"""Understat scraper for xG (Expected Goals) data.

Uses Playwright to render JavaScript and extract data from Understat pages.
"""

import asyncio
from datetime import datetime
from decimal import Decimal
from typing import Any, Optional

import structlog
from playwright.async_api import async_playwright, Browser, Page
from tenacity import retry, stop_after_attempt, wait_exponential

logger = structlog.get_logger()


class UnderstatScraper:
    """Scraper for Understat xG data using Playwright."""

    BASE_URL = "https://understat.com"
    LEAGUE = "EPL"

    def __init__(self):
        self._browser: Optional[Browser] = None
        self._playwright = None

    async def _get_browser(self) -> Browser:
        """Get or create browser instance."""
        if self._browser is None:
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(headless=True)
        return self._browser

    async def close(self) -> None:
        """Close browser and playwright."""
        if self._browser:
            await self._browser.close()
            self._browser = None
        if self._playwright:
            await self._playwright.stop()
            self._playwright = None

    async def _load_page_with_data(self, url: str, data_var: str, timeout: int = 120000) -> Any:
        """Load a page and extract JavaScript data.

        Args:
            url: URL to load
            data_var: JavaScript variable name to extract
            timeout: Page load timeout in ms (default 2 minutes)

        Returns:
            Extracted data or None
        """
        browser = await self._get_browser()
        page = await browser.new_page()

        try:
            logger.info(f"Loading {url}")

            # Try multiple times with increasing timeouts
            for attempt in range(3):
                try:
                    await page.goto(url, wait_until="domcontentloaded", timeout=timeout)
                    break
                except Exception as e:
                    if attempt < 2:
                        logger.warning(f"Attempt {attempt + 1} failed, retrying...")
                        await asyncio.sleep(2)
                    else:
                        raise

            # Wait for data to be populated by JavaScript
            for i in range(20):
                await page.wait_for_timeout(1000)
                has_data = await page.evaluate(
                    f"() => typeof {data_var} !== 'undefined' && {data_var} !== null"
                )
                if has_data:
                    logger.info(f"Data loaded after {i+1} seconds")
                    break

            # Extract the data
            data = await page.evaluate(f"() => {data_var}")
            return data

        except Exception as e:
            logger.error(f"Error loading {url}: {e}")
            return None

        finally:
            await page.close()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def get_league_matches(self, season: str = "2024") -> list[dict[str, Any]]:
        """Get all matches for a season with xG data.

        Args:
            season: Season start year (e.g., "2024" for 2024-25)

        Returns:
            List of match dictionaries with xG data
        """
        url = f"{self.BASE_URL}/league/{self.LEAGUE}/{season}"
        matches = await self._load_page_with_data(url, "datesData")

        if not matches:
            logger.warning(f"Could not extract matches data for season {season}")
            return []

        logger.info(f"Found {len(matches)} matches for {self.LEAGUE} {season}")
        return matches

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def get_team_stats(self, team_name: str, season: str = "2024") -> Optional[dict[str, Any]]:
        """Get team xG statistics for a season.

        Args:
            team_name: Team name (e.g., "Arsenal")
            season: Season start year

        Returns:
            Team statistics or None
        """
        # Convert team name to URL format
        url_name = team_name.replace(" ", "_")
        url = f"{self.BASE_URL}/team/{url_name}/{season}"

        stats = await self._load_page_with_data(url, "statisticsData")
        matches = await self._load_page_with_data(url, "datesData")

        if not stats and not matches:
            logger.warning(f"Could not fetch team data for {team_name}")
            return None

        return {
            "team": team_name,
            "season": season,
            "statistics": stats,
            "matches": matches,
        }


def parse_understat_match(match_data: dict[str, Any]) -> dict[str, Any]:
    """Parse match data from Understat to our format.

    Understat match data structure:
    {
        "id": "12345",
        "isResult": true,
        "h": {"id": "123", "title": "Arsenal", "short_title": "ARS"},
        "a": {"id": "456", "title": "Chelsea", "short_title": "CHE"},
        "goals": {"h": "2", "a": "1"},
        "xG": {"h": "1.85", "a": "1.23"},
        "datetime": "2024-01-15 15:00:00"
    }
    """
    return {
        "understat_id": int(match_data["id"]),
        "is_finished": match_data.get("isResult", False),
        "home_team": match_data["h"]["title"],
        "away_team": match_data["a"]["title"],
        "home_goals": int(match_data["goals"]["h"]) if match_data.get("isResult") else None,
        "away_goals": int(match_data["goals"]["a"]) if match_data.get("isResult") else None,
        "home_xg": Decimal(match_data["xG"]["h"]) if match_data.get("xG", {}).get("h") else None,
        "away_xg": Decimal(match_data["xG"]["a"]) if match_data.get("xG", {}).get("a") else None,
        "datetime": datetime.strptime(match_data["datetime"], "%Y-%m-%d %H:%M:%S"),
    }


def match_understat_to_fixture(
    understat_match: dict[str, Any],
    fixtures: list[dict[str, Any]],
    tolerance_hours: int = 24,
) -> Optional[int]:
    """Match an Understat match to a database fixture.

    Args:
        understat_match: Parsed Understat match data
        fixtures: List of fixture dicts with 'id', 'home_team', 'away_team', 'kickoff_time'
        tolerance_hours: Time window for matching

    Returns:
        Fixture ID if matched, None otherwise
    """
    from datetime import timedelta

    us_time = understat_match["datetime"]
    us_home = understat_match["home_team"].lower()
    us_away = understat_match["away_team"].lower()

    for fixture in fixtures:
        fx_home = fixture["home_team"].lower()
        fx_away = fixture["away_team"].lower()
        fx_time = fixture["kickoff_time"]

        # Check team names match
        if us_home not in fx_home and fx_home not in us_home:
            continue
        if us_away not in fx_away and fx_away not in us_away:
            continue

        # Check time is close
        time_diff = abs((us_time - fx_time).total_seconds()) / 3600
        if time_diff <= tolerance_hours:
            return fixture["id"]

    return None
