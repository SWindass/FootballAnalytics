"""Understat scraper for xG (Expected Goals) data."""

import json
import re
from datetime import datetime
from decimal import Decimal
from typing import Any, Optional

import httpx
import structlog
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential

logger = structlog.get_logger()


class UnderstatScraper:
    """Scraper for Understat xG data."""

    BASE_URL = "https://understat.com"
    LEAGUE = "EPL"

    def __init__(self):
        self._session: Optional[httpx.AsyncClient] = None

    async def _get_session(self) -> httpx.AsyncClient:
        if self._session is None:
            self._session = httpx.AsyncClient(
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                },
                timeout=30,
            )
        return self._session

    async def close(self) -> None:
        if self._session:
            await self._session.aclose()
            self._session = None

    def _extract_json_data(self, html: str, var_name: str) -> Any:
        """Extract JSON data from JavaScript variable in HTML."""
        pattern = rf"var {var_name}\s*=\s*JSON\.parse\('(.+?)'\)"
        match = re.search(pattern, html)
        if not match:
            return None

        # Decode escaped unicode
        json_str = match.group(1).encode().decode("unicode_escape")
        return json.loads(json_str)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def get_league_matches(self, season: str = "2024") -> list[dict[str, Any]]:
        """Get all matches for a season with xG data.

        Args:
            season: Season start year (e.g., "2024" for 2024-25)
        """
        session = await self._get_session()
        url = f"{self.BASE_URL}/league/{self.LEAGUE}/{season}"

        response = await session.get(url)
        response.raise_for_status()

        matches_data = self._extract_json_data(response.text, "datesData")
        if not matches_data:
            logger.warning("Could not extract matches data from Understat")
            return []

        return matches_data

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def get_match_details(self, match_id: int) -> Optional[dict[str, Any]]:
        """Get detailed xG data for a specific match."""
        session = await self._get_session()
        url = f"{self.BASE_URL}/match/{match_id}"

        response = await session.get(url)
        response.raise_for_status()

        shots_data = self._extract_json_data(response.text, "shotsData")
        match_info = self._extract_json_data(response.text, "match_info")

        if not shots_data:
            return None

        return {
            "match_id": match_id,
            "match_info": match_info,
            "shots": shots_data,
        }

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def get_team_stats(self, team_name: str, season: str = "2024") -> Optional[dict[str, Any]]:
        """Get team xG statistics for a season."""
        session = await self._get_session()

        # Convert team name to URL format
        url_name = team_name.replace(" ", "_")
        url = f"{self.BASE_URL}/team/{url_name}/{season}"

        try:
            response = await session.get(url)
            response.raise_for_status()
        except httpx.HTTPStatusError:
            logger.warning(f"Could not fetch team data for {team_name}")
            return None

        stats_data = self._extract_json_data(response.text, "statisticsData")
        dates_data = self._extract_json_data(response.text, "datesData")

        return {
            "team": team_name,
            "season": season,
            "statistics": stats_data,
            "matches": dates_data,
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
