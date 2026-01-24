"""The Odds API client for betting odds data."""

from datetime import datetime
from decimal import Decimal
from typing import Any, Optional

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import get_settings

logger = structlog.get_logger()
settings = get_settings()


class OddsApiClient:
    """Client for The Odds API."""

    BASE_URL = "https://api.the-odds-api.com/v4"
    SPORT = "soccer_epl"  # EPL identifier

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.odds_api_key
        self._requests_remaining: Optional[int] = None
        self._requests_used: Optional[int] = None

    def _update_quota(self, headers: dict) -> None:
        """Update API quota tracking from response headers."""
        if "x-requests-remaining" in headers:
            self._requests_remaining = int(headers["x-requests-remaining"])
        if "x-requests-used" in headers:
            self._requests_used = int(headers["x-requests-used"])
        logger.debug(
            "Odds API quota",
            remaining=self._requests_remaining,
            used=self._requests_used,
        )

    @property
    def requests_remaining(self) -> Optional[int]:
        return self._requests_remaining

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def _make_request(
        self, endpoint: str, params: Optional[dict] = None
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Make API request with retries."""
        url = f"{self.BASE_URL}/{endpoint}"
        request_params = {"apiKey": self.api_key}
        if params:
            request_params.update(params)

        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=request_params, timeout=30)
            self._update_quota(dict(response.headers))
            response.raise_for_status()
            return response.json()

    async def get_odds(
        self,
        markets: str = "h2h,totals",
        regions: str = "uk,eu",
        odds_format: str = "decimal",
        bookmakers: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Get current odds for EPL matches.

        Args:
            markets: Comma-separated markets (h2h, spreads, totals)
            regions: Comma-separated regions (uk, eu, us, au)
            odds_format: decimal or american
            bookmakers: Optional comma-separated bookmaker keys
        """
        params = {
            "markets": markets,
            "regions": regions,
            "oddsFormat": odds_format,
        }
        if bookmakers:
            params["bookmakers"] = bookmakers

        return await self._make_request(f"sports/{self.SPORT}/odds", params)

    async def get_scores(
        self,
        days_from: int = 3,
    ) -> list[dict[str, Any]]:
        """Get scores for recent/upcoming EPL matches.

        Args:
            days_from: Number of days in the past to include completed matches
        """
        params = {"daysFrom": days_from}
        return await self._make_request(f"sports/{self.SPORT}/scores", params)

    async def get_events(self) -> list[dict[str, Any]]:
        """Get upcoming EPL events (no odds, lower quota cost)."""
        return await self._make_request(f"sports/{self.SPORT}/events")


def parse_odds(event_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Parse odds data from API response.

    Returns a list of odds records, one per bookmaker.
    """
    records = []

    commence_time = datetime.fromisoformat(event_data["commence_time"].replace("Z", "+00:00"))
    home_team = event_data["home_team"]
    away_team = event_data["away_team"]

    for bookmaker in event_data.get("bookmakers", []):
        record = {
            "event_id": event_data["id"],
            "home_team": home_team,
            "away_team": away_team,
            "commence_time": commence_time,
            "bookmaker": bookmaker["key"],
            "bookmaker_title": bookmaker["title"],
            "last_update": datetime.fromisoformat(
                bookmaker["last_update"].replace("Z", "+00:00")
            ),
        }

        # Parse markets
        for market in bookmaker.get("markets", []):
            if market["key"] == "h2h":
                # Match winner odds
                for outcome in market["outcomes"]:
                    if outcome["name"] == home_team:
                        record["home_odds"] = Decimal(str(outcome["price"]))
                    elif outcome["name"] == away_team:
                        record["away_odds"] = Decimal(str(outcome["price"]))
                    elif outcome["name"] == "Draw":
                        record["draw_odds"] = Decimal(str(outcome["price"]))

            elif market["key"] == "totals":
                # Over/Under 2.5 goals
                for outcome in market["outcomes"]:
                    if outcome.get("point") == 2.5:
                        if outcome["name"] == "Over":
                            record["over_2_5_odds"] = Decimal(str(outcome["price"]))
                        elif outcome["name"] == "Under":
                            record["under_2_5_odds"] = Decimal(str(outcome["price"]))

        records.append(record)

    return records


def match_team_names(
    odds_home: str,
    odds_away: str,
    db_teams: list[dict[str, str]],
) -> tuple[Optional[int], Optional[int]]:
    """Match team names from odds API to database team IDs.

    Args:
        odds_home: Home team name from odds API
        odds_away: Away team name from odds API
        db_teams: List of team dicts with 'id', 'name', 'short_name'

    Returns:
        Tuple of (home_team_id, away_team_id) or (None, None) if not matched
    """
    # Common name mappings
    name_mappings = {
        "Manchester United": ["Man United", "Man Utd"],
        "Manchester City": ["Man City"],
        "Tottenham Hotspur": ["Tottenham", "Spurs"],
        "Newcastle United": ["Newcastle"],
        "West Ham United": ["West Ham"],
        "Wolverhampton Wanderers": ["Wolves", "Wolverhampton"],
        "Brighton and Hove Albion": ["Brighton"],
        "Nottingham Forest": ["Nott'm Forest", "Nottm Forest"],
        "Leicester City": ["Leicester"],
        "Crystal Palace": ["C Palace"],
        "AFC Bournemouth": ["Bournemouth"],
        "Ipswich Town": ["Ipswich"],
    }

    def find_team(name: str) -> Optional[int]:
        name_lower = name.lower()
        for team in db_teams:
            if (
                team["name"].lower() == name_lower
                or team["short_name"].lower() == name_lower
            ):
                return team["id"]

            # Check mappings
            for canonical, aliases in name_mappings.items():
                if name_lower in [a.lower() for a in aliases]:
                    if team["name"].lower() == canonical.lower():
                        return team["id"]
                if team["name"].lower() == canonical.lower():
                    if name_lower in [a.lower() for a in aliases]:
                        return team["id"]

        return None

    home_id = find_team(odds_home)
    away_id = find_team(odds_away)

    return home_id, away_id
