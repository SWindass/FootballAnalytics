"""Football-data.org API client for fixtures and results."""

import time
from datetime import datetime
from typing import Any

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import get_settings

logger = structlog.get_logger()
settings = get_settings()


class FootballDataClient:
    """Client for football-data.org API."""

    BASE_URL = "https://api.football-data.org/v4"
    COMPETITION_CODE = "PL"  # Premier League

    # Supported competitions for fixture congestion tracking
    COMPETITIONS = {
        "PL": "Premier League",
        "CL": "UEFA Champions League",
        "EL": "UEFA Europa League",  # Not on free tier but included for completeness
    }

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or settings.football_data_api_key
        self.rate_limit = settings.football_data_rate_limit
        self._last_request_time = 0.0

    def _get_headers(self) -> dict[str, str]:
        return {"X-Auth-Token": self.api_key}

    def _rate_limit_wait(self) -> None:
        """Ensure we don't exceed rate limit (10 requests/minute)."""
        min_interval = 60.0 / self.rate_limit
        elapsed = time.time() - self._last_request_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_request_time = time.time()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def _make_request(self, endpoint: str, params: dict | None = None) -> dict[str, Any]:
        """Make API request with rate limiting and retries."""
        self._rate_limit_wait()

        url = f"{self.BASE_URL}/{endpoint}"
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=self._get_headers(), params=params, timeout=30)
            response.raise_for_status()
            return response.json()

    async def get_competition_info(self) -> dict[str, Any]:
        """Get Premier League competition info."""
        return await self._make_request(f"competitions/{self.COMPETITION_CODE}")

    async def get_teams(self) -> list[dict[str, Any]]:
        """Get all teams in the Premier League."""
        data = await self._make_request(f"competitions/{self.COMPETITION_CODE}/teams")
        return data.get("teams", [])

    async def get_matches(
        self,
        season: str | None = None,
        matchday: int | None = None,
        status: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get matches with optional filters.

        Args:
            season: Season year (e.g., "2024" for 2024-25)
            matchday: Specific matchday (1-38)
            status: SCHEDULED, LIVE, IN_PLAY, PAUSED, FINISHED, POSTPONED, CANCELLED
            date_from: ISO date string
            date_to: ISO date string
        """
        params = {}
        if season:
            params["season"] = season
        if matchday:
            params["matchday"] = matchday
        if status:
            params["status"] = status
        if date_from:
            params["dateFrom"] = date_from
        if date_to:
            params["dateTo"] = date_to

        data = await self._make_request(f"competitions/{self.COMPETITION_CODE}/matches", params)
        return data.get("matches", [])

    async def get_match(self, match_id: int) -> dict[str, Any]:
        """Get a specific match by ID."""
        return await self._make_request(f"matches/{match_id}")

    async def get_standings(self, season: str | None = None) -> dict[str, Any]:
        """Get current league standings."""
        params = {"season": season} if season else {}
        data = await self._make_request(
            f"competitions/{self.COMPETITION_CODE}/standings", params
        )
        return data.get("standings", [])

    async def get_team_matches(
        self,
        team_id: int,
        season: str | None = None,
        status: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get matches for a specific team."""
        params = {"limit": limit}
        if season:
            params["season"] = season
        if status:
            params["status"] = status

        data = await self._make_request(f"teams/{team_id}/matches", params)
        return data.get("matches", [])

    async def get_competition_matches(
        self,
        competition: str,
        season: str | None = None,
        matchday: int | None = None,
        status: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get matches for a specific competition.

        Args:
            competition: Competition code (PL, CL, etc.)
            season: Season year (e.g., "2024" for 2024-25)
            matchday: Specific matchday/gameweek
            status: SCHEDULED, FINISHED, etc.
            date_from: ISO date string
            date_to: ISO date string
        """
        params = {}
        if season:
            params["season"] = season
        if matchday:
            params["matchday"] = matchday
        if status:
            params["status"] = status
        if date_from:
            params["dateFrom"] = date_from
        if date_to:
            params["dateTo"] = date_to

        data = await self._make_request(f"competitions/{competition}/matches", params)
        return data.get("matches", [])

    async def get_all_team_matches(
        self,
        team_external_id: int,
        season: str | None = None,
        competitions: list[str] = None,
    ) -> list[dict[str, Any]]:
        """Get ALL matches for a team across multiple competitions.

        This is useful for calculating accurate rest days.

        Args:
            team_external_id: Team's external ID from football-data.org
            season: Season year
            competitions: List of competition codes to include

        Returns:
            List of all matches sorted by date
        """
        if competitions is None:
            competitions = ["PL", "CL"]  # Default to PL and CL

        all_matches = []

        # Fetch from each competition
        for comp in competitions:
            try:
                matches = await self.get_competition_matches(
                    competition=comp,
                    season=season,
                    status="FINISHED",
                )
                # Filter to matches involving this team
                team_matches = [
                    m for m in matches
                    if m.get("homeTeam", {}).get("id") == team_external_id
                    or m.get("awayTeam", {}).get("id") == team_external_id
                ]
                all_matches.extend(team_matches)
            except Exception as e:
                logger.warning(f"Failed to fetch {comp} matches: {e}")

        # Sort by date
        all_matches.sort(key=lambda m: m.get("utcDate", ""))

        return all_matches


def parse_match(match_data: dict[str, Any]) -> dict[str, Any]:
    """Parse match data from API response to our schema format."""
    utc_date = datetime.fromisoformat(match_data["utcDate"].replace("Z", "+00:00"))

    score = match_data.get("score", {})
    full_time = score.get("fullTime", {})
    half_time = score.get("halfTime", {})

    # Map API status to our status
    status_map = {
        "SCHEDULED": "scheduled",
        "TIMED": "scheduled",
        "LIVE": "in_play",
        "IN_PLAY": "in_play",
        "PAUSED": "paused",
        "FINISHED": "finished",
        "POSTPONED": "postponed",
        "CANCELLED": "cancelled",
    }

    # Extract referee info (first referee in list is usually the main one)
    referees = match_data.get("referees", [])
    main_referee = None
    for ref in referees:
        if ref.get("type") == "REFEREE" or not main_referee:
            main_referee = ref
            if ref.get("type") == "REFEREE":
                break

    # Extract coach info
    home_coach = match_data.get("homeTeam", {}).get("coach")
    away_coach = match_data.get("awayTeam", {}).get("coach")

    return {
        "external_id": match_data["id"],
        "season": f"{match_data['season']['startDate'][:4]}-{match_data['season']['endDate'][2:4]}",
        "matchweek": match_data.get("matchday", 0),
        "kickoff_time": utc_date,
        "status": status_map.get(match_data.get("status", "SCHEDULED"), "scheduled"),
        "home_team_external_id": match_data["homeTeam"]["id"],
        "away_team_external_id": match_data["awayTeam"]["id"],
        "home_score": full_time.get("home"),
        "away_score": full_time.get("away"),
        "home_ht_score": half_time.get("home"),
        "away_ht_score": half_time.get("away"),
        # New fields
        "referee": parse_referee(main_referee) if main_referee else None,
        "home_coach": parse_coach(home_coach) if home_coach else None,
        "away_coach": parse_coach(away_coach) if away_coach else None,
    }


def parse_referee(referee_data: dict[str, Any]) -> dict[str, Any]:
    """Parse referee data from API response."""
    return {
        "external_id": referee_data.get("id"),
        "name": referee_data.get("name", "Unknown"),
        "nationality": referee_data.get("nationality"),
    }


def parse_coach(coach_data: dict[str, Any]) -> dict[str, Any]:
    """Parse coach/manager data from API response."""
    return {
        "external_id": coach_data.get("id"),
        "name": coach_data.get("name", "Unknown"),
        "nationality": coach_data.get("nationality"),
        "date_of_birth": coach_data.get("dateOfBirth"),
    }


def parse_team(team_data: dict[str, Any]) -> dict[str, Any]:
    """Parse team data from API response to our schema format."""
    return {
        "external_id": team_data["id"],
        "name": team_data["name"],
        "short_name": team_data.get("shortName", team_data["name"]),
        "tla": team_data.get("tla", team_data["name"][:3].upper()),
        "crest_url": team_data.get("crest"),
        "venue": team_data.get("venue"),
        "founded": team_data.get("founded"),
    }
