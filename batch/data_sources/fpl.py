"""Fantasy Premier League API client.

Free, official API providing player-level data including:
- Player statistics (goals, assists, clean sheets, etc.)
- ICT Index (Influence, Creativity, Threat)
- Expected stats (xG, xA per player)
- Player prices and ownership
- Match-by-match performance
"""

from datetime import datetime
from typing import Any

import httpx
import structlog

logger = structlog.get_logger()


class FPLClient:
    """Client for Fantasy Premier League API."""

    BASE_URL = "https://fantasy.premierleague.com/api"

    def __init__(self):
        self._cache = {}
        self._cache_time = None
        self._cache_ttl = 3600  # 1 hour cache

    async def _make_request(self, endpoint: str) -> dict[str, Any]:
        """Make API request with caching."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.BASE_URL}/{endpoint}",
                timeout=30,
                headers={"User-Agent": "FootballAnalytics/1.0"}
            )
            response.raise_for_status()
            return response.json()

    async def get_bootstrap_static(self, use_cache: bool = True) -> dict[str, Any]:
        """Get main bootstrap data - teams, players, gameweeks.

        This is the main endpoint containing:
        - elements: All players with current season stats
        - teams: All teams
        - events: All gameweeks
        - element_types: Player positions
        """
        cache_key = "bootstrap_static"
        now = datetime.utcnow()

        if use_cache and cache_key in self._cache and self._cache_time:
            age = (now - self._cache_time).total_seconds()
            if age < self._cache_ttl:
                return self._cache[cache_key]

        data = await self._make_request("bootstrap-static/")
        self._cache[cache_key] = data
        self._cache_time = now
        return data

    async def get_player_summary(self, player_id: int) -> dict[str, Any]:
        """Get detailed player data including match history.

        Returns:
            - fixtures: Upcoming fixtures
            - history: This season's match-by-match data
            - history_past: Previous seasons' data
        """
        return await self._make_request(f"element-summary/{player_id}/")

    async def get_gameweek_live(self, gameweek: int) -> dict[str, Any]:
        """Get live data for a specific gameweek.

        Returns real-time stats during matches.
        """
        return await self._make_request(f"event/{gameweek}/live/")

    async def get_fixtures(self, gameweek: int | None = None) -> list[dict[str, Any]]:
        """Get fixtures, optionally filtered by gameweek."""
        endpoint = "fixtures/"
        if gameweek:
            endpoint += f"?event={gameweek}"
        return await self._make_request(endpoint)

    async def get_all_players(self) -> list[dict[str, Any]]:
        """Get all players with current stats."""
        data = await self.get_bootstrap_static()
        return data.get("elements", [])

    async def get_all_teams(self) -> list[dict[str, Any]]:
        """Get all FPL teams."""
        data = await self.get_bootstrap_static()
        return data.get("teams", [])

    async def get_current_gameweek(self) -> int | None:
        """Get the current gameweek number."""
        data = await self.get_bootstrap_static()
        events = data.get("events", [])
        for event in events:
            if event.get("is_current"):
                return event.get("id")
        return None

    async def get_player_history(self, player_id: int) -> list[dict[str, Any]]:
        """Get a player's match-by-match history for current season."""
        data = await self.get_player_summary(player_id)
        return data.get("history", [])


def parse_fpl_player(player_data: dict[str, Any], teams: dict[int, str]) -> dict[str, Any]:
    """Parse FPL player data to our schema.

    Args:
        player_data: Raw player data from FPL API
        teams: Mapping of FPL team_id to team name
    """
    return {
        "fpl_id": player_data["id"],
        "name": f"{player_data['first_name']} {player_data['second_name']}",
        "web_name": player_data["web_name"],  # Short display name
        "team_fpl_id": player_data["team"],
        "team_name": teams.get(player_data["team"], "Unknown"),
        "position": _position_map.get(player_data["element_type"], "Unknown"),
        "price": player_data["now_cost"] / 10,  # Price in millions
        "total_points": player_data["total_points"],
        "points_per_game": float(player_data["points_per_game"]),
        "selected_by_percent": float(player_data["selected_by_percent"]),
        # Form and ICT
        "form": float(player_data["form"]) if player_data["form"] else 0.0,
        "influence": float(player_data["influence"]),
        "creativity": float(player_data["creativity"]),
        "threat": float(player_data["threat"]),
        "ict_index": float(player_data["ict_index"]),
        # Expected stats
        "expected_goals": float(player_data.get("expected_goals", 0) or 0),
        "expected_assists": float(player_data.get("expected_assists", 0) or 0),
        "expected_goal_involvements": float(player_data.get("expected_goal_involvements", 0) or 0),
        "expected_goals_conceded": float(player_data.get("expected_goals_conceded", 0) or 0),
        # Actual stats
        "goals_scored": player_data["goals_scored"],
        "assists": player_data["assists"],
        "clean_sheets": player_data["clean_sheets"],
        "goals_conceded": player_data["goals_conceded"],
        "minutes": player_data["minutes"],
        "starts": player_data.get("starts", 0),
        # Status
        "status": player_data["status"],  # a=available, d=doubtful, i=injured, s=suspended, u=unavailable
        "chance_of_playing": player_data.get("chance_of_playing_next_round"),
        "news": player_data.get("news", ""),
    }


def parse_fpl_match_performance(history_entry: dict[str, Any]) -> dict[str, Any]:
    """Parse a single match performance from player history."""
    return {
        "gameweek": history_entry["round"],
        "opponent_team_fpl_id": history_entry["opponent_team"],
        "was_home": history_entry["was_home"],
        "minutes": history_entry["minutes"],
        "total_points": history_entry["total_points"],
        "goals_scored": history_entry["goals_scored"],
        "assists": history_entry["assists"],
        "clean_sheets": history_entry["clean_sheets"],
        "goals_conceded": history_entry["goals_conceded"],
        "bonus": history_entry["bonus"],
        "bps": history_entry["bps"],  # Bonus points system score
        "influence": float(history_entry["influence"]),
        "creativity": float(history_entry["creativity"]),
        "threat": float(history_entry["threat"]),
        "ict_index": float(history_entry["ict_index"]),
        "expected_goals": float(history_entry.get("expected_goals", 0) or 0),
        "expected_assists": float(history_entry.get("expected_assists", 0) or 0),
        "expected_goal_involvements": float(history_entry.get("expected_goal_involvements", 0) or 0),
        "value": history_entry["value"] / 10,  # Price at time of match
        "selected": history_entry["selected"],  # Number of managers who selected
    }


# Position mapping
_position_map = {
    1: "GKP",  # Goalkeeper
    2: "DEF",  # Defender
    3: "MID",  # Midfielder
    4: "FWD",  # Forward
}


async def fetch_team_player_ratings(fpl_team_id: int) -> dict[str, Any]:
    """Get aggregated player ratings for a team.

    Useful for pre-match analysis - summarizes team strength
    based on player-level data.
    """
    client = FPLClient()
    players = await client.get_all_players()

    team_players = [p for p in players if p["team"] == fpl_team_id]

    if not team_players:
        return {}

    # Calculate team aggregates
    total_ict = sum(float(p["ict_index"]) for p in team_players)
    avg_form = sum(float(p["form"] or 0) for p in team_players) / len(team_players)
    total_xg = sum(float(p.get("expected_goals", 0) or 0) for p in team_players)
    total_xa = sum(float(p.get("expected_assists", 0) or 0) for p in team_players)

    # Key players (top 5 by ICT)
    key_players = sorted(team_players, key=lambda p: float(p["ict_index"]), reverse=True)[:5]

    # Injury concerns
    injured = [p for p in team_players if p["status"] in ("i", "d", "s")]

    return {
        "total_ict_index": total_ict,
        "avg_form": avg_form,
        "total_expected_goals": total_xg,
        "total_expected_assists": total_xa,
        "key_players": [p["web_name"] for p in key_players],
        "injury_concerns": [
            {"name": p["web_name"], "status": p["status"], "news": p.get("news", "")}
            for p in injured
        ],
    }
