"""Integration tests for data sources."""

import pytest
from unittest.mock import patch, AsyncMock

from batch.data_sources.football_data_org import (
    FootballDataClient,
    parse_match,
    parse_team,
)
from batch.data_sources.the_odds_api import (
    OddsApiClient,
    parse_odds,
    match_team_names,
)


class TestFootballDataClient:
    """Tests for football-data.org client."""

    def test_parse_match(self):
        """Test parsing match data from API response."""
        api_response = {
            "id": 12345,
            "season": {"startDate": "2024-08-01", "endDate": "2025-05-31"},
            "matchday": 15,
            "utcDate": "2025-01-25T15:00:00Z",
            "status": "FINISHED",
            "homeTeam": {"id": 1, "name": "Arsenal"},
            "awayTeam": {"id": 2, "name": "Chelsea"},
            "score": {
                "fullTime": {"home": 2, "away": 1},
                "halfTime": {"home": 1, "away": 0},
            },
        }

        parsed = parse_match(api_response)

        assert parsed["external_id"] == 12345
        assert parsed["season"] == "2024-25"
        assert parsed["matchweek"] == 15
        assert parsed["status"] == "finished"
        assert parsed["home_team_external_id"] == 1
        assert parsed["away_team_external_id"] == 2
        assert parsed["home_score"] == 2
        assert parsed["away_score"] == 1

    def test_parse_team(self):
        """Test parsing team data from API response."""
        api_response = {
            "id": 57,
            "name": "Arsenal FC",
            "shortName": "Arsenal",
            "tla": "ARS",
            "crest": "https://crests.football-data.org/57.png",
            "venue": "Emirates Stadium",
            "founded": 1886,
        }

        parsed = parse_team(api_response)

        assert parsed["external_id"] == 57
        assert parsed["name"] == "Arsenal FC"
        assert parsed["short_name"] == "Arsenal"
        assert parsed["tla"] == "ARS"
        assert parsed["venue"] == "Emirates Stadium"


class TestOddsApiClient:
    """Tests for The Odds API client."""

    def test_parse_odds(self):
        """Test parsing odds data from API response."""
        api_response = {
            "id": "abc123",
            "commence_time": "2025-01-25T15:00:00Z",
            "home_team": "Arsenal",
            "away_team": "Chelsea",
            "bookmakers": [
                {
                    "key": "bet365",
                    "title": "Bet365",
                    "last_update": "2025-01-24T10:00:00Z",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Arsenal", "price": 2.1},
                                {"name": "Draw", "price": 3.4},
                                {"name": "Chelsea", "price": 3.5},
                            ],
                        },
                        {
                            "key": "totals",
                            "outcomes": [
                                {"name": "Over", "point": 2.5, "price": 1.9},
                                {"name": "Under", "point": 2.5, "price": 1.95},
                            ],
                        },
                    ],
                }
            ],
        }

        records = parse_odds(api_response)

        assert len(records) == 1
        record = records[0]
        assert record["bookmaker"] == "bet365"
        assert record["home_team"] == "Arsenal"
        assert record["away_team"] == "Chelsea"
        assert float(record["home_odds"]) == 2.1
        assert float(record["draw_odds"]) == 3.4
        assert float(record["away_odds"]) == 3.5
        assert float(record["over_2_5_odds"]) == 1.9

    def test_match_team_names(self):
        """Test team name matching."""
        db_teams = [
            {"id": 1, "name": "Arsenal FC", "short_name": "Arsenal"},
            {"id": 2, "name": "Chelsea FC", "short_name": "Chelsea"},
            {"id": 3, "name": "Manchester United FC", "short_name": "Man United"},
        ]

        # Direct match
        home_id, away_id = match_team_names("Arsenal", "Chelsea", db_teams)
        assert home_id == 1
        assert away_id == 2

        # Using short name
        home_id, away_id = match_team_names("Man United", "Arsenal", db_teams)
        assert home_id == 3
        assert away_id == 1

    def test_match_team_names_not_found(self):
        """Test team name matching when not found."""
        db_teams = [
            {"id": 1, "name": "Arsenal FC", "short_name": "Arsenal"},
        ]

        home_id, away_id = match_team_names("Liverpool", "Everton", db_teams)
        assert home_id is None
        assert away_id is None
