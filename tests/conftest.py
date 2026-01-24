"""Pytest configuration and fixtures."""

import pytest
from datetime import datetime
from decimal import Decimal


@pytest.fixture
def sample_match():
    """Sample match data for testing."""
    return {
        "id": 1,
        "external_id": 12345,
        "season": "2024-25",
        "matchweek": 15,
        "kickoff_time": datetime(2025, 1, 25, 15, 0, 0),
        "status": "scheduled",
        "home_team_id": 1,
        "away_team_id": 2,
        "home_score": None,
        "away_score": None,
    }


@pytest.fixture
def sample_team():
    """Sample team data for testing."""
    return {
        "id": 1,
        "external_id": 57,
        "name": "Arsenal FC",
        "short_name": "Arsenal",
        "tla": "ARS",
        "venue": "Emirates Stadium",
    }


@pytest.fixture
def sample_team_stats():
    """Sample team statistics for testing."""
    return {
        "team_id": 1,
        "season": "2024-25",
        "matchweek": 14,
        "form": "WWDWL",
        "form_points": 10,
        "goals_scored": 35,
        "goals_conceded": 18,
        "avg_goals_scored": Decimal("2.50"),
        "avg_goals_conceded": Decimal("1.29"),
        "home_wins": 6,
        "home_draws": 1,
        "home_losses": 0,
        "away_wins": 4,
        "away_draws": 2,
        "away_losses": 1,
        "clean_sheets": 6,
        "failed_to_score": 2,
    }


@pytest.fixture
def sample_odds():
    """Sample odds data for testing."""
    return {
        "match_id": 1,
        "bookmaker": "bet365",
        "home_odds": Decimal("2.10"),
        "draw_odds": Decimal("3.40"),
        "away_odds": Decimal("3.50"),
        "over_2_5_odds": Decimal("1.90"),
        "under_2_5_odds": Decimal("1.95"),
    }


@pytest.fixture
def sample_analysis():
    """Sample match analysis for testing."""
    return {
        "match_id": 1,
        "elo_home_prob": Decimal("0.4500"),
        "elo_draw_prob": Decimal("0.2800"),
        "elo_away_prob": Decimal("0.2700"),
        "poisson_home_prob": Decimal("0.4800"),
        "poisson_draw_prob": Decimal("0.2500"),
        "poisson_away_prob": Decimal("0.2700"),
        "poisson_over_2_5_prob": Decimal("0.5500"),
        "poisson_btts_prob": Decimal("0.5200"),
        "consensus_home_prob": Decimal("0.4650"),
        "consensus_draw_prob": Decimal("0.2650"),
        "consensus_away_prob": Decimal("0.2700"),
        "predicted_home_goals": Decimal("1.75"),
        "predicted_away_goals": Decimal("1.25"),
    }
