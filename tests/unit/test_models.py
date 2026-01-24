"""Unit tests for prediction models."""

import pytest
from decimal import Decimal

from batch.models.elo import EloRatingSystem, EloConfig
from batch.models.poisson import PoissonModel, PoissonConfig
from batch.betting.kelly_criterion import KellyCalculator, KellyConfig


class TestEloRatingSystem:
    """Tests for ELO rating system."""

    def test_initial_rating(self):
        """Test teams start with default rating."""
        elo = EloRatingSystem()
        assert elo.get_rating(1) == 1500.0

    def test_expected_score_equal_teams(self):
        """Test expected score for equal teams."""
        elo = EloRatingSystem()
        elo.set_rating(1, 1500)
        elo.set_rating(2, 1500)

        # Without home advantage
        home_exp, away_exp = elo.expected_score(1500, 1500, include_home_advantage=False)
        assert abs(home_exp - 0.5) < 0.01
        assert abs(away_exp - 0.5) < 0.01

    def test_expected_score_with_home_advantage(self):
        """Test home advantage affects expected score."""
        elo = EloRatingSystem()

        # With home advantage, home team should have higher expected score
        home_exp, away_exp = elo.expected_score(1500, 1500, include_home_advantage=True)
        assert home_exp > 0.5
        assert away_exp < 0.5

    def test_update_ratings_home_win(self):
        """Test rating update after home win."""
        elo = EloRatingSystem()
        elo.set_rating(1, 1500)
        elo.set_rating(2, 1500)

        home_change, away_change = elo.update_ratings(1, 2, 2, 0)

        # Home team should gain rating, away should lose
        assert home_change > 0
        assert away_change < 0
        assert elo.get_rating(1) > 1500
        assert elo.get_rating(2) < 1500

    def test_update_ratings_draw(self):
        """Test rating update after draw."""
        elo = EloRatingSystem()
        elo.set_rating(1, 1500)
        elo.set_rating(2, 1500)

        home_change, away_change = elo.update_ratings(1, 2, 1, 1)

        # Draw against same-rated team with home advantage means home loses rating
        assert home_change < 0
        assert away_change > 0

    def test_match_probabilities_sum_to_one(self):
        """Test probabilities sum to 1."""
        elo = EloRatingSystem()
        elo.set_rating(1, 1600)
        elo.set_rating(2, 1400)

        home_prob, draw_prob, away_prob = elo.match_probabilities(1, 2)

        assert abs(home_prob + draw_prob + away_prob - 1.0) < 0.001

    def test_goal_difference_multiplier(self):
        """Test larger goal differences have larger impact."""
        elo = EloRatingSystem()

        mult_1 = elo.goal_difference_multiplier(1)
        mult_3 = elo.goal_difference_multiplier(3)

        assert mult_3 > mult_1


class TestPoissonModel:
    """Tests for Poisson goal distribution model."""

    def test_expected_goals_calculation(self):
        """Test expected goals calculation."""
        poisson = PoissonModel()

        # Strong attack vs weak defense should produce high expected
        home_exp, away_exp = poisson.calculate_expected_goals(
            home_attack=1.5,
            home_defense=0.8,
            away_attack=1.0,
            away_defense=1.2,
        )

        assert home_exp > away_exp
        assert home_exp > 1.0

    def test_match_probabilities_sum_to_one(self):
        """Test probabilities sum to 1."""
        poisson = PoissonModel()

        home_prob, draw_prob, away_prob = poisson.match_probabilities(1.5, 1.2)

        assert abs(home_prob + draw_prob + away_prob - 1.0) < 0.001

    def test_over_under_probability(self):
        """Test over/under probabilities sum to 1."""
        poisson = PoissonModel()

        over_prob, under_prob = poisson.over_under_probability(1.5, 1.5, line=2.5)

        assert abs(over_prob + under_prob - 1.0) < 0.001

    def test_btts_probability(self):
        """Test BTTS probabilities sum to 1."""
        poisson = PoissonModel()

        btts_yes, btts_no = poisson.btts_probability(1.5, 1.5)

        assert abs(btts_yes + btts_no - 1.0) < 0.001

    def test_most_likely_scores(self):
        """Test most likely scores are reasonable."""
        poisson = PoissonModel()

        scores = poisson.most_likely_scores(1.5, 1.2, top_n=5)

        assert len(scores) == 5
        # Probabilities should be in descending order
        for i in range(len(scores) - 1):
            assert scores[i][2] >= scores[i + 1][2]


class TestKellyCalculator:
    """Tests for Kelly Criterion calculator."""

    def test_calculate_edge_value_bet(self):
        """Test edge calculation for value bet."""
        kelly = KellyCalculator()

        # 50% model probability, 2.5 odds (40% implied)
        edge = kelly.calculate_edge(0.5, 2.5)

        assert edge > 0  # Positive edge
        assert abs(edge - 0.1) < 0.01  # 10% edge

    def test_calculate_edge_no_value(self):
        """Test edge calculation for no-value bet."""
        kelly = KellyCalculator()

        # 40% model probability, 2.5 odds (40% implied)
        edge = kelly.calculate_edge(0.4, 2.5)

        assert abs(edge) < 0.01  # No significant edge

    def test_kelly_stake_positive_edge(self):
        """Test Kelly stake for positive edge bet."""
        kelly = KellyCalculator(KellyConfig(fraction=1.0, min_edge=0.0))

        # 60% probability at 2.0 odds
        stake = kelly.calculate_kelly_stake(0.6, 2.0)

        assert stake > 0
        assert stake <= 1.0

    def test_kelly_stake_negative_edge(self):
        """Test Kelly stake for negative edge bet."""
        kelly = KellyCalculator()

        # 30% probability at 2.0 odds (50% implied)
        stake = kelly.calculate_kelly_stake(0.3, 2.0)

        assert stake == 0  # No bet on negative edge

    def test_fractional_kelly(self):
        """Test fractional Kelly reduces stake."""
        kelly_full = KellyCalculator(KellyConfig(fraction=1.0, min_edge=0.0, min_stake=0.0))
        kelly_quarter = KellyCalculator(KellyConfig(fraction=0.25, min_edge=0.0, min_stake=0.0))

        full_stake = kelly_full.calculate_kelly_stake(0.6, 2.0)
        quarter_stake = kelly_quarter.calculate_kelly_stake(0.6, 2.0)

        assert abs(quarter_stake - full_stake * 0.25) < 0.001

    def test_implied_probability(self):
        """Test implied probability calculation."""
        kelly = KellyCalculator()

        implied = kelly.implied_probability(2.0)
        assert abs(implied - 0.5) < 0.001

        implied = kelly.implied_probability(4.0)
        assert abs(implied - 0.25) < 0.001

    def test_is_value_bet(self):
        """Test value bet detection."""
        kelly = KellyCalculator(KellyConfig(min_edge=0.05))

        # 55% probability at 2.0 odds (50% implied) = 5% edge
        assert kelly.is_value_bet(0.55, 2.0)

        # 52% probability at 2.0 odds = 2% edge (below threshold)
        assert not kelly.is_value_bet(0.52, 2.0)
