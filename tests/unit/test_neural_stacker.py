"""Unit tests for neural stacker prediction logic."""

import pytest
from decimal import Decimal
from unittest.mock import Mock, MagicMock
from typing import Optional

# Skip all tests if torch is not available (not installed in CI)
torch = pytest.importorskip("torch", reason="torch not installed")
from batch.models.neural_stacker import NeuralStacker


class MockMatchAnalysis:
    """Mock MatchAnalysis for testing without database."""

    def __init__(
        self,
        elo_home_prob: Optional[Decimal] = None,
        elo_draw_prob: Optional[Decimal] = None,
        elo_away_prob: Optional[Decimal] = None,
        poisson_home_prob: Optional[Decimal] = None,
        poisson_draw_prob: Optional[Decimal] = None,
        poisson_away_prob: Optional[Decimal] = None,
        features: Optional[dict] = None,
    ):
        self.elo_home_prob = elo_home_prob
        self.elo_draw_prob = elo_draw_prob
        self.elo_away_prob = elo_away_prob
        self.poisson_home_prob = poisson_home_prob
        self.poisson_draw_prob = poisson_draw_prob
        self.poisson_away_prob = poisson_away_prob
        self.features = features


class MockTeamStats:
    """Mock TeamStats for testing without database."""

    def __init__(
        self,
        home_wins: int = 0,
        home_draws: int = 0,
        home_losses: int = 0,
        away_wins: int = 0,
        away_draws: int = 0,
        away_losses: int = 0,
    ):
        self.home_wins = home_wins
        self.home_draws = home_draws
        self.home_losses = home_losses
        self.away_wins = away_wins
        self.away_draws = away_draws
        self.away_losses = away_losses


class MockEloRating:
    """Mock EloRating for testing without database."""

    def __init__(self, rating: Decimal = Decimal("1500.00")):
        self.rating = rating


class TestFallbackPrediction:
    """Tests for _fallback_prediction method."""

    @pytest.fixture
    def stacker(self):
        """Create a NeuralStacker without loading model."""
        stacker = NeuralStacker.__new__(NeuralStacker)
        stacker.model = None
        stacker.device = "cpu"
        stacker.draw_threshold = 0.08
        stacker.temperature = 1.0
        return stacker

    def test_fallback_with_market_odds_returns_market_weighted(self, stacker):
        """Market odds should dominate (80%) when available."""
        analysis = MockMatchAnalysis(
            elo_home_prob=Decimal("0.50"),
            elo_draw_prob=Decimal("0.25"),
            elo_away_prob=Decimal("0.25"),
            features={
                "historical_odds": {
                    "implied_home_prob": 0.40,
                    "implied_draw_prob": 0.30,
                    "implied_away_prob": 0.30,
                }
            }
        )

        home, draw, away = stacker._fallback_prediction(analysis)

        # Should be closer to market (0.40) than ELO (0.50)
        assert home < 0.45  # Market pull towards 0.40
        assert abs(home + draw + away - 1.0) < 0.001

    def test_fallback_without_market_odds_uses_elo_poisson(self, stacker):
        """Without market odds, should blend ELO (45%) and Poisson (55%)."""
        analysis = MockMatchAnalysis(
            elo_home_prob=Decimal("0.50"),
            elo_draw_prob=Decimal("0.25"),
            elo_away_prob=Decimal("0.25"),
            poisson_home_prob=Decimal("0.40"),
            poisson_draw_prob=Decimal("0.30"),
            poisson_away_prob=Decimal("0.30"),
            features=None,  # No market odds
        )

        home, draw, away = stacker._fallback_prediction(analysis)

        # ELO: 0.50, Poisson: 0.40, weighted: 0.50*0.45 + 0.40*0.55 = 0.445
        expected_home = 0.50 * 0.45 + 0.40 * 0.55
        assert abs(home - expected_home) < 0.01
        assert abs(home + draw + away - 1.0) < 0.001

    def test_fallback_with_missing_elo_uses_defaults(self, stacker):
        """Missing ELO should use market odds as defaults."""
        analysis = MockMatchAnalysis(
            elo_home_prob=None,
            elo_draw_prob=None,
            elo_away_prob=None,
            features={
                "historical_odds": {
                    "implied_home_prob": 0.45,
                    "implied_draw_prob": 0.28,
                    "implied_away_prob": 0.27,
                }
            }
        )

        home, draw, away = stacker._fallback_prediction(analysis)

        # With no ELO, should fall back to market values for ELO portion
        # Result should be very close to pure market odds
        assert abs(home - 0.45) < 0.02
        assert abs(home + draw + away - 1.0) < 0.001

    def test_fallback_with_no_data_returns_uniform(self, stacker):
        """With no data, should return near-uniform distribution."""
        analysis = MockMatchAnalysis(
            elo_home_prob=None,
            elo_draw_prob=None,
            elo_away_prob=None,
            poisson_home_prob=None,
            poisson_draw_prob=None,
            poisson_away_prob=None,
            features=None,
        )

        home, draw, away = stacker._fallback_prediction(analysis)

        # Should be close to 0.33 each (using defaults)
        assert abs(home - 0.33) < 0.02
        assert abs(draw - 0.33) < 0.02
        assert abs(away - 0.33) < 0.02
        assert abs(home + draw + away - 1.0) < 0.001

    def test_fallback_probabilities_sum_to_one(self, stacker):
        """All fallback predictions must sum to 1.0."""
        test_cases = [
            # Strong home favorite
            MockMatchAnalysis(
                elo_home_prob=Decimal("0.70"),
                elo_draw_prob=Decimal("0.18"),
                elo_away_prob=Decimal("0.12"),
                poisson_home_prob=Decimal("0.65"),
                poisson_draw_prob=Decimal("0.20"),
                poisson_away_prob=Decimal("0.15"),
            ),
            # Even match
            MockMatchAnalysis(
                elo_home_prob=Decimal("0.35"),
                elo_draw_prob=Decimal("0.30"),
                elo_away_prob=Decimal("0.35"),
                poisson_home_prob=Decimal("0.33"),
                poisson_draw_prob=Decimal("0.34"),
                poisson_away_prob=Decimal("0.33"),
            ),
            # Away favorite with market odds
            MockMatchAnalysis(
                elo_home_prob=Decimal("0.25"),
                elo_draw_prob=Decimal("0.28"),
                elo_away_prob=Decimal("0.47"),
                features={
                    "historical_odds": {
                        "implied_home_prob": 0.22,
                        "implied_draw_prob": 0.26,
                        "implied_away_prob": 0.52,
                    }
                }
            ),
        ]

        for analysis in test_cases:
            home, draw, away = stacker._fallback_prediction(analysis)
            assert abs(home + draw + away - 1.0) < 0.001

    def test_fallback_market_odds_weight(self, stacker):
        """Verify 80/20 weighting between market and ELO."""
        # Create analysis where market and ELO disagree significantly
        analysis = MockMatchAnalysis(
            elo_home_prob=Decimal("0.60"),  # ELO says home favorite
            elo_draw_prob=Decimal("0.25"),
            elo_away_prob=Decimal("0.15"),
            features={
                "historical_odds": {
                    "implied_home_prob": 0.30,  # Market says slight away
                    "implied_draw_prob": 0.30,
                    "implied_away_prob": 0.40,
                }
            }
        )

        home, draw, away = stacker._fallback_prediction(analysis)

        # Expected: 0.30 * 0.8 + 0.60 * 0.2 = 0.36 (before normalization)
        # Market heavily influences result
        assert home < 0.40  # Closer to market's 0.30 than ELO's 0.60


class TestColdStartDetection:
    """Tests for _is_cold_start method."""

    @pytest.fixture
    def stacker(self):
        """Create a NeuralStacker without loading model."""
        stacker = NeuralStacker.__new__(NeuralStacker)
        stacker.model = None
        return stacker

    def test_cold_start_with_no_stats(self, stacker):
        """No team stats = cold start."""
        result = stacker._is_cold_start(None, None, None, None)
        assert result is True

    def test_cold_start_with_few_games(self, stacker):
        """Less than 3 games each = cold start."""
        home_stats = MockTeamStats(home_wins=1, away_draws=1)  # 2 games
        away_stats = MockTeamStats(home_wins=1)  # 1 game

        result = stacker._is_cold_start(home_stats, away_stats, None, None)
        assert result is True

    def test_not_cold_start_with_enough_games(self, stacker):
        """3+ games each = not cold start."""
        home_stats = MockTeamStats(home_wins=2, away_wins=1)  # 3 games
        away_stats = MockTeamStats(home_wins=1, home_draws=1, away_wins=1)  # 3 games

        result = stacker._is_cold_start(home_stats, away_stats, None, None)
        assert result is False

    def test_not_cold_start_one_team_many_games(self, stacker):
        """One team with many games = not cold start (AND logic)."""
        home_stats = MockTeamStats(home_wins=5, away_wins=4)  # 9 games
        away_stats = MockTeamStats()  # 0 games

        # Current logic: cold start only if BOTH have < 3 games
        # If one team has enough data, we can use neural stacker
        result = stacker._is_cold_start(home_stats, away_stats, None, None)
        assert result is False

    def test_not_cold_start_one_team_with_stats(self, stacker):
        """One team with enough stats = not cold start."""
        home_stats = MockTeamStats(home_wins=2, home_draws=1, away_wins=2)  # 5 games

        result = stacker._is_cold_start(home_stats, None, None, None)
        # Home has 5 games >= 3, so not cold start (even if away is None = 0 games)
        assert result is False

    def test_not_cold_start_mid_season(self, stacker):
        """Mid-season stats = not cold start."""
        home_stats = MockTeamStats(
            home_wins=6, home_draws=2, home_losses=1,
            away_wins=4, away_draws=3, away_losses=2
        )  # 18 games
        away_stats = MockTeamStats(
            home_wins=5, home_draws=3, home_losses=1,
            away_wins=5, away_draws=2, away_losses=2
        )  # 18 games

        result = stacker._is_cold_start(home_stats, away_stats, None, None)
        assert result is False


class TestSmartDrawPrediction:
    """Tests for _apply_smart_draw method."""

    @pytest.fixture
    def stacker(self):
        """Create a NeuralStacker with draw threshold."""
        stacker = NeuralStacker.__new__(NeuralStacker)
        stacker.draw_threshold = 0.08
        return stacker

    def test_smart_draw_boosts_close_match(self, stacker):
        """Draw should be boosted when home/away are close."""
        probs = (0.38, 0.28, 0.34)  # Close match, diff = 0.04 < 0.08

        result = stacker._apply_smart_draw(probs)

        # Draw should be boosted
        assert result[1] > probs[1]
        # Home and away should be reduced
        assert result[0] < probs[0]
        assert result[2] < probs[2]
        # Still sums to 1
        assert abs(sum(result) - 1.0) < 0.001

    def test_smart_draw_no_boost_for_clear_favorite(self, stacker):
        """No draw boost when there's a clear favorite."""
        probs = (0.55, 0.25, 0.20)  # Clear home favorite, diff = 0.35 > 0.08

        result = stacker._apply_smart_draw(probs)

        # Should return unchanged
        assert result == probs

    def test_smart_draw_no_boost_low_draw_prob(self, stacker):
        """No draw boost when draw probability is low."""
        probs = (0.40, 0.20, 0.40)  # Even match but draw < 25%

        result = stacker._apply_smart_draw(probs)

        # Should return unchanged because draw < 0.25
        assert result == probs

    def test_smart_draw_max_boost_for_equal_probs(self, stacker):
        """Maximum boost when home and away are equal."""
        probs = (0.35, 0.30, 0.35)  # Perfectly equal, diff = 0

        result = stacker._apply_smart_draw(probs)

        # Maximum 8% boost to draw
        # closeness = 1 - 0/0.08 = 1.0, boost = 0.08 * 1.0 = 0.08
        assert result[1] > probs[1]
        assert result[1] - probs[1] < 0.085  # Max ~8% boost
        assert abs(sum(result) - 1.0) < 0.001

    def test_smart_draw_partial_boost(self, stacker):
        """Partial boost when home/away are somewhat close."""
        probs = (0.38, 0.28, 0.34)  # diff = 0.04, closeness = 0.5

        result = stacker._apply_smart_draw(probs)

        # Partial boost: closeness = 1 - 0.04/0.08 = 0.5, boost = 0.08 * 0.5 = 0.04
        boost = result[1] - probs[1]
        assert 0.03 < boost < 0.05  # ~4% boost
        assert abs(sum(result) - 1.0) < 0.001

    def test_smart_draw_probabilities_valid(self, stacker):
        """All probabilities remain valid (0-1) after adjustment."""
        test_cases = [
            (0.36, 0.28, 0.36),  # Equal
            (0.38, 0.26, 0.36),  # Slight difference
            (0.37, 0.30, 0.33),  # Different draw
        ]

        for probs in test_cases:
            result = stacker._apply_smart_draw(probs)

            assert all(0 <= p <= 1 for p in result)
            assert abs(sum(result) - 1.0) < 0.001


class TestModelAgreement:
    """Tests for model agreement calculation logic."""

    def test_models_agree_all_same(self):
        """All models predict same outcome = agreement."""
        import numpy as np

        elo_probs = [0.50, 0.25, 0.25]  # Home favorite
        poisson_probs = [0.48, 0.27, 0.25]  # Home favorite
        market_probs = [0.52, 0.24, 0.24]  # Home favorite

        elo_pred = np.argmax(elo_probs)
        poisson_pred = np.argmax(poisson_probs)
        market_pred = np.argmax(market_probs)

        assert elo_pred == poisson_pred == market_pred == 0

    def test_models_disagree_different_outcomes(self):
        """Models predict different outcomes = disagreement."""
        import numpy as np

        elo_probs = [0.45, 0.25, 0.30]  # Home favorite
        poisson_probs = [0.30, 0.35, 0.35]  # Draw favorite
        market_probs = [0.25, 0.30, 0.45]  # Away favorite

        elo_pred = np.argmax(elo_probs)
        poisson_pred = np.argmax(poisson_probs)
        market_pred = np.argmax(market_probs)

        assert elo_pred != poisson_pred or poisson_pred != market_pred

    def test_models_agree_on_draw(self):
        """All models can agree on draw prediction."""
        import numpy as np

        elo_probs = [0.30, 0.40, 0.30]  # Draw favorite
        poisson_probs = [0.28, 0.42, 0.30]  # Draw favorite
        market_probs = [0.32, 0.38, 0.30]  # Draw favorite

        elo_pred = np.argmax(elo_probs)
        poisson_pred = np.argmax(poisson_probs)
        market_pred = np.argmax(market_probs)

        assert elo_pred == poisson_pred == market_pred == 1


class TestProbabilityNormalization:
    """Tests to ensure all predictions sum to 1.0."""

    @pytest.fixture
    def stacker(self):
        """Create a NeuralStacker without loading model."""
        stacker = NeuralStacker.__new__(NeuralStacker)
        stacker.model = None
        stacker.draw_threshold = 0.08
        return stacker

    def test_normalization_after_fallback(self, stacker):
        """Fallback predictions must be normalized."""
        # Test with various edge cases
        analyses = [
            MockMatchAnalysis(
                elo_home_prob=Decimal("0.45"),
                elo_draw_prob=Decimal("0.28"),
                elo_away_prob=Decimal("0.27"),
                poisson_home_prob=Decimal("0.50"),
                poisson_draw_prob=Decimal("0.25"),
                poisson_away_prob=Decimal("0.25"),
            ),
            MockMatchAnalysis(
                features={
                    "historical_odds": {
                        "implied_home_prob": 0.45,
                        "implied_draw_prob": 0.28,
                        "implied_away_prob": 0.27,
                    }
                }
            ),
        ]

        for analysis in analyses:
            home, draw, away = stacker._fallback_prediction(analysis)
            total = home + draw + away
            assert abs(total - 1.0) < 0.001, f"Sum was {total}, expected 1.0"

    def test_normalization_after_smart_draw(self, stacker):
        """Smart draw adjustments must maintain normalization."""
        test_probs = [
            (0.36, 0.28, 0.36),
            (0.38, 0.30, 0.32),
            (0.35, 0.30, 0.35),
            (0.40, 0.26, 0.34),
        ]

        for probs in test_probs:
            result = stacker._apply_smart_draw(probs)
            total = sum(result)
            assert abs(total - 1.0) < 0.001, f"Sum was {total}, expected 1.0"
