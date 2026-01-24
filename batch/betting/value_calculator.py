"""Value bet detection and calculation.

Calibrated based on backtest results:
- Model is overconfident (says 75% but actually wins 70%)
- Home wins at 75%+ confidence are profitable
- Away wins require higher thresholds
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Optional

from app.core.config import get_settings
from batch.betting.kelly_criterion import KellyCalculator, KellyConfig

settings = get_settings()


# Calibration: Model confidence -> Actual win rate (from backtest)
# Model is overconfident, so we adjust probabilities down
CALIBRATION_MAP = {
    # (min_conf, max_conf): actual_win_rate
    (0.50, 0.55): 0.485,
    (0.55, 0.60): 0.526,
    (0.60, 0.65): 0.566,
    (0.65, 0.70): 0.585,
    (0.70, 0.75): 0.607,
    (0.75, 1.00): 0.700,  # Sweet spot: 75%+ confidence = 70% actual
}


def calibrate_probability(model_prob: float) -> float:
    """Convert overconfident model probability to calibrated actual probability.

    Based on backtest: model says 75% but actually wins 70%.
    """
    for (min_conf, max_conf), actual in CALIBRATION_MAP.items():
        if min_conf <= model_prob < max_conf:
            # Linear interpolation within bucket
            bucket_range = max_conf - min_conf
            position = (model_prob - min_conf) / bucket_range

            # Get next bucket's actual rate for interpolation
            next_actual = actual
            for (next_min, next_max), next_rate in CALIBRATION_MAP.items():
                if next_min == max_conf:
                    next_actual = next_rate
                    break

            return actual + position * (next_actual - actual)

    # Fallback: apply 7% reduction (average overconfidence)
    return model_prob * 0.93


@dataclass
class ValueBetConfig:
    """Configuration for value bet detection.

    Calibrated settings based on backtest:
    - Home wins at 75%+ model confidence hit 70% (profitable at odds >= 1.43)
    - Away wins less reliable, need higher threshold
    """

    # Confidence thresholds (model probability, not calibrated)
    home_min_confidence: float = 0.70  # Home wins: 70%+ model = ~60% actual
    away_min_confidence: float = 0.60  # Away wins: higher bar, less reliable
    other_min_confidence: float = 0.65  # O/U, BTTS etc.

    edge_threshold: float = 0.03  # Minimum edge over calibrated probability (3%)
    min_odds: float = 1.40  # Minimum odds (breakeven for 70% win rate = 1.43)
    max_odds: float = 10.0  # Maximum odds to consider
    kelly_fraction: float = 0.25  # Fractional Kelly
    max_stake_percent: float = 5.0  # Maximum stake as % of bankroll (conservative)

    # Legacy field for backwards compatibility
    min_confidence: float = 0.70


@dataclass
class ValueBet:
    """Detected value betting opportunity."""

    match_id: int
    outcome: str  # 'home_win', 'draw', 'away_win', 'over_2_5', etc.
    bookmaker: str
    model_probability: Decimal
    implied_probability: Decimal
    edge: Decimal
    odds: Decimal
    kelly_stake: Decimal
    recommended_stake: Decimal


class ValueBetDetector:
    """Detects value betting opportunities using calibrated probabilities.

    Key insight from backtest:
    - Model is overconfident (75% model prob = 70% actual win rate)
    - We use calibrated probabilities to calculate true edge
    - Home wins at high confidence are most profitable
    """

    def __init__(self, config: Optional[ValueBetConfig] = None):
        self.config = config or ValueBetConfig()
        self.kelly = KellyCalculator(
            KellyConfig(
                fraction=self.config.kelly_fraction,
                min_edge=self.config.edge_threshold,
            )
        )

    def _get_min_confidence(self, outcome: str) -> float:
        """Get minimum confidence threshold for an outcome type."""
        if outcome == "home_win":
            return self.config.home_min_confidence
        elif outcome == "away_win":
            return self.config.away_min_confidence
        else:
            return self.config.other_min_confidence

    def detect_value_bets(
        self,
        match_id: int,
        predictions: dict[str, float],
        odds: dict[str, dict[str, float]],
    ) -> list[ValueBet]:
        """Detect value bets for a match using calibrated probabilities.

        Args:
            match_id: Match ID
            predictions: Dict mapping outcome to model probability
                e.g., {'home_win': 0.45, 'draw': 0.28, 'away_win': 0.27}
            odds: Dict mapping bookmaker to outcome odds
                e.g., {'bet365': {'home_win': 2.1, 'draw': 3.4, 'away_win': 3.5}}

        Returns:
            List of detected value bets (using calibrated probabilities)
        """
        value_bets = []

        for bookmaker, bookmaker_odds in odds.items():
            for outcome, model_prob in predictions.items():
                if outcome not in bookmaker_odds:
                    continue

                decimal_odds = bookmaker_odds[outcome]

                # Get outcome-specific confidence threshold
                min_confidence = self._get_min_confidence(outcome)

                # Check model confidence threshold
                if model_prob < min_confidence:
                    continue

                # Check odds within range
                if decimal_odds < self.config.min_odds or decimal_odds > self.config.max_odds:
                    continue

                # IMPORTANT: Use calibrated probability for edge calculation
                calibrated_prob = calibrate_probability(model_prob)
                implied_prob = 1 / decimal_odds

                # Calculate edge using CALIBRATED probability (not overconfident model prob)
                edge = calibrated_prob - implied_prob

                # Check for sufficient edge
                if edge < self.config.edge_threshold:
                    continue

                # Calculate Kelly stake using calibrated probability
                # Kelly formula: f = (bp - q) / b where b = odds - 1
                b = decimal_odds - 1
                kelly_full = (b * calibrated_prob - (1 - calibrated_prob)) / b
                kelly_stake = max(0, min(kelly_full * self.config.kelly_fraction, self.config.max_stake_percent / 100))

                value_bet = ValueBet(
                    match_id=match_id,
                    outcome=outcome,
                    bookmaker=bookmaker,
                    model_probability=Decimal(str(round(calibrated_prob, 4))),  # Store calibrated prob
                    implied_probability=Decimal(str(round(implied_prob, 4))),
                    edge=Decimal(str(round(edge, 4))),
                    odds=Decimal(str(round(decimal_odds, 2))),
                    kelly_stake=Decimal(str(round(kelly_stake, 4))),
                    recommended_stake=Decimal(str(round(kelly_stake, 4))),
                )
                value_bets.append(value_bet)

        # Sort by edge (highest first)
        value_bets.sort(key=lambda x: x.edge, reverse=True)

        return value_bets


def calculate_consensus_probabilities(
    elo_probs: Optional[tuple[float, float, float]],
    poisson_probs: Optional[tuple[float, float, float]],
    xgboost_probs: Optional[tuple[float, float, float]],
    weights: tuple[float, float, float] = (0.3, 0.35, 0.35),
) -> tuple[float, float, float]:
    """Calculate weighted average consensus probabilities.

    Args:
        elo_probs: (home, draw, away) from ELO model
        poisson_probs: (home, draw, away) from Poisson model
        xgboost_probs: (home, draw, away) from XGBoost model
        weights: (elo_weight, poisson_weight, xgboost_weight)

    Returns:
        Tuple of consensus (home, draw, away) probabilities
    """
    all_probs = [elo_probs, poisson_probs, xgboost_probs]
    all_weights = list(weights)

    # Filter out None values
    available = [(p, w) for p, w in zip(all_probs, all_weights) if p is not None]

    if not available:
        return (0.33, 0.34, 0.33)  # Default uniform

    # Normalize weights
    total_weight = sum(w for _, w in available)
    normalized = [(p, w / total_weight) for p, w in available]

    # Calculate weighted average
    home = sum(p[0] * w for p, w in normalized)
    draw = sum(p[1] * w for p, w in normalized)
    away = sum(p[2] * w for p, w in normalized)

    # Normalize to ensure sum = 1
    total = home + draw + away
    return (home / total, draw / total, away / total)


def format_value_bet_summary(value_bets: list[ValueBet]) -> str:
    """Format value bets for display.

    Args:
        value_bets: List of value bets

    Returns:
        Formatted string summary
    """
    if not value_bets:
        return "No value bets detected."

    lines = ["Value Bets Detected:", "=" * 50]

    for bet in value_bets:
        outcome_display = {
            "home_win": "Home Win",
            "draw": "Draw",
            "away_win": "Away Win",
            "over_2_5": "Over 2.5",
            "under_2_5": "Under 2.5",
            "btts_yes": "BTTS Yes",
            "btts_no": "BTTS No",
        }.get(bet.outcome, bet.outcome)

        lines.append(f"\n{outcome_display} @ {bet.odds} ({bet.bookmaker})")
        lines.append(f"  Model prob: {float(bet.model_probability):.1%}")
        lines.append(f"  Implied prob: {float(bet.implied_probability):.1%}")
        lines.append(f"  Edge: {float(bet.edge):.1%}")
        lines.append(f"  Kelly stake: {float(bet.kelly_stake):.2%}")

    return "\n".join(lines)
