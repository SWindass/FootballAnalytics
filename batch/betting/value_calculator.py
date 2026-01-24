"""Value bet detection and calculation."""

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Optional

from app.core.config import get_settings
from batch.betting.kelly_criterion import KellyCalculator, KellyConfig

settings = get_settings()


@dataclass
class ValueBetConfig:
    """Configuration for value bet detection."""

    edge_threshold: float = 0.05  # Minimum edge (5%)
    min_odds: float = 1.5  # Minimum odds to consider
    max_odds: float = 10.0  # Maximum odds to consider
    kelly_fraction: float = 0.25  # Fractional Kelly
    max_stake_percent: float = 10.0  # Maximum stake as % of bankroll


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
    """Detects value betting opportunities by comparing model probabilities to odds."""

    def __init__(self, config: Optional[ValueBetConfig] = None):
        self.config = config or ValueBetConfig(
            edge_threshold=settings.edge_threshold,
            min_odds=settings.min_odds,
            max_odds=settings.max_odds,
            kelly_fraction=settings.kelly_fraction,
        )
        self.kelly = KellyCalculator(
            KellyConfig(
                fraction=self.config.kelly_fraction,
                min_edge=self.config.edge_threshold,
            )
        )

    def detect_value_bets(
        self,
        match_id: int,
        predictions: dict[str, float],
        odds: dict[str, dict[str, float]],
    ) -> list[ValueBet]:
        """Detect value bets for a match.

        Args:
            match_id: Match ID
            predictions: Dict mapping outcome to probability
                e.g., {'home_win': 0.45, 'draw': 0.28, 'away_win': 0.27}
            odds: Dict mapping bookmaker to outcome odds
                e.g., {'bet365': {'home_win': 2.1, 'draw': 3.4, 'away_win': 3.5}}

        Returns:
            List of detected value bets
        """
        value_bets = []

        for bookmaker, bookmaker_odds in odds.items():
            for outcome, model_prob in predictions.items():
                if outcome not in bookmaker_odds:
                    continue

                decimal_odds = bookmaker_odds[outcome]

                # Check odds within range
                if decimal_odds < self.config.min_odds or decimal_odds > self.config.max_odds:
                    continue

                # Check for value
                if not self.kelly.is_value_bet(model_prob, decimal_odds):
                    continue

                # Calculate stake
                kelly_stake = self.kelly.calculate_kelly_stake(model_prob, decimal_odds)
                edge = self.kelly.calculate_edge(model_prob, decimal_odds)
                implied_prob = self.kelly.implied_probability(decimal_odds)

                value_bet = ValueBet(
                    match_id=match_id,
                    outcome=outcome,
                    bookmaker=bookmaker,
                    model_probability=Decimal(str(round(model_prob, 4))),
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
