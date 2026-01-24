"""Kelly Criterion calculator for optimal bet sizing."""

from dataclasses import dataclass
from decimal import Decimal
from typing import Optional


@dataclass
class KellyConfig:
    """Configuration for Kelly Criterion calculations."""

    fraction: float = 0.25  # Fractional Kelly (risk reduction)
    max_stake: float = 0.1  # Maximum stake as fraction of bankroll
    min_stake: float = 0.01  # Minimum stake to consider
    min_edge: float = 0.05  # Minimum edge required


class KellyCalculator:
    """Kelly Criterion calculator for optimal bet sizing.

    The Kelly Criterion determines the optimal stake size to maximize
    long-term growth rate while managing risk.

    Formula: f* = (bp - q) / b

    Where:
    - f* = fraction of bankroll to bet
    - b = decimal odds - 1 (net odds)
    - p = probability of winning
    - q = probability of losing (1 - p)
    """

    def __init__(self, config: Optional[KellyConfig] = None):
        self.config = config or KellyConfig()

    def calculate_kelly_stake(
        self,
        probability: float,
        decimal_odds: float,
    ) -> float:
        """Calculate the Kelly stake for a bet.

        Args:
            probability: Estimated probability of winning (0-1)
            decimal_odds: Decimal odds offered by bookmaker (e.g., 2.5)

        Returns:
            Optimal stake as fraction of bankroll (0-1)
        """
        if probability <= 0 or probability >= 1:
            return 0.0

        if decimal_odds <= 1:
            return 0.0

        # Kelly formula
        b = decimal_odds - 1  # Net odds
        p = probability
        q = 1 - p

        kelly = (b * p - q) / b

        # Apply fractional Kelly for risk management
        kelly *= self.config.fraction

        # Apply constraints
        if kelly < self.config.min_stake:
            return 0.0

        return min(kelly, self.config.max_stake)

    def calculate_edge(
        self,
        probability: float,
        decimal_odds: float,
    ) -> float:
        """Calculate the edge (expected value) of a bet.

        Args:
            probability: Estimated probability of winning (0-1)
            decimal_odds: Decimal odds offered by bookmaker

        Returns:
            Edge as a decimal (e.g., 0.05 = 5% edge)
        """
        if probability <= 0 or decimal_odds <= 1:
            return 0.0

        # Implied probability from odds
        implied_prob = 1 / decimal_odds

        # Edge = our probability - implied probability
        return probability - implied_prob

    def implied_probability(self, decimal_odds: float) -> float:
        """Convert decimal odds to implied probability.

        Args:
            decimal_odds: Decimal odds (e.g., 2.5)

        Returns:
            Implied probability (0-1)
        """
        if decimal_odds <= 1:
            return 1.0
        return 1 / decimal_odds

    def is_value_bet(
        self,
        probability: float,
        decimal_odds: float,
    ) -> bool:
        """Determine if a bet represents value.

        Args:
            probability: Estimated probability of winning
            decimal_odds: Decimal odds offered

        Returns:
            True if the bet has positive expected value above threshold
        """
        edge = self.calculate_edge(probability, decimal_odds)
        return edge >= self.config.min_edge

    def calculate_expected_value(
        self,
        probability: float,
        decimal_odds: float,
        stake: float = 1.0,
    ) -> float:
        """Calculate expected value of a bet.

        Args:
            probability: Estimated probability of winning
            decimal_odds: Decimal odds offered
            stake: Stake amount

        Returns:
            Expected value (profit/loss)
        """
        # EV = (p * win_amount) - (q * stake)
        win_amount = stake * (decimal_odds - 1)
        lose_amount = stake

        ev = (probability * win_amount) - ((1 - probability) * lose_amount)
        return ev


def calculate_optimal_stakes(
    bets: list[dict],
    bankroll: float = 1000.0,
    kelly_fraction: float = 0.25,
) -> list[dict]:
    """Calculate optimal stakes for multiple bets.

    Args:
        bets: List of bet dicts with 'probability' and 'odds'
        bankroll: Total bankroll amount
        kelly_fraction: Fraction of Kelly to use

    Returns:
        List of bets with added 'kelly_stake' and 'recommended_stake' fields
    """
    calculator = KellyCalculator(KellyConfig(fraction=kelly_fraction))

    results = []
    for bet in bets:
        prob = bet["probability"]
        odds = bet["odds"]

        kelly = calculator.calculate_kelly_stake(prob, odds)
        edge = calculator.calculate_edge(prob, odds)

        result = bet.copy()
        result["kelly_stake"] = kelly
        result["recommended_stake"] = kelly * bankroll if kelly > 0 else 0
        result["edge"] = edge
        result["is_value"] = calculator.is_value_bet(prob, odds)
        result["expected_value"] = calculator.calculate_expected_value(prob, odds, result["recommended_stake"])

        results.append(result)

    return results
