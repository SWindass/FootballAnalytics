"""
Dynamic Seasonal Recalibration for Match Prediction Models.

Addresses the 2025-26 performance drop (47.6% vs 54.6% historical) by:
1. Tracking rolling match statistics (draw rate, goals, home advantage)
2. Comparing to historical baselines
3. Dynamically adjusting Dixon-Coles rho and draw probability

Key Finding: 2025-26 has elevated draw rate (42% in recent matches vs 23.4% historical)
which causes models to underestimate draws.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class MatchResult:
    """Minimal match result for tracking statistics."""
    home_goals: int
    away_goals: int
    home_xg: float
    away_xg: float

    @property
    def is_draw(self) -> bool:
        return self.home_goals == self.away_goals

    @property
    def is_home_win(self) -> bool:
        return self.home_goals > self.away_goals

    @property
    def is_away_win(self) -> bool:
        return self.home_goals < self.away_goals

    @property
    def total_goals(self) -> int:
        return self.home_goals + self.away_goals


@dataclass
class LeagueBaseline:
    """Historical baseline statistics for comparison."""
    draw_rate: float = 0.234  # Historical EPL draw rate
    home_win_rate: float = 0.447  # Historical home win rate
    away_win_rate: float = 0.319  # Historical away win rate
    goals_per_game: float = 2.68  # Historical goals per game
    avg_xg_per_team: float = 1.35  # Historical xG per team


@dataclass
class RecalibrationFactors:
    """Calculated adjustment factors for model parameters."""
    rho_adjustment: float = 1.0  # Multiplier for Dixon-Coles rho
    draw_boost: float = 1.0  # Multiplier for draw probability
    avg_goals_adjustment: float = 1.0  # Adjustment for expected goals
    home_advantage_adjustment: float = 1.0  # Adjustment for home advantage

    # Diagnostic info
    recent_draw_rate: float = 0.0
    recent_home_win_rate: float = 0.0
    recent_goals_per_game: float = 0.0
    deviation_from_baseline: str = ""


class SeasonalRecalibration:
    """
    Dynamic recalibration based on recent match statistics.

    Tracks rolling statistics over recent matches and adjusts model
    parameters when significant deviations from historical norms occur.

    Parameters
    ----------
    window_size : int
        Number of recent matches to track (default: 50)
    baseline : LeagueBaseline
        Historical baseline statistics for comparison
    sensitivity : float
        How sensitive to deviations (0.5 = less reactive, 2.0 = more reactive)
    """

    def __init__(
        self,
        window_size: int = 50,
        baseline: Optional[LeagueBaseline] = None,
        sensitivity: float = 1.0,
    ):
        self.window_size = window_size
        self.baseline = baseline or LeagueBaseline()
        self.sensitivity = sensitivity

        # Rolling window of recent matches
        self._recent_matches: deque[MatchResult] = deque(maxlen=window_size)

        # Thresholds for triggering recalibration
        self._draw_rate_threshold = 0.03  # Trigger if draw rate differs by 3pp
        self._goals_threshold = 0.20  # Trigger if goals differ by 0.2
        self._home_advantage_threshold = 0.05  # Trigger if home win rate differs by 5pp

    def add_match(
        self,
        home_goals: int,
        away_goals: int,
        home_xg: float = 0.0,
        away_xg: float = 0.0,
    ) -> None:
        """Add a completed match to tracking window."""
        self._recent_matches.append(MatchResult(
            home_goals=home_goals,
            away_goals=away_goals,
            home_xg=home_xg,
            away_xg=away_xg,
        ))

    def get_current_statistics(self) -> dict:
        """Calculate current rolling statistics."""
        if len(self._recent_matches) < 10:
            return {
                'draw_rate': self.baseline.draw_rate,
                'home_win_rate': self.baseline.home_win_rate,
                'away_win_rate': self.baseline.away_win_rate,
                'goals_per_game': self.baseline.goals_per_game,
                'sample_size': len(self._recent_matches),
            }

        n = len(self._recent_matches)
        draws = sum(1 for m in self._recent_matches if m.is_draw)
        home_wins = sum(1 for m in self._recent_matches if m.is_home_win)
        away_wins = sum(1 for m in self._recent_matches if m.is_away_win)
        total_goals = sum(m.total_goals for m in self._recent_matches)

        return {
            'draw_rate': draws / n,
            'home_win_rate': home_wins / n,
            'away_win_rate': away_wins / n,
            'goals_per_game': total_goals / n,
            'sample_size': n,
        }

    def calculate_recalibration(self) -> RecalibrationFactors:
        """
        Calculate recalibration factors based on recent vs historical.

        Returns
        -------
        RecalibrationFactors
            Adjustment factors for model parameters
        """
        stats = self.get_current_statistics()
        factors = RecalibrationFactors()

        # Store recent stats for diagnostics
        factors.recent_draw_rate = stats['draw_rate']
        factors.recent_home_win_rate = stats['home_win_rate']
        factors.recent_goals_per_game = stats['goals_per_game']

        if stats['sample_size'] < 20:
            factors.deviation_from_baseline = "Insufficient data for recalibration"
            return factors

        deviations = []

        # 1. Draw rate adjustment
        draw_diff = stats['draw_rate'] - self.baseline.draw_rate
        if abs(draw_diff) > self._draw_rate_threshold:
            # Calculate how much to adjust
            # If draws are 42% vs 23.4% baseline, that's 80% more draws
            draw_ratio = stats['draw_rate'] / self.baseline.draw_rate

            # Apply sensitivity-scaled adjustment
            # For elevated draws: boost draw probability and increase |rho|
            if draw_diff > 0:
                # Draw probability boost based on ratio
                factors.draw_boost = 1.0 + (draw_ratio - 1.0) * self.sensitivity

                # Rho adjustment: more negative rho = more low-score correlation = more draws
                # Increase magnitude by similar factor
                factors.rho_adjustment = 1.0 + (draw_ratio - 1.0) * self.sensitivity * 0.3

                deviations.append(f"Elevated draws (+{draw_diff*100:.1f}pp)")
            else:
                # Fewer draws: reduce boost and |rho|
                factors.draw_boost = max(0.7, draw_ratio)
                factors.rho_adjustment = max(0.7, draw_ratio)
                deviations.append(f"Reduced draws ({draw_diff*100:.1f}pp)")

        # 2. Goals per game adjustment
        goals_diff = stats['goals_per_game'] - self.baseline.goals_per_game
        if abs(goals_diff) > self._goals_threshold:
            goals_ratio = stats['goals_per_game'] / self.baseline.goals_per_game
            factors.avg_goals_adjustment = goals_ratio

            if goals_diff < 0:
                deviations.append(f"Lower scoring ({goals_diff:.2f} goals/game)")
            else:
                deviations.append(f"Higher scoring (+{goals_diff:.2f} goals/game)")

        # 3. Home advantage adjustment
        home_diff = stats['home_win_rate'] - self.baseline.home_win_rate
        if abs(home_diff) > self._home_advantage_threshold:
            home_ratio = stats['home_win_rate'] / self.baseline.home_win_rate
            factors.home_advantage_adjustment = home_ratio

            if home_diff < 0:
                deviations.append(f"Weakened home advantage ({home_diff*100:.1f}pp)")
            else:
                deviations.append(f"Stronger home advantage (+{home_diff*100:.1f}pp)")

        factors.deviation_from_baseline = "; ".join(deviations) if deviations else "Within normal range"

        return factors

    def adjust_probabilities(
        self,
        home_prob: float,
        draw_prob: float,
        away_prob: float,
        factors: Optional[RecalibrationFactors] = None,
    ) -> tuple[float, float, float]:
        """
        Apply recalibration factors to predicted probabilities.

        Parameters
        ----------
        home_prob, draw_prob, away_prob : float
            Raw model probabilities
        factors : RecalibrationFactors, optional
            Pre-calculated factors (will compute if not provided)

        Returns
        -------
        tuple[float, float, float]
            Adjusted (home, draw, away) probabilities
        """
        if factors is None:
            factors = self.calculate_recalibration()

        # Apply draw boost
        draw_prob *= factors.draw_boost

        # Slightly reduce home/away to compensate
        adjustment_pool = (factors.draw_boost - 1.0) * draw_prob / (home_prob + away_prob + 0.001)
        home_prob -= home_prob * adjustment_pool * 0.5
        away_prob -= away_prob * adjustment_pool * 0.5

        # Apply home advantage adjustment
        if factors.home_advantage_adjustment < 1.0:
            # Weakened home advantage: reduce home, boost away
            shift = (1.0 - factors.home_advantage_adjustment) * 0.5
            home_prob *= (1.0 - shift)
            away_prob *= (1.0 + shift * 0.5)

        # Normalize
        total = home_prob + draw_prob + away_prob
        if total > 0:
            home_prob /= total
            draw_prob /= total
            away_prob /= total

        return home_prob, draw_prob, away_prob

    def adjust_rho(self, base_rho: float, factors: Optional[RecalibrationFactors] = None) -> float:
        """
        Adjust Dixon-Coles rho parameter based on recent statistics.

        Parameters
        ----------
        base_rho : float
            Base rho value (typically -0.11 to -0.13)
        factors : RecalibrationFactors, optional
            Pre-calculated factors

        Returns
        -------
        float
            Adjusted rho value
        """
        if factors is None:
            factors = self.calculate_recalibration()

        # More negative rho = more correlation = more low-score draws
        # factors.rho_adjustment > 1 when draws are elevated
        adjusted = base_rho * factors.rho_adjustment

        # Clamp to reasonable range
        return np.clip(adjusted, -0.25, 0.0)

    def adjust_avg_goals(
        self,
        base_avg: float,
        factors: Optional[RecalibrationFactors] = None,
    ) -> float:
        """
        Adjust league average goals based on recent statistics.

        Parameters
        ----------
        base_avg : float
            Base average goals per team (typically 1.35)
        factors : RecalibrationFactors, optional
            Pre-calculated factors

        Returns
        -------
        float
            Adjusted average goals
        """
        if factors is None:
            factors = self.calculate_recalibration()

        adjusted = base_avg * factors.avg_goals_adjustment

        # Clamp to reasonable range
        return np.clip(adjusted, 1.0, 1.7)

    def get_diagnostic_summary(self) -> str:
        """Get human-readable summary of current recalibration state."""
        stats = self.get_current_statistics()
        factors = self.calculate_recalibration()

        lines = [
            "=" * 60,
            "SEASONAL RECALIBRATION STATUS",
            "=" * 60,
            f"Sample size: {stats['sample_size']} matches",
            "",
            "Current vs Historical:",
            f"  Draw rate:    {stats['draw_rate']*100:.1f}% vs {self.baseline.draw_rate*100:.1f}% baseline",
            f"  Home win:     {stats['home_win_rate']*100:.1f}% vs {self.baseline.home_win_rate*100:.1f}% baseline",
            f"  Goals/game:   {stats['goals_per_game']:.2f} vs {self.baseline.goals_per_game:.2f} baseline",
            "",
            "Active Adjustments:",
            f"  Rho adjustment:     {factors.rho_adjustment:.2f}x (base -0.11 → {-0.11 * factors.rho_adjustment:.3f})",
            f"  Draw boost:         {factors.draw_boost:.2f}x",
            f"  Goals adjustment:   {factors.avg_goals_adjustment:.2f}x",
            f"  Home adv adjustment:{factors.home_advantage_adjustment:.2f}x",
            "",
            f"Deviation: {factors.deviation_from_baseline}",
            "=" * 60,
        ]

        return "\n".join(lines)

    def reset(self) -> None:
        """Clear tracking window."""
        self._recent_matches.clear()


class ConservativeRecalibration(SeasonalRecalibration):
    """
    More conservative recalibration that only adjusts when clearly needed.

    Uses tighter thresholds and smaller adjustment factors to avoid
    over-correction that hurts accuracy on non-anomalous seasons.
    """

    def __init__(
        self,
        window_size: int = 50,
        baseline: Optional[LeagueBaseline] = None,
    ):
        super().__init__(
            window_size=window_size,
            baseline=baseline,
            sensitivity=0.5,  # Half as reactive
        )

        # Tighter thresholds - only trigger on significant deviations
        self._draw_rate_threshold = 0.05  # 5pp instead of 3pp
        self._goals_threshold = 0.30  # 0.3 instead of 0.2
        self._home_advantage_threshold = 0.08  # 8pp instead of 5pp

    def calculate_recalibration(self) -> RecalibrationFactors:
        """Calculate conservative recalibration factors."""
        stats = self.get_current_statistics()
        factors = RecalibrationFactors()

        factors.recent_draw_rate = stats['draw_rate']
        factors.recent_home_win_rate = stats['home_win_rate']
        factors.recent_goals_per_game = stats['goals_per_game']

        if stats['sample_size'] < 30:
            factors.deviation_from_baseline = "Insufficient data"
            return factors

        deviations = []

        # Draw rate adjustment - conservative
        draw_diff = stats['draw_rate'] - self.baseline.draw_rate
        if draw_diff > self._draw_rate_threshold:
            # Only apply when draws are elevated, not reduced
            draw_ratio = stats['draw_rate'] / self.baseline.draw_rate

            # Cap the boost at 1.3x to avoid over-correction
            factors.draw_boost = min(1.3, 1.0 + (draw_ratio - 1.0) * self.sensitivity)
            factors.rho_adjustment = min(1.2, 1.0 + (draw_ratio - 1.0) * self.sensitivity * 0.2)

            deviations.append(f"Elevated draws (+{draw_diff*100:.1f}pp)")

        factors.deviation_from_baseline = "; ".join(deviations) if deviations else "Within normal range"

        return factors


def create_recalibrated_predictor(
    base_predictor,
    recalibration: SeasonalRecalibration,
):
    """
    Wrap a predictor with dynamic recalibration.

    Parameters
    ----------
    base_predictor : object
        Base prediction model with predict_match method
    recalibration : SeasonalRecalibration
        Recalibration instance

    Returns
    -------
    callable
        Wrapped prediction function
    """
    factors = recalibration.calculate_recalibration()

    def recalibrated_predict(home_team: str, away_team: str, **kwargs):
        # Get base prediction
        pred = base_predictor.predict_match(home_team, away_team, **kwargs)

        # Apply recalibration
        home, draw, away = recalibration.adjust_probabilities(
            pred.home_win, pred.draw, pred.away_win, factors
        )

        # Update prediction with adjusted probabilities
        pred.home_win = home
        pred.draw = draw
        pred.away_win = away

        return pred

    return recalibrated_predict


def apply_draw_threshold_adjustment(
    home_prob: float,
    draw_prob: float,
    away_prob: float,
    draw_threshold: float = 0.26,
    parity_threshold: float = 0.08,
) -> str:
    """
    Get predicted outcome with adjusted draw threshold.

    When home and away probabilities are close (within parity_threshold)
    and draw probability is above draw_threshold, predict draw.

    This addresses the issue where models rarely predict draws because
    H/A probabilities are typically higher even for evenly-matched teams.

    Parameters
    ----------
    home_prob, draw_prob, away_prob : float
        Predicted probabilities
    draw_threshold : float
        Minimum draw probability to consider draw prediction
    parity_threshold : float
        Maximum difference between home and away to consider "even"

    Returns
    -------
    str
        Predicted outcome: 'H', 'D', or 'A'
    """
    # Check if teams are evenly matched
    is_parity = abs(home_prob - away_prob) < parity_threshold

    # Check if draw probability is substantial
    draw_is_significant = draw_prob >= draw_threshold

    # Predict draw when teams are even and draw is likely enough
    if is_parity and draw_is_significant:
        return 'D'

    # Also predict draw if it's actually the highest probability
    if draw_prob >= home_prob and draw_prob >= away_prob:
        return 'D'

    # Otherwise predict home or away
    if home_prob >= away_prob:
        return 'H'
    return 'A'
