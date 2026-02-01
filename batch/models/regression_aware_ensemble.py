"""
Regression-Aware Ensemble Predictor.

Dynamically adjusts ensemble weights based on whether teams are
over/underperforming their xG (regression candidates).

When regression is likely, boosts xG model weight since it excels
at identifying teams due for regression to the mean.
"""

from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional

import numpy as np


@dataclass
class TeamRegressionStats:
    """Tracks a team's over/underperformance vs xG."""
    # Recent match data (newest first)
    goals_scored: list[float] = field(default_factory=list)
    goals_conceded: list[float] = field(default_factory=list)
    xg_for: list[float] = field(default_factory=list)
    xg_against: list[float] = field(default_factory=list)

    # Calculated regression indicators
    attack_luck: float = 0.0  # goals_scored - xG (positive = lucky)
    defense_luck: float = 0.0  # goals_conceded - xG_against (negative = lucky)

    # Flags
    is_attack_regression_candidate: bool = False
    is_defense_regression_candidate: bool = False

    matches_played: int = 0


@dataclass
class RegressionAwarePrediction:
    """Prediction with regression-aware weighting."""
    home_prob: float
    draw_prob: float
    away_prob: float

    # Which weights were used
    weights_used: dict = field(default_factory=dict)
    regression_boost_applied: bool = False

    # Regression info
    home_attack_luck: float = 0.0
    home_defense_luck: float = 0.0
    away_attack_luck: float = 0.0
    away_defense_luck: float = 0.0


class RegressionAwareEnsemble:
    """
    Ensemble predictor that adapts weights based on regression likelihood.

    When either team is significantly over/underperforming their xG,
    boosts the xG model weight since it better captures true underlying
    performance.

    Parameters
    ----------
    window : int
        Number of matches for regression calculation (default: 10)
    regression_threshold : float
        Luck score threshold to trigger regression weighting (default: 0.25)
    standard_weights : dict
        Weights to use when no regression candidates
    regression_weights : dict
        Weights to use when regression is likely
    """

    def __init__(
        self,
        window: int = 10,
        regression_threshold: float = 0.25,
        standard_weights: Optional[dict] = None,
        regression_weights: Optional[dict] = None,
    ):
        self.window = window
        self.regression_threshold = regression_threshold

        # Default weights from optimization
        self.standard_weights = standard_weights or {
            'xg': 0.205,
            'pidc': 0.434,
            'elo': 0.197,
            'pi': 0.163,
        }

        # Boosted xG weights for regression scenarios
        self.regression_weights = regression_weights or {
            'xg': 0.40,  # Boosted from 20.5% to 40%
            'pidc': 0.30,
            'elo': 0.15,
            'pi': 0.15,
        }

        # Team regression tracking
        self._team_stats: dict[str, TeamRegressionStats] = defaultdict(TeamRegressionStats)

    def update_match(
        self,
        home_team: str,
        away_team: str,
        home_goals: int,
        away_goals: int,
        home_xg: float,
        away_xg: float,
    ):
        """Update team regression stats after a match."""
        # Update home team
        home_stats = self._team_stats[home_team]
        home_stats.goals_scored.insert(0, home_goals)
        home_stats.goals_conceded.insert(0, away_goals)
        home_stats.xg_for.insert(0, home_xg)
        home_stats.xg_against.insert(0, away_xg)
        home_stats.matches_played += 1

        # Trim to window
        for lst in [home_stats.goals_scored, home_stats.goals_conceded,
                    home_stats.xg_for, home_stats.xg_against]:
            if len(lst) > self.window:
                lst[:] = lst[:self.window]

        # Update away team
        away_stats = self._team_stats[away_team]
        away_stats.goals_scored.insert(0, away_goals)
        away_stats.goals_conceded.insert(0, home_goals)
        away_stats.xg_for.insert(0, away_xg)
        away_stats.xg_against.insert(0, home_xg)
        away_stats.matches_played += 1

        for lst in [away_stats.goals_scored, away_stats.goals_conceded,
                    away_stats.xg_for, away_stats.xg_against]:
            if len(lst) > self.window:
                lst[:] = lst[:self.window]

        # Recalculate regression indicators
        self._update_regression_indicators(home_team)
        self._update_regression_indicators(away_team)

    def _update_regression_indicators(self, team: str):
        """Calculate luck scores and regression flags."""
        stats = self._team_stats[team]

        if len(stats.goals_scored) < 3:
            return

        # Attack luck: scoring more than xG suggests (positive = lucky)
        total_goals = sum(stats.goals_scored)
        total_xg = sum(stats.xg_for)
        stats.attack_luck = (total_goals - total_xg) / len(stats.goals_scored)

        # Defense luck: conceding less than xG against (negative = lucky/good)
        total_conceded = sum(stats.goals_conceded)
        total_xg_against = sum(stats.xg_against)
        stats.defense_luck = (total_conceded - total_xg_against) / len(stats.goals_conceded)

        # Flag regression candidates
        stats.is_attack_regression_candidate = abs(stats.attack_luck) > self.regression_threshold
        stats.is_defense_regression_candidate = abs(stats.defense_luck) > self.regression_threshold

    def get_team_stats(self, team: str) -> TeamRegressionStats:
        """Get regression stats for a team."""
        return self._team_stats.get(team, TeamRegressionStats())

    def is_regression_match(self, home_team: str, away_team: str) -> bool:
        """Check if either team is a regression candidate."""
        home_stats = self._team_stats.get(home_team, TeamRegressionStats())
        away_stats = self._team_stats.get(away_team, TeamRegressionStats())

        return (
            home_stats.is_attack_regression_candidate or
            home_stats.is_defense_regression_candidate or
            away_stats.is_attack_regression_candidate or
            away_stats.is_defense_regression_candidate
        )

    def predict(
        self,
        home_team: str,
        away_team: str,
        model_predictions: dict[str, tuple[float, float, float]],
    ) -> RegressionAwarePrediction:
        """
        Generate ensemble prediction with adaptive weights.

        Parameters
        ----------
        home_team : str
        away_team : str
        model_predictions : dict
            Keys: 'xg', 'pidc', 'elo', 'pi'
            Values: (home_prob, draw_prob, away_prob)

        Returns
        -------
        RegressionAwarePrediction
        """
        home_stats = self._team_stats.get(home_team, TeamRegressionStats())
        away_stats = self._team_stats.get(away_team, TeamRegressionStats())

        # Determine which weights to use
        use_regression = self.is_regression_match(home_team, away_team)
        weights = self.regression_weights if use_regression else self.standard_weights

        # Calculate weighted ensemble
        home_prob = 0.0
        draw_prob = 0.0
        away_prob = 0.0

        for model, weight in weights.items():
            if model in model_predictions:
                h, d, a = model_predictions[model]
                home_prob += weight * h
                draw_prob += weight * d
                away_prob += weight * a

        # Normalize
        total = home_prob + draw_prob + away_prob
        if total > 0:
            home_prob /= total
            draw_prob /= total
            away_prob /= total

        return RegressionAwarePrediction(
            home_prob=home_prob,
            draw_prob=draw_prob,
            away_prob=away_prob,
            weights_used=weights,
            regression_boost_applied=use_regression,
            home_attack_luck=home_stats.attack_luck,
            home_defense_luck=home_stats.defense_luck,
            away_attack_luck=away_stats.attack_luck,
            away_defense_luck=away_stats.defense_luck,
        )

    def get_regression_summary(self) -> dict:
        """Get summary of teams with regression signals."""
        summary = {
            'lucky_scorers': [],  # Scoring above xG
            'unlucky_scorers': [],  # Scoring below xG
            'lucky_defenders': [],  # Conceding below xG
            'unlucky_defenders': [],  # Conceding above xG
        }

        for team, stats in self._team_stats.items():
            if stats.matches_played < 5:
                continue

            if stats.attack_luck > self.regression_threshold:
                summary['lucky_scorers'].append((team, stats.attack_luck))
            elif stats.attack_luck < -self.regression_threshold:
                summary['unlucky_scorers'].append((team, stats.attack_luck))

            if stats.defense_luck < -self.regression_threshold:
                summary['lucky_defenders'].append((team, stats.defense_luck))
            elif stats.defense_luck > self.regression_threshold:
                summary['unlucky_defenders'].append((team, stats.defense_luck))

        # Sort by magnitude
        for key in summary:
            summary[key].sort(key=lambda x: abs(x[1]), reverse=True)

        return summary
