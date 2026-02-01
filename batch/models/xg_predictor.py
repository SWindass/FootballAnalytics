"""
xG-Based Match Predictor using Understat Expected Goals data.

Uses professional xG data to calculate expected goals and generate
match outcome probabilities via Dixon-Coles Poisson model.

Features:
- Rolling average xG for/against (last N matches)
- xG overperformance tracking (regression candidates)
- Shot quality metrics
- Dixon-Coles low-score correlation correction
"""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
from scipy.stats import poisson


@dataclass
class TeamXGStats:
    """Rolling xG statistics for a team."""
    # Recent match xG data (newest first)
    xg_for: list[float] = field(default_factory=list)
    xg_against: list[float] = field(default_factory=list)
    goals_for: list[float] = field(default_factory=list)
    goals_against: list[float] = field(default_factory=list)

    # Calculated metrics
    avg_xg_for: float = 0.0
    avg_xg_against: float = 0.0
    avg_goals_for: float = 0.0
    avg_goals_against: float = 0.0

    # xG overperformance (actual - expected)
    xg_overperformance: float = 0.0  # Positive = scoring more than expected
    xg_underperformance_against: float = 0.0  # Positive = conceding less than expected

    # Match count
    matches_played: int = 0


@dataclass
class XGMatchPrediction:
    """xG-based match prediction."""
    home_xg: float
    away_xg: float
    home_prob: float
    draw_prob: float
    away_prob: float

    # Score matrix for exact score probabilities
    score_matrix: Optional[np.ndarray] = None

    # Confidence metrics
    confidence: float = 0.0
    home_regression_risk: float = 0.0  # How much home team is overperforming
    away_regression_risk: float = 0.0


class XGPredictor:
    """
    Match predictor using Understat xG data.

    Uses rolling xG averages to estimate expected goals, then applies
    Dixon-Coles Poisson model for probability calculation.

    Parameters
    ----------
    window : int
        Number of recent matches for rolling averages (default: 10)
    rho : float
        Dixon-Coles correlation parameter (default: -0.11)
    home_advantage : float
        Multiplier for home team xG (default: 1.10)
    regression_weight : float
        How much to regress overperformers toward xG (default: 0.3)
    """

    def __init__(
        self,
        window: int = 10,
        rho: float = -0.11,
        home_advantage: float = 1.10,
        regression_weight: float = 0.3,
    ):
        self.window = window
        self.rho = rho
        self.home_advantage = home_advantage
        self.regression_weight = regression_weight

        # Team statistics: team_name -> TeamXGStats
        self._team_stats: dict[str, TeamXGStats] = defaultdict(TeamXGStats)

        # League averages
        self._league_xg_avg: float = 1.35  # Updated dynamically
        self._all_match_xg: list[float] = []

    def update_match(
        self,
        home_team: str,
        away_team: str,
        home_goals: int,
        away_goals: int,
        home_xg: float,
        away_xg: float,
    ):
        """
        Update team statistics after a match.

        Call this for each historical match in chronological order.
        """
        # Update home team stats
        home_stats = self._team_stats[home_team]
        home_stats.xg_for.insert(0, home_xg)
        home_stats.xg_against.insert(0, away_xg)
        home_stats.goals_for.insert(0, home_goals)
        home_stats.goals_against.insert(0, away_goals)
        home_stats.matches_played += 1

        # Trim to window size
        if len(home_stats.xg_for) > self.window:
            home_stats.xg_for = home_stats.xg_for[:self.window]
            home_stats.xg_against = home_stats.xg_against[:self.window]
            home_stats.goals_for = home_stats.goals_for[:self.window]
            home_stats.goals_against = home_stats.goals_against[:self.window]

        # Update away team stats
        away_stats = self._team_stats[away_team]
        away_stats.xg_for.insert(0, away_xg)
        away_stats.xg_against.insert(0, home_xg)
        away_stats.goals_for.insert(0, away_goals)
        away_stats.goals_against.insert(0, home_goals)
        away_stats.matches_played += 1

        if len(away_stats.xg_for) > self.window:
            away_stats.xg_for = away_stats.xg_for[:self.window]
            away_stats.xg_against = away_stats.xg_against[:self.window]
            away_stats.goals_for = away_stats.goals_for[:self.window]
            away_stats.goals_against = away_stats.goals_against[:self.window]

        # Update calculated metrics
        self._update_team_metrics(home_team)
        self._update_team_metrics(away_team)

        # Update league average
        self._all_match_xg.append(home_xg)
        self._all_match_xg.append(away_xg)
        if len(self._all_match_xg) > 1000:
            self._all_match_xg = self._all_match_xg[-1000:]
        self._league_xg_avg = np.mean(self._all_match_xg)

    def _update_team_metrics(self, team: str):
        """Recalculate team's derived metrics."""
        stats = self._team_stats[team]

        if not stats.xg_for:
            return

        stats.avg_xg_for = np.mean(stats.xg_for)
        stats.avg_xg_against = np.mean(stats.xg_against)
        stats.avg_goals_for = np.mean(stats.goals_for)
        stats.avg_goals_against = np.mean(stats.goals_against)

        # xG overperformance: scoring more than expected
        stats.xg_overperformance = stats.avg_goals_for - stats.avg_xg_for

        # Defensive overperformance: conceding less than expected (positive = good)
        stats.xg_underperformance_against = stats.avg_xg_against - stats.avg_goals_against

    def get_team_stats(self, team: str) -> TeamXGStats:
        """Get current xG statistics for a team."""
        return self._team_stats.get(team, TeamXGStats())

    def predict_match(
        self,
        home_team: str,
        away_team: str,
        apply_regression: bool = True,
    ) -> XGMatchPrediction:
        """
        Predict match outcome using xG-based expected goals.

        Formula:
        home_xG = (home_xG_for / league_avg) × (away_xG_against / league_avg) × league_avg × home_adv
        away_xG = (away_xG_for / league_avg) × (home_xG_against / league_avg) × league_avg × away_disadv

        Parameters
        ----------
        home_team : str
            Home team name
        away_team : str
            Away team name
        apply_regression : bool
            Whether to regress overperformers toward their xG

        Returns
        -------
        XGMatchPrediction
            Prediction with probabilities and xG values
        """
        home_stats = self._team_stats.get(home_team, TeamXGStats())
        away_stats = self._team_stats.get(away_team, TeamXGStats())

        # Need minimum matches for reliable prediction
        min_matches = 3
        if home_stats.matches_played < min_matches or away_stats.matches_played < min_matches:
            # Return default prediction
            return XGMatchPrediction(
                home_xg=self._league_xg_avg * self.home_advantage,
                away_xg=self._league_xg_avg * (2 - self.home_advantage),
                home_prob=0.45,
                draw_prob=0.25,
                away_prob=0.30,
                confidence=0.0,
            )

        league_avg = self._league_xg_avg

        # Calculate attack/defense ratios
        home_attack_ratio = home_stats.avg_xg_for / league_avg if league_avg > 0 else 1.0
        home_defense_ratio = home_stats.avg_xg_against / league_avg if league_avg > 0 else 1.0
        away_attack_ratio = away_stats.avg_xg_for / league_avg if league_avg > 0 else 1.0
        away_defense_ratio = away_stats.avg_xg_against / league_avg if league_avg > 0 else 1.0

        # Calculate expected goals
        # Home xG = home attack strength × away defensive weakness × league avg × home advantage
        home_xg = (
            home_attack_ratio *
            away_defense_ratio *
            league_avg *
            self.home_advantage
        )

        # Away xG = away attack strength × home defensive weakness × league avg × away disadvantage
        away_xg = (
            away_attack_ratio *
            home_defense_ratio *
            league_avg *
            (2 - self.home_advantage)  # e.g., 0.90 if home_advantage is 1.10
        )

        # Apply regression for overperformers
        home_regression_risk = 0.0
        away_regression_risk = 0.0

        if apply_regression:
            # If home team is overperforming their xG, regress expected output
            if home_stats.xg_overperformance > 0.2:
                regression = home_stats.xg_overperformance * self.regression_weight
                home_xg -= regression * 0.5
                home_regression_risk = home_stats.xg_overperformance

            # If away team is overperforming, regress
            if away_stats.xg_overperformance > 0.2:
                regression = away_stats.xg_overperformance * self.regression_weight
                away_xg -= regression * 0.5
                away_regression_risk = away_stats.xg_overperformance

        # Clamp to reasonable range
        home_xg = np.clip(home_xg, 0.5, 3.5)
        away_xg = np.clip(away_xg, 0.3, 3.0)

        # Generate probability matrix with Dixon-Coles
        matrix = self._calculate_score_matrix(home_xg, away_xg)

        # Extract outcome probabilities
        home_prob, draw_prob, away_prob = self._matrix_to_probs(matrix)

        # Calculate confidence based on data quality
        data_quality = min(home_stats.matches_played, away_stats.matches_played, self.window) / self.window
        confidence = data_quality

        return XGMatchPrediction(
            home_xg=home_xg,
            away_xg=away_xg,
            home_prob=home_prob,
            draw_prob=draw_prob,
            away_prob=away_prob,
            score_matrix=matrix,
            confidence=confidence,
            home_regression_risk=home_regression_risk,
            away_regression_risk=away_regression_risk,
        )

    def _calculate_score_matrix(
        self,
        home_xg: float,
        away_xg: float,
        max_goals: int = 7,
    ) -> np.ndarray:
        """Calculate score probability matrix with Dixon-Coles correction."""
        matrix = np.zeros((max_goals + 1, max_goals + 1))

        for h in range(max_goals + 1):
            for a in range(max_goals + 1):
                base_prob = poisson.pmf(h, home_xg) * poisson.pmf(a, away_xg)
                correction = self._dixon_coles_correction(h, a, home_xg, away_xg)
                matrix[h, a] = base_prob * correction

        # Normalize
        total = matrix.sum()
        if total > 0:
            matrix /= total

        return matrix

    def _dixon_coles_correction(
        self,
        home_goals: int,
        away_goals: int,
        home_xg: float,
        away_xg: float,
    ) -> float:
        """Apply Dixon-Coles low-score correlation correction."""
        if home_goals == 0 and away_goals == 0:
            return 1 - home_xg * away_xg * self.rho
        elif home_goals == 0 and away_goals == 1:
            return 1 + home_xg * self.rho
        elif home_goals == 1 and away_goals == 0:
            return 1 + away_xg * self.rho
        elif home_goals == 1 and away_goals == 1:
            return 1 - self.rho
        return 1.0

    def _matrix_to_probs(self, matrix: np.ndarray) -> tuple[float, float, float]:
        """Convert score matrix to outcome probabilities."""
        home_win = draw = away_win = 0.0

        for h in range(matrix.shape[0]):
            for a in range(matrix.shape[1]):
                if h > a:
                    home_win += matrix[h, a]
                elif h == a:
                    draw += matrix[h, a]
                else:
                    away_win += matrix[h, a]

        return home_win, draw, away_win

    def get_regression_candidates(self, threshold: float = 0.3) -> dict[str, dict]:
        """
        Identify teams that are likely to regress.

        Returns teams overperforming or underperforming their xG by more than threshold.
        """
        candidates = {}

        for team, stats in self._team_stats.items():
            if stats.matches_played < 5:
                continue

            if abs(stats.xg_overperformance) > threshold:
                candidates[team] = {
                    'type': 'attacking',
                    'direction': 'down' if stats.xg_overperformance > 0 else 'up',
                    'magnitude': stats.xg_overperformance,
                    'avg_goals': stats.avg_goals_for,
                    'avg_xg': stats.avg_xg_for,
                }

            if abs(stats.xg_underperformance_against) > threshold:
                if team not in candidates:
                    candidates[team] = {}
                candidates[team].update({
                    'defense_type': 'defensive',
                    'defense_direction': 'worse' if stats.xg_underperformance_against > 0 else 'better',
                    'defense_magnitude': stats.xg_underperformance_against,
                    'avg_goals_against': stats.avg_goals_against,
                    'avg_xg_against': stats.avg_xg_against,
                })

        return candidates

    def season_reset(self, regression_factor: float = 0.5):
        """
        Apply season regression - partially reset team stats.

        Call this at the start of a new season.
        """
        for team, stats in self._team_stats.items():
            # Keep some history but regress toward league average
            if stats.xg_for:
                # Regress xG toward league average
                regressed_xg_for = (
                    regression_factor * self._league_xg_avg +
                    (1 - regression_factor) * stats.avg_xg_for
                )
                regressed_xg_against = (
                    regression_factor * self._league_xg_avg +
                    (1 - regression_factor) * stats.avg_xg_against
                )

                # Keep last few matches but add regressed values
                stats.xg_for = [regressed_xg_for] * 3 + stats.xg_for[:2]
                stats.xg_against = [regressed_xg_against] * 3 + stats.xg_against[:2]
                stats.goals_for = [regressed_xg_for] * 3 + stats.goals_for[:2]
                stats.goals_against = [regressed_xg_against] * 3 + stats.goals_against[:2]

                self._update_team_metrics(team)
