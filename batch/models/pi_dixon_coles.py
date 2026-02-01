"""Pi Rating + Dixon-Coles model with enhanced draw prediction.

Combines Pi Ratings with Dixon-Coles Poisson model and a trained
draw probability multiplier for improved outcome predictions.

Architecture:
1. Pi Ratings → Expected goal difference
2. Convert to expected goals per team
3. Dixon-Coles Poisson → Probability matrix with low-score correlation
4. Draw-specific features → Trained multiplier
5. Final calibrated probabilities
"""

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression

from batch.models.pi_rating import PiRating


@dataclass
class MatchFeatures:
    """Features for a single match prediction."""

    # Pi Rating features
    home_rating: float
    away_rating: float
    rating_diff: float
    expected_gd: float

    # Expected goals
    home_xg: float
    away_xg: float
    total_xg: float

    # Draw-specific features
    strength_parity: float  # 1 / (1 + |rating_diff|)
    low_scoring: int  # 1 if total_xg < 2.3
    both_defensive: int  # 1 if both teams defensive
    mid_table_clash: int  # 1 if both teams mid-table

    # Team context (optional)
    home_league_pos: int | None = None
    away_league_pos: int | None = None
    home_xga_vs_avg: float | None = None
    away_xga_vs_avg: float | None = None


@dataclass
class MatchProbabilities:
    """Outcome probabilities for a match."""

    home_win: float
    draw: float
    away_win: float

    # Score matrix (optional)
    score_matrix: np.ndarray | None = None

    # Component probabilities for analysis
    poisson_home: float = 0.0
    poisson_draw: float = 0.0
    poisson_away: float = 0.0
    dixon_coles_draw: float = 0.0
    draw_multiplier: float = 1.0

    # Draw confidence score (for explicit draw selection)
    draw_confidence: float = 0.0

    def to_dict(self) -> dict:
        return {
            "home_win": self.home_win,
            "draw": self.draw,
            "away_win": self.away_win,
        }

    def get_prediction(self, draw_threshold: float = 0.28) -> str:
        """Get predicted outcome with configurable draw threshold.

        Parameters
        ----------
        draw_threshold : float
            If draw probability exceeds this AND draw_confidence is high,
            predict draw even if not highest probability.

        Returns
        -------
        str
            "H", "D", or "A"
        """
        # If draw probability is high and conditions favor draw, predict draw
        if self.draw >= draw_threshold and self.draw_confidence > 0.6:
            return "D"

        # Also predict draw if it's close to the maximum
        max_prob = max(self.home_win, self.draw, self.away_win)
        if self.draw > max_prob * 0.92 and self.draw >= 0.26:
            return "D"

        # Otherwise pick highest probability
        if self.home_win >= self.away_win:
            return "H"
        return "A"


class PiDixonColesModel:
    """Combined Pi Rating + Dixon-Coles model with draw enhancement.

    This model:
    1. Uses Pi Ratings for team strength estimation
    2. Converts rating difference to expected goals
    3. Applies Dixon-Coles Poisson for score probabilities
    4. Enhances draw predictions with trained multiplier

    Parameters
    ----------
    pi_lambda : float
        Pi Rating learning rate parameter.
    pi_gamma : float
        Pi Rating away venue multiplier.
    rho : float
        Dixon-Coles correlation parameter (negative boosts low scores).
    home_advantage_goals : float
        Base home advantage in expected goals.
    avg_goals : float
        League average goals per team per match.
    """

    def __init__(
        self,
        pi_lambda: float = 0.06,
        pi_gamma: float = 0.6,
        rho: float = -0.13,
        home_advantage_goals: float = 0.25,
        avg_goals: float = 1.35,
    ):
        self.pi_lambda = pi_lambda
        self.pi_gamma = pi_gamma
        self.rho = rho
        self.home_advantage_goals = home_advantage_goals
        self.avg_goals = avg_goals

        # Initialize Pi Rating system
        self.pi_rating = PiRating(
            lambda_param=pi_lambda,
            gamma_param=pi_gamma,
        )

        # Draw multiplier model (trained separately)
        self.draw_model: LogisticRegression | None = None
        self.draw_model_trained = False

        # League context for features
        self.team_positions: dict[str, int] = {}
        self.team_xga: dict[str, float] = {}
        self.league_avg_xga: float = 1.35

        # Training data
        self._training_features: list[np.ndarray] = []
        self._training_labels: list[int] = []
        self._training_weights: list[float] = []

    def rating_to_expected_goals(
        self,
        home_team: str,
        away_team: str,
    ) -> tuple[float, float]:
        """Convert Pi Ratings to expected goals for each team.

        Uses the rating difference to adjust from league average.
        A team with higher rating scores more, concedes less.

        Parameters
        ----------
        home_team : str
            Home team name.
        away_team : str
            Away team name.

        Returns
        -------
        tuple[float, float]
            (home_xg, away_xg)
        """
        # Get Pi Rating expected goal difference
        expected_gd = self.pi_rating.calculate_expected_goal_diff(home_team, away_team)

        # Convert to individual team xG
        # GD = home_xg - away_xg
        # Total goals typically around 2.7 per match
        total_xg = 2 * self.avg_goals

        # Distribute based on expected GD
        # If GD = 0.5, home scores more of the total
        home_xg = (total_xg + expected_gd) / 2 + self.home_advantage_goals / 2
        away_xg = (total_xg - expected_gd) / 2 - self.home_advantage_goals / 2

        # Ensure non-negative
        home_xg = max(0.3, home_xg)
        away_xg = max(0.3, away_xg)

        return home_xg, away_xg

    def poisson_probability(
        self,
        home_goals: int,
        away_goals: int,
        home_xg: float,
        away_xg: float,
    ) -> float:
        """Calculate Poisson probability for a specific scoreline.

        Parameters
        ----------
        home_goals, away_goals : int
            Scoreline to calculate probability for.
        home_xg, away_xg : float
            Expected goals for each team.

        Returns
        -------
        float
            Probability of this scoreline.
        """
        return poisson.pmf(home_goals, home_xg) * poisson.pmf(away_goals, away_xg)

    def dixon_coles_correction(
        self,
        home_goals: int,
        away_goals: int,
        home_xg: float,
        away_xg: float,
        rho: float,
    ) -> float:
        """Apply Dixon-Coles correlation correction.

        Adjusts probabilities for 0-0, 1-0, 0-1, 1-1 scorelines
        to account for the observed correlation between home and
        away goals (teams adjust tactics based on scoreline).

        Parameters
        ----------
        home_goals, away_goals : int
            Scoreline.
        home_xg, away_xg : float
            Expected goals.
        rho : float
            Correlation parameter (typically negative, ~-0.13).

        Returns
        -------
        float
            Correction multiplier for this scoreline.
        """
        if home_goals == 0 and away_goals == 0:
            return 1 - home_xg * away_xg * rho
        elif home_goals == 0 and away_goals == 1:
            return 1 + home_xg * rho
        elif home_goals == 1 and away_goals == 0:
            return 1 + away_xg * rho
        elif home_goals == 1 and away_goals == 1:
            return 1 - rho
        else:
            return 1.0

    def calculate_score_matrix(
        self,
        home_xg: float,
        away_xg: float,
        max_goals: int = 7,
    ) -> np.ndarray:
        """Calculate full score probability matrix with Dixon-Coles.

        Parameters
        ----------
        home_xg, away_xg : float
            Expected goals for each team.
        max_goals : int
            Maximum goals to consider per team.

        Returns
        -------
        np.ndarray
            Matrix of shape (max_goals+1, max_goals+1) with probabilities.
            matrix[i, j] = P(home scores i, away scores j)
        """
        matrix = np.zeros((max_goals + 1, max_goals + 1))

        for h in range(max_goals + 1):
            for a in range(max_goals + 1):
                base_prob = self.poisson_probability(h, a, home_xg, away_xg)
                correction = self.dixon_coles_correction(h, a, home_xg, away_xg, self.rho)
                matrix[h, a] = base_prob * correction

        # Normalize to sum to 1
        matrix /= matrix.sum()

        return matrix

    def matrix_to_probabilities(
        self,
        matrix: np.ndarray,
    ) -> tuple[float, float, float]:
        """Convert score matrix to outcome probabilities.

        Parameters
        ----------
        matrix : np.ndarray
            Score probability matrix.

        Returns
        -------
        tuple[float, float, float]
            (home_win_prob, draw_prob, away_win_prob)
        """
        home_win = 0.0
        draw = 0.0
        away_win = 0.0

        for h in range(matrix.shape[0]):
            for a in range(matrix.shape[1]):
                if h > a:
                    home_win += matrix[h, a]
                elif h == a:
                    draw += matrix[h, a]
                else:
                    away_win += matrix[h, a]

        return home_win, draw, away_win

    def calculate_draw_features(
        self,
        home_team: str,
        away_team: str,
        home_xg: float,
        away_xg: float,
    ) -> MatchFeatures:
        """Calculate draw-specific features for a match.

        Parameters
        ----------
        home_team, away_team : str
            Team names.
        home_xg, away_xg : float
            Expected goals.

        Returns
        -------
        MatchFeatures
            Features for draw prediction.
        """
        # Get ratings
        home_r = self.pi_rating.get_team_rating(home_team)
        away_r = self.pi_rating.get_team_rating(away_team)

        home_overall = home_r.overall if home_r else 0.0
        away_overall = away_r.overall if away_r else 0.0
        rating_diff = home_overall - away_overall

        # Strength parity: closer teams more likely to draw
        strength_parity = 1 / (1 + abs(rating_diff))

        # Low scoring indicator
        total_xg = home_xg + away_xg
        low_scoring = 1 if total_xg < 2.3 else 0

        # Defensive teams indicator
        home_xga = self.team_xga.get(home_team, self.league_avg_xga)
        away_xga = self.team_xga.get(away_team, self.league_avg_xga)
        both_defensive = 1 if (home_xga < self.league_avg_xga and
                               away_xga < self.league_avg_xga) else 0

        # Mid-table clash
        home_pos = self.team_positions.get(home_team, 10)
        away_pos = self.team_positions.get(away_team, 10)
        mid_table_clash = 1 if (7 <= home_pos <= 14 and 7 <= away_pos <= 14) else 0

        return MatchFeatures(
            home_rating=home_overall,
            away_rating=away_overall,
            rating_diff=rating_diff,
            expected_gd=home_xg - away_xg,
            home_xg=home_xg,
            away_xg=away_xg,
            total_xg=total_xg,
            strength_parity=strength_parity,
            low_scoring=low_scoring,
            both_defensive=both_defensive,
            mid_table_clash=mid_table_clash,
            home_league_pos=home_pos,
            away_league_pos=away_pos,
            home_xga_vs_avg=home_xga - self.league_avg_xga,
            away_xga_vs_avg=away_xga - self.league_avg_xga,
        )

    def features_to_array(self, features: MatchFeatures) -> np.ndarray:
        """Convert MatchFeatures to numpy array for model input."""
        return np.array([
            features.strength_parity,
            features.low_scoring,
            features.both_defensive,
            features.mid_table_clash,
            abs(features.rating_diff),
            features.total_xg,
        ])

    def predict_match(
        self,
        home_team: str,
        away_team: str,
        apply_draw_model: bool = True,
        draw_boost_threshold: float = 0.65,
    ) -> MatchProbabilities:
        """Predict match outcome probabilities.

        Parameters
        ----------
        home_team, away_team : str
            Team names.
        apply_draw_model : bool
            Whether to apply trained draw multiplier.
        draw_boost_threshold : float
            Minimum draw model confidence to boost draw prediction.

        Returns
        -------
        MatchProbabilities
            Predicted probabilities.
        """
        # Step 1: Get expected goals from Pi Rating
        home_xg, away_xg = self.rating_to_expected_goals(home_team, away_team)

        # Step 2: Calculate score matrix with Dixon-Coles
        matrix = self.calculate_score_matrix(home_xg, away_xg)

        # Step 3: Get base probabilities
        home_win, draw, away_win = self.matrix_to_probabilities(matrix)

        # Store Dixon-Coles draw for comparison
        dc_draw = draw

        # Step 4: Apply draw enhancement if conditions are favorable
        draw_mult = 1.0
        features = self.calculate_draw_features(home_team, away_team, home_xg, away_xg)

        # Rule-based draw boost based on key indicators
        # These conditions are known to correlate with draws
        draw_boost = 0.0

        # Strength parity boost (teams closely matched)
        if features.strength_parity > 0.7:  # Rating diff < 0.43
            draw_boost += 0.08

        # Low scoring matches favor draws
        if features.total_xg < 2.4:
            draw_boost += 0.05

        # Mid-table clashes
        if features.mid_table_clash:
            draw_boost += 0.03

        # Close expected GD
        if abs(features.expected_gd) < 0.3:
            draw_boost += 0.06

        # Apply trained model boost if available
        if apply_draw_model and self.draw_model_trained:
            feat_array = self.features_to_array(features).reshape(1, -1)
            draw_boost_prob = self.draw_model.predict_proba(feat_array)[0, 1]

            # Strong model confidence adds additional boost
            if draw_boost_prob > draw_boost_threshold:
                draw_boost += (draw_boost_prob - 0.5) * 0.15

        # Apply boost (multiplicative)
        draw_mult = 1.0 + draw_boost
        draw *= draw_mult

        # Step 5: Renormalize
        total = home_win + draw + away_win
        home_win /= total
        draw /= total
        away_win /= total

        # Calculate draw confidence score based on features
        draw_confidence = 0.0
        if features.strength_parity > 0.7:
            draw_confidence += 0.3
        if features.total_xg < 2.4:
            draw_confidence += 0.2
        if features.mid_table_clash:
            draw_confidence += 0.15
        if abs(features.expected_gd) < 0.25:
            draw_confidence += 0.25
        if features.low_scoring:
            draw_confidence += 0.1

        # Trained model contribution
        if apply_draw_model and self.draw_model_trained:
            feat_array = self.features_to_array(features).reshape(1, -1)
            draw_boost_prob = self.draw_model.predict_proba(feat_array)[0, 1]
            draw_confidence = 0.4 * draw_confidence + 0.6 * draw_boost_prob

        return MatchProbabilities(
            home_win=home_win,
            draw=draw,
            away_win=away_win,
            score_matrix=matrix,
            poisson_home=home_win,
            poisson_draw=dc_draw,
            poisson_away=away_win,
            dixon_coles_draw=dc_draw,
            draw_multiplier=draw_mult,
            draw_confidence=min(1.0, draw_confidence),
        )

    def update_after_match(
        self,
        home_team: str,
        away_team: str,
        home_goals: int,
        away_goals: int,
        match_date: datetime | None = None,
        collect_training_data: bool = True,
    ) -> None:
        """Update model after a match result.

        Updates Pi Ratings and optionally collects training data
        for the draw multiplier model.

        Parameters
        ----------
        home_team, away_team : str
            Team names.
        home_goals, away_goals : int
            Match result.
        match_date : datetime, optional
            Match date.
        collect_training_data : bool
            Whether to collect data for draw model training.
        """
        # Collect training data before updating ratings
        if collect_training_data:
            home_xg, away_xg = self.rating_to_expected_goals(home_team, away_team)
            features = self.calculate_draw_features(home_team, away_team, home_xg, away_xg)
            feat_array = self.features_to_array(features)

            is_draw = 1 if home_goals == away_goals else 0
            self._training_features.append(feat_array)
            self._training_labels.append(is_draw)

            # Weight draws more heavily (class imbalance)
            weight = 2.5 if is_draw else 1.0
            self._training_weights.append(weight)

        # Update Pi Ratings
        self.pi_rating.update_ratings(
            home_team, away_team, home_goals, away_goals, match_date, store_history=False
        )

    def train_draw_model(
        self,
        min_samples: int = 200,
    ) -> dict:
        """Train the draw probability multiplier model.

        Uses collected training data to fit a logistic regression
        model that predicts when draws are more/less likely.

        Parameters
        ----------
        min_samples : int
            Minimum samples required before training.

        Returns
        -------
        dict
            Training metrics.
        """
        if len(self._training_features) < min_samples:
            return {"status": "insufficient_data", "samples": len(self._training_features)}

        X = np.array(self._training_features)
        y = np.array(self._training_labels)
        weights = np.array(self._training_weights)

        # Train logistic regression
        self.draw_model = LogisticRegression(
            class_weight=None,  # Using manual weights instead
            max_iter=1000,
            random_state=42,
        )
        self.draw_model.fit(X, y, sample_weight=weights)
        self.draw_model_trained = True

        # Calculate metrics
        y_pred = self.draw_model.predict(X)
        self.draw_model.predict_proba(X)[:, 1]

        accuracy = (y_pred == y).mean()
        draw_recall = y_pred[y == 1].mean() if y.sum() > 0 else 0

        return {
            "status": "trained",
            "samples": len(y),
            "draws": int(y.sum()),
            "draw_rate": y.mean(),
            "accuracy": accuracy,
            "draw_recall": draw_recall,
            "feature_importance": dict(zip(
                ["strength_parity", "low_scoring", "both_defensive",
                 "mid_table_clash", "rating_diff_abs", "total_xg"],
                self.draw_model.coef_[0].tolist(), strict=False
            )),
        }

    def set_league_context(
        self,
        positions: dict[str, int],
        xga: dict[str, float],
        league_avg_xga: float = 1.35,
    ) -> None:
        """Set current league context for feature calculation.

        Parameters
        ----------
        positions : dict[str, int]
            Team name -> league position (1-20).
        xga : dict[str, float]
            Team name -> expected goals against per game.
        league_avg_xga : float
            League average xGA.
        """
        self.team_positions = positions
        self.team_xga = xga
        self.league_avg_xga = league_avg_xga

    def optimize_rho(
        self,
        matches_df: pd.DataFrame,
        rho_range: tuple[float, float] = (-0.25, 0.0),
    ) -> float:
        """Optimize Dixon-Coles rho parameter on historical data.

        Parameters
        ----------
        matches_df : pd.DataFrame
            Historical matches with home_goals, away_goals columns.
        rho_range : tuple[float, float]
            Search range for rho.

        Returns
        -------
        float
            Optimal rho value.
        """
        def neg_log_likelihood(rho_arr):
            rho = rho_arr[0]
            total_ll = 0

            for _, row in matches_df.iterrows():
                home_xg, away_xg = self.rating_to_expected_goals(
                    row["home_team"], row["away_team"]
                )

                h, a = int(row["home_goals"]), int(row["away_goals"])

                # Clamp to reasonable range
                h = min(h, 7)
                a = min(a, 7)

                base_prob = self.poisson_probability(h, a, home_xg, away_xg)
                correction = self.dixon_coles_correction(h, a, home_xg, away_xg, rho)
                prob = base_prob * correction

                if prob > 0:
                    total_ll += np.log(prob)

            return -total_ll

        result = minimize(
            neg_log_likelihood,
            x0=[self.rho],
            bounds=[rho_range],
            method="L-BFGS-B",
        )

        return result.x[0]

    def reset(self) -> None:
        """Reset model state."""
        self.pi_rating.reset()
        self._training_features.clear()
        self._training_labels.clear()
        self._training_weights.clear()
        self.draw_model = None
        self.draw_model_trained = False


def create_calibration_plot(
    predictions: list[MatchProbabilities],
    actuals: list[str],
    n_bins: int = 10,
):
    """Create calibration plot data for draw predictions.

    Parameters
    ----------
    predictions : list[MatchProbabilities]
        Model predictions.
    actuals : list[str]
        Actual outcomes ("H", "D", "A").
    n_bins : int
        Number of calibration bins.

    Returns
    -------
    dict
        Calibration data for plotting.
    """
    draw_probs = [p.draw for p in predictions]
    draw_actuals = [1 if a == "D" else 0 for a in actuals]

    # Calculate calibration curve
    prob_true, prob_pred = calibration_curve(
        draw_actuals, draw_probs, n_bins=n_bins, strategy="uniform"
    )

    # Brier score for draws
    brier = np.mean([(p - a) ** 2 for p, a in zip(draw_probs, draw_actuals, strict=False)])

    return {
        "prob_true": prob_true.tolist(),
        "prob_pred": prob_pred.tolist(),
        "brier_score": brier,
        "n_samples": len(predictions),
        "actual_draw_rate": np.mean(draw_actuals),
        "predicted_draw_rate": np.mean(draw_probs),
    }
