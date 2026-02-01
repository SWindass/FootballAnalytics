"""
Zero-Inflated Poisson + Dixon-Coles Model.

Models the excess 0-0 draws separately from the regular Poisson process,
then applies Dixon-Coles correlation correction.

The key insight: 0-0 draws occur more often than Poisson predicts because:
1. Some matches have "structural zeros" (defensive tactics, weather, etc.)
2. Standard Poisson underestimates the probability of scoreless draws

Architecture:
1. Logistic regression predicts P(structural_zero) based on match features
2. ZIP combines structural zeros with Poisson process
3. Dixon-Coles corrects low-score correlations
4. Final probabilities for all outcomes
"""

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.stats import poisson
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


@dataclass
class ZIPMatchFeatures:
    """Features for ZIP structural zero prediction."""
    rating_diff_abs: float  # Absolute rating difference (evenly matched = low)
    total_xg: float  # Expected total goals (low = more likely 0-0)
    both_defensive: int  # 1 if both teams have xGA < league avg
    mid_table_clash: int  # 1 if both teams positioned 8-14
    recent_draw_rate: float  # From recalibration system (elevated = more 0-0s)

    # Additional features
    home_xg: float = 1.35
    away_xg: float = 1.35
    strength_parity: float = 0.5  # 1/(1+|rating_diff|)

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for model input."""
        return np.array([
            self.rating_diff_abs,
            self.total_xg,
            self.both_defensive,
            self.mid_table_clash,
            self.recent_draw_rate,
            self.strength_parity,
        ])


@dataclass
class ZIPPrediction:
    """Prediction from ZIP+Dixon-Coles model."""
    home_prob: float
    draw_prob: float
    away_prob: float

    # Score matrix
    score_matrix: np.ndarray | None = None

    # Component probabilities
    p_structural_zero: float = 0.0  # P(structural 0-0)
    p_poisson_0_0: float = 0.0  # P(0-0) from Poisson alone
    p_combined_0_0: float = 0.0  # Final P(0-0) after ZIP

    # Confidence
    confidence: float = 0.0

    def get_prediction(self, draw_threshold: float = 0.26, parity_threshold: float = 0.08) -> str:
        """Get predicted outcome with draw threshold adjustment."""
        # Check if teams are evenly matched
        is_parity = abs(self.home_prob - self.away_prob) < parity_threshold
        draw_is_significant = self.draw_prob >= draw_threshold

        if is_parity and draw_is_significant:
            return 'D'

        if self.draw_prob >= self.home_prob and self.draw_prob >= self.away_prob:
            return 'D'

        if self.home_prob >= self.away_prob:
            return 'H'
        return 'A'


class ZIPDixonColesModel:
    """
    Zero-Inflated Poisson with Dixon-Coles correction.

    Combines:
    1. Logistic regression for structural zero probability
    2. ZIP distribution for score probabilities
    3. Dixon-Coles correction for low-score correlation

    Parameters
    ----------
    rho : float
        Dixon-Coles correlation parameter (typically -0.11 to -0.15)
    structural_zero_weight : float
        How much to weight structural zeros vs Poisson (0.0-1.0)
    min_samples_for_training : int
        Minimum samples before training the structural zero model
    """

    def __init__(
        self,
        rho: float = -0.13,
        structural_zero_weight: float = 0.5,
        min_samples_for_training: int = 200,
    ):
        self.rho = rho
        self.structural_zero_weight = structural_zero_weight
        self.min_samples_for_training = min_samples_for_training

        # Structural zero model
        self._zero_model: LogisticRegression | None = None
        self._scaler = StandardScaler()
        self._is_trained = False

        # Training data
        self._training_features: list[np.ndarray] = []
        self._training_labels: list[int] = []  # 1 if 0-0, else 0

        # League context
        self._league_avg_xg: float = 1.35
        self._recent_draw_rate: float = 0.234  # Historical baseline

        # Team context (for features)
        self._team_positions: dict[str, int] = {}
        self._team_xga: dict[str, float] = {}

    def set_league_context(
        self,
        positions: dict[str, int],
        xga: dict[str, float],
        league_avg_xg: float = 1.35,
        recent_draw_rate: float = 0.234,
    ) -> None:
        """Set current league context for feature calculation."""
        self._team_positions = positions
        self._team_xga = xga
        self._league_avg_xg = league_avg_xg
        self._recent_draw_rate = recent_draw_rate

    def set_rho(self, rho: float) -> None:
        """Update Dixon-Coles rho (e.g., from recalibration)."""
        self.rho = rho

    def calculate_features(
        self,
        home_team: str,
        away_team: str,
        home_xg: float,
        away_xg: float,
        rating_diff: float = 0.0,
    ) -> ZIPMatchFeatures:
        """Calculate features for a match."""
        # Get team positions
        home_pos = self._team_positions.get(home_team, 10)
        away_pos = self._team_positions.get(away_team, 10)

        # Get team defensive stats
        home_xga = self._team_xga.get(home_team, self._league_avg_xg)
        away_xga = self._team_xga.get(away_team, self._league_avg_xg)

        return ZIPMatchFeatures(
            rating_diff_abs=abs(rating_diff),
            total_xg=home_xg + away_xg,
            both_defensive=1 if (home_xga < self._league_avg_xg and away_xga < self._league_avg_xg) else 0,
            mid_table_clash=1 if (8 <= home_pos <= 14 and 8 <= away_pos <= 14) else 0,
            recent_draw_rate=self._recent_draw_rate,
            home_xg=home_xg,
            away_xg=away_xg,
            strength_parity=1.0 / (1.0 + abs(rating_diff)),
        )

    def collect_training_sample(
        self,
        features: ZIPMatchFeatures,
        home_goals: int,
        away_goals: int,
    ) -> None:
        """Collect a training sample for the structural zero model."""
        is_0_0 = 1 if (home_goals == 0 and away_goals == 0) else 0

        self._training_features.append(features.to_array())
        self._training_labels.append(is_0_0)

    def train_structural_zero_model(self) -> dict:
        """Train the logistic regression model for structural zeros."""
        if len(self._training_features) < self.min_samples_for_training:
            return {
                'status': 'insufficient_data',
                'samples': len(self._training_features),
                'required': self.min_samples_for_training,
            }

        X = np.array(self._training_features)
        y = np.array(self._training_labels)

        # Scale features
        X_scaled = self._scaler.fit_transform(X)

        # Train with class weight to handle imbalance (0-0 is rare)
        self._zero_model = LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=42,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._zero_model.fit(X_scaled, y)

        self._is_trained = True

        # Calculate metrics
        y_pred = self._zero_model.predict(X_scaled)
        self._zero_model.predict_proba(X_scaled)[:, 1]

        accuracy = (y_pred == y).mean()
        zero_zero_count = y.sum()
        zero_zero_rate = zero_zero_count / len(y)

        # Recall for 0-0 matches
        true_positives = ((y_pred == 1) & (y == 1)).sum()
        recall = true_positives / zero_zero_count if zero_zero_count > 0 else 0

        return {
            'status': 'trained',
            'samples': len(y),
            'zero_zero_count': int(zero_zero_count),
            'zero_zero_rate': zero_zero_rate,
            'accuracy': accuracy,
            'recall_0_0': recall,
            'feature_importance': dict(zip(
                ['rating_diff_abs', 'total_xg', 'both_defensive',
                 'mid_table_clash', 'recent_draw_rate', 'strength_parity'],
                self._zero_model.coef_[0].tolist(), strict=False
            )),
        }

    def predict_structural_zero_prob(self, features: ZIPMatchFeatures) -> float:
        """Predict probability of structural zero (0-0)."""
        if not self._is_trained:
            # Use simple heuristic if model not trained
            base_prob = 0.08  # Historical 0-0 rate

            # Adjust based on features
            if features.total_xg < 2.3:
                base_prob *= 1.3
            if features.both_defensive:
                base_prob *= 1.2
            if features.strength_parity > 0.7:
                base_prob *= 1.15
            if features.recent_draw_rate > 0.30:
                base_prob *= (features.recent_draw_rate / 0.234)

            return min(0.25, base_prob)

        # Use trained model
        X = features.to_array().reshape(1, -1)
        X_scaled = self._scaler.transform(X)
        prob = self._zero_model.predict_proba(X_scaled)[0, 1]

        # Apply weight to blend with Poisson
        return prob * self.structural_zero_weight

    def poisson_probability(
        self,
        home_goals: int,
        away_goals: int,
        home_xg: float,
        away_xg: float,
    ) -> float:
        """Calculate base Poisson probability for a scoreline."""
        return poisson.pmf(home_goals, home_xg) * poisson.pmf(away_goals, away_xg)

    def dixon_coles_correction(
        self,
        home_goals: int,
        away_goals: int,
        home_xg: float,
        away_xg: float,
    ) -> float:
        """Apply Dixon-Coles correlation correction."""
        if home_goals == 0 and away_goals == 0:
            return 1 - home_xg * away_xg * self.rho
        elif home_goals == 0 and away_goals == 1:
            return 1 + home_xg * self.rho
        elif home_goals == 1 and away_goals == 0:
            return 1 + away_xg * self.rho
        elif home_goals == 1 and away_goals == 1:
            return 1 - self.rho
        return 1.0

    def calculate_zip_probability(
        self,
        home_goals: int,
        away_goals: int,
        home_xg: float,
        away_xg: float,
        p_structural: float,
    ) -> float:
        """
        Calculate ZIP probability for a scoreline.

        For 0-0:
            P(0,0) = p_structural + (1-p_structural) × Poisson(0,0)

        For other scores:
            P(h,a) = (1-p_structural) × Poisson(h,a)
        """
        poisson_prob = self.poisson_probability(home_goals, away_goals, home_xg, away_xg)

        if home_goals == 0 and away_goals == 0:
            # Zero-inflated: structural zeros + Poisson zeros
            return p_structural + (1 - p_structural) * poisson_prob
        else:
            # Non-zero scores: only from Poisson process
            return (1 - p_structural) * poisson_prob

    def calculate_score_matrix(
        self,
        home_xg: float,
        away_xg: float,
        p_structural: float,
        max_goals: int = 7,
    ) -> np.ndarray:
        """Calculate full score probability matrix with ZIP + Dixon-Coles."""
        matrix = np.zeros((max_goals + 1, max_goals + 1))

        for h in range(max_goals + 1):
            for a in range(max_goals + 1):
                # ZIP probability
                zip_prob = self.calculate_zip_probability(h, a, home_xg, away_xg, p_structural)

                # Dixon-Coles correction
                dc_correction = self.dixon_coles_correction(h, a, home_xg, away_xg)

                matrix[h, a] = zip_prob * dc_correction

        # Normalize
        total = matrix.sum()
        if total > 0:
            matrix /= total

        return matrix

    def matrix_to_probabilities(self, matrix: np.ndarray) -> tuple[float, float, float]:
        """Convert score matrix to outcome probabilities."""
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

    def predict_match(
        self,
        home_team: str,
        away_team: str,
        home_xg: float,
        away_xg: float,
        rating_diff: float = 0.0,
    ) -> ZIPPrediction:
        """
        Predict match outcome using ZIP + Dixon-Coles.

        Parameters
        ----------
        home_team, away_team : str
            Team names
        home_xg, away_xg : float
            Expected goals for each team
        rating_diff : float
            Rating difference (home - away), used for features

        Returns
        -------
        ZIPPrediction
            Prediction with probabilities and diagnostics
        """
        # Calculate features
        features = self.calculate_features(
            home_team, away_team, home_xg, away_xg, rating_diff
        )

        # Predict structural zero probability
        p_structural = self.predict_structural_zero_prob(features)

        # Calculate score matrix
        matrix = self.calculate_score_matrix(home_xg, away_xg, p_structural)

        # Get outcome probabilities
        home_prob, draw_prob, away_prob = self.matrix_to_probabilities(matrix)

        # Calculate component probabilities for diagnostics
        p_poisson_0_0 = self.poisson_probability(0, 0, home_xg, away_xg)
        p_combined_0_0 = matrix[0, 0]

        # Confidence based on model training and feature quality
        confidence = 0.5
        if self._is_trained:
            confidence = 0.7
        if features.strength_parity > 0.6:
            confidence *= 0.9  # Lower confidence for even matches

        return ZIPPrediction(
            home_prob=home_prob,
            draw_prob=draw_prob,
            away_prob=away_prob,
            score_matrix=matrix,
            p_structural_zero=p_structural,
            p_poisson_0_0=p_poisson_0_0,
            p_combined_0_0=p_combined_0_0,
            confidence=confidence,
        )

    def update_after_match(
        self,
        home_team: str,
        away_team: str,
        home_goals: int,
        away_goals: int,
        home_xg: float,
        away_xg: float,
        rating_diff: float = 0.0,
    ) -> None:
        """Update model after a match result."""
        features = self.calculate_features(
            home_team, away_team, home_xg, away_xg, rating_diff
        )
        self.collect_training_sample(features, home_goals, away_goals)

    def reset(self) -> None:
        """Reset model state."""
        self._zero_model = None
        self._is_trained = False
        self._training_features.clear()
        self._training_labels.clear()
        self._scaler = StandardScaler()

    def get_0_0_boost_factor(self) -> float:
        """Get the current boost factor for 0-0 draws vs standard Poisson."""
        if not self._is_trained:
            return 1.0 + self.structural_zero_weight

        # Calculate average boost from training data
        avg_structural = np.mean([
            self.predict_structural_zero_prob(
                ZIPMatchFeatures(*f[:6], f[0], f[1], f[5])
            ) for f in self._training_features[-100:]  # Last 100 samples
        ]) if self._training_features else 0.1

        return 1.0 + avg_structural / 0.08  # Relative to baseline 8% 0-0 rate
