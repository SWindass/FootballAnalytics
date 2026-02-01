"""XGBoost classifier for match outcome prediction."""

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


@dataclass
class XGBoostConfig:
    """Configuration for XGBoost model."""

    n_estimators: int = 100
    max_depth: int = 5
    learning_rate: float = 0.1
    min_child_weight: int = 1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    random_state: int = 42


class MatchOutcomeClassifier:
    """XGBoost classifier for predicting match outcomes (Home/Draw/Away)."""

    FEATURE_COLUMNS = [
        # ELO ratings
        "home_elo",
        "away_elo",
        "elo_diff",
        # Form
        "home_form_points",
        "away_form_points",
        "form_diff",
        # Goals
        "home_avg_scored",
        "home_avg_conceded",
        "away_avg_scored",
        "away_avg_conceded",
        "goal_diff_attack",
        "goal_diff_defense",
        # xG
        "home_avg_xg_for",
        "home_avg_xg_against",
        "away_avg_xg_for",
        "away_avg_xg_against",
        "xg_diff_for",
        "xg_diff_against",
        # Home/Away performance
        "home_home_ppg",  # Points per game at home
        "away_away_ppg",  # Points per game away
        # Head to head (optional)
        "h2h_home_wins",
        "h2h_draws",
        "h2h_away_wins",
    ]

    def __init__(self, config: XGBoostConfig | None = None):
        self.config = config or XGBoostConfig()
        self.model: XGBClassifier | None = None
        self.scaler = StandardScaler()
        self.feature_importance: dict[str, float] | None = None

    def _create_model(self) -> XGBClassifier:
        """Create XGBoost classifier with configured parameters."""
        return XGBClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            min_child_weight=self.config.min_child_weight,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            random_state=self.config.random_state,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            use_label_encoder=False,
        )

    def prepare_features(self, matches_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare feature matrix from match data.

        Expected columns in matches_df:
        - All columns in FEATURE_COLUMNS
        - outcome: 0=Home, 1=Draw, 2=Away (for training)
        """
        # Select feature columns that exist
        available_features = [c for c in self.FEATURE_COLUMNS if c in matches_df.columns]

        features = matches_df[available_features].copy()

        # Fill missing values with sensible defaults
        features = features.fillna(features.mean())

        return features

    def train(
        self,
        matches_df: pd.DataFrame,
        test_size: float = 0.2,
    ) -> dict[str, Any]:
        """Train the classifier on historical match data.

        Args:
            matches_df: DataFrame with features and 'outcome' column
            test_size: Fraction of data to use for testing

        Returns:
            Dict with training metrics
        """
        X = self.prepare_features(matches_df)
        y = matches_df["outcome"].values

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.config.random_state
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train model
        self.model = self._create_model()
        self.model.fit(
            X_train_scaled,
            y_train,
            eval_set=[(X_test_scaled, y_test)],
            verbose=False,
        )

        # Calculate metrics
        train_accuracy = self.model.score(X_train_scaled, y_train)
        test_accuracy = self.model.score(X_test_scaled, y_test)

        # Cross-validation
        cv_scores = cross_val_score(
            self._create_model(),
            self.scaler.fit_transform(X),
            y,
            cv=5,
            scoring="accuracy",
        )

        # Feature importance
        self.feature_importance = dict(
            zip(X.columns, self.model.feature_importances_, strict=False)
        )

        return {
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "feature_importance": self.feature_importance,
        }

    def predict_probabilities(
        self,
        match_features: pd.DataFrame,
    ) -> np.ndarray:
        """Predict outcome probabilities for matches.

        Args:
            match_features: DataFrame with feature columns

        Returns:
            Array of shape (n_matches, 3) with [home, draw, away] probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        X = self.prepare_features(match_features)
        X_scaled = self.scaler.transform(X)

        return self.model.predict_proba(X_scaled)

    def predict_match(
        self,
        features: dict[str, float],
    ) -> tuple[float, float, float]:
        """Predict probabilities for a single match.

        Args:
            features: Dict mapping feature names to values

        Returns:
            Tuple of (home_prob, draw_prob, away_prob)
        """
        df = pd.DataFrame([features])
        probs = self.predict_probabilities(df)[0]
        return tuple(probs)

    def save(self, path: Path) -> None:
        """Save model and scaler to disk."""
        if self.model is None:
            raise ValueError("No model to save")

        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "config": self.config,
            "feature_importance": self.feature_importance,
        }

        with open(path, "wb") as f:
            pickle.dump(model_data, f)

    def load(self, path: Path) -> None:
        """Load model and scaler from disk."""
        with open(path, "rb") as f:
            model_data = pickle.load(f)

        self.model = model_data["model"]
        self.scaler = model_data["scaler"]
        self.config = model_data["config"]
        self.feature_importance = model_data.get("feature_importance")


def build_feature_dataframe(
    match: dict,
    home_stats: dict,
    away_stats: dict,
    home_elo: float,
    away_elo: float,
    h2h_stats: dict | None = None,
) -> pd.DataFrame:
    """Build feature DataFrame for a single match prediction.

    Args:
        match: Match dict with basic info
        home_stats: Home team statistics
        away_stats: Away team statistics
        home_elo: Home team ELO rating
        away_elo: Away team ELO rating
        h2h_stats: Optional head-to-head statistics

    Returns:
        Single-row DataFrame with all features
    """
    features = {
        # ELO
        "home_elo": home_elo,
        "away_elo": away_elo,
        "elo_diff": home_elo - away_elo,
        # Form
        "home_form_points": home_stats.get("form_points", 7.5),
        "away_form_points": away_stats.get("form_points", 7.5),
        "form_diff": home_stats.get("form_points", 7.5) - away_stats.get("form_points", 7.5),
        # Goals
        "home_avg_scored": home_stats.get("avg_goals_scored", 1.4),
        "home_avg_conceded": home_stats.get("avg_goals_conceded", 1.4),
        "away_avg_scored": away_stats.get("avg_goals_scored", 1.4),
        "away_avg_conceded": away_stats.get("avg_goals_conceded", 1.4),
        "goal_diff_attack": home_stats.get("avg_goals_scored", 1.4)
        - away_stats.get("avg_goals_scored", 1.4),
        "goal_diff_defense": away_stats.get("avg_goals_conceded", 1.4)
        - home_stats.get("avg_goals_conceded", 1.4),
        # xG
        "home_avg_xg_for": home_stats.get("avg_xg_for", 1.4),
        "home_avg_xg_against": home_stats.get("avg_xg_against", 1.4),
        "away_avg_xg_for": away_stats.get("avg_xg_for", 1.4),
        "away_avg_xg_against": away_stats.get("avg_xg_against", 1.4),
        "xg_diff_for": home_stats.get("avg_xg_for", 1.4) - away_stats.get("avg_xg_for", 1.4),
        "xg_diff_against": away_stats.get("avg_xg_against", 1.4)
        - home_stats.get("avg_xg_against", 1.4),
        # Home/Away performance
        "home_home_ppg": _calculate_ppg(
            home_stats.get("home_wins", 0),
            home_stats.get("home_draws", 0),
            home_stats.get("home_losses", 0),
        ),
        "away_away_ppg": _calculate_ppg(
            away_stats.get("away_wins", 0),
            away_stats.get("away_draws", 0),
            away_stats.get("away_losses", 0),
        ),
        # H2H
        "h2h_home_wins": h2h_stats.get("home_wins", 0) if h2h_stats else 0,
        "h2h_draws": h2h_stats.get("draws", 0) if h2h_stats else 0,
        "h2h_away_wins": h2h_stats.get("away_wins", 0) if h2h_stats else 0,
    }

    return pd.DataFrame([features])


def _calculate_ppg(wins: int, draws: int, losses: int) -> float:
    """Calculate points per game."""
    games = wins + draws + losses
    if games == 0:
        return 1.5  # League average
    return (wins * 3 + draws) / games
