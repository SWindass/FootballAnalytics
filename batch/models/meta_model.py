"""Meta-model: Predicts when our model is more accurate than the market.

The meta-model learns patterns in model-market disagreements to identify
situations where betting on our model's prediction is likely profitable.

Key insight: The market is usually right. We don't try to beat it overall.
Instead, we learn which situations our model handles better than the market.
"""

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import structlog
from sqlalchemy import select

from app.db.database import SyncSessionLocal
from app.db.models import EloRating, Match, MatchAnalysis, MatchStatus, Team, TeamStats

logger = structlog.get_logger()

# Model save paths
MODEL_DIR = Path(__file__).parent / "saved"
META_MODEL_PATH = MODEL_DIR / "meta_model.npz"
META_METADATA_PATH = MODEL_DIR / "meta_model_meta.json"

# Big 6 teams (tend to be more predictable)
BIG_6_TEAMS = {"Arsenal", "Chelsea", "Liverpool", "Manchester City", "Manchester United", "Tottenham Hotspur"}


@dataclass
class MetaModelConfig:
    """Configuration for meta-model."""

    # Minimum disagreement to consider (ignore tiny differences)
    min_disagreement: float = 0.03

    # Training settings
    learning_rate: float = 0.01
    n_iterations: int = 1000
    regularization: float = 0.01

    # Feature weights initialization
    init_bias: float = 0.0


class MetaFeatures:
    """Feature extraction for meta-model."""

    @staticmethod
    def extract_features(
        match: Match,
        analysis: MatchAnalysis,
        outcome: str,
        home_stats: TeamStats | None,
        away_stats: TeamStats | None,
        home_elo: EloRating | None,
        away_elo: EloRating | None,
        home_team: Team | None,
        away_team: Team | None,
    ) -> np.ndarray | None:
        """Extract meta-features for predicting model vs market accuracy.

        Features capture:
        1. Size and nature of disagreement
        2. Match characteristics
        3. Team characteristics
        4. Seasonal context
        """
        try:
            features = []

            # 1. Disagreement features
            model_prob = MetaFeatures._get_model_prob(analysis, outcome)
            market_prob = MetaFeatures._get_market_prob(analysis, outcome)

            if model_prob is None or market_prob is None:
                return None

            disagreement = model_prob - market_prob
            features.append(disagreement)  # Signed disagreement
            features.append(abs(disagreement))  # Magnitude
            features.append(1.0 if disagreement > 0 else 0.0)  # Direction (model higher)

            # 2. Model agreement features
            elo_prob = MetaFeatures._get_elo_prob(analysis, outcome)
            poisson_prob = MetaFeatures._get_poisson_prob(analysis, outcome)

            elo_agrees = 1.0 if elo_prob and elo_prob > market_prob else 0.0
            poisson_agrees = 1.0 if poisson_prob and poisson_prob > market_prob else 0.0
            features.append(elo_agrees)
            features.append(poisson_agrees)
            features.append(elo_agrees + poisson_agrees)  # Count of agreeing models

            # ELO-Poisson agreement
            if elo_prob and poisson_prob:
                features.append(1.0 if abs(elo_prob - poisson_prob) < 0.05 else 0.0)
            else:
                features.append(0.0)

            # 3. Match type features
            is_home_favorite = 1.0 if outcome == "home_win" else 0.0
            is_draw = 1.0 if outcome == "draw" else 0.0
            is_away_win = 1.0 if outcome == "away_win" else 0.0
            features.extend([is_home_favorite, is_draw, is_away_win])

            # 4. Team strength features
            if home_elo and away_elo:
                elo_diff = (float(home_elo.rating) - float(away_elo.rating)) / 400
                is_close_match = 1.0 if abs(elo_diff) < 0.125 else 0.0  # Within 50 ELO points
            else:
                elo_diff = 0.0
                is_close_match = 0.0

            features.append(elo_diff)
            features.append(is_close_match)

            # 5. Team type features (big 6, promoted, etc.)
            is_big_6_home = 1.0 if home_team and home_team.name in BIG_6_TEAMS else 0.0
            is_big_6_away = 1.0 if away_team and away_team.name in BIG_6_TEAMS else 0.0
            big_6_involved = 1.0 if is_big_6_home or is_big_6_away else 0.0
            features.extend([is_big_6_home, is_big_6_away, big_6_involved])

            # 6. Form features
            if home_stats:
                home_form = (home_stats.form_points or 7.5) / 15
            else:
                home_form = 0.5
            if away_stats:
                away_form = (away_stats.form_points or 7.5) / 15
            else:
                away_form = 0.5

            features.append(home_form)
            features.append(away_form)
            features.append(home_form - away_form)  # Form advantage

            # 7. Seasonal context (early season = less reliable form data)
            matchweek = match.matchweek or 20
            is_early_season = 1.0 if matchweek <= 6 else 0.0
            is_late_season = 1.0 if matchweek >= 32 else 0.0
            features.extend([matchweek / 38, is_early_season, is_late_season])

            # 8. xG regression potential
            if outcome == "home_win" and home_stats:
                xg_overperform = MetaFeatures._calculate_xg_overperform(home_stats)
            elif outcome == "away_win" and away_stats:
                xg_overperform = MetaFeatures._calculate_xg_overperform(away_stats)
            else:
                xg_overperform = 0.0

            features.append(xg_overperform)

            # 9. Injury impact
            home_injuries = (home_stats.key_players_out or 0) / 5 if home_stats else 0.0
            away_injuries = (away_stats.key_players_out or 0) / 5 if away_stats else 0.0
            features.extend([home_injuries, away_injuries])

            return np.array(features, dtype=np.float32)

        except Exception as e:
            logger.warning(f"Failed to extract meta-features: {e}")
            return None

    @staticmethod
    def _get_model_prob(analysis: MatchAnalysis, outcome: str) -> float | None:
        """Get consensus model probability for outcome."""
        if outcome == "home_win":
            return float(analysis.consensus_home_prob) if analysis.consensus_home_prob else None
        elif outcome == "draw":
            return float(analysis.consensus_draw_prob) if analysis.consensus_draw_prob else None
        elif outcome == "away_win":
            return float(analysis.consensus_away_prob) if analysis.consensus_away_prob else None
        return None

    @staticmethod
    def _get_market_prob(analysis: MatchAnalysis, outcome: str) -> float | None:
        """Get market probability from historical odds."""
        if not analysis.features:
            return None

        hist_odds = analysis.features.get("historical_odds", {})
        if not hist_odds:
            return None

        if outcome == "home_win":
            return hist_odds.get("implied_home_prob")
        elif outcome == "draw":
            return hist_odds.get("implied_draw_prob")
        elif outcome == "away_win":
            return hist_odds.get("implied_away_prob")
        return None

    @staticmethod
    def _get_elo_prob(analysis: MatchAnalysis, outcome: str) -> float | None:
        """Get ELO model probability for outcome."""
        if outcome == "home_win":
            return float(analysis.elo_home_prob) if analysis.elo_home_prob else None
        elif outcome == "draw":
            return float(analysis.elo_draw_prob) if analysis.elo_draw_prob else None
        elif outcome == "away_win":
            return float(analysis.elo_away_prob) if analysis.elo_away_prob else None
        return None

    @staticmethod
    def _get_poisson_prob(analysis: MatchAnalysis, outcome: str) -> float | None:
        """Get Poisson model probability for outcome."""
        if outcome == "home_win":
            return float(analysis.poisson_home_prob) if analysis.poisson_home_prob else None
        elif outcome == "draw":
            return float(analysis.poisson_draw_prob) if analysis.poisson_draw_prob else None
        elif outcome == "away_win":
            return float(analysis.poisson_away_prob) if analysis.poisson_away_prob else None
        return None

    @staticmethod
    def _calculate_xg_overperform(stats: TeamStats) -> float:
        """Calculate xG over/underperformance."""
        if not stats.xg_for or not stats.goals_scored:
            return 0.0

        games = (stats.home_wins or 0) + (stats.home_draws or 0) + \
                (stats.home_losses or 0) + (stats.away_wins or 0) + \
                (stats.away_draws or 0) + (stats.away_losses or 0)

        if games < 3:
            return 0.0

        goals_pg = stats.goals_scored / games
        xg_pg = float(stats.xg_for) / games
        return (goals_pg - xg_pg) / 2  # Normalize


class MetaModel:
    """Logistic regression meta-model for predicting when to trust our model.

    Uses simple logistic regression for interpretability and to avoid overfitting
    on the relatively small training set of model-market disagreements.
    """

    FEATURE_NAMES = [
        "disagreement",
        "abs_disagreement",
        "model_higher",
        "elo_agrees",
        "poisson_agrees",
        "models_agreeing_count",
        "elo_poisson_agreement",
        "is_home_favorite",
        "is_draw",
        "is_away_win",
        "elo_diff",
        "is_close_match",
        "is_big_6_home",
        "is_big_6_away",
        "big_6_involved",
        "home_form",
        "away_form",
        "form_advantage",
        "matchweek_normalized",
        "is_early_season",
        "is_late_season",
        "xg_overperform",
        "home_injuries",
        "away_injuries",
    ]

    def __init__(self, config: MetaModelConfig | None = None):
        self.config = config or MetaModelConfig()
        self.weights = None
        self.bias = self.config.init_bias
        self.n_features = len(self.FEATURE_NAMES)

    def _ensure_model_dir(self):
        """Ensure model directory exists."""
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

    def load_model(self) -> bool:
        """Load saved model if exists."""
        if not META_MODEL_PATH.exists():
            logger.warning("No saved meta-model found")
            return False

        try:
            data = np.load(META_MODEL_PATH)
            self.weights = data["weights"]
            self.bias = float(data["bias"])
            logger.info("Loaded meta-model")
            return True
        except Exception as e:
            logger.error(f"Failed to load meta-model: {e}")
            return False

    def save_model(self, metadata: dict = None):
        """Save model and metadata."""
        self._ensure_model_dir()

        np.savez(
            META_MODEL_PATH,
            weights=self.weights,
            bias=np.array([self.bias])
        )

        if metadata is None:
            metadata = {}
        metadata["saved_at"] = datetime.now(UTC).isoformat()
        metadata["n_features"] = self.n_features
        metadata["feature_names"] = self.FEATURE_NAMES

        with open(META_METADATA_PATH, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved meta-model to {META_MODEL_PATH}")

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid function."""
        return np.where(
            x >= 0,
            1 / (1 + np.exp(-x)),
            np.exp(x) / (1 + np.exp(x))
        )

    def prepare_training_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Prepare training data from historical matches.

        For each match where model disagreed with market:
        - Label = 1 if model was correct (betting on model would have been profitable)
        - Label = 0 if market was correct
        """
        with SyncSessionLocal() as session:
            logger.info("Loading historical matches for meta-model training...")

            # Get finished matches with analysis and market odds
            stmt = (
                select(Match, MatchAnalysis)
                .join(MatchAnalysis, Match.id == MatchAnalysis.match_id)
                .where(Match.status == MatchStatus.FINISHED)
                .where(MatchAnalysis.consensus_home_prob.isnot(None))
                .order_by(Match.kickoff_time)
            )
            results = list(session.execute(stmt).all())
            logger.info(f"Found {len(results)} finished matches")

            # Bulk load supporting data
            all_team_stats = list(session.execute(select(TeamStats)).scalars().all())
            team_stats_lookup = {
                (ts.team_id, ts.season, ts.matchweek): ts for ts in all_team_stats
            }

            all_elo_ratings = list(session.execute(select(EloRating)).scalars().all())
            elo_lookup = {
                (er.team_id, er.season, er.matchweek): er for er in all_elo_ratings
            }

            all_teams = list(session.execute(select(Team)).scalars().all())
            team_lookup = {t.id: t for t in all_teams}

            features = []
            labels = []
            skipped_no_odds = 0
            skipped_small_disagreement = 0

            for match, analysis in results:
                # Skip if no market odds
                if not analysis.features or "historical_odds" not in analysis.features:
                    skipped_no_odds += 1
                    continue

                hist_odds = analysis.features.get("historical_odds", {})
                if not hist_odds.get("implied_home_prob"):
                    skipped_no_odds += 1
                    continue

                # Get supporting data
                home_stats = team_stats_lookup.get(
                    (match.home_team_id, match.season, match.matchweek - 1)
                )
                away_stats = team_stats_lookup.get(
                    (match.away_team_id, match.season, match.matchweek - 1)
                )
                home_elo = elo_lookup.get(
                    (match.home_team_id, match.season, match.matchweek - 1)
                )
                away_elo = elo_lookup.get(
                    (match.away_team_id, match.season, match.matchweek - 1)
                )
                home_team = team_lookup.get(match.home_team_id)
                away_team = team_lookup.get(match.away_team_id)

                # Determine actual result
                if match.home_score > match.away_score:
                    actual = "home_win"
                elif match.home_score == match.away_score:
                    actual = "draw"
                else:
                    actual = "away_win"

                # Check each outcome for model-market disagreement
                for outcome in ["home_win", "draw", "away_win"]:
                    model_prob = MetaFeatures._get_model_prob(analysis, outcome)
                    market_prob = MetaFeatures._get_market_prob(analysis, outcome)

                    if model_prob is None or market_prob is None:
                        continue

                    disagreement = model_prob - market_prob

                    # Only train on significant disagreements
                    if abs(disagreement) < self.config.min_disagreement:
                        skipped_small_disagreement += 1
                        continue

                    # Extract features
                    feature = MetaFeatures.extract_features(
                        match, analysis, outcome,
                        home_stats, away_stats,
                        home_elo, away_elo,
                        home_team, away_team,
                    )

                    if feature is None:
                        continue

                    # Label: 1 if model was right (higher prob for actual outcome)
                    # Model "wins" if it gave higher probability to what actually happened
                    model_was_right = (
                        (disagreement > 0 and outcome == actual) or
                        (disagreement < 0 and outcome != actual)
                    )

                    features.append(feature)
                    labels.append(1.0 if model_was_right else 0.0)

            logger.info(f"Prepared {len(features)} training samples")
            logger.info(f"Skipped {skipped_no_odds} matches without odds")
            logger.info(f"Skipped {skipped_small_disagreement} small disagreements")

            if len(features) < 100:
                logger.warning("Very few training samples for meta-model")

            return np.array(features, dtype=np.float32), np.array(labels, dtype=np.float32)

    def train(self) -> dict:
        """Train the meta-model using logistic regression with gradient descent."""
        X, y = self.prepare_training_data()

        if len(X) < 50:
            raise ValueError(f"Not enough training data: {len(X)} samples")

        # Initialize weights
        self.weights = np.zeros(X.shape[1], dtype=np.float32)
        self.bias = 0.0

        # Add small noise for symmetry breaking
        self.weights += np.random.randn(X.shape[1]).astype(np.float32) * 0.01

        # Train/val split (time-series: use last 20% for validation)
        n_val = int(len(X) * 0.2)
        X_train, y_train = X[:-n_val], y[:-n_val]
        X_val, y_val = X[-n_val:], y[-n_val:]

        logger.info(f"Training meta-model: {len(X_train)} train, {len(X_val)} val samples")

        best_val_acc = 0
        best_weights = self.weights.copy()
        best_bias = self.bias

        for iteration in range(self.config.n_iterations):
            # Forward pass
            z = np.dot(X_train, self.weights) + self.bias
            predictions = self._sigmoid(z)

            # Calculate gradients
            error = predictions - y_train
            grad_weights = np.dot(X_train.T, error) / len(X_train)
            grad_weights += self.config.regularization * self.weights  # L2 regularization
            grad_bias = np.mean(error)

            # Update
            self.weights -= self.config.learning_rate * grad_weights
            self.bias -= self.config.learning_rate * grad_bias

            # Evaluate on validation set every 100 iterations
            if iteration % 100 == 0:
                val_preds = self._sigmoid(np.dot(X_val, self.weights) + self.bias)
                val_acc = np.mean((val_preds > 0.5) == y_val)

                train_preds = predictions
                train_acc = np.mean((train_preds > 0.5) == y_train)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_weights = self.weights.copy()
                    best_bias = self.bias

                logger.info(f"Iteration {iteration}: train_acc={train_acc:.3f}, val_acc={val_acc:.3f}")

        # Restore best model
        self.weights = best_weights
        self.bias = best_bias

        # Save model
        self.save_model({
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "best_val_acc": float(best_val_acc),
        })

        # Feature importance
        importance = list(zip(self.FEATURE_NAMES, np.abs(self.weights), strict=False))
        importance.sort(key=lambda x: x[1], reverse=True)

        logger.info("\nTop 10 most important features:")
        for name, imp in importance[:10]:
            logger.info(f"  {name}: {imp:.4f}")

        return {
            "best_val_acc": float(best_val_acc),
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "feature_importance": importance,
        }

    def predict(
        self,
        match: Match,
        analysis: MatchAnalysis,
        outcome: str,
        home_stats: TeamStats | None,
        away_stats: TeamStats | None,
        home_elo: EloRating | None,
        away_elo: EloRating | None,
        home_team: Team | None,
        away_team: Team | None,
    ) -> float | None:
        """Predict probability that betting on model's prediction is profitable.

        Args:
            match: Match object
            analysis: Match analysis
            outcome: Outcome to predict ('home_win', 'draw', 'away_win')
            home_stats, away_stats: Team statistics
            home_elo, away_elo: ELO ratings
            home_team, away_team: Team objects

        Returns:
            Probability (0-1) that model is more accurate than market for this bet,
            or None if features couldn't be extracted.
        """
        if self.weights is None:
            if not self.load_model():
                logger.warning("Meta-model not available")
                return None

        features = MetaFeatures.extract_features(
            match, analysis, outcome,
            home_stats, away_stats,
            home_elo, away_elo,
            home_team, away_team,
        )

        if features is None:
            return None

        z = np.dot(features, self.weights) + self.bias
        return float(self._sigmoid(z))


def train_meta_model():
    """Train the meta-model."""
    model = MetaModel()
    metrics = model.train()
    return metrics


if __name__ == "__main__":
    import logging
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

    print("Training Meta-Model...")
    metrics = train_meta_model()
    print("\nTraining complete:")
    print(f"  Best validation accuracy: {metrics['best_val_acc']:.1%}")
    print(f"  Training samples: {metrics['train_samples']}")
    print(f"  Validation samples: {metrics['val_samples']}")
