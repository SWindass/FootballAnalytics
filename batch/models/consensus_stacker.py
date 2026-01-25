"""Consensus-Aware Neural Stacker - learns to boost confidence when models agree.

Key insight: When ELO, Poisson, and market odds all point to the same outcome,
the prediction is more reliable. This model learns to:
1. Detect agreement between prediction sources
2. Boost confidence when sources align
3. Reduce confidence (or abstain) when sources disagree

Features include explicit agreement metrics that let the network learn
the relationship between model consensus and prediction accuracy.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from typing import Optional
from datetime import datetime, timezone
from decimal import Decimal

import structlog
from sqlalchemy import select

from app.db.database import SyncSessionLocal
from app.db.models import Match, MatchAnalysis, MatchStatus, EloRating, TeamStats

logger = structlog.get_logger()

MODEL_DIR = Path(__file__).parent / "saved"
MODEL_PATH = MODEL_DIR / "consensus_stacker.pt"
METADATA_PATH = MODEL_DIR / "consensus_stacker_meta.json"


class ConsensusNet(nn.Module):
    """Neural network that learns from model agreement.

    Architecture:
    - Input: Model predictions + agreement features
    - Hidden layers learn optimal combination
    - Output: Calibrated probabilities with confidence
    """

    def __init__(self, input_size: int, hidden_sizes: list = [64, 32], dropout: float = 0.3):
        super().__init__()

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, 3))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.network(x)
        return torch.softmax(logits, dim=1)


class ConsensusStacker:
    """Stacker that explicitly models agreement between prediction sources."""

    FEATURE_NAMES = [
        # === RAW PREDICTIONS (9 features) ===
        "elo_home_prob", "elo_draw_prob", "elo_away_prob",
        "poisson_home_prob", "poisson_draw_prob", "poisson_away_prob",
        "market_home_prob", "market_draw_prob", "market_away_prob",

        # === AGREEMENT FEATURES (10 features) ===
        # Average predictions (ensemble baseline)
        "avg_home_prob", "avg_draw_prob", "avg_away_prob",

        # Disagreement metrics (lower = more agreement)
        "home_std",  # Std dev of home predictions across 3 sources
        "draw_std",  # Std dev of draw predictions
        "away_std",  # Std dev of away predictions

        # Agreement signals (higher = more confidence)
        "max_agreement",  # Max probability where all 3 sources agree on favorite
        "favorite_agreement",  # 1 if all 3 pick same favorite, else 0
        "prediction_entropy",  # Entropy of average prediction (lower = clearer favorite)

        # Disagreement-draw correlation (empirical: +5pp draw rate when disagree)
        "disagreement_draw_boost",  # 1 if models disagree (draw more likely), else 0

        # === STRENGTH FEATURES (3 features) ===
        "elo_diff",  # Normalized ELO difference
        "market_favorite_strength",  # How strongly market favors one outcome
        "model_confidence_gap",  # Gap between top 2 average probabilities
    ]

    def __init__(self):
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = len(self.FEATURE_NAMES)

    def _ensure_model_dir(self):
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

    def build_model(self, hidden_sizes: list = [64, 32], dropout: float = 0.3):
        self.model = ConsensusNet(
            input_size=self.input_size,
            hidden_sizes=hidden_sizes,
            dropout=dropout
        ).to(self.device)
        return self.model

    def load_model(self) -> bool:
        if not MODEL_PATH.exists():
            logger.warning("No saved consensus model found")
            return False

        try:
            self.build_model()
            self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
            self.model.eval()
            logger.info("Loaded consensus stacker model")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def save_model(self, metadata: dict = None):
        self._ensure_model_dir()
        torch.save(self.model.state_dict(), MODEL_PATH)

        if metadata:
            with open(METADATA_PATH, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)

        logger.info(f"Saved model to {MODEL_PATH}")

    def prepare_training_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Prepare training data with agreement features."""

        with SyncSessionLocal() as session:
            logger.info("Loading training data...")

            # Get finished matches with all predictions
            stmt = (
                select(Match, MatchAnalysis)
                .join(MatchAnalysis, Match.id == MatchAnalysis.match_id)
                .where(Match.status == MatchStatus.FINISHED)
                .where(MatchAnalysis.elo_home_prob.isnot(None))
                .where(MatchAnalysis.poisson_home_prob.isnot(None))
                .order_by(Match.kickoff_time)
            )
            results = list(session.execute(stmt).all())
            logger.info(f"Found {len(results)} matches")

            # Load ELO ratings
            all_elo = list(session.execute(select(EloRating)).scalars().all())
            elo_lookup = {
                (e.team_id, e.season, e.matchweek): float(e.rating)
                for e in all_elo
            }

            features = []
            labels = []

            for match, analysis in results:
                feature = self._build_feature_vector(match, analysis, elo_lookup)
                if feature is not None:
                    features.append(feature)

                    # Label
                    if match.home_score > match.away_score:
                        labels.append(0)
                    elif match.home_score == match.away_score:
                        labels.append(1)
                    else:
                        labels.append(2)

            logger.info(f"Prepared {len(features)} samples with agreement features")
            return np.array(features, dtype=np.float32), np.array(labels, dtype=np.int64)

    def _build_feature_vector(
        self,
        match: Match,
        analysis: MatchAnalysis,
        elo_lookup: dict,
    ) -> Optional[list]:
        """Build feature vector with agreement metrics."""

        def safe_float(val, default=0.0):
            if val is None:
                return default
            if isinstance(val, Decimal):
                return float(val)
            return float(val)

        try:
            # === RAW PREDICTIONS ===
            elo_home = safe_float(analysis.elo_home_prob, 0.4)
            elo_draw = safe_float(analysis.elo_draw_prob, 0.27)
            elo_away = safe_float(analysis.elo_away_prob, 0.33)

            poisson_home = safe_float(analysis.poisson_home_prob, 0.4)
            poisson_draw = safe_float(analysis.poisson_draw_prob, 0.27)
            poisson_away = safe_float(analysis.poisson_away_prob, 0.33)

            # Market odds (from historical data or defaults)
            market_home = 0.4
            market_draw = 0.27
            market_away = 0.33

            if analysis.features:
                hist_odds = analysis.features.get("historical_odds", {})
                if hist_odds:
                    market_home = hist_odds.get("implied_home_prob", 0.4)
                    market_draw = hist_odds.get("implied_draw_prob", 0.27)
                    market_away = hist_odds.get("implied_away_prob", 0.33)

            # === AGREEMENT FEATURES ===
            # Average predictions (simple ensemble)
            avg_home = (elo_home + poisson_home + market_home) / 3
            avg_draw = (elo_draw + poisson_draw + market_draw) / 3
            avg_away = (elo_away + poisson_away + market_away) / 3

            # Standard deviation (disagreement measure)
            home_std = np.std([elo_home, poisson_home, market_home])
            draw_std = np.std([elo_draw, poisson_draw, market_draw])
            away_std = np.std([elo_away, poisson_away, market_away])

            # Agreement signals
            # Which outcome does each model favor?
            elo_favorite = np.argmax([elo_home, elo_draw, elo_away])
            poisson_favorite = np.argmax([poisson_home, poisson_draw, poisson_away])
            market_favorite = np.argmax([market_home, market_draw, market_away])

            # Do all 3 agree on the favorite?
            favorite_agreement = 1.0 if (elo_favorite == poisson_favorite == market_favorite) else 0.0

            # Maximum probability where all agree
            if favorite_agreement:
                if elo_favorite == 0:
                    max_agreement = min(elo_home, poisson_home, market_home)
                elif elo_favorite == 1:
                    max_agreement = min(elo_draw, poisson_draw, market_draw)
                else:
                    max_agreement = min(elo_away, poisson_away, market_away)
            else:
                max_agreement = 0.0

            # Entropy of average prediction (lower = clearer favorite)
            avg_probs = np.array([avg_home, avg_draw, avg_away])
            avg_probs = np.clip(avg_probs, 1e-10, 1.0)  # Avoid log(0)
            prediction_entropy = -np.sum(avg_probs * np.log(avg_probs))

            # Disagreement-draw boost: when models disagree, draws are +5pp more likely
            # Analysis shows: agree=23.7% draws, disagree=28.6% draws
            disagreement_draw_boost = 0.0 if favorite_agreement else 1.0

            # === STRENGTH FEATURES ===
            # ELO difference
            home_elo = elo_lookup.get(
                (match.home_team_id, match.season, match.matchweek - 1), 1500
            )
            away_elo = elo_lookup.get(
                (match.away_team_id, match.season, match.matchweek - 1), 1500
            )
            elo_diff = (home_elo - away_elo) / 400  # Normalized

            # Market favorite strength (how strongly market favors one outcome)
            market_favorite_strength = max(market_home, market_draw, market_away)

            # Gap between top 2 average probabilities
            sorted_avg = sorted([avg_home, avg_draw, avg_away], reverse=True)
            model_confidence_gap = sorted_avg[0] - sorted_avg[1]

            return [
                # Raw predictions
                elo_home, elo_draw, elo_away,
                poisson_home, poisson_draw, poisson_away,
                market_home, market_draw, market_away,
                # Agreement features
                avg_home, avg_draw, avg_away,
                home_std, draw_std, away_std,
                max_agreement, favorite_agreement, prediction_entropy,
                disagreement_draw_boost,
                # Strength features
                elo_diff, market_favorite_strength, model_confidence_gap,
            ]

        except Exception as e:
            logger.warning(f"Failed to build features: {e}")
            return None

    def train(
        self,
        epochs: int = 100,
        batch_size: int = 64,
        lr: float = 0.001,
        hidden_sizes: list = [64, 32],
    ) -> dict:
        """Train the consensus stacker."""

        X, y = self.prepare_training_data()

        # Time-series split (last 20% for validation)
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}")

        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.LongTensor(y_val)
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Build model
        self.build_model(hidden_sizes=hidden_sizes)

        # Calculate class weights to handle imbalanced classes
        # Home wins (~46%), Draws (~26%), Away wins (~28%)
        # Use mild weighting to slightly boost draw predictions without over-correcting
        # Target: predict draws ~20-25% of the time (actual is 25.6%)
        class_counts = np.bincount(y_train, minlength=3)
        total_samples = len(y_train)

        # Mild inverse frequency weighting with dampening factor
        # sqrt() dampens the effect so we don't over-correct
        raw_weights = total_samples / (3 * class_counts + 1e-6)
        dampened_weights = np.sqrt(raw_weights)
        # Normalize so mean = 1
        dampened_weights = dampened_weights / dampened_weights.mean()
        class_weights = torch.FloatTensor(dampened_weights).to(self.device)
        logger.info(f"Class weights: H={class_weights[0]:.2f}, D={class_weights[1]:.2f}, A={class_weights[2]:.2f}")

        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

        best_val_acc = 0
        best_epoch = 0

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Validation
            self.model.eval()
            val_loss = 0
            correct = 0
            total = 0

            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)

                    outputs = self.model(X_batch)
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs, 1)
                    total += y_batch.size(0)
                    correct += (predicted == y_batch).sum().item()

            val_acc = correct / total
            scheduler.step(val_loss)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                self.save_model({
                    "val_accuracy": val_acc,
                    "epoch": epoch,
                    "trained_at": datetime.now(timezone.utc).isoformat(),
                })

            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch}: val_acc={val_acc:.3f}, best={best_val_acc:.3f}")

        # Load best model
        self.load_model()
        logger.info(f"Training complete. Best val_acc={best_val_acc:.3f} at epoch {best_epoch}")

        return {
            "val_accuracy": best_val_acc,
            "best_epoch": best_epoch,
            "train_samples": len(X_train),
            "val_samples": len(X_val),
        }

    def predict(
        self,
        elo_probs: tuple[float, float, float],
        poisson_probs: tuple[float, float, float],
        market_probs: tuple[float, float, float],
        elo_diff: float = 0.0,
    ) -> tuple[float, float, float, float]:
        """Make prediction with confidence score.

        Returns:
            Tuple of (home_prob, draw_prob, away_prob, confidence)
            where confidence is higher when models agree
        """
        if self.model is None:
            raise ValueError("Model not loaded")

        # Build feature vector
        elo_home, elo_draw, elo_away = elo_probs
        poisson_home, poisson_draw, poisson_away = poisson_probs
        market_home, market_draw, market_away = market_probs

        # Agreement features
        avg_home = (elo_home + poisson_home + market_home) / 3
        avg_draw = (elo_draw + poisson_draw + market_draw) / 3
        avg_away = (elo_away + poisson_away + market_away) / 3

        home_std = np.std([elo_home, poisson_home, market_home])
        draw_std = np.std([elo_draw, poisson_draw, market_draw])
        away_std = np.std([elo_away, poisson_away, market_away])

        elo_favorite = np.argmax([elo_home, elo_draw, elo_away])
        poisson_favorite = np.argmax([poisson_home, poisson_draw, poisson_away])
        market_favorite = np.argmax([market_home, market_draw, market_away])

        favorite_agreement = 1.0 if (elo_favorite == poisson_favorite == market_favorite) else 0.0

        if favorite_agreement:
            probs_list = [
                [elo_home, poisson_home, market_home],
                [elo_draw, poisson_draw, market_draw],
                [elo_away, poisson_away, market_away],
            ]
            max_agreement = min(probs_list[elo_favorite])
        else:
            max_agreement = 0.0

        avg_probs = np.array([avg_home, avg_draw, avg_away])
        avg_probs = np.clip(avg_probs, 1e-10, 1.0)
        prediction_entropy = -np.sum(avg_probs * np.log(avg_probs))

        # Disagreement-draw boost
        disagreement_draw_boost = 0.0 if favorite_agreement else 1.0

        market_favorite_strength = max(market_home, market_draw, market_away)
        sorted_avg = sorted([avg_home, avg_draw, avg_away], reverse=True)
        model_confidence_gap = sorted_avg[0] - sorted_avg[1]

        features = [
            elo_home, elo_draw, elo_away,
            poisson_home, poisson_draw, poisson_away,
            market_home, market_draw, market_away,
            avg_home, avg_draw, avg_away,
            home_std, draw_std, away_std,
            max_agreement, favorite_agreement, prediction_entropy,
            disagreement_draw_boost,
            elo_diff / 400, market_favorite_strength, model_confidence_gap,
        ]

        # Predict
        self.model.eval()
        with torch.no_grad():
            X = torch.FloatTensor([features]).to(self.device)
            probs = self.model(X)[0].cpu().numpy()

        # Calculate confidence based on agreement and prediction strength
        # Higher when: models agree + prediction is decisive
        agreement_score = 1 - np.mean([home_std, draw_std, away_std])
        decisiveness = max(probs) - sorted(probs)[-2]
        confidence = (agreement_score + decisiveness) / 2

        return probs[0], probs[1], probs[2], confidence

    def analyze_agreement_accuracy(self):
        """Analyze how accuracy and draw rates vary with model agreement."""

        X, y = self.prepare_training_data()

        # Use validation set
        split_idx = int(len(X) * 0.8)
        X_val, y_val = X[split_idx:], y[split_idx:]

        # Agreement is stored in feature index 17 (favorite_agreement)
        agreement_idx = self.FEATURE_NAMES.index("favorite_agreement")

        agreed = X_val[:, agreement_idx] == 1.0
        disagreed = X_val[:, agreement_idx] == 0.0

        # Simple prediction: use average probs (features 9-11)
        avg_home_idx = self.FEATURE_NAMES.index("avg_home_prob")
        avg_preds = np.argmax(X_val[:, avg_home_idx:avg_home_idx+3], axis=1)

        # Accuracy when models agree vs disagree
        if agreed.sum() > 0:
            agree_acc = (avg_preds[agreed] == y_val[agreed]).mean()
            agree_draw_rate = (y_val[agreed] == 1).mean()
        else:
            agree_acc = 0
            agree_draw_rate = 0

        if disagreed.sum() > 0:
            disagree_acc = (avg_preds[disagreed] == y_val[disagreed]).mean()
            disagree_draw_rate = (y_val[disagreed] == 1).mean()
        else:
            disagree_acc = 0
            disagree_draw_rate = 0

        return {
            "total_matches": len(y_val),
            "matches_agreed": int(agreed.sum()),
            "matches_disagreed": int(disagreed.sum()),
            "accuracy_when_agreed": agree_acc,
            "accuracy_when_disagreed": disagree_acc,
            "draw_rate_when_agreed": agree_draw_rate,
            "draw_rate_when_disagreed": disagree_draw_rate,
            "agreement_rate": agreed.mean(),
        }


def main():
    """Train and evaluate the consensus stacker."""

    stacker = ConsensusStacker()

    # First, analyze agreement patterns
    print("=" * 60)
    print("AGREEMENT ANALYSIS")
    print("=" * 60)

    agreement = stacker.analyze_agreement_accuracy()
    print(f"Total validation matches: {agreement['total_matches']}")
    print(f"Models agreed: {agreement['matches_agreed']} ({agreement['agreement_rate']:.1%})")
    print(f"Models disagreed: {agreement['matches_disagreed']}")
    print()
    print(f"Accuracy when ELO + Poisson + Market AGREE: {agreement['accuracy_when_agreed']:.1%}")
    print(f"Accuracy when models DISAGREE:             {agreement['accuracy_when_disagreed']:.1%}")
    print(f"Improvement from agreement:                {agreement['accuracy_when_agreed'] - agreement['accuracy_when_disagreed']:.1%}")
    print()
    print(f"Draw rate when models AGREE:    {agreement['draw_rate_when_agreed']:.1%}")
    print(f"Draw rate when models DISAGREE: {agreement['draw_rate_when_disagreed']:.1%}")
    print(f"Draw rate increase:             +{(agreement['draw_rate_when_disagreed'] - agreement['draw_rate_when_agreed']):.1%}")

    # Train the model
    print()
    print("=" * 60)
    print("TRAINING CONSENSUS STACKER")
    print("=" * 60)

    result = stacker.train(epochs=100, batch_size=64)

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Best validation accuracy: {result['val_accuracy']:.1%}")
    print(f"At epoch: {result['best_epoch']}")
    print()
    print("Comparison:")
    print("  ELO alone:            51.8%")
    print("  Neural stacker v9:    53.1%")
    print(f"  Consensus stacker:    {result['val_accuracy']:.1%}")


if __name__ == "__main__":
    main()
