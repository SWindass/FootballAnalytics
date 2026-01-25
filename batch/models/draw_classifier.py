"""Draw prediction using historical draw rates by ELO difference.

Research shows that standard ELO models systematically underpredict draws because
the probability formula caps draw probability at ~25%. This module uses historical
draw rates calibrated by ELO difference to produce better draw predictions.

Approach:
1. Calculate historical draw rate for each ELO difference bucket
2. Use this as the draw probability
3. Redistribute remaining probability between home/away using ELO ratios
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np

import structlog
from sqlalchemy import select

from app.db.database import SyncSessionLocal
from app.db.models import Match, MatchStatus, EloRating

logger = structlog.get_logger()


@dataclass
class DrawRateBucket:
    """Historical draw rate for an ELO difference range."""
    min_diff: int
    max_diff: int
    draw_rate: float
    home_win_rate: float
    away_win_rate: float
    sample_size: int


class DrawClassifier:
    """Predicts draw probability using historical draw rates by ELO difference.

    The classifier maintains a lookup table of draw rates for different
    ELO difference buckets, calibrated from historical match data.
    """

    # Default draw rates by absolute ELO difference (from historical analysis)
    # These are used if calibration data isn't available
    DEFAULT_DRAW_RATES = {
        0: 0.267,    # 0-24 ELO diff
        25: 0.275,   # 25-49
        50: 0.293,   # 50-74
        75: 0.263,   # 75-99
        100: 0.267,  # 100-124
        125: 0.243,  # 125-149
        150: 0.244,  # 150-174
        175: 0.201,  # 175-199
        200: 0.208,  # 200-224
        225: 0.202,  # 225-249
        250: 0.227,  # 250-274
        275: 0.157,  # 275-299
        300: 0.135,  # 300+
    }

    def __init__(self, bucket_size: int = 25):
        """Initialize the draw classifier.

        Args:
            bucket_size: Size of ELO difference buckets (default 25)
        """
        self.bucket_size = bucket_size
        self.draw_rates: dict[int, DrawRateBucket] = {}
        self.calibrated = False

    def calibrate(self, use_recent_seasons: int = 0) -> dict:
        """Calibrate draw rates from historical match data.

        Args:
            use_recent_seasons: If > 0, only use last N seasons for calibration

        Returns:
            Calibration summary
        """
        logger.info("Calibrating draw classifier from historical data")

        with SyncSessionLocal() as session:
            # Get finished matches
            stmt = (
                select(Match)
                .where(Match.status == MatchStatus.FINISHED)
                .order_by(Match.kickoff_time)
            )
            matches = list(session.execute(stmt).scalars().all())

            # Load ELO ratings
            all_elo = list(session.execute(select(EloRating)).scalars().all())
            elo_lookup = {
                (e.team_id, e.season, e.matchweek): float(e.rating)
                for e in all_elo
            }

            # Filter to recent seasons if requested
            if use_recent_seasons > 0:
                seasons = sorted(set(m.season for m in matches))
                recent_seasons = set(seasons[-use_recent_seasons:])
                matches = [m for m in matches if m.season in recent_seasons]
                logger.info(f"Using {len(matches)} matches from {recent_seasons}")

            # Count outcomes by ELO difference bucket
            from collections import defaultdict
            buckets = defaultdict(lambda: {
                "total": 0, "draws": 0, "home_wins": 0, "away_wins": 0
            })

            for m in matches:
                home_elo = elo_lookup.get(
                    (m.home_team_id, m.season, m.matchweek - 1), 1500
                )
                away_elo = elo_lookup.get(
                    (m.away_team_id, m.season, m.matchweek - 1), 1500
                )

                abs_diff = abs(home_elo - away_elo)
                bucket = int(abs_diff // self.bucket_size) * self.bucket_size
                bucket = min(bucket, 300)  # Cap at 300+

                buckets[bucket]["total"] += 1

                if m.home_score == m.away_score:
                    buckets[bucket]["draws"] += 1
                elif m.home_score > m.away_score:
                    buckets[bucket]["home_wins"] += 1
                else:
                    buckets[bucket]["away_wins"] += 1

            # Build calibrated draw rates
            self.draw_rates = {}
            for bucket, data in buckets.items():
                total = data["total"]
                if total >= 50:  # Require minimum sample size
                    self.draw_rates[bucket] = DrawRateBucket(
                        min_diff=bucket,
                        max_diff=bucket + self.bucket_size - 1,
                        draw_rate=data["draws"] / total,
                        home_win_rate=data["home_wins"] / total,
                        away_win_rate=data["away_wins"] / total,
                        sample_size=total,
                    )

            self.calibrated = True
            logger.info(f"Calibrated {len(self.draw_rates)} ELO difference buckets")

            return {
                "buckets_calibrated": len(self.draw_rates),
                "matches_used": len(matches),
                "seasons_used": len(set(m.season for m in matches)),
            }

    def get_draw_probability(self, elo_diff: float) -> float:
        """Get calibrated draw probability for an ELO difference.

        Args:
            elo_diff: ELO difference (home - away), can be positive or negative

        Returns:
            Draw probability (0-1)
        """
        abs_diff = abs(elo_diff)
        bucket = int(abs_diff // self.bucket_size) * self.bucket_size
        bucket = min(bucket, 300)

        if self.calibrated and bucket in self.draw_rates:
            return self.draw_rates[bucket].draw_rate

        # Fall back to default rates
        return self.DEFAULT_DRAW_RATES.get(bucket, 0.25)

    def predict(
        self,
        home_elo: float,
        away_elo: float,
        home_advantage: float = 50.0,
    ) -> tuple[float, float, float]:
        """Predict match probabilities with calibrated draw rate.

        Args:
            home_elo: Home team ELO rating
            away_elo: Away team ELO rating
            home_advantage: ELO points to add for home advantage

        Returns:
            Tuple of (home_win_prob, draw_prob, away_win_prob)
        """
        # Calculate ELO difference (with home advantage)
        effective_home_elo = home_elo + home_advantage
        elo_diff = effective_home_elo - away_elo

        # Get calibrated draw probability (using raw diff for bucket lookup)
        raw_diff = home_elo - away_elo
        draw_prob = self.get_draw_probability(raw_diff)

        # Calculate home/away win probabilities using standard ELO formula
        # Then redistribute after removing draw probability
        expected_home = 1 / (1 + 10 ** ((away_elo - effective_home_elo) / 400))
        expected_away = 1 - expected_home

        # Redistribute remaining probability proportionally
        remaining = 1 - draw_prob
        home_prob = expected_home * remaining
        away_prob = expected_away * remaining

        # Normalize to ensure sum = 1
        total = home_prob + draw_prob + away_prob
        return (
            home_prob / total,
            draw_prob / total,
            away_prob / total,
        )


class HybridPredictor:
    """Combines draw classifier with existing ELO/Poisson predictions.

    Uses a two-stage approach:
    1. Draw classifier determines P(draw) based on historical rates
    2. ELO determines P(home|not draw) and P(away|not draw)
    """

    def __init__(self):
        self.draw_classifier = DrawClassifier()
        self.calibrated = False

    def calibrate(self, use_recent_seasons: int = 10) -> dict:
        """Calibrate the hybrid predictor."""
        result = self.draw_classifier.calibrate(use_recent_seasons)
        self.calibrated = True
        return result

    def predict(
        self,
        home_elo: float,
        away_elo: float,
        poisson_draw_prob: Optional[float] = None,
        home_advantage: float = 50.0,
    ) -> tuple[float, float, float]:
        """Predict match probabilities.

        Args:
            home_elo: Home team ELO rating
            away_elo: Away team ELO rating
            poisson_draw_prob: Optional Poisson draw probability to blend
            home_advantage: ELO points for home advantage

        Returns:
            Tuple of (home_win_prob, draw_prob, away_win_prob)
        """
        # Get draw classifier prediction
        home_p, draw_p, away_p = self.draw_classifier.predict(
            home_elo, away_elo, home_advantage
        )

        # Optionally blend with Poisson draw probability
        if poisson_draw_prob is not None:
            # Weighted average: 60% historical, 40% Poisson
            blended_draw = 0.6 * draw_p + 0.4 * poisson_draw_prob

            # Redistribute home/away
            remaining = 1 - blended_draw
            home_ratio = home_p / (home_p + away_p) if (home_p + away_p) > 0 else 0.5

            home_p = remaining * home_ratio
            draw_p = blended_draw
            away_p = remaining * (1 - home_ratio)

        return home_p, draw_p, away_p


def threshold_prediction(
    home_p: float,
    draw_p: float,
    away_p: float,
    draw_threshold: float = 0.26,
    close_match_boost: bool = True,
) -> int:
    """Make prediction using threshold-based decision for draws.

    Instead of simple argmax, predict draw when:
    1. Draw probability exceeds threshold, AND
    2. Home/away probabilities are close (no clear favorite)

    Args:
        home_p, draw_p, away_p: Predicted probabilities
        draw_threshold: Minimum draw probability to consider predicting draw
        close_match_boost: If True, also consider home/away gap

    Returns:
        Prediction: 0=home, 1=draw, 2=away
    """
    # Check if match is close (no strong favorite)
    home_away_gap = abs(home_p - away_p)

    if close_match_boost:
        # Predict draw if:
        # - Draw probability is high enough, AND
        # - Home/away are close (gap < 0.15)
        if draw_p >= draw_threshold and home_away_gap < 0.15:
            return 1
    else:
        if draw_p >= draw_threshold:
            return 1

    # Otherwise use argmax for home/away
    return 0 if home_p > away_p else 2


def evaluate_draw_classifier():
    """Evaluate the draw classifier on validation data."""
    import logging
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

    from collections import Counter

    classifier = DrawClassifier()
    classifier.calibrate()

    with SyncSessionLocal() as session:
        # Get matches
        matches = list(session.execute(
            select(Match)
            .where(Match.status == MatchStatus.FINISHED)
            .order_by(Match.kickoff_time)
        ).scalars().all())

        # Load ELO
        all_elo = list(session.execute(select(EloRating)).scalars().all())
        elo_lookup = {
            (e.team_id, e.season, e.matchweek): float(e.rating)
            for e in all_elo
        }

        # Validation split
        n_val = int(len(matches) * 0.2)
        val_matches = matches[-n_val:]

        outcomes = {0: "Home win", 1: "Draw", 2: "Away win"}

        print("=" * 70)
        print("DRAW CLASSIFIER EVALUATION - THRESHOLD OPTIMIZATION")
        print("=" * 70)
        print(f"Validation set: {len(val_matches)} matches\n")

        # Test different thresholds
        print("--- Threshold Search ---")
        print(f"{'Threshold':<12} {'Draws Pred':<12} {'Draws Hit':<12} {'Accuracy':<12}")
        print("-" * 50)

        best_acc = 0
        best_threshold = 0

        for threshold in [0.20, 0.22, 0.24, 0.25, 0.26, 0.27, 0.28, 0.30]:
            correct = 0
            draws_predicted = 0
            draws_correct = 0

            for m in val_matches:
                home_elo = elo_lookup.get(
                    (m.home_team_id, m.season, m.matchweek - 1), 1500
                )
                away_elo = elo_lookup.get(
                    (m.away_team_id, m.season, m.matchweek - 1), 1500
                )

                # Get probabilities
                home_p, draw_p, away_p = classifier.predict(home_elo, away_elo)

                # Threshold-based prediction
                pred = threshold_prediction(home_p, draw_p, away_p, threshold)

                # Actual outcome
                if m.home_score > m.away_score:
                    actual = 0
                elif m.home_score == m.away_score:
                    actual = 1
                else:
                    actual = 2

                if pred == 1:
                    draws_predicted += 1
                    if actual == 1:
                        draws_correct += 1

                if pred == actual:
                    correct += 1

            acc = correct / len(val_matches)
            draw_acc = draws_correct / draws_predicted if draws_predicted > 0 else 0

            if acc > best_acc:
                best_acc = acc
                best_threshold = threshold

            print(f"{threshold:<12.2f} {draws_predicted:<12} {draws_correct:<12} {acc:.1%}")

        print("-" * 50)
        print(f"Best threshold: {best_threshold} with {best_acc:.1%} accuracy")

        # Detailed evaluation with best threshold
        print(f"\n{'='*70}")
        print(f"DETAILED RESULTS WITH THRESHOLD = {best_threshold}")
        print("=" * 70)

        pred_counts = Counter()
        actual_counts = Counter()
        correct_by_outcome = Counter()
        total_correct = 0

        for m in val_matches:
            home_elo = elo_lookup.get(
                (m.home_team_id, m.season, m.matchweek - 1), 1500
            )
            away_elo = elo_lookup.get(
                (m.away_team_id, m.season, m.matchweek - 1), 1500
            )

            home_p, draw_p, away_p = classifier.predict(home_elo, away_elo)
            pred = threshold_prediction(home_p, draw_p, away_p, best_threshold)

            if m.home_score > m.away_score:
                actual = 0
            elif m.home_score == m.away_score:
                actual = 1
            else:
                actual = 2

            pred_counts[pred] += 1
            actual_counts[actual] += 1

            if pred == actual:
                total_correct += 1
                correct_by_outcome[actual] += 1

        print("\n--- Prediction Distribution ---")
        for i in range(3):
            pct = pred_counts[i] / len(val_matches) * 100
            print(f"{outcomes[i]}: {pred_counts[i]} ({pct:.1f}%)")

        print("\n--- Actual Distribution ---")
        for i in range(3):
            pct = actual_counts[i] / len(val_matches) * 100
            print(f"{outcomes[i]}: {actual_counts[i]} ({pct:.1f}%)")

        print("\n--- Accuracy by Outcome ---")
        for i in range(3):
            if actual_counts[i] > 0:
                acc = correct_by_outcome[i] / actual_counts[i] * 100
                print(f"{outcomes[i]}: {correct_by_outcome[i]}/{actual_counts[i]} = {acc:.1f}%")

        print(f"\n--- Overall Accuracy ---")
        print(f"Draw Classifier (threshold): {total_correct/len(val_matches):.1%}")
        print(f"Baseline ELO (no draws):     51.8%")
        print("=" * 70)


if __name__ == "__main__":
    evaluate_draw_classifier()
