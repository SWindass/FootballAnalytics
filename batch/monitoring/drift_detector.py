"""Drift detection for betting strategy performance.

Implements statistical methods to detect when a strategy's performance
has significantly degraded from its historical baseline.
"""

from dataclasses import dataclass
from decimal import Decimal

import numpy as np
import structlog

logger = structlog.get_logger()


@dataclass
class DriftResult:
    """Result of drift detection analysis."""

    z_score: float
    cusum_statistic: float
    is_drift_detected: bool
    drift_type: str | None = None  # "sudden" or "gradual"
    confidence: float = 0.0


class DriftDetector:
    """Detects performance drift in betting strategies.

    Uses two complementary methods:
    1. Z-score: Detects sudden, large changes in performance
    2. CUSUM: Detects gradual, sustained shifts in performance
    """

    # Detection thresholds (can be tuned)
    Z_SCORE_THRESHOLD = 2.0  # ~2.5% false positive rate one-sided
    CUSUM_THRESHOLD = 4.0  # Accumulated deviation threshold

    def __init__(
        self,
        z_score_threshold: float = Z_SCORE_THRESHOLD,
        cusum_threshold: float = CUSUM_THRESHOLD,
    ):
        """Initialize drift detector.

        Args:
            z_score_threshold: Threshold for z-score detection (default 2.0)
            cusum_threshold: Threshold for CUSUM detection (default 4.0)
        """
        self.z_score_threshold = z_score_threshold
        self.cusum_threshold = cusum_threshold

    def detect_drift(
        self,
        recent_roi: float,
        historical_mean_roi: float,
        historical_std_roi: float,
        recent_results: list[float],
        target_roi: float = 0.0,
    ) -> DriftResult:
        """Detect performance drift using z-score and CUSUM.

        Args:
            recent_roi: ROI of recent period (e.g., last 30 or 50 bets)
            historical_mean_roi: Long-term average ROI
            historical_std_roi: Standard deviation of historical ROI
            recent_results: List of individual bet P/L values (normalized, e.g., -1 for loss, +odds-1 for win)
            target_roi: Target ROI to detect drift from (default 0 = break-even)

        Returns:
            DriftResult with detection statistics and verdict
        """
        # Calculate z-score
        z_score = self._calculate_z_score(recent_roi, historical_mean_roi, historical_std_roi)

        # Calculate CUSUM statistic
        cusum = self._calculate_cusum(recent_results, target_roi)

        # Determine if drift is detected
        is_z_score_drift = z_score < -self.z_score_threshold  # Negative z = underperformance
        is_cusum_drift = cusum > self.cusum_threshold

        is_drift = is_z_score_drift or is_cusum_drift

        # Determine drift type
        drift_type = None
        if is_drift:
            if is_z_score_drift and not is_cusum_drift:
                drift_type = "sudden"
            elif is_cusum_drift and not is_z_score_drift:
                drift_type = "gradual"
            else:
                drift_type = "both"

        # Calculate confidence (how far beyond threshold)
        z_confidence = max(0, (-z_score - self.z_score_threshold) / self.z_score_threshold)
        cusum_confidence = max(0, (cusum - self.cusum_threshold) / self.cusum_threshold)
        confidence = max(z_confidence, cusum_confidence) if is_drift else 0.0

        return DriftResult(
            z_score=z_score,
            cusum_statistic=cusum,
            is_drift_detected=is_drift,
            drift_type=drift_type,
            confidence=min(1.0, confidence),
        )

    def _calculate_z_score(
        self,
        recent_roi: float,
        historical_mean: float,
        historical_std: float,
    ) -> float:
        """Calculate z-score for recent performance.

        Z-score measures how many standard deviations the recent ROI
        is from the historical mean. A large negative z-score indicates
        significant underperformance.

        Args:
            recent_roi: ROI from recent period
            historical_mean: Long-term average ROI
            historical_std: Standard deviation of historical ROI

        Returns:
            Z-score (negative = underperformance)
        """
        if historical_std == 0 or historical_std is None:
            # Can't calculate meaningful z-score without variance
            return 0.0

        z = (recent_roi - historical_mean) / historical_std
        return z

    def _calculate_cusum(
        self,
        results: list[float],
        target: float = 0.0,
    ) -> float:
        """Calculate CUSUM statistic for gradual drift detection.

        CUSUM (Cumulative Sum) detects sustained small shifts that might
        not trigger a z-score alert but indicate degrading performance.

        Uses the lower CUSUM (detecting negative shifts) since we care
        about underperformance.

        Args:
            results: List of individual bet P/L values (normalized)
            target: Target value to detect drift from

        Returns:
            CUSUM statistic (higher = more evidence of negative drift)
        """
        if not results:
            return 0.0

        # Calculate deviations from target
        deviations = np.array(results) - target

        # Calculate lower CUSUM (detects negative shifts)
        # S_L = max(0, S_L_prev + (target - x_i))
        cusum_lower = 0.0
        max_cusum = 0.0

        for dev in deviations:
            cusum_lower = max(0, cusum_lower - dev)  # Accumulate negative deviations
            max_cusum = max(max_cusum, cusum_lower)

        return max_cusum

    def analyze_rolling_window(
        self,
        all_results: list[tuple[Decimal, Decimal]],  # List of (profit_loss, odds)
        window_size: int = 50,
        baseline_size: int = 100,
    ) -> DriftResult:
        """Analyze drift using a rolling window approach.

        Args:
            all_results: List of (profit_loss, odds) tuples ordered by time
            window_size: Number of recent bets to analyze
            baseline_size: Number of historical bets for baseline

        Returns:
            DriftResult for the rolling window
        """
        if len(all_results) < window_size:
            return DriftResult(
                z_score=0.0,
                cusum_statistic=0.0,
                is_drift_detected=False,
            )

        # Split into baseline and recent
        recent = all_results[-window_size:]
        baseline = all_results[:-window_size]

        if len(baseline) < baseline_size // 2:
            # Not enough baseline data
            return DriftResult(
                z_score=0.0,
                cusum_statistic=0.0,
                is_drift_detected=False,
            )

        # Calculate ROIs
        def calculate_roi(results: list[tuple[Decimal, Decimal]]) -> float:
            """Calculate ROI from (profit_loss, odds) tuples."""
            total_profit = sum(float(pl) for pl, _ in results)
            n_bets = len(results)
            return total_profit / n_bets if n_bets > 0 else 0.0

        recent_roi = calculate_roi(recent)

        # Calculate baseline statistics using sliding windows
        baseline_rois = []
        for i in range(0, len(baseline) - window_size + 1, window_size // 2):
            window = baseline[i : i + window_size]
            if len(window) == window_size:
                baseline_rois.append(calculate_roi(window))

        if len(baseline_rois) < 2:
            historical_mean = calculate_roi(baseline)
            historical_std = 0.1  # Default std when not enough windows
        else:
            historical_mean = np.mean(baseline_rois)
            historical_std = np.std(baseline_rois)

        # Normalize recent results for CUSUM
        recent_normalized = []
        for pl, _odds in recent:
            # Normalize to standard bet size
            # Win: profit / stake = odds - 1
            # Loss: -1 (lost stake)
            stake = 1.0
            recent_normalized.append(float(pl) / stake)

        return self.detect_drift(
            recent_roi=recent_roi,
            historical_mean_roi=historical_mean,
            historical_std_roi=historical_std,
            recent_results=recent_normalized,
            target_roi=0.0,  # Target is break-even
        )
