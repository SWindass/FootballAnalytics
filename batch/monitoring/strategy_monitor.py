"""Strategy performance monitoring.

Core monitoring logic for:
- Rolling ROI calculation
- Auto-disable checks
- Creating weekly snapshots
"""

from datetime import datetime
from decimal import Decimal
from typing import Optional

import structlog
from sqlalchemy import select, func, and_
from sqlalchemy.orm import Session

from app.db.models import (
    BettingStrategy,
    StrategyMonitoringSnapshot,
    StrategyStatus,
    SnapshotType,
    ValueBet,
    Match,
    MatchStatus,
)
from batch.monitoring.drift_detector import DriftDetector, DriftResult
from batch.monitoring.alerting import AlertManager, AlertType

logger = structlog.get_logger()


class StrategyMonitor:
    """Monitors betting strategy performance and triggers alerts/actions."""

    # Thresholds for auto-disable
    ROLLING_WINDOW_SIZE = 50
    MIN_BETS_FOR_DISABLE = 50
    DISABLE_ROI_THRESHOLD = 0.0  # Disable if rolling ROI < 0

    def __init__(self, session: Session):
        self.session = session
        self.drift_detector = DriftDetector()
        self.alert_manager = AlertManager()

    def create_weekly_snapshot(
        self,
        strategy: BettingStrategy,
        snapshot_date: Optional[datetime] = None,
    ) -> StrategyMonitoringSnapshot:
        """Create a weekly performance snapshot for a strategy.

        Args:
            strategy: The strategy to snapshot
            snapshot_date: Date for the snapshot (default: now)

        Returns:
            The created snapshot
        """
        snapshot_date = snapshot_date or datetime.utcnow()

        # Get all settled bets for this strategy
        bets = self._get_settled_bets(strategy.id)

        if not bets:
            logger.info(f"No settled bets for strategy {strategy.name}")
            return self._create_empty_snapshot(strategy.id, snapshot_date)

        # Calculate rolling metrics
        rolling_30 = self._calculate_rolling_metrics(bets, window_size=30)
        rolling_50 = self._calculate_rolling_metrics(bets, window_size=50)
        cumulative = self._calculate_cumulative_metrics(bets)

        # Detect drift
        drift_result = self._detect_drift(bets, strategy)

        # Create snapshot
        snapshot = StrategyMonitoringSnapshot(
            strategy_id=strategy.id,
            snapshot_date=snapshot_date,
            snapshot_type=SnapshotType.WEEKLY,
            rolling_30_bets=rolling_30["n_bets"],
            rolling_30_wins=rolling_30["n_wins"],
            rolling_30_roi=Decimal(str(round(rolling_30["roi"], 4))) if rolling_30["roi"] else None,
            rolling_30_profit=Decimal(str(round(rolling_30["profit"], 2))) if rolling_30["profit"] else None,
            rolling_50_bets=rolling_50["n_bets"],
            rolling_50_wins=rolling_50["n_wins"],
            rolling_50_roi=Decimal(str(round(rolling_50["roi"], 4))) if rolling_50["roi"] else None,
            rolling_50_profit=Decimal(str(round(rolling_50["profit"], 2))) if rolling_50["profit"] else None,
            cumulative_bets=cumulative["n_bets"],
            cumulative_roi=Decimal(str(round(cumulative["roi"], 4))) if cumulative["roi"] else None,
            z_score=Decimal(str(round(drift_result.z_score, 4))),
            cusum_statistic=Decimal(str(round(drift_result.cusum_statistic, 4))),
            is_drift_detected=drift_result.is_drift_detected,
        )

        self.session.add(snapshot)

        # Check for alerts
        self._check_and_trigger_alerts(strategy, snapshot, drift_result)

        logger.info(
            f"Created weekly snapshot for {strategy.name}",
            rolling_50_roi=rolling_50["roi"],
            cumulative_bets=cumulative["n_bets"],
            drift_detected=drift_result.is_drift_detected,
        )

        return snapshot

    def check_auto_disable(self, strategy: BettingStrategy) -> bool:
        """Check if a strategy should be auto-disabled.

        A strategy is disabled when:
        - Rolling 50-bet ROI is negative
        - At least 50 bets have been made

        Args:
            strategy: The strategy to check

        Returns:
            True if strategy was disabled, False otherwise
        """
        if strategy.status != StrategyStatus.ACTIVE:
            return False

        bets = self._get_settled_bets(strategy.id)
        if len(bets) < self.MIN_BETS_FOR_DISABLE:
            return False

        rolling_50 = self._calculate_rolling_metrics(bets, window_size=50)
        if rolling_50["roi"] is not None and rolling_50["roi"] < self.DISABLE_ROI_THRESHOLD:
            # Disable the strategy
            strategy.status = StrategyStatus.DISABLED
            strategy.status_reason = (
                f"Auto-disabled: Rolling 50-bet ROI of {rolling_50['roi']:.2%} "
                f"is below threshold of {self.DISABLE_ROI_THRESHOLD:.2%}"
            )
            strategy.rolling_50_roi = Decimal(str(round(rolling_50["roi"], 4)))

            self.alert_manager.send_alert(
                AlertType.STRATEGY_DISABLED,
                strategy_name=strategy.name,
                reason=strategy.status_reason,
                rolling_roi=rolling_50["roi"],
            )

            logger.warning(
                f"Auto-disabled strategy {strategy.name}",
                rolling_roi=rolling_50["roi"],
            )
            return True

        return False

    def update_strategy_stats(self, strategy: BettingStrategy) -> None:
        """Update aggregate statistics on the strategy record.

        Args:
            strategy: The strategy to update
        """
        bets = self._get_settled_bets(strategy.id)

        if not bets:
            return

        # Calculate totals
        total_bets = len(bets)
        total_wins = sum(1 for b in bets if b.result == "won")
        total_profit = sum(float(b.profit_loss or 0) for b in bets)
        historical_roi = total_profit / total_bets if total_bets > 0 else 0.0

        # Calculate rolling 50
        rolling_50 = self._calculate_rolling_metrics(bets, window_size=50)

        # Calculate losing streak
        losing_streak = self._calculate_losing_streak(bets)

        # Update strategy
        strategy.total_bets = total_bets
        strategy.total_wins = total_wins
        strategy.total_profit = Decimal(str(round(total_profit, 2)))
        strategy.historical_roi = Decimal(str(round(historical_roi, 4)))
        strategy.rolling_50_roi = (
            Decimal(str(round(rolling_50["roi"], 4))) if rolling_50["roi"] else None
        )
        strategy.consecutive_losing_streak = losing_streak

    def _get_settled_bets(self, strategy_id: int) -> list[ValueBet]:
        """Get all settled bets for a strategy, ordered by match date."""
        stmt = (
            select(ValueBet)
            .join(Match, ValueBet.match_id == Match.id)
            .where(
                and_(
                    ValueBet.strategy_id == strategy_id,
                    ValueBet.result.in_(["won", "lost"]),
                    Match.status == MatchStatus.FINISHED,
                )
            )
            .order_by(Match.kickoff_time)
        )
        return list(self.session.execute(stmt).scalars().all())

    def _calculate_rolling_metrics(
        self, bets: list[ValueBet], window_size: int
    ) -> dict:
        """Calculate rolling window metrics."""
        if len(bets) < window_size:
            recent = bets
        else:
            recent = bets[-window_size:]

        n_bets = len(recent)
        if n_bets == 0:
            return {"n_bets": 0, "n_wins": 0, "roi": None, "profit": None}

        n_wins = sum(1 for b in recent if b.result == "won")
        profit = sum(float(b.profit_loss or 0) for b in recent)
        roi = profit / n_bets

        return {
            "n_bets": n_bets,
            "n_wins": n_wins,
            "roi": roi,
            "profit": profit,
        }

    def _calculate_cumulative_metrics(self, bets: list[ValueBet]) -> dict:
        """Calculate cumulative metrics for all bets."""
        n_bets = len(bets)
        if n_bets == 0:
            return {"n_bets": 0, "roi": None}

        profit = sum(float(b.profit_loss or 0) for b in bets)
        roi = profit / n_bets

        return {"n_bets": n_bets, "roi": roi}

    def _calculate_losing_streak(self, bets: list[ValueBet]) -> int:
        """Calculate current consecutive losing streak."""
        streak = 0
        for bet in reversed(bets):
            if bet.result == "lost":
                streak += 1
            else:
                break
        return streak

    def _detect_drift(
        self, bets: list[ValueBet], strategy: BettingStrategy
    ) -> DriftResult:
        """Run drift detection on the bet history."""
        if len(bets) < 50:
            return DriftResult(
                z_score=0.0,
                cusum_statistic=0.0,
                is_drift_detected=False,
            )

        # Convert to format expected by drift detector
        results = [(b.profit_loss, b.odds) for b in bets]
        return self.drift_detector.analyze_rolling_window(results)

    def _check_and_trigger_alerts(
        self,
        strategy: BettingStrategy,
        snapshot: StrategyMonitoringSnapshot,
        drift_result: DriftResult,
    ) -> None:
        """Check various alert conditions and trigger if needed."""
        alerts_triggered = []

        # Alert 1: Drift detected
        if drift_result.is_drift_detected:
            self.alert_manager.send_alert(
                AlertType.DRIFT_DETECTED,
                strategy_name=strategy.name,
                drift_type=drift_result.drift_type,
                z_score=drift_result.z_score,
                cusum=drift_result.cusum_statistic,
            )
            alerts_triggered.append("drift_detected")

        # Alert 2: Rolling ROI negative
        if snapshot.rolling_50_roi and float(snapshot.rolling_50_roi) < 0:
            self.alert_manager.send_alert(
                AlertType.NEGATIVE_ROI,
                strategy_name=strategy.name,
                rolling_roi=float(snapshot.rolling_50_roi),
                n_bets=snapshot.rolling_50_bets,
            )
            alerts_triggered.append("negative_roi")

        # Alert 3: Long losing streak (10+)
        if strategy.consecutive_losing_streak >= 10:
            self.alert_manager.send_alert(
                AlertType.LOSING_STREAK,
                strategy_name=strategy.name,
                streak_length=strategy.consecutive_losing_streak,
            )
            alerts_triggered.append("losing_streak")

        if alerts_triggered:
            snapshot.alert_triggered = True
            snapshot.alert_type = ",".join(alerts_triggered)

    def _create_empty_snapshot(
        self, strategy_id: int, snapshot_date: datetime
    ) -> StrategyMonitoringSnapshot:
        """Create an empty snapshot when no bets exist."""
        snapshot = StrategyMonitoringSnapshot(
            strategy_id=strategy_id,
            snapshot_date=snapshot_date,
            snapshot_type=SnapshotType.WEEKLY,
            rolling_30_bets=0,
            rolling_30_wins=0,
            rolling_50_bets=0,
            rolling_50_wins=0,
            cumulative_bets=0,
            z_score=Decimal("0"),
            cusum_statistic=Decimal("0"),
            is_drift_detected=False,
        )
        self.session.add(snapshot)
        return snapshot
