"""Quarterly strategy validation job.

Runs on Jan/Apr/Jul/Oct 1st at 4AM UTC to:
- Perform comprehensive backtest validation
- Compare current performance to historical baseline
- Recommend strategy adjustments or retirement
"""

import argparse
from datetime import datetime, timedelta
from decimal import Decimal

import structlog
from sqlalchemy import and_, select
from sqlalchemy.orm import Session

from app.db.database import SyncSessionLocal
from app.db.models import (
    BettingStrategy,
    Match,
    MatchStatus,
    SnapshotType,
    StrategyMonitoringSnapshot,
    StrategyStatus,
    ValueBet,
)
from batch.monitoring.alerting import AlertManager, AlertType

logger = structlog.get_logger()


class QuarterlyValidationJob:
    """Quarterly comprehensive strategy validation."""

    # Validation thresholds
    MIN_ROI_THRESHOLD = -0.05  # Fail if ROI < -5%
    MIN_WIN_RATE_THRESHOLD = 0.40  # Warn if win rate < 40%
    MIN_BETS_FOR_VALIDATION = 30  # Need at least 30 bets in the quarter

    def __init__(self, session: Session | None = None):
        self.session = session or SyncSessionLocal()
        self.alert_manager = AlertManager()

    def run(self) -> dict:
        """Execute the quarterly validation job.

        Returns:
            Summary of job results
        """
        logger.info("Starting quarterly strategy validation")
        start_time = datetime.utcnow()

        # Get all strategies
        stmt = select(BettingStrategy)
        strategies = list(self.session.execute(stmt).scalars().all())

        logger.info(f"Validating {len(strategies)} strategies")

        validated = 0
        passed = 0
        warned = 0
        failed = 0

        for strategy in strategies:
            try:
                result = self._validate_strategy(strategy)
                validated += 1

                if result["status"] == "passed":
                    passed += 1
                elif result["status"] == "warned":
                    warned += 1
                else:
                    failed += 1

                # Create quarterly snapshot
                self._create_quarterly_snapshot(strategy, result)

            except Exception as e:
                logger.error(
                    f"Failed to validate strategy {strategy.name}",
                    error=str(e),
                )

        self.session.commit()

        duration = (datetime.utcnow() - start_time).total_seconds()
        result = {
            "status": "success",
            "strategies_validated": validated,
            "passed": passed,
            "warned": warned,
            "failed": failed,
            "duration_seconds": round(duration, 1),
        }

        logger.info("Quarterly validation completed", **result)
        return result

    def _validate_strategy(self, strategy: BettingStrategy) -> dict:
        """Validate a single strategy.

        Args:
            strategy: The strategy to validate

        Returns:
            Validation result dict
        """
        # Get quarterly performance
        quarter_start = datetime.utcnow() - timedelta(days=90)

        stmt = (
            select(ValueBet)
            .join(Match, ValueBet.match_id == Match.id)
            .where(
                and_(
                    ValueBet.strategy_id == strategy.id,
                    ValueBet.result.in_(["won", "lost"]),
                    Match.kickoff_time >= quarter_start,
                    Match.status == MatchStatus.FINISHED,
                )
            )
            .order_by(Match.kickoff_time)
        )
        bets = list(self.session.execute(stmt).scalars().all())

        n_bets = len(bets)
        if n_bets < self.MIN_BETS_FOR_VALIDATION:
            logger.info(
                f"Insufficient bets for validation: {n_bets}",
                strategy=strategy.name,
            )
            return {
                "status": "skipped",
                "reason": f"Only {n_bets} bets (need {self.MIN_BETS_FOR_VALIDATION})",
                "n_bets": n_bets,
            }

        # Calculate metrics
        n_wins = sum(1 for b in bets if b.result == "won")
        total_profit = sum(float(b.profit_loss or 0) for b in bets)
        roi = total_profit / n_bets
        win_rate = n_wins / n_bets

        logger.info(
            f"Quarterly metrics for {strategy.name}",
            n_bets=n_bets,
            win_rate=win_rate,
            roi=roi,
        )

        # Determine validation status
        status = "passed"
        issues = []

        if roi < self.MIN_ROI_THRESHOLD:
            status = "failed"
            issues.append(f"ROI of {roi:.2%} is below threshold of {self.MIN_ROI_THRESHOLD:.2%}")

        if win_rate < self.MIN_WIN_RATE_THRESHOLD and status != "failed":
            status = "warned"
            issues.append(f"Win rate of {win_rate:.2%} is below threshold of {self.MIN_WIN_RATE_THRESHOLD:.2%}")

        # Send alerts if needed
        if status == "failed":
            self.alert_manager.send_alert(
                AlertType.VALIDATION_FAILED,
                strategy_name=strategy.name,
                reason="; ".join(issues),
                roi=roi,
                win_rate=win_rate,
                n_bets=n_bets,
            )

            # Pause the strategy
            if strategy.status == StrategyStatus.ACTIVE:
                strategy.status = StrategyStatus.PAUSED
                strategy.status_reason = f"Quarterly validation failed: {'; '.join(issues)}"

        # Update last backtest date
        strategy.last_backtest_at = datetime.utcnow()

        return {
            "status": status,
            "n_bets": n_bets,
            "n_wins": n_wins,
            "win_rate": win_rate,
            "roi": roi,
            "issues": issues,
        }

    def _create_quarterly_snapshot(
        self, strategy: BettingStrategy, validation_result: dict
    ) -> None:
        """Create a quarterly snapshot with validation results."""
        if validation_result["status"] == "skipped":
            return

        snapshot = StrategyMonitoringSnapshot(
            strategy_id=strategy.id,
            snapshot_date=datetime.utcnow(),
            snapshot_type=SnapshotType.QUARTERLY,
            cumulative_bets=validation_result["n_bets"],
            cumulative_roi=Decimal(str(round(validation_result["roi"], 4))),
            rolling_30_bets=validation_result["n_bets"],
            rolling_30_wins=validation_result["n_wins"],
            rolling_30_roi=Decimal(str(round(validation_result["roi"], 4))),
            rolling_50_bets=validation_result["n_bets"],
            rolling_50_wins=validation_result["n_wins"],
            rolling_50_roi=Decimal(str(round(validation_result["roi"], 4))),
            z_score=Decimal("0"),
            cusum_statistic=Decimal("0"),
            is_drift_detected=validation_result["status"] == "failed",
            alert_triggered=validation_result["status"] in ["failed", "warned"],
            alert_type=f"quarterly_{validation_result['status']}",
        )
        self.session.add(snapshot)


def run_quarterly_validation() -> dict:
    """Entry point for quarterly validation job."""
    with SyncSessionLocal() as session:
        job = QuarterlyValidationJob(session)
        return job.run()


def main():
    parser = argparse.ArgumentParser(description="Run quarterly strategy validation")
    parser.parse_args()

    import logging
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

    result = run_quarterly_validation()
    print(f"Validation complete: {result}")


if __name__ == "__main__":
    main()
