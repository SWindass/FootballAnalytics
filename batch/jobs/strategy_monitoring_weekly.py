"""Weekly strategy monitoring job.

Runs every Monday at 9AM UTC to:
- Create performance snapshots for each active strategy
- Check for drift in recent performance
- Auto-disable strategies with sustained negative ROI
- Send alerts for concerning patterns
"""

import argparse
from datetime import datetime
from typing import Optional

import structlog
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.database import SyncSessionLocal
from app.db.models import BettingStrategy, StrategyStatus
from batch.monitoring.strategy_monitor import StrategyMonitor

logger = structlog.get_logger()


class WeeklyMonitoringJob:
    """Weekly strategy monitoring job."""

    def __init__(self, session: Optional[Session] = None):
        self.session = session or SyncSessionLocal()
        self.monitor = StrategyMonitor(self.session)

    def run(self) -> dict:
        """Execute the weekly monitoring job.

        Returns:
            Summary of job results
        """
        logger.info("Starting weekly strategy monitoring")
        start_time = datetime.utcnow()

        # Get all strategies (including paused ones - they still need monitoring)
        stmt = select(BettingStrategy).where(
            BettingStrategy.status.in_([StrategyStatus.ACTIVE, StrategyStatus.PAUSED])
        )
        strategies = list(self.session.execute(stmt).scalars().all())

        logger.info(f"Monitoring {len(strategies)} strategies")

        snapshots_created = 0
        strategies_disabled = 0
        alerts_triggered = 0

        for strategy in strategies:
            try:
                # Create weekly snapshot
                snapshot = self.monitor.create_weekly_snapshot(strategy)
                snapshots_created += 1

                if snapshot.alert_triggered:
                    alerts_triggered += 1

                # Update strategy stats
                self.monitor.update_strategy_stats(strategy)

                # Check for auto-disable (only for active strategies)
                if strategy.status == StrategyStatus.ACTIVE:
                    if self.monitor.check_auto_disable(strategy):
                        strategies_disabled += 1

            except Exception as e:
                logger.error(
                    f"Failed to monitor strategy {strategy.name}",
                    error=str(e),
                )

        self.session.commit()

        duration = (datetime.utcnow() - start_time).total_seconds()
        result = {
            "status": "success",
            "strategies_monitored": len(strategies),
            "snapshots_created": snapshots_created,
            "strategies_disabled": strategies_disabled,
            "alerts_triggered": alerts_triggered,
            "duration_seconds": round(duration, 1),
        }

        logger.info("Weekly monitoring completed", **result)
        return result


def run_weekly_monitoring() -> dict:
    """Entry point for weekly monitoring job."""
    with SyncSessionLocal() as session:
        job = WeeklyMonitoringJob(session)
        return job.run()


def main():
    parser = argparse.ArgumentParser(description="Run weekly strategy monitoring")
    parser.parse_args()

    import logging
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

    result = run_weekly_monitoring()
    print(f"Monitoring complete: {result}")


if __name__ == "__main__":
    main()
