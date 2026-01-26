"""Monthly strategy optimization job.

Runs on the 1st of each month at 3AM UTC to:
- Re-optimize strategy parameters using Bayesian optimization
- Apply new parameters if improvement >= 2%
- Record optimization runs for analysis
"""

import argparse
from datetime import datetime
from typing import Optional

import structlog
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.database import SyncSessionLocal
from app.db.models import BettingStrategy, StrategyStatus
from batch.monitoring.parameter_optimizer import ParameterOptimizer
from batch.monitoring.alerting import AlertManager, AlertType

logger = structlog.get_logger()


class MonthlyOptimizationJob:
    """Monthly strategy parameter optimization job."""

    DEFAULT_N_TRIALS = 100
    DEFAULT_LOOKBACK_YEARS = 2.0

    def __init__(self, session: Optional[Session] = None):
        self.session = session or SyncSessionLocal()
        self.optimizer = ParameterOptimizer(self.session)
        self.alert_manager = AlertManager()

    def run(
        self,
        n_trials: int = DEFAULT_N_TRIALS,
        lookback_years: float = DEFAULT_LOOKBACK_YEARS,
        strategy_name: Optional[str] = None,
    ) -> dict:
        """Execute the monthly optimization job.

        Args:
            n_trials: Number of Optuna trials per strategy
            lookback_years: Years of historical data to use
            strategy_name: Optional specific strategy to optimize

        Returns:
            Summary of job results
        """
        logger.info(
            "Starting monthly strategy optimization",
            n_trials=n_trials,
            lookback_years=lookback_years,
        )
        start_time = datetime.utcnow()

        # Get strategies to optimize
        stmt = select(BettingStrategy).where(
            BettingStrategy.status.in_([StrategyStatus.ACTIVE, StrategyStatus.PAUSED])
        )
        if strategy_name:
            stmt = stmt.where(BettingStrategy.name == strategy_name)

        strategies = list(self.session.execute(stmt).scalars().all())

        if not strategies:
            logger.warning("No strategies found to optimize")
            return {"status": "skipped", "reason": "No strategies found"}

        logger.info(f"Optimizing {len(strategies)} strategies")

        optimized = 0
        parameters_updated = 0
        failed = 0

        for strategy in strategies:
            try:
                result = self.optimizer.optimize_strategy(
                    strategy,
                    n_trials=n_trials,
                    lookback_years=lookback_years,
                )

                if result is None:
                    failed += 1
                    continue

                optimized += 1

                # Apply if improvement is significant
                if result.should_apply:
                    self.optimizer.apply_optimization(strategy, result)
                    parameters_updated += 1

                    self.alert_manager.send_alert(
                        AlertType.PARAMETERS_UPDATED,
                        strategy_name=strategy.name,
                        version=strategy.optimization_version,
                        improvement=result.improvement,
                        roi_before=result.backtest_roi_before,
                        roi_after=result.backtest_roi_after,
                    )

                # Send optimization complete alert
                self.alert_manager.send_alert(
                    AlertType.OPTIMIZATION_COMPLETE,
                    strategy_name=strategy.name,
                    improvement=result.improvement,
                    was_applied=result.should_apply,
                    reason=result.not_applied_reason,
                )

            except Exception as e:
                logger.error(
                    f"Failed to optimize strategy {strategy.name}",
                    error=str(e),
                )
                failed += 1

        self.session.commit()

        duration = (datetime.utcnow() - start_time).total_seconds()
        result = {
            "status": "success",
            "strategies_optimized": optimized,
            "parameters_updated": parameters_updated,
            "failed": failed,
            "n_trials": n_trials,
            "lookback_years": lookback_years,
            "duration_seconds": round(duration, 1),
        }

        logger.info("Monthly optimization completed", **result)
        return result


def run_monthly_optimization(
    n_trials: int = 100,
    lookback_years: float = 2.0,
    strategy_name: Optional[str] = None,
) -> dict:
    """Entry point for monthly optimization job."""
    with SyncSessionLocal() as session:
        job = MonthlyOptimizationJob(session)
        return job.run(
            n_trials=n_trials,
            lookback_years=lookback_years,
            strategy_name=strategy_name,
        )


def main():
    parser = argparse.ArgumentParser(description="Run monthly strategy optimization")
    parser.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Number of Optuna trials per strategy (default: 100)",
    )
    parser.add_argument(
        "--lookback-years",
        type=float,
        default=2.0,
        help="Years of historical data to use (default: 2.0)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        help="Specific strategy name to optimize (default: all)",
    )
    args = parser.parse_args()

    import logging
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("optuna").setLevel(logging.WARNING)

    result = run_monthly_optimization(
        n_trials=args.n_trials,
        lookback_years=args.lookback_years,
        strategy_name=args.strategy,
    )
    print(f"Optimization complete: {result}")


if __name__ == "__main__":
    main()
