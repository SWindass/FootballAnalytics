"""Strategy monitoring timer triggers.

Three triggers for adaptive strategy management:
- Weekly monitoring: Monday 9AM UTC
- Monthly optimization: 1st of month 3AM UTC
- Quarterly validation: Jan/Apr/Jul/Oct 1st 4AM UTC
"""

import logging
import sys
from pathlib import Path

import azure.functions as func

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from batch.jobs.strategy_monitoring_weekly import run_weekly_monitoring
from batch.jobs.strategy_optimization_monthly import run_monthly_optimization
from batch.jobs.strategy_validation_quarterly import run_quarterly_validation

app = func.FunctionApp()


@app.timer_trigger(
    schedule="0 0 9 * * 1",  # Every Monday at 9AM UTC
    arg_name="timer",
    run_on_startup=False,
)
def strategy_monitoring_weekly_trigger(timer: func.TimerRequest) -> None:
    """Execute weekly strategy monitoring job."""
    logging.info("Weekly strategy monitoring trigger started")

    if timer.past_due:
        logging.warning("Timer is past due!")

    try:
        result = run_weekly_monitoring()
        logging.info(f"Weekly monitoring completed: {result}")
    except Exception as e:
        logging.error(f"Weekly monitoring failed: {e}")
        raise


@app.timer_trigger(
    schedule="0 0 3 1 * *",  # 1st of every month at 3AM UTC
    arg_name="timer",
    run_on_startup=False,
)
def strategy_optimization_monthly_trigger(timer: func.TimerRequest) -> None:
    """Execute monthly strategy optimization job."""
    logging.info("Monthly strategy optimization trigger started")

    if timer.past_due:
        logging.warning("Timer is past due!")

    try:
        result = run_monthly_optimization(n_trials=100, lookback_years=2.0)
        logging.info(f"Monthly optimization completed: {result}")
    except Exception as e:
        logging.error(f"Monthly optimization failed: {e}")
        raise


@app.timer_trigger(
    schedule="0 0 4 1 1,4,7,10 *",  # Jan/Apr/Jul/Oct 1st at 4AM UTC
    arg_name="timer",
    run_on_startup=False,
)
def strategy_validation_quarterly_trigger(timer: func.TimerRequest) -> None:
    """Execute quarterly strategy validation job."""
    logging.info("Quarterly strategy validation trigger started")

    if timer.past_due:
        logging.warning("Timer is past due!")

    try:
        result = run_quarterly_validation()
        logging.info(f"Quarterly validation completed: {result}")
    except Exception as e:
        logging.error(f"Quarterly validation failed: {e}")
        raise
