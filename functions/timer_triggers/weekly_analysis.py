"""Weekly analysis timer trigger - runs Tuesday 5PM UK time."""

import logging
import sys
from pathlib import Path

import azure.functions as func

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from batch.jobs.weekly_analysis import run_weekly_analysis

app = func.FunctionApp()


@app.timer_trigger(
    schedule="0 0 17 * * 2",  # Every Tuesday at 5PM UTC
    arg_name="timer",
    run_on_startup=False,
)
def weekly_analysis_trigger(timer: func.TimerRequest) -> None:
    """Execute weekly analysis job."""
    logging.info("Weekly analysis trigger started")

    if timer.past_due:
        logging.warning("Timer is past due!")

    try:
        result = run_weekly_analysis()
        logging.info(f"Weekly analysis completed: {result}")
    except Exception as e:
        logging.error(f"Weekly analysis failed: {e}")
        raise
