"""Odds refresh timer trigger - runs Saturday 8AM UK time."""

import logging
import sys
from pathlib import Path

import azure.functions as func

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from batch.jobs.odds_refresh import run_odds_refresh

app = func.FunctionApp()


@app.timer_trigger(
    schedule="0 0 8 * * 6",  # Every Saturday at 8AM UTC
    arg_name="timer",
    run_on_startup=False,
)
def odds_refresh_trigger(timer: func.TimerRequest) -> None:
    """Execute odds refresh job."""
    logging.info("Odds refresh trigger started")

    if timer.past_due:
        logging.warning("Timer is past due!")

    try:
        result = run_odds_refresh()
        logging.info(f"Odds refresh completed: {result}")
    except Exception as e:
        logging.error(f"Odds refresh failed: {e}")
        raise
