"""Injury update timer trigger - runs Friday 3PM UK time."""

import logging
import sys
from pathlib import Path

import azure.functions as func

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from batch.jobs.injury_update import run_injury_update

app = func.FunctionApp()


@app.timer_trigger(
    schedule="0 0 15 * * 5",  # Every Friday at 3PM UTC
    arg_name="timer",
    run_on_startup=False,
)
def injury_update_trigger(timer: func.TimerRequest) -> None:
    """Execute injury update job."""
    logging.info("Injury update trigger started")

    if timer.past_due:
        logging.warning("Timer is past due!")

    try:
        result = run_injury_update()
        logging.info(f"Injury update completed: {result}")
    except Exception as e:
        logging.error(f"Injury update failed: {e}")
        raise
