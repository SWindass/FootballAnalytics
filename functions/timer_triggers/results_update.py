"""Results update timer trigger - runs every 6 hours."""

import logging
import sys
from pathlib import Path

import azure.functions as func

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from batch.jobs.results_update import run_results_update

app = func.FunctionApp()


@app.timer_trigger(
    schedule="0 0 */6 * * *",  # Every 6 hours
    arg_name="timer",
    run_on_startup=False,
)
def results_update_trigger(timer: func.TimerRequest) -> None:
    """Execute results update job."""
    logging.info("Results update trigger started")

    if timer.past_due:
        logging.warning("Timer is past due!")

    try:
        result = run_results_update()
        logging.info(f"Results update completed: {result}")
    except Exception as e:
        logging.error(f"Results update failed: {e}")
        raise
