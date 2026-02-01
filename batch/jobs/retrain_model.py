"""Retrain neural stacker model - can be run manually or scheduled.

This job retrains the neural network that combines ELO and Poisson predictions
to learn optimal weighting from historical match results.

Usage:
    python batch/jobs/retrain_model.py [--epochs N] [--force]

Options:
    --epochs N    Number of training epochs (default: 100)
    --force       Retrain even if minimum data threshold not met
"""

import argparse
from datetime import datetime

import structlog
from sqlalchemy import select

from app.db.database import SyncSessionLocal
from app.db.models import Match, MatchStatus
from batch.models.neural_stacker import NeuralStacker

logger = structlog.get_logger()

MIN_MATCHES_DEFAULT = 100


def retrain_neural_stacker(
    epochs: int = 100,
    min_matches: int = MIN_MATCHES_DEFAULT,
    force: bool = False,
) -> dict:
    """Retrain the neural stacker model.

    Args:
        epochs: Number of training epochs
        min_matches: Minimum finished matches required
        force: Retrain even if below threshold

    Returns:
        Training metrics dict
    """
    start_time = datetime.utcnow()

    # Check data availability
    with SyncSessionLocal() as session:
        stmt = select(Match).where(Match.status == MatchStatus.FINISHED)
        total_finished = len(list(session.execute(stmt).scalars().all()))

    logger.info(f"Found {total_finished} finished matches")

    if total_finished < min_matches and not force:
        logger.warning(
            f"Only {total_finished} finished matches, need {min_matches}+ for training. "
            "Use --force to override."
        )
        return {
            "status": "skipped",
            "reason": f"Insufficient data ({total_finished} < {min_matches})",
        }

    # Train model
    logger.info(f"Training neural stacker for {epochs} epochs...")
    stacker = NeuralStacker()

    try:
        metrics = stacker.train(
            epochs=epochs,
            batch_size=32,
            learning_rate=0.001,
            validation_split=0.2,
        )

        duration = (datetime.utcnow() - start_time).total_seconds()

        logger.info(
            "Neural stacker training complete",
            val_accuracy=f"{metrics['best_val_acc']:.1%}",
            best_epoch=metrics['best_epoch'],
            train_samples=metrics['train_samples'],
            val_samples=metrics['val_samples'],
            duration_seconds=round(duration, 1),
        )

        return {
            "status": "success",
            "val_accuracy": metrics['best_val_acc'],
            "best_epoch": metrics['best_epoch'],
            "train_samples": metrics['train_samples'],
            "val_samples": metrics['val_samples'],
            "duration_seconds": duration,
        }

    except ValueError as e:
        logger.error(f"Training failed: {e}")
        return {
            "status": "error",
            "reason": str(e),
        }


def main():
    parser = argparse.ArgumentParser(
        description="Retrain neural stacker model"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)",
    )
    parser.add_argument(
        "--min-matches",
        type=int,
        default=MIN_MATCHES_DEFAULT,
        help=f"Minimum finished matches required (default: {MIN_MATCHES_DEFAULT})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain even if below minimum data threshold",
    )

    args = parser.parse_args()

    # Suppress SQLAlchemy logging
    import logging
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

    print("Retraining Neural Stacker Model")
    print("================================")
    print(f"Epochs: {args.epochs}")
    print(f"Min matches: {args.min_matches}")
    print()

    result = retrain_neural_stacker(
        epochs=args.epochs,
        min_matches=args.min_matches,
        force=args.force,
    )

    if result["status"] == "success":
        print("\nTraining complete!")
        print(f"  Validation accuracy: {result['val_accuracy']:.1%}")
        print(f"  Best epoch: {result['best_epoch']}")
        print(f"  Training samples: {result['train_samples']}")
        print(f"  Validation samples: {result['val_samples']}")
        print(f"  Duration: {result['duration_seconds']:.1f}s")
    elif result["status"] == "skipped":
        print(f"\nSkipped: {result['reason']}")
    else:
        print(f"\nError: {result['reason']}")


if __name__ == "__main__":
    main()
