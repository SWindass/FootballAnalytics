"""Seed initial betting strategies into the database.

Creates the two proven profitable strategies:
1. Away wins with 5-12% edge, excluding home form 4-6
2. Home wins with form 12+ and negative edge
"""

import argparse
from datetime import datetime

import structlog
from sqlalchemy import select

from app.db.database import SyncSessionLocal
from app.db.models import BettingStrategy, StrategyStatus

logger = structlog.get_logger()

# Strategy definitions from CLAUDE.md
STRATEGIES = [
    {
        "name": "away_win_edge_5_12",
        "description": (
            "Away wins with 5-12% positive edge. "
            "Excludes bets when home team form is 4-6 (poor but not terrible). "
            "Backtested 2020-2025: 85 bets, 54.1% win rate, +32.9% ROI."
        ),
        "outcome_type": "away_win",
        "parameters": {
            "min_edge": 0.05,
            "max_edge": 0.12,
            "min_odds": 1.50,
            "max_odds": 8.00,
            "exclude_home_form_min": 4,
            "exclude_home_form_max": 6,
        },
    },
    {
        "name": "home_win_form_12_negative_edge",
        "description": (
            "Home wins when team is on hot streak (12+ form points) "
            "AND market values them more than our model (negative edge). "
            "Counterintuitive but profitable - trust market on momentum. "
            "Backtested 2020-2025: 94 bets, 67.0% win rate, +30.4% ROI."
        ),
        "outcome_type": "home_win",
        "parameters": {
            "max_edge": 0.0,  # Negative edge required
            "min_form": 12,  # Hot streak required
            "min_odds": 1.01,
            "max_odds": 10.00,
        },
    },
]


def seed_strategies(force: bool = False) -> dict:
    """Seed initial strategies into the database.

    Args:
        force: If True, update existing strategies with new parameters

    Returns:
        Summary of seeding results
    """
    logger.info("Starting strategy seeding")

    with SyncSessionLocal() as session:
        created = 0
        updated = 0
        skipped = 0

        for strategy_def in STRATEGIES:
            # Check if strategy already exists
            stmt = select(BettingStrategy).where(
                BettingStrategy.name == strategy_def["name"]
            )
            existing = session.execute(stmt).scalar_one_or_none()

            if existing:
                if force:
                    # Update existing strategy
                    existing.description = strategy_def["description"]
                    existing.outcome_type = strategy_def["outcome_type"]
                    existing.parameters = strategy_def["parameters"]
                    existing.updated_at = datetime.utcnow()
                    updated += 1
                    logger.info(f"Updated strategy: {strategy_def['name']}")
                else:
                    skipped += 1
                    logger.info(f"Skipped existing strategy: {strategy_def['name']}")
            else:
                # Create new strategy
                strategy = BettingStrategy(
                    name=strategy_def["name"],
                    description=strategy_def["description"],
                    outcome_type=strategy_def["outcome_type"],
                    parameters=strategy_def["parameters"],
                    status=StrategyStatus.ACTIVE,
                )
                session.add(strategy)
                created += 1
                logger.info(f"Created strategy: {strategy_def['name']}")

        session.commit()

        result = {
            "status": "success",
            "created": created,
            "updated": updated,
            "skipped": skipped,
        }
        logger.info("Strategy seeding complete", **result)
        return result


def main():
    parser = argparse.ArgumentParser(description="Seed betting strategies")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Update existing strategies with new parameters",
    )
    args = parser.parse_args()

    result = seed_strategies(force=args.force)
    print(f"Seeding complete: {result}")


if __name__ == "__main__":
    main()
