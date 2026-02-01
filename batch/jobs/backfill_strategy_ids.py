"""Backfill strategy_id on historical value bets.

Assigns the appropriate strategy_id to value_bets that were created
before the strategy monitoring system was implemented.
"""

import argparse
from datetime import datetime

import structlog
from sqlalchemy import select

from app.db.database import SyncSessionLocal
from app.db.models import BettingStrategy, Match, TeamStats, ValueBet

logger = structlog.get_logger()


def backfill_strategy_ids(dry_run: bool = False) -> dict:
    """Backfill strategy_id on historical value bets.

    Args:
        dry_run: If True, don't commit changes

    Returns:
        Summary of backfill results
    """
    logger.info("Starting strategy ID backfill", dry_run=dry_run)
    start_time = datetime.utcnow()

    with SyncSessionLocal() as session:
        # Load strategies
        strategies = {
            s.outcome_type: s
            for s in session.execute(select(BettingStrategy)).scalars().all()
        }

        if not strategies:
            logger.error("No strategies found. Run seed_strategies first.")
            return {"status": "error", "reason": "No strategies found"}

        logger.info(f"Loaded {len(strategies)} strategies")

        # Get value bets without strategy_id
        stmt = (
            select(ValueBet)
            .where(ValueBet.strategy_id.is_(None))
            .order_by(ValueBet.created_at)
        )
        bets = list(session.execute(stmt).scalars().all())
        logger.info(f"Found {len(bets)} value bets without strategy_id")

        if not bets:
            return {"status": "success", "updated": 0, "skipped": 0}

        # Load all matches for these bets
        match_ids = [b.match_id for b in bets]
        matches = {
            m.id: m
            for m in session.execute(
                select(Match).where(Match.id.in_(match_ids))
            ).scalars().all()
        }

        # Load team stats for form data
        seasons = {m.season for m in matches.values()}
        stats = {
            (ts.team_id, ts.season, ts.matchweek): ts
            for ts in session.execute(
                select(TeamStats).where(TeamStats.season.in_(seasons))
            ).scalars().all()
        }

        updated = 0
        skipped = 0

        for bet in bets:
            match = matches.get(bet.match_id)
            if not match:
                skipped += 1
                continue

            # Get outcome type from bet
            outcome = str(bet.outcome.value) if hasattr(bet.outcome, 'value') else str(bet.outcome)

            # Get strategy for this outcome type
            strategy = strategies.get(outcome)
            if not strategy:
                skipped += 1
                continue

            # Get home form for form-based strategies
            home_stats = stats.get(
                (match.home_team_id, match.season, match.matchweek - 1)
            )
            home_form = home_stats.form_points if home_stats else 0

            # Check if bet qualifies for the strategy
            if _qualifies_for_strategy(bet, strategy, home_form):
                bet.strategy_id = strategy.id
                updated += 1
            else:
                skipped += 1

            # Progress logging
            if (updated + skipped) % 500 == 0:
                logger.info(f"Progress: {updated} updated, {skipped} skipped")

        if not dry_run:
            session.commit()
            logger.info("Changes committed")
        else:
            session.rollback()
            logger.info("Dry run - changes rolled back")

        duration = (datetime.utcnow() - start_time).total_seconds()
        result = {
            "status": "success",
            "updated": updated,
            "skipped": skipped,
            "duration_seconds": round(duration, 1),
        }
        logger.info("Strategy ID backfill complete", **result)
        return result


def _qualifies_for_strategy(
    bet: ValueBet, strategy: BettingStrategy, home_form: int
) -> bool:
    """Check if a bet qualifies for a strategy.

    Args:
        bet: The value bet to check
        strategy: The strategy to check against
        home_form: Home team form points

    Returns:
        True if bet qualifies for the strategy
    """
    params = strategy.parameters
    edge = float(bet.edge)
    odds = float(bet.odds)
    outcome = str(bet.outcome.value) if hasattr(bet.outcome, 'value') else str(bet.outcome)

    # Check outcome type
    if outcome != strategy.outcome_type:
        return False

    # Check odds range
    if odds < params.get("min_odds", 1.01) or odds > params.get("max_odds", 100.0):
        return False

    if outcome == "away_win":
        # Away win: positive edge required (5-12%)
        min_edge = params.get("min_edge", 0.05)
        max_edge = params.get("max_edge", 0.12)
        if edge < min_edge or edge > max_edge:
            return False

        # Enhanced: exclude when home team form is 4-6
        exclude_min = params.get("exclude_home_form_min")
        exclude_max = params.get("exclude_home_form_max")
        if exclude_min is not None and exclude_max is not None:
            if exclude_min <= home_form <= exclude_max:
                return False

    elif outcome == "home_win":
        # Home win: negative edge AND hot streak required
        max_edge = params.get("max_edge", 0.0)
        min_form = params.get("min_form", 12)

        if edge > max_edge:
            return False
        if home_form < min_form:
            return False

    return True


def main():
    parser = argparse.ArgumentParser(description="Backfill strategy IDs on value bets")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be updated without making changes",
    )
    args = parser.parse_args()

    result = backfill_strategy_ids(dry_run=args.dry_run)
    print(f"Backfill complete: {result}")


if __name__ == "__main__":
    main()
