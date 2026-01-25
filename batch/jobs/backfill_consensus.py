"""Backfill consensus predictions using the trained consensus stacker model.

This script re-runs all historical predictions through the newly trained
consensus stacker model to update the consensus probabilities.

Run with: PYTHONPATH=. python batch/jobs/backfill_consensus.py
"""

import argparse
from decimal import Decimal
from typing import Optional

import numpy as np
import structlog
from sqlalchemy import select

from app.db.database import SyncSessionLocal
from app.db.models import Match, MatchAnalysis, EloRating
from batch.models.consensus_stacker import ConsensusStacker

logger = structlog.get_logger()


def backfill_with_consensus_stacker(
    season: Optional[str] = None,
    dry_run: bool = False,
    batch_size: int = 500,
) -> dict:
    """Backfill all predictions using the trained consensus stacker.

    Args:
        season: If specified, only process this season. Otherwise ALL seasons.
        dry_run: If True, only report what would be changed.
        batch_size: Number of predictions to update per commit.

    Returns:
        Summary of changes made.
    """
    # Load the trained consensus stacker
    stacker = ConsensusStacker()
    if not stacker.load_model():
        logger.error("Failed to load consensus stacker model")
        return {"error": "Model not loaded"}

    logger.info("Loaded consensus stacker model")

    with SyncSessionLocal() as session:
        # Load ELO ratings for computing elo_diff
        all_elo = list(session.execute(select(EloRating)).scalars().all())
        elo_lookup = {
            (e.team_id, e.season, e.matchweek): float(e.rating)
            for e in all_elo
        }
        logger.info(f"Loaded {len(elo_lookup)} ELO ratings")

        # Get all matches with ELO and Poisson predictions
        stmt = (
            select(Match, MatchAnalysis)
            .join(MatchAnalysis, Match.id == MatchAnalysis.match_id)
            .where(MatchAnalysis.elo_home_prob.isnot(None))
            .where(MatchAnalysis.poisson_home_prob.isnot(None))
            .order_by(Match.kickoff_time)
        )

        if season:
            stmt = stmt.where(Match.season == season)

        results = list(session.execute(stmt).all())
        logger.info(f"Found {len(results)} matches to process")

        updated = 0
        unchanged = 0
        errors = 0

        for i, (match, analysis) in enumerate(results):
            try:
                # Get ELO and Poisson predictions
                elo_probs = (
                    float(analysis.elo_home_prob),
                    float(analysis.elo_draw_prob),
                    float(analysis.elo_away_prob),
                )
                poisson_probs = (
                    float(analysis.poisson_home_prob),
                    float(analysis.poisson_draw_prob),
                    float(analysis.poisson_away_prob),
                )

                # Get market probs if available, otherwise use defaults
                market_probs = (0.4, 0.27, 0.33)  # Default
                if analysis.features and isinstance(analysis.features, dict):
                    if "market_home_prob" in analysis.features:
                        market_probs = (
                            analysis.features["market_home_prob"],
                            analysis.features["market_draw_prob"],
                            analysis.features["market_away_prob"],
                        )

                # Get ELO diff
                prev_mw = max(1, match.matchweek - 1)
                home_elo = elo_lookup.get(
                    (match.home_team_id, match.season, prev_mw), 1500
                )
                away_elo = elo_lookup.get(
                    (match.away_team_id, match.season, prev_mw), 1500
                )
                elo_diff = home_elo - away_elo

                # Run through consensus stacker
                home_prob, draw_prob, away_prob, confidence = stacker.predict(
                    elo_probs, poisson_probs, market_probs, elo_diff
                )

                # Check if significantly changed
                old_consensus = (
                    float(analysis.consensus_home_prob),
                    float(analysis.consensus_draw_prob),
                    float(analysis.consensus_away_prob),
                )
                new_consensus = (home_prob, draw_prob, away_prob)

                diff = sum(abs(old_consensus[j] - new_consensus[j]) for j in range(3))

                if diff > 0.01:  # Changed by more than 1%
                    if not dry_run:
                        analysis.consensus_home_prob = Decimal(str(round(home_prob, 4)))
                        analysis.consensus_draw_prob = Decimal(str(round(draw_prob, 4)))
                        analysis.consensus_away_prob = Decimal(str(round(away_prob, 4)))
                        analysis.confidence = Decimal(str(round(confidence, 4)))
                    updated += 1
                else:
                    unchanged += 1

            except Exception as e:
                logger.warning(f"Error processing match {match.id}: {e}")
                errors += 1

            # Commit in batches
            if not dry_run and (i + 1) % batch_size == 0:
                session.commit()
                logger.info(f"Committed batch {(i + 1) // batch_size}, updated {updated} so far")

        # Final commit
        if not dry_run:
            session.commit()

        logger.info(
            f"{'Dry run: would update' if dry_run else 'Updated'} {updated} predictions, "
            f"{unchanged} unchanged, {errors} errors"
        )

        return {"updated": updated, "unchanged": unchanged, "errors": errors}


def main():
    parser = argparse.ArgumentParser(
        description="Backfill predictions using trained consensus stacker"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without making changes",
    )
    parser.add_argument(
        "--season",
        type=str,
        default=None,
        help="Season to process (e.g., '2024-25'). If not specified, process ALL seasons.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Number of predictions to update per commit (default: 500)",
    )
    args = parser.parse_args()

    result = backfill_with_consensus_stacker(
        season=args.season,
        dry_run=args.dry_run,
        batch_size=args.batch_size,
    )

    print(f"\nResult: {result}")


if __name__ == "__main__":
    main()
