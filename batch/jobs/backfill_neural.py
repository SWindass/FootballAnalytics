"""Backfill predictions using the trained neural stacker model.

This script re-runs all historical predictions through the neural stacker
which uses rich features (form, goals, H2H, etc.) from TeamStats.

Run with: PYTHONPATH=. python batch/jobs/backfill_neural.py
"""

import argparse
from collections import defaultdict
from decimal import Decimal
from typing import Optional

import structlog
from sqlalchemy import select

from app.db.database import SyncSessionLocal
from app.db.models import Match, MatchAnalysis, MatchStatus, EloRating, TeamStats, Referee
from batch.models.neural_stacker import NeuralStacker

logger = structlog.get_logger()


def backfill_with_neural_stacker(
    season: Optional[str] = None,
    dry_run: bool = False,
    batch_size: int = 500,
) -> dict:
    """Backfill all predictions using the trained neural stacker.

    Args:
        season: If specified, only process this season. Otherwise ALL seasons.
        dry_run: If True, only report what would be changed.
        batch_size: Number of predictions to update per commit.

    Returns:
        Summary of changes made.
    """
    # Load the trained neural stacker
    stacker = NeuralStacker()
    if not stacker.load_model():
        logger.error("Failed to load neural stacker model")
        return {"error": "Model not loaded"}

    logger.info("Loaded neural stacker model")

    with SyncSessionLocal() as session:
        # Load ELO ratings
        all_elo = list(session.execute(select(EloRating)).scalars().all())
        elo_by_team_mw = {}
        for e in all_elo:
            key = (e.team_id, e.season, e.matchweek)
            elo_by_team_mw[key] = e
        logger.info(f"Loaded {len(elo_by_team_mw)} ELO ratings")

        # Load TeamStats
        all_stats = list(session.execute(select(TeamStats)).scalars().all())
        stats_by_team_mw = {}
        for s in all_stats:
            key = (s.team_id, s.season, s.matchweek)
            stats_by_team_mw[key] = s
        logger.info(f"Loaded {len(stats_by_team_mw)} TeamStats records")

        # Load Referees (may be empty)
        all_refs = list(session.execute(select(Referee)).scalars().all())
        refs_by_id = {r.id: r for r in all_refs}
        logger.info(f"Loaded {len(refs_by_id)} referees")

        # Load all finished matches for H2H/venue/recency calculations
        all_finished_matches = list(
            session.execute(
                select(Match)
                .where(Match.status == MatchStatus.FINISHED)
                .order_by(Match.kickoff_time)
            ).scalars().all()
        )
        logger.info(f"Loaded {len(all_finished_matches)} finished matches for feature calculation")

        # Build match lookup structures for efficient feature calculation
        h2h_matches = defaultdict(list)
        team_home_matches = defaultdict(list)
        team_away_matches = defaultdict(list)
        team_all_matches = defaultdict(list)

        for m in all_finished_matches:
            # H2H: store with canonical key (sorted team ids)
            key = tuple(sorted([m.home_team_id, m.away_team_id]))
            h2h_matches[key].append(m)

            # Venue-specific
            team_home_matches[m.home_team_id].append(m)
            team_away_matches[m.away_team_id].append(m)

            # All matches
            team_all_matches[m.home_team_id].append(m)
            team_all_matches[m.away_team_id].append(m)

        logger.info("Built match lookup structures")

        # Get all matches with predictions
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
        skipped = 0
        errors = 0

        # Helper to find most recent stats for a team/season
        def get_team_stats(team_id: int, season: str, matchweek: int):
            """Get TeamStats, falling back to most recent available if exact MW missing.

            Note: We intentionally don't fall back to previous season stats because
            previous season's form data doesn't predict early season matches well
            (summer transfers and squad changes make old form unreliable).
            The neural stacker will detect the cold start and use base models instead.
            """
            # Try exact matchweek first
            stats = stats_by_team_mw.get((team_id, season, matchweek))
            if stats:
                return stats

            # Fall back to most recent available matchweek in current season
            for mw in range(matchweek - 1, 0, -1):
                stats = stats_by_team_mw.get((team_id, season, mw))
                if stats:
                    return stats

            # No current season stats - return None so neural stacker uses cold start fallback
            return None

        for i, (match, analysis) in enumerate(results):
            try:
                # Get TeamStats for this match's matchweek (or most recent)
                home_stats = get_team_stats(
                    match.home_team_id, match.season, match.matchweek
                )
                away_stats = get_team_stats(
                    match.away_team_id, match.season, match.matchweek
                )

                # Get ELO ratings (use previous matchweek)
                prev_mw = max(1, match.matchweek - 1)
                home_elo = elo_by_team_mw.get(
                    (match.home_team_id, match.season, prev_mw)
                )
                away_elo = elo_by_team_mw.get(
                    (match.away_team_id, match.season, prev_mw)
                )

                # Get referee if assigned
                referee = refs_by_id.get(match.referee_id) if match.referee_id else None

                # Note: We no longer skip matches without stats - the neural stacker
                # will detect cold start situations and fall back to base model predictions

                # Calculate H2H features
                h2h_features = stacker._calculate_h2h_from_cache(
                    h2h_matches, match.home_team_id, match.away_team_id, match.kickoff_time
                )

                # Calculate venue-specific form
                home_home_ppg = stacker._calculate_venue_form_from_cache(
                    team_home_matches[match.home_team_id],
                    match.home_team_id, is_home=True,
                    before_date=match.kickoff_time, season=match.season
                )
                away_away_ppg = stacker._calculate_venue_form_from_cache(
                    team_away_matches[match.away_team_id],
                    match.away_team_id, is_home=False,
                    before_date=match.kickoff_time, season=match.season
                )

                # Calculate recency-weighted form
                home_recency = stacker._calculate_recency_from_cache(
                    team_all_matches[match.home_team_id],
                    match.home_team_id, match.kickoff_time
                )
                away_recency = stacker._calculate_recency_from_cache(
                    team_all_matches[match.away_team_id],
                    match.away_team_id, match.kickoff_time
                )

                # Run through neural stacker with all features
                consensus = stacker.predict(
                    analysis, home_stats, away_stats, home_elo, away_elo,
                    referee, home_rest_days=None, away_rest_days=None,
                    home_team_id=match.home_team_id, away_team_id=match.away_team_id,
                    h2h_features=h2h_features,
                    home_home_ppg=home_home_ppg, away_away_ppg=away_away_ppg,
                    home_recency=home_recency, away_recency=away_recency
                )

                if consensus is None:
                    skipped += 1
                    continue

                home_prob, draw_prob, away_prob = consensus

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
            f"{unchanged} unchanged, {skipped} skipped (no stats), {errors} errors"
        )

        return {
            "updated": updated,
            "unchanged": unchanged,
            "skipped": skipped,
            "errors": errors,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Backfill predictions using trained neural stacker"
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

    result = backfill_with_neural_stacker(
        season=args.season,
        dry_run=args.dry_run,
        batch_size=args.batch_size,
    )

    print(f"\nResult: {result}")


if __name__ == "__main__":
    main()
