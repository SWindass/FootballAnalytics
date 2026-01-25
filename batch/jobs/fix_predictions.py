"""Fix predictions with extreme or suspicious values.

This script recalculates predictions for matches that have:
1. Extreme probabilities (>90% for any outcome)
2. Missing or corrupted consensus values

Run with: PYTHONPATH=. python batch/jobs/fix_predictions.py
"""

import argparse
from decimal import Decimal
from typing import Optional

import structlog
from sqlalchemy import select

from app.core.config import get_settings
from app.db.database import SyncSessionLocal
from app.db.models import EloRating, Match, MatchAnalysis, Team, TeamStats
from batch.models.elo import EloRatingSystem
from batch.models.poisson import PoissonModel, calculate_team_strengths

logger = structlog.get_logger()
settings = get_settings()


def fix_extreme_predictions(
    extreme_threshold: float = 0.90,
    dry_run: bool = False,
) -> dict:
    """Fix predictions with extreme probability values.

    Args:
        extreme_threshold: Predictions above this threshold are considered extreme
        dry_run: If True, only report what would be fixed without making changes

    Returns:
        Summary of fixes applied
    """
    with SyncSessionLocal() as session:
        # Find matches with extreme predictions
        stmt = (
            select(Match, MatchAnalysis)
            .join(MatchAnalysis, Match.id == MatchAnalysis.match_id)
            .where(
                (MatchAnalysis.consensus_home_prob >= extreme_threshold) |
                (MatchAnalysis.consensus_away_prob >= extreme_threshold)
            )
        )
        extreme_matches = list(session.execute(stmt).all())

        if not extreme_matches:
            logger.info("No extreme predictions found")
            return {"fixed": 0, "skipped": 0}

        logger.info(f"Found {len(extreme_matches)} matches with extreme predictions")

        # Load all teams
        teams = {t.id: t for t in session.execute(select(Team)).scalars().all()}

        # Initialize models
        elo = EloRatingSystem()
        poisson = PoissonModel()

        # Load ELO ratings
        all_elo = list(session.execute(select(EloRating)).scalars().all())
        elo_by_team_mw = {}
        for e in all_elo:
            key = (e.team_id, e.season, e.matchweek)
            elo_by_team_mw[key] = float(e.rating)

        # Load completed matches for Poisson strengths
        completed_stmt = (
            select(Match)
            .where(Match.season == settings.current_season)
            .where(Match.status == "finished")
        )
        completed_matches = [
            {
                "home_team_id": m.home_team_id,
                "away_team_id": m.away_team_id,
                "home_score": m.home_score,
                "away_score": m.away_score,
            }
            for m in session.execute(completed_stmt).scalars().all()
        ]
        poisson_strengths = calculate_team_strengths(
            completed_matches, league_avg_scored=1.4, league_avg_conceded=1.4
        )

        fixed = 0
        skipped = 0

        for match, analysis in extreme_matches:
            home_team = teams[match.home_team_id]
            away_team = teams[match.away_team_id]

            old_home = float(analysis.consensus_home_prob)
            old_draw = float(analysis.consensus_draw_prob)
            old_away = float(analysis.consensus_away_prob)

            logger.info(
                f"Fixing: {home_team.short_name} vs {away_team.short_name} "
                f"(H={old_home:.2f}, D={old_draw:.2f}, A={old_away:.2f})"
            )

            # Get ELO ratings for this matchweek
            prev_mw = match.matchweek - 1
            home_elo = elo_by_team_mw.get(
                (match.home_team_id, match.season, prev_mw), 1500
            )
            away_elo = elo_by_team_mw.get(
                (match.away_team_id, match.season, prev_mw), 1500
            )

            # Set ELO ratings
            elo.set_rating(match.home_team_id, home_elo)
            elo.set_rating(match.away_team_id, away_elo)

            # Calculate ELO probabilities
            elo_probs = elo.match_probabilities(match.home_team_id, match.away_team_id)

            # Calculate Poisson probabilities
            home_strength = poisson_strengths.get(match.home_team_id, (1.0, 1.0))
            away_strength = poisson_strengths.get(match.away_team_id, (1.0, 1.0))
            home_exp, away_exp = poisson.calculate_expected_goals(
                home_strength[0], home_strength[1],
                away_strength[0], away_strength[1],
            )
            poisson_probs = poisson.match_probabilities(home_exp, away_exp)

            # Simple weighted average (no neural stacker to avoid the original issue)
            consensus = (
                0.5 * elo_probs[0] + 0.5 * poisson_probs[0],
                0.5 * elo_probs[1] + 0.5 * poisson_probs[1],
                0.5 * elo_probs[2] + 0.5 * poisson_probs[2],
            )

            # Normalize
            total = sum(consensus)
            consensus = (consensus[0] / total, consensus[1] / total, consensus[2] / total)

            # Apply draw adjustment for close matches
            home_away_gap = abs(consensus[0] - consensus[2])
            if home_away_gap < 0.10:
                # Boost draw for close match
                draw_boost = 0.05
                new_draw = consensus[1] + draw_boost
                home_share = consensus[0] / (consensus[0] + consensus[2])
                new_home = consensus[0] - (draw_boost * home_share)
                new_away = consensus[2] - (draw_boost * (1 - home_share))
                total = new_home + new_draw + new_away
                consensus = (new_home / total, new_draw / total, new_away / total)

            logger.info(
                f"  New: H={consensus[0]:.2f}, D={consensus[1]:.2f}, A={consensus[2]:.2f}"
            )

            if not dry_run:
                analysis.consensus_home_prob = Decimal(str(round(consensus[0], 4)))
                analysis.consensus_draw_prob = Decimal(str(round(consensus[1], 4)))
                analysis.consensus_away_prob = Decimal(str(round(consensus[2], 4)))
                analysis.elo_home_prob = Decimal(str(round(elo_probs[0], 4)))
                analysis.elo_draw_prob = Decimal(str(round(elo_probs[1], 4)))
                analysis.elo_away_prob = Decimal(str(round(elo_probs[2], 4)))
                analysis.poisson_home_prob = Decimal(str(round(poisson_probs[0], 4)))
                analysis.poisson_draw_prob = Decimal(str(round(poisson_probs[1], 4)))
                analysis.poisson_away_prob = Decimal(str(round(poisson_probs[2], 4)))
                fixed += 1
            else:
                skipped += 1

        if not dry_run:
            session.commit()
            logger.info(f"Fixed {fixed} predictions")
        else:
            logger.info(f"Dry run: would fix {skipped} predictions")

        return {"fixed": fixed, "skipped": skipped}


def apply_draw_adjustment_to_all(
    min_matchweek: int = 1,
    season: Optional[str] = None,
    dry_run: bool = False,
) -> dict:
    """Apply draw adjustment to all existing predictions.

    This recalculates consensus probabilities with the new draw boost logic
    for close matches and model disagreement.

    Args:
        min_matchweek: Only process matches from this matchweek onwards
        season: If specified, only process this season. Otherwise process ALL seasons.
        dry_run: If True, only report what would be changed

    Returns:
        Summary of changes made
    """
    import numpy as np

    with SyncSessionLocal() as session:
        # Get all finished matches with predictions
        stmt = (
            select(Match, MatchAnalysis)
            .join(MatchAnalysis, Match.id == MatchAnalysis.match_id)
            .where(Match.matchweek >= min_matchweek)
            .where(MatchAnalysis.elo_home_prob.isnot(None))
            .where(MatchAnalysis.poisson_home_prob.isnot(None))
            .order_by(Match.kickoff_time)
        )

        # Optionally filter by season
        if season:
            stmt = stmt.where(Match.season == season)
        results = list(session.execute(stmt).all())

        logger.info(f"Found {len(results)} matches to process")

        updated = 0
        unchanged = 0

        for match, analysis in results:
            # Get original model predictions
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

            # Get market probs if available
            market_probs = None
            if analysis.features:
                features = analysis.features if isinstance(analysis.features, dict) else {}
                if "market_home_prob" in features:
                    market_probs = (
                        features["market_home_prob"],
                        features["market_draw_prob"],
                        features["market_away_prob"],
                    )

            # Calculate base consensus (simple average without market, or with market blend)
            if market_probs:
                base_consensus = (
                    0.3 * elo_probs[0] + 0.3 * poisson_probs[0] + 0.4 * market_probs[0],
                    0.3 * elo_probs[1] + 0.3 * poisson_probs[1] + 0.4 * market_probs[1],
                    0.3 * elo_probs[2] + 0.3 * poisson_probs[2] + 0.4 * market_probs[2],
                )
            else:
                base_consensus = (
                    0.5 * elo_probs[0] + 0.5 * poisson_probs[0],
                    0.5 * elo_probs[1] + 0.5 * poisson_probs[1],
                    0.5 * elo_probs[2] + 0.5 * poisson_probs[2],
                )

            # Normalize
            total = sum(base_consensus)
            base_consensus = tuple(p / total for p in base_consensus)

            # Apply draw adjustment
            home_prob, draw_prob, away_prob = base_consensus

            # Model disagreement check
            if not market_probs:
                market_probs = (0.4, 0.27, 0.33)

            elo_fav = np.argmax(elo_probs)
            poisson_fav = np.argmax(poisson_probs)
            market_fav = np.argmax(market_probs)

            models_agree = (elo_fav == poisson_fav == market_fav)
            disagreement_boost = 0.0 if models_agree else 0.05

            # Close match boost
            home_away_gap = abs(home_prob - away_prob)
            if home_away_gap < 0.05:
                closeness_boost = 0.08
            elif home_away_gap < 0.10:
                closeness_boost = 0.05
            elif home_away_gap < 0.15:
                closeness_boost = 0.03
            else:
                closeness_boost = 0.0

            total_boost = min(0.10, disagreement_boost + closeness_boost)

            if total_boost > 0:
                new_draw = draw_prob + total_boost
                home_share = home_prob / (home_prob + away_prob) if (home_prob + away_prob) > 0 else 0.5
                new_home = home_prob - (total_boost * home_share)
                new_away = away_prob - (total_boost * (1 - home_share))

                new_home = max(0.05, new_home)
                new_away = max(0.05, new_away)
                new_draw = min(0.50, new_draw)

                total = new_home + new_draw + new_away
                new_consensus = (new_home / total, new_draw / total, new_away / total)
            else:
                new_consensus = base_consensus

            # Check if changed significantly
            old_consensus = (
                float(analysis.consensus_home_prob),
                float(analysis.consensus_draw_prob),
                float(analysis.consensus_away_prob),
            )

            diff = sum(abs(old_consensus[i] - new_consensus[i]) for i in range(3))

            if diff > 0.01:  # Changed by more than 1%
                if not dry_run:
                    analysis.consensus_home_prob = Decimal(str(round(new_consensus[0], 4)))
                    analysis.consensus_draw_prob = Decimal(str(round(new_consensus[1], 4)))
                    analysis.consensus_away_prob = Decimal(str(round(new_consensus[2], 4)))
                updated += 1
            else:
                unchanged += 1

        if not dry_run:
            session.commit()
            logger.info(f"Updated {updated} predictions, {unchanged} unchanged")
        else:
            logger.info(f"Dry run: would update {updated} predictions, {unchanged} unchanged")

        return {"updated": updated, "unchanged": unchanged}


def main():
    parser = argparse.ArgumentParser(description="Fix predictions")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be fixed without making changes",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.90,
        help="Threshold for extreme predictions (default: 0.90)",
    )
    parser.add_argument(
        "--apply-draw-adjustment",
        action="store_true",
        help="Apply draw adjustment to all predictions",
    )
    parser.add_argument(
        "--min-matchweek",
        type=int,
        default=1,
        help="Minimum matchweek to process (default: 1)",
    )
    parser.add_argument(
        "--season",
        type=str,
        default=None,
        help="Season to process (e.g., '2024-25'). If not specified, process ALL seasons.",
    )
    args = parser.parse_args()

    if args.apply_draw_adjustment:
        result = apply_draw_adjustment_to_all(
            min_matchweek=args.min_matchweek,
            season=args.season,
            dry_run=args.dry_run,
        )
    else:
        result = fix_extreme_predictions(
            extreme_threshold=args.threshold,
            dry_run=args.dry_run,
        )

    print(f"\nResult: {result}")


if __name__ == "__main__":
    main()
