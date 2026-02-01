"""Backfill Dixon-Coles and Pi Rating predictions for existing matches.

This script populates the new prediction columns in match_analyses table
with predictions from the Dixon-Coles and Pi Rating models.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from decimal import Decimal
from datetime import datetime

import structlog
from sqlalchemy import select, update

from app.core.config import get_settings
from app.db.database import SyncSessionLocal
from app.db.models import Match, MatchAnalysis, MatchStatus, Team, TeamStats

logger = structlog.get_logger()
settings = get_settings()


def get_team_name_mapping(session) -> dict[int, str]:
    """Get mapping of team ID to team name."""
    teams = list(session.execute(select(Team)).scalars().all())
    return {t.id: t.name for t in teams}


def calculate_dixon_coles_predictions(
    home_xg: float,
    away_xg: float,
    rho: float = -0.13,
) -> tuple[float, float, float]:
    """Calculate Dixon-Coles adjusted probabilities.

    Dixon-Coles adds a correlation parameter (rho) that adjusts
    for the dependency between home and away goals, particularly
    for low-scoring matches (0-0, 1-0, 0-1, 1-1).

    Parameters
    ----------
    home_xg : float
        Expected goals for home team
    away_xg : float
        Expected goals for away team
    rho : float
        Correlation parameter (typically -0.1 to -0.15 for EPL)
        Negative rho increases probability of low-scoring outcomes
    """
    import numpy as np
    from scipy.stats import poisson

    # Calculate score probabilities up to 10 goals each
    max_goals = 10

    # Create Poisson probability matrices
    home_probs = poisson.pmf(np.arange(max_goals + 1), home_xg)
    away_probs = poisson.pmf(np.arange(max_goals + 1), away_xg)

    # Create score matrix
    score_matrix = np.outer(home_probs, away_probs)

    # Apply Dixon-Coles adjustment for low scores (0-0, 1-0, 0-1, 1-1)
    def tau(x, y, lambda_h, lambda_a, rho):
        if x == 0 and y == 0:
            return 1 - lambda_h * lambda_a * rho
        elif x == 0 and y == 1:
            return 1 + lambda_h * rho
        elif x == 1 and y == 0:
            return 1 + lambda_a * rho
        elif x == 1 and y == 1:
            return 1 - rho
        return 1.0

    # Apply tau adjustment to low scores
    for i in range(min(2, max_goals + 1)):
        for j in range(min(2, max_goals + 1)):
            score_matrix[i, j] *= tau(i, j, home_xg, away_xg, rho)

    # Normalize
    score_matrix /= score_matrix.sum()

    # Calculate outcome probabilities
    home_win = np.sum(np.tril(score_matrix, -1))  # Below diagonal
    draw = np.sum(np.diag(score_matrix))           # Diagonal
    away_win = np.sum(np.triu(score_matrix, 1))    # Above diagonal

    # Ensure they sum to 1
    total = home_win + draw + away_win
    return (home_win / total, draw / total, away_win / total)


def calculate_pi_rating_predictions(
    home_elo: float,
    away_elo: float,
    home_advantage: float = 65.0,
) -> tuple[float, float, float]:
    """Calculate Pi Rating style predictions based on ELO difference.

    Pi Ratings work by converting rating difference to expected goal difference,
    then to win probabilities. We approximate this using ELO ratings.

    Parameters
    ----------
    home_elo : float
        Home team ELO rating
    away_elo : float
        Away team ELO rating
    home_advantage : float
        Home advantage in ELO points (typically 50-100)
    """
    import numpy as np

    # Effective rating difference (including home advantage)
    rating_diff = (home_elo + home_advantage) - away_elo

    # Convert to expected score using logistic function
    # In ELO, 400 point difference = 10:1 odds
    expected_home_score = 1 / (1 + 10 ** (-rating_diff / 400))

    # Convert expected score to win/draw/away probabilities
    # Using approximation: draw probability peaks when teams are equal
    # and decreases as rating difference increases

    # Base draw probability (around 26% for EPL)
    base_draw = 0.26

    # Draw probability decreases with rating difference
    draw_reduction = min(0.15, abs(rating_diff) / 800)
    draw_prob = max(0.15, base_draw - draw_reduction)

    # Remaining probability split based on expected score
    remaining = 1 - draw_prob
    home_win = remaining * expected_home_score
    away_win = remaining * (1 - expected_home_score)

    # Normalize
    total = home_win + draw_prob + away_win
    return (home_win / total, draw_prob / total, away_win / total)


def backfill_predictions(batch_size: int = 500, dry_run: bool = False) -> dict:
    """Backfill Dixon-Coles and Pi Rating predictions.

    Parameters
    ----------
    batch_size : int
        Number of records to process per batch
    dry_run : bool
        If True, don't commit changes

    Returns
    -------
    dict
        Summary of backfill results
    """
    logger.info("Starting prediction backfill", batch_size=batch_size, dry_run=dry_run)

    with SyncSessionLocal() as session:
        # Get all analyses that need updating
        stmt = (
            select(MatchAnalysis)
            .where(MatchAnalysis.poisson_home_prob.isnot(None))  # Has Poisson predictions
            .where(MatchAnalysis.dixon_coles_home_prob.is_(None))  # Missing Dixon-Coles
        )

        analyses = list(session.execute(stmt).scalars().all())
        total = len(analyses)
        logger.info(f"Found {total} analyses to backfill")

        if total == 0:
            return {"status": "success", "updated": 0, "message": "No analyses need updating"}

        # Process in batches
        updated = 0
        errors = 0

        for i, analysis in enumerate(analyses):
            try:
                # Get the match for ELO lookup
                match = session.get(Match, analysis.match_id)
                if not match:
                    continue

                # Get expected goals from existing Poisson predictions
                # (we use these as inputs to Dixon-Coles)
                if analysis.predicted_home_goals and analysis.predicted_away_goals:
                    home_xg = float(analysis.predicted_home_goals)
                    away_xg = float(analysis.predicted_away_goals)
                else:
                    # Estimate from probabilities if xG not available
                    home_xg = 1.4  # League average
                    away_xg = 1.4

                # Calculate Dixon-Coles predictions
                dc_probs = calculate_dixon_coles_predictions(home_xg, away_xg)
                analysis.dixon_coles_home_prob = Decimal(str(round(dc_probs[0], 4)))
                analysis.dixon_coles_draw_prob = Decimal(str(round(dc_probs[1], 4)))
                analysis.dixon_coles_away_prob = Decimal(str(round(dc_probs[2], 4)))

                # Get ELO ratings from features if available
                if analysis.features and "home_elo" in analysis.features:
                    home_elo = analysis.features["home_elo"]
                    away_elo = analysis.features["away_elo"]
                else:
                    # Use default ELO if not available
                    home_elo = 1500
                    away_elo = 1500

                # Calculate Pi Rating predictions
                pi_probs = calculate_pi_rating_predictions(home_elo, away_elo)
                analysis.pi_rating_home_prob = Decimal(str(round(pi_probs[0], 4)))
                analysis.pi_rating_draw_prob = Decimal(str(round(pi_probs[1], 4)))
                analysis.pi_rating_away_prob = Decimal(str(round(pi_probs[2], 4)))

                updated += 1

                # Commit in batches
                if updated % batch_size == 0:
                    if not dry_run:
                        session.commit()
                    logger.info(f"Progress: {updated}/{total} ({100*updated/total:.1f}%)")

            except Exception as e:
                errors += 1
                logger.error(f"Error processing analysis {analysis.id}: {e}")

        # Final commit
        if not dry_run:
            session.commit()
            logger.info(f"Backfill complete: {updated} updated, {errors} errors")
        else:
            logger.info(f"Dry run complete: would update {updated}, {errors} errors")

        return {
            "status": "success",
            "total": total,
            "updated": updated,
            "errors": errors,
            "dry_run": dry_run,
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Backfill Dixon-Coles and Pi Rating predictions")
    parser.add_argument("--dry-run", action="store_true", help="Don't commit changes")
    parser.add_argument("--batch-size", type=int, default=500, help="Batch size for commits")
    args = parser.parse_args()

    result = backfill_predictions(batch_size=args.batch_size, dry_run=args.dry_run)
    print(f"Result: {result}")
