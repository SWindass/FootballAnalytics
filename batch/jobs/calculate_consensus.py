"""Calculate consensus probabilities from multiple prediction models.

Combines ELO, Poisson, and XGBoost predictions into weighted averages.
"""

import argparse
from decimal import Decimal

import structlog
from sqlalchemy import select

from app.core.config import get_settings
from app.db.database import SyncSessionLocal
from app.db.models import EloRating, Match, MatchAnalysis, MatchStatus, Team
from batch.models.elo import EloRatingSystem

logger = structlog.get_logger()
settings = get_settings()

# Default weights for consensus calculation
DEFAULT_WEIGHTS = {
    "elo": 0.35,
    "poisson": 0.40,
    "xgboost": 0.25,
}


def get_elo_ratings_for_matchweek(
    session,
    season: str,
    matchweek: int,
) -> dict[int, float]:
    """Get ELO ratings for all teams at a specific matchweek.

    Uses the most recent rating before or at the matchweek.
    """
    # Get the latest rating for each team up to this matchweek
    from sqlalchemy import func

    subq = (
        select(
            EloRating.team_id,
            func.max(EloRating.matchweek).label("max_mw"),
        )
        .where(EloRating.season == season)
        .where(EloRating.matchweek < matchweek)
        .group_by(EloRating.team_id)
        .subquery()
    )

    stmt = (
        select(EloRating)
        .join(
            subq,
            (EloRating.team_id == subq.c.team_id)
            & (EloRating.matchweek == subq.c.max_mw),
        )
        .where(EloRating.season == season)
    )

    ratings = {r.team_id: float(r.rating) for r in session.execute(stmt).scalars().all()}
    return ratings


def calculate_elo_predictions(
    season: str,
    matchweek: int | None = None,
    backfill: bool = False,
) -> dict:
    """Calculate ELO-based match predictions.

    Args:
        season: Season to calculate for
        matchweek: Specific matchweek (None = latest)
        backfill: Recalculate all matchweeks

    Returns:
        Summary dict
    """
    with SyncSessionLocal() as session:
        {t.id: t for t in session.execute(select(Team)).scalars().all()}

        # Determine matchweeks to process
        if matchweek is not None:
            matchweeks_to_process = [matchweek]
        elif backfill:
            stmt = (
                select(Match.matchweek)
                .where(Match.season == season)
                .distinct()
                .order_by(Match.matchweek)
            )
            matchweeks_to_process = [mw for (mw,) in session.execute(stmt).all()]
        else:
            stmt = (
                select(Match.matchweek)
                .where(Match.season == season)
                .order_by(Match.matchweek.desc())
                .limit(1)
            )
            result = session.execute(stmt).first()
            matchweeks_to_process = [result[0]] if result else []

        if not matchweeks_to_process:
            print(f"No matchweeks found for season {season}")
            return {"status": "no_matchweeks"}

        print(f"Calculating ELO predictions for matchweeks: {matchweeks_to_process}")

        elo_system = EloRatingSystem()
        predictions_updated = 0

        for mw in matchweeks_to_process:
            # Get ELO ratings up to this matchweek
            elo_ratings = get_elo_ratings_for_matchweek(session, season, mw)

            if not elo_ratings:
                print(f"  MW{mw}: No ELO ratings available, using defaults")

            # Initialize ELO system with ratings
            elo_system.ratings = elo_ratings.copy()

            # Get matches for this matchweek
            stmt = (
                select(Match)
                .where(Match.season == season)
                .where(Match.matchweek == mw)
                .order_by(Match.kickoff_time)
            )
            matches = list(session.execute(stmt).scalars().all())

            for match in matches:
                # Calculate ELO probabilities
                home_prob, draw_prob, away_prob = elo_system.match_probabilities(
                    match.home_team_id,
                    match.away_team_id,
                )

                # Get or create MatchAnalysis
                existing = session.execute(
                    select(MatchAnalysis).where(MatchAnalysis.match_id == match.id)
                ).scalar_one_or_none()

                if existing:
                    existing.elo_home_prob = Decimal(str(round(home_prob, 4)))
                    existing.elo_draw_prob = Decimal(str(round(draw_prob, 4)))
                    existing.elo_away_prob = Decimal(str(round(away_prob, 4)))
                    predictions_updated += 1
                else:
                    analysis = MatchAnalysis(
                        match_id=match.id,
                        elo_home_prob=Decimal(str(round(home_prob, 4))),
                        elo_draw_prob=Decimal(str(round(draw_prob, 4))),
                        elo_away_prob=Decimal(str(round(away_prob, 4))),
                    )
                    session.add(analysis)
                    predictions_updated += 1

        session.commit()
        print(f"Updated {predictions_updated} ELO predictions")
        return {"status": "success", "elo_predictions_updated": predictions_updated}


def calculate_consensus(
    season: str,
    matchweek: int | None = None,
    backfill: bool = False,
    weights: dict[str, float] | None = None,
) -> dict:
    """Calculate weighted consensus from available model predictions.

    Args:
        season: Season to calculate for
        matchweek: Specific matchweek (None = latest)
        backfill: Recalculate all matchweeks
        weights: Custom weights for models (default: elo=0.35, poisson=0.40, xgboost=0.25)

    Returns:
        Summary dict
    """
    weights = weights or DEFAULT_WEIGHTS.copy()

    with SyncSessionLocal() as session:
        {t.id: t for t in session.execute(select(Team)).scalars().all()}

        # Determine matchweeks to process
        if matchweek is not None:
            matchweeks_to_process = [matchweek]
        elif backfill:
            stmt = (
                select(Match.matchweek)
                .where(Match.season == season)
                .distinct()
                .order_by(Match.matchweek)
            )
            matchweeks_to_process = [mw for (mw,) in session.execute(stmt).all()]
        else:
            stmt = (
                select(Match.matchweek)
                .where(Match.season == season)
                .order_by(Match.matchweek.desc())
                .limit(1)
            )
            result = session.execute(stmt).first()
            matchweeks_to_process = [result[0]] if result else []

        if not matchweeks_to_process:
            return {"status": "no_matchweeks"}

        print(f"\nCalculating consensus for matchweeks: {matchweeks_to_process}")
        print(f"Weights: ELO={weights['elo']:.0%}, Poisson={weights['poisson']:.0%}, XGBoost={weights['xgboost']:.0%}")

        consensus_updated = 0

        for mw in matchweeks_to_process:
            # Get all analyses for this matchweek
            stmt = (
                select(MatchAnalysis)
                .join(Match, Match.id == MatchAnalysis.match_id)
                .where(Match.season == season)
                .where(Match.matchweek == mw)
            )
            analyses = list(session.execute(stmt).scalars().all())

            for analysis in analyses:
                # Collect available predictions
                models = {}

                if analysis.elo_home_prob is not None:
                    models["elo"] = {
                        "home": float(analysis.elo_home_prob),
                        "draw": float(analysis.elo_draw_prob),
                        "away": float(analysis.elo_away_prob),
                    }

                if analysis.poisson_home_prob is not None:
                    models["poisson"] = {
                        "home": float(analysis.poisson_home_prob),
                        "draw": float(analysis.poisson_draw_prob),
                        "away": float(analysis.poisson_away_prob),
                    }

                if analysis.xgboost_home_prob is not None:
                    models["xgboost"] = {
                        "home": float(analysis.xgboost_home_prob),
                        "draw": float(analysis.xgboost_draw_prob),
                        "away": float(analysis.xgboost_away_prob),
                    }

                if not models:
                    continue

                # Calculate weighted consensus
                # Normalize weights based on available models
                available_weight = sum(weights[m] for m in models.keys())
                if available_weight == 0:
                    continue

                consensus_home = 0.0
                consensus_draw = 0.0
                consensus_away = 0.0

                for model_name, probs in models.items():
                    normalized_weight = weights[model_name] / available_weight
                    consensus_home += probs["home"] * normalized_weight
                    consensus_draw += probs["draw"] * normalized_weight
                    consensus_away += probs["away"] * normalized_weight

                # Normalize to ensure sum = 1.0
                total = consensus_home + consensus_draw + consensus_away
                if total > 0:
                    consensus_home /= total
                    consensus_draw /= total
                    consensus_away /= total

                # Update analysis
                analysis.consensus_home_prob = Decimal(str(round(consensus_home, 4)))
                analysis.consensus_draw_prob = Decimal(str(round(consensus_draw, 4)))
                analysis.consensus_away_prob = Decimal(str(round(consensus_away, 4)))

                # Calculate confidence based on model agreement
                # Higher agreement = higher confidence
                if len(models) > 1:
                    home_variance = sum(
                        (probs["home"] - consensus_home) ** 2 for probs in models.values()
                    ) / len(models)
                    draw_variance = sum(
                        (probs["draw"] - consensus_draw) ** 2 for probs in models.values()
                    ) / len(models)
                    away_variance = sum(
                        (probs["away"] - consensus_away) ** 2 for probs in models.values()
                    ) / len(models)
                    avg_variance = (home_variance + draw_variance + away_variance) / 3
                    # Convert variance to confidence (0-1, lower variance = higher confidence)
                    confidence = max(0, 1 - (avg_variance * 10))
                    analysis.confidence = Decimal(str(round(confidence, 3)))
                else:
                    analysis.confidence = Decimal("0.5")  # Single model = medium confidence

                consensus_updated += 1

        session.commit()
        print(f"Updated {consensus_updated} consensus predictions")

        return {
            "status": "success",
            "consensus_updated": consensus_updated,
            "models_used": list(weights.keys()),
        }


def print_predictions(season: str, matchweek: int) -> None:
    """Print predictions comparison for a matchweek."""
    with SyncSessionLocal() as session:
        teams = {t.id: t for t in session.execute(select(Team)).scalars().all()}

        stmt = (
            select(Match, MatchAnalysis)
            .join(MatchAnalysis, Match.id == MatchAnalysis.match_id)
            .where(Match.season == season)
            .where(Match.matchweek == matchweek)
            .order_by(Match.kickoff_time)
        )
        results = list(session.execute(stmt).all())

        if not results:
            print(f"No predictions found for {season} MW{matchweek}")
            return

        print(f"\n{'='*100}")
        print(f"Predictions for {season} Matchweek {matchweek}")
        print("=" * 100)
        print(f"{'Match':<35} {'ELO':^20} {'Poisson':^20} {'Consensus':^20}")
        print(f"{'':<35} {'H/D/A':^20} {'H/D/A':^20} {'H/D/A':^20}")
        print("-" * 100)

        for match, analysis in results:
            home_name = teams.get(match.home_team_id).short_name if match.home_team_id in teams else "?"
            away_name = teams.get(match.away_team_id).short_name if match.away_team_id in teams else "?"
            match_str = f"{home_name} vs {away_name}"

            def fmt_probs(h, d, a):
                if h is None:
                    return "N/A"
                return f"{float(h)*100:4.1f}/{float(d)*100:4.1f}/{float(a)*100:4.1f}"

            elo_str = fmt_probs(analysis.elo_home_prob, analysis.elo_draw_prob, analysis.elo_away_prob)
            poisson_str = fmt_probs(analysis.poisson_home_prob, analysis.poisson_draw_prob, analysis.poisson_away_prob)
            consensus_str = fmt_probs(analysis.consensus_home_prob, analysis.consensus_draw_prob, analysis.consensus_away_prob)

            # Add actual result if available
            result_str = ""
            if match.status == MatchStatus.FINISHED and match.home_score is not None:
                result_str = f" [{match.home_score}-{match.away_score}]"

            print(f"{match_str:<35} {elo_str:^20} {poisson_str:^20} {consensus_str:^20}{result_str}")

        print("=" * 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate consensus predictions")
    parser.add_argument("--season", type=str, default="2024-25", help="Season to calculate")
    parser.add_argument("--matchweek", type=int, default=None, help="Specific matchweek")
    parser.add_argument("--backfill", action="store_true", help="Recalculate all matchweeks")
    parser.add_argument("--elo-only", action="store_true", help="Only calculate ELO predictions")
    parser.add_argument("--show", action="store_true", help="Show predictions for matchweek")
    parser.add_argument("--elo-weight", type=float, default=0.35, help="ELO weight (default: 0.35)")
    parser.add_argument("--poisson-weight", type=float, default=0.40, help="Poisson weight (default: 0.40)")
    parser.add_argument("--xgboost-weight", type=float, default=0.25, help="XGBoost weight (default: 0.25)")

    args = parser.parse_args()

    weights = {
        "elo": args.elo_weight,
        "poisson": args.poisson_weight,
        "xgboost": args.xgboost_weight,
    }

    # Step 1: Calculate ELO predictions
    print("Step 1: Calculating ELO predictions...")
    elo_result = calculate_elo_predictions(
        season=args.season,
        matchweek=args.matchweek,
        backfill=args.backfill,
    )
    print(f"ELO result: {elo_result}")

    if not args.elo_only:
        # Step 2: Calculate consensus
        print("\nStep 2: Calculating consensus...")
        consensus_result = calculate_consensus(
            season=args.season,
            matchweek=args.matchweek,
            backfill=args.backfill,
            weights=weights,
        )
        print(f"Consensus result: {consensus_result}")

    # Show predictions if requested
    if args.show:
        mw = args.matchweek
        if mw is None:
            with SyncSessionLocal() as session:
                stmt = (
                    select(Match.matchweek)
                    .where(Match.season == args.season)
                    .order_by(Match.matchweek.desc())
                    .limit(1)
                )
                result = session.execute(stmt).first()
                mw = result[0] if result else 1
        print_predictions(args.season, mw)
