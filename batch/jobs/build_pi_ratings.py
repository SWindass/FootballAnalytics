"""Build Pi Ratings from historical match data.

Loads match results from the database and builds Pi ratings for all teams.
Can optionally optimize parameters and export results.
"""

import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import select

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from app.db.database import SyncSessionLocal
from app.db.models import Match, Team
from batch.models.pi_rating import PiRating, optimize_parameters, plot_rating_history


def load_matches_from_db(
    session,
    seasons: list[str] = None,
    completed_only: bool = True,
) -> pd.DataFrame:
    """Load match data from database.

    Parameters
    ----------
    session : Session
        SQLAlchemy session.
    seasons : list[str], optional
        Filter to specific seasons (e.g., ["2023-24", "2024-25"]).
    completed_only : bool
        Only include completed matches with scores.

    Returns
    -------
    pd.DataFrame
        Match data ready for Pi Rating processing.
    """
    # Build query
    query = (
        select(
            Match.id,
            Match.kickoff_time,
            Match.matchweek,
            Match.status,
            Match.home_score,
            Match.away_score,
            Match.season,
        )
        .add_columns(
            Team.name.label("home_team"),
        )
    )

    # Join home team
    query = query.join(Team, Match.home_team_id == Team.id)

    if completed_only:
        query = query.where(Match.status == "finished")
        query = query.where(Match.home_score.isnot(None))
        query = query.where(Match.away_score.isnot(None))

    if seasons:
        query = query.where(Match.season.in_(seasons))

    query = query.order_by(Match.kickoff_time)

    # Execute and get results
    results = session.execute(query).all()

    # Need to get away team names separately
    match_ids = [r.id for r in results]
    away_teams = {}

    if match_ids:
        away_query = (
            select(Match.id, Team.name.label("away_team"))
            .join(Team, Match.away_team_id == Team.id)
            .where(Match.id.in_(match_ids))
        )
        away_results = session.execute(away_query).all()
        away_teams = {r.id: r.away_team for r in away_results}

    # Build DataFrame
    data = []
    for r in results:
        data.append({
            "date": r.kickoff_time,
            "season": r.season,
            "matchday": r.matchweek,
            "home_team": r.home_team,
            "away_team": away_teams.get(r.id, "Unknown"),
            "home_goals": r.home_score,
            "away_goals": r.away_score,
        })

    return pd.DataFrame(data)


def build_pi_ratings(
    seasons: list[str] = None,
    lambda_param: float = 0.06,
    gamma_param: float = 0.6,
    optimize: bool = False,
) -> dict:
    """Build Pi ratings from database match history.

    Parameters
    ----------
    seasons : list[str], optional
        Seasons to include. If None, uses all available.
    lambda_param : float
        Learning rate parameter.
    gamma_param : float
        Away venue multiplier.
    optimize : bool
        Whether to optimize parameters.

    Returns
    -------
    dict
        Results including ratings, history, and performance metrics.
    """
    with SyncSessionLocal() as session:
        print("Loading matches from database...")
        matches_df = load_matches_from_db(session, seasons=seasons)

        if matches_df.empty:
            print("No matches found!")
            return {"error": "No matches found"}

        print(f"Loaded {len(matches_df)} matches")

        if seasons:
            print(f"Seasons: {', '.join(seasons)}")
        else:
            available_seasons = sorted(matches_df["season"].unique())
            print(f"Seasons: {', '.join(available_seasons)}")

        # Optimize parameters if requested
        if optimize:
            print("\nOptimizing parameters...")
            opt_result = optimize_parameters(
                matches_df,
                lambda_range=(0.02, 0.12),
                gamma_range=(0.4, 0.8),
                n_trials=100,
            )
            best = opt_result["best_params"]
            print(f"Optimal lambda: {best['lambda']:.4f}")
            print(f"Optimal gamma: {best['gamma']:.4f}")
            print(f"Best MSE: {best['mse']:.4f}")

            lambda_param = best["lambda"]
            gamma_param = best["gamma"]

        # Build ratings
        print(f"\nBuilding Pi ratings (λ={lambda_param}, γ={gamma_param})...")
        pi = PiRating(lambda_param=lambda_param, gamma_param=gamma_param)
        pi.process_matches(matches_df)

        # Get results
        ratings_df = pi.get_ratings_dataframe()
        history_df = pi.get_history_dataframe()
        predictions_df = pi.get_predictions_dataframe()

        print(f"\nProcessed {pi._match_count} matches")
        print(f"Teams rated: {len(ratings_df)}")
        print(f"MSE: {pi.calculate_mse():.4f}")
        print(f"MAE: {pi.calculate_mae():.4f}")

        return {
            "pi_rating": pi,
            "ratings": ratings_df,
            "history": history_df,
            "predictions": predictions_df,
            "mse": pi.calculate_mse(),
            "mae": pi.calculate_mae(),
            "match_count": pi._match_count,
            "params": {"lambda": lambda_param, "gamma": gamma_param},
        }


def predict_upcoming_matches(pi: PiRating, session=None) -> pd.DataFrame:
    """Predict upcoming (scheduled) matches.

    Parameters
    ----------
    pi : PiRating
        Trained PiRating model.
    session : Session, optional
        Database session. Creates one if not provided.

    Returns
    -------
    pd.DataFrame
        Predictions for upcoming matches.
    """
    close_session = False
    if session is None:
        session = SyncSessionLocal()
        close_session = True

    try:
        # Get scheduled matches
        query = (
            select(
                Match.id,
                Match.kickoff_time,
                Match.matchweek,
            )
            .add_columns(
                Team.name.label("home_team"),
            )
            .join(Team, Match.home_team_id == Team.id)
            .where(Match.status.in_(["scheduled", "timed"]))
            .order_by(Match.kickoff_time)
        )

        results = session.execute(query).all()

        if not results:
            return pd.DataFrame()

        # Get away teams
        match_ids = [r.id for r in results]
        away_query = (
            select(Match.id, Team.name.label("away_team"))
            .join(Team, Match.away_team_id == Team.id)
            .where(Match.id.in_(match_ids))
        )
        away_results = session.execute(away_query).all()
        away_teams = {r.id: r.away_team for r in away_results}

        # Generate predictions
        predictions = []
        for r in results:
            away_team = away_teams.get(r.id, "Unknown")
            pred = pi.predict_match(r.home_team, away_team)

            # Convert goal diff to win probabilities (simplified)
            # Using a basic logistic-style conversion
            import math
            home_prob = 1 / (1 + math.exp(-pred.predicted_goal_diff * 0.7))
            away_prob = 1 / (1 + math.exp(pred.predicted_goal_diff * 0.7))
            draw_prob = 1 - home_prob - away_prob
            if draw_prob < 0:
                # Normalize
                total = home_prob + away_prob
                home_prob /= total
                away_prob /= total
                draw_prob = 0

            predictions.append({
                "date": r.kickoff_time,
                "matchday": r.matchweek,
                "home_team": r.home_team,
                "away_team": away_team,
                "predicted_gd": round(pred.predicted_goal_diff, 2),
                "home_rating": round(pred.home_rating, 3),
                "away_rating": round(pred.away_rating, 3),
                "home_win_prob": round(home_prob, 3),
                "draw_prob": round(max(0, draw_prob), 3),
                "away_win_prob": round(away_prob, 3),
            })

        return pd.DataFrame(predictions)

    finally:
        if close_session:
            session.close()


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Build Pi Ratings from match history")
    parser.add_argument(
        "--seasons",
        nargs="*",
        help="Seasons to include (e.g., 2023-24 2024-25)",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Optimize lambda and gamma parameters",
    )
    parser.add_argument(
        "--lambda",
        dest="lambda_param",
        type=float,
        default=0.06,
        help="Lambda parameter (default: 0.06)",
    )
    parser.add_argument(
        "--gamma",
        dest="gamma_param",
        type=float,
        default=0.6,
        help="Gamma parameter (default: 0.6)",
    )
    parser.add_argument(
        "--predict",
        action="store_true",
        help="Show predictions for upcoming matches",
    )
    parser.add_argument(
        "--plot",
        nargs="*",
        metavar="TEAM",
        help="Plot rating history for specified teams",
    )
    parser.add_argument(
        "--export",
        type=str,
        help="Export ratings to CSV file",
    )

    args = parser.parse_args()

    # Build ratings
    result = build_pi_ratings(
        seasons=args.seasons,
        lambda_param=args.lambda_param,
        gamma_param=args.gamma_param,
        optimize=args.optimize,
    )

    if "error" in result:
        return

    pi = result["pi_rating"]
    ratings_df = result["ratings"]

    # Display top teams
    print("\nTop 10 Teams by Pi Rating:")
    print("=" * 70)
    print(ratings_df.head(10).to_string(index=False))

    # Show predictions if requested
    if args.predict:
        print("\nUpcoming Match Predictions:")
        print("=" * 70)
        predictions = predict_upcoming_matches(pi)
        if predictions.empty:
            print("No upcoming matches found")
        else:
            print(predictions.to_string(index=False))

    # Plot if requested
    if args.plot:
        print(f"\nGenerating plot for: {', '.join(args.plot)}")
        try:
            fig = plot_rating_history(pi, args.plot)
            output_path = project_root / "pi_ratings_plot.png"
            fig.savefig(output_path)
            print(f"Plot saved to: {output_path}")
        except Exception as e:
            print(f"Error generating plot: {e}")

    # Export if requested
    if args.export:
        ratings_df.to_csv(args.export, index=False)
        print(f"\nRatings exported to: {args.export}")


if __name__ == "__main__":
    main()
