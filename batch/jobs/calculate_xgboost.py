"""Train XGBoost model and generate predictions.

Builds features from historical match data, trains the classifier,
and generates match outcome predictions.
"""

import argparse
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import pandas as pd
import structlog
from sqlalchemy import select

from app.core.config import get_settings
from app.db.database import SyncSessionLocal
from app.db.models import EloRating, Match, MatchAnalysis, MatchStatus
from batch.models.xgboost_model import MatchOutcomeClassifier

logger = structlog.get_logger()
settings = get_settings()

MODEL_PATH = Path("models/xgboost_match_predictor.pkl")


def get_team_form(
    session,
    team_id: int,
    before_date: datetime,
    num_matches: int = 5,
) -> dict:
    """Calculate team's recent form before a given date.

    Returns:
        Dict with form stats (points, goals, xG, etc.)
    """
    # Get last N matches for this team before the date
    stmt = (
        select(Match)
        .where(
            ((Match.home_team_id == team_id) | (Match.away_team_id == team_id))
            & (Match.status == MatchStatus.FINISHED)
            & (Match.kickoff_time < before_date)
        )
        .order_by(Match.kickoff_time.desc())
        .limit(num_matches)
    )
    matches = list(session.execute(stmt).scalars().all())

    if not matches:
        return {
            "form_points": 7.5,  # Average (1.5 ppg * 5)
            "avg_scored": 1.4,
            "avg_conceded": 1.4,
            "avg_xg_for": 1.4,
            "avg_xg_against": 1.4,
            "home_wins": 0,
            "home_draws": 0,
            "home_losses": 0,
            "away_wins": 0,
            "away_draws": 0,
            "away_losses": 0,
        }

    points = 0
    goals_scored = 0
    goals_conceded = 0
    xg_for = 0.0
    xg_against = 0.0
    xg_count = 0
    home_wins = home_draws = home_losses = 0
    away_wins = away_draws = away_losses = 0

    for match in matches:
        is_home = match.home_team_id == team_id
        if is_home:
            scored = match.home_score or 0
            conceded = match.away_score or 0
            # xG data
            if match.home_xg is not None:
                xg_for += float(match.home_xg)
                xg_against += float(match.away_xg) if match.away_xg else 0
                xg_count += 1
        else:
            scored = match.away_score or 0
            conceded = match.home_score or 0
            # xG data
            if match.away_xg is not None:
                xg_for += float(match.away_xg)
                xg_against += float(match.home_xg) if match.home_xg else 0
                xg_count += 1

        goals_scored += scored
        goals_conceded += conceded

        if scored > conceded:
            points += 3
            if is_home:
                home_wins += 1
            else:
                away_wins += 1
        elif scored == conceded:
            points += 1
            if is_home:
                home_draws += 1
            else:
                away_draws += 1
        else:
            if is_home:
                home_losses += 1
            else:
                away_losses += 1

    num = len(matches)
    return {
        "form_points": points,
        "avg_scored": goals_scored / num if num > 0 else 1.4,
        "avg_conceded": goals_conceded / num if num > 0 else 1.4,
        "avg_xg_for": xg_for / xg_count if xg_count > 0 else goals_scored / num if num > 0 else 1.4,
        "avg_xg_against": xg_against / xg_count if xg_count > 0 else goals_conceded / num if num > 0 else 1.4,
        "home_wins": home_wins,
        "home_draws": home_draws,
        "home_losses": home_losses,
        "away_wins": away_wins,
        "away_draws": away_draws,
        "away_losses": away_losses,
    }


def get_elo_rating(session, team_id: int, season: str, matchweek: int) -> float:
    """Get team's ELO rating before a matchweek."""
    stmt = (
        select(EloRating.rating)
        .where(EloRating.team_id == team_id)
        .where(EloRating.season == season)
        .where(EloRating.matchweek < matchweek)
        .order_by(EloRating.matchweek.desc())
        .limit(1)
    )
    result = session.execute(stmt).scalar_one_or_none()
    return float(result) if result else 1500.0


def get_h2h_stats(
    session,
    home_team_id: int,
    away_team_id: int,
    before_date: datetime,
    num_matches: int = 5,
) -> dict:
    """Get head-to-head statistics between two teams."""
    stmt = (
        select(Match)
        .where(
            (
                ((Match.home_team_id == home_team_id) & (Match.away_team_id == away_team_id))
                | ((Match.home_team_id == away_team_id) & (Match.away_team_id == home_team_id))
            )
            & (Match.status == MatchStatus.FINISHED)
            & (Match.kickoff_time < before_date)
        )
        .order_by(Match.kickoff_time.desc())
        .limit(num_matches)
    )
    matches = list(session.execute(stmt).scalars().all())

    home_wins = draws = away_wins = 0
    for match in matches:
        if match.home_score > match.away_score:
            if match.home_team_id == home_team_id:
                home_wins += 1
            else:
                away_wins += 1
        elif match.home_score < match.away_score:
            if match.away_team_id == away_team_id:
                away_wins += 1
            else:
                home_wins += 1
        else:
            draws += 1

    return {
        "h2h_home_wins": home_wins,
        "h2h_draws": draws,
        "h2h_away_wins": away_wins,
    }


def build_training_data(
    seasons: list[str],
    min_matchweek: int = 6,
) -> pd.DataFrame:
    """Build training dataset from historical matches.

    Args:
        seasons: List of seasons to include
        min_matchweek: Minimum matchweek to include (need history for features)

    Returns:
        DataFrame with features and outcome labels
    """
    print(f"Building training data for seasons: {seasons}")

    with SyncSessionLocal() as session:
        all_data = []

        for season in seasons:
            print(f"  Processing {season}...")

            # Get finished matches from this season
            stmt = (
                select(Match)
                .where(Match.season == season)
                .where(Match.status == MatchStatus.FINISHED)
                .where(Match.matchweek >= min_matchweek)
                .order_by(Match.matchweek, Match.kickoff_time)
            )
            matches = list(session.execute(stmt).scalars().all())

            for match in matches:
                # Get ELO ratings
                home_elo = get_elo_rating(session, match.home_team_id, season, match.matchweek)
                away_elo = get_elo_rating(session, match.away_team_id, season, match.matchweek)

                # Get form stats
                home_form = get_team_form(session, match.home_team_id, match.kickoff_time)
                away_form = get_team_form(session, match.away_team_id, match.kickoff_time)

                # Get H2H stats
                h2h = get_h2h_stats(session, match.home_team_id, match.away_team_id, match.kickoff_time)

                # Determine outcome
                if match.home_score > match.away_score:
                    outcome = 0  # Home win
                elif match.home_score < match.away_score:
                    outcome = 2  # Away win
                else:
                    outcome = 1  # Draw

                # Build feature row
                row = {
                    "match_id": match.id,
                    "season": season,
                    "matchweek": match.matchweek,
                    # ELO features
                    "home_elo": home_elo,
                    "away_elo": away_elo,
                    "elo_diff": home_elo - away_elo,
                    # Form features
                    "home_form_points": home_form["form_points"],
                    "away_form_points": away_form["form_points"],
                    "form_diff": home_form["form_points"] - away_form["form_points"],
                    # Goals features
                    "home_avg_scored": home_form["avg_scored"],
                    "home_avg_conceded": home_form["avg_conceded"],
                    "away_avg_scored": away_form["avg_scored"],
                    "away_avg_conceded": away_form["avg_conceded"],
                    "goal_diff_attack": home_form["avg_scored"] - away_form["avg_scored"],
                    "goal_diff_defense": away_form["avg_conceded"] - home_form["avg_conceded"],
                    # xG features (uses actual xG data when available)
                    "home_avg_xg_for": home_form["avg_xg_for"],
                    "home_avg_xg_against": home_form["avg_xg_against"],
                    "away_avg_xg_for": away_form["avg_xg_for"],
                    "away_avg_xg_against": away_form["avg_xg_against"],
                    "xg_diff_for": home_form["avg_xg_for"] - away_form["avg_xg_for"],
                    "xg_diff_against": away_form["avg_xg_against"] - home_form["avg_xg_against"],
                    # Home/Away PPG
                    "home_home_ppg": _calc_ppg(home_form["home_wins"], home_form["home_draws"], home_form["home_losses"]),
                    "away_away_ppg": _calc_ppg(away_form["away_wins"], away_form["away_draws"], away_form["away_losses"]),
                    # H2H
                    "h2h_home_wins": h2h["h2h_home_wins"],
                    "h2h_draws": h2h["h2h_draws"],
                    "h2h_away_wins": h2h["h2h_away_wins"],
                    # Target
                    "outcome": outcome,
                }
                all_data.append(row)

            print(f"    Added {len(matches)} matches")

    df = pd.DataFrame(all_data)
    print(f"Total training samples: {len(df)}")
    return df


def _calc_ppg(wins: int, draws: int, losses: int) -> float:
    """Calculate points per game."""
    games = wins + draws + losses
    if games == 0:
        return 1.5
    return (wins * 3 + draws) / games


def train_model(
    training_seasons: list[str],
    test_size: float = 0.2,
) -> dict:
    """Train XGBoost model on historical data.

    Args:
        training_seasons: Seasons to use for training
        test_size: Fraction for test split

    Returns:
        Training metrics
    """
    # Build training data
    df = build_training_data(training_seasons)

    if len(df) < 100:
        print("Not enough training data")
        return {"status": "insufficient_data", "samples": len(df)}

    # Train model
    print("\nTraining XGBoost model...")
    classifier = MatchOutcomeClassifier()
    metrics = classifier.train(df, test_size=test_size)

    print("\nTraining Results:")
    print(f"  Train Accuracy: {metrics['train_accuracy']:.1%}")
    print(f"  Test Accuracy: {metrics['test_accuracy']:.1%}")
    print(f"  CV Mean: {metrics['cv_mean']:.1%} (+/- {metrics['cv_std']:.1%})")

    print("\nTop 10 Feature Importance:")
    sorted_features = sorted(
        metrics["feature_importance"].items(),
        key=lambda x: x[1],
        reverse=True,
    )[:10]
    for name, importance in sorted_features:
        print(f"  {name}: {importance:.4f}")

    # Save model
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    classifier.save(MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")

    return {
        "status": "success",
        "samples": len(df),
        "train_accuracy": metrics["train_accuracy"],
        "test_accuracy": metrics["test_accuracy"],
        "cv_mean": metrics["cv_mean"],
    }


def generate_predictions(
    season: str,
    matchweek: int | None = None,
    backfill: bool = False,
) -> dict:
    """Generate XGBoost predictions for matches.

    Args:
        season: Season to predict
        matchweek: Specific matchweek (None = latest)
        backfill: Predict all matchweeks

    Returns:
        Summary dict
    """
    # Load model
    if not MODEL_PATH.exists():
        print(f"Model not found at {MODEL_PATH}. Run with --train first.")
        return {"status": "no_model"}

    classifier = MatchOutcomeClassifier()
    classifier.load(MODEL_PATH)
    print(f"Loaded model from {MODEL_PATH}")

    with SyncSessionLocal() as session:
        # Determine matchweeks
        if matchweek is not None:
            matchweeks_to_process = [matchweek]
        elif backfill:
            stmt = (
                select(Match.matchweek)
                .where(Match.season == season)
                .where(Match.matchweek >= 6)  # Need history for features
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

        print(f"Generating predictions for matchweeks: {matchweeks_to_process}")
        predictions_updated = 0

        for mw in matchweeks_to_process:
            stmt = (
                select(Match)
                .where(Match.season == season)
                .where(Match.matchweek == mw)
                .order_by(Match.kickoff_time)
            )
            matches = list(session.execute(stmt).scalars().all())

            for match in matches:
                # Build features
                home_elo = get_elo_rating(session, match.home_team_id, season, mw)
                away_elo = get_elo_rating(session, match.away_team_id, season, mw)
                home_form = get_team_form(session, match.home_team_id, match.kickoff_time)
                away_form = get_team_form(session, match.away_team_id, match.kickoff_time)
                h2h = get_h2h_stats(session, match.home_team_id, match.away_team_id, match.kickoff_time)

                features = {
                    "home_elo": home_elo,
                    "away_elo": away_elo,
                    "elo_diff": home_elo - away_elo,
                    "home_form_points": home_form["form_points"],
                    "away_form_points": away_form["form_points"],
                    "form_diff": home_form["form_points"] - away_form["form_points"],
                    "home_avg_scored": home_form["avg_scored"],
                    "home_avg_conceded": home_form["avg_conceded"],
                    "away_avg_scored": away_form["avg_scored"],
                    "away_avg_conceded": away_form["avg_conceded"],
                    "goal_diff_attack": home_form["avg_scored"] - away_form["avg_scored"],
                    "goal_diff_defense": away_form["avg_conceded"] - home_form["avg_conceded"],
                    "home_avg_xg_for": home_form["avg_scored"],
                    "home_avg_xg_against": home_form["avg_conceded"],
                    "away_avg_xg_for": away_form["avg_scored"],
                    "away_avg_xg_against": away_form["avg_conceded"],
                    "xg_diff_for": home_form["avg_scored"] - away_form["avg_scored"],
                    "xg_diff_against": away_form["avg_conceded"] - home_form["avg_conceded"],
                    "home_home_ppg": _calc_ppg(home_form["home_wins"], home_form["home_draws"], home_form["home_losses"]),
                    "away_away_ppg": _calc_ppg(away_form["away_wins"], away_form["away_draws"], away_form["away_losses"]),
                    "h2h_home_wins": h2h["h2h_home_wins"],
                    "h2h_draws": h2h["h2h_draws"],
                    "h2h_away_wins": h2h["h2h_away_wins"],
                }

                # Get predictions
                home_prob, draw_prob, away_prob = classifier.predict_match(features)

                # Update MatchAnalysis
                existing = session.execute(
                    select(MatchAnalysis).where(MatchAnalysis.match_id == match.id)
                ).scalar_one_or_none()

                if existing:
                    existing.xgboost_home_prob = Decimal(str(round(home_prob, 4)))
                    existing.xgboost_draw_prob = Decimal(str(round(draw_prob, 4)))
                    existing.xgboost_away_prob = Decimal(str(round(away_prob, 4)))
                else:
                    analysis = MatchAnalysis(
                        match_id=match.id,
                        xgboost_home_prob=Decimal(str(round(home_prob, 4))),
                        xgboost_draw_prob=Decimal(str(round(draw_prob, 4))),
                        xgboost_away_prob=Decimal(str(round(away_prob, 4))),
                    )
                    session.add(analysis)

                predictions_updated += 1

        session.commit()
        print(f"Updated {predictions_updated} XGBoost predictions")

        return {
            "status": "success",
            "predictions_updated": predictions_updated,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XGBoost model and generate predictions")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument(
        "--train-seasons",
        type=str,
        default="2020-21,2021-22,2022-23,2023-24",
        help="Comma-separated seasons for training",
    )
    parser.add_argument("--predict", action="store_true", help="Generate predictions")
    parser.add_argument("--season", type=str, default="2024-25", help="Season to predict")
    parser.add_argument("--matchweek", type=int, default=None, help="Specific matchweek")
    parser.add_argument("--backfill", action="store_true", help="Predict all matchweeks")

    args = parser.parse_args()

    if args.train:
        seasons = [s.strip() for s in args.train_seasons.split(",")]
        train_model(seasons)

    if args.predict:
        generate_predictions(
            season=args.season,
            matchweek=args.matchweek,
            backfill=args.backfill,
        )

    if not args.train and not args.predict:
        print("Specify --train and/or --predict")
        print("Example: python calculate_xgboost.py --train --predict --backfill")
