"""Backtest Pi Rating against ELO and actual results.

Compares predictive accuracy of Pi Rating vs ELO using walk-forward validation.
"""

import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import select

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from app.db.database import SyncSessionLocal
from app.db.models import Match, Team
from batch.models.pi_rating import PiRating
from batch.models.elo import EloRatingSystem


def load_matches(session, seasons: list[str] = None) -> pd.DataFrame:
    """Load completed matches from database."""
    query = (
        select(
            Match.id,
            Match.kickoff_time,
            Match.matchweek,
            Match.season,
            Match.home_team_id,
            Match.away_team_id,
            Match.home_score,
            Match.away_score,
        )
        .where(Match.status == "finished")
        .where(Match.home_score.isnot(None))
        .where(Match.away_score.isnot(None))
    )

    if seasons:
        query = query.where(Match.season.in_(seasons))

    query = query.order_by(Match.kickoff_time)
    results = session.execute(query).all()

    # Get team names
    teams_query = select(Team.id, Team.name)
    team_map = {r.id: r.name for r in session.execute(teams_query).all()}

    data = []
    for r in results:
        data.append({
            "match_id": r.id,
            "date": r.kickoff_time,
            "season": r.season,
            "matchweek": r.matchweek,
            "home_team_id": r.home_team_id,
            "away_team_id": r.away_team_id,
            "home_team": team_map.get(r.home_team_id, "Unknown"),
            "away_team": team_map.get(r.away_team_id, "Unknown"),
            "home_goals": r.home_score,
            "away_goals": r.away_score,
        })

    return pd.DataFrame(data)


def backtest_pi_rating(
    matches_df: pd.DataFrame,
    lambda_param: float = 0.06,
    gamma_param: float = 0.6,
    warmup_matches: int = 100,
) -> pd.DataFrame:
    """Run walk-forward backtest of Pi Rating.

    For each match:
    1. Make prediction using current ratings
    2. Record prediction and actual result
    3. Update ratings with actual result

    Parameters
    ----------
    matches_df : pd.DataFrame
        Match data sorted chronologically.
    lambda_param : float
        Pi Rating learning rate.
    gamma_param : float
        Pi Rating away multiplier.
    warmup_matches : int
        Number of initial matches for warmup (not included in results).

    Returns
    -------
    pd.DataFrame
        Backtest results with predictions and actual outcomes.
    """
    pi = PiRating(lambda_param=lambda_param, gamma_param=gamma_param)

    results = []

    for idx, row in matches_df.iterrows():
        # Get prediction BEFORE updating ratings
        predicted_gd = pi.calculate_expected_goal_diff(row["home_team"], row["away_team"])

        # Convert to win probabilities (simplified logistic)
        import math
        home_prob = 1 / (1 + math.exp(-predicted_gd * 0.7))
        away_prob = 1 / (1 + math.exp(predicted_gd * 0.7))
        # Simple draw estimate
        draw_prob = max(0, 0.28 - 0.1 * abs(predicted_gd))
        total = home_prob + draw_prob + away_prob
        home_prob, draw_prob, away_prob = home_prob/total, draw_prob/total, away_prob/total

        # Actual result
        actual_gd = row["home_goals"] - row["away_goals"]
        if actual_gd > 0:
            actual_outcome = "H"
        elif actual_gd < 0:
            actual_outcome = "A"
        else:
            actual_outcome = "D"

        # Predicted outcome (highest probability)
        probs = {"H": home_prob, "D": draw_prob, "A": away_prob}
        predicted_outcome = max(probs, key=probs.get)

        # Update ratings with actual result
        pi.update_ratings(
            row["home_team"],
            row["away_team"],
            row["home_goals"],
            row["away_goals"],
            row["date"],
            store_history=False,
        )

        # Skip warmup matches for accuracy calculation
        if idx >= warmup_matches:
            results.append({
                "match_id": row["match_id"],
                "date": row["date"],
                "season": row["season"],
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "home_goals": row["home_goals"],
                "away_goals": row["away_goals"],
                "actual_gd": actual_gd,
                "actual_outcome": actual_outcome,
                "pi_predicted_gd": predicted_gd,
                "pi_home_prob": home_prob,
                "pi_draw_prob": draw_prob,
                "pi_away_prob": away_prob,
                "pi_predicted_outcome": predicted_outcome,
                "pi_correct": predicted_outcome == actual_outcome,
                "pi_gd_error": predicted_gd - actual_gd,
                "pi_gd_sq_error": (predicted_gd - actual_gd) ** 2,
            })

    return pd.DataFrame(results)


def backtest_elo(
    matches_df: pd.DataFrame,
    k_factor: float = 28.0,
    home_advantage: float = 50.0,
    warmup_matches: int = 100,
) -> pd.DataFrame:
    """Run walk-forward backtest of ELO.

    Parameters
    ----------
    matches_df : pd.DataFrame
        Match data sorted chronologically.
    k_factor : float
        ELO K-factor.
    home_advantage : float
        ELO home advantage points.
    warmup_matches : int
        Number of initial matches for warmup.

    Returns
    -------
    pd.DataFrame
        Backtest results with predictions and actual outcomes.
    """
    from batch.models.elo import EloConfig

    config = EloConfig(k_factor=k_factor, home_advantage=home_advantage)
    elo = EloRatingSystem(config)

    # Create team_id mapping for ELO (uses numeric IDs)
    team_ids = {}
    next_id = 1

    results = []

    for idx, row in matches_df.iterrows():
        home_team = row["home_team"]
        away_team = row["away_team"]

        # Assign numeric IDs if needed
        if home_team not in team_ids:
            team_ids[home_team] = next_id
            next_id += 1
        if away_team not in team_ids:
            team_ids[away_team] = next_id
            next_id += 1

        home_id = team_ids[home_team]
        away_id = team_ids[away_team]

        # Get prediction BEFORE updating ratings
        home_prob, draw_prob, away_prob = elo.match_probabilities(home_id, away_id)

        # Calculate expected goal diff from ELO (rough approximation)
        home_rating = elo.get_rating(home_id)
        away_rating = elo.get_rating(away_id)
        rating_diff = (home_rating + config.home_advantage) - away_rating
        # Convert rating diff to expected GD (empirically ~0.003 per ELO point)
        predicted_gd = rating_diff * 0.003

        # Actual result
        actual_gd = row["home_goals"] - row["away_goals"]
        if actual_gd > 0:
            actual_outcome = "H"
        elif actual_gd < 0:
            actual_outcome = "A"
        else:
            actual_outcome = "D"

        # Predicted outcome
        probs = {"H": home_prob, "D": draw_prob, "A": away_prob}
        predicted_outcome = max(probs, key=probs.get)

        # Update ratings
        elo.update_ratings(home_id, away_id, row["home_goals"], row["away_goals"])

        # Skip warmup
        if idx >= warmup_matches:
            results.append({
                "match_id": row["match_id"],
                "elo_predicted_gd": predicted_gd,
                "elo_home_prob": home_prob,
                "elo_draw_prob": draw_prob,
                "elo_away_prob": away_prob,
                "elo_predicted_outcome": predicted_outcome,
                "elo_correct": predicted_outcome == actual_outcome,
                "elo_gd_error": predicted_gd - actual_gd,
                "elo_gd_sq_error": (predicted_gd - actual_gd) ** 2,
            })

    return pd.DataFrame(results)


def calculate_metrics(df: pd.DataFrame, prefix: str) -> dict:
    """Calculate accuracy metrics for a model.

    Parameters
    ----------
    df : pd.DataFrame
        Backtest results.
    prefix : str
        Column prefix (e.g., "pi_" or "elo_").

    Returns
    -------
    dict
        Dictionary of metrics.
    """
    correct_col = f"{prefix}correct"
    gd_error_col = f"{prefix}gd_error"
    gd_sq_error_col = f"{prefix}gd_sq_error"
    outcome_col = f"{prefix}predicted_outcome"
    home_prob_col = f"{prefix}home_prob"
    draw_prob_col = f"{prefix}draw_prob"
    away_prob_col = f"{prefix}away_prob"

    # Basic accuracy
    accuracy = df[correct_col].mean()

    # MSE and MAE for goal difference
    mse = df[gd_sq_error_col].mean()
    mae = df[gd_error_col].abs().mean()
    rmse = np.sqrt(mse)

    # Accuracy by outcome type
    home_matches = df[df["actual_outcome"] == "H"]
    draw_matches = df[df["actual_outcome"] == "D"]
    away_matches = df[df["actual_outcome"] == "A"]

    home_accuracy = home_matches[correct_col].mean() if len(home_matches) > 0 else 0
    draw_accuracy = draw_matches[correct_col].mean() if len(draw_matches) > 0 else 0
    away_accuracy = away_matches[correct_col].mean() if len(away_matches) > 0 else 0

    # Brier score for probabilities
    brier = 0
    for _, row in df.iterrows():
        actual = row["actual_outcome"]
        home_actual = 1 if actual == "H" else 0
        draw_actual = 1 if actual == "D" else 0
        away_actual = 1 if actual == "A" else 0

        brier += (row[home_prob_col] - home_actual) ** 2
        brier += (row[draw_prob_col] - draw_actual) ** 2
        brier += (row[away_prob_col] - away_actual) ** 2

    brier /= len(df)

    # Log loss (cross entropy)
    log_loss = 0
    epsilon = 1e-10
    for _, row in df.iterrows():
        actual = row["actual_outcome"]
        if actual == "H":
            log_loss -= np.log(max(row[home_prob_col], epsilon))
        elif actual == "D":
            log_loss -= np.log(max(row[draw_prob_col], epsilon))
        else:
            log_loss -= np.log(max(row[away_prob_col], epsilon))

    log_loss /= len(df)

    return {
        "accuracy": accuracy,
        "home_accuracy": home_accuracy,
        "draw_accuracy": draw_accuracy,
        "away_accuracy": away_accuracy,
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "brier_score": brier,
        "log_loss": log_loss,
    }


def compare_models(
    seasons: list[str] = None,
    warmup_matches: int = 200,
    pi_lambda: float = 0.06,
    pi_gamma: float = 0.6,
    elo_k: float = 28.0,
    elo_home_adv: float = 50.0,
) -> dict:
    """Compare Pi Rating and ELO models.

    Parameters
    ----------
    seasons : list[str], optional
        Seasons to include.
    warmup_matches : int
        Warmup period for both models.
    pi_lambda, pi_gamma : float
        Pi Rating parameters.
    elo_k, elo_home_adv : float
        ELO parameters.

    Returns
    -------
    dict
        Comparison results.
    """
    print("Loading matches from database...")
    with SyncSessionLocal() as session:
        matches_df = load_matches(session, seasons)

    print(f"Loaded {len(matches_df)} matches")

    if len(matches_df) < warmup_matches + 100:
        print("Not enough matches for meaningful backtest")
        return {}

    # Run backtests
    print(f"\nRunning Pi Rating backtest (λ={pi_lambda}, γ={pi_gamma})...")
    pi_results = backtest_pi_rating(matches_df, pi_lambda, pi_gamma, warmup_matches)

    print(f"Running ELO backtest (K={elo_k}, home_adv={elo_home_adv})...")
    elo_results = backtest_elo(matches_df, elo_k, elo_home_adv, warmup_matches)

    # Merge results
    combined = pi_results.merge(elo_results, on="match_id")

    print(f"\nAnalyzing {len(combined)} matches (after {warmup_matches} warmup)")

    # Calculate metrics
    pi_metrics = calculate_metrics(combined, "pi_")
    elo_metrics = calculate_metrics(combined, "elo_")

    # Compare
    print("\n" + "=" * 70)
    print("MODEL COMPARISON: Pi Rating vs ELO")
    print("=" * 70)

    metrics_to_compare = [
        ("Overall Accuracy", "accuracy", True),
        ("Home Win Accuracy", "home_accuracy", True),
        ("Draw Accuracy", "draw_accuracy", True),
        ("Away Win Accuracy", "away_accuracy", True),
        ("Goal Diff MSE", "mse", False),
        ("Goal Diff MAE", "mae", False),
        ("Goal Diff RMSE", "rmse", False),
        ("Brier Score", "brier_score", False),
        ("Log Loss", "log_loss", False),
    ]

    print(f"\n{'Metric':<25} {'Pi Rating':>12} {'ELO':>12} {'Difference':>12} {'Winner':>10}")
    print("-" * 70)

    for name, key, higher_better in metrics_to_compare:
        pi_val = pi_metrics[key]
        elo_val = elo_metrics[key]
        diff = pi_val - elo_val

        if higher_better:
            winner = "Pi" if diff > 0 else "ELO" if diff < 0 else "Tie"
        else:
            winner = "Pi" if diff < 0 else "ELO" if diff > 0 else "Tie"

        print(f"{name:<25} {pi_val:>12.4f} {elo_val:>12.4f} {diff:>+12.4f} {winner:>10}")

    # Season-by-season breakdown
    print("\n" + "=" * 70)
    print("SEASON-BY-SEASON ACCURACY")
    print("=" * 70)

    print(f"\n{'Season':<12} {'Matches':>8} {'Pi Rating':>12} {'ELO':>12} {'Winner':>10}")
    print("-" * 55)

    for season in sorted(combined["season"].unique()):
        season_data = combined[combined["season"] == season]
        pi_acc = season_data["pi_correct"].mean()
        elo_acc = season_data["elo_correct"].mean()
        winner = "Pi" if pi_acc > elo_acc else "ELO" if elo_acc > pi_acc else "Tie"
        print(f"{season:<12} {len(season_data):>8} {pi_acc:>12.1%} {elo_acc:>12.1%} {winner:>10}")

    # Analysis by match type
    print("\n" + "=" * 70)
    print("PERFORMANCE BY MATCH TYPE")
    print("=" * 70)

    # Close matches (small rating/predicted GD)
    close_matches = combined[combined["pi_predicted_gd"].abs() < 0.3]
    decisive_matches = combined[combined["pi_predicted_gd"].abs() >= 0.5]

    print(f"\nClose matches (|predicted GD| < 0.3): {len(close_matches)} matches")
    if len(close_matches) > 0:
        pi_close = close_matches["pi_correct"].mean()
        elo_close = close_matches["elo_correct"].mean()
        print(f"  Pi Rating: {pi_close:.1%}, ELO: {elo_close:.1%}")

    print(f"\nDecisive matches (|predicted GD| >= 0.5): {len(decisive_matches)} matches")
    if len(decisive_matches) > 0:
        pi_dec = decisive_matches["pi_correct"].mean()
        elo_dec = decisive_matches["elo_correct"].mean()
        print(f"  Pi Rating: {pi_dec:.1%}, ELO: {elo_dec:.1%}")

    # Home vs Away predictions
    print("\n" + "=" * 70)
    print("CALIBRATION CHECK")
    print("=" * 70)

    # Check how well predicted probabilities match actual outcomes
    prob_bins = [(0, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 1.0)]

    print(f"\nPi Rating - Home Win Calibration:")
    print(f"{'Predicted Prob':<20} {'Actual Win %':>15} {'Count':>10}")
    print("-" * 45)
    for low, high in prob_bins:
        mask = (combined["pi_home_prob"] >= low) & (combined["pi_home_prob"] < high)
        subset = combined[mask]
        if len(subset) > 10:
            actual = (subset["actual_outcome"] == "H").mean()
            print(f"{low:.0%} - {high:.0%}           {actual:>15.1%} {len(subset):>10}")

    return {
        "pi_metrics": pi_metrics,
        "elo_metrics": elo_metrics,
        "combined_results": combined,
        "match_count": len(combined),
    }


def optimize_pi_parameters(
    seasons: list[str] = None,
    warmup_matches: int = 200,
) -> dict:
    """Find optimal Pi Rating parameters.

    Tests combinations of lambda and gamma to minimize log loss.
    """
    print("Loading matches...")
    with SyncSessionLocal() as session:
        matches_df = load_matches(session, seasons)

    print(f"Loaded {len(matches_df)} matches")

    # Parameter grid
    lambdas = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
    gammas = [0.4, 0.5, 0.6, 0.7, 0.8]

    best_result = {"lambda": 0.06, "gamma": 0.6, "log_loss": float("inf")}
    results = []

    print("\nOptimizing parameters...")
    for lam in lambdas:
        for gam in gammas:
            pi_results = backtest_pi_rating(matches_df, lam, gam, warmup_matches)
            metrics = calculate_metrics(pi_results, "pi_")

            results.append({
                "lambda": lam,
                "gamma": gam,
                "accuracy": metrics["accuracy"],
                "log_loss": metrics["log_loss"],
                "brier": metrics["brier_score"],
            })

            if metrics["log_loss"] < best_result["log_loss"]:
                best_result = {
                    "lambda": lam,
                    "gamma": gam,
                    "log_loss": metrics["log_loss"],
                    "accuracy": metrics["accuracy"],
                }

    print(f"\nBest parameters: λ={best_result['lambda']}, γ={best_result['gamma']}")
    print(f"Log loss: {best_result['log_loss']:.4f}, Accuracy: {best_result['accuracy']:.1%}")

    return {"best": best_result, "all_results": pd.DataFrame(results)}


def test_ensemble(
    seasons: list[str] = None,
    warmup_matches: int = 200,
    pi_weight: float = 0.5,
) -> dict:
    """Test ensemble of Pi Rating and ELO.

    Combines probabilities from both models.
    """
    print("Loading matches...")
    with SyncSessionLocal() as session:
        matches_df = load_matches(session, seasons)

    print(f"Loaded {len(matches_df)} matches")

    # Run backtests
    print(f"\nRunning Pi Rating backtest...")
    pi_results = backtest_pi_rating(matches_df, 0.07, 0.7, warmup_matches)

    print(f"Running ELO backtest...")
    elo_results = backtest_elo(matches_df, 28.0, 50.0, warmup_matches)

    # Merge results
    combined = pi_results.merge(elo_results, on="match_id")

    # Create ensemble predictions
    elo_weight = 1 - pi_weight

    combined["ens_home_prob"] = pi_weight * combined["pi_home_prob"] + elo_weight * combined["elo_home_prob"]
    combined["ens_draw_prob"] = pi_weight * combined["pi_draw_prob"] + elo_weight * combined["elo_draw_prob"]
    combined["ens_away_prob"] = pi_weight * combined["pi_away_prob"] + elo_weight * combined["elo_away_prob"]

    # Normalize
    total = combined["ens_home_prob"] + combined["ens_draw_prob"] + combined["ens_away_prob"]
    combined["ens_home_prob"] /= total
    combined["ens_draw_prob"] /= total
    combined["ens_away_prob"] /= total

    # Predicted outcome
    def get_predicted(row):
        probs = {"H": row["ens_home_prob"], "D": row["ens_draw_prob"], "A": row["ens_away_prob"]}
        return max(probs, key=probs.get)

    combined["ens_predicted_outcome"] = combined.apply(get_predicted, axis=1)
    combined["ens_correct"] = combined["ens_predicted_outcome"] == combined["actual_outcome"]

    # Ensemble goal diff (average of both)
    combined["ens_gd_error"] = (combined["pi_gd_error"] + combined["elo_gd_error"]) / 2
    combined["ens_gd_sq_error"] = combined["ens_gd_error"] ** 2

    # Calculate metrics
    ens_metrics = calculate_metrics(combined, "ens_")
    pi_metrics = calculate_metrics(combined, "pi_")
    elo_metrics = calculate_metrics(combined, "elo_")

    print("\n" + "=" * 80)
    print(f"ENSEMBLE COMPARISON (Pi weight: {pi_weight:.0%}, ELO weight: {elo_weight:.0%})")
    print("=" * 80)

    print(f"\n{'Metric':<25} {'Pi Rating':>12} {'ELO':>12} {'Ensemble':>12} {'Best':>10}")
    print("-" * 80)

    metrics_list = [
        ("Overall Accuracy", "accuracy", True),
        ("Goal Diff MSE", "mse", False),
        ("Brier Score", "brier_score", False),
        ("Log Loss", "log_loss", False),
    ]

    for name, key, higher_better in metrics_list:
        pi_val = pi_metrics[key]
        elo_val = elo_metrics[key]
        ens_val = ens_metrics[key]

        vals = {"Pi": pi_val, "ELO": elo_val, "Ensemble": ens_val}
        if higher_better:
            best = max(vals, key=vals.get)
        else:
            best = min(vals, key=vals.get)

        print(f"{name:<25} {pi_val:>12.4f} {elo_val:>12.4f} {ens_val:>12.4f} {best:>10}")

    # Find optimal ensemble weight
    print("\n" + "=" * 80)
    print("OPTIMIZING ENSEMBLE WEIGHT")
    print("=" * 80)

    weights = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
    best_weight = 0.5
    best_log_loss = float("inf")

    print(f"\n{'Pi Weight':>12} {'Accuracy':>12} {'Log Loss':>12}")
    print("-" * 40)

    for w in weights:
        # Recalculate ensemble
        ew = 1 - w
        hp = w * combined["pi_home_prob"] + ew * combined["elo_home_prob"]
        dp = w * combined["pi_draw_prob"] + ew * combined["elo_draw_prob"]
        ap = w * combined["pi_away_prob"] + ew * combined["elo_away_prob"]
        total = hp + dp + ap
        hp, dp, ap = hp/total, dp/total, ap/total

        # Log loss
        log_loss = 0
        epsilon = 1e-10
        for idx, row in combined.iterrows():
            actual = row["actual_outcome"]
            if actual == "H":
                log_loss -= np.log(max(hp.iloc[idx - combined.index[0]], epsilon))
            elif actual == "D":
                log_loss -= np.log(max(dp.iloc[idx - combined.index[0]], epsilon))
            else:
                log_loss -= np.log(max(ap.iloc[idx - combined.index[0]], epsilon))
        log_loss /= len(combined)

        # Accuracy
        correct = 0
        for idx, row in combined.iterrows():
            i = idx - combined.index[0]
            probs = {"H": hp.iloc[i], "D": dp.iloc[i], "A": ap.iloc[i]}
            pred = max(probs, key=probs.get)
            if pred == row["actual_outcome"]:
                correct += 1
        acc = correct / len(combined)

        print(f"{w:>12.0%} {acc:>12.1%} {log_loss:>12.4f}")

        if log_loss < best_log_loss:
            best_log_loss = log_loss
            best_weight = w

    print(f"\nOptimal Pi weight: {best_weight:.0%} (Log loss: {best_log_loss:.4f})")

    return {
        "pi_metrics": pi_metrics,
        "elo_metrics": elo_metrics,
        "ensemble_metrics": ens_metrics,
        "optimal_pi_weight": best_weight,
    }


def main():
    """Main entry point."""
    import argparse
    import logging

    # Suppress SQLAlchemy logging
    logging.getLogger("sqlalchemy").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(description="Backtest Pi Rating vs ELO")
    parser.add_argument(
        "--seasons",
        nargs="*",
        help="Seasons to include",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=200,
        help="Warmup matches (default: 200)",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Optimize Pi Rating parameters",
    )
    parser.add_argument(
        "--ensemble",
        action="store_true",
        help="Test ensemble of Pi Rating and ELO",
    )

    args = parser.parse_args()

    if args.optimize:
        optimize_pi_parameters(args.seasons, args.warmup)
    elif args.ensemble:
        test_ensemble(args.seasons, args.warmup)
    else:
        compare_models(args.seasons, args.warmup)


if __name__ == "__main__":
    main()
