"""Evaluate FPL-enhanced ensemble predictor.

Combines:
- Current ensemble (Pi+DC, Pi, ELO)
- FPL-based predictions
With optimized weighting and confidence adjustment.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import select

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from app.db.database import SyncSessionLocal
from app.db.models import Match, Player, PlayerMatchPerformance, Team
from batch.models.elo import EloConfig, EloRatingSystem
from batch.models.fpl_predictor import FPLPredictor
from batch.models.pi_dixon_coles import PiDixonColesModel
from batch.models.pi_rating import PiRating


def load_fpl_data(session) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load FPL player and performance data."""
    # Load players
    players_query = select(
        Player.id,
        Player.name,
        Player.position,
        Player.team_id,
        Team.name.label("team_name"),
    ).outerjoin(Team, Player.team_id == Team.id)

    players = session.execute(players_query).all()
    players_df = pd.DataFrame([
        {"id": p.id, "name": p.name, "position": p.position,
         "team_id": p.team_id, "team_name": p.team_name}
        for p in players
    ])

    # Load performances
    perf_query = select(
        PlayerMatchPerformance.player_id,
        PlayerMatchPerformance.season,
        PlayerMatchPerformance.gameweek,
        PlayerMatchPerformance.total_points,
        PlayerMatchPerformance.goals_scored,
        PlayerMatchPerformance.assists,
        PlayerMatchPerformance.clean_sheets,
        PlayerMatchPerformance.minutes,
        PlayerMatchPerformance.influence,
        PlayerMatchPerformance.creativity,
        PlayerMatchPerformance.threat,
        PlayerMatchPerformance.ict_index,
        PlayerMatchPerformance.expected_goals,
        PlayerMatchPerformance.expected_assists,
    )

    performances = session.execute(perf_query).all()
    perf_df = pd.DataFrame([
        {
            "player_id": p.player_id,
            "season": p.season,
            "gameweek": p.gameweek,
            "total_points": p.total_points,
            "goals_scored": p.goals_scored,
            "assists": p.assists,
            "clean_sheets": p.clean_sheets,
            "minutes": p.minutes,
            "influence": float(p.influence) if p.influence else 0,
            "creativity": float(p.creativity) if p.creativity else 0,
            "threat": float(p.threat) if p.threat else 0,
            "ict_index": float(p.ict_index) if p.ict_index else 0,
            "expected_goals": float(p.expected_goals) if p.expected_goals else 0,
            "expected_assists": float(p.expected_assists) if p.expected_assists else 0,
        }
        for p in performances
    ])

    return players_df, perf_df


def load_matches(session, seasons=None):
    """Load match data."""
    query = (
        select(
            Match.id, Match.kickoff_time, Match.matchweek, Match.season,
            Match.home_team_id, Match.away_team_id, Match.home_score, Match.away_score,
        )
        .where(Match.status == "finished")
        .where(Match.home_score.isnot(None))
    )
    if seasons:
        query = query.where(Match.season.in_(seasons))
    query = query.order_by(Match.kickoff_time)

    results = session.execute(query).all()
    teams = {r.id: r.name for r in session.execute(select(Team.id, Team.name)).all()}

    data = []
    for r in results:
        data.append({
            "match_id": r.id,
            "date": r.kickoff_time,
            "season": r.season,
            "matchweek": r.matchweek,
            "home_team": teams.get(r.home_team_id, "Unknown"),
            "away_team": teams.get(r.away_team_id, "Unknown"),
            "home_goals": r.home_score,
            "away_goals": r.away_score,
        })

    return pd.DataFrame(data)


def generate_all_predictions(matches_df, fpl_predictor, warmup=300):
    """Generate predictions from all models including FPL."""
    # Initialize models
    pidc = PiDixonColesModel(pi_lambda=0.07, pi_gamma=0.7, rho=-0.11)
    pi = PiRating(lambda_param=0.07, gamma_param=0.7)
    elo = EloRatingSystem(EloConfig(k_factor=28.0, home_advantage=50.0))

    team_ids = {}
    next_id = 1

    results = []

    for idx, row in matches_df.iterrows():
        actual = "H" if row["home_goals"] > row["away_goals"] else (
            "A" if row["home_goals"] < row["away_goals"] else "D"
        )

        # --- Pi+DC Prediction ---
        pidc_pred = pidc.predict_match(
            row["home_team"], row["away_team"],
            apply_draw_model=pidc.draw_model_trained
        )

        # --- Pi Baseline ---
        pi_gd = pi.calculate_expected_goal_diff(row["home_team"], row["away_team"])
        import math
        pi_h = 1 / (1 + math.exp(-pi_gd * 0.7))
        pi_a = 1 / (1 + math.exp(pi_gd * 0.7))
        pi_d = max(0, 0.28 - 0.1 * abs(pi_gd))
        pi_total = pi_h + pi_d + pi_a
        pi_h, pi_d, pi_a = pi_h/pi_total, pi_d/pi_total, pi_a/pi_total

        # --- ELO ---
        if row["home_team"] not in team_ids:
            team_ids[row["home_team"]] = next_id
            next_id += 1
        if row["away_team"] not in team_ids:
            team_ids[row["away_team"]] = next_id
            next_id += 1

        elo_h, elo_d, elo_a = elo.match_probabilities(
            team_ids[row["home_team"]], team_ids[row["away_team"]]
        )

        # --- FPL Prediction ---
        fpl_pred = fpl_predictor.predict_match(
            row["home_team"], row["away_team"],
            row["season"], row["matchweek"]
        )

        # Skip warmup
        if idx >= warmup:
            results.append({
                "match_id": row["match_id"],
                "date": row["date"],
                "season": row["season"],
                "matchweek": row["matchweek"],
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "home_goals": row["home_goals"],
                "away_goals": row["away_goals"],
                "actual": actual,

                # Pi+DC
                "pidc_home": pidc_pred.home_win,
                "pidc_draw": pidc_pred.draw,
                "pidc_away": pidc_pred.away_win,
                "pidc_draw_conf": pidc_pred.draw_confidence,

                # Pi Baseline
                "pi_home": pi_h,
                "pi_draw": pi_d,
                "pi_away": pi_a,

                # ELO
                "elo_home": elo_h,
                "elo_draw": elo_d,
                "elo_away": elo_a,

                # FPL
                "fpl_home": fpl_pred.home_prob,
                "fpl_draw": fpl_pred.draw_prob,
                "fpl_away": fpl_pred.away_prob,
                "fpl_home_xg": fpl_pred.home_xg,
                "fpl_away_xg": fpl_pred.away_xg,
                "fpl_confidence": fpl_pred.confidence,
            })

        # Update models
        pidc.update_after_match(
            row["home_team"], row["away_team"],
            row["home_goals"], row["away_goals"],
            row["date"], collect_training_data=True
        )
        pi.update_ratings(
            row["home_team"], row["away_team"],
            row["home_goals"], row["away_goals"],
            row["date"], store_history=False
        )
        elo.update_ratings(
            team_ids[row["home_team"]], team_ids[row["away_team"]],
            row["home_goals"], row["away_goals"]
        )

        # Train draw model periodically
        if (idx + 1) % 200 == 0 and idx >= warmup:
            pidc.train_draw_model(min_samples=100)

    return pd.DataFrame(results)


def create_fpl_ensemble(
    row,
    base_weights=(0.575, 0.05, 0.375),
    fpl_weight=0.30,
    confidence_adjust=True,
):
    """Create FPL-enhanced ensemble prediction.

    final = (1 - fpl_weight) * base_ensemble + fpl_weight * fpl_pred
    With optional confidence adjustment.
    """
    # Base ensemble (Pi+DC, Pi, ELO)
    base_home = (
        base_weights[0] * row["pidc_home"] +
        base_weights[1] * row["pi_home"] +
        base_weights[2] * row["elo_home"]
    )
    base_draw = (
        base_weights[0] * row["pidc_draw"] +
        base_weights[1] * row["pi_draw"] +
        base_weights[2] * row["elo_draw"]
    )
    base_away = (
        base_weights[0] * row["pidc_away"] +
        base_weights[1] * row["pi_away"] +
        base_weights[2] * row["elo_away"]
    )

    # Normalize base
    base_total = base_home + base_draw + base_away
    base_home, base_draw, base_away = base_home/base_total, base_draw/base_total, base_away/base_total

    # FPL contribution
    fpl_home = row["fpl_home"]
    fpl_draw = row["fpl_draw"]
    fpl_away = row["fpl_away"]

    # Adjust FPL weight by confidence
    effective_fpl_weight = fpl_weight
    if confidence_adjust:
        effective_fpl_weight *= row["fpl_confidence"]

    # Combine
    base_weight = 1 - effective_fpl_weight
    final_home = base_weight * base_home + effective_fpl_weight * fpl_home
    final_draw = base_weight * base_draw + effective_fpl_weight * fpl_draw
    final_away = base_weight * base_away + effective_fpl_weight * fpl_away

    # Normalize
    total = final_home + final_draw + final_away
    return final_home/total, final_draw/total, final_away/total


def calculate_metrics(df, home_col, draw_col, away_col, name):
    """Calculate metrics for a model."""
    def get_pred(row):
        h, d, a = row[home_col], row[draw_col], row[away_col]
        # Aggressive draw prediction
        if d >= 0.26 and abs(h - a) < 0.08:
            return "D"
        if d >= 0.25 and d >= max(h, a) * 0.95:
            return "D"
        if h >= a:
            return "H"
        return "A"

    df[f"{name}_pred"] = df.apply(get_pred, axis=1)
    df[f"{name}_correct"] = df[f"{name}_pred"] == df["actual"]

    accuracy = df[f"{name}_correct"].mean()

    # Brier
    brier = 0
    for _, row in df.iterrows():
        h_act = 1 if row["actual"] == "H" else 0
        d_act = 1 if row["actual"] == "D" else 0
        a_act = 1 if row["actual"] == "A" else 0
        brier += (row[home_col] - h_act)**2 + (row[draw_col] - d_act)**2 + (row[away_col] - a_act)**2
    brier /= len(df)

    # Log loss
    log_loss = 0
    for _, row in df.iterrows():
        eps = 1e-10
        if row["actual"] == "H":
            log_loss -= np.log(max(row[home_col], eps))
        elif row["actual"] == "D":
            log_loss -= np.log(max(row[draw_col], eps))
        else:
            log_loss -= np.log(max(row[away_col], eps))
    log_loss /= len(df)

    # Draw metrics
    draw_preds = df[df[f"{name}_pred"] == "D"]
    draw_actual = df[df["actual"] == "D"]
    draw_precision = (draw_preds["actual"] == "D").mean() if len(draw_preds) > 0 else 0
    draw_recall = (draw_actual[f"{name}_pred"] == "D").mean() if len(draw_actual) > 0 else 0

    return {
        "accuracy": accuracy,
        "brier": brier,
        "log_loss": log_loss,
        "draw_precision": draw_precision,
        "draw_recall": draw_recall,
        "draw_preds": len(draw_preds),
    }


def optimize_fpl_weight(train_df, base_weights=(0.575, 0.05, 0.375)):
    """Find optimal FPL weight."""
    from scipy.optimize import minimize_scalar

    def objective(fpl_weight):
        total_brier = 0
        for _, row in train_df.iterrows():
            h, d, a = create_fpl_ensemble(row, base_weights, fpl_weight, confidence_adjust=True)

            h_act = 1 if row["actual"] == "H" else 0
            d_act = 1 if row["actual"] == "D" else 0
            a_act = 1 if row["actual"] == "A" else 0

            total_brier += (h - h_act)**2 + (d - d_act)**2 + (a - a_act)**2

        return total_brier / len(train_df)

    result = minimize_scalar(objective, bounds=(0.0, 0.5), method="bounded")
    return result.x


def main():
    import argparse
    import logging

    logging.getLogger("sqlalchemy").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=200)
    args = parser.parse_args()

    print("Loading data...")
    with SyncSessionLocal() as session:
        # Load FPL data
        print("  Loading FPL player data...")
        players_df, perf_df = load_fpl_data(session)
        print(f"  Loaded {len(players_df)} players, {len(perf_df)} performances")

        # Load matches - only seasons with FPL data
        fpl_seasons = sorted(perf_df["season"].unique())
        print(f"  FPL data available for: {', '.join(fpl_seasons)}")

        print("  Loading matches...")
        matches_df = load_matches(session, seasons=fpl_seasons)
        print(f"  Loaded {len(matches_df)} matches")

    # Initialize FPL predictor
    print("\nInitializing FPL predictor...")
    fpl = FPLPredictor(form_window=5, rho=-0.11)
    fpl.load_fpl_data(perf_df, players_df)

    # Generate predictions
    print("\nGenerating predictions from all models...")
    results_df = generate_all_predictions(matches_df, fpl, args.warmup)
    print(f"Generated predictions for {len(results_df)} matches")

    # Filter to matches with FPL confidence > 0
    fpl_matches = results_df[results_df["fpl_confidence"] > 0].copy()
    print(f"Matches with FPL data: {len(fpl_matches)}")

    if len(fpl_matches) < 100:
        print("Not enough FPL data for meaningful analysis")
        return

    # Split train/test
    split_idx = int(len(fpl_matches) * 0.7)
    train_df = fpl_matches.iloc[:split_idx].copy()
    test_df = fpl_matches.iloc[split_idx:].copy()

    print(f"\nTrain: {len(train_df)}, Test: {len(test_df)}")

    # Optimize FPL weight
    print("\nOptimizing FPL weight...")
    optimal_fpl_weight = optimize_fpl_weight(train_df)
    print(f"Optimal FPL weight: {optimal_fpl_weight:.1%}")

    # Create ensemble predictions for test set
    print("\nEvaluating on test set...")

    # Add ensemble columns
    for idx, row in test_df.iterrows():
        # Base ensemble (no FPL)
        h, d, a = create_fpl_ensemble(row, fpl_weight=0.0)
        test_df.loc[idx, "base_home"] = h
        test_df.loc[idx, "base_draw"] = d
        test_df.loc[idx, "base_away"] = a

        # FPL ensemble (optimal weight)
        h, d, a = create_fpl_ensemble(row, fpl_weight=optimal_fpl_weight, confidence_adjust=True)
        test_df.loc[idx, "fpl_ens_home"] = h
        test_df.loc[idx, "fpl_ens_draw"] = d
        test_df.loc[idx, "fpl_ens_away"] = a

        # Fixed 30% FPL weight
        h, d, a = create_fpl_ensemble(row, fpl_weight=0.30, confidence_adjust=True)
        test_df.loc[idx, "fpl30_home"] = h
        test_df.loc[idx, "fpl30_draw"] = d
        test_df.loc[idx, "fpl30_away"] = a

    # Calculate metrics for all models
    print("\n" + "=" * 90)
    print("MODEL COMPARISON (Test Set)")
    print("=" * 90)

    models = [
        ("Pi+DC", "pidc_home", "pidc_draw", "pidc_away"),
        ("Pi Baseline", "pi_home", "pi_draw", "pi_away"),
        ("ELO", "elo_home", "elo_draw", "elo_away"),
        ("FPL Only", "fpl_home", "fpl_draw", "fpl_away"),
        ("Base Ensemble", "base_home", "base_draw", "base_away"),
        (f"FPL Ensemble ({optimal_fpl_weight:.0%})", "fpl_ens_home", "fpl_ens_draw", "fpl_ens_away"),
        ("FPL Ensemble (30%)", "fpl30_home", "fpl30_draw", "fpl30_away"),
    ]

    print(f"\n{'Model':<25} {'Accuracy':>10} {'Brier':>10} {'Log Loss':>10} {'Draw Prec':>10} {'Draw Rec':>10}")
    print("-" * 85)

    all_metrics = {}
    for name, h_col, d_col, a_col in models:
        metrics = calculate_metrics(test_df.copy(), h_col, d_col, a_col, name.replace(" ", "_").replace("(", "").replace(")", "").replace("%", ""))
        all_metrics[name] = metrics

        print(f"{name:<25} {metrics['accuracy']:>10.1%} {metrics['brier']:>10.4f} "
              f"{metrics['log_loss']:>10.4f} {metrics['draw_precision']:>10.1%} {metrics['draw_recall']:>10.1%}")

    # Compare FPL ensemble vs base
    print("\n" + "=" * 90)
    print("FPL ENHANCEMENT ANALYSIS")
    print("=" * 90)

    base_metrics = all_metrics["Base Ensemble"]
    fpl_metrics = all_metrics[f"FPL Ensemble ({optimal_fpl_weight:.0%})"]

    print(f"\nBase Ensemble vs FPL Ensemble (optimal weight {optimal_fpl_weight:.0%}):")
    print(f"  Accuracy:      {base_metrics['accuracy']:.1%} → {fpl_metrics['accuracy']:.1%} "
          f"({(fpl_metrics['accuracy'] - base_metrics['accuracy'])*100:+.2f}pp)")
    print(f"  Brier Score:   {base_metrics['brier']:.4f} → {fpl_metrics['brier']:.4f} "
          f"({(fpl_metrics['brier'] - base_metrics['brier'])*100:+.2f}%)")
    print(f"  Draw Precision:{base_metrics['draw_precision']:.1%} → {fpl_metrics['draw_precision']:.1%}")
    print(f"  Draw Recall:   {base_metrics['draw_recall']:.1%} → {fpl_metrics['draw_recall']:.1%}")

    # FPL prediction quality
    print("\n" + "=" * 90)
    print("FPL MODEL ANALYSIS")
    print("=" * 90)

    fpl_only = all_metrics["FPL Only"]
    print("\nFPL-only model performance:")
    print(f"  Accuracy:    {fpl_only['accuracy']:.1%}")
    print(f"  Brier Score: {fpl_only['brier']:.4f}")
    print(f"  Log Loss:    {fpl_only['log_loss']:.4f}")

    # Compare FPL xG vs actual goals
    print("\nFPL xG vs Actual Goals:")
    test_df["total_xg"] = test_df["fpl_home_xg"] + test_df["fpl_away_xg"]
    test_df["total_goals"] = test_df["home_goals"] + test_df["away_goals"]

    xg_corr = test_df["total_xg"].corr(test_df["total_goals"])
    print(f"  Total xG correlation: {xg_corr:.3f}")

    home_xg_corr = test_df["fpl_home_xg"].corr(test_df["home_goals"])
    away_xg_corr = test_df["fpl_away_xg"].corr(test_df["away_goals"])
    print(f"  Home xG correlation:  {home_xg_corr:.3f}")
    print(f"  Away xG correlation:  {away_xg_corr:.3f}")

    # Season breakdown
    print("\n" + "=" * 90)
    print("SEASON BREAKDOWN")
    print("=" * 90)

    print(f"\n{'Season':<12} {'Matches':>8} {'Base Acc':>10} {'FPL Ens Acc':>12} {'Improvement':>12}")
    print("-" * 60)

    for season in sorted(test_df["season"].unique()):
        season_df = test_df[test_df["season"] == season].copy()

        if len(season_df) < 20:
            continue

        base_m = calculate_metrics(season_df.copy(), "base_home", "base_draw", "base_away", "base")
        fpl_m = calculate_metrics(season_df.copy(), "fpl_ens_home", "fpl_ens_draw", "fpl_ens_away", "fpl")

        improvement = (fpl_m["accuracy"] - base_m["accuracy"]) * 100

        print(f"{season:<12} {len(season_df):>8} {base_m['accuracy']:>10.1%} "
              f"{fpl_m['accuracy']:>12.1%} {improvement:>+12.2f}pp")

    # Final summary
    print("\n" + "=" * 90)
    print("FINAL SUMMARY")
    print("=" * 90)

    # Find best model
    best_brier = min(all_metrics.items(), key=lambda x: x[1]["brier"])
    best_acc = max(all_metrics.items(), key=lambda x: x[1]["accuracy"])

    print(f"\nBest Brier Score:  {best_brier[0]} ({best_brier[1]['brier']:.4f})")
    print(f"Best Accuracy:     {best_acc[0]} ({best_acc[1]['accuracy']:.1%})")

    improvement_brier = (base_metrics["brier"] - fpl_metrics["brier"]) / base_metrics["brier"] * 100
    improvement_acc = (fpl_metrics["accuracy"] - base_metrics["accuracy"]) * 100

    print("\nFPL Enhancement Impact:")
    print(f"  Brier Score Improvement: {improvement_brier:+.2f}%")
    print(f"  Accuracy Improvement:    {improvement_acc:+.2f}pp")
    print(f"  Optimal FPL Weight:      {optimal_fpl_weight:.1%}")


if __name__ == "__main__":
    main()
