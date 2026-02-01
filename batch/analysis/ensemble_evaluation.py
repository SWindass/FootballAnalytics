"""Comprehensive ensemble evaluation.

Compares ensemble predictor against individual models using multiple
weighting strategies and evaluation metrics.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import select

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from app.db.database import SyncSessionLocal
from app.db.models import Match, Team
from batch.models.elo import EloConfig, EloRatingSystem
from batch.models.ensemble_predictor import (
    EnsemblePredictor,
    analyze_ensemble_vs_individuals,
    evaluate_ensemble,
    optimize_ensemble_weights,
)
from batch.models.pi_dixon_coles import PiDixonColesModel
from batch.models.pi_rating import PiRating


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


def generate_predictions(matches_df, warmup=500):
    """Generate predictions from all three models."""
    # Initialize models
    pidc = PiDixonColesModel(pi_lambda=0.07, pi_gamma=0.7, rho=-0.11)
    pi = PiRating(lambda_param=0.07, gamma_param=0.7)
    elo = EloRatingSystem(EloConfig(k_factor=28.0, home_advantage=50.0))

    team_ids = {}
    next_id = 1

    results = []

    for idx, row in matches_df.iterrows():
        # Actual outcome
        actual = "H" if row["home_goals"] > row["away_goals"] else (
            "A" if row["home_goals"] < row["away_goals"] else "D"
        )

        # --- Pi+DC Prediction ---
        pidc_pred = pidc.predict_match(
            row["home_team"], row["away_team"],
            apply_draw_model=pidc.draw_model_trained
        )

        # --- Pi Baseline Prediction ---
        pi_gd = pi.calculate_expected_goal_diff(row["home_team"], row["away_team"])
        import math
        pi_h = 1 / (1 + math.exp(-pi_gd * 0.7))
        pi_a = 1 / (1 + math.exp(pi_gd * 0.7))
        pi_d = max(0, 0.28 - 0.1 * abs(pi_gd))
        pi_total = pi_h + pi_d + pi_a
        pi_h, pi_d, pi_a = pi_h/pi_total, pi_d/pi_total, pi_a/pi_total

        # --- ELO Prediction ---
        if row["home_team"] not in team_ids:
            team_ids[row["home_team"]] = next_id
            next_id += 1
        if row["away_team"] not in team_ids:
            team_ids[row["away_team"]] = next_id
            next_id += 1

        elo_h, elo_d, elo_a = elo.match_probabilities(
            team_ids[row["home_team"]], team_ids[row["away_team"]]
        )

        # Skip warmup
        if idx >= warmup:
            results.append({
                "match_id": row["match_id"],
                "date": row["date"],
                "season": row["season"],
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "home_goals": row["home_goals"],
                "away_goals": row["away_goals"],
                "actual": actual,

                "pidc_home_prob": pidc_pred.home_win,
                "pidc_draw_prob": pidc_pred.draw,
                "pidc_away_prob": pidc_pred.away_win,
                "pidc_draw_conf": pidc_pred.draw_confidence,

                "pi_home_prob": pi_h,
                "pi_draw_prob": pi_d,
                "pi_away_prob": pi_a,

                "elo_home_prob": elo_h,
                "elo_draw_prob": elo_d,
                "elo_away_prob": elo_a,
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


def calculate_individual_metrics(df, prefix):
    """Calculate metrics for individual model."""
    def get_pred(row):
        h, d, a = row[f"{prefix}_home_prob"], row[f"{prefix}_draw_prob"], row[f"{prefix}_away_prob"]
        if h >= d and h >= a:
            return "H"
        elif d >= a:
            return "D"
        return "A"

    df[f"{prefix}_pred"] = df.apply(get_pred, axis=1)
    df[f"{prefix}_correct"] = df[f"{prefix}_pred"] == df["actual"]

    accuracy = df[f"{prefix}_correct"].mean()

    # Brier
    brier = 0
    for _, row in df.iterrows():
        h_act = 1 if row["actual"] == "H" else 0
        d_act = 1 if row["actual"] == "D" else 0
        a_act = 1 if row["actual"] == "A" else 0
        brier += (row[f"{prefix}_home_prob"] - h_act)**2
        brier += (row[f"{prefix}_draw_prob"] - d_act)**2
        brier += (row[f"{prefix}_away_prob"] - a_act)**2
    brier /= len(df)

    # Log loss
    log_loss = 0
    for _, row in df.iterrows():
        eps = 1e-10
        if row["actual"] == "H":
            log_loss -= np.log(max(row[f"{prefix}_home_prob"], eps))
        elif row["actual"] == "D":
            log_loss -= np.log(max(row[f"{prefix}_draw_prob"], eps))
        else:
            log_loss -= np.log(max(row[f"{prefix}_away_prob"], eps))
    log_loss /= len(df)

    # Draw specific
    draw_preds = df[df[f"{prefix}_pred"] == "D"]
    draw_actual = df[df["actual"] == "D"]
    draw_precision = (draw_preds["actual"] == "D").mean() if len(draw_preds) > 0 else 0
    draw_recall = (draw_actual[f"{prefix}_pred"] == "D").mean() if len(draw_actual) > 0 else 0

    return {
        "accuracy": accuracy,
        "brier_score": brier,
        "log_loss": log_loss,
        "draw_precision": draw_precision,
        "draw_recall": draw_recall,
        "draw_predictions": len(draw_preds),
    }


def print_confusion_matrix(confusion, title):
    """Print confusion matrix."""
    print(f"\n{title}")
    print("            Predicted")
    print("            H      D      A")
    print("Actual  +" + "-" * 24)
    for actual in ["H", "D", "A"]:
        row = confusion[actual]
        print(f"    {actual}   | {row['H']:>5} {row['D']:>5} {row['A']:>5}")


def main():
    import argparse
    import logging

    logging.getLogger("sqlalchemy").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", nargs="*")
    parser.add_argument("--warmup", type=int, default=500)
    args = parser.parse_args()

    print("Loading matches...")
    with SyncSessionLocal() as session:
        matches_df = load_matches(session, args.seasons)

    print(f"Loaded {len(matches_df)} matches")

    print("\nGenerating predictions from all models...")
    results_df = generate_predictions(matches_df, args.warmup)
    print(f"Generated predictions for {len(results_df)} matches")

    # Split into train/test for optimization
    split_idx = int(len(results_df) * 0.7)
    train_df = results_df.iloc[:split_idx].copy()
    test_df = results_df.iloc[split_idx:].copy()

    print(f"\nTrain set: {len(train_df)} matches")
    print(f"Test set: {len(test_df)} matches")

    # Calculate individual model metrics
    print("\n" + "=" * 80)
    print("INDIVIDUAL MODEL PERFORMANCE (Test Set)")
    print("=" * 80)

    pidc_metrics = calculate_individual_metrics(test_df.copy(), "pidc")
    pi_metrics = calculate_individual_metrics(test_df.copy(), "pi")
    elo_metrics = calculate_individual_metrics(test_df.copy(), "elo")

    print(f"\n{'Model':<15} {'Accuracy':>10} {'Brier':>10} {'Log Loss':>10} {'Draw Prec':>10} {'Draw Rec':>10}")
    print("-" * 75)
    print(f"{'Pi+DC':<15} {pidc_metrics['accuracy']:>10.1%} {pidc_metrics['brier_score']:>10.4f} "
          f"{pidc_metrics['log_loss']:>10.4f} {pidc_metrics['draw_precision']:>10.1%} {pidc_metrics['draw_recall']:>10.1%}")
    print(f"{'Pi Baseline':<15} {pi_metrics['accuracy']:>10.1%} {pi_metrics['brier_score']:>10.4f} "
          f"{pi_metrics['log_loss']:>10.4f} {pi_metrics['draw_precision']:>10.1%} {pi_metrics['draw_recall']:>10.1%}")
    print(f"{'ELO':<15} {elo_metrics['accuracy']:>10.1%} {elo_metrics['brier_score']:>10.4f} "
          f"{elo_metrics['log_loss']:>10.4f} {elo_metrics['draw_precision']:>10.1%} {elo_metrics['draw_recall']:>10.1%}")

    # Optimize ensemble weights on training set
    print("\n" + "=" * 80)
    print("OPTIMIZING ENSEMBLE WEIGHTS (on Training Set)")
    print("=" * 80)

    print("\nOptimizing for Brier Score...")
    brier_opt = optimize_ensemble_weights(train_df, metric="brier")
    print(f"Optimal weights: Pi+DC={brier_opt['pidc_weight']:.3f}, "
          f"Pi={brier_opt['pi_weight']:.3f}, ELO={brier_opt['elo_weight']:.3f}")

    print("\nOptimizing for Log Loss...")
    logloss_opt = optimize_ensemble_weights(train_df, metric="log_loss")
    print(f"Optimal weights: Pi+DC={logloss_opt['pidc_weight']:.3f}, "
          f"Pi={logloss_opt['pi_weight']:.3f}, ELO={logloss_opt['elo_weight']:.3f}")

    print("\nOptimizing for Accuracy...")
    acc_opt = optimize_ensemble_weights(train_df, metric="accuracy")
    print(f"Optimal weights: Pi+DC={acc_opt['pidc_weight']:.3f}, "
          f"Pi={acc_opt['pi_weight']:.3f}, ELO={acc_opt['elo_weight']:.3f}")

    # Evaluate different ensemble strategies on test set
    print("\n" + "=" * 80)
    print("ENSEMBLE STRATEGIES COMPARISON (Test Set)")
    print("=" * 80)

    strategies = [
        ("Equal Weights", (0.33, 0.33, 0.34), "fixed"),
        ("Brier Optimized", tuple(brier_opt['weights']), "fixed"),
        ("LogLoss Optimized", tuple(logloss_opt['weights']), "fixed"),
        ("Accuracy Optimized", tuple(acc_opt['weights']), "fixed"),
        ("Situational", (0.40, 0.30, 0.30), "situational"),
        ("Confidence-Based", (0.33, 0.33, 0.34), "confidence"),
    ]

    print(f"\n{'Strategy':<20} {'Accuracy':>10} {'Brier':>10} {'Log Loss':>10} {'Draw Prec':>10} {'Draw Rec':>10}")
    print("-" * 80)

    best_brier = float("inf")
    best_strategy = None
    best_results = None

    for name, weights, strategy in strategies:
        eval_result = evaluate_ensemble(test_df, weights, strategy)

        print(f"{name:<20} {eval_result['accuracy']:>10.1%} {eval_result['brier_score']:>10.4f} "
              f"{eval_result['log_loss']:>10.4f} {eval_result['draw_precision']:>10.1%} "
              f"{eval_result['draw_recall']:>10.1%}")

        if eval_result["brier_score"] < best_brier:
            best_brier = eval_result["brier_score"]
            best_strategy = (name, weights, strategy)
            best_results = eval_result

    print(f"\nBest Strategy: {best_strategy[0]} (Brier: {best_brier:.4f})")

    # Compare best ensemble vs individual models
    print("\n" + "=" * 80)
    print("BEST ENSEMBLE vs INDIVIDUAL MODELS")
    print("=" * 80)

    print(f"\n{'Metric':<20} {'Ensemble':>12} {'Pi+DC':>12} {'Pi Base':>12} {'ELO':>12} {'Best':>10}")
    print("-" * 80)

    metrics_compare = [
        ("Accuracy", best_results["accuracy"], pidc_metrics["accuracy"],
         pi_metrics["accuracy"], elo_metrics["accuracy"]),
        ("Brier Score", best_results["brier_score"], pidc_metrics["brier_score"],
         pi_metrics["brier_score"], elo_metrics["brier_score"]),
        ("Log Loss", best_results["log_loss"], pidc_metrics["log_loss"],
         pi_metrics["log_loss"], elo_metrics["log_loss"]),
        ("Draw Precision", best_results["draw_precision"], pidc_metrics["draw_precision"],
         pi_metrics["draw_precision"], elo_metrics["draw_precision"]),
        ("Draw Recall", best_results["draw_recall"], pidc_metrics["draw_recall"],
         pi_metrics["draw_recall"], elo_metrics["draw_recall"]),
    ]

    for name, ens, pidc, pi_val, elo_val in metrics_compare:
        vals = {"Ensemble": ens, "Pi+DC": pidc, "Pi Base": pi_val, "ELO": elo_val}
        if name in ["Brier Score", "Log Loss"]:
            best = min(vals, key=vals.get)
        else:
            best = max(vals, key=vals.get)

        print(f"{name:<20} {ens:>12.4f} {pidc:>12.4f} {pi_val:>12.4f} {elo_val:>12.4f} {best:>10}")

    # Confusion matrices
    print("\n" + "=" * 80)
    print("CONFUSION MATRICES")
    print("=" * 80)

    print_confusion_matrix(best_results["confusion_matrix"], "Best Ensemble")

    # Calculate individual confusion matrices
    for prefix, name in [("pidc", "Pi+DC"), ("pi", "Pi Baseline"), ("elo", "ELO")]:
        conf = {"H": {"H": 0, "D": 0, "A": 0},
                "D": {"H": 0, "D": 0, "A": 0},
                "A": {"H": 0, "D": 0, "A": 0}}

        for _, row in test_df.iterrows():
            h, d, a = row[f"{prefix}_home_prob"], row[f"{prefix}_draw_prob"], row[f"{prefix}_away_prob"]
            pred = "H" if h >= d and h >= a else ("D" if d >= a else "A")
            conf[row["actual"]][pred] += 1

        print_confusion_matrix(conf, name)

    # Analyze where ensemble beats/loses
    print("\n" + "=" * 80)
    print("ENSEMBLE VS INDIVIDUAL ANALYSIS")
    print("=" * 80)

    analysis = analyze_ensemble_vs_individuals(test_df, best_strategy[1])

    print("\nModel Agreement Analysis:")
    print(f"  All models agree & correct:     {analysis['all_agree_correct']:>6}")
    print(f"  All models agree & wrong:       {analysis['all_agree_wrong']:>6}")
    print(f"  Models disagree, ensemble right:{analysis['disagreement_ensemble_correct']:>6}")
    print(f"  Models disagree, ensemble wrong:{analysis['disagreement_ensemble_wrong']:>6}")

    print(f"\nEnsemble Unique Wins: {len(analysis['ensemble_only_correct'])}")
    if analysis['ensemble_only_correct']:
        print("  Examples (ensemble right, all individuals wrong):")
        for case in analysis['ensemble_only_correct'][:5]:
            print(f"    {case['match']}: Actual={case['actual']}, "
                  f"Ens={case['ens_pred']}, "
                  f"PiDC={case['pidc_pred']}, Pi={case['pi_pred']}, ELO={case['elo_pred']}")

    print(f"\nEnsemble Losses (someone was right): {len(analysis['ensemble_wrong_individual_correct'])}")
    if analysis['ensemble_wrong_individual_correct']:
        print("  Examples (ensemble wrong, at least one individual right):")
        for case in analysis['ensemble_wrong_individual_correct'][:5]:
            right_models = []
            if case['pidc_correct']:
                right_models.append("PiDC")
            if case['pi_correct']:
                right_models.append("Pi")
            if case['elo_correct']:
                right_models.append("ELO")
            print(f"    {case['match']}: Actual={case['actual']}, "
                  f"Ens={case['ens_pred']}, Right: {', '.join(right_models)}")

    # Calibration analysis
    print("\n" + "=" * 80)
    print("CALIBRATION ANALYSIS")
    print("=" * 80)

    # Combine predictions for best ensemble
    ensemble = EnsemblePredictor(weights=best_strategy[1], strategy=best_strategy[2])

    print("\nEnsemble Draw Calibration:")
    print(f"{'Predicted Range':<20} {'Actual Draw %':>15} {'Count':>10}")
    print("-" * 50)

    # Calculate ensemble draw probs
    ens_draw_probs = []
    for _, row in test_df.iterrows():
        pidc = (row["pidc_home_prob"], row["pidc_draw_prob"], row["pidc_away_prob"])
        pi = (row["pi_home_prob"], row["pi_draw_prob"], row["pi_away_prob"])
        elo = (row["elo_home_prob"], row["elo_draw_prob"], row["elo_away_prob"])
        pred = ensemble.combine_predictions(pidc, pi, elo)
        ens_draw_probs.append(pred.draw_prob)

    test_df["ens_draw_prob"] = ens_draw_probs

    bins = [(0.15, 0.20), (0.20, 0.25), (0.25, 0.30), (0.30, 0.35), (0.35, 0.40)]
    for low, high in bins:
        mask = (test_df["ens_draw_prob"] >= low) & (test_df["ens_draw_prob"] < high)
        subset = test_df[mask]
        if len(subset) >= 20:
            actual = (subset["actual"] == "D").mean()
            print(f"{low:.0%} - {high:.0%}           {actual:>15.1%} {len(subset):>10}")

    # Final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    print(f"\nBest Ensemble Strategy: {best_strategy[0]}")
    print(f"Weights: Pi+DC={best_strategy[1][0]:.1%}, Pi={best_strategy[1][1]:.1%}, ELO={best_strategy[1][2]:.1%}")

    # Calculate improvements
    ens_brier = best_results["brier_score"]
    best_individual_brier = min(pidc_metrics["brier_score"], pi_metrics["brier_score"], elo_metrics["brier_score"])
    brier_improvement = (best_individual_brier - ens_brier) / best_individual_brier * 100

    print(f"\nBrier Score Improvement vs Best Individual: {brier_improvement:+.2f}%")
    print(f"  Ensemble: {ens_brier:.4f}")
    print(f"  Best Individual (Pi+DC): {best_individual_brier:.4f}")

    if best_results["draw_predictions"] > 0:
        print("\nDraw Prediction Performance:")
        print(f"  Predictions: {best_results['draw_predictions']}")
        print(f"  Correct: {best_results['draw_correct']}")
        print(f"  Precision: {best_results['draw_precision']:.1%}")
        print(f"  Recall: {best_results['draw_recall']:.1%}")


if __name__ == "__main__":
    main()
