"""Backtest Pi + Dixon-Coles model with draw enhancement.

Evaluates:
1. Does enhanced draw prediction improve accuracy?
2. Does it beat bookmaker draw odds?
3. Calibration curves
4. Brier score improvements
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import select

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from app.db.database import SyncSessionLocal
from app.db.models import Match, OddsHistory, Team
from batch.models.elo import EloConfig, EloRatingSystem
from batch.models.pi_dixon_coles import (
    PiDixonColesModel,
)
from batch.models.pi_rating import PiRating


def load_matches_with_odds(
    session,
    seasons: list[str] = None,
) -> pd.DataFrame:
    """Load matches with bookmaker odds."""

    # Get matches
    match_query = (
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
    )

    if seasons:
        match_query = match_query.where(Match.season.in_(seasons))

    match_query = match_query.order_by(Match.kickoff_time)
    matches = session.execute(match_query).all()

    # Get team names
    teams = {r.id: r.name for r in session.execute(select(Team.id, Team.name)).all()}

    # Get odds
    odds_query = select(
        OddsHistory.match_id,
        OddsHistory.home_odds,
        OddsHistory.draw_odds,
        OddsHistory.away_odds,
    )
    odds_results = session.execute(odds_query).all()
    odds_map = {r.match_id: (r.home_odds, r.draw_odds, r.away_odds) for r in odds_results}

    data = []
    for m in matches:
        odds = odds_map.get(m.id)

        # Convert odds to implied probabilities (with margin removal)
        if odds and all(o and float(o) > 1 for o in odds):
            h_odds, d_odds, a_odds = [float(o) for o in odds]
            h_impl = 1 / h_odds
            d_impl = 1 / d_odds
            a_impl = 1 / a_odds
            total = h_impl + d_impl + a_impl

            # Remove margin (normalize)
            h_impl /= total
            d_impl /= total
            a_impl /= total
        else:
            h_impl, d_impl, a_impl = None, None, None
            h_odds, d_odds, a_odds = None, None, None

        data.append({
            "match_id": m.id,
            "date": m.kickoff_time,
            "season": m.season,
            "matchweek": m.matchweek,
            "home_team": teams.get(m.home_team_id, "Unknown"),
            "away_team": teams.get(m.away_team_id, "Unknown"),
            "home_goals": m.home_score,
            "away_goals": m.away_score,
            "home_odds": h_odds,
            "draw_odds": d_odds,
            "away_odds": a_odds,
            "market_home_prob": h_impl,
            "market_draw_prob": d_impl,
            "market_away_prob": a_impl,
        })

    return pd.DataFrame(data)


def backtest_model(
    matches_df: pd.DataFrame,
    warmup_matches: int = 300,
    train_draw_every: int = 200,
    optimize_rho: bool = True,
) -> dict:
    """Run walk-forward backtest of Pi + Dixon-Coles model.

    Parameters
    ----------
    matches_df : pd.DataFrame
        Match data with optional odds.
    warmup_matches : int
        Initial warmup period.
    train_draw_every : int
        Retrain draw model every N matches.
    optimize_rho : bool
        Whether to optimize Dixon-Coles rho parameter.

    Returns
    -------
    dict
        Backtest results.
    """
    # Initialize models
    pi_dc = PiDixonColesModel(
        pi_lambda=0.07,
        pi_gamma=0.7,
        rho=-0.13,
    )

    # Also track baseline Pi Rating for comparison
    pi_baseline = PiRating(lambda_param=0.07, gamma_param=0.7)

    # ELO baseline
    elo = EloRatingSystem(EloConfig(k_factor=28.0, home_advantage=50.0))
    team_ids = {}
    next_id = 1

    results = []

    # Optimize rho on first portion of data
    if optimize_rho and len(matches_df) > warmup_matches:
        warmup_data = matches_df.iloc[:warmup_matches].copy()
        # Process warmup through Pi Rating first
        for _, row in warmup_data.iterrows():
            pi_dc.pi_rating.update_ratings(
                row["home_team"], row["away_team"],
                row["home_goals"], row["away_goals"],
                row["date"], store_history=False
            )

        # Now optimize rho
        print("Optimizing Dixon-Coles rho parameter...")
        optimal_rho = pi_dc.optimize_rho(warmup_data)
        print(f"Optimal rho: {optimal_rho:.4f}")
        pi_dc.rho = optimal_rho

        # Reset for fresh start
        pi_dc.reset()

    for idx, row in matches_df.iterrows():
        # Actual outcome
        if row["home_goals"] > row["away_goals"]:
            actual = "H"
        elif row["home_goals"] < row["away_goals"]:
            actual = "A"
        else:
            actual = "D"

        actual_gd = row["home_goals"] - row["away_goals"]

        # ---- Pi + Dixon-Coles Prediction ----
        pi_dc_pred = pi_dc.predict_match(
            row["home_team"], row["away_team"],
            apply_draw_model=pi_dc.draw_model_trained
        )

        # ---- Baseline Pi Rating Prediction ----
        pi_gd = pi_baseline.calculate_expected_goal_diff(row["home_team"], row["away_team"])
        import math
        pi_h = 1 / (1 + math.exp(-pi_gd * 0.7))
        pi_a = 1 / (1 + math.exp(pi_gd * 0.7))
        pi_d = max(0, 0.28 - 0.1 * abs(pi_gd))
        pi_total = pi_h + pi_d + pi_a
        pi_h, pi_d, pi_a = pi_h/pi_total, pi_d/pi_total, pi_a/pi_total

        # ---- ELO Prediction ----
        if row["home_team"] not in team_ids:
            team_ids[row["home_team"]] = next_id
            next_id += 1
        if row["away_team"] not in team_ids:
            team_ids[row["away_team"]] = next_id
            next_id += 1

        elo_h, elo_d, elo_a = elo.match_probabilities(
            team_ids[row["home_team"]],
            team_ids[row["away_team"]]
        )

        # Skip warmup for results
        if idx >= warmup_matches:
            result = {
                "match_id": row["match_id"],
                "date": row["date"],
                "season": row["season"],
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "home_goals": row["home_goals"],
                "away_goals": row["away_goals"],
                "actual": actual,
                "actual_gd": actual_gd,

                # Pi + Dixon-Coles
                "pidc_home_prob": pi_dc_pred.home_win,
                "pidc_draw_prob": pi_dc_pred.draw,
                "pidc_away_prob": pi_dc_pred.away_win,
                "pidc_raw_draw": pi_dc_pred.dixon_coles_draw,
                "pidc_draw_mult": pi_dc_pred.draw_multiplier,
                "pidc_draw_conf": pi_dc_pred.draw_confidence,

                # Baseline Pi
                "pi_home_prob": pi_h,
                "pi_draw_prob": pi_d,
                "pi_away_prob": pi_a,

                # ELO
                "elo_home_prob": elo_h,
                "elo_draw_prob": elo_d,
                "elo_away_prob": elo_a,

                # Market (if available)
                "market_home_prob": row["market_home_prob"],
                "market_draw_prob": row["market_draw_prob"],
                "market_away_prob": row["market_away_prob"],
                "draw_odds": row["draw_odds"],
            }
            results.append(result)

        # Update models
        pi_dc.update_after_match(
            row["home_team"], row["away_team"],
            row["home_goals"], row["away_goals"],
            row["date"], collect_training_data=True
        )

        pi_baseline.update_ratings(
            row["home_team"], row["away_team"],
            row["home_goals"], row["away_goals"],
            row["date"], store_history=False
        )

        elo.update_ratings(
            team_ids[row["home_team"]],
            team_ids[row["away_team"]],
            row["home_goals"], row["away_goals"]
        )

        # Periodically train draw model
        if (idx + 1) % train_draw_every == 0 and idx >= warmup_matches:
            train_result = pi_dc.train_draw_model(min_samples=100)
            if train_result["status"] == "trained":
                print(f"Match {idx+1}: Draw model retrained "
                      f"(accuracy={train_result['accuracy']:.1%}, "
                      f"recall={train_result['draw_recall']:.1%})")

    return {
        "results": pd.DataFrame(results),
        "model": pi_dc,
        "rho": pi_dc.rho,
    }


def calculate_metrics(results_df: pd.DataFrame, prefix: str, use_draw_threshold: bool = False) -> dict:
    """Calculate prediction metrics for a model."""
    h_col = f"{prefix}_home_prob"
    d_col = f"{prefix}_draw_prob"
    a_col = f"{prefix}_away_prob"

    # Predicted outcome
    def get_pred(row):
        probs = {"H": row[h_col], "D": row[d_col], "A": row[a_col]}

        # For Pi+DC, use special draw threshold logic
        if use_draw_threshold and prefix == "pidc":
            draw_conf = row.get("pidc_draw_conf", 0)
            draw_prob = row[d_col]

            # Predict draw if confidence is high and probability is reasonable
            if draw_prob >= 0.26 and draw_conf > 0.55:
                return "D"

            # Also predict draw if very close to max
            max_prob = max(probs.values())
            if draw_prob > max_prob * 0.90 and draw_prob >= 0.25:
                return "D"

        return max(probs, key=probs.get)

    results_df[f"{prefix}_pred"] = results_df.apply(get_pred, axis=1)
    results_df[f"{prefix}_correct"] = results_df[f"{prefix}_pred"] == results_df["actual"]

    # Overall accuracy
    accuracy = results_df[f"{prefix}_correct"].mean()

    # Accuracy by outcome
    home_acc = results_df[results_df["actual"] == "H"][f"{prefix}_correct"].mean()
    draw_acc = results_df[results_df["actual"] == "D"][f"{prefix}_correct"].mean()
    away_acc = results_df[results_df["actual"] == "A"][f"{prefix}_correct"].mean()

    # Brier score (lower is better)
    brier = 0
    for _, row in results_df.iterrows():
        h_act = 1 if row["actual"] == "H" else 0
        d_act = 1 if row["actual"] == "D" else 0
        a_act = 1 if row["actual"] == "A" else 0

        brier += (row[h_col] - h_act) ** 2
        brier += (row[d_col] - d_act) ** 2
        brier += (row[a_col] - a_act) ** 2

    brier /= len(results_df)

    # Log loss
    log_loss = 0
    epsilon = 1e-10
    for _, row in results_df.iterrows():
        if row["actual"] == "H":
            log_loss -= np.log(max(row[h_col], epsilon))
        elif row["actual"] == "D":
            log_loss -= np.log(max(row[d_col], epsilon))
        else:
            log_loss -= np.log(max(row[a_col], epsilon))
    log_loss /= len(results_df)

    # Draw-specific Brier
    draw_brier = np.mean([
        (row[d_col] - (1 if row["actual"] == "D" else 0)) ** 2
        for _, row in results_df.iterrows()
    ])

    return {
        "accuracy": accuracy,
        "home_accuracy": home_acc,
        "draw_accuracy": draw_acc,
        "away_accuracy": away_acc,
        "brier_score": brier,
        "log_loss": log_loss,
        "draw_brier": draw_brier,
    }


def analyze_draw_betting(results_df: pd.DataFrame) -> dict:
    """Analyze if model can beat bookmaker on draw bets."""
    # Filter to matches with odds
    with_odds = results_df[results_df["draw_odds"].notna()].copy()

    if len(with_odds) == 0:
        return {"status": "no_odds_data"}

    analysis = {
        "total_matches": len(with_odds),
        "actual_draws": (with_odds["actual"] == "D").sum(),
        "draw_rate": (with_odds["actual"] == "D").mean(),
    }

    # Find value bets: model draw prob > implied market prob
    with_odds["pidc_edge"] = with_odds["pidc_draw_prob"] - with_odds["market_draw_prob"]
    with_odds["pi_edge"] = with_odds["pi_draw_prob"] - with_odds["market_draw_prob"]
    with_odds["elo_edge"] = with_odds["elo_draw_prob"] - with_odds["market_draw_prob"]

    # Betting simulation for different edge thresholds
    for model_name, edge_col in [("pidc", "pidc_edge"), ("pi", "pi_edge"), ("elo", "elo_edge")]:
        model_results = []

        for min_edge in [0.02, 0.03, 0.05, 0.07, 0.10]:
            value_bets = with_odds[with_odds[edge_col] >= min_edge]

            if len(value_bets) < 10:
                continue

            wins = (value_bets["actual"] == "D").sum()
            total = len(value_bets)
            win_rate = wins / total

            # Calculate ROI (flat staking)
            returns = sum(
                float(row["draw_odds"]) - 1 if row["actual"] == "D" else -1
                for _, row in value_bets.iterrows()
            )
            roi = returns / total if total > 0 else 0

            model_results.append({
                "min_edge": min_edge,
                "bets": total,
                "wins": wins,
                "win_rate": win_rate,
                "roi": roi,
                "avg_odds": value_bets["draw_odds"].astype(float).mean(),
            })

        analysis[f"{model_name}_betting"] = model_results

    return analysis


def print_calibration_table(results_df: pd.DataFrame, prefix: str, n_bins: int = 5):
    """Print calibration table for draw predictions."""
    d_col = f"{prefix}_draw_prob"

    # Create bins
    results_df["_bin"] = pd.cut(results_df[d_col], bins=n_bins)

    print(f"\n{prefix.upper()} Draw Calibration:")
    print(f"{'Predicted Range':<20} {'Actual Draw %':>15} {'Count':>10}")
    print("-" * 50)

    for bin_label in sorted(results_df["_bin"].unique()):
        bin_data = results_df[results_df["_bin"] == bin_label]
        if len(bin_data) >= 10:
            actual_rate = (bin_data["actual"] == "D").mean()
            print(f"{str(bin_label):<20} {actual_rate:>15.1%} {len(bin_data):>10}")

    results_df.drop("_bin", axis=1, inplace=True)


def main():
    import argparse
    import logging

    logging.getLogger("sqlalchemy").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", nargs="*")
    parser.add_argument("--warmup", type=int, default=300)
    parser.add_argument("--no-optimize-rho", action="store_true")

    args = parser.parse_args()

    print("Loading matches with odds...")
    with SyncSessionLocal() as session:
        matches_df = load_matches_with_odds(session, args.seasons)

    print(f"Loaded {len(matches_df)} matches")

    has_odds = matches_df["draw_odds"].notna().sum()
    print(f"Matches with odds data: {has_odds}")

    # Run backtest
    print("\nRunning backtest...")
    backtest = backtest_model(
        matches_df,
        warmup_matches=args.warmup,
        optimize_rho=not args.no_optimize_rho,
    )

    results_df = backtest["results"]
    print(f"\nAnalyzing {len(results_df)} matches (after {args.warmup} warmup)")

    # Calculate metrics for each model
    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)

    pidc_metrics = calculate_metrics(results_df, "pidc", use_draw_threshold=True)
    pi_metrics = calculate_metrics(results_df, "pi")
    elo_metrics = calculate_metrics(results_df, "elo")

    # Market baseline (if available)
    market_metrics = None
    if results_df["market_draw_prob"].notna().any():
        market_df = results_df[results_df["market_draw_prob"].notna()].copy()
        market_metrics = calculate_metrics(market_df, "market")

    print(f"\n{'Metric':<20} {'Pi+DC':>10} {'Pi Base':>10} {'ELO':>10}", end="")
    if market_metrics:
        print(f" {'Market':>10}", end="")
    print(f" {'Best':>10}")
    print("-" * 80)

    metrics_to_show = [
        ("Accuracy", "accuracy", True),
        ("Draw Accuracy", "draw_accuracy", True),
        ("Brier Score", "brier_score", False),
        ("Log Loss", "log_loss", False),
        ("Draw Brier", "draw_brier", False),
    ]

    for name, key, higher_better in metrics_to_show:
        vals = {
            "Pi+DC": pidc_metrics[key],
            "Pi Base": pi_metrics[key],
            "ELO": elo_metrics[key],
        }
        if market_metrics:
            vals["Market"] = market_metrics[key]

        if higher_better:
            best = max(vals, key=vals.get)
        else:
            best = min(vals, key=vals.get)

        print(f"{name:<20} {vals['Pi+DC']:>10.4f} {vals['Pi Base']:>10.4f} {vals['ELO']:>10.4f}", end="")
        if market_metrics:
            print(f" {vals['Market']:>10.4f}", end="")
        print(f" {best:>10}")

    # Draw-specific analysis
    print("\n" + "=" * 80)
    print("DRAW PREDICTION ANALYSIS")
    print("=" * 80)

    total_draws = (results_df["actual"] == "D").sum()
    total_matches = len(results_df)
    print(f"\nActual draws: {total_draws}/{total_matches} ({total_draws/total_matches:.1%})")

    # Compare predicted draw rates
    print("\nAverage predicted draw probability:")
    print(f"  Pi+DC:    {results_df['pidc_draw_prob'].mean():.1%}")
    print(f"  Pi Base:  {results_df['pi_draw_prob'].mean():.1%}")
    print(f"  ELO:      {results_df['elo_draw_prob'].mean():.1%}")
    if results_df["market_draw_prob"].notna().any():
        print(f"  Market:   {results_df['market_draw_prob'].mean():.1%}")

    # Calibration tables
    print_calibration_table(results_df, "pidc")
    print_calibration_table(results_df, "pi")
    print_calibration_table(results_df, "elo")

    # Betting analysis
    print("\n" + "=" * 80)
    print("DRAW BETTING ANALYSIS")
    print("=" * 80)

    betting_analysis = analyze_draw_betting(results_df)

    if betting_analysis.get("status") == "no_odds_data":
        print("\nNo odds data available for betting analysis")
    else:
        print(f"\nMatches with odds: {betting_analysis['total_matches']}")
        print(f"Actual draw rate: {betting_analysis['draw_rate']:.1%}")

        for model_name in ["pidc", "pi", "elo"]:
            betting_results = betting_analysis.get(f"{model_name}_betting", [])
            if betting_results:
                print(f"\n{model_name.upper()} Value Betting (model > market):")
                print(f"{'Min Edge':>10} {'Bets':>8} {'Win%':>8} {'ROI':>10} {'Avg Odds':>10}")
                print("-" * 50)
                for r in betting_results:
                    print(f"{r['min_edge']:>10.0%} {r['bets']:>8} {r['win_rate']:>8.1%} "
                          f"{r['roi']:>+10.1%} {r['avg_odds']:>10.2f}")

    # Season breakdown
    print("\n" + "=" * 80)
    print("SEASON BREAKDOWN - DRAW ACCURACY")
    print("=" * 80)

    print(f"\n{'Season':<12} {'Matches':>8} {'Draws':>8} {'Pi+DC':>10} {'Pi Base':>10} {'ELO':>10}")
    print("-" * 65)

    for season in sorted(results_df["season"].unique()):
        season_data = results_df[results_df["season"] == season]
        draws = (season_data["actual"] == "D").sum()

        pidc_pred_draws = (season_data["pidc_pred"] == "D").sum() if "pidc_pred" in season_data else 0
        pi_pred_draws = (season_data["pi_pred"] == "D").sum() if "pi_pred" in season_data else 0
        elo_pred_draws = (season_data["elo_pred"] == "D").sum() if "elo_pred" in season_data else 0

        # Draw accuracy (what % of predicted draws were correct)
        pidc_corr = season_data[(season_data["pidc_pred"] == "D") & (season_data["actual"] == "D")]
        pi_corr = season_data[(season_data["pi_pred"] == "D") & (season_data["actual"] == "D")]
        elo_corr = season_data[(season_data["elo_pred"] == "D") & (season_data["actual"] == "D")]

        pidc_acc = len(pidc_corr) / pidc_pred_draws if pidc_pred_draws > 0 else 0
        pi_acc = len(pi_corr) / pi_pred_draws if pi_pred_draws > 0 else 0
        elo_acc = len(elo_corr) / elo_pred_draws if elo_pred_draws > 0 else 0

        print(f"{season:<12} {len(season_data):>8} {draws:>8} "
              f"{pidc_acc:>10.1%} {pi_acc:>10.1%} {elo_acc:>10.1%}")

    # Feature importance from draw model
    print("\n" + "=" * 80)
    print("DRAW MODEL FEATURE IMPORTANCE")
    print("=" * 80)

    if backtest["model"].draw_model_trained:
        features = ["strength_parity", "low_scoring", "both_defensive",
                   "mid_table_clash", "rating_diff_abs", "total_xg"]
        coefs = backtest["model"].draw_model.coef_[0]

        print(f"\n{'Feature':<20} {'Coefficient':>15} {'Direction':>15}")
        print("-" * 50)
        for feat, coef in sorted(zip(features, coefs, strict=False), key=lambda x: -abs(x[1])):
            direction = "More draws" if coef > 0 else "Fewer draws"
            print(f"{feat:<20} {coef:>+15.4f} {direction:>15}")

    # Final summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    pidc_brier_improvement = (pi_metrics["brier_score"] - pidc_metrics["brier_score"]) / pi_metrics["brier_score"] * 100
    pidc_draw_improvement = (pi_metrics["draw_brier"] - pidc_metrics["draw_brier"]) / pi_metrics["draw_brier"] * 100

    print("\nPi + Dixon-Coles vs Pi Baseline:")
    print(f"  Brier Score Improvement: {pidc_brier_improvement:+.2f}%")
    print(f"  Draw Brier Improvement:  {pidc_draw_improvement:+.2f}%")
    print(f"  Optimal rho:             {backtest['rho']:.4f}")


if __name__ == "__main__":
    main()
