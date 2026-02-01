"""
Regression-Aware Ensemble Evaluation.

Tests the adaptive weighting strategy that boosts xG model weight
when teams are over/underperforming their xG.
"""

import math
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import log_loss

from sqlalchemy import text

from app.db.database import SyncSessionLocal
from batch.models.xg_predictor import XGPredictor
from batch.models.regression_aware_ensemble import RegressionAwareEnsemble
from batch.models.elo import EloRatingSystem, EloConfig
from batch.models.pi_rating import PiRating
from batch.models.pi_dixon_coles import PiDixonColesModel

warnings.filterwarnings("ignore")


def load_matches_with_xg():
    """Load matches that have xG data."""
    print("Loading matches with xG data...")

    with SyncSessionLocal() as session:
        query = text("""
            SELECT m.id, m.kickoff_time, m.matchweek, m.season,
                   m.home_score, m.away_score, m.home_xg, m.away_xg,
                   ht.name as home_team, at.name as away_team
            FROM matches m
            JOIN teams ht ON m.home_team_id = ht.id
            JOIN teams at ON m.away_team_id = at.id
            WHERE m.status = 'finished'
            AND m.home_score IS NOT NULL
            AND m.home_xg IS NOT NULL
            AND m.away_xg IS NOT NULL
            ORDER BY m.kickoff_time
        """)

        result = session.execute(query)
        rows = result.fetchall()

        matches = []
        for row in rows:
            matches.append({
                'id': row[0],
                'kickoff_time': row[1],
                'matchweek': row[2],
                'season': row[3],
                'home_score': row[4],
                'away_score': row[5],
                'home_xg': float(row[6]),
                'away_xg': float(row[7]),
                'home_team': row[8],
                'away_team': row[9],
            })

        print(f"  Loaded {len(matches)} matches")
        return matches


def generate_predictions(matches: list[dict], warmup: int = 380):
    """Generate predictions from all models including regression-aware ensemble."""
    print(f"\nGenerating predictions (warmup={warmup})...")

    # Initialize models
    xg_predictor = XGPredictor(window=10, rho=-0.11, home_advantage=1.10)
    elo = EloRatingSystem(EloConfig(k_factor=28.0, home_advantage=50.0))
    pi = PiRating(lambda_param=0.07, gamma_param=0.7)
    pidc = PiDixonColesModel(pi_lambda=0.07, pi_gamma=0.7, rho=-0.11)

    # Regression-aware ensemble
    # Optimal threshold from testing: 0.50 (43% of matches flagged)
    regression_ensemble = RegressionAwareEnsemble(
        window=10,
        regression_threshold=0.50,  # ~0.5 goals per game over/under xG
        standard_weights={'xg': 0.205, 'pidc': 0.434, 'elo': 0.197, 'pi': 0.163},
        regression_weights={'xg': 0.45, 'pidc': 0.25, 'elo': 0.15, 'pi': 0.15},
    )

    # Team IDs for ELO
    team_ids = {}
    next_id = 1

    results = []

    for idx, match in enumerate(matches):
        home = match['home_team']
        away = match['away_team']

        if home not in team_ids:
            team_ids[home] = next_id
            next_id += 1
        if away not in team_ids:
            team_ids[away] = next_id
            next_id += 1

        try:
            # Get individual model predictions
            xg_pred = xg_predictor.predict_match(home, away)

            pidc_pred = pidc.predict_match(home, away, apply_draw_model=pidc.draw_model_trained)

            pi_gd = pi.calculate_expected_goal_diff(home, away)
            pi_h = 1 / (1 + math.exp(-pi_gd * 0.7))
            pi_a = 1 / (1 + math.exp(pi_gd * 0.7))
            pi_d = max(0, 0.28 - 0.1 * abs(pi_gd))
            pi_total = pi_h + pi_d + pi_a
            pi_h, pi_d, pi_a = pi_h / pi_total, pi_d / pi_total, pi_a / pi_total

            elo_h, elo_d, elo_a = elo.match_probabilities(
                team_ids[home], team_ids[away]
            )

            # Regression-aware ensemble prediction
            model_preds = {
                'xg': (xg_pred.home_prob, xg_pred.draw_prob, xg_pred.away_prob),
                'pidc': (pidc_pred.home_win, pidc_pred.draw, pidc_pred.away_win),
                'elo': (elo_h, elo_d, elo_a),
                'pi': (pi_h, pi_d, pi_a),
            }

            reg_pred = regression_ensemble.predict(home, away, model_preds)

            # Standard fixed-weight ensemble (for comparison)
            std_weights = {'xg': 0.205, 'pidc': 0.434, 'elo': 0.197, 'pi': 0.163}
            std_h = sum(w * model_preds[m][0] for m, w in std_weights.items())
            std_d = sum(w * model_preds[m][1] for m, w in std_weights.items())
            std_a = sum(w * model_preds[m][2] for m, w in std_weights.items())
            std_total = std_h + std_d + std_a
            std_h, std_d, std_a = std_h / std_total, std_d / std_total, std_a / std_total

            # Actual outcome
            if match['home_score'] > match['away_score']:
                actual = 'H'
            elif match['home_score'] == match['away_score']:
                actual = 'D'
            else:
                actual = 'A'

            if idx >= warmup:
                results.append({
                    'match_id': match['id'],
                    'season': match['season'],
                    'home_team': home,
                    'away_team': away,
                    'actual': actual,

                    # Individual models
                    'xg_home': xg_pred.home_prob,
                    'xg_draw': xg_pred.draw_prob,
                    'xg_away': xg_pred.away_prob,

                    'pidc_home': pidc_pred.home_win,
                    'pidc_draw': pidc_pred.draw,
                    'pidc_away': pidc_pred.away_win,

                    'elo_home': elo_h,
                    'elo_draw': elo_d,
                    'elo_away': elo_a,

                    'pi_home': pi_h,
                    'pi_draw': pi_d,
                    'pi_away': pi_a,

                    # Standard ensemble
                    'std_home': std_h,
                    'std_draw': std_d,
                    'std_away': std_a,

                    # Regression-aware ensemble
                    'reg_home': reg_pred.home_prob,
                    'reg_draw': reg_pred.draw_prob,
                    'reg_away': reg_pred.away_prob,
                    'regression_boost': reg_pred.regression_boost_applied,
                    'home_attack_luck': reg_pred.home_attack_luck,
                    'home_defense_luck': reg_pred.home_defense_luck,
                    'away_attack_luck': reg_pred.away_attack_luck,
                    'away_defense_luck': reg_pred.away_defense_luck,
                })

        except Exception as e:
            pass

        # Update all models
        xg_predictor.update_match(
            home, away,
            match['home_score'], match['away_score'],
            match['home_xg'], match['away_xg']
        )
        regression_ensemble.update_match(
            home, away,
            match['home_score'], match['away_score'],
            match['home_xg'], match['away_xg']
        )
        pidc.update_after_match(
            home, away, match['home_score'], match['away_score'],
            match['kickoff_time'], collect_training_data=True
        )
        pi.update_ratings(
            home, away, match['home_score'], match['away_score'],
            match['kickoff_time'], store_history=False
        )
        elo.update_ratings(
            team_ids[home], team_ids[away],
            match['home_score'], match['away_score']
        )

        if (idx + 1) % 500 == 0:
            print(f"  Processed {idx + 1}/{len(matches)}")

    print(f"  Generated {len(results)} predictions")
    return pd.DataFrame(results)


def calculate_metrics(df: pd.DataFrame, prefix: str) -> dict:
    """Calculate accuracy, Brier, log loss."""
    y_true = []
    y_pred = []
    predictions = []

    for _, row in df.iterrows():
        probs = [row[f'{prefix}_home'], row[f'{prefix}_draw'], row[f'{prefix}_away']]
        total = sum(probs)
        probs = [p / total for p in probs]
        y_pred.append(probs)

        if row['actual'] == 'H':
            y_true.append(0)
        elif row['actual'] == 'D':
            y_true.append(1)
        else:
            y_true.append(2)

        predictions.append(np.argmax(probs))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    predictions = np.array(predictions)

    accuracy = (predictions == y_true).mean()

    y_true_onehot = np.zeros((len(y_true), 3))
    y_true_onehot[np.arange(len(y_true)), y_true] = 1
    brier = ((y_pred - y_true_onehot) ** 2).sum(axis=1).mean()

    y_pred_clipped = np.clip(y_pred, 1e-10, 1 - 1e-10)
    logloss = log_loss(y_true, y_pred_clipped)

    return {'accuracy': accuracy, 'brier': brier, 'logloss': logloss}


def optimize_regression_weights(df: pd.DataFrame) -> tuple[dict, dict]:
    """
    Optimize weights separately for regression and non-regression matches.
    """
    regression_df = df[df['regression_boost'] == True]
    normal_df = df[df['regression_boost'] == False]

    models = ['xg', 'pidc', 'elo', 'pi']

    def optimize_weights(subset_df):
        def ensemble_brier(weights):
            weights = np.array(weights)
            weights = weights / weights.sum()

            total_brier = 0.0
            for _, row in subset_df.iterrows():
                home_prob = sum(w * row[f'{m}_home'] for w, m in zip(weights, models))
                draw_prob = sum(w * row[f'{m}_draw'] for w, m in zip(weights, models))
                away_prob = sum(w * row[f'{m}_away'] for w, m in zip(weights, models))

                total = home_prob + draw_prob + away_prob
                home_prob /= total
                draw_prob /= total
                away_prob /= total

                actual = [0, 0, 0]
                if row['actual'] == 'H':
                    actual[0] = 1
                elif row['actual'] == 'D':
                    actual[1] = 1
                else:
                    actual[2] = 1

                brier = sum((p - a) ** 2 for p, a in zip(
                    [home_prob, draw_prob, away_prob], actual
                )) / 3
                total_brier += brier

            return total_brier / len(subset_df)

        result = minimize(
            ensemble_brier,
            x0=[0.25, 0.25, 0.25, 0.25],
            bounds=[(0, 1)] * 4,
            method='L-BFGS-B'
        )

        weights = np.array(result.x)
        weights = weights / weights.sum()
        return {m: w for m, w in zip(models, weights)}

    print("\nOptimizing weights for regression matches...")
    reg_weights = optimize_weights(regression_df) if len(regression_df) > 50 else None

    print("Optimizing weights for normal matches...")
    normal_weights = optimize_weights(normal_df) if len(normal_df) > 50 else None

    return reg_weights, normal_weights


def main():
    matches = load_matches_with_xg()

    if len(matches) < 500:
        print("Not enough data!")
        return

    df = generate_predictions(matches, warmup=380)

    if df.empty:
        print("No predictions!")
        return

    # Split by season
    all_seasons = sorted(df['season'].unique())
    test_seasons = all_seasons[-3:]
    train_seasons = all_seasons[:-3]

    train_df = df[df['season'].isin(train_seasons)]
    test_df = df[df['season'].isin(test_seasons)]

    print(f"\nSeasons: Train={train_seasons}, Test={test_seasons}")
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")

    # Count regression matches
    regression_count = test_df['regression_boost'].sum()
    normal_count = len(test_df) - regression_count

    print(f"\nTest set breakdown:")
    print(f"  Regression matches: {regression_count} ({regression_count/len(test_df)*100:.1f}%)")
    print(f"  Normal matches:     {normal_count} ({normal_count/len(test_df)*100:.1f}%)")

    # Overall performance comparison
    print("\n" + "=" * 80)
    print("OVERALL PERFORMANCE (Test Set)")
    print("=" * 80)

    std_metrics = calculate_metrics(test_df, 'std')
    reg_metrics = calculate_metrics(test_df, 'reg')

    print(f"\n{'Ensemble':<30} {'Accuracy':>10} {'Brier':>10} {'Log Loss':>10}")
    print("-" * 70)
    print(f"{'Standard Fixed Weights':<30} {std_metrics['accuracy']*100:>9.1f}% "
          f"{std_metrics['brier']:>10.4f} {std_metrics['logloss']:>10.4f}")
    print(f"{'Regression-Aware':<30} {reg_metrics['accuracy']*100:>9.1f}% "
          f"{reg_metrics['brier']:>10.4f} {reg_metrics['logloss']:>10.4f}")

    # Performance breakdown by match type
    print("\n" + "=" * 80)
    print("PERFORMANCE BY MATCH TYPE")
    print("=" * 80)

    regression_df = test_df[test_df['regression_boost'] == True]
    normal_df = test_df[test_df['regression_boost'] == False]

    print(f"\n--- REGRESSION MATCHES (n={len(regression_df)}) ---")
    if len(regression_df) > 0:
        std_reg = calculate_metrics(regression_df, 'std')
        reg_reg = calculate_metrics(regression_df, 'reg')
        xg_reg = calculate_metrics(regression_df, 'xg')
        pidc_reg = calculate_metrics(regression_df, 'pidc')

        print(f"{'Model':<25} {'Accuracy':>10} {'Brier':>10}")
        print("-" * 50)
        print(f"{'xG Only':<25} {xg_reg['accuracy']*100:>9.1f}% {xg_reg['brier']:>10.4f}")
        print(f"{'Pi+DC Only':<25} {pidc_reg['accuracy']*100:>9.1f}% {pidc_reg['brier']:>10.4f}")
        print(f"{'Standard Ensemble':<25} {std_reg['accuracy']*100:>9.1f}% {std_reg['brier']:>10.4f}")
        print(f"{'Regression-Aware (xG 40%)':<25} {reg_reg['accuracy']*100:>9.1f}% {reg_reg['brier']:>10.4f}")

    print(f"\n--- NORMAL MATCHES (n={len(normal_df)}) ---")
    if len(normal_df) > 0:
        std_norm = calculate_metrics(normal_df, 'std')
        reg_norm = calculate_metrics(normal_df, 'reg')

        print(f"{'Model':<25} {'Accuracy':>10} {'Brier':>10}")
        print("-" * 50)
        print(f"{'Standard Ensemble':<25} {std_norm['accuracy']*100:>9.1f}% {std_norm['brier']:>10.4f}")
        print(f"{'Regression-Aware':<25} {reg_norm['accuracy']*100:>9.1f}% {reg_norm['brier']:>10.4f}")

    # Optimize weights on training data
    print("\n" + "=" * 80)
    print("OPTIMIZED WEIGHTS BY MATCH TYPE")
    print("=" * 80)

    train_regression = train_df[train_df['regression_boost'] == True]
    train_normal = train_df[train_df['regression_boost'] == False]

    opt_reg_weights, opt_norm_weights = optimize_regression_weights(train_df)

    if opt_reg_weights:
        print(f"\nOptimal weights for REGRESSION matches:")
        for m, w in opt_reg_weights.items():
            print(f"  {m}: {w*100:.1f}%")

    if opt_norm_weights:
        print(f"\nOptimal weights for NORMAL matches:")
        for m, w in opt_norm_weights.items():
            print(f"  {m}: {w*100:.1f}%")

    # Season breakdown
    print("\n" + "=" * 80)
    print("SEASON BREAKDOWN")
    print("=" * 80)

    print(f"\n{'Season':<12} {'Matches':>8} {'Reg%':>8} {'Std Acc':>10} {'Reg Acc':>10} {'Std Brier':>10} {'Reg Brier':>10}")
    print("-" * 80)

    for season in test_seasons:
        season_df = test_df[test_df['season'] == season]
        if len(season_df) > 0:
            reg_pct = season_df['regression_boost'].mean() * 100
            std_m = calculate_metrics(season_df, 'std')
            reg_m = calculate_metrics(season_df, 'reg')
            print(f"{season:<12} {len(season_df):>8} {reg_pct:>7.1f}% "
                  f"{std_m['accuracy']*100:>9.1f}% {reg_m['accuracy']*100:>9.1f}% "
                  f"{std_m['brier']:>10.4f} {reg_m['brier']:>10.4f}")

    # Final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    brier_improvement = (std_metrics['brier'] - reg_metrics['brier']) / std_metrics['brier'] * 100
    acc_improvement = (reg_metrics['accuracy'] - std_metrics['accuracy']) * 100

    print(f"""
Standard Fixed-Weight Ensemble:
  Accuracy: {std_metrics['accuracy']*100:.1f}%
  Brier:    {std_metrics['brier']:.4f}

Regression-Aware Ensemble:
  Accuracy: {reg_metrics['accuracy']*100:.1f}%
  Brier:    {reg_metrics['brier']:.4f}

IMPROVEMENT:
  Accuracy: {std_metrics['accuracy']*100:.1f}% → {reg_metrics['accuracy']*100:.1f}% ({acc_improvement:+.1f}pp)
  Brier:    {std_metrics['brier']:.4f} → {reg_metrics['brier']:.4f} ({brier_improvement:+.1f}%)

Regression matches ({regression_count}):
  - These are matches where teams are significantly over/underperforming xG
  - xG weight boosted from 20.5% to 40%
""")

    if len(regression_df) > 0:
        print(f"  Performance on regression matches:")
        print(f"    Standard: {std_reg['accuracy']*100:.1f}% acc, {std_reg['brier']:.4f} Brier")
        print(f"    Regression-Aware: {reg_reg['accuracy']*100:.1f}% acc, {reg_reg['brier']:.4f} Brier")


def test_thresholds():
    """Test multiple regression thresholds to find optimal."""
    matches = load_matches_with_xg()

    print("\nTesting different regression thresholds...")
    print("=" * 80)

    thresholds = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

    for threshold in thresholds:
        # Re-run with different threshold
        xg_predictor = XGPredictor(window=10, rho=-0.11, home_advantage=1.10)
        elo = EloRatingSystem(EloConfig(k_factor=28.0, home_advantage=50.0))
        pi = PiRating(lambda_param=0.07, gamma_param=0.7)
        pidc = PiDixonColesModel(pi_lambda=0.07, pi_gamma=0.7, rho=-0.11)

        regression_ensemble = RegressionAwareEnsemble(
            window=10,
            regression_threshold=threshold,
            standard_weights={'xg': 0.205, 'pidc': 0.434, 'elo': 0.197, 'pi': 0.163},
            regression_weights={'xg': 0.45, 'pidc': 0.25, 'elo': 0.15, 'pi': 0.15},
        )

        team_ids = {}
        next_id = 1
        results = []

        for idx, match in enumerate(matches):
            home = match['home_team']
            away = match['away_team']

            if home not in team_ids:
                team_ids[home] = next_id
                next_id += 1
            if away not in team_ids:
                team_ids[away] = next_id
                next_id += 1

            try:
                xg_pred = xg_predictor.predict_match(home, away)
                pidc_pred = pidc.predict_match(home, away, apply_draw_model=pidc.draw_model_trained)

                pi_gd = pi.calculate_expected_goal_diff(home, away)
                pi_h = 1 / (1 + math.exp(-pi_gd * 0.7))
                pi_a = 1 / (1 + math.exp(pi_gd * 0.7))
                pi_d = max(0, 0.28 - 0.1 * abs(pi_gd))
                pi_total = pi_h + pi_d + pi_a
                pi_h, pi_d, pi_a = pi_h / pi_total, pi_d / pi_total, pi_a / pi_total

                elo_h, elo_d, elo_a = elo.match_probabilities(team_ids[home], team_ids[away])

                model_preds = {
                    'xg': (xg_pred.home_prob, xg_pred.draw_prob, xg_pred.away_prob),
                    'pidc': (pidc_pred.home_win, pidc_pred.draw, pidc_pred.away_win),
                    'elo': (elo_h, elo_d, elo_a),
                    'pi': (pi_h, pi_d, pi_a),
                }

                reg_pred = regression_ensemble.predict(home, away, model_preds)

                # Standard ensemble
                std_weights = {'xg': 0.205, 'pidc': 0.434, 'elo': 0.197, 'pi': 0.163}
                std_h = sum(w * model_preds[m][0] for m, w in std_weights.items())
                std_d = sum(w * model_preds[m][1] for m, w in std_weights.items())
                std_a = sum(w * model_preds[m][2] for m, w in std_weights.items())
                std_total = std_h + std_d + std_a
                std_h, std_d, std_a = std_h / std_total, std_d / std_total, std_a / std_total

                if match['home_score'] > match['away_score']:
                    actual = 'H'
                elif match['home_score'] == match['away_score']:
                    actual = 'D'
                else:
                    actual = 'A'

                if idx >= 380 and match['season'] in ['2023-24', '2024-25', '2025-26']:
                    results.append({
                        'actual': actual,
                        'std_home': std_h, 'std_draw': std_d, 'std_away': std_a,
                        'reg_home': reg_pred.home_prob, 'reg_draw': reg_pred.draw_prob, 'reg_away': reg_pred.away_prob,
                        'regression_boost': reg_pred.regression_boost_applied,
                    })

            except:
                pass

            xg_predictor.update_match(home, away, match['home_score'], match['away_score'], match['home_xg'], match['away_xg'])
            regression_ensemble.update_match(home, away, match['home_score'], match['away_score'], match['home_xg'], match['away_xg'])
            pidc.update_after_match(home, away, match['home_score'], match['away_score'], match['kickoff_time'], collect_training_data=True)
            pi.update_ratings(home, away, match['home_score'], match['away_score'], match['kickoff_time'], store_history=False)
            elo.update_ratings(team_ids[home], team_ids[away], match['home_score'], match['away_score'])

        df = pd.DataFrame(results)

        if len(df) > 0:
            reg_pct = df['regression_boost'].mean() * 100
            std_m = calculate_metrics(df, 'std')
            reg_m = calculate_metrics(df, 'reg')

            brier_imp = (std_m['brier'] - reg_m['brier']) / std_m['brier'] * 100
            acc_imp = (reg_m['accuracy'] - std_m['accuracy']) * 100

            print(f"Threshold {threshold:.2f}: Reg%={reg_pct:5.1f}% | "
                  f"Std: {std_m['accuracy']*100:.1f}%/{std_m['brier']:.4f} | "
                  f"Reg: {reg_m['accuracy']*100:.1f}%/{reg_m['brier']:.4f} | "
                  f"Δ Acc={acc_imp:+.1f}pp, Δ Brier={brier_imp:+.1f}%")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--test-thresholds':
        test_thresholds()
    else:
        main()
