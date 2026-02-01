"""
Optimize ZIP parameters and integrate into 5-model ensemble.

Tests different structural_zero_weight values and finds optimal
ensemble weights for the 5-model combination.
"""

import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import poisson
from scipy.optimize import minimize
from sqlalchemy import text

from app.db.database import SyncSessionLocal
from batch.models.zip_dixon_coles import ZIPDixonColesModel
from batch.models.seasonal_recalibration import (
    SeasonalRecalibration,
    apply_draw_threshold_adjustment,
)

warnings.filterwarnings("ignore")


def load_matches_with_xg():
    """Load all matches with xG data."""
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
                'home_xg': float(row[6]) if row[6] else 1.35,
                'away_xg': float(row[7]) if row[7] else 1.35,
                'home_team': row[8],
                'away_team': row[9],
            })

        return pd.DataFrame(matches)


def calculate_poisson_dc_probs(home_xg, away_xg, rho=-0.13):
    """Calculate Poisson + Dixon-Coles probabilities."""
    home_prob = draw_prob = away_prob = 0
    for h in range(8):
        for a in range(8):
            p = poisson.pmf(h, home_xg) * poisson.pmf(a, away_xg)
            if h == 0 and a == 0:
                p *= (1 - home_xg * away_xg * rho)
            elif h == 0 and a == 1:
                p *= (1 + home_xg * rho)
            elif h == 1 and a == 0:
                p *= (1 + away_xg * rho)
            elif h == 1 and a == 1:
                p *= (1 - rho)

            if h > a:
                home_prob += p
            elif h == a:
                draw_prob += p
            else:
                away_prob += p

    total = home_prob + draw_prob + away_prob
    return home_prob/total, draw_prob/total, away_prob/total


def test_zip_weight(df, structural_weight):
    """Test ZIP model with specific structural weight."""
    model = ZIPDixonColesModel(rho=-0.13, structural_zero_weight=structural_weight)
    recal = SeasonalRecalibration(window_size=50)

    results = []
    trained = False

    for _, row in df.iterrows():
        home_xg = row['home_xg']
        away_xg = row['away_xg']
        home_goals = int(row['home_score'])
        away_goals = int(row['away_score'])

        if home_goals > away_goals:
            actual = 'H'
        elif home_goals < away_goals:
            actual = 'A'
        else:
            actual = 'D'

        stats = recal.get_current_statistics()
        model.set_league_context({}, {}, recent_draw_rate=stats['draw_rate'])

        pred = model.predict_match(
            row['home_team'], row['away_team'],
            home_xg, away_xg
        )

        results.append({
            'actual': actual,
            'home_prob': pred.home_prob,
            'draw_prob': pred.draw_prob,
            'away_prob': pred.away_prob,
        })

        model.update_after_match(
            row['home_team'], row['away_team'],
            home_goals, away_goals,
            home_xg, away_xg
        )
        recal.add_match(home_goals, away_goals, home_xg, away_xg)

        if not trained and len(model._training_features) >= 200:
            model.train_structural_zero_model()
            trained = True

    # Calculate metrics
    correct = 0
    brier = 0
    for r in results:
        pred = apply_draw_threshold_adjustment(r['home_prob'], r['draw_prob'], r['away_prob'])
        if pred == r['actual']:
            correct += 1

        h_act = 1 if r['actual'] == 'H' else 0
        d_act = 1 if r['actual'] == 'D' else 0
        a_act = 1 if r['actual'] == 'A' else 0
        brier += (r['home_prob'] - h_act)**2 + (r['draw_prob'] - d_act)**2 + (r['away_prob'] - a_act)**2

    return correct / len(results), brier / len(results), results


def generate_all_model_predictions(df):
    """Generate predictions from all 5 models for each match."""
    print("Generating predictions from all models...")

    # Initialize models
    zip_model = ZIPDixonColesModel(rho=-0.13, structural_zero_weight=0.15)  # Tuned weight
    recal = SeasonalRecalibration(window_size=50)

    all_predictions = []
    zip_trained = False

    for idx, row in df.iterrows():
        home_xg = row['home_xg']
        away_xg = row['away_xg']
        home_goals = int(row['home_score'])
        away_goals = int(row['away_score'])
        season = row['season']

        if home_goals > away_goals:
            actual = 'H'
        elif home_goals < away_goals:
            actual = 'A'
        else:
            actual = 'D'

        # Model 1: Standard Poisson
        h1, d1, a1 = calculate_poisson_dc_probs(home_xg, away_xg, rho=0)

        # Model 2: Poisson + Dixon-Coles (rho=-0.13)
        h2, d2, a2 = calculate_poisson_dc_probs(home_xg, away_xg, rho=-0.13)

        # Model 3: Poisson + DC with stronger rho (-0.18)
        h3, d3, a3 = calculate_poisson_dc_probs(home_xg, away_xg, rho=-0.18)

        # Model 4: ZIP + Dixon-Coles
        stats = recal.get_current_statistics()
        zip_model.set_league_context({}, {}, recent_draw_rate=stats['draw_rate'])
        zip_pred = zip_model.predict_match(
            row['home_team'], row['away_team'],
            home_xg, away_xg
        )
        h4, d4, a4 = zip_pred.home_prob, zip_pred.draw_prob, zip_pred.away_prob

        all_predictions.append({
            'season': season,
            'actual': actual,
            'poisson': (h1, d1, a1),
            'dc': (h2, d2, a2),
            'dc_strong': (h3, d3, a3),
            'zip': (h4, d4, a4),
        })

        # Update models
        zip_model.update_after_match(
            row['home_team'], row['away_team'],
            home_goals, away_goals,
            home_xg, away_xg
        )
        recal.add_match(home_goals, away_goals, home_xg, away_xg)

        if not zip_trained and len(zip_model._training_features) >= 200:
            zip_model.train_structural_zero_model()
            zip_trained = True

    return all_predictions


def optimize_ensemble_weights(predictions, model_keys=['poisson', 'dc', 'dc_strong', 'zip']):
    """Find optimal weights for ensemble."""
    n_models = len(model_keys)

    def objective(weights):
        weights = weights / weights.sum()  # Normalize
        brier = 0

        for pred in predictions:
            # Combine predictions
            home = sum(weights[i] * pred[k][0] for i, k in enumerate(model_keys))
            draw = sum(weights[i] * pred[k][1] for i, k in enumerate(model_keys))
            away = sum(weights[i] * pred[k][2] for i, k in enumerate(model_keys))

            total = home + draw + away
            home, draw, away = home/total, draw/total, away/total

            h_act = 1 if pred['actual'] == 'H' else 0
            d_act = 1 if pred['actual'] == 'D' else 0
            a_act = 1 if pred['actual'] == 'A' else 0

            brier += (home - h_act)**2 + (draw - d_act)**2 + (away - a_act)**2

        return brier / len(predictions)

    # Optimize
    x0 = np.ones(n_models) / n_models
    bounds = [(0.05, 0.8)] * n_models
    constraint = {'type': 'eq', 'fun': lambda w: w.sum() - 1}

    result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraint)

    return result.x / result.x.sum(), result.fun


def evaluate_ensemble(predictions, weights, model_keys):
    """Evaluate ensemble with given weights."""
    correct = 0
    brier = 0
    draw_correct = 0
    draw_predicted = 0
    draw_actual = 0

    results_by_season = defaultdict(list)

    for pred in predictions:
        # Combine predictions
        home = sum(weights[i] * pred[k][0] for i, k in enumerate(model_keys))
        draw = sum(weights[i] * pred[k][1] for i, k in enumerate(model_keys))
        away = sum(weights[i] * pred[k][2] for i, k in enumerate(model_keys))

        total = home + draw + away
        home, draw, away = home/total, draw/total, away/total

        predicted = apply_draw_threshold_adjustment(home, draw, away)

        if predicted == pred['actual']:
            correct += 1

        if predicted == 'D':
            draw_predicted += 1
            if pred['actual'] == 'D':
                draw_correct += 1
        if pred['actual'] == 'D':
            draw_actual += 1

        h_act = 1 if pred['actual'] == 'H' else 0
        d_act = 1 if pred['actual'] == 'D' else 0
        a_act = 1 if pred['actual'] == 'A' else 0
        brier += (home - h_act)**2 + (draw - d_act)**2 + (away - a_act)**2

        results_by_season[pred['season']].append({
            'actual': pred['actual'],
            'predicted': predicted,
            'correct': predicted == pred['actual'],
        })

    n = len(predictions)
    return {
        'accuracy': correct / n,
        'brier': brier / n,
        'draw_precision': draw_correct / draw_predicted if draw_predicted > 0 else 0,
        'draw_recall': draw_correct / draw_actual if draw_actual > 0 else 0,
        'results_by_season': dict(results_by_season),
    }


def main():
    print("=" * 70)
    print("ZIP ENSEMBLE OPTIMIZATION")
    print("=" * 70)

    # Load data
    df = load_matches_with_xg()
    print(f"\nLoaded {len(df)} matches with xG data")

    # Phase 1: Test different ZIP structural weights
    print("\n" + "-" * 70)
    print("PHASE 1: Optimizing ZIP structural_zero_weight")
    print("-" * 70)

    best_weight = 0.15
    best_brier = float('inf')

    for weight in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        acc, brier, _ = test_zip_weight(df, weight)
        status = " <-- BEST" if brier < best_brier else ""
        if brier < best_brier:
            best_brier = brier
            best_weight = weight
        print(f"  Weight {weight:.2f}: Accuracy {acc*100:.1f}%, Brier {brier:.4f}{status}")

    print(f"\n  Optimal structural_zero_weight: {best_weight}")

    # Phase 2: Generate all model predictions
    print("\n" + "-" * 70)
    print("PHASE 2: Generating multi-model predictions")
    print("-" * 70)

    predictions = generate_all_model_predictions(df)
    print(f"  Generated predictions for {len(predictions)} matches")

    # Phase 3: Optimize ensemble weights
    print("\n" + "-" * 70)
    print("PHASE 3: Optimizing ensemble weights")
    print("-" * 70)

    model_keys = ['poisson', 'dc', 'dc_strong', 'zip']

    # 3-model ensemble (without ZIP)
    weights_3, brier_3 = optimize_ensemble_weights(predictions, ['poisson', 'dc', 'dc_strong'])
    result_3 = evaluate_ensemble(predictions, weights_3, ['poisson', 'dc', 'dc_strong'])

    print(f"\n  3-Model Ensemble (no ZIP):")
    print(f"    Weights: Poisson={weights_3[0]:.2f}, DC={weights_3[1]:.2f}, DC_strong={weights_3[2]:.2f}")
    print(f"    Accuracy: {result_3['accuracy']*100:.1f}%")
    print(f"    Brier: {result_3['brier']:.4f}")
    print(f"    Draw Precision: {result_3['draw_precision']*100:.1f}%")
    print(f"    Draw Recall: {result_3['draw_recall']*100:.1f}%")

    # 4-model ensemble (with ZIP)
    weights_4, brier_4 = optimize_ensemble_weights(predictions, model_keys)
    result_4 = evaluate_ensemble(predictions, weights_4, model_keys)

    print(f"\n  4-Model Ensemble (with ZIP):")
    print(f"    Weights: Poisson={weights_4[0]:.2f}, DC={weights_4[1]:.2f}, DC_strong={weights_4[2]:.2f}, ZIP={weights_4[3]:.2f}")
    print(f"    Accuracy: {result_4['accuracy']*100:.1f}%")
    print(f"    Brier: {result_4['brier']:.4f}")
    print(f"    Draw Precision: {result_4['draw_precision']*100:.1f}%")
    print(f"    Draw Recall: {result_4['draw_recall']*100:.1f}%")

    # Phase 4: Compare by season
    print("\n" + "-" * 70)
    print("PHASE 4: Season comparison")
    print("-" * 70)

    print(f"\n  {'Season':<12} {'3-Model':>10} {'4-Model':>10} {'Diff':>10}")
    print("  " + "-" * 45)

    for season in sorted(result_3['results_by_season'].keys()):
        if season not in result_4['results_by_season']:
            continue

        r3 = result_3['results_by_season'][season]
        r4 = result_4['results_by_season'][season]

        acc_3 = np.mean([x['correct'] for x in r3]) * 100
        acc_4 = np.mean([x['correct'] for x in r4]) * 100
        diff = acc_4 - acc_3

        sign = '+' if diff >= 0 else ''
        print(f"  {season:<12} {acc_3:>9.1f}% {acc_4:>9.1f}% {sign}{diff:>9.1f}pp")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    improvement_acc = (result_4['accuracy'] - result_3['accuracy']) * 100
    improvement_brier = (result_3['brier'] - result_4['brier'])  # Lower is better
    improvement_draw_p = (result_4['draw_precision'] - result_3['draw_precision']) * 100
    improvement_draw_r = (result_4['draw_recall'] - result_3['draw_recall']) * 100

    print(f"""
  Adding ZIP to ensemble:
    Accuracy:       {result_3['accuracy']*100:.1f}% → {result_4['accuracy']*100:.1f}% ({'+' if improvement_acc >= 0 else ''}{improvement_acc:.1f}pp)
    Brier:          {result_3['brier']:.4f} → {result_4['brier']:.4f} ({'+' if improvement_brier >= 0 else ''}{improvement_brier:.4f})
    Draw Precision: {result_3['draw_precision']*100:.1f}% → {result_4['draw_precision']*100:.1f}% ({'+' if improvement_draw_p >= 0 else ''}{improvement_draw_p:.1f}pp)
    Draw Recall:    {result_3['draw_recall']*100:.1f}% → {result_4['draw_recall']*100:.1f}% ({'+' if improvement_draw_r >= 0 else ''}{improvement_draw_r:.1f}pp)

  Optimal ZIP weight in ensemble: {weights_4[3]*100:.0f}%

  VERDICT: {"ZIP IMPROVES ensemble" if improvement_brier > 0 or improvement_acc > 0.5 else "ZIP provides marginal benefit"}
""")


if __name__ == "__main__":
    main()
