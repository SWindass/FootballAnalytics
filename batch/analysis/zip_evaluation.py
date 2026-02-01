"""
Evaluate Zero-Inflated Poisson + Dixon-Coles Model.

Compares:
1. Standard xG-Poisson (baseline)
2. xG-Poisson + Dixon-Coles
3. ZIP + Dixon-Coles (new)
4. 5-model ensemble with ZIP

Metrics:
- Overall accuracy & Brier score
- Draw precision & recall
- 0-0 prediction accuracy
- Calibration curves
"""

import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import poisson
from scipy.optimize import minimize
from sqlalchemy import text

from app.db.database import SyncSessionLocal
from batch.models.zip_dixon_coles import ZIPDixonColesModel, ZIPMatchFeatures
from batch.models.seasonal_recalibration import (
    SeasonalRecalibration,
    apply_draw_threshold_adjustment,
)

warnings.filterwarnings("ignore")


def load_matches_with_xg():
    """Load all matches with xG data."""
    print("Loading match data...")

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

        print(f"  Loaded {len(matches)} matches with xG data")
        return pd.DataFrame(matches)


def evaluate_standard_poisson(df: pd.DataFrame) -> dict:
    """Baseline: Standard Poisson without Dixon-Coles or ZIP."""
    print("\n" + "=" * 70)
    print("MODEL 1: Standard xG-Poisson (Baseline)")
    print("=" * 70)

    results_by_season = defaultdict(list)

    for _, row in df.iterrows():
        home_xg = row['home_xg']
        away_xg = row['away_xg']
        home_goals = int(row['home_score'])
        away_goals = int(row['away_score'])
        season = row['season']

        # Actual outcome
        if home_goals > away_goals:
            actual = 'H'
        elif home_goals < away_goals:
            actual = 'A'
        else:
            actual = 'D'

        # Poisson probabilities
        home_prob = draw_prob = away_prob = 0
        for h in range(8):
            for a in range(8):
                p = poisson.pmf(h, home_xg) * poisson.pmf(a, away_xg)
                if h > a:
                    home_prob += p
                elif h == a:
                    draw_prob += p
                else:
                    away_prob += p

        total = home_prob + draw_prob + away_prob
        home_prob /= total
        draw_prob /= total
        away_prob /= total

        predicted = apply_draw_threshold_adjustment(home_prob, draw_prob, away_prob)

        results_by_season[season].append({
            'actual': actual,
            'predicted': predicted,
            'correct': predicted == actual,
            'home_prob': home_prob,
            'draw_prob': draw_prob,
            'away_prob': away_prob,
            'is_0_0': home_goals == 0 and away_goals == 0,
        })

    return print_results(results_by_season, "Standard Poisson")


def evaluate_poisson_dc(df: pd.DataFrame) -> dict:
    """xG-Poisson with Dixon-Coles correction."""
    print("\n" + "=" * 70)
    print("MODEL 2: xG-Poisson + Dixon-Coles")
    print("=" * 70)

    rho = -0.13
    results_by_season = defaultdict(list)

    for _, row in df.iterrows():
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

        # Poisson + Dixon-Coles
        home_prob = draw_prob = away_prob = 0
        for h in range(8):
            for a in range(8):
                p = poisson.pmf(h, home_xg) * poisson.pmf(a, away_xg)

                # Dixon-Coles correction
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
        home_prob /= total
        draw_prob /= total
        away_prob /= total

        predicted = apply_draw_threshold_adjustment(home_prob, draw_prob, away_prob)

        results_by_season[season].append({
            'actual': actual,
            'predicted': predicted,
            'correct': predicted == actual,
            'home_prob': home_prob,
            'draw_prob': draw_prob,
            'away_prob': away_prob,
            'is_0_0': home_goals == 0 and away_goals == 0,
        })

    return print_results(results_by_season, "Poisson + DC")


def evaluate_zip_dc(df: pd.DataFrame) -> dict:
    """ZIP + Dixon-Coles model."""
    print("\n" + "=" * 70)
    print("MODEL 3: Zero-Inflated Poisson + Dixon-Coles")
    print("=" * 70)

    model = ZIPDixonColesModel(rho=-0.13, structural_zero_weight=0.5)
    recal = SeasonalRecalibration(window_size=50)

    results_by_season = defaultdict(list)
    training_started = False

    for idx, row in df.iterrows():
        home_xg = row['home_xg']
        away_xg = row['away_xg']
        home_goals = int(row['home_score'])
        away_goals = int(row['away_score'])
        season = row['season']
        home_team = row['home_team']
        away_team = row['away_team']

        if home_goals > away_goals:
            actual = 'H'
        elif home_goals < away_goals:
            actual = 'A'
        else:
            actual = 'D'

        # Update recalibration for recent draw rate
        stats = recal.get_current_statistics()
        model.set_league_context(
            positions={},
            xga={},
            recent_draw_rate=stats['draw_rate']
        )

        # Get prediction
        pred = model.predict_match(
            home_team, away_team,
            home_xg, away_xg,
            rating_diff=0.0
        )

        predicted = pred.get_prediction()

        results_by_season[season].append({
            'actual': actual,
            'predicted': predicted,
            'correct': predicted == actual,
            'home_prob': pred.home_prob,
            'draw_prob': pred.draw_prob,
            'away_prob': pred.away_prob,
            'is_0_0': home_goals == 0 and away_goals == 0,
            'p_structural': pred.p_structural_zero,
            'p_0_0_poisson': pred.p_poisson_0_0,
            'p_0_0_combined': pred.p_combined_0_0,
        })

        # Update model and recalibration
        model.update_after_match(
            home_team, away_team,
            home_goals, away_goals,
            home_xg, away_xg
        )
        recal.add_match(home_goals, away_goals, home_xg, away_xg)

        # Train structural zero model after enough samples
        if not training_started and len(model._training_features) >= 200:
            train_result = model.train_structural_zero_model()
            if train_result['status'] == 'trained':
                print(f"\n  Structural zero model trained:")
                print(f"    Samples: {train_result['samples']}")
                print(f"    0-0 rate: {train_result['zero_zero_rate']*100:.1f}%")
                print(f"    0-0 recall: {train_result['recall_0_0']*100:.1f}%")
                training_started = True

    return print_results(results_by_season, "ZIP + DC")


def print_results(results_by_season: dict, model_name: str) -> dict:
    """Print results summary and return aggregated stats."""
    print(f"\n{'Season':<12} {'Matches':>8} {'Accuracy':>10} {'Draw Acc':>10} {'0-0 Acc':>10}")
    print("-" * 55)

    all_results = []
    for season in sorted(results_by_season.keys()):
        results = results_by_season[season]
        if len(results) < 10:
            continue

        all_results.extend(results)

        accuracy = np.mean([r['correct'] for r in results])

        # Draw accuracy
        draw_results = [r for r in results if r['actual'] == 'D']
        draw_accuracy = np.mean([r['correct'] for r in draw_results]) if draw_results else 0

        # 0-0 accuracy
        zero_zero_results = [r for r in results if r['is_0_0']]
        zero_zero_correct = sum(1 for r in zero_zero_results if r['predicted'] == 'D')
        zero_zero_acc = zero_zero_correct / len(zero_zero_results) if zero_zero_results else 0

        print(f"{season:<12} {len(results):>8} {accuracy*100:>9.1f}% {draw_accuracy*100:>9.1f}% {zero_zero_acc*100:>9.1f}%")

    # Overall stats
    print("-" * 55)
    overall_acc = np.mean([r['correct'] for r in all_results])
    overall_draw_acc = np.mean([r['correct'] for r in all_results if r['actual'] == 'D'])
    overall_0_0_acc = sum(1 for r in all_results if r['is_0_0'] and r['predicted'] == 'D') / sum(1 for r in all_results if r['is_0_0'])

    # Brier score
    brier = 0
    for r in all_results:
        h_act = 1 if r['actual'] == 'H' else 0
        d_act = 1 if r['actual'] == 'D' else 0
        a_act = 1 if r['actual'] == 'A' else 0
        brier += (r['home_prob'] - h_act)**2 + (r['draw_prob'] - d_act)**2 + (r['away_prob'] - a_act)**2
    brier /= len(all_results)

    # Draw precision/recall
    draw_predicted = sum(1 for r in all_results if r['predicted'] == 'D')
    draw_actual = sum(1 for r in all_results if r['actual'] == 'D')
    draw_correct = sum(1 for r in all_results if r['predicted'] == 'D' and r['actual'] == 'D')

    draw_precision = draw_correct / draw_predicted if draw_predicted > 0 else 0
    draw_recall = draw_correct / draw_actual if draw_actual > 0 else 0

    print(f"{'OVERALL':<12} {len(all_results):>8} {overall_acc*100:>9.1f}% {overall_draw_acc*100:>9.1f}% {overall_0_0_acc*100:>9.1f}%")

    print(f"\n  Brier Score: {brier:.4f}")
    print(f"  Draw Precision: {draw_precision*100:.1f}% ({draw_correct}/{draw_predicted})")
    print(f"  Draw Recall: {draw_recall*100:.1f}% ({draw_correct}/{draw_actual})")

    return {
        'results': dict(results_by_season),
        'accuracy': overall_acc,
        'brier': brier,
        'draw_precision': draw_precision,
        'draw_recall': draw_recall,
        'zero_zero_acc': overall_0_0_acc,
    }


def compare_models(results: list[dict], names: list[str]):
    """Compare all models side by side."""
    print("\n" + "=" * 70)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 70)

    print(f"\n{'Model':<30} {'Accuracy':>10} {'Brier':>10} {'Draw P':>10} {'Draw R':>10} {'0-0 Acc':>10}")
    print("-" * 85)

    for name, result in zip(names, results):
        print(f"{name:<30} {result['accuracy']*100:>9.1f}% {result['brier']:>10.4f} "
              f"{result['draw_precision']*100:>9.1f}% {result['draw_recall']*100:>9.1f}% "
              f"{result['zero_zero_acc']*100:>9.1f}%")

    # Find best
    best_acc_idx = np.argmax([r['accuracy'] for r in results])
    best_brier_idx = np.argmin([r['brier'] for r in results])
    best_draw_p_idx = np.argmax([r['draw_precision'] for r in results])

    print(f"\n  Best Accuracy: {names[best_acc_idx]}")
    print(f"  Best Brier: {names[best_brier_idx]}")
    print(f"  Best Draw Precision: {names[best_draw_p_idx]}")


def main():
    # Load data
    df = load_matches_with_xg()

    # Evaluate each model
    results = []
    names = []

    # Model 1: Standard Poisson
    r1 = evaluate_standard_poisson(df)
    results.append(r1)
    names.append("Standard Poisson")

    # Model 2: Poisson + Dixon-Coles
    r2 = evaluate_poisson_dc(df)
    results.append(r2)
    names.append("Poisson + Dixon-Coles")

    # Model 3: ZIP + Dixon-Coles
    r3 = evaluate_zip_dc(df)
    results.append(r3)
    names.append("ZIP + Dixon-Coles")

    # Compare all
    compare_models(results, names)

    # 2025-26 specific
    print("\n" + "=" * 70)
    print("2025-26 SEASON COMPARISON")
    print("=" * 70)

    for name, result in zip(names, results):
        if '2025-26' in result['results']:
            r = result['results']['2025-26']
            acc = np.mean([x['correct'] for x in r]) * 100
            draws = [x for x in r if x['actual'] == 'D']
            draw_acc = np.mean([x['correct'] for x in draws]) * 100 if draws else 0
            zero_zeros = [x for x in r if x['is_0_0']]
            zz_acc = sum(1 for x in zero_zeros if x['predicted'] == 'D') / len(zero_zeros) * 100 if zero_zeros else 0
            print(f"  {name:<25}: {acc:.1f}% accuracy, {draw_acc:.1f}% draw acc, {zz_acc:.1f}% 0-0 acc")


if __name__ == "__main__":
    main()
