"""
Evaluate Seasonal Recalibration Fix for 2025-26 Performance Drop.

Tests whether the dynamic recalibration improves accuracy from 47.6% to target 52-53%.
"""

import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sqlalchemy import text

from app.db.database import SyncSessionLocal
from batch.models.pi_dixon_coles import PiDixonColesModel
from batch.models.seasonal_recalibration import (
    ConservativeRecalibration,
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
            ORDER BY m.kickoff_time
        """)

        result = session.execute(query)
        rows = result.fetchall()

        matches = []
        for row in rows:
            home_xg = float(row[6]) if row[6] is not None else None
            away_xg = float(row[7]) if row[7] is not None else None

            matches.append({
                'id': row[0],
                'kickoff_time': row[1],
                'matchweek': row[2],
                'season': row[3],
                'home_score': row[4],
                'away_score': row[5],
                'home_xg': home_xg,
                'away_xg': away_xg,
                'home_team': row[8],
                'away_team': row[9],
            })

        print(f"  Loaded {len(matches)} matches")
        return pd.DataFrame(matches)


def evaluate_without_recalibration(df: pd.DataFrame) -> dict:
    """Evaluate base Pi+Dixon-Coles model without recalibration."""
    print("\n" + "=" * 70)
    print("BASELINE: Pi+Dixon-Coles WITHOUT Recalibration")
    print("=" * 70)

    model = PiDixonColesModel(rho=-0.11)
    results_by_season = defaultdict(list)

    # Process matches chronologically
    for _, row in df.iterrows():
        if pd.isna(row['home_score']):
            continue

        home_team = row['home_team']
        away_team = row['away_team']
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

        # Get prediction BEFORE updating
        try:
            pred = model.predict_match(home_team, away_team, apply_draw_model=True)

            # Predicted outcome
            if pred.home_win >= pred.draw and pred.home_win >= pred.away_win:
                predicted = 'H'
            elif pred.draw >= pred.home_win and pred.draw >= pred.away_win:
                predicted = 'D'
            else:
                predicted = 'A'

            results_by_season[season].append({
                'actual': actual,
                'predicted': predicted,
                'correct': predicted == actual,
                'home_prob': pred.home_win,
                'draw_prob': pred.draw,
                'away_prob': pred.away_win,
            })

        except Exception:
            pass

        # Update model after match
        model.update_after_match(
            home_team, away_team, home_goals, away_goals,
            collect_training_data=True
        )

    # Calculate accuracy by season
    print(f"\n{'Season':<12} {'Matches':>8} {'Accuracy':>10} {'Draw Acc':>10}")
    print("-" * 45)

    for season in sorted(results_by_season.keys()):
        results = results_by_season[season]
        if len(results) < 10:
            continue

        accuracy = np.mean([r['correct'] for r in results])
        draw_results = [r for r in results if r['actual'] == 'D']
        draw_accuracy = np.mean([r['correct'] for r in draw_results]) if draw_results else 0

        print(f"{season:<12} {len(results):>8} {accuracy*100:>9.1f}% {draw_accuracy*100:>9.1f}%")

    return dict(results_by_season)


def evaluate_with_recalibration(df: pd.DataFrame) -> dict:
    """Evaluate Pi+Dixon-Coles WITH dynamic recalibration."""
    print("\n" + "=" * 70)
    print("WITH RECALIBRATION: Dynamic Seasonal Adjustment")
    print("=" * 70)

    model = PiDixonColesModel(rho=-0.11)
    recal = SeasonalRecalibration(window_size=50, sensitivity=1.0)

    results_by_season = defaultdict(list)
    recal_applied_count = 0

    # Process matches chronologically
    for _idx, row in df.iterrows():
        if pd.isna(row['home_score']):
            continue

        home_team = row['home_team']
        away_team = row['away_team']
        home_goals = int(row['home_score'])
        away_goals = int(row['away_score'])
        season = row['season']
        home_xg = row['home_xg'] if not pd.isna(row['home_xg']) else 1.35
        away_xg = row['away_xg'] if not pd.isna(row['away_xg']) else 1.35

        # Actual outcome
        if home_goals > away_goals:
            actual = 'H'
        elif home_goals < away_goals:
            actual = 'A'
        else:
            actual = 'D'

        # Get recalibration factors BEFORE prediction
        factors = recal.calculate_recalibration()

        # Get prediction
        try:
            # Adjust rho based on recalibration
            adjusted_rho = recal.adjust_rho(-0.11, factors)
            model.rho = adjusted_rho

            pred = model.predict_match(home_team, away_team, apply_draw_model=True)

            # Apply probability recalibration
            home_prob, draw_prob, away_prob = recal.adjust_probabilities(
                pred.home_win, pred.draw, pred.away_win, factors
            )

            # Predicted outcome
            if home_prob >= draw_prob and home_prob >= away_prob:
                predicted = 'H'
            elif draw_prob >= home_prob and draw_prob >= away_prob:
                predicted = 'D'
            else:
                predicted = 'A'

            # Track if recalibration was actually applied
            if factors.draw_boost > 1.01 or factors.rho_adjustment > 1.01:
                recal_applied_count += 1

            results_by_season[season].append({
                'actual': actual,
                'predicted': predicted,
                'correct': predicted == actual,
                'home_prob': home_prob,
                'draw_prob': draw_prob,
                'away_prob': away_prob,
                'draw_boost': factors.draw_boost,
                'rho_adj': factors.rho_adjustment,
            })

        except Exception:
            pass

        # Update model and recalibration AFTER match
        model.update_after_match(
            home_team, away_team, home_goals, away_goals,
            collect_training_data=True
        )
        recal.add_match(home_goals, away_goals, home_xg, away_xg)

    # Calculate accuracy by season
    print(f"\n{'Season':<12} {'Matches':>8} {'Accuracy':>10} {'Draw Acc':>10} {'Recal Applied':>14}")
    print("-" * 60)

    for season in sorted(results_by_season.keys()):
        results = results_by_season[season]
        if len(results) < 10:
            continue

        accuracy = np.mean([r['correct'] for r in results])
        draw_results = [r for r in results if r['actual'] == 'D']
        draw_accuracy = np.mean([r['correct'] for r in draw_results]) if draw_results else 0
        recal_pct = np.mean([r['draw_boost'] > 1.01 for r in results]) * 100

        print(f"{season:<12} {len(results):>8} {accuracy*100:>9.1f}% {draw_accuracy*100:>9.1f}% {recal_pct:>13.1f}%")

    print(f"\nTotal predictions with recalibration applied: {recal_applied_count}")

    return dict(results_by_season)


def evaluate_simple_xg_with_recalibration(df: pd.DataFrame) -> dict:
    """Test recalibration with simple xG-Poisson model."""
    print("\n" + "=" * 70)
    print("SIMPLE XG-POISSON WITH RECALIBRATION")
    print("=" * 70)

    recal = SeasonalRecalibration(window_size=50, sensitivity=1.2)
    results_by_season = defaultdict(list)

    # Process matches chronologically
    for _, row in df.iterrows():
        if pd.isna(row['home_xg']) or pd.isna(row['home_score']):
            continue

        row['home_team']
        row['away_team']
        home_goals = int(row['home_score'])
        away_goals = int(row['away_score'])
        season = row['season']
        home_xg = float(row['home_xg'])
        away_xg = float(row['away_xg'])

        # Actual outcome
        if home_goals > away_goals:
            actual = 'H'
        elif home_goals < away_goals:
            actual = 'A'
        else:
            actual = 'D'

        # Get recalibration factors
        factors = recal.calculate_recalibration()

        # Adjust rho for Dixon-Coles
        base_rho = -0.11
        adjusted_rho = recal.adjust_rho(base_rho, factors)

        # Calculate Poisson probabilities with Dixon-Coles
        home_prob = 0
        draw_prob = 0
        away_prob = 0

        for h in range(8):
            for a in range(8):
                p = poisson.pmf(h, home_xg) * poisson.pmf(a, away_xg)

                # Dixon-Coles correction
                if h == 0 and a == 0:
                    p *= (1 - home_xg * away_xg * adjusted_rho)
                elif h == 0 and a == 1:
                    p *= (1 + home_xg * adjusted_rho)
                elif h == 1 and a == 0:
                    p *= (1 + away_xg * adjusted_rho)
                elif h == 1 and a == 1:
                    p *= (1 - adjusted_rho)

                if h > a:
                    home_prob += p
                elif h == a:
                    draw_prob += p
                else:
                    away_prob += p

        # Normalize
        total = home_prob + draw_prob + away_prob
        home_prob /= total
        draw_prob /= total
        away_prob /= total

        # Apply probability recalibration
        home_prob, draw_prob, away_prob = recal.adjust_probabilities(
            home_prob, draw_prob, away_prob, factors
        )

        # Predicted outcome
        if home_prob >= draw_prob and home_prob >= away_prob:
            predicted = 'H'
        elif draw_prob >= home_prob and draw_prob >= away_prob:
            predicted = 'D'
        else:
            predicted = 'A'

        results_by_season[season].append({
            'actual': actual,
            'predicted': predicted,
            'correct': predicted == actual,
            'home_prob': home_prob,
            'draw_prob': draw_prob,
            'away_prob': away_prob,
        })

        # Update recalibration tracking AFTER prediction
        recal.add_match(home_goals, away_goals, home_xg, away_xg)

    # Calculate accuracy by season
    print(f"\n{'Season':<12} {'Matches':>8} {'Accuracy':>10} {'Draw Acc':>10}")
    print("-" * 45)

    for season in sorted(results_by_season.keys()):
        results = results_by_season[season]
        if len(results) < 10:
            continue

        accuracy = np.mean([r['correct'] for r in results])
        draw_results = [r for r in results if r['actual'] == 'D']
        draw_accuracy = np.mean([r['correct'] for r in draw_results]) if draw_results else 0

        print(f"{season:<12} {len(results):>8} {accuracy*100:>9.1f}% {draw_accuracy*100:>9.1f}%")

    return dict(results_by_season)


def compare_approaches(baseline: dict, recalibrated: dict):
    """Compare baseline vs recalibrated performance."""
    print("\n" + "=" * 70)
    print("COMPARISON: Baseline vs Recalibrated")
    print("=" * 70)

    print(f"\n{'Season':<12} {'Baseline':>10} {'Recalib':>10} {'Change':>10}")
    print("-" * 45)

    for season in sorted(baseline.keys()):
        if season not in recalibrated:
            continue

        base_results = baseline[season]
        recal_results = recalibrated[season]

        if len(base_results) < 10 or len(recal_results) < 10:
            continue

        base_acc = np.mean([r['correct'] for r in base_results]) * 100
        recal_acc = np.mean([r['correct'] for r in recal_results]) * 100
        change = recal_acc - base_acc

        sign = '+' if change >= 0 else ''
        print(f"{season:<12} {base_acc:>9.1f}% {recal_acc:>9.1f}% {sign}{change:>9.1f}pp")

    # Focus on 2025-26
    if '2025-26' in baseline and '2025-26' in recalibrated:
        print("\n" + "-" * 45)
        print("2025-26 SPECIFIC ANALYSIS:")

        base_25 = baseline['2025-26']
        recal_25 = recalibrated['2025-26']

        base_acc = np.mean([r['correct'] for r in base_25]) * 100
        recal_acc = np.mean([r['correct'] for r in recal_25]) * 100

        # Draw-specific
        base_draws = [r for r in base_25 if r['actual'] == 'D']
        recal_draws = [r for r in recal_25 if r['actual'] == 'D']

        base_draw_acc = np.mean([r['correct'] for r in base_draws]) * 100 if base_draws else 0
        recal_draw_acc = np.mean([r['correct'] for r in recal_draws]) * 100 if recal_draws else 0

        print(f"  Overall accuracy: {base_acc:.1f}% → {recal_acc:.1f}% ({recal_acc - base_acc:+.1f}pp)")
        print(f"  Draw accuracy:    {base_draw_acc:.1f}% → {recal_draw_acc:.1f}% ({recal_draw_acc - base_draw_acc:+.1f}pp)")

        target_met = "YES" if recal_acc >= 52 else "NO"
        print(f"\n  TARGET (52-53%): {target_met} (achieved {recal_acc:.1f}%)")


def evaluate_xg_with_draw_threshold(df: pd.DataFrame) -> dict:
    """Test xG model with draw threshold adjustment (no recalibration)."""
    print("\n" + "=" * 70)
    print("XG-POISSON WITH DRAW THRESHOLD ADJUSTMENT")
    print("=" * 70)

    results_by_season = defaultdict(list)

    for _, row in df.iterrows():
        if pd.isna(row['home_xg']) or pd.isna(row['home_score']):
            continue

        home_goals = int(row['home_score'])
        away_goals = int(row['away_score'])
        season = row['season']
        home_xg = float(row['home_xg'])
        away_xg = float(row['away_xg'])

        if home_goals > away_goals:
            actual = 'H'
        elif home_goals < away_goals:
            actual = 'A'
        else:
            actual = 'D'

        # Poisson with Dixon-Coles
        rho = -0.13
        home_prob = 0
        draw_prob = 0
        away_prob = 0

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
        home_prob /= total
        draw_prob /= total
        away_prob /= total

        # Use draw threshold adjustment for prediction
        predicted = apply_draw_threshold_adjustment(
            home_prob, draw_prob, away_prob,
            draw_threshold=0.26,
            parity_threshold=0.08
        )

        results_by_season[season].append({
            'actual': actual,
            'predicted': predicted,
            'correct': predicted == actual,
            'home_prob': home_prob,
            'draw_prob': draw_prob,
            'away_prob': away_prob,
        })

    print(f"\n{'Season':<12} {'Matches':>8} {'Accuracy':>10} {'Draw Acc':>10} {'Draws Pred':>12}")
    print("-" * 55)

    for season in sorted(results_by_season.keys()):
        results = results_by_season[season]
        if len(results) < 10:
            continue

        accuracy = np.mean([r['correct'] for r in results])
        draw_results = [r for r in results if r['actual'] == 'D']
        draw_accuracy = np.mean([r['correct'] for r in draw_results]) if draw_results else 0
        draws_predicted = sum(1 for r in results if r['predicted'] == 'D')

        print(f"{season:<12} {len(results):>8} {accuracy*100:>9.1f}% {draw_accuracy*100:>9.1f}% {draws_predicted:>12}")

    return dict(results_by_season)


def evaluate_conservative_recal(df: pd.DataFrame) -> dict:
    """Test xG model with conservative recalibration and draw threshold."""
    print("\n" + "=" * 70)
    print("XG-POISSON + CONSERVATIVE RECAL + DRAW THRESHOLD")
    print("=" * 70)

    recal = ConservativeRecalibration(window_size=50)
    results_by_season = defaultdict(list)

    for _, row in df.iterrows():
        if pd.isna(row['home_xg']) or pd.isna(row['home_score']):
            continue

        home_goals = int(row['home_score'])
        away_goals = int(row['away_score'])
        season = row['season']
        home_xg = float(row['home_xg'])
        away_xg = float(row['away_xg'])

        if home_goals > away_goals:
            actual = 'H'
        elif home_goals < away_goals:
            actual = 'A'
        else:
            actual = 'D'

        factors = recal.calculate_recalibration()
        adjusted_rho = recal.adjust_rho(-0.13, factors)

        home_prob = 0
        draw_prob = 0
        away_prob = 0

        for h in range(8):
            for a in range(8):
                p = poisson.pmf(h, home_xg) * poisson.pmf(a, away_xg)

                if h == 0 and a == 0:
                    p *= (1 - home_xg * away_xg * adjusted_rho)
                elif h == 0 and a == 1:
                    p *= (1 + home_xg * adjusted_rho)
                elif h == 1 and a == 0:
                    p *= (1 + away_xg * adjusted_rho)
                elif h == 1 and a == 1:
                    p *= (1 - adjusted_rho)

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

        # Apply probability recalibration
        home_prob, draw_prob, away_prob = recal.adjust_probabilities(
            home_prob, draw_prob, away_prob, factors
        )

        # Use draw threshold for prediction
        predicted = apply_draw_threshold_adjustment(
            home_prob, draw_prob, away_prob,
            draw_threshold=0.26,
            parity_threshold=0.08
        )

        results_by_season[season].append({
            'actual': actual,
            'predicted': predicted,
            'correct': predicted == actual,
        })

        recal.add_match(home_goals, away_goals, home_xg, away_xg)

    print(f"\n{'Season':<12} {'Matches':>8} {'Accuracy':>10} {'Draw Acc':>10}")
    print("-" * 45)

    for season in sorted(results_by_season.keys()):
        results = results_by_season[season]
        if len(results) < 10:
            continue

        accuracy = np.mean([r['correct'] for r in results])
        draw_results = [r for r in results if r['actual'] == 'D']
        draw_accuracy = np.mean([r['correct'] for r in draw_results]) if draw_results else 0

        print(f"{season:<12} {len(results):>8} {accuracy*100:>9.1f}% {draw_accuracy*100:>9.1f}%")

    return dict(results_by_season)


def main():
    # Load data
    df = load_matches_with_xg()

    # Test draw threshold adjustment alone
    print("\n\n")
    draw_thresh_results = evaluate_xg_with_draw_threshold(df)

    # Test conservative recalibration + draw threshold
    print("\n\n")
    conservative_results = evaluate_conservative_recal(df)

    # Also test simple xG approach with original recalibration for comparison
    print("\n\n")
    xg_recal_results = evaluate_simple_xg_with_recalibration(df)

    # Summary comparison for 2025-26
    print("\n" + "=" * 70)
    print("2025-26 SUMMARY COMPARISON")
    print("=" * 70)

    approaches = [
        ("xG + Draw Threshold", draw_thresh_results),
        ("xG + Conservative Recal", conservative_results),
        ("xG + Original Recal", xg_recal_results),
    ]

    print(f"\n{'Approach':<30} {'Accuracy':>10} {'Draw Acc':>10}")
    print("-" * 55)

    for name, results in approaches:
        if '2025-26' in results:
            r = results['2025-26']
            acc = np.mean([x['correct'] for x in r]) * 100
            draws = [x for x in r if x['actual'] == 'D']
            draw_acc = np.mean([x['correct'] for x in draws]) * 100 if draws else 0
            print(f"{name:<30} {acc:>9.1f}% {draw_acc:>9.1f}%")

    print("\n" + "-" * 55)
    print("TARGET: 52-53% accuracy for 2025-26")


if __name__ == "__main__":
    main()
