"""
xG-Based Predictor Evaluation.

Evaluates the Understat xG predictor against Pi+DC, ELO, and builds
an enhanced ensemble with xG data.

Expected improvement: 54.9% → 57-58% accuracy, Brier: 0.5755 → 0.550
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
from batch.models.elo import EloConfig, EloRatingSystem
from batch.models.pi_dixon_coles import PiDixonColesModel
from batch.models.pi_rating import PiRating
from batch.models.xg_predictor import XGPredictor

warnings.filterwarnings("ignore")


def load_matches_with_xg():
    """Load matches that have xG data from database."""
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

        print(f"  Loaded {len(matches)} matches with xG data")

        # Get season breakdown
        seasons = defaultdict(int)
        for m in matches:
            seasons[m['season']] += 1
        print(f"  Seasons: {', '.join(sorted(seasons.keys()))}")

        return matches


def generate_all_predictions(matches: list[dict], warmup: int = 380):
    """
    Generate predictions from all models.

    Returns DataFrame with predictions from xG, Pi+DC, ELO, and Pi.
    """
    print(f"\nGenerating predictions (warmup={warmup} matches)...")

    # Initialize models
    xg_predictor = XGPredictor(window=10, rho=-0.11, home_advantage=1.10)
    elo = EloRatingSystem(EloConfig(k_factor=28.0, home_advantage=50.0))
    pi = PiRating(lambda_param=0.07, gamma_param=0.7)
    pidc = PiDixonColesModel(pi_lambda=0.07, pi_gamma=0.7, rho=-0.11)

    # Team ID mapping for ELO
    team_ids = {}
    next_id = 1

    results = []

    for idx, match in enumerate(matches):
        home = match['home_team']
        away = match['away_team']

        # Ensure teams have IDs for ELO
        if home not in team_ids:
            team_ids[home] = next_id
            next_id += 1
        if away not in team_ids:
            team_ids[away] = next_id
            next_id += 1

        # Get predictions BEFORE updating (to avoid lookahead)
        try:
            # xG prediction
            xg_pred = xg_predictor.predict_match(home, away)

            # Pi+DC prediction
            pidc_pred = pidc.predict_match(home, away, apply_draw_model=pidc.draw_model_trained)

            # Pi baseline
            pi_gd = pi.calculate_expected_goal_diff(home, away)
            pi_h = 1 / (1 + math.exp(-pi_gd * 0.7))
            pi_a = 1 / (1 + math.exp(pi_gd * 0.7))
            pi_d = max(0, 0.28 - 0.1 * abs(pi_gd))
            pi_total = pi_h + pi_d + pi_a
            pi_h, pi_d, pi_a = pi_h / pi_total, pi_d / pi_total, pi_a / pi_total

            # ELO prediction
            elo_h, elo_d, elo_a = elo.match_probabilities(
                team_ids[home], team_ids[away]
            )

            # Actual outcome
            if match['home_score'] > match['away_score']:
                actual = 'H'
            elif match['home_score'] == match['away_score']:
                actual = 'D'
            else:
                actual = 'A'

            # Only record after warmup
            if idx >= warmup:
                results.append({
                    'match_id': match['id'],
                    'season': match['season'],
                    'matchweek': match['matchweek'],
                    'home_team': home,
                    'away_team': away,
                    'home_score': match['home_score'],
                    'away_score': match['away_score'],
                    'home_xg_actual': match['home_xg'],
                    'away_xg_actual': match['away_xg'],
                    'actual': actual,

                    # xG predictions
                    'xg_home': xg_pred.home_prob,
                    'xg_draw': xg_pred.draw_prob,
                    'xg_away': xg_pred.away_prob,
                    'xg_home_xg': xg_pred.home_xg,
                    'xg_away_xg': xg_pred.away_xg,
                    'xg_confidence': xg_pred.confidence,
                    'xg_home_regression': xg_pred.home_regression_risk,
                    'xg_away_regression': xg_pred.away_regression_risk,

                    # Pi+DC predictions
                    'pidc_home': pidc_pred.home_win,
                    'pidc_draw': pidc_pred.draw,
                    'pidc_away': pidc_pred.away_win,

                    # Pi baseline
                    'pi_home': pi_h,
                    'pi_draw': pi_d,
                    'pi_away': pi_a,

                    # ELO predictions
                    'elo_home': elo_h,
                    'elo_draw': elo_d,
                    'elo_away': elo_a,
                })

        except Exception:
            pass

        # Update all models with actual result
        xg_predictor.update_match(
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
            print(f"  Processed {idx + 1}/{len(matches)} matches")

    print(f"  Generated {len(results)} predictions")
    return pd.DataFrame(results)


def calculate_metrics(df: pd.DataFrame, model_prefix: str) -> dict:
    """Calculate accuracy, Brier score, log loss for a model."""
    y_true = []
    y_pred = []
    predictions = []

    for _, row in df.iterrows():
        probs = [
            row[f'{model_prefix}_home'],
            row[f'{model_prefix}_draw'],
            row[f'{model_prefix}_away'],
        ]

        # Normalize
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

    # Accuracy
    accuracy = (predictions == y_true).mean()

    # Brier score
    y_true_onehot = np.zeros((len(y_true), 3))
    y_true_onehot[np.arange(len(y_true)), y_true] = 1
    brier = ((y_pred - y_true_onehot) ** 2).sum(axis=1).mean()

    # Log loss
    y_pred_clipped = np.clip(y_pred, 1e-10, 1 - 1e-10)
    logloss = log_loss(y_true, y_pred_clipped)

    # Draw metrics
    draw_pred = (predictions == 1)
    draw_actual = (y_true == 1)
    draw_precision = draw_pred[draw_actual].sum() / draw_pred.sum() if draw_pred.sum() > 0 else 0
    draw_recall = draw_pred[draw_actual].sum() / draw_actual.sum() if draw_actual.sum() > 0 else 0

    return {
        'accuracy': accuracy,
        'brier': brier,
        'logloss': logloss,
        'draw_precision': draw_precision,
        'draw_recall': draw_recall,
    }


def optimize_ensemble_weights(df: pd.DataFrame, models: list[str]) -> dict[str, float]:
    """Optimize ensemble weights using Brier score."""
    n_models = len(models)

    def ensemble_brier(weights):
        weights = np.array(weights)
        weights = weights / weights.sum()

        total_brier = 0.0

        for _, row in df.iterrows():
            home_prob = sum(w * row[f'{m}_home'] for w, m in zip(weights, models, strict=False))
            draw_prob = sum(w * row[f'{m}_draw'] for w, m in zip(weights, models, strict=False))
            away_prob = sum(w * row[f'{m}_away'] for w, m in zip(weights, models, strict=False))

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

            brier = (
                (home_prob - actual[0])**2 +
                (draw_prob - actual[1])**2 +
                (away_prob - actual[2])**2
            ) / 3

            total_brier += brier

        return total_brier / len(df)

    x0 = [1.0 / n_models] * n_models

    result = minimize(
        ensemble_brier,
        x0=x0,
        bounds=[(0, 1)] * n_models,
        method='L-BFGS-B',
        options={'maxiter': 1000}
    )

    weights = np.array(result.x)
    weights = weights / weights.sum()

    return dict(zip(models, weights, strict=False))


def add_ensemble_predictions(df: pd.DataFrame, weights: dict, prefix: str) -> pd.DataFrame:
    """Add ensemble predictions to dataframe."""
    df = df.copy()

    models = list(weights.keys())

    for outcome in ['home', 'draw', 'away']:
        df[f'{prefix}_{outcome}'] = sum(
            weights[m] * df[f'{m}_{outcome}'] for m in models
        )

    # Normalize
    total = df[f'{prefix}_home'] + df[f'{prefix}_draw'] + df[f'{prefix}_away']
    df[f'{prefix}_home'] /= total
    df[f'{prefix}_draw'] /= total
    df[f'{prefix}_away'] /= total

    return df


def analyze_xg_performance(df: pd.DataFrame):
    """Analyze when xG predictor performs best."""
    print("\n" + "=" * 80)
    print("xG PREDICTOR ANALYSIS")
    print("=" * 80)

    # xG prediction vs actual xG
    print("\nPredicted xG vs Actual xG correlation:")
    home_xg_corr = np.corrcoef(df['xg_home_xg'], df['home_xg_actual'])[0, 1]
    away_xg_corr = np.corrcoef(df['xg_away_xg'], df['away_xg_actual'])[0, 1]
    print(f"  Home xG: {home_xg_corr:.3f}")
    print(f"  Away xG: {away_xg_corr:.3f}")

    # xG prediction vs actual goals
    print("\nPredicted xG vs Actual Goals correlation:")
    home_goals_corr = np.corrcoef(df['xg_home_xg'], df['home_score'])[0, 1]
    away_goals_corr = np.corrcoef(df['xg_away_xg'], df['away_score'])[0, 1]
    print(f"  Home: {home_goals_corr:.3f}")
    print(f"  Away: {away_goals_corr:.3f}")

    # xG vs Pi+DC comparison
    xg_better = 0
    pidc_better = 0

    for _, row in df.iterrows():
        actual_vec = [0, 0, 0]
        if row['actual'] == 'H':
            actual_vec[0] = 1
        elif row['actual'] == 'D':
            actual_vec[1] = 1
        else:
            actual_vec[2] = 1

        xg_brier = (
            (row['xg_home'] - actual_vec[0])**2 +
            (row['xg_draw'] - actual_vec[1])**2 +
            (row['xg_away'] - actual_vec[2])**2
        )

        pidc_brier = (
            (row['pidc_home'] - actual_vec[0])**2 +
            (row['pidc_draw'] - actual_vec[1])**2 +
            (row['pidc_away'] - actual_vec[2])**2
        )

        if xg_brier < pidc_brier:
            xg_better += 1
        elif pidc_brier < xg_brier:
            pidc_better += 1

    print("\nMatch-by-match comparison (xG vs Pi+DC):")
    print(f"  xG better:   {xg_better} ({xg_better/len(df)*100:.1f}%)")
    print(f"  Pi+DC better: {pidc_better} ({pidc_better/len(df)*100:.1f}%)")

    # Regression analysis
    print("\nPerformance on regression candidates:")
    high_regression = df[
        (df['xg_home_regression'].abs() > 0.3) | (df['xg_away_regression'].abs() > 0.3)
    ]
    if len(high_regression) > 20:
        xg_metrics = calculate_metrics(high_regression, 'xg')
        pidc_metrics = calculate_metrics(high_regression, 'pidc')
        print(f"  Matches with regression candidates: {len(high_regression)}")
        print(f"  xG Brier:   {xg_metrics['brier']:.4f}")
        print(f"  Pi+DC Brier: {pidc_metrics['brier']:.4f}")


def main():
    # Load data
    matches = load_matches_with_xg()

    if len(matches) < 500:
        print("Not enough matches with xG data!")
        return

    # Generate predictions
    df = generate_all_predictions(matches, warmup=380)

    if df.empty:
        print("No predictions generated!")
        return

    # Split into train/test by season
    all_seasons = sorted(df['season'].unique())
    print(f"\nSeasons: {', '.join(all_seasons)}")

    # Use last 3 seasons for test
    test_seasons = all_seasons[-3:]
    train_seasons = all_seasons[:-3]

    print(f"Train seasons: {', '.join(train_seasons)}")
    print(f"Test seasons:  {', '.join(test_seasons)}")

    train_df = df[df['season'].isin(train_seasons)]
    test_df = df[df['season'].isin(test_seasons)]

    print(f"\nTrain: {len(train_df)}, Test: {len(test_df)}")

    # Individual model performance
    print("\n" + "=" * 80)
    print("INDIVIDUAL MODEL PERFORMANCE (Test Set)")
    print("=" * 80)

    models = ['xg', 'pidc', 'elo', 'pi']
    model_names = ['xG Model', 'Pi+DC', 'ELO', 'Pi Baseline']

    print(f"\n{'Model':<15} {'Accuracy':>10} {'Brier':>10} {'Log Loss':>10} {'Draw P':>10} {'Draw R':>10}")
    print("-" * 75)

    for model, name in zip(models, model_names, strict=False):
        metrics = calculate_metrics(test_df, model)
        print(f"{name:<15} {metrics['accuracy']*100:>9.1f}% {metrics['brier']:>10.4f} "
              f"{metrics['logloss']:>10.4f} {metrics['draw_precision']*100:>9.1f}% "
              f"{metrics['draw_recall']*100:>9.1f}%")

    # Ensemble optimization
    print("\n" + "=" * 80)
    print("ENSEMBLE OPTIMIZATION")
    print("=" * 80)

    # 3-model ensemble (without xG)
    weights_3model = optimize_ensemble_weights(train_df, ['pidc', 'elo', 'pi'])
    print("\n3-Model Ensemble (Pi+DC, ELO, Pi):")
    for m, w in weights_3model.items():
        print(f"  {m}: {w*100:.1f}%")

    # 4-model ensemble (with xG)
    weights_4model = optimize_ensemble_weights(train_df, ['xg', 'pidc', 'elo', 'pi'])
    print("\n4-Model Ensemble (+xG):")
    for m, w in weights_4model.items():
        print(f"  {m}: {w*100:.1f}%")

    # Add ensemble predictions
    test_df = add_ensemble_predictions(test_df, weights_3model, 'ens3')
    test_df = add_ensemble_predictions(test_df, weights_4model, 'ens4')

    # Compare ensembles
    print("\n" + "=" * 80)
    print("ENSEMBLE COMPARISON (Test Set)")
    print("=" * 80)

    ens3_metrics = calculate_metrics(test_df, 'ens3')
    ens4_metrics = calculate_metrics(test_df, 'ens4')

    print(f"\n{'Ensemble':<25} {'Accuracy':>10} {'Brier':>10} {'Log Loss':>10}")
    print("-" * 65)
    print(f"{'3-Model (no xG)':<25} {ens3_metrics['accuracy']*100:>9.1f}% "
          f"{ens3_metrics['brier']:>10.4f} {ens3_metrics['logloss']:>10.4f}")
    print(f"{'4-Model (with xG)':<25} {ens4_metrics['accuracy']*100:>9.1f}% "
          f"{ens4_metrics['brier']:>10.4f} {ens4_metrics['logloss']:>10.4f}")

    # Improvement calculation
    brier_improvement = (ens3_metrics['brier'] - ens4_metrics['brier']) / ens3_metrics['brier'] * 100
    acc_improvement = (ens4_metrics['accuracy'] - ens3_metrics['accuracy']) * 100

    print("\nxG Contribution:")
    print(f"  Brier improvement: {brier_improvement:+.2f}%")
    print(f"  Accuracy improvement: {acc_improvement:+.2f}pp")

    # Analyze xG performance
    analyze_xg_performance(test_df)

    # Season breakdown
    print("\n" + "=" * 80)
    print("SEASON BREAKDOWN")
    print("=" * 80)

    print(f"\n{'Season':<12} {'Matches':>8} {'xG Acc':>10} {'Pi+DC Acc':>10} {'4-Ens Acc':>10} {'xG Weight':>10}")
    print("-" * 70)

    for season in test_seasons:
        season_df = test_df[test_df['season'] == season]
        if len(season_df) > 0:
            xg_acc = calculate_metrics(season_df, 'xg')['accuracy']
            pidc_acc = calculate_metrics(season_df, 'pidc')['accuracy']
            ens4_acc = calculate_metrics(season_df, 'ens4')['accuracy']
            xg_weight = weights_4model.get('xg', 0) * 100
            print(f"{season:<12} {len(season_df):>8} {xg_acc*100:>9.1f}% "
                  f"{pidc_acc*100:>9.1f}% {ens4_acc*100:>9.1f}% {xg_weight:>9.1f}%")

    # Final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    xg_metrics = calculate_metrics(test_df, 'xg')
    pidc_metrics = calculate_metrics(test_df, 'pidc')

    print(f"""
xG Model (Understat data):
  Accuracy: {xg_metrics['accuracy']*100:.1f}%
  Brier:    {xg_metrics['brier']:.4f}
  Optimal weight in ensemble: {weights_4model.get('xg', 0)*100:.1f}%

Pi+DC Model:
  Accuracy: {pidc_metrics['accuracy']*100:.1f}%
  Brier:    {pidc_metrics['brier']:.4f}

3-Model Ensemble (Pi+DC + ELO + Pi):
  Accuracy: {ens3_metrics['accuracy']*100:.1f}%
  Brier:    {ens3_metrics['brier']:.4f}

4-Model Ensemble (+ xG @ {weights_4model.get('xg', 0)*100:.1f}%):
  Accuracy: {ens4_metrics['accuracy']*100:.1f}%
  Brier:    {ens4_metrics['brier']:.4f}

IMPROVEMENT from xG:
  Accuracy: {ens3_metrics['accuracy']*100:.1f}% → {ens4_metrics['accuracy']*100:.1f}% ({acc_improvement:+.1f}pp)
  Brier:    {ens3_metrics['brier']:.4f} → {ens4_metrics['brier']:.4f} ({brier_improvement:+.1f}%)
""")


if __name__ == "__main__":
    main()
