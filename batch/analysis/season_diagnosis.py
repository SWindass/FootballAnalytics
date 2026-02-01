"""
Diagnose 2025-26 Season Performance Drop.

Investigates:
1. Data quality (xG values, systematic biases)
2. League statistics (goals, draws, home advantage)
3. Implements dynamic recalibration
"""

import warnings

import pandas as pd
from scipy.stats import poisson
from sqlalchemy import text

from app.db.database import SyncSessionLocal

warnings.filterwarnings("ignore")


def load_all_matches():
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


def analyze_league_statistics(df: pd.DataFrame):
    """Calculate and compare league statistics by season."""
    print("\n" + "=" * 80)
    print("LEAGUE STATISTICS BY SEASON")
    print("=" * 80)

    # Filter to seasons with xG data
    df = df[df['home_xg'].notna()].copy()

    # Calculate derived metrics
    df['total_goals'] = df['home_score'] + df['away_score']
    df['total_xg'] = df['home_xg'] + df['away_xg']
    df['is_draw'] = (df['home_score'] == df['away_score']).astype(int)
    df['home_win'] = (df['home_score'] > df['away_score']).astype(int)
    df['away_win'] = (df['home_score'] < df['away_score']).astype(int)

    # Group by season
    season_stats = df.groupby('season').agg({
        'id': 'count',
        'home_score': 'mean',
        'away_score': 'mean',
        'total_goals': 'mean',
        'home_xg': 'mean',
        'away_xg': 'mean',
        'total_xg': 'mean',
        'is_draw': 'mean',
        'home_win': 'mean',
        'away_win': 'mean',
    }).round(3)

    season_stats.columns = [
        'matches', 'home_goals', 'away_goals', 'total_goals',
        'home_xg', 'away_xg', 'total_xg', 'draw_rate', 'home_win_rate', 'away_win_rate'
    ]

    # Calculate xG accuracy
    season_stats['xg_vs_goals'] = (season_stats['total_goals'] - season_stats['total_xg']).round(3)

    print(f"\n{'Season':<10} {'Matches':>8} {'Goals/G':>9} {'xG/G':>8} {'xG Diff':>9} {'Draw%':>8} {'Home%':>8} {'Away%':>8}")
    print("-" * 85)

    for season in sorted(season_stats.index):
        s = season_stats.loc[season]
        print(f"{season:<10} {int(s['matches']):>8} {s['total_goals']:>9.2f} {s['total_xg']:>8.2f} "
              f"{s['xg_vs_goals']:>+9.2f} {s['draw_rate']*100:>7.1f}% "
              f"{s['home_win_rate']*100:>7.1f}% {s['away_win_rate']*100:>7.1f}%")

    # Historical averages (2014-2024)
    historical = season_stats[~season_stats.index.isin(['2025-26'])]
    hist_avg = {
        'goals': historical['total_goals'].mean(),
        'xg': historical['total_xg'].mean(),
        'draw_rate': historical['draw_rate'].mean(),
        'home_win': historical['home_win_rate'].mean(),
        'away_win': historical['away_win_rate'].mean(),
    }

    print("\n" + "-" * 85)
    print(f"{'HISTORICAL':<10} {'':>8} {hist_avg['goals']:>9.2f} {hist_avg['xg']:>8.2f} "
          f"{'':>9} {hist_avg['draw_rate']*100:>7.1f}% "
          f"{hist_avg['home_win']*100:>7.1f}% {hist_avg['away_win']*100:>7.1f}%")

    return season_stats, hist_avg


def analyze_2025_26_detailed(df: pd.DataFrame, hist_avg: dict):
    """Deep dive into 2025-26 season."""
    print("\n" + "=" * 80)
    print("2025-26 SEASON DETAILED ANALYSIS")
    print("=" * 80)

    season_df = df[df['season'] == '2025-26'].copy()

    if len(season_df) == 0:
        print("No 2025-26 data available!")
        return

    print(f"\nMatches played: {len(season_df)}")

    # Basic stats
    total_goals = season_df['home_score'].sum() + season_df['away_score'].sum()
    draws = ((season_df['home_score'] == season_df['away_score']).sum())
    home_wins = ((season_df['home_score'] > season_df['away_score']).sum())
    away_wins = ((season_df['home_score'] < season_df['away_score']).sum())

    goals_per_game = total_goals / len(season_df)
    draw_rate = draws / len(season_df)

    print(f"\nGoals per game:    {goals_per_game:.2f} (historical: {hist_avg['goals']:.2f}, diff: {goals_per_game - hist_avg['goals']:+.2f})")
    print(f"Draw rate:         {draw_rate*100:.1f}% (historical: {hist_avg['draw_rate']*100:.1f}%, diff: {(draw_rate - hist_avg['draw_rate'])*100:+.1f}pp)")
    print(f"Home win rate:     {home_wins/len(season_df)*100:.1f}% (historical: {hist_avg['home_win']*100:.1f}%)")
    print(f"Away win rate:     {away_wins/len(season_df)*100:.1f}% (historical: {hist_avg['away_win']*100:.1f}%)")

    # xG analysis
    if season_df['home_xg'].notna().any():
        xg_df = season_df[season_df['home_xg'].notna()]
        avg_home_xg = xg_df['home_xg'].mean()
        avg_away_xg = xg_df['away_xg'].mean()
        total_xg = avg_home_xg + avg_away_xg

        # Goals vs xG
        actual_home = xg_df['home_score'].mean()
        actual_away = xg_df['away_score'].mean()

        print(f"\nxG Analysis ({len(xg_df)} matches with xG):")
        print(f"  Home xG:    {avg_home_xg:.2f}, Actual: {actual_home:.2f}, Diff: {actual_home - avg_home_xg:+.2f}")
        print(f"  Away xG:    {avg_away_xg:.2f}, Actual: {actual_away:.2f}, Diff: {actual_away - avg_away_xg:+.2f}")
        print(f"  Total xG:   {total_xg:.2f}, Actual: {actual_home + actual_away:.2f}")

    # Outcome distribution
    print("\nOutcome distribution:")
    print(f"  Home wins: {home_wins} ({home_wins/len(season_df)*100:.1f}%)")
    print(f"  Draws:     {draws} ({draws/len(season_df)*100:.1f}%)")
    print(f"  Away wins: {away_wins} ({away_wins/len(season_df)*100:.1f}%)")

    # Score distribution
    print("\nMost common scorelines:")
    season_df['scoreline'] = season_df['home_score'].astype(str) + '-' + season_df['away_score'].astype(str)
    scoreline_counts = season_df['scoreline'].value_counts().head(10)
    for scoreline, count in scoreline_counts.items():
        pct = count / len(season_df) * 100
        print(f"  {scoreline}: {count} ({pct:.1f}%)")

    return {
        'goals_per_game': goals_per_game,
        'draw_rate': draw_rate,
        'home_win_rate': home_wins / len(season_df),
        'away_win_rate': away_wins / len(season_df),
    }


def analyze_prediction_errors(df: pd.DataFrame):
    """Analyze where models are going wrong in 2025-26."""
    print("\n" + "=" * 80)
    print("PREDICTION ERROR ANALYSIS")
    print("=" * 80)

    # Only look at 2025-26
    season_df = df[df['season'] == '2025-26'].copy()

    if len(season_df) == 0:
        print("No 2025-26 data!")
        return

    # Determine actual outcomes
    season_df['actual'] = 'D'
    season_df.loc[season_df['home_score'] > season_df['away_score'], 'actual'] = 'H'
    season_df.loc[season_df['home_score'] < season_df['away_score'], 'actual'] = 'A'

    # Calculate expected outcome probabilities using simple xG model
    results = []

    for _, row in season_df.iterrows():
        if pd.isna(row['home_xg']):
            continue

        home_xg = row['home_xg']
        away_xg = row['away_xg']

        # Simple Poisson probabilities
        home_prob = 0
        draw_prob = 0
        away_prob = 0

        for h in range(8):
            for a in range(8):
                p = poisson.pmf(h, home_xg) * poisson.pmf(a, away_xg)
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

        # Prediction
        if home_prob >= draw_prob and home_prob >= away_prob:
            predicted = 'H'
        elif draw_prob >= home_prob and draw_prob >= away_prob:
            predicted = 'D'
        else:
            predicted = 'A'

        results.append({
            'actual': row['actual'],
            'predicted': predicted,
            'home_prob': home_prob,
            'draw_prob': draw_prob,
            'away_prob': away_prob,
            'correct': predicted == row['actual'],
        })

    results_df = pd.DataFrame(results)

    if len(results_df) == 0:
        return

    print(f"\nSimple xG-Poisson model accuracy: {results_df['correct'].mean()*100:.1f}%")

    # Confusion matrix
    print("\nConfusion matrix (Predicted vs Actual):")
    confusion = pd.crosstab(results_df['predicted'], results_df['actual'], margins=True)
    print(confusion)

    # Error analysis by actual outcome
    print("\nAccuracy by actual outcome:")
    for actual in ['H', 'D', 'A']:
        subset = results_df[results_df['actual'] == actual]
        if len(subset) > 0:
            acc = subset['correct'].mean()
            print(f"  {actual}: {acc*100:.1f}% ({len(subset)} matches)")

    # Calibration check
    print("\nProbability calibration:")
    for outcome, col in [('Home', 'home_prob'), ('Draw', 'draw_prob'), ('Away', 'away_prob')]:
        actual_letter = outcome[0]
        predicted_prob = results_df[col].mean()
        actual_freq = (results_df['actual'] == actual_letter).mean()
        print(f"  {outcome}: Predicted {predicted_prob*100:.1f}%, Actual {actual_freq*100:.1f}% (diff: {(predicted_prob - actual_freq)*100:+.1f}pp)")


def compare_recent_vs_historical(df: pd.DataFrame):
    """Compare last 50 matches to historical baseline."""
    print("\n" + "=" * 80)
    print("RECENT FORM vs HISTORICAL (Last 50 Matches)")
    print("=" * 80)

    # Get last 50 matches
    recent = df.tail(50).copy()

    if len(recent) < 50:
        print(f"Only {len(recent)} recent matches available")

    recent_goals = (recent['home_score'] + recent['away_score']).mean()
    recent_draws = (recent['home_score'] == recent['away_score']).mean()
    recent_home_wins = (recent['home_score'] > recent['away_score']).mean()

    # Historical (excluding recent)
    historical = df.iloc[:-50].copy()
    hist_goals = (historical['home_score'] + historical['away_score']).mean()
    hist_draws = (historical['home_score'] == historical['away_score']).mean()
    hist_home_wins = (historical['home_score'] > historical['away_score']).mean()

    print(f"\n{'Metric':<20} {'Recent 50':>12} {'Historical':>12} {'Difference':>12}")
    print("-" * 60)
    print(f"{'Goals/match':<20} {recent_goals:>12.2f} {hist_goals:>12.2f} {recent_goals - hist_goals:>+12.2f}")
    print(f"{'Draw rate':<20} {recent_draws*100:>11.1f}% {hist_draws*100:>11.1f}% {(recent_draws - hist_draws)*100:>+11.1f}pp")
    print(f"{'Home win rate':<20} {recent_home_wins*100:>11.1f}% {hist_home_wins*100:>11.1f}% {(recent_home_wins - hist_home_wins)*100:>+11.1f}pp")

    # Diagnosis
    print("\n" + "-" * 60)
    print("DIAGNOSIS:")

    issues = []
    if recent_draws > hist_draws + 0.03:
        issues.append(f"Draw rate elevated (+{(recent_draws - hist_draws)*100:.1f}pp)")
    if recent_goals < hist_goals - 0.2:
        issues.append(f"Lower scoring ({recent_goals - hist_goals:.2f} goals/game)")
    if recent_home_wins < hist_home_wins - 0.03:
        issues.append(f"Home advantage weakened ({(recent_home_wins - hist_home_wins)*100:.1f}pp)")

    if issues:
        print("  Detected issues:")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print("  No significant deviations from historical norms")

    return {
        'recent_goals': recent_goals,
        'recent_draws': recent_draws,
        'hist_goals': hist_goals,
        'hist_draws': hist_draws,
    }


def suggest_recalibration(stats: dict):
    """Suggest model recalibration based on recent trends."""
    print("\n" + "=" * 80)
    print("RECOMMENDED RECALIBRATION")
    print("=" * 80)

    recent_goals = stats.get('recent_goals', 2.68)
    recent_draws = stats.get('recent_draws', 0.257)
    hist_goals = stats.get('hist_goals', 2.68)
    hist_draws = stats.get('hist_draws', 0.257)

    print("\nCurrent model parameters:")
    print("  - Dixon-Coles rho: -0.11")
    print("  - League avg goals: 1.35 per team")
    print("  - Home advantage: 10%")

    recommendations = []

    # Dixon-Coles rho adjustment
    draw_diff = recent_draws - hist_draws
    if draw_diff > 0.03:
        new_rho = -0.11 * 1.15  # More negative = more draw correlation
        recommendations.append(f"Increase Dixon-Coles |rho| to {new_rho:.3f} (more draws)")
    elif draw_diff < -0.03:
        new_rho = -0.11 * 0.85
        recommendations.append(f"Decrease Dixon-Coles |rho| to {new_rho:.3f} (fewer draws)")

    # Goals adjustment
    goals_diff = recent_goals - hist_goals
    if goals_diff < -0.2:
        new_avg = 1.35 * (recent_goals / hist_goals)
        recommendations.append(f"Reduce league_avg_goals to {new_avg:.2f} (lower scoring)")
    elif goals_diff > 0.2:
        new_avg = 1.35 * (recent_goals / hist_goals)
        recommendations.append(f"Increase league_avg_goals to {new_avg:.2f} (higher scoring)")

    # Draw boost
    if recent_draws > hist_draws + 0.02:
        boost = (recent_draws / hist_draws - 1) * 100
        recommendations.append(f"Apply {boost:.0f}% draw probability boost")

    print("\nRecommended adjustments:")
    if recommendations:
        for rec in recommendations:
            print(f"  → {rec}")
    else:
        print("  No adjustments needed - recent form matches historical baseline")

    return recommendations


def main():
    # Load data
    df = load_all_matches()

    # Analyze league statistics
    season_stats, hist_avg = analyze_league_statistics(df)

    # Deep dive into 2025-26
    current_stats = analyze_2025_26_detailed(df, hist_avg)

    # Compare recent vs historical
    recent_stats = compare_recent_vs_historical(df)

    # Analyze prediction errors
    analyze_prediction_errors(df)

    # Suggest recalibration
    suggest_recalibration(recent_stats)

    # Final summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if current_stats:
        print(f"""
2025-26 Season Analysis:
  - Goals/game: {current_stats['goals_per_game']:.2f} (historical avg: {hist_avg['goals']:.2f})
  - Draw rate:  {current_stats['draw_rate']*100:.1f}% (historical avg: {hist_avg['draw_rate']*100:.1f}%)

Key Finding:
""")

        goals_diff = current_stats['goals_per_game'] - hist_avg['goals']
        draw_diff = current_stats['draw_rate'] - hist_avg['draw_rate']

        if abs(goals_diff) < 0.15 and abs(draw_diff) < 0.02:
            print("  Season characteristics are NORMAL - poor prediction performance")
            print("  is likely due to model limitations, not data issues.")
        elif goals_diff < -0.2:
            print("  Season is LOWER SCORING than historical average.")
            print("  Models may be overestimating attacking strength.")
        elif draw_diff > 0.03:
            print("  Season has MORE DRAWS than historical average.")
            print("  Dixon-Coles rho should be adjusted to increase draw probability.")
        else:
            print("  Season characteristics are within normal range.")


if __name__ == "__main__":
    main()
