"""Plot Poisson team strengths and prediction accuracy over time."""

import argparse

import matplotlib.pyplot as plt
from sqlalchemy import select

from app.db.database import SyncSessionLocal
from app.db.models import Match, MatchAnalysis, MatchStatus, Team
from batch.jobs.calculate_poisson import calculate_team_strengths_for_season


def plot_team_strengths(season: str = "2024-25", output_file: str = "poisson_strengths.png"):
    """Plot attack vs defense strengths for all teams."""
    with SyncSessionLocal() as session:
        teams = {t.id: t.short_name for t in session.execute(select(Team)).scalars().all()}

    # Get current strengths (all completed matches)
    strengths = calculate_team_strengths_for_season(season)

    if not strengths:
        print(f"No data found for season {season}")
        return

    # Team colors
    colors = {
        "Liverpool": "#C8102E",
        "Man City": "#6CABDD",
        "Arsenal": "#EF0107",
        "Chelsea": "#034694",
        "Tottenham": "#132257",
        "Man United": "#DA291C",
        "Aston Villa": "#95BFE5",
        "Newcastle": "#241F20",
        "Brighton Hove": "#0057B8",
        "West Ham": "#7A263A",
        "Crystal Palace": "#1B458F",
        "Fulham": "#000000",
        "Brentford": "#E30613",
        "Nottingham": "#DD0000",
        "Bournemouth": "#DA291C",
        "Wolverhampton": "#FDB913",
        "Everton": "#003399",
        "Leicester": "#003090",
        "Southampton": "#D71920",
        "Ipswich": "#0033A0",
    }

    # Create scatter plot
    plt.figure(figsize=(12, 10))
    plt.style.use('seaborn-v0_8-whitegrid')

    for team_id, stats in strengths.items():
        team_name = teams.get(team_id, f"Team {team_id}")
        color = colors.get(team_name, "#888888")

        plt.scatter(
            stats["attack"],
            stats["defense"],
            s=150,
            c=color,
            edgecolors='white',
            linewidths=1.5,
            zorder=3,
        )
        plt.annotate(
            team_name,
            (stats["attack"], stats["defense"]),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=8,
            fontweight='bold',
        )

    # Reference lines at 1.0 (league average)
    plt.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

    # Quadrant labels
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    plt.text(xlim[1] - 0.05, ylim[0] + 0.05, "Strong Attack\nStrong Defense",
             ha='right', va='bottom', fontsize=9, alpha=0.6, style='italic')
    plt.text(xlim[0] + 0.05, ylim[0] + 0.05, "Weak Attack\nStrong Defense",
             ha='left', va='bottom', fontsize=9, alpha=0.6, style='italic')
    plt.text(xlim[1] - 0.05, ylim[1] - 0.05, "Strong Attack\nWeak Defense",
             ha='right', va='top', fontsize=9, alpha=0.6, style='italic')
    plt.text(xlim[0] + 0.05, ylim[1] - 0.05, "Weak Attack\nWeak Defense",
             ha='left', va='top', fontsize=9, alpha=0.6, style='italic')

    plt.xlabel("Attack Strength (>1 = above league avg)", fontsize=12)
    plt.ylabel("Defense Strength (>1 = concedes more than avg)", fontsize=12)
    plt.title(f"Poisson Team Strengths - {season} Season", fontsize=14, fontweight='bold')
    plt.tight_layout()

    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_file}")


def plot_strength_progression(
    season: str = "2024-25",
    output_file: str = "poisson_progression.png",
    top_n: int = 6,
):
    """Plot how team strengths evolve over matchweeks."""
    with SyncSessionLocal() as session:
        teams = {t.id: t.short_name for t in session.execute(select(Team)).scalars().all()}

        # Get all matchweeks
        stmt = (
            select(Match.matchweek)
            .where(Match.season == season)
            .where(Match.status == MatchStatus.FINISHED)
            .distinct()
            .order_by(Match.matchweek)
        )
        matchweeks = [mw for (mw,) in session.execute(stmt).all()]

    if len(matchweeks) < 3:
        print("Need at least 3 matchweeks of data")
        return

    # Calculate strengths at each matchweek checkpoint
    checkpoints = [mw for mw in matchweeks if mw >= 5]  # Start from MW5 for stability
    team_attack_history: dict[int, list[tuple[int, float]]] = {}
    team_defense_history: dict[int, list[tuple[int, float]]] = {}

    for mw in checkpoints:
        strengths = calculate_team_strengths_for_season(season, up_to_matchweek=mw + 1)
        for team_id, stats in strengths.items():
            if team_id not in team_attack_history:
                team_attack_history[team_id] = []
                team_defense_history[team_id] = []
            team_attack_history[team_id].append((mw, stats["attack"]))
            team_defense_history[team_id].append((mw, stats["defense"]))

    # Get final strengths to select top teams
    final_strengths = calculate_team_strengths_for_season(season)
    sorted_by_attack = sorted(
        final_strengths.items(),
        key=lambda x: x[1]["attack"],
        reverse=True,
    )
    top_teams = [t[0] for t in sorted_by_attack[:top_n]]
    bottom_teams = [t[0] for t in sorted_by_attack[-3:]]
    teams_to_plot = top_teams + bottom_teams

    colors = {
        "Liverpool": "#C8102E",
        "Man City": "#6CABDD",
        "Arsenal": "#EF0107",
        "Chelsea": "#034694",
        "Tottenham": "#132257",
        "Man United": "#DA291C",
        "Aston Villa": "#95BFE5",
        "Newcastle": "#241F20",
        "Brighton Hove": "#0057B8",
        "West Ham": "#7A263A",
        "Crystal Palace": "#1B458F",
        "Fulham": "#000000",
        "Brentford": "#E30613",
        "Nottingham": "#DD0000",
        "Bournemouth": "#DA291C",
        "Wolverhampton": "#FDB913",
        "Everton": "#003399",
        "Leicester": "#003090",
        "Southampton": "#D71920",
        "Ipswich": "#0033A0",
    }

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    plt.style.use('seaborn-v0_8-whitegrid')

    # Attack strength progression
    for team_id in teams_to_plot:
        if team_id not in team_attack_history:
            continue
        team_name = teams.get(team_id, f"Team {team_id}")
        color = colors.get(team_name, "#888888")
        mws, attacks = zip(*team_attack_history[team_id])
        ax1.plot(mws, attacks, label=team_name, color=color, linewidth=2, marker='o', markersize=4)

    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel("Matchweek", fontsize=11)
    ax1.set_ylabel("Attack Strength", fontsize=11)
    ax1.set_title("Attack Strength Progression", fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=8)

    # Defense strength progression
    for team_id in teams_to_plot:
        if team_id not in team_defense_history:
            continue
        team_name = teams.get(team_id, f"Team {team_id}")
        color = colors.get(team_name, "#888888")
        mws, defenses = zip(*team_defense_history[team_id])
        ax2.plot(mws, defenses, label=team_name, color=color, linewidth=2, marker='o', markersize=4)

    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel("Matchweek", fontsize=11)
    ax2.set_ylabel("Defense Strength (lower = better)", fontsize=11)
    ax2.set_title("Defense Strength Progression", fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=8)

    plt.suptitle(f"Poisson Strength Progression - {season}", fontsize=14, fontweight='bold')
    plt.tight_layout()

    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_file}")


def plot_prediction_accuracy(season: str = "2024-25", output_file: str = "poisson_accuracy.png"):
    """Plot Poisson prediction accuracy over matchweeks."""
    with SyncSessionLocal() as session:
        # Get finished matches with predictions
        stmt = (
            select(Match, MatchAnalysis)
            .join(MatchAnalysis, Match.id == MatchAnalysis.match_id)
            .where(Match.season == season)
            .where(Match.status == MatchStatus.FINISHED)
            .where(MatchAnalysis.poisson_home_prob.isnot(None))
            .order_by(Match.matchweek)
        )
        results = list(session.execute(stmt).all())

    if not results:
        print("No matches with Poisson predictions found")
        return

    # Track accuracy by matchweek
    matchweek_stats: dict[int, dict] = {}

    for match, analysis in results:
        mw = match.matchweek
        if mw not in matchweek_stats:
            matchweek_stats[mw] = {"correct": 0, "total": 0, "brier": 0.0}

        # Determine actual result
        if match.home_score > match.away_score:
            actual = "home"
            actual_probs = (1, 0, 0)
        elif match.home_score < match.away_score:
            actual = "away"
            actual_probs = (0, 0, 1)
        else:
            actual = "draw"
            actual_probs = (0, 1, 0)

        # Determine predicted result
        probs = {
            "home": float(analysis.poisson_home_prob),
            "draw": float(analysis.poisson_draw_prob),
            "away": float(analysis.poisson_away_prob),
        }
        predicted = max(probs, key=probs.get)

        # Update stats
        matchweek_stats[mw]["total"] += 1
        if predicted == actual:
            matchweek_stats[mw]["correct"] += 1

        # Brier score
        brier = (
            (probs["home"] - actual_probs[0]) ** 2
            + (probs["draw"] - actual_probs[1]) ** 2
            + (probs["away"] - actual_probs[2]) ** 2
        )
        matchweek_stats[mw]["brier"] += brier

    # Calculate cumulative accuracy
    mws = sorted(matchweek_stats.keys())
    accuracies = []
    brier_scores = []
    cumulative_correct = 0
    cumulative_total = 0
    cumulative_brier = 0

    for mw in mws:
        cumulative_correct += matchweek_stats[mw]["correct"]
        cumulative_total += matchweek_stats[mw]["total"]
        cumulative_brier += matchweek_stats[mw]["brier"]
        accuracies.append(cumulative_correct / cumulative_total * 100)
        brier_scores.append(cumulative_brier / cumulative_total)

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    plt.style.use('seaborn-v0_8-whitegrid')

    # Accuracy plot
    ax1.plot(mws, accuracies, color='#2E86AB', linewidth=2.5, marker='o', markersize=5)
    ax1.axhline(y=33.3, color='gray', linestyle='--', alpha=0.5, label='Random guess (33.3%)')
    ax1.fill_between(mws, 33.3, accuracies, alpha=0.3, color='#2E86AB')
    ax1.set_ylabel("Cumulative Accuracy (%)", fontsize=11)
    ax1.set_title("Prediction Accuracy Over Season", fontsize=12, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.set_ylim(0, 70)

    # Brier score plot
    ax2.plot(mws, brier_scores, color='#E94F37', linewidth=2.5, marker='o', markersize=5)
    ax2.axhline(y=0.667, color='gray', linestyle='--', alpha=0.5, label='Random guess Brier')
    ax2.set_xlabel("Matchweek", fontsize=11)
    ax2.set_ylabel("Cumulative Brier Score (lower = better)", fontsize=11)
    ax2.set_title("Brier Score Over Season", fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right')

    plt.suptitle(f"Poisson Model Performance - {season}", fontsize=14, fontweight='bold')
    plt.tight_layout()

    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_file}")

    # Print summary
    print(f"\nSeason Summary:")
    print(f"  Matches predicted: {cumulative_total}")
    print(f"  Correct predictions: {cumulative_correct}")
    print(f"  Accuracy: {accuracies[-1]:.1f}%")
    print(f"  Brier Score: {brier_scores[-1]:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Poisson visualizations")
    parser.add_argument("--season", type=str, default="2024-25", help="Season to plot")
    parser.add_argument(
        "--type",
        choices=["strengths", "progression", "accuracy", "all"],
        default="all",
        help="Type of plot to generate",
    )

    args = parser.parse_args()

    if args.type in ("strengths", "all"):
        plot_team_strengths(args.season)

    if args.type in ("progression", "all"):
        plot_strength_progression(args.season)

    if args.type in ("accuracy", "all"):
        plot_prediction_accuracy(args.season)
