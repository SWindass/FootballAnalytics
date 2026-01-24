"""Plot ELO ratings over time for EPL teams."""

import matplotlib.pyplot as plt
from sqlalchemy import select

from app.db.database import SyncSessionLocal
from app.db.models import EloRating, Team


def plot_elo_ratings(season: str = "2024-25", output_file: str = "elo_ratings.png"):
    """Generate ELO ratings plot for all teams."""
    with SyncSessionLocal() as session:
        # Get all teams
        teams = {t.id: t.short_name for t in session.execute(select(Team)).scalars().all()}

        # Get all ratings for the season
        stmt = (
            select(EloRating)
            .where(EloRating.season == season)
            .order_by(EloRating.team_id, EloRating.matchweek)
        )
        ratings = list(session.execute(stmt).scalars().all())

        if not ratings:
            print(f"No ratings found for season {season}")
            return

        # Organize by team
        team_data: dict[int, tuple[list[int], list[float]]] = {}
        for rating in ratings:
            if rating.team_id not in team_data:
                team_data[rating.team_id] = ([], [])
            team_data[rating.team_id][0].append(rating.matchweek)
            team_data[rating.team_id][1].append(float(rating.rating))

        # Get final ratings for sorting legend
        final_ratings = {
            team_id: data[1][-1] if data[1] else 1500
            for team_id, data in team_data.items()
        }

        # Color scheme - top teams get distinct colors
        colors = {
            # Top 6 traditional colors
            "Liverpool": "#C8102E",
            "Man City": "#6CABDD",
            "Arsenal": "#EF0107",
            "Chelsea": "#034694",
            "Tottenham": "#132257",
            "Man United": "#DA291C",
            # Other teams
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

        # Create figure
        plt.figure(figsize=(14, 8))
        plt.style.use('seaborn-v0_8-whitegrid')

        # Sort teams by final rating for legend order
        sorted_teams = sorted(team_data.keys(), key=lambda t: final_ratings[t], reverse=True)

        for team_id in sorted_teams:
            matchweeks, elo_values = team_data[team_id]
            team_name = teams.get(team_id, f"Team {team_id}")
            color = colors.get(team_name, "#888888")

            # Thicker lines for top/bottom teams
            final = final_ratings[team_id]
            linewidth = 2.5 if final > 1550 or final < 1400 else 1.2
            alpha = 1.0 if final > 1550 or final < 1400 else 0.6

            plt.plot(
                matchweeks,
                elo_values,
                label=f"{team_name} ({final:.0f})",
                color=color,
                linewidth=linewidth,
                alpha=alpha,
            )

        # Styling
        plt.axhline(y=1500, color='gray', linestyle='--', alpha=0.5, label='_nolegend_')
        plt.xlabel("Matchweek", fontsize=12)
        plt.ylabel("ELO Rating", fontsize=12)
        plt.title(f"EPL ELO Ratings - {season} Season", fontsize=14, fontweight='bold')
        plt.legend(
            loc='center left',
            bbox_to_anchor=(1.02, 0.5),
            fontsize=9,
            title="Team (Final ELO)",
            title_fontsize=10,
        )
        plt.xlim(1, 38)
        plt.xticks(range(1, 39, 2))
        plt.tight_layout()

        # Save
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_file}")

        # Also show path for viewing
        import os
        full_path = os.path.abspath(output_file)
        print(f"Full path: {full_path}")


if __name__ == "__main__":
    plot_elo_ratings()
