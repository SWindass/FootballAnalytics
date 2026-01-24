"""Interactive ELO ratings dashboard."""

import streamlit as st
import plotly.graph_objects as go
from sqlalchemy import select

from app.db.database import SyncSessionLocal
from app.db.models import EloRating, Team


# Team colors for consistent styling
TEAM_COLORS = {
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
    "Fulham": "#CC0000",
    "Brentford": "#E30613",
    "Nottingham": "#DD0000",
    "Bournemouth": "#B50E12",
    "Wolverhampton": "#FDB913",
    "Everton": "#003399",
    "Leicester": "#003090",
    "Southampton": "#D71920",
    "Ipswich": "#0033A0",
}


@st.cache_data(ttl=60)
def load_data():
    """Load teams and ELO ratings from database."""
    with SyncSessionLocal() as session:
        # Get all teams
        teams = list(session.execute(select(Team)).scalars().all())
        team_dict = {t.id: {"name": t.name, "short_name": t.short_name} for t in teams}

        # Get all ratings
        stmt = select(EloRating).order_by(EloRating.season, EloRating.matchweek)
        ratings = list(session.execute(stmt).scalars().all())

        # Organize data
        data = []
        for r in ratings:
            team_info = team_dict.get(r.team_id, {})
            data.append({
                "team_id": r.team_id,
                "team_name": team_info.get("name", f"Team {r.team_id}"),
                "short_name": team_info.get("short_name", f"Team {r.team_id}"),
                "season": r.season,
                "matchweek": r.matchweek,
                "rating": float(r.rating),
            })

        return team_dict, data


def get_team_color(short_name: str) -> str:
    """Get color for a team."""
    return TEAM_COLORS.get(short_name, "#888888")


def main():
    st.set_page_config(
        page_title="EPL ELO Ratings",
        page_icon="",
        layout="wide",
    )

    st.title("EPL ELO Ratings Dashboard")
    st.markdown("Track team strength across seasons using ELO ratings with cross-season carryover.")

    # Load data
    team_dict, ratings_data = load_data()

    if not ratings_data:
        st.warning("No ELO ratings found in database. Run the ELO calculation first.")
        return

    # Get unique seasons and teams
    seasons = sorted(set(r["season"] for r in ratings_data))
    teams_by_name = {t["short_name"]: tid for tid, t in team_dict.items()}

    # Sidebar controls
    st.sidebar.header("Filters")

    # View mode
    view_mode = st.sidebar.radio(
        "View Mode",
        ["Single Season", "All Seasons"],
        index=0,
    )

    # Season selector (for single season mode)
    if view_mode == "Single Season":
        selected_season = st.sidebar.selectbox(
            "Season",
            seasons,
            index=len(seasons) - 1,  # Default to latest
        )
    else:
        selected_season = None  # All seasons

    # Filter data for selected season(s)
    if view_mode == "Single Season":
        season_data = [r for r in ratings_data if r["season"] == selected_season]
        if not season_data:
            st.warning(f"No data for season {selected_season}")
            return
    else:
        season_data = ratings_data  # All data

    # Get matchweek range
    matchweeks = sorted(set(r["matchweek"] for r in season_data))
    min_mw, max_mw = min(matchweeks), max(matchweeks)

    if view_mode == "Single Season":
        # Matchweek range slider for single season
        mw_range = st.sidebar.slider(
            "Matchweek Range",
            min_value=min_mw,
            max_value=max_mw,
            value=(min_mw, max_mw),
        )
    else:
        mw_range = (min_mw, max_mw)  # All matchweeks in multi-season view

    # Get teams with data, sorted by current rating
    teams_in_data = sorted(set(r["short_name"] for r in season_data))

    # Calculate current ratings for sorting and default selection
    latest_season = max(seasons)
    latest_data = [r for r in ratings_data if r["season"] == latest_season]
    if latest_data:
        max_mw_latest = max(r["matchweek"] for r in latest_data)
        current_ratings = {}
        for r in latest_data:
            if r["matchweek"] == max_mw_latest:
                current_ratings[r["short_name"]] = r["rating"]
        # Sort teams by rating (highest first)
        teams_in_data = sorted(teams_in_data, key=lambda t: current_ratings.get(t, 0), reverse=True)
        top_teams = teams_in_data[:6]
    else:
        top_teams = teams_in_data[:6]
        current_ratings = {}

    # Team selector with checkboxes
    st.sidebar.markdown("**Select Teams**")

    # Quick select buttons
    col1, col2 = st.sidebar.columns(2)
    if col1.button("Top 6", use_container_width=True):
        st.session_state.selected_teams = set(top_teams)
    if col2.button("Clear", use_container_width=True):
        st.session_state.selected_teams = set()

    # Initialize session state for selected teams
    if "selected_teams" not in st.session_state:
        st.session_state.selected_teams = set(top_teams)

    # Team checkboxes in expander
    with st.sidebar.expander("Teams", expanded=True):
        selected_teams = []
        for team in teams_in_data:
            default = team in st.session_state.selected_teams
            if st.checkbox(team, value=default, key=f"team_{team}"):
                selected_teams.append(team)

        # Update session state
        st.session_state.selected_teams = set(selected_teams)

    if not selected_teams:
        st.info("Select at least one team to display.")
        return

    # Filter data
    filtered_data = [
        r for r in season_data
        if r["short_name"] in selected_teams
        and mw_range[0] <= r["matchweek"] <= mw_range[1]
    ]

    # Build chart
    fig = go.Figure()

    if view_mode == "Single Season":
        # Single season view - x-axis is matchweek
        for team_name in selected_teams:
            team_data = [r for r in filtered_data if r["short_name"] == team_name]
            team_data.sort(key=lambda r: r["matchweek"])

            if not team_data:
                continue

            mws = [r["matchweek"] for r in team_data]
            elos = [r["rating"] for r in team_data]
            final_elo = elos[-1] if elos else 1500

            fig.add_trace(go.Scatter(
                x=mws,
                y=elos,
                mode='lines+markers',
                name=f"{team_name} ({final_elo:.0f})",
                line=dict(color=get_team_color(team_name), width=2.5),
                marker=dict(size=6),
                hovertemplate=(
                    f"<b>{team_name}</b><br>"
                    "Matchweek: %{x}<br>"
                    "ELO: %{y:.1f}<br>"
                    "<extra></extra>"
                ),
            ))

        fig.update_xaxes(dtick=2, range=[mw_range[0] - 0.5, mw_range[1] + 0.5])
        chart_title = f"ELO Ratings - {selected_season} Season"
        x_title = "Matchweek"

    else:
        # Multi-season view - x-axis is season + matchweek label
        season_order = sorted(seasons)

        for team_name in selected_teams:
            team_data = [r for r in filtered_data if r["short_name"] == team_name]
            # Sort by season then matchweek
            team_data.sort(key=lambda r: (r["season"], r["matchweek"]))

            if not team_data:
                continue

            # Create x-axis labels as "Season MW#" and numeric x for plotting
            x_vals = []
            x_labels = []
            elos = []
            for r in team_data:
                season_idx = season_order.index(r["season"])
                x_numeric = season_idx * 38 + r["matchweek"]
                x_vals.append(x_numeric)
                x_labels.append(f"{r['season']} MW{r['matchweek']}")
                elos.append(r["rating"])

            final_elo = elos[-1] if elos else 1500

            fig.add_trace(go.Scatter(
                x=x_vals,
                y=elos,
                mode='lines',
                name=f"{team_name} ({final_elo:.0f})",
                line=dict(color=get_team_color(team_name), width=2),
                hovertemplate=(
                    f"<b>{team_name}</b><br>"
                    "%{text}<br>"
                    "ELO: %{y:.1f}<br>"
                    "<extra></extra>"
                ),
                text=x_labels,
            ))

        # Add vertical lines for season boundaries
        for i, season in enumerate(season_order[1:], 1):
            fig.add_vline(
                x=i * 38 + 0.5,
                line_dash="dash",
                line_color="gray",
                opacity=0.3,
            )

        # Custom x-axis ticks at season starts
        tickvals = [i * 38 + 19 for i in range(len(season_order))]
        ticktext = season_order
        fig.update_xaxes(tickvals=tickvals, ticktext=ticktext)

        chart_title = "ELO Ratings - All Seasons (with carryover)"
        x_title = "Season"

    # Add baseline
    fig.add_hline(
        y=1500,
        line_dash="dash",
        line_color="gray",
        opacity=0.5,
        annotation_text="Baseline (1500)",
        annotation_position="bottom right",
    )

    # Layout
    fig.update_layout(
        title=chart_title,
        xaxis_title=x_title,
        yaxis_title="ELO Rating",
        hovermode="x unified",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02,
        ),
        height=600,
    )

    # Display chart
    st.plotly_chart(fig, use_container_width=True)

    # Stats table
    if view_mode == "Single Season":
        st.subheader(f"Season Summary - {selected_season}")
        summary_source = [r for r in season_data if r["short_name"] in selected_teams]
    else:
        st.subheader("Overall Summary (All Seasons)")
        summary_source = [r for r in season_data if r["short_name"] in selected_teams]

    summary_data = []
    for team_name in selected_teams:
        team_data = [r for r in summary_source if r["short_name"] == team_name]
        if not team_data:
            continue

        team_data.sort(key=lambda r: (r["season"], r["matchweek"]))
        start_elo = team_data[0]["rating"]
        end_elo = team_data[-1]["rating"]
        max_elo = max(r["rating"] for r in team_data)
        min_elo = min(r["rating"] for r in team_data)
        peak_entry = next(r for r in team_data if r["rating"] == max_elo)
        peak_label = f"MW{peak_entry['matchweek']}" if view_mode == "Single Season" else f"{peak_entry['season'][:4]} MW{peak_entry['matchweek']}"

        summary_data.append({
            "Team": team_name,
            "Start": f"{start_elo:.0f}",
            "Current": f"{end_elo:.0f}",
            "Change": f"{end_elo - start_elo:+.0f}",
            "Peak": f"{max_elo:.0f} ({peak_label})",
            "Low": f"{min_elo:.0f}",
            "Range": f"{max_elo - min_elo:.0f}",
        })

    # Sort by current rating
    summary_data.sort(key=lambda x: float(x["Current"]), reverse=True)

    st.dataframe(
        summary_data,
        use_container_width=True,
        hide_index=True,
    )

    # Info box
    with st.expander("About ELO Ratings"):
        st.markdown("""
        **ELO Rating System**

        - All teams start at **1500** (league average)
        - Ratings change after each match based on:
          - Expected vs actual result
          - Goal difference (larger margins = bigger changes)
          - Home advantage (~65 points boost)
        - **K-factor**: 32 (maximum points exchanged per match)

        **Interpretation**:
        - **1600+**: Elite team, title contender
        - **1550-1600**: Top 4 quality
        - **1500-1550**: Upper mid-table
        - **1450-1500**: Lower mid-table
        - **<1450**: Relegation battle
        """)


if __name__ == "__main__":
    main()
