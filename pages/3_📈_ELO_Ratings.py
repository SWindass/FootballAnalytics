"""ELO Ratings Dashboard Page."""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
scripts_dir = str(project_root / "scripts")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

# Initialize database with Streamlit secrets BEFORE other imports
import db_init  # noqa: F401

import streamlit as st
import plotly.graph_objects as go
from sqlalchemy import select

from app.db.database import SyncSessionLocal
from app.db.models import EloRating, Team
from auth import require_auth, show_user_info
from pwa import inject_pwa_tags

st.set_page_config(page_title="ELO Ratings", page_icon="📈", layout="wide")

# PWA support
inject_pwa_tags()

# Auth check - admin only
require_auth(allowed_roles=["admin"])
show_user_info()

# Team colors
TEAM_COLORS = {
    "Liverpool": "#C8102E", "Man City": "#6CABDD", "Arsenal": "#EF0107",
    "Chelsea": "#034694", "Tottenham": "#132257", "Man United": "#DA291C",
    "Aston Villa": "#95BFE5", "Newcastle": "#241F20", "Brighton Hove": "#0057B8",
    "West Ham": "#7A263A", "Crystal Palace": "#1B458F", "Fulham": "#CC0000",
    "Brentford": "#E30613", "Nottingham": "#DD0000", "Bournemouth": "#B50E12",
    "Wolverhampton": "#FDB913", "Everton": "#003399", "Leicester": "#003090",
    "Southampton": "#D71920", "Ipswich": "#0033A0",
}


@st.cache_data(ttl=60)
def load_data():
    with SyncSessionLocal() as session:
        teams = list(session.execute(select(Team)).scalars().all())
        team_dict = {t.id: {"name": t.name, "short_name": t.short_name} for t in teams}

        stmt = select(EloRating).order_by(EloRating.season, EloRating.matchweek)
        ratings = list(session.execute(stmt).scalars().all())

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
                "rating_change": float(r.rating_change) if r.rating_change else 0,
            })

        return team_dict, data


def get_team_color(short_name: str) -> str:
    return TEAM_COLORS.get(short_name, "#888888")


# Main content
st.title("📈 EPL ELO Ratings Dashboard")
st.markdown("Track team strength across seasons using ELO ratings with cross-season carryover.")

team_dict, ratings_data = load_data()

if not ratings_data:
    st.warning("No ELO ratings found in database. Run the ELO calculation first.")
    st.stop()

# Get unique seasons and teams
seasons = sorted(set(r["season"] for r in ratings_data))

# Sidebar controls
st.sidebar.header("Filters")

view_mode = st.sidebar.radio("View Mode", ["Single Season", "All Seasons"], index=0)

if view_mode == "Single Season":
    selected_season = st.sidebar.selectbox("Season", seasons, index=len(seasons) - 1)
else:
    selected_season = None

# Filter data by season
if view_mode == "Single Season":
    season_data = [r for r in ratings_data if r["season"] == selected_season]
    if not season_data:
        st.warning(f"No data for season {selected_season}")
        st.stop()
else:
    season_data = ratings_data

matchweeks = sorted(set(r["matchweek"] for r in season_data))
min_mw, max_mw = min(matchweeks), max(matchweeks)

# Matchweek selector for single season view
if view_mode == "Single Season":
    selected_matchweek = st.sidebar.slider(
        "View Table at Matchweek",
        min_value=min_mw,
        max_value=max_mw,
        value=max_mw
    )
    mw_range = st.sidebar.slider(
        "Chart Matchweek Range",
        min_value=min_mw,
        max_value=max_mw,
        value=(min_mw, max_mw)
    )
else:
    selected_matchweek = max_mw
    mw_range = (min_mw, max_mw)

# Build league table - get most recent rating for each team up to selected matchweek
def get_latest_ratings(data, up_to_matchweek):
    """Get most recent rating for each team up to a given matchweek."""
    team_latest = {}
    for r in data:
        if r["matchweek"] <= up_to_matchweek:
            team = r["short_name"]
            if team not in team_latest or r["matchweek"] > team_latest[team]["matchweek"]:
                team_latest[team] = r
    return list(team_latest.values())

if view_mode == "Single Season":
    table_data = get_latest_ratings(season_data, selected_matchweek)
else:
    # For all seasons view, use most recent matchweek of the latest season
    latest_season = max(seasons)
    latest_season_data = [r for r in ratings_data if r["season"] == latest_season]
    latest_mw = max(r["matchweek"] for r in latest_season_data)
    table_data = get_latest_ratings(latest_season_data, latest_mw)

# Sort by ELO rating (league position proxy)
table_data = sorted(table_data, key=lambda r: r["rating"], reverse=True)
teams_in_data = [r["short_name"] for r in table_data]

# Initialize selected teams in session state
if "selected_teams" not in st.session_state:
    st.session_state.selected_teams = set(teams_in_data[:6])  # Default top 6

# Quick select buttons
st.sidebar.markdown("**Quick Select**")
col1, col2, col3 = st.sidebar.columns(3)
if col1.button("Top 6", use_container_width=True):
    st.session_state.selected_teams = set(teams_in_data[:6])
    st.rerun()
if col2.button("All", use_container_width=True):
    st.session_state.selected_teams = set(teams_in_data)
    st.rerun()
if col3.button("Clear", use_container_width=True):
    st.session_state.selected_teams = set()
    st.rerun()

# Get currently selected teams
selected_teams = list(st.session_state.selected_teams)

# Build chart with selected teams
fig = go.Figure()

if selected_teams:
    filtered_data = [
        r for r in season_data
        if r["short_name"] in selected_teams
        and mw_range[0] <= r["matchweek"] <= mw_range[1]
    ]

    if view_mode == "Single Season":
        for team_name in selected_teams:
            team_data = sorted([r for r in filtered_data if r["short_name"] == team_name], key=lambda r: r["matchweek"])
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
                hovertemplate=f"<b>{team_name}</b><br>Matchweek: %{{x}}<br>ELO: %{{y:.1f}}<extra></extra>",
            ))

        fig.update_xaxes(dtick=2, range=[mw_range[0] - 0.5, mw_range[1] + 0.5])
        chart_title = f"ELO Ratings - {selected_season} Season"
        x_title = "Matchweek"
    else:
        season_order = sorted(seasons)
        for team_name in selected_teams:
            team_data = sorted([r for r in filtered_data if r["short_name"] == team_name], key=lambda r: (r["season"], r["matchweek"]))
            if not team_data:
                continue

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
                hovertemplate=f"<b>{team_name}</b><br>%{{text}}<br>ELO: %{{y:.1f}}<extra></extra>",
                text=x_labels,
            ))

        for i in range(1, len(season_order)):
            fig.add_vline(x=i * 38 + 0.5, line_dash="dash", line_color="gray", opacity=0.3)

        tickvals = [i * 38 + 19 for i in range(len(season_order))]
        fig.update_xaxes(tickvals=tickvals, ticktext=season_order)
        chart_title = "ELO Ratings - All Seasons (with carryover)"
        x_title = "Season"
else:
    chart_title = "ELO Ratings - Select teams from the table below"
    x_title = "Matchweek"

fig.add_hline(y=1500, line_dash="dash", line_color="gray", opacity=0.5, annotation_text="Baseline (1500)", annotation_position="bottom right")

fig.update_layout(
    title=chart_title,
    xaxis_title=x_title,
    yaxis_title="ELO Rating",
    hovermode="x unified",
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.02),
    height=550,
)

st.plotly_chart(fig, use_container_width=True)

# League table with checkboxes
if view_mode == "Single Season":
    st.subheader(f"League Table by ELO - Matchweek {selected_matchweek}")
else:
    st.subheader(f"Current ELO Rankings")

st.caption("Select teams to add to the chart above")

# Build the full table data
full_table = []
for pos, entry in enumerate(table_data, 1):
    team_name = entry["short_name"]

    # Get season data for this team to calculate stats
    team_season_data = sorted(
        [r for r in season_data if r["short_name"] == team_name],
        key=lambda r: (r["season"], r["matchweek"])
    )

    if team_season_data:
        start_elo = team_season_data[0]["rating"]
        current_elo = entry["rating"]
        max_elo = max(r["rating"] for r in team_season_data)
        min_elo = min(r["rating"] for r in team_season_data)
        peak_entry = next(r for r in team_season_data if r["rating"] == max_elo)
        peak_mw = peak_entry["matchweek"]

        # Get change from previous matchweek
        change = entry.get("rating_change", 0)
    else:
        start_elo = current_elo = max_elo = min_elo = 1500
        peak_mw = 0
        change = 0

    full_table.append({
        "pos": pos,
        "team": team_name,
        "current_elo": current_elo,
        "change": change,
        "start_elo": start_elo,
        "season_change": current_elo - start_elo,
        "peak": max_elo,
        "peak_mw": peak_mw,
        "low": min_elo,
        "range": max_elo - min_elo,
        "selected": team_name in selected_teams,
    })

# Display table with checkboxes using columns
header_cols = st.columns([0.5, 0.8, 2, 1.2, 1, 1.2, 1.2, 1, 1])
header_cols[0].markdown("**#**")
header_cols[1].markdown("**Show**")
header_cols[2].markdown("**Team**")
header_cols[3].markdown("**ELO**")
header_cols[4].markdown("**Chg**")
header_cols[5].markdown("**Season +/-**")
header_cols[6].markdown("**Peak**")
header_cols[7].markdown("**Low**")
header_cols[8].markdown("**Range**")

# Track changes to selection
new_selected = set()

for row in full_table:
    cols = st.columns([0.5, 0.8, 2, 1.2, 1, 1.2, 1.2, 1, 1])

    cols[0].write(f"{row['pos']}")

    # Checkbox for selection
    is_checked = cols[1].checkbox(
        "Show",
        value=row["selected"],
        key=f"check_{row['team']}",
        label_visibility="collapsed"
    )
    if is_checked:
        new_selected.add(row["team"])

    # Team name with color indicator
    color = get_team_color(row["team"])
    cols[2].markdown(f"<span style='color:{color}'>●</span> {row['team']}", unsafe_allow_html=True)

    # Current ELO
    cols[3].write(f"{row['current_elo']:.0f}")

    # Change indicator
    chg = row["change"]
    if chg > 0:
        cols[4].markdown(f"<span style='color:green'>▲ {chg:.0f}</span>", unsafe_allow_html=True)
    elif chg < 0:
        cols[4].markdown(f"<span style='color:red'>▼ {abs(chg):.0f}</span>", unsafe_allow_html=True)
    else:
        cols[4].write("—")

    # Season change
    season_chg = row["season_change"]
    if season_chg > 0:
        cols[5].markdown(f"<span style='color:green'>+{season_chg:.0f}</span>", unsafe_allow_html=True)
    elif season_chg < 0:
        cols[5].markdown(f"<span style='color:red'>{season_chg:.0f}</span>", unsafe_allow_html=True)
    else:
        cols[5].write("0")

    # Peak
    cols[6].write(f"{row['peak']:.0f} (MW{row['peak_mw']})")

    # Low
    cols[7].write(f"{row['low']:.0f}")

    # Range
    cols[8].write(f"{row['range']:.0f}")

# Update session state if selection changed
if new_selected != st.session_state.selected_teams:
    st.session_state.selected_teams = new_selected
    st.rerun()

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
