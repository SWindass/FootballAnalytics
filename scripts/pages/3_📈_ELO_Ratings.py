"""ELO Ratings Dashboard Page."""

import streamlit as st
import plotly.graph_objects as go
from sqlalchemy import select

from app.db.database import SyncSessionLocal
from app.db.models import EloRating, Team

st.set_page_config(page_title="ELO Ratings", page_icon="📈", layout="wide")

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

# Filter data
if view_mode == "Single Season":
    season_data = [r for r in ratings_data if r["season"] == selected_season]
    if not season_data:
        st.warning(f"No data for season {selected_season}")
        st.stop()
else:
    season_data = ratings_data

matchweeks = sorted(set(r["matchweek"] for r in season_data))
min_mw, max_mw = min(matchweeks), max(matchweeks)

if view_mode == "Single Season":
    mw_range = st.sidebar.slider("Matchweek Range", min_value=min_mw, max_value=max_mw, value=(min_mw, max_mw))
else:
    mw_range = (min_mw, max_mw)

# Get teams sorted by current rating
teams_in_data = sorted(set(r["short_name"] for r in season_data))
latest_season = max(seasons)
latest_data = [r for r in ratings_data if r["season"] == latest_season]

if latest_data:
    max_mw_latest = max(r["matchweek"] for r in latest_data)
    current_ratings = {r["short_name"]: r["rating"] for r in latest_data if r["matchweek"] == max_mw_latest}
    teams_in_data = sorted(teams_in_data, key=lambda t: current_ratings.get(t, 0), reverse=True)
    top_teams = teams_in_data[:6]
else:
    top_teams = teams_in_data[:6]
    current_ratings = {}

# Team selector
st.sidebar.markdown("**Select Teams**")
col1, col2 = st.sidebar.columns(2)
if col1.button("Top 6", use_container_width=True):
    st.session_state.selected_teams = set(top_teams)
if col2.button("Clear", use_container_width=True):
    st.session_state.selected_teams = set()

if "selected_teams" not in st.session_state:
    st.session_state.selected_teams = set(top_teams)

with st.sidebar.expander("Teams", expanded=True):
    selected_teams = []
    for team in teams_in_data:
        default = team in st.session_state.selected_teams
        if st.checkbox(team, value=default, key=f"team_{team}"):
            selected_teams.append(team)
    st.session_state.selected_teams = set(selected_teams)

if not selected_teams:
    st.info("Select at least one team to display.")
    st.stop()

# Filter data
filtered_data = [
    r for r in season_data
    if r["short_name"] in selected_teams
    and mw_range[0] <= r["matchweek"] <= mw_range[1]
]

# Build chart
fig = go.Figure()

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

fig.add_hline(y=1500, line_dash="dash", line_color="gray", opacity=0.5, annotation_text="Baseline (1500)", annotation_position="bottom right")

fig.update_layout(
    title=chart_title,
    xaxis_title=x_title,
    yaxis_title="ELO Rating",
    hovermode="x unified",
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.02),
    height=600,
)

st.plotly_chart(fig, use_container_width=True)

# Stats table
if view_mode == "Single Season":
    st.subheader(f"Season Summary - {selected_season}")
else:
    st.subheader("Overall Summary (All Seasons)")

summary_data = []
for team_name in selected_teams:
    team_data = sorted([r for r in season_data if r["short_name"] == team_name], key=lambda r: (r["season"], r["matchweek"]))
    if not team_data:
        continue

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

summary_data.sort(key=lambda x: float(x["Current"]), reverse=True)
st.dataframe(summary_data, use_container_width=True, hide_index=True)

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
