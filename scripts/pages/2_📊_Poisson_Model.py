"""Poisson Model Dashboard Page - wrapper for multi-page app."""

import streamlit as st
import plotly.graph_objects as go
from sqlalchemy import select

from app.db.database import SyncSessionLocal
from app.db.models import Match, MatchAnalysis, MatchStatus, Team
from batch.jobs.calculate_poisson import calculate_team_strengths_for_season
from auth import require_auth, show_user_info
from pwa import inject_pwa_tags

st.set_page_config(page_title="Poisson Model", page_icon="📊", layout="wide")

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
def load_teams():
    with SyncSessionLocal() as session:
        teams = list(session.execute(select(Team)).scalars().all())
        return {t.id: {"name": t.name, "short_name": t.short_name} for t in teams}


@st.cache_data(ttl=60)
def load_seasons():
    with SyncSessionLocal() as session:
        stmt = select(Match.season).distinct().order_by(Match.season)
        return [s for (s,) in session.execute(stmt).all()]


@st.cache_data(ttl=60)
def load_matchweeks(season: str):
    with SyncSessionLocal() as session:
        stmt = (
            select(Match.matchweek)
            .where(Match.season == season)
            .where(Match.status == MatchStatus.FINISHED)
            .distinct()
            .order_by(Match.matchweek)
        )
        return [mw for (mw,) in session.execute(stmt).all()]


@st.cache_data(ttl=60)
def get_team_strengths(season: str, up_to_matchweek: int = None):
    return calculate_team_strengths_for_season(season, up_to_matchweek)


def get_team_color(short_name: str) -> str:
    return TEAM_COLORS.get(short_name, "#888888")


# Main content
st.title("📊 Poisson Model Dashboard")
st.markdown("Team attack/defense strengths and prediction analysis using the Poisson distribution model.")

teams = load_teams()
seasons = load_seasons()

if not seasons:
    st.warning("No seasons found in database.")
    st.stop()

# Sidebar
st.sidebar.header("Settings")
selected_season = st.sidebar.selectbox("Season", seasons, index=len(seasons) - 1)

matchweeks = load_matchweeks(selected_season)
if not matchweeks:
    st.warning(f"No finished matches for season {selected_season}")
    st.stop()

# Tabs
tab1, tab2 = st.tabs(["Team Strengths", "Strength Rankings"])

with tab1:
    st.subheader("Attack vs Defense Strength")

    col1, col2 = st.columns([1, 3])
    with col1:
        mw_option = st.radio("Data up to", ["Current", "Select Matchweek"], key="strength_mw_option")

    with col2:
        if mw_option == "Select Matchweek":
            selected_mw = st.slider("Matchweek", min_value=min(matchweeks), max_value=max(matchweeks), value=max(matchweeks))
        else:
            selected_mw = None

    strengths = get_team_strengths(selected_season, selected_mw)

    if not strengths:
        st.info("Not enough match data to calculate strengths yet.")
    else:
        fig = go.Figure()

        for team_id, stats in strengths.items():
            team_info = teams.get(team_id, {})
            team_name = team_info.get("short_name", f"Team {team_id}")
            color = get_team_color(team_name)

            fig.add_trace(go.Scatter(
                x=[stats["attack"]],
                y=[stats["defense"]],
                mode='markers+text',
                name=team_name,
                text=[team_name],
                textposition="top center",
                marker=dict(size=15, color=color, line=dict(width=2, color='white')),
                hovertemplate=(
                    f"<b>{team_name}</b><br>"
                    f"Attack: {stats['attack']:.3f}<br>"
                    f"Defense: {stats['defense']:.3f}<br>"
                    f"Games: {stats['games']}<br>"
                    "<extra></extra>"
                ),
                showlegend=False,
            ))

        fig.add_hline(y=1.0, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_vline(x=1.0, line_dash="dash", line_color="gray", opacity=0.5)

        fig.add_annotation(x=1.3, y=0.7, text="Strong Attack<br>Strong Defense", showarrow=False, opacity=0.5)
        fig.add_annotation(x=0.7, y=0.7, text="Weak Attack<br>Strong Defense", showarrow=False, opacity=0.5)
        fig.add_annotation(x=1.3, y=1.3, text="Strong Attack<br>Weak Defense", showarrow=False, opacity=0.5)
        fig.add_annotation(x=0.7, y=1.3, text="Weak Attack<br>Weak Defense", showarrow=False, opacity=0.5)

        fig.update_layout(
            title=f"Team Strengths - {selected_season}" + (f" (up to MW{selected_mw})" if selected_mw else ""),
            xaxis_title="Attack Strength (>1 = above league avg)",
            yaxis_title="Defense Strength (>1 = concedes more)",
            height=600,
        )

        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Strength Rankings")

    strengths = get_team_strengths(selected_season, None)

    if strengths:
        table_data = []
        for team_id, stats in strengths.items():
            team_info = teams.get(team_id, {})
            table_data.append({
                "Team": team_info.get("short_name", f"Team {team_id}"),
                "Attack": round(stats["attack"], 3),
                "Defense": round(stats["defense"], 3),
                "Games": stats["games"],
                "Net": round(stats["attack"] - stats["defense"], 3),
            })

        table_data.sort(key=lambda x: x["Net"], reverse=True)
        st.dataframe(table_data, use_container_width=True, hide_index=True)

# Info box
with st.sidebar.expander("About Poisson Model"):
    st.markdown("""
    **Poisson Distribution Model**

    Predicts match outcomes based on:
    - **Attack Strength**: Goals scored vs league avg
    - **Defense Strength**: Goals conceded vs league avg
    - **Home Advantage**: +0.25 expected goals

    **Interpretation**:
    - Attack > 1.0 = scores more than average
    - Defense < 1.0 = concedes less than average
    - Defense > 1.0 = leaky defense
    """)
