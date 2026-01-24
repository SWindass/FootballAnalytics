"""Upcoming fixtures dashboard with predictions and value bets."""

import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta, timezone
from sqlalchemy import select

from app.db.database import SyncSessionLocal
from app.db.models import Match, MatchAnalysis, MatchStatus, Team, OddsHistory, ValueBet


# Team colors
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
    "Leicester City": "#003090",
    "Southampton": "#D71920",
    "Ipswich Town": "#0033A0",
    "Leeds United": "#FFCD00",
}


st.set_page_config(page_title="Upcoming Fixtures", page_icon="📅", layout="wide")


@st.cache_data(ttl=60)
def load_teams():
    """Load teams from database."""
    with SyncSessionLocal() as session:
        teams = list(session.execute(select(Team)).scalars().all())
        return {t.id: {"name": t.name, "short_name": t.short_name} for t in teams}


@st.cache_data(ttl=60)
def load_upcoming_fixtures(days: int = 7):
    """Load upcoming fixtures with predictions."""
    with SyncSessionLocal() as session:
        now = datetime.now(timezone.utc)

        stmt = (
            select(Match)
            .where(Match.status == MatchStatus.SCHEDULED)
            .where(Match.kickoff_time > now)
            .where(Match.kickoff_time < now + timedelta(days=days))
            .order_by(Match.kickoff_time)
        )
        matches = list(session.execute(stmt).scalars().all())

        fixtures = []
        for match in matches:
            # Get analysis
            analysis = session.execute(
                select(MatchAnalysis).where(MatchAnalysis.match_id == match.id)
            ).scalar_one_or_none()

            # Get best odds
            odds = session.execute(
                select(OddsHistory)
                .where(OddsHistory.match_id == match.id)
                .order_by(OddsHistory.recorded_at.desc())
                .limit(1)
            ).scalar_one_or_none()

            # Get value bets
            value_bets = list(session.execute(
                select(ValueBet)
                .where(ValueBet.match_id == match.id)
                .where(ValueBet.is_active == True)
                .order_by(ValueBet.edge.desc())
            ).scalars().all())

            fixtures.append({
                "id": match.id,
                "matchweek": match.matchweek,
                "kickoff": match.kickoff_time,
                "home_team_id": match.home_team_id,
                "away_team_id": match.away_team_id,
                "analysis": analysis,
                "odds": odds,
                "value_bets": value_bets,
            })

        return fixtures


def get_team_color(short_name: str) -> str:
    """Get color for a team."""
    return TEAM_COLORS.get(short_name, "#888888")


def render_probability_bar(home_prob: float, draw_prob: float, away_prob: float, home_name: str, away_name: str):
    """Render a horizontal probability bar."""
    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=["Probability"],
        x=[home_prob * 100],
        orientation='h',
        name=home_name,
        marker_color=get_team_color(home_name),
        text=f"{home_prob:.0%}",
        textposition='inside',
        insidetextanchor='middle',
        hovertemplate=f"{home_name} Win: {home_prob:.1%}<extra></extra>",
    ))

    fig.add_trace(go.Bar(
        y=["Probability"],
        x=[draw_prob * 100],
        orientation='h',
        name="Draw",
        marker_color="#888888",
        text=f"{draw_prob:.0%}",
        textposition='inside',
        insidetextanchor='middle',
        hovertemplate=f"Draw: {draw_prob:.1%}<extra></extra>",
    ))

    fig.add_trace(go.Bar(
        y=["Probability"],
        x=[away_prob * 100],
        orientation='h',
        name=away_name,
        marker_color=get_team_color(away_name),
        text=f"{away_prob:.0%}",
        textposition='inside',
        insidetextanchor='middle',
        hovertemplate=f"{away_name} Win: {away_prob:.1%}<extra></extra>",
    ))

    fig.update_layout(
        barmode='stack',
        height=60,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )

    return fig


# Main content
st.title("📅 Upcoming Fixtures & Predictions")
st.markdown("Match predictions and value betting opportunities for the upcoming week.")

# Load data
teams = load_teams()

# Sidebar options
st.sidebar.header("Settings")
days_ahead = st.sidebar.slider("Days ahead", 1, 14, 7)
show_all_odds = st.sidebar.checkbox("Show all bookmaker odds", False)

fixtures = load_upcoming_fixtures(days_ahead)

if not fixtures:
    st.info("No upcoming fixtures found.")
    st.stop()

# Summary metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Fixtures", len(fixtures))

total_value_bets = sum(len(f["value_bets"]) for f in fixtures)
col2.metric("Value Bets", total_value_bets)

with_predictions = sum(1 for f in fixtures if f["analysis"] and f["analysis"].consensus_home_prob)
col3.metric("With Predictions", f"{with_predictions}/{len(fixtures)}")

# Highest edge
max_edge = 0
max_edge_match = ""
for f in fixtures:
    if f["value_bets"]:
        edge = float(f["value_bets"][0].edge)
        if edge > max_edge:
            max_edge = edge
            home = teams.get(f["home_team_id"], {}).get("short_name", "?")
            away = teams.get(f["away_team_id"], {}).get("short_name", "?")
            max_edge_match = f"{home} vs {away}"

col4.metric("Best Edge", f"{max_edge:.1%}" if max_edge > 0 else "N/A", max_edge_match if max_edge > 0 else None)

st.divider()

# Group fixtures by date
fixtures_by_date = {}
for f in fixtures:
    date_str = f["kickoff"].strftime("%A, %B %d")
    if date_str not in fixtures_by_date:
        fixtures_by_date[date_str] = []
    fixtures_by_date[date_str].append(f)

# Render each day
for date_str, day_fixtures in fixtures_by_date.items():
    st.subheader(f"{date_str}")

    for fixture in day_fixtures:
        home = teams.get(fixture["home_team_id"], {})
        away = teams.get(fixture["away_team_id"], {})
        home_name = home.get("short_name", "Unknown")
        away_name = away.get("short_name", "Unknown")
        kickoff_time = fixture["kickoff"].strftime("%H:%M")

        analysis = fixture["analysis"]
        odds = fixture["odds"]
        value_bets = fixture["value_bets"]

        # Match card
        with st.container():
            # Header row
            header_col1, header_col2, header_col3 = st.columns([2, 1, 1])

            with header_col1:
                st.markdown(f"### {home_name} vs {away_name}")
                st.caption(f"Kickoff: {kickoff_time} | Matchweek {fixture['matchweek']}")

            with header_col2:
                if odds:
                    st.markdown("**Best Odds**")
                    st.text(f"H: {float(odds.home_odds):.2f}  D: {float(odds.draw_odds):.2f}  A: {float(odds.away_odds):.2f}")

            with header_col3:
                if value_bets:
                    best_vb = value_bets[0]
                    st.markdown(f"**Value Bet**")
                    outcome_display = best_vb.outcome.replace("_", " ").title()
                    st.text(f"{outcome_display} @ {float(best_vb.odds):.2f}")
                    st.caption(f"Edge: {float(best_vb.edge):.1%}")

            # Predictions
            if analysis and analysis.consensus_home_prob:
                home_prob = float(analysis.consensus_home_prob)
                draw_prob = float(analysis.consensus_draw_prob)
                away_prob = float(analysis.consensus_away_prob)

                # Probability bar
                prob_fig = render_probability_bar(home_prob, draw_prob, away_prob, home_name, away_name)
                st.plotly_chart(prob_fig, use_container_width=True, key=f"prob_{fixture['id']}")

                # Detailed predictions in expander
                with st.expander("View detailed predictions"):
                    detail_cols = st.columns(4)

                    # Consensus
                    with detail_cols[0]:
                        st.markdown("**Consensus**")
                        st.text(f"Home: {home_prob:.1%}")
                        st.text(f"Draw: {draw_prob:.1%}")
                        st.text(f"Away: {away_prob:.1%}")

                    # ELO
                    with detail_cols[1]:
                        if analysis.elo_home_prob:
                            st.markdown("**ELO Model**")
                            st.text(f"Home: {float(analysis.elo_home_prob):.1%}")
                            st.text(f"Draw: {float(analysis.elo_draw_prob):.1%}")
                            st.text(f"Away: {float(analysis.elo_away_prob):.1%}")

                    # Poisson
                    with detail_cols[2]:
                        if analysis.poisson_home_prob:
                            st.markdown("**Poisson Model**")
                            st.text(f"Home: {float(analysis.poisson_home_prob):.1%}")
                            st.text(f"Draw: {float(analysis.poisson_draw_prob):.1%}")
                            st.text(f"Away: {float(analysis.poisson_away_prob):.1%}")
                            if analysis.predicted_home_goals and analysis.predicted_away_goals:
                                st.caption(f"xG: {float(analysis.predicted_home_goals):.1f} - {float(analysis.predicted_away_goals):.1f}")

                    # XGBoost
                    with detail_cols[3]:
                        if analysis.xgboost_home_prob:
                            st.markdown("**XGBoost Model**")
                            st.text(f"Home: {float(analysis.xgboost_home_prob):.1%}")
                            st.text(f"Draw: {float(analysis.xgboost_draw_prob):.1%}")
                            st.text(f"Away: {float(analysis.xgboost_away_prob):.1%}")

                    # O/U and BTTS
                    if analysis.poisson_over_2_5_prob or analysis.poisson_btts_prob:
                        st.markdown("---")
                        ou_cols = st.columns(2)
                        with ou_cols[0]:
                            if analysis.poisson_over_2_5_prob:
                                st.markdown(f"**Over 2.5 Goals:** {float(analysis.poisson_over_2_5_prob):.1%}")
                        with ou_cols[1]:
                            if analysis.poisson_btts_prob:
                                st.markdown(f"**BTTS:** {float(analysis.poisson_btts_prob):.1%}")

                    # Value bets for this match
                    if value_bets:
                        st.markdown("---")
                        st.markdown("**All Value Bets**")
                        vb_data = []
                        for vb in value_bets[:5]:  # Show top 5
                            vb_data.append({
                                "Outcome": vb.outcome.replace("_", " ").title(),
                                "Bookmaker": vb.bookmaker,
                                "Odds": f"{float(vb.odds):.2f}",
                                "Model": f"{float(vb.model_probability):.1%}",
                                "Implied": f"{float(vb.implied_probability):.1%}",
                                "Edge": f"{float(vb.edge):.1%}",
                                "Kelly": f"{float(vb.kelly_stake):.2%}",
                            })
                        st.dataframe(vb_data, use_container_width=True, hide_index=True)

                    # AI Analysis narrative
                    if analysis.narrative:
                        st.markdown("---")
                        st.markdown("**AI Match Preview**")
                        st.markdown(analysis.narrative)
                        if analysis.narrative_generated_at:
                            st.caption(f"Generated: {analysis.narrative_generated_at.strftime('%Y-%m-%d %H:%M')}")
            else:
                st.caption("No predictions available for this match")

            st.divider()

# Value Bets Summary
st.subheader("All Value Bets This Week")

all_value_bets = []
for f in fixtures:
    home = teams.get(f["home_team_id"], {}).get("short_name", "?")
    away = teams.get(f["away_team_id"], {}).get("short_name", "?")
    match_str = f"{home} vs {away}"

    for vb in f["value_bets"]:
        all_value_bets.append({
            "match": match_str,
            "kickoff": f["kickoff"],
            "outcome": vb.outcome,
            "bookmaker": vb.bookmaker,
            "odds": float(vb.odds),
            "model_prob": float(vb.model_probability),
            "implied_prob": float(vb.implied_probability),
            "edge": float(vb.edge),
            "kelly": float(vb.kelly_stake),
        })

if all_value_bets:
    # Sort by edge
    all_value_bets.sort(key=lambda x: x["edge"], reverse=True)

    # Show only best per match/outcome
    seen = set()
    unique_bets = []
    for vb in all_value_bets:
        key = (vb["match"], vb["outcome"])
        if key not in seen:
            seen.add(key)
            unique_bets.append(vb)

    display_data = []
    for vb in unique_bets:
        display_data.append({
            "Match": vb["match"],
            "Kickoff": vb["kickoff"].strftime("%a %H:%M"),
            "Bet": vb["outcome"].replace("_", " ").title(),
            "Best Odds": f"{vb['odds']:.2f}",
            "Bookmaker": vb["bookmaker"],
            "Model": f"{vb['model_prob']:.1%}",
            "Implied": f"{vb['implied_prob']:.1%}",
            "Edge": f"{vb['edge']:.1%}",
            "Kelly Stake": f"{vb['kelly']:.2%}",
        })

    st.dataframe(display_data, use_container_width=True, hide_index=True)

    # Kelly summary
    total_kelly = sum(vb["kelly"] for vb in unique_bets)
    st.info(f"**Total Kelly allocation:** {total_kelly:.1%} of bankroll across {len(unique_bets)} bets")
else:
    st.info("No value bets found for upcoming fixtures. Run `python batch/jobs/odds_refresh.py` to detect value bets.")

# Sidebar info
with st.sidebar.expander("About Predictions"):
    st.markdown("""
    **Consensus Model**

    Combines three models:
    - **ELO** (35%): Team strength ratings
    - **Poisson** (40%): Goal distribution model
    - **XGBoost** (25%): ML classifier

    **Value Bets**

    A bet is flagged when:
    - Model probability > implied probability + 5%
    - Model confidence >= 60%

    **Kelly Criterion**

    Recommended stake based on edge and odds.
    Using 0.25 fractional Kelly for risk management.
    """)
