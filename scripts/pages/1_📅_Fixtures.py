"""Upcoming fixtures dashboard with predictions and value bets."""

import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta, timezone
from sqlalchemy import select

from app.db.database import SyncSessionLocal
from app.db.models import Match, MatchAnalysis, MatchStatus, Team, OddsHistory, ValueBet, EloRating

# Standard color scheme
COLORS = {
    "home": "#2E86AB",    # Blue
    "draw": "#A0A0A0",    # Gray
    "away": "#E94F37",    # Red
    "accent": "#F9A03F",  # Orange (for highlights)
}


def decimal_to_fraction(decimal_odds: float) -> str:
    """Convert decimal odds to fractional odds."""
    from fractions import Fraction

    if decimal_odds <= 1:
        return "N/A"

    # Convert to fractional: (decimal - 1) as fraction
    frac = Fraction(decimal_odds - 1).limit_denominator(100)

    if frac.denominator == 1:
        return f"{frac.numerator}/1"

    return f"{frac.numerator}/{frac.denominator}"

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
            analysis = session.execute(
                select(MatchAnalysis).where(MatchAnalysis.match_id == match.id)
            ).scalar_one_or_none()

            odds = session.execute(
                select(OddsHistory)
                .where(OddsHistory.match_id == match.id)
                .order_by(OddsHistory.recorded_at.desc())
                .limit(1)
            ).scalar_one_or_none()

            value_bets = list(session.execute(
                select(ValueBet)
                .where(ValueBet.match_id == match.id)
                .where(ValueBet.is_active == True)
                .order_by(ValueBet.edge.desc())
            ).scalars().all())

            fixtures.append({
                "id": match.id,
                "season": match.season,
                "matchweek": match.matchweek,
                "kickoff": match.kickoff_time,
                "home_team_id": match.home_team_id,
                "away_team_id": match.away_team_id,
                "analysis": analysis,
                "odds": odds,
                "value_bets": value_bets,
            })

        return fixtures


@st.cache_data(ttl=300)
def load_elo_history(team_id: int, season: str):
    """Load ELO rating history for a team."""
    with SyncSessionLocal() as session:
        stmt = (
            select(EloRating)
            .where(EloRating.team_id == team_id)
            .where(EloRating.season == season)
            .order_by(EloRating.matchweek)
        )
        ratings = list(session.execute(stmt).scalars().all())
        return [{"matchweek": r.matchweek, "rating": float(r.rating)} for r in ratings]


def render_elo_chart(home_elo: list, away_elo: list, home_name: str, away_name: str):
    """Render ELO history chart for two teams."""
    fig = go.Figure()

    if home_elo:
        fig.add_trace(go.Scatter(
            x=[e["matchweek"] for e in home_elo],
            y=[e["rating"] for e in home_elo],
            mode='lines+markers',
            name=home_name,
            line=dict(color=COLORS["home"], width=2),
            marker=dict(size=5),
        ))

    if away_elo:
        fig.add_trace(go.Scatter(
            x=[e["matchweek"] for e in away_elo],
            y=[e["rating"] for e in away_elo],
            mode='lines+markers',
            name=away_name,
            line=dict(color=COLORS["away"], width=2),
            marker=dict(size=5),
        ))

    fig.update_layout(
        height=200,
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        xaxis=dict(title="Matchweek", dtick=5),
        yaxis=dict(title="ELO"),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )

    return fig


def render_probability_bar(home_prob: float, draw_prob: float, away_prob: float):
    """Render a compact horizontal probability bar."""
    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=[""],
        x=[home_prob * 100],
        orientation='h',
        name="Home",
        marker_color=COLORS["home"],
        text=f"H {home_prob:.0%}",
        textposition='inside',
        textfont=dict(color='white', size=11),
        hovertemplate=f"Home: {home_prob:.1%}<extra></extra>",
    ))

    fig.add_trace(go.Bar(
        y=[""],
        x=[draw_prob * 100],
        orientation='h',
        name="Draw",
        marker_color=COLORS["draw"],
        text=f"D {draw_prob:.0%}",
        textposition='inside',
        textfont=dict(color='white', size=11),
        hovertemplate=f"Draw: {draw_prob:.1%}<extra></extra>",
    ))

    fig.add_trace(go.Bar(
        y=[""],
        x=[away_prob * 100],
        orientation='h',
        name="Away",
        marker_color=COLORS["away"],
        text=f"A {away_prob:.0%}",
        textposition='inside',
        textfont=dict(color='white', size=11),
        hovertemplate=f"Away: {away_prob:.1%}<extra></extra>",
    ))

    fig.update_layout(
        barmode='stack',
        height=35,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False, range=[0, 100]),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )

    return fig


# Main content
st.title("📅 Fixtures & Predictions")

# Load data
teams = load_teams()

# Sidebar
with st.sidebar:
    days_ahead = st.slider("Days ahead", 1, 14, 7)

    with st.expander("About"):
        st.markdown("""
        **Models:** ELO (35%) + Poisson (40%) + XGBoost (25%)

        **Value Bet:** Model prob > implied + 5%

        **Kelly:** 0.25 fractional for risk management
        """)

fixtures = load_upcoming_fixtures(days_ahead)

if not fixtures:
    st.info("No upcoming fixtures found.")
    st.stop()

# Compact summary
total_vb = sum(len(f["value_bets"]) for f in fixtures)
with_pred = sum(1 for f in fixtures if f["analysis"] and f["analysis"].consensus_home_prob)
cols = st.columns(4)
cols[0].metric("Fixtures", len(fixtures))
cols[1].metric("Value Bets", total_vb)
cols[2].metric("Predictions", f"{with_pred}/{len(fixtures)}")

max_edge = 0
for f in fixtures:
    if f["value_bets"]:
        edge = float(f["value_bets"][0].edge)
        if edge > max_edge:
            max_edge = edge
cols[3].metric("Best Edge", f"{max_edge:.1%}" if max_edge > 0 else "-")

st.divider()

# Group by date
fixtures_by_date = {}
for f in fixtures:
    date_str = f["kickoff"].strftime("%a %d %b")
    if date_str not in fixtures_by_date:
        fixtures_by_date[date_str] = []
    fixtures_by_date[date_str].append(f)

# Render fixtures
for date_str, day_fixtures in fixtures_by_date.items():
    st.markdown(f"**{date_str}**")

    for fixture in day_fixtures:
        home = teams.get(fixture["home_team_id"], {})
        away = teams.get(fixture["away_team_id"], {})
        home_name = home.get("short_name", "?")
        away_name = away.get("short_name", "?")
        kickoff_time = fixture["kickoff"].strftime("%H:%M")

        analysis = fixture["analysis"]
        odds = fixture["odds"]
        value_bets = fixture["value_bets"]

        # Compact match row
        c1, c2, c3, c4 = st.columns([3, 2, 2, 1])

        with c1:
            st.markdown(f"**{home_name}** vs **{away_name}**")
            st.caption(f"{kickoff_time} | MW{fixture['matchweek']}")

        with c2:
            if analysis and analysis.consensus_home_prob:
                prob_fig = render_probability_bar(
                    float(analysis.consensus_home_prob),
                    float(analysis.consensus_draw_prob),
                    float(analysis.consensus_away_prob)
                )
                st.plotly_chart(prob_fig, use_container_width=True, key=f"prob_{fixture['id']}")

        with c3:
            if odds:
                h, d, a = float(odds.home_odds), float(odds.draw_odds), float(odds.away_odds)
                st.caption(f"Odds: {h:.2f} / {d:.2f} / {a:.2f}")
                st.caption(f"({decimal_to_fraction(h)} / {decimal_to_fraction(d)} / {decimal_to_fraction(a)})")
            if analysis and analysis.predicted_home_goals:
                st.caption(f"xG: {float(analysis.predicted_home_goals):.1f} - {float(analysis.predicted_away_goals):.1f}")

        with c4:
            if value_bets:
                best = value_bets[0]
                st.markdown(f"**{float(best.edge):.0%}** edge")

        # Expandable details
        if analysis and analysis.consensus_home_prob:
            with st.expander("Details", expanded=False):
                # Model comparison in compact table
                model_data = []
                if analysis.elo_home_prob:
                    model_data.append({
                        "Model": "ELO",
                        "Home": f"{float(analysis.elo_home_prob):.0%}",
                        "Draw": f"{float(analysis.elo_draw_prob):.0%}",
                        "Away": f"{float(analysis.elo_away_prob):.0%}",
                    })
                if analysis.poisson_home_prob:
                    model_data.append({
                        "Model": "Poisson",
                        "Home": f"{float(analysis.poisson_home_prob):.0%}",
                        "Draw": f"{float(analysis.poisson_draw_prob):.0%}",
                        "Away": f"{float(analysis.poisson_away_prob):.0%}",
                    })
                if analysis.xgboost_home_prob:
                    model_data.append({
                        "Model": "XGBoost",
                        "Home": f"{float(analysis.xgboost_home_prob):.0%}",
                        "Draw": f"{float(analysis.xgboost_draw_prob):.0%}",
                        "Away": f"{float(analysis.xgboost_away_prob):.0%}",
                    })
                model_data.append({
                    "Model": "**Consensus**",
                    "Home": f"**{float(analysis.consensus_home_prob):.0%}**",
                    "Draw": f"**{float(analysis.consensus_draw_prob):.0%}**",
                    "Away": f"**{float(analysis.consensus_away_prob):.0%}**",
                })

                col1, col2 = st.columns([1, 1])
                with col1:
                    st.dataframe(model_data, use_container_width=True, hide_index=True)

                with col2:
                    extras = []
                    if analysis.poisson_over_2_5_prob:
                        extras.append(f"Over 2.5: {float(analysis.poisson_over_2_5_prob):.0%}")
                    if analysis.poisson_btts_prob:
                        extras.append(f"BTTS: {float(analysis.poisson_btts_prob):.0%}")
                    if extras:
                        st.markdown(" | ".join(extras))

                    if value_bets:
                        st.markdown("**Value Bets:**")
                        for vb in value_bets[:3]:
                            outcome = vb.outcome.replace("_", " ").title()
                            odds_dec = float(vb.odds)
                            odds_frac = decimal_to_fraction(odds_dec)
                            st.markdown(f"- {outcome} @ {odds_dec:.2f} ({odds_frac}) via {vb.bookmaker} — {float(vb.edge):.1%} edge")

                # ELO History Chart
                home_elo = load_elo_history(fixture["home_team_id"], fixture["season"])
                away_elo = load_elo_history(fixture["away_team_id"], fixture["season"])
                if home_elo or away_elo:
                    st.markdown("---")
                    st.markdown("**ELO Rating History**")
                    elo_fig = render_elo_chart(home_elo, away_elo, home_name, away_name)
                    st.plotly_chart(elo_fig, use_container_width=True, key=f"elo_{fixture['id']}")

                # AI Narrative
                if analysis.narrative:
                    st.markdown("---")
                    st.markdown(analysis.narrative)

    st.divider()

# Value bets summary
if total_vb > 0:
    st.markdown("### Value Bets Summary")

    all_vb = []
    for f in fixtures:
        home = teams.get(f["home_team_id"], {}).get("short_name", "?")
        away = teams.get(f["away_team_id"], {}).get("short_name", "?")
        for vb in f["value_bets"]:
            odds_dec = float(vb.odds)
            all_vb.append({
                "Match": f"{home} v {away}",
                "Kick": f["kickoff"].strftime("%a %H:%M"),
                "Bet": vb.outcome.replace("_", " ").title(),
                "Odds": f"{odds_dec:.2f} ({decimal_to_fraction(odds_dec)})",
                "Book": vb.bookmaker,
                "Edge": f"{float(vb.edge):.1%}",
                "Kelly": f"{float(vb.kelly_stake):.1%}",
            })

    all_vb.sort(key=lambda x: float(x["Edge"].rstrip("%")), reverse=True)

    # Dedupe by match+bet
    seen = set()
    unique = []
    for vb in all_vb:
        key = (vb["Match"], vb["Bet"])
        if key not in seen:
            seen.add(key)
            unique.append(vb)

    st.dataframe(unique, use_container_width=True, hide_index=True)

    total_kelly = sum(float(vb["Kelly"].rstrip("%")) for vb in unique)
    st.caption(f"Total Kelly: {total_kelly:.1f}% across {len(unique)} bets")
