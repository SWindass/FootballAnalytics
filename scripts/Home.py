"""Football Analytics Dashboard - Home Page."""

import sys
import os

# Ensure the project root is in the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import traceback

# Debug: Show any import errors
try:
    from datetime import datetime, timezone, timedelta
    from sqlalchemy import select, func
    from app.db.database import SyncSessionLocal
    from app.db.models import Match, MatchStatus, ValueBet, Team
except Exception as e:
    st.error(f"Import error: {e}")
    st.code(traceback.format_exc())
    st.stop()

st.set_page_config(
    page_title="Football Analytics",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("⚽ Football Analytics Dashboard")
st.markdown("EPL Value Bet Finder - Predictions powered by ELO, Poisson & XGBoost models")

# Quick stats
with SyncSessionLocal() as session:
    now = datetime.now(timezone.utc)

    # Upcoming matches
    upcoming = session.execute(
        select(func.count(Match.id))
        .where(Match.status == MatchStatus.SCHEDULED)
        .where(Match.kickoff_time > now)
        .where(Match.kickoff_time < now + timedelta(days=7))
    ).scalar()

    # Active value bets
    value_bets = session.execute(
        select(func.count(ValueBet.id))
        .where(ValueBet.is_active == True)
    ).scalar()

    # Best edge
    best_vb = session.execute(
        select(ValueBet)
        .where(ValueBet.is_active == True)
        .order_by(ValueBet.edge.desc())
        .limit(1)
    ).scalar_one_or_none()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Upcoming Matches", upcoming)
col2.metric("Active Value Bets", value_bets)

if best_vb:
    col3.metric("Best Edge", f"{float(best_vb.edge):.1%}")
    col4.metric("Kelly Stake", f"{float(best_vb.kelly_stake):.2%}")

st.divider()

# Navigation cards
st.subheader("Quick Navigation")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### 📅 Upcoming Fixtures
    View this week's matches with predictions and value bets.

    - Match probabilities
    - Best odds comparison
    - Value bet opportunities

    👉 **Select "Fixtures" in the sidebar**
    """)

with col2:
    st.markdown("""
    ### 📊 Poisson Model
    Team attack/defense strengths and goal predictions.

    - Attack vs Defense scatter
    - Strength progression over time
    - Model accuracy analysis

    👉 **Select "Poisson Model" in the sidebar**
    """)

with col3:
    st.markdown("""
    ### 📈 ELO Ratings
    Team strength rankings based on match results.

    - Current ELO standings
    - Rating changes over time
    - Head-to-head comparisons

    👉 **Select "ELO Ratings" in the sidebar**
    """)

st.divider()

# Top value bets preview
st.subheader("Top Value Bets Right Now")

with SyncSessionLocal() as session:
    top_bets = list(session.execute(
        select(ValueBet)
        .where(ValueBet.is_active == True)
        .order_by(ValueBet.edge.desc())
        .limit(5)
    ).scalars().all())

    if top_bets:
        teams = {t.id: t.short_name for t in session.execute(select(Team)).scalars().all()}

        for vb in top_bets:
            match = session.get(Match, vb.match_id)
            home = teams.get(match.home_team_id, "?")
            away = teams.get(match.away_team_id, "?")
            outcome = vb.outcome.replace("_", " ").title()

            col1, col2, col3, col4 = st.columns([3, 2, 1, 1])
            col1.write(f"**{home} vs {away}**")
            col2.write(f"{outcome} @ {float(vb.odds):.2f}")
            col3.write(f"Edge: {float(vb.edge):.1%}")
            col4.write(f"Kelly: {float(vb.kelly_stake):.2%}")
    else:
        st.info("No active value bets. Run `python batch/jobs/odds_refresh.py` to find value bets.")

st.divider()

# Commands reference
with st.expander("Useful Commands"):
    st.code("""
# Refresh odds and find value bets
PYTHONPATH=. python batch/jobs/odds_refresh.py

# Run weekly analysis (predictions for upcoming matches)
PYTHONPATH=. python batch/jobs/weekly_analysis.py

# Recalculate ELO ratings
PYTHONPATH=. python batch/jobs/calculate_elo.py

# Recalculate Poisson predictions
PYTHONPATH=. python batch/jobs/calculate_poisson.py --backfill
    """, language="bash")
