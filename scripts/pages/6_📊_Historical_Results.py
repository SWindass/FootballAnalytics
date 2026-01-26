"""Historical results browser - view past matchweeks and value bet performance."""

import streamlit as st
import pandas as pd
from datetime import datetime, timezone
from sqlalchemy import select, func
from sqlalchemy.orm import joinedload
from collections import defaultdict

from app.db.database import SyncSessionLocal
from app.db.models import Match, MatchAnalysis, MatchStatus, Team, ValueBet
from app.core.config import get_settings

settings = get_settings()

st.set_page_config(page_title="Historical Results", page_icon="📊", layout="wide")


@st.cache_data(ttl=300)
def load_teams():
    """Load teams from database."""
    with SyncSessionLocal() as session:
        teams = list(session.execute(select(Team)).scalars().all())
        return {t.id: {"name": t.name, "short_name": t.short_name} for t in teams}


@st.cache_data(ttl=300)
def get_available_seasons():
    """Get list of seasons with data."""
    with SyncSessionLocal() as session:
        stmt = (
            select(Match.season)
            .distinct()
            .order_by(Match.season.asc())
        )
        return [s for (s,) in session.execute(stmt).all()]


@st.cache_data(ttl=300)
def get_matchweeks_for_season(season: str):
    """Get list of matchweeks for a season."""
    with SyncSessionLocal() as session:
        stmt = (
            select(Match.matchweek)
            .where(Match.season == season)
            .distinct()
            .order_by(Match.matchweek)
        )
        return [mw for (mw,) in session.execute(stmt).all()]


@st.cache_data(ttl=60)
def load_matchweek_results(season: str, matchweek: int):
    """Load results for a specific matchweek with single query."""
    with SyncSessionLocal() as session:
        # Single query with joins
        stmt = (
            select(Match, MatchAnalysis)
            .outerjoin(MatchAnalysis, Match.id == MatchAnalysis.match_id)
            .where(Match.season == season)
            .where(Match.matchweek == matchweek)
            .order_by(Match.kickoff_time)
        )
        rows = list(session.execute(stmt).all())

        # Get match IDs for value bet lookup
        match_ids = [m.id for m, _ in rows]

        # Batch load value bets
        vb_stmt = select(ValueBet).where(ValueBet.match_id.in_(match_ids))
        all_value_bets = list(session.execute(vb_stmt).scalars().all())
        vb_by_match = defaultdict(list)
        for vb in all_value_bets:
            vb_by_match[vb.match_id].append({
                "outcome": vb.outcome,
                "odds": float(vb.odds),
                "edge": float(vb.edge),
                "bookmaker": vb.bookmaker,
            })

        results = []
        for match, analysis in rows:
            is_finished = match.status == MatchStatus.FINISHED

            analysis_data = None
            if analysis and analysis.consensus_home_prob:
                analysis_data = {
                    "home_prob": float(analysis.consensus_home_prob),
                    "draw_prob": float(analysis.consensus_draw_prob),
                    "away_prob": float(analysis.consensus_away_prob),
                }

            results.append({
                "id": match.id,
                "kickoff": match.kickoff_time,
                "home_team_id": match.home_team_id,
                "away_team_id": match.away_team_id,
                "home_score": match.home_score,
                "away_score": match.away_score,
                "is_finished": is_finished,
                "analysis": analysis_data,
                "value_bets": vb_by_match.get(match.id, []),
            })

        return results


@st.cache_data(ttl=300)
def load_season_summary(season: str):
    """Load summary stats for a season."""
    with SyncSessionLocal() as session:
        stmt = (
            select(Match.matchweek,
                   MatchAnalysis.consensus_home_prob,
                   MatchAnalysis.consensus_draw_prob,
                   MatchAnalysis.consensus_away_prob,
                   Match.home_score, Match.away_score)
            .join(MatchAnalysis, Match.id == MatchAnalysis.match_id)
            .where(Match.season == season)
            .where(Match.status == MatchStatus.FINISHED)
            .where(MatchAnalysis.consensus_home_prob.isnot(None))
            .order_by(Match.matchweek)
        )
        rows = list(session.execute(stmt).all())

        mw_stats = defaultdict(lambda: {"correct": 0, "total": 0})
        for mw, home_prob, draw_prob, away_prob, home_score, away_score in rows:
            h, d, a = float(home_prob), float(draw_prob), float(away_prob)
            predicted = "home" if h > d and h > a else ("away" if a > d else "draw")
            actual = "home" if home_score > away_score else ("away" if away_score > home_score else "draw")

            mw_stats[mw]["total"] += 1
            if predicted == actual:
                mw_stats[mw]["correct"] += 1

        return dict(mw_stats)


@st.cache_data(ttl=300)
def load_season_value_bets(season: str):
    """Load value bet performance for a season."""
    with SyncSessionLocal() as session:
        stmt = (
            select(ValueBet.outcome, ValueBet.odds, ValueBet.edge, ValueBet.bookmaker,
                   Match.matchweek, Match.home_team_id, Match.away_team_id,
                   Match.home_score, Match.away_score, Match.id)
            .join(Match, ValueBet.match_id == Match.id)
            .where(Match.season == season)
            .where(Match.status == MatchStatus.FINISHED)
            .order_by(Match.matchweek, Match.kickoff_time)
        )
        rows = list(session.execute(stmt).all())

        performance = []
        for outcome, odds, edge, bookmaker, mw, home_id, away_id, home_score, away_score, match_id in rows:
            actual = "home_win" if home_score > away_score else ("draw" if home_score == away_score else "away_win")
            won = outcome == actual
            stake = 10.0
            profit = stake * (float(odds) - 1) if won else -stake

            performance.append({
                "matchweek": mw,
                "match_id": match_id,
                "home_team_id": home_id,
                "away_team_id": away_id,
                "outcome": outcome,
                "odds": float(odds),
                "edge": float(edge),
                "won": won,
                "profit": profit,
                "bookmaker": bookmaker,
            })

        return performance


# Load data
teams = load_teams()
seasons = get_available_seasons()

if not seasons:
    st.warning("No seasons found in database")
    st.stop()

st.title("📊 Historical Results")

# Default to current season from settings
default_season = settings.current_season if settings.current_season in seasons else seasons[-1]

# Season and Matchweek navigation
col1, col2, col3, col4, col5, col6 = st.columns([1, 2, 1, 1, 2, 1])

with col1:
    prev_season = st.button("◀", key="prev_s", help="Previous season")
with col2:
    season_idx = seasons.index(default_season) if "season" not in st.session_state else seasons.index(st.session_state.season) if st.session_state.season in seasons else len(seasons) - 1

    # Handle prev/next
    if prev_season and season_idx > 0:
        season_idx -= 1
    if "next_s_clicked" in st.session_state and st.session_state.next_s_clicked and season_idx < len(seasons) - 1:
        season_idx += 1
        st.session_state.next_s_clicked = False

    selected_season = st.selectbox("Season", seasons, index=season_idx, key="season", label_visibility="collapsed")
with col3:
    if st.button("▶", key="next_s", help="Next season"):
        st.session_state.next_s_clicked = True
        st.rerun()

# Get matchweeks for selected season
matchweeks = get_matchweeks_for_season(selected_season)
if not matchweeks:
    st.warning("No matchweeks found")
    st.stop()

# Default to last matchweek
default_mw_idx = len(matchweeks) - 1

with col4:
    prev_mw = st.button("◀", key="prev_m", help="Previous matchweek")
with col5:
    mw_idx = default_mw_idx if "mw" not in st.session_state else (matchweeks.index(st.session_state.mw) if st.session_state.mw in matchweeks else default_mw_idx)

    if prev_mw and mw_idx > 0:
        mw_idx -= 1
    if "next_m_clicked" in st.session_state and st.session_state.next_m_clicked and mw_idx < len(matchweeks) - 1:
        mw_idx += 1
        st.session_state.next_m_clicked = False

    selected_mw = st.selectbox("Matchweek", matchweeks, index=mw_idx, format_func=lambda x: f"MW {x}", key="mw", label_visibility="collapsed")
with col6:
    if st.button("▶", key="next_m", help="Next matchweek"):
        st.session_state.next_m_clicked = True
        st.rerun()

st.markdown("---")

# Content tabs
tab1, tab2, tab3 = st.tabs(["📋 Results", "📈 Season Overview", "💰 Value Bets"])

with tab1:
    st.subheader(f"Matchweek {selected_mw} - {selected_season}")

    with st.spinner("Loading results..."):
        results = load_matchweek_results(selected_season, selected_mw)

    if not results:
        st.info("No matches found")
    else:
        finished = [r for r in results if r["is_finished"]]
        pending = [r for r in results if not r["is_finished"]]

        # Prediction accuracy
        correct = sum(1 for r in finished if r["analysis"] and (
            ("home" if r["analysis"]["home_prob"] > r["analysis"]["draw_prob"] and r["analysis"]["home_prob"] > r["analysis"]["away_prob"] else ("away" if r["analysis"]["away_prob"] > r["analysis"]["draw_prob"] else "draw"))
            == ("home" if r["home_score"] > r["away_score"] else ("away" if r["away_score"] > r["home_score"] else "draw"))
        ))

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Matches", len(results))
        col2.metric("Completed", len(finished))
        col3.metric("Pending", len(pending))
        if finished:
            col4.metric("Accuracy", f"{correct}/{len(finished)}")

        st.markdown("---")

        for result in results:
            home_name = teams.get(result["home_team_id"], {}).get("short_name", "?")
            away_name = teams.get(result["away_team_id"], {}).get("short_name", "?")

            col1, col2, col3, col4 = st.columns([3, 2, 2, 1])

            with col1:
                if result["is_finished"]:
                    st.markdown(f"**{home_name}** {result['home_score']} - {result['away_score']} **{away_name}**")
                else:
                    st.markdown(f"**{home_name}** vs **{away_name}** ({result['kickoff'].strftime('%a %H:%M')})")

            with col2:
                if result["analysis"]:
                    a = result["analysis"]
                    st.caption(f"H {a['home_prob']:.0%} D {a['draw_prob']:.0%} A {a['away_prob']:.0%}")

            with col3:
                for vb in result["value_bets"][:2]:
                    st.caption(f"{vb['outcome'][0].upper()} +{vb['edge']:.0%} @ {vb['odds']:.2f}")

            with col4:
                if result["is_finished"] and result["analysis"]:
                    a = result["analysis"]
                    predicted = "home" if a["home_prob"] > a["draw_prob"] and a["home_prob"] > a["away_prob"] else ("away" if a["away_prob"] > a["draw_prob"] else "draw")
                    actual = "home" if result["home_score"] > result["away_score"] else ("away" if result["away_score"] > result["home_score"] else "draw")
                    st.markdown("✅" if predicted == actual else "❌")

        # Value bet results
        vb_results = []
        for r in finished:
            if r["value_bets"]:
                actual = "home_win" if r["home_score"] > r["away_score"] else ("draw" if r["home_score"] == r["away_score"] else "away_win")
                home = teams.get(r["home_team_id"], {}).get("short_name", "?")
                away = teams.get(r["away_team_id"], {}).get("short_name", "?")
                for vb in r["value_bets"]:
                    won = vb["outcome"] == actual
                    profit = 10 * (vb["odds"] - 1) if won else -10
                    vb_results.append({
                        "Match": f"{home} v {away}",
                        "Bet": vb["outcome"].replace("_", " ").title(),
                        "Odds": f"{vb['odds']:.2f}",
                        "Edge": f"+{vb['edge']:.0%}",
                        "Result": "✅" if won else "❌",
                        "P/L": f"£{profit:+.0f}",
                    })

        if vb_results:
            st.markdown("---")
            st.subheader("💰 Value Bets")
            st.dataframe(vb_results, use_container_width=True, hide_index=True)
            total = sum(float(v["P/L"].replace("£", "").replace("+", "")) for v in vb_results)
            wins = sum(1 for v in vb_results if v["Result"] == "✅")
            st.markdown(f"**{wins}/{len(vb_results)} won, £{total:+.0f}**")

with tab2:
    st.subheader(f"Season Overview - {selected_season}")

    with st.spinner("Loading season data..."):
        mw_stats = load_season_summary(selected_season)

    if not mw_stats:
        st.info("No data found")
    else:
        total_correct = sum(s["correct"] for s in mw_stats.values())
        total_matches = sum(s["total"] for s in mw_stats.values())

        col1, col2, col3 = st.columns(3)
        col1.metric("Matches", total_matches)
        col2.metric("Correct", total_correct)
        col3.metric("Accuracy", f"{100*total_correct/total_matches:.1f}%" if total_matches else "N/A")

        chart_data = pd.DataFrame([
            {"MW": mw, "Accuracy": s["correct"]/s["total"]*100 if s["total"] else 0}
            for mw, s in sorted(mw_stats.items())
        ])
        st.bar_chart(chart_data.set_index("MW"))

with tab3:
    st.subheader(f"Value Bet Performance - {selected_season}")

    with st.spinner("Loading value bets..."):
        performance = load_season_value_bets(selected_season)

    if not performance:
        st.info("No value bets found")
    else:
        total_bets = len(performance)
        wins = sum(1 for p in performance if p["won"])
        total_profit = sum(p["profit"] for p in performance)
        roi = total_profit / (total_bets * 10) * 100 if total_bets else 0

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Bets", total_bets)
        col2.metric("Win Rate", f"{100*wins/total_bets:.0f}%")
        col3.metric("ROI", f"{roi:+.1f}%")
        col4.metric("Profit", f"£{total_profit:+.0f}")

        # Cumulative profit
        mw_profit = defaultdict(float)
        for p in performance:
            mw_profit[p["matchweek"]] += p["profit"]

        cumulative = []
        running = 0
        for mw in sorted(mw_profit.keys()):
            running += mw_profit[mw]
            cumulative.append({"MW": mw, "Profit": running})

        if cumulative:
            st.line_chart(pd.DataFrame(cumulative).set_index("MW"))

        # By outcome
        by_outcome = defaultdict(lambda: {"bets": 0, "wins": 0, "profit": 0})
        for p in performance:
            by_outcome[p["outcome"]]["bets"] += 1
            by_outcome[p["outcome"]]["wins"] += int(p["won"])
            by_outcome[p["outcome"]]["profit"] += p["profit"]

        outcome_data = []
        for o in ["home_win", "draw", "away_win"]:
            if o in by_outcome:
                d = by_outcome[o]
                outcome_data.append({
                    "Outcome": o.replace("_", " ").title(),
                    "Bets": d["bets"],
                    "Wins": d["wins"],
                    "ROI": f"{d['profit']/(d['bets']*10)*100:+.0f}%" if d["bets"] else "N/A",
                })

        if outcome_data:
            st.dataframe(outcome_data, use_container_width=True, hide_index=True)
