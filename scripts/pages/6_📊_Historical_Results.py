"""Historical results browser - view past matchweeks and value bet performance."""

import streamlit as st
import pandas as pd
from datetime import datetime, timezone
from sqlalchemy import select, func
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
    """Get list of seasons with data, oldest first (most recent on right)."""
    with SyncSessionLocal() as session:
        stmt = (
            select(Match.season)
            .distinct()
            .order_by(Match.season.asc())  # Oldest first, most recent last (on right)
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
    """Load results for a specific matchweek."""
    with SyncSessionLocal() as session:
        stmt = (
            select(Match)
            .where(Match.season == season)
            .where(Match.matchweek == matchweek)
            .order_by(Match.kickoff_time)
        )
        matches = list(session.execute(stmt).scalars().all())

        results = []
        for match in matches:
            analysis = session.execute(
                select(MatchAnalysis).where(MatchAnalysis.match_id == match.id)
            ).scalar_one_or_none()

            # Get value bets for this match
            value_bets = list(session.execute(
                select(ValueBet).where(ValueBet.match_id == match.id)
            ).scalars().all())

            is_finished = match.status == MatchStatus.FINISHED

            results.append({
                "id": match.id,
                "kickoff": match.kickoff_time,
                "home_team_id": match.home_team_id,
                "away_team_id": match.away_team_id,
                "home_score": match.home_score,
                "away_score": match.away_score,
                "is_finished": is_finished,
                "analysis": analysis,
                "value_bets": value_bets,
            })

        return results


@st.cache_data(ttl=300)
def load_season_summary(season: str):
    """Load summary stats for a season."""
    with SyncSessionLocal() as session:
        # Get all finished matches with analysis
        stmt = (
            select(Match, MatchAnalysis)
            .join(MatchAnalysis, Match.id == MatchAnalysis.match_id)
            .where(Match.season == season)
            .where(Match.status == MatchStatus.FINISHED)
            .order_by(Match.matchweek)
        )
        results = list(session.execute(stmt).all())

        # Calculate accuracy by matchweek
        mw_stats = defaultdict(lambda: {"correct": 0, "total": 0, "matches": []})

        for match, analysis in results:
            if analysis.consensus_home_prob:
                home_prob = float(analysis.consensus_home_prob)
                draw_prob = float(analysis.consensus_draw_prob)
                away_prob = float(analysis.consensus_away_prob)

                predicted = "home" if home_prob > draw_prob and home_prob > away_prob else (
                    "away" if away_prob > draw_prob else "draw"
                )
                actual = "home" if match.home_score > match.away_score else (
                    "away" if match.away_score > match.home_score else "draw"
                )

                mw_stats[match.matchweek]["total"] += 1
                if predicted == actual:
                    mw_stats[match.matchweek]["correct"] += 1

        return dict(mw_stats)


@st.cache_data(ttl=300)
def load_season_value_bets(season: str):
    """Load value bet performance for a season."""
    with SyncSessionLocal() as session:
        stmt = (
            select(ValueBet, Match)
            .join(Match, ValueBet.match_id == Match.id)
            .where(Match.season == season)
            .where(Match.status == MatchStatus.FINISHED)
            .order_by(Match.matchweek, Match.kickoff_time)
        )
        results = list(session.execute(stmt).all())

        performance = []
        for vb, match in results:
            if match.home_score > match.away_score:
                actual = "home_win"
            elif match.home_score == match.away_score:
                actual = "draw"
            else:
                actual = "away_win"

            won = vb.outcome == actual
            stake = 10.0
            profit = stake * (float(vb.odds) - 1) if won else -stake

            performance.append({
                "matchweek": match.matchweek,
                "match_id": match.id,
                "home_team_id": match.home_team_id,
                "away_team_id": match.away_team_id,
                "outcome": vb.outcome,
                "odds": float(vb.odds),
                "edge": float(vb.edge),
                "won": won,
                "profit": profit,
                "bookmaker": vb.bookmaker,
            })

        return performance


# Initialize session state for navigation
if "selected_season_idx" not in st.session_state:
    st.session_state.selected_season_idx = None  # Will be set to last (most recent) season
if "selected_mw_idx" not in st.session_state:
    st.session_state.selected_mw_idx = None  # Will be set to last matchweek


teams = load_teams()
seasons = get_available_seasons()

# Initialize season index to most recent (last in list)
if st.session_state.selected_season_idx is None:
    st.session_state.selected_season_idx = len(seasons) - 1 if seasons else 0

# Clamp season index to valid range
st.session_state.selected_season_idx = max(0, min(st.session_state.selected_season_idx, len(seasons) - 1))

if not seasons:
    st.warning("No seasons found in database")
    st.stop()

st.title("📊 Historical Results")

# Season navigation with arrows
st.markdown("### Season")
col1, col2, col3 = st.columns([1, 3, 1])

with col1:
    # Prev goes to earlier season (lower index)
    if st.button("◀ Prev", key="prev_season", disabled=st.session_state.selected_season_idx <= 0):
        st.session_state.selected_season_idx -= 1
        st.session_state.selected_mw_idx = None  # Reset matchweek
        st.rerun()

with col2:
    # Season tabs
    season_tabs = st.tabs(seasons)

with col3:
    # Next goes to later season (higher index)
    if st.button("Next ▶", key="next_season", disabled=st.session_state.selected_season_idx >= len(seasons) - 1):
        st.session_state.selected_season_idx += 1
        st.session_state.selected_mw_idx = None  # Reset matchweek
        st.rerun()

# Get current season from tab selection
selected_season = seasons[st.session_state.selected_season_idx]

# Display content in the selected season tab
with season_tabs[st.session_state.selected_season_idx]:
    matchweeks = get_matchweeks_for_season(selected_season)

    if not matchweeks:
        st.warning("No matchweeks found for this season")
        st.stop()

    # Initialize matchweek index if not set
    if st.session_state.selected_mw_idx is None:
        st.session_state.selected_mw_idx = len(matchweeks) - 1

    # Clamp matchweek index to valid range
    st.session_state.selected_mw_idx = max(0, min(st.session_state.selected_mw_idx, len(matchweeks) - 1))

    selected_mw = matchweeks[st.session_state.selected_mw_idx]

    # Matchweek navigation with arrows
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 3, 1])

    with col1:
        if st.button("◀ Prev MW", key="prev_mw", disabled=st.session_state.selected_mw_idx <= 0):
            st.session_state.selected_mw_idx -= 1
            st.rerun()

    with col2:
        # Matchweek selector
        mw_options = [f"MW {mw}" for mw in matchweeks]
        selected_mw_label = st.selectbox(
            "Matchweek",
            mw_options,
            index=st.session_state.selected_mw_idx,
            label_visibility="collapsed",
        )
        # Update index if changed via selectbox
        new_idx = mw_options.index(selected_mw_label)
        if new_idx != st.session_state.selected_mw_idx:
            st.session_state.selected_mw_idx = new_idx
            st.rerun()

    with col3:
        if st.button("Next MW ▶", key="next_mw", disabled=st.session_state.selected_mw_idx >= len(matchweeks) - 1):
            st.session_state.selected_mw_idx += 1
            st.rerun()

    # Content tabs: Results | Season Overview | Value Bets
    st.markdown("---")
    content_tab1, content_tab2, content_tab3 = st.tabs(["📋 Results", "📈 Season Overview", "💰 Value Bets"])

    # TAB 1: Matchweek Results
    with content_tab1:
        st.subheader(f"Matchweek {selected_mw} Results")

        results = load_matchweek_results(selected_season, selected_mw)

        if not results:
            st.info("No matches found for this matchweek")
        else:
            # Summary metrics
            finished = [r for r in results if r["is_finished"]]
            pending = [r for r in results if not r["is_finished"]]

            # Prediction accuracy
            correct = 0
            for r in finished:
                if r["analysis"] and r["analysis"].consensus_home_prob:
                    home_prob = float(r["analysis"].consensus_home_prob)
                    draw_prob = float(r["analysis"].consensus_draw_prob)
                    away_prob = float(r["analysis"].consensus_away_prob)

                    predicted = "home" if home_prob > draw_prob and home_prob > away_prob else (
                        "away" if away_prob > draw_prob else "draw"
                    )
                    actual = "home" if r["home_score"] > r["away_score"] else (
                        "away" if r["away_score"] > r["home_score"] else "draw"
                    )
                    if predicted == actual:
                        correct += 1

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Matches", len(results))
            col2.metric("Completed", len(finished))
            col3.metric("Pending", len(pending))
            if finished:
                col4.metric("Accuracy", f"{correct}/{len(finished)} ({100*correct/len(finished):.0f}%)")

            st.markdown("---")

            # Results table
            for result in results:
                home = teams.get(result["home_team_id"], {})
                away = teams.get(result["away_team_id"], {})
                home_name = home.get("short_name", "?")
                away_name = away.get("short_name", "?")

                analysis = result["analysis"]
                value_bets = result["value_bets"]

                col1, col2, col3, col4 = st.columns([3, 2, 2, 1])

                with col1:
                    if result["is_finished"]:
                        st.markdown(f"**{home_name}** {result['home_score']} - {result['away_score']} **{away_name}**")
                    else:
                        kickoff = result["kickoff"].strftime("%a %H:%M")
                        st.markdown(f"**{home_name}** vs **{away_name}** ({kickoff})")

                with col2:
                    if analysis and analysis.consensus_home_prob:
                        h = float(analysis.consensus_home_prob)
                        d = float(analysis.consensus_draw_prob)
                        a = float(analysis.consensus_away_prob)
                        st.caption(f"Pred: H {h:.0%} D {d:.0%} A {a:.0%}")

                with col3:
                    if value_bets:
                        for vb in value_bets[:2]:
                            outcome_short = {"home_win": "H", "draw": "D", "away_win": "A"}.get(vb.outcome, "?")
                            edge_pct = float(vb.edge) * 100
                            st.caption(f"Value: {outcome_short} +{edge_pct:.0f}% @ {float(vb.odds):.2f}")

                with col4:
                    if result["is_finished"] and analysis and analysis.consensus_home_prob:
                        home_prob = float(analysis.consensus_home_prob)
                        draw_prob = float(analysis.consensus_draw_prob)
                        away_prob = float(analysis.consensus_away_prob)

                        predicted = "home" if home_prob > draw_prob and home_prob > away_prob else (
                            "away" if away_prob > draw_prob else "draw"
                        )
                        actual = "home" if result["home_score"] > result["away_score"] else (
                            "away" if result["away_score"] > result["home_score"] else "draw"
                        )

                        if predicted == actual:
                            st.markdown("✅")
                        else:
                            st.markdown("❌")

            # Value bet results for this matchweek
            value_bet_results = []
            for result in finished:
                if result["value_bets"]:
                    home = teams.get(result["home_team_id"], {}).get("short_name", "?")
                    away = teams.get(result["away_team_id"], {}).get("short_name", "?")

                    if result["home_score"] > result["away_score"]:
                        actual = "home_win"
                    elif result["home_score"] == result["away_score"]:
                        actual = "draw"
                    else:
                        actual = "away_win"

                    for vb in result["value_bets"]:
                        won = vb.outcome == actual
                        stake = 10.0
                        profit = stake * (float(vb.odds) - 1) if won else -stake

                        value_bet_results.append({
                            "Match": f"{home} v {away}",
                            "Bet": vb.outcome.replace("_", " ").title(),
                            "Odds": f"{float(vb.odds):.2f}",
                            "Edge": f"+{float(vb.edge):.1%}",
                            "Result": "✅ Won" if won else "❌ Lost",
                            "Profit": f"£{profit:+.2f}",
                        })

            if value_bet_results:
                st.markdown("---")
                st.subheader("💰 Value Bet Results")
                st.dataframe(value_bet_results, use_container_width=True, hide_index=True)

                total_profit = sum(float(r["Profit"].replace("£", "").replace("+", "")) for r in value_bet_results)
                wins = sum(1 for r in value_bet_results if "Won" in r["Result"])
                st.markdown(f"**Total: {wins}/{len(value_bet_results)} won, £{total_profit:+.2f}**")

    # TAB 2: Season Overview
    with content_tab2:
        st.subheader(f"Season Overview - {selected_season}")

        mw_stats = load_season_summary(selected_season)

        if not mw_stats:
            st.info("No completed matches found")
        else:
            # Overall metrics
            total_correct = sum(s["correct"] for s in mw_stats.values())
            total_matches = sum(s["total"] for s in mw_stats.values())

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Matches", total_matches)
            col2.metric("Correct Predictions", total_correct)
            col3.metric("Overall Accuracy", f"{100*total_correct/total_matches:.1f}%" if total_matches > 0 else "N/A")

            st.markdown("---")

            # Accuracy by matchweek chart
            chart_data = []
            for mw in sorted(mw_stats.keys()):
                stats = mw_stats[mw]
                pct = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
                chart_data.append({
                    "Matchweek": mw,
                    "Accuracy %": pct,
                    "Correct": stats["correct"],
                    "Total": stats["total"],
                })

            df = pd.DataFrame(chart_data)

            st.markdown("**Prediction Accuracy by Matchweek**")
            st.bar_chart(df.set_index("Matchweek")["Accuracy %"])

            # Table
            st.dataframe(df, use_container_width=True, hide_index=True)

    # TAB 3: Value Bets
    with content_tab3:
        st.subheader(f"Value Bet Performance - {selected_season}")

        performance = load_season_value_bets(selected_season)

        if not performance:
            st.info("No value bets found for this season")
        else:
            # Summary metrics
            total_bets = len(performance)
            wins = sum(1 for p in performance if p["won"])
            total_profit = sum(p["profit"] for p in performance)
            total_staked = total_bets * 10.0
            roi = (total_profit / total_staked) * 100 if total_staked > 0 else 0

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Bets", total_bets)
            col2.metric("Win Rate", f"{100*wins/total_bets:.1f}%")
            col3.metric("ROI", f"{roi:+.1f}%")
            col4.metric("Net Profit", f"£{total_profit:+.2f}")

            st.markdown("---")

            # Cumulative profit chart
            mw_perf = defaultdict(lambda: {"bets": 0, "wins": 0, "profit": 0})
            for p in performance:
                mw_perf[p["matchweek"]]["bets"] += 1
                if p["won"]:
                    mw_perf[p["matchweek"]]["wins"] += 1
                mw_perf[p["matchweek"]]["profit"] += p["profit"]

            chart_data = []
            cumulative_profit = 0
            for mw in sorted(mw_perf.keys()):
                perf = mw_perf[mw]
                cumulative_profit += perf["profit"]
                chart_data.append({
                    "Matchweek": mw,
                    "Bets": perf["bets"],
                    "Wins": perf["wins"],
                    "MW Profit": f"£{perf['profit']:+.2f}",
                    "Cumulative": cumulative_profit,
                })

            df = pd.DataFrame(chart_data)

            st.markdown("**Cumulative Profit**")
            st.line_chart(df.set_index("Matchweek")["Cumulative"])

            # Performance by matchweek table
            st.markdown("**Performance by Matchweek**")
            st.dataframe(df, use_container_width=True, hide_index=True)

            st.markdown("---")

            # Performance by outcome
            st.markdown("**Performance by Outcome**")
            outcome_perf = defaultdict(lambda: {"bets": 0, "wins": 0, "profit": 0})
            for p in performance:
                outcome_perf[p["outcome"]]["bets"] += 1
                if p["won"]:
                    outcome_perf[p["outcome"]]["wins"] += 1
                outcome_perf[p["outcome"]]["profit"] += p["profit"]

            outcome_data = []
            for outcome in ["home_win", "draw", "away_win"]:
                if outcome in outcome_perf:
                    perf = outcome_perf[outcome]
                    staked = perf["bets"] * 10.0
                    outcome_roi = (perf["profit"] / staked) * 100 if staked > 0 else 0
                    outcome_data.append({
                        "Outcome": outcome.replace("_", " ").title(),
                        "Bets": perf["bets"],
                        "Wins": perf["wins"],
                        "Win Rate": f"{100*perf['wins']/perf['bets']:.1f}%" if perf["bets"] > 0 else "N/A",
                        "Profit": f"£{perf['profit']:+.2f}",
                        "ROI": f"{outcome_roi:+.1f}%",
                    })

            if outcome_data:
                st.dataframe(outcome_data, use_container_width=True, hide_index=True)

            st.markdown("---")

            # All bets table
            with st.expander("All Value Bets", expanded=False):
                bet_data = []
                for p in performance:
                    home = teams.get(p["home_team_id"], {}).get("short_name", "?")
                    away = teams.get(p["away_team_id"], {}).get("short_name", "?")
                    bet_data.append({
                        "MW": p["matchweek"],
                        "Match": f"{home} v {away}",
                        "Bet": p["outcome"].replace("_", " ").title(),
                        "Odds": f"{p['odds']:.2f}",
                        "Edge": f"+{p['edge']:.1%}",
                        "Result": "✅" if p["won"] else "❌",
                        "Profit": f"£{p['profit']:+.2f}",
                    })

                st.dataframe(bet_data, use_container_width=True, hide_index=True)
