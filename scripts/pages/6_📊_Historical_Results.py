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
    """Load value bet performance for a season (deduplicated - best odds per outcome)."""
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

        # Group by match_id + outcome, keep best odds
        best_bets = {}
        for outcome, odds, edge, bookmaker, mw, home_id, away_id, home_score, away_score, match_id in rows:
            key = (match_id, outcome)
            if key not in best_bets or float(odds) > best_bets[key]["odds"]:
                actual = "home_win" if home_score > away_score else ("draw" if home_score == away_score else "away_win")
                won = outcome == actual
                stake = 10.0
                profit = stake * (float(odds) - 1) if won else -stake

                best_bets[key] = {
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
                }

        # Sort by matchweek
        return sorted(best_bets.values(), key=lambda x: (x["matchweek"], x["match_id"]))


# Load data
teams = load_teams()
seasons = get_available_seasons()

if not seasons:
    st.warning("No seasons found in database")
    st.stop()

st.title("📊 Historical Results")

# Default to current season from settings
default_season = settings.current_season if settings.current_season in seasons else seasons[-1]

# Initialize session state
if "hist_season" not in st.session_state:
    st.session_state.hist_season = default_season
if "hist_mw" not in st.session_state:
    st.session_state.hist_mw = None  # Will be set after we know matchweeks

# Ensure season is valid
if st.session_state.hist_season not in seasons:
    st.session_state.hist_season = default_season

# Season navigation
col1, col2, col3, col4, col5, col6 = st.columns([1, 2, 1, 1, 2, 1])

with col1:
    season_idx = seasons.index(st.session_state.hist_season)
    if st.button("◀", key="prev_s", help="Previous season", disabled=season_idx == 0):
        st.session_state.hist_season = seasons[season_idx - 1]
        st.session_state.hist_mw = None  # Reset matchweek when season changes
        st.rerun()

with col2:
    new_season = st.selectbox(
        "Season", seasons,
        index=seasons.index(st.session_state.hist_season),
        label_visibility="collapsed"
    )
    if new_season != st.session_state.hist_season:
        st.session_state.hist_season = new_season
        st.session_state.hist_mw = None
        st.rerun()

with col3:
    season_idx = seasons.index(st.session_state.hist_season)
    if st.button("▶", key="next_s", help="Next season", disabled=season_idx == len(seasons) - 1):
        st.session_state.hist_season = seasons[season_idx + 1]
        st.session_state.hist_mw = None
        st.rerun()

selected_season = st.session_state.hist_season

# Get matchweeks for selected season
matchweeks = get_matchweeks_for_season(selected_season)
if not matchweeks:
    st.warning("No matchweeks found")
    st.stop()

# Initialize/validate matchweek
if st.session_state.hist_mw is None or st.session_state.hist_mw not in matchweeks:
    st.session_state.hist_mw = matchweeks[-1]  # Default to last matchweek

with col4:
    mw_idx = matchweeks.index(st.session_state.hist_mw)
    if st.button("◀", key="prev_m", help="Previous matchweek", disabled=mw_idx == 0):
        st.session_state.hist_mw = matchweeks[mw_idx - 1]
        st.rerun()

with col5:
    new_mw = st.selectbox(
        "Matchweek", matchweeks,
        index=matchweeks.index(st.session_state.hist_mw),
        format_func=lambda x: f"MW {x}",
        label_visibility="collapsed"
    )
    if new_mw != st.session_state.hist_mw:
        st.session_state.hist_mw = new_mw
        st.rerun()

with col6:
    mw_idx = matchweeks.index(st.session_state.hist_mw)
    if st.button("▶", key="next_m", help="Next matchweek", disabled=mw_idx == len(matchweeks) - 1):
        st.session_state.hist_mw = matchweeks[mw_idx + 1]
        st.rerun()

selected_mw = st.session_state.hist_mw

st.markdown("---")

# Helper functions for formatting
def format_outcome(outcome: str) -> str:
    """Format outcome for display."""
    return {"home_win": "Home", "draw": "Draw", "away_win": "Away"}.get(outcome, outcome)

def get_best_value_bet(value_bets: list, outcome: str) -> dict | None:
    """Get best value bet for an outcome (highest odds)."""
    bets = [vb for vb in value_bets if vb["outcome"] == outcome]
    return max(bets, key=lambda x: x["odds"]) if bets else None

def dedupe_value_bets(value_bets: list) -> list:
    """Keep only best odds per outcome."""
    best = {}
    for vb in value_bets:
        outcome = vb["outcome"]
        if outcome not in best or vb["odds"] > best[outcome]["odds"]:
            best[outcome] = vb
    return list(best.values())

# Content tabs
tab1, tab2, tab3 = st.tabs(["Results", "Season Overview", "Value Bets"])

with tab1:
    st.subheader(f"Matchweek {selected_mw}")

    with st.spinner("Loading results..."):
        results = load_matchweek_results(selected_season, selected_mw)

    if not results:
        st.info("No matches found")
    else:
        # Build results table
        table_data = []
        for r in results:
            home = teams.get(r["home_team_id"], {}).get("short_name", "?")
            away = teams.get(r["away_team_id"], {}).get("short_name", "?")

            # Score or kickoff time
            if r["is_finished"]:
                score = f"{r['home_score']} - {r['away_score']}"
            else:
                score = r["kickoff"].strftime("%a %H:%M")

            # Prediction
            prediction = ""
            correct = None
            if r["analysis"]:
                a = r["analysis"]
                probs = {"H": a["home_prob"], "D": a["draw_prob"], "A": a["away_prob"]}
                pred_key = max(probs, key=probs.get)
                prediction = f"{pred_key} {probs[pred_key]:.0%}"

                if r["is_finished"]:
                    actual = "H" if r["home_score"] > r["away_score"] else ("A" if r["away_score"] > r["home_score"] else "D")
                    correct = pred_key == actual

            # Best value bet (deduplicated)
            value_bet = ""
            vb_won = None
            if r["value_bets"]:
                best_vbs = dedupe_value_bets(r["value_bets"])
                if best_vbs:
                    vb = best_vbs[0]  # Show first one
                    value_bet = f"{format_outcome(vb['outcome'])} @ {vb['odds']:.2f} (+{vb['edge']:.0%})"
                    if r["is_finished"]:
                        actual = "home_win" if r["home_score"] > r["away_score"] else ("draw" if r["home_score"] == r["away_score"] else "away_win")
                        vb_won = vb["outcome"] == actual

            row = {
                "Home": home,
                "Score": score,
                "Away": away,
                "Prediction": prediction,
            }

            # Add result indicator
            if correct is not None:
                row[""] = "✓" if correct else "✗"
            elif not r["is_finished"]:
                row[""] = ""
            else:
                row[""] = "-"

            # Add value bet if any
            if value_bet:
                row["Value Bet"] = value_bet
                if vb_won is not None:
                    row["Won"] = "✓" if vb_won else "✗"

            table_data.append(row)

        # Summary metrics
        finished = [r for r in results if r["is_finished"]]
        with_analysis = [r for r in finished if r["analysis"]]
        correct_count = sum(1 for r in with_analysis if (
            ("H" if r["analysis"]["home_prob"] > r["analysis"]["draw_prob"] and r["analysis"]["home_prob"] > r["analysis"]["away_prob"] else ("A" if r["analysis"]["away_prob"] > r["analysis"]["draw_prob"] else "D"))
            == ("H" if r["home_score"] > r["away_score"] else ("A" if r["away_score"] > r["home_score"] else "D"))
        ))

        col1, col2, col3 = st.columns(3)
        col1.metric("Matches", len(results))
        col2.metric("Completed", len(finished))
        if with_analysis:
            col3.metric("Predictions", f"{correct_count}/{len(with_analysis)} correct")

        # Display table
        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Value bet summary for this matchweek
        vb_results = []
        for r in finished:
            if r["value_bets"]:
                actual = "home_win" if r["home_score"] > r["away_score"] else ("draw" if r["home_score"] == r["away_score"] else "away_win")
                home = teams.get(r["home_team_id"], {}).get("short_name", "?")
                away = teams.get(r["away_team_id"], {}).get("short_name", "?")

                for vb in dedupe_value_bets(r["value_bets"]):
                    won = vb["outcome"] == actual
                    profit = 10 * (vb["odds"] - 1) if won else -10
                    vb_results.append({"won": won, "profit": profit})

        if vb_results:
            wins = sum(1 for v in vb_results if v["won"])
            total_profit = sum(v["profit"] for v in vb_results)
            st.caption(f"Value bets: {wins}/{len(vb_results)} won, £{total_profit:+.0f}")

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

        # By outcome summary
        by_outcome = defaultdict(lambda: {"bets": 0, "wins": 0, "profit": 0})
        for p in performance:
            by_outcome[p["outcome"]]["bets"] += 1
            by_outcome[p["outcome"]]["wins"] += int(p["won"])
            by_outcome[p["outcome"]]["profit"] += p["profit"]

        st.subheader("By Outcome")
        outcome_cols = st.columns(3)
        for i, (outcome, label) in enumerate([("home_win", "Home"), ("draw", "Draw"), ("away_win", "Away")]):
            if outcome in by_outcome:
                d = by_outcome[outcome]
                win_rate = d["wins"] / d["bets"] * 100 if d["bets"] else 0
                outcome_roi = d["profit"] / (d["bets"] * 10) * 100 if d["bets"] else 0
                with outcome_cols[i]:
                    st.metric(label, f"{d['bets']} bets")
                    st.caption(f"{d['wins']} wins ({win_rate:.0f}%), ROI: {outcome_roi:+.0f}%")

        # Recent bets table
        st.subheader("Bet History")
        bet_table = []
        for p in performance[-50:]:  # Last 50 bets
            home = teams.get(p["home_team_id"], {}).get("short_name", "?")
            away = teams.get(p["away_team_id"], {}).get("short_name", "?")
            bet_table.append({
                "MW": p["matchweek"],
                "Match": f"{home} v {away}",
                "Bet": format_outcome(p["outcome"]),
                "Odds": f"{p['odds']:.2f}",
                "Edge": f"+{p['edge']:.0%}",
                "Result": "✓" if p["won"] else "✗",
                "P/L": f"£{p['profit']:+.0f}",
            })

        if bet_table:
            st.dataframe(pd.DataFrame(bet_table), use_container_width=True, hide_index=True)
