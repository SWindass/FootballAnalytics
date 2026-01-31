"""Historical results browser - view past matchweeks and value bet performance."""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / "scripts"))

# Initialize database with Streamlit secrets BEFORE other imports
import db_init  # noqa: F401

import streamlit as st
import pandas as pd
from datetime import datetime, timezone
from sqlalchemy import select, func
from sqlalchemy.orm import joinedload
from collections import defaultdict

from app.db.database import SyncSessionLocal
from app.db.models import Match, MatchAnalysis, MatchStatus, Team, ValueBet
from app.core.config import get_settings
from auth import require_auth, show_user_info
from pwa import inject_pwa_tags

settings = get_settings()

st.set_page_config(page_title="Historical Results", page_icon="📊", layout="wide")

# PWA support
inject_pwa_tags()

# Auth check - admin only
require_auth(allowed_roles=["admin"])
show_user_info()


def _check_bet_won(outcome: str, home_score: int, away_score: int) -> bool:
    """Check if a bet won based on outcome type and scores."""
    if outcome == "home_win":
        return home_score > away_score
    elif outcome == "away_win":
        return away_score > home_score
    elif outcome == "draw":
        return home_score == away_score
    elif outcome == "over_2_5":
        return (home_score + away_score) > 2.5
    elif outcome == "under_2_5":
        return (home_score + away_score) < 2.5
    elif outcome == "btts_yes":
        return home_score > 0 and away_score > 0
    elif outcome == "btts_no":
        return home_score == 0 or away_score == 0
    return False


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
    """Get list of completed matchweeks for a season (all matches finished)."""
    with SyncSessionLocal() as session:
        # Get matchweeks that have at least one finished match
        # and no scheduled/in-progress matches
        from sqlalchemy import and_, not_, exists

        # Subquery: matchweeks with any non-finished matches
        incomplete_mw = (
            select(Match.matchweek)
            .where(Match.season == season)
            .where(Match.status != MatchStatus.FINISHED)
            .distinct()
        ).subquery()

        stmt = (
            select(Match.matchweek)
            .where(Match.season == season)
            .where(Match.status == MatchStatus.FINISHED)
            .where(Match.matchweek.not_in(select(incomplete_mw)))
            .distinct()
            .order_by(Match.matchweek)
        )
        return [mw for (mw,) in session.execute(stmt).all()]


@st.cache_data(ttl=300)
def get_season_data_availability(season: str) -> dict:
    """Check what data is available for a season."""
    with SyncSessionLocal() as session:
        from sqlalchemy import text

        # Count matches
        match_count = session.execute(
            select(func.count(Match.id))
            .where(Match.season == season)
            .where(Match.status == MatchStatus.FINISHED)
        ).scalar() or 0

        # Count matches with xG
        xg_count = session.execute(
            select(func.count(Match.id))
            .where(Match.season == season)
            .where(Match.status == MatchStatus.FINISHED)
            .where(Match.home_xg.isnot(None))
        ).scalar() or 0

        # Count matches with odds
        odds_count = session.execute(text('''
            SELECT COUNT(*)
            FROM match_analyses ma
            JOIN matches m ON m.id = ma.match_id
            WHERE m.season = :season
            AND m.status = 'finished'
            AND ma.features IS NOT NULL
            AND ma.features->>'historical_odds' IS NOT NULL
        '''), {'season': season}).scalar() or 0

        return {
            "matches": match_count,
            "has_xg": xg_count > 0,
            "xg_count": xg_count,
            "xg_pct": (xg_count / match_count * 100) if match_count > 0 else 0,
            "has_odds": odds_count > 0,
            "odds_count": odds_count,
            "odds_pct": (odds_count / match_count * 100) if match_count > 0 else 0,
        }


@st.cache_data(ttl=60)
def get_current_matchweek():
    """Get the current matchweek number for the current season."""
    with SyncSessionLocal() as session:
        now = datetime.now(timezone.utc)

        # Find matchweek with scheduled matches (upcoming)
        stmt = (
            select(Match.matchweek)
            .where(Match.season == settings.current_season)
            .where(Match.status == MatchStatus.SCHEDULED)
            .order_by(Match.kickoff_time)
            .limit(1)
        )
        result = session.execute(stmt).scalar_one_or_none()
        if result:
            return result

        # Otherwise get the latest matchweek with finished matches
        stmt = (
            select(Match.matchweek)
            .where(Match.season == settings.current_season)
            .where(Match.status == MatchStatus.FINISHED)
            .order_by(Match.matchweek.desc())
            .limit(1)
        )
        return session.execute(stmt).scalar_one_or_none()


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

        # Batch load all value bets (including inactive - historic bets are inactive)
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
    """Load value bet summary stats by matchweek for a season.

    For current season: uses ValueBet table.
    For historical seasons: calculates on-the-fly from MatchAnalysis.features.
    """
    with SyncSessionLocal() as session:
        mw_stats = defaultdict(lambda: {"bets": 0, "wins": 0, "profit": 0})

        # First try ValueBet table (for current season)
        stmt = (
            select(ValueBet.outcome, ValueBet.odds,
                   Match.matchweek, Match.home_score, Match.away_score, Match.id)
            .join(Match, ValueBet.match_id == Match.id)
            .where(Match.season == season)
            .where(Match.status == MatchStatus.FINISHED)
            .order_by(Match.matchweek)
        )
        vb_rows = list(session.execute(stmt).all())

        if vb_rows:
            # Dedupe: best odds per match/outcome
            best_bets = {}
            for outcome, odds, mw, home_score, away_score, match_id in vb_rows:
                key = (match_id, outcome)
                if key not in best_bets or float(odds) > best_bets[key]["odds"]:
                    won = _check_bet_won(outcome, home_score, away_score)
                    profit = 10 * (float(odds) - 1) if won else -10
                    best_bets[key] = {"mw": mw, "won": won, "profit": profit, "odds": float(odds)}

            for bet in best_bets.values():
                mw_stats[bet["mw"]]["bets"] += 1
                mw_stats[bet["mw"]]["wins"] += int(bet["won"])
                mw_stats[bet["mw"]]["profit"] += bet["profit"]

            return dict(mw_stats)

        # Fall back to calculating from MatchAnalysis (for historical seasons)
        stmt = (
            select(Match.id, Match.matchweek, Match.home_score, Match.away_score,
                   MatchAnalysis.consensus_away_prob, MatchAnalysis.features)
            .join(MatchAnalysis, Match.id == MatchAnalysis.match_id)
            .where(Match.season == season)
            .where(Match.status == MatchStatus.FINISHED)
            .where(MatchAnalysis.consensus_home_prob.isnot(None))
            .order_by(Match.matchweek)
        )
        rows = list(session.execute(stmt).all())

        for match_id, mw, home_score, away_score, away_prob, features in rows:
            hist_odds = features.get("historical_odds", {}) if features else {}
            away_odds = hist_odds.get("avg_away_odds") or hist_odds.get("b365_away_odds")

            if not away_odds:
                continue

            actual = "home_win" if home_score > away_score else ("draw" if home_score == away_score else "away_win")

            # Away win strategy: 5-12% edge
            market_prob = 1.0 / float(away_odds)
            edge = float(away_prob) - market_prob

            if 0.05 <= edge <= 0.12:
                won = "away_win" == actual
                profit = 10 * (float(away_odds) - 1) if won else -10
                mw_stats[mw]["bets"] += 1
                mw_stats[mw]["wins"] += int(won)
                mw_stats[mw]["profit"] += profit

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
                won = _check_bet_won(outcome, home_score, away_score)
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
col1, col2, col3 = st.columns([1, 3, 1])

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
    # Default to current matchweek if on current season, otherwise last matchweek
    if selected_season == settings.current_season:
        current_mw = get_current_matchweek()
        st.session_state.hist_mw = current_mw if current_mw in matchweeks else matchweeks[-1]
    else:
        st.session_state.hist_mw = matchweeks[-1]

selected_mw = st.session_state.hist_mw

st.markdown("---")

# Check data availability and show warnings
data_avail = get_season_data_availability(selected_season)
warnings = []
if not data_avail["has_odds"]:
    warnings.append("**No odds data** - Historical betting odds not available for this season (pre-2000)")
elif data_avail["odds_pct"] < 100:
    warnings.append(f"**Partial odds data** - {data_avail['odds_count']}/{data_avail['matches']} matches ({data_avail['odds_pct']:.0f}%)")

if not data_avail["has_xg"]:
    warnings.append("**No xG data** - Expected goals data not available for this season (pre-2014)")
elif data_avail["xg_pct"] < 100:
    warnings.append(f"**Partial xG data** - {data_avail['xg_count']}/{data_avail['matches']} matches ({data_avail['xg_pct']:.0f}%)")

if warnings:
    st.info(" · ".join(warnings))

# Helper functions for formatting
def format_outcome(outcome: str) -> str:
    """Format outcome for display."""
    return {
        "home_win": "Home",
        "draw": "Draw",
        "away_win": "Away",
        "over_2_5": "Over 2.5",
        "under_2_5": "Under 2.5",
        "btts_yes": "BTTS Yes",
        "btts_no": "BTTS No",
    }.get(outcome, outcome)

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

def check_bet_won(outcome: str, home_score: int, away_score: int) -> bool:
    """Check if a bet won based on outcome type and scores."""
    return _check_bet_won(outcome, home_score, away_score)


def render_results_table(results: list, teams: dict) -> tuple[str, int, int]:
    """Render results as an HTML table. Returns (html, vb_wins, vb_total)."""
    vb_wins = 0
    vb_total = 0

    # CSS styles
    css = """
    <style>
    .results-table {
        border-collapse: collapse;
        font-size: 14px;
        background-color: #0e1117;
    }
    .results-table th {
        background-color: #262730;
        color: #fafafa;
        padding: 8px 10px;
        text-align: left;
        border-bottom: 2px solid #4a4a5a;
    }
    .results-table td {
        padding: 6px 10px;
        border-bottom: 1px solid #3a3a4a;
        vertical-align: top;
        color: #fafafa;
        background-color: #0e1117;
    }
    .results-table tr:hover td {
        background-color: #404050;
        color: #ffffff;
    }
    .results-table .score {
        text-align: center;
        font-weight: bold;
        white-space: nowrap;
    }
    .results-table .vb-line {
        margin: 2px 0;
        white-space: nowrap;
    }
    .results-table .vb-win {
        color: #00cc66;
    }
    .results-table .vb-loss {
        color: #ff4444;
    }
    .results-table .vb-pending {
        color: #aaaaaa;
    }
    .results-table tr:hover .vb-win {
        color: #00ff77;
    }
    .results-table tr:hover .vb-loss {
        color: #ff6666;
    }
    </style>
    """

    # Table header
    html = css + """
    <table class="results-table">
        <thead>
            <tr>
                <th>Home</th>
                <th style="text-align: center;">Score</th>
                <th>Away</th>
                <th>Value Bet</th>
            </tr>
        </thead>
        <tbody>
    """

    for r in results:
        home = teams.get(r["home_team_id"], {}).get("short_name", "?")
        away = teams.get(r["away_team_id"], {}).get("short_name", "?")

        # Score or kickoff time
        if r["is_finished"]:
            score = f"{r['home_score']} - {r['away_score']}"
        else:
            score = r["kickoff"].strftime("%a %H:%M")

        # Value bets - each on its own line
        vb_html = ""
        if r["value_bets"]:
            best_vbs = dedupe_value_bets(r["value_bets"])
            for vb in best_vbs:
                outcome_text = format_outcome(vb['outcome'])
                odds_text = f"@{vb['odds']:.2f}"

                if r["is_finished"]:
                    won = check_bet_won(vb["outcome"], r["home_score"], r["away_score"])
                    vb_total += 1
                    if won:
                        vb_wins += 1
                        css_class = "vb-win"
                        result_text = "✓"
                    else:
                        css_class = "vb-loss"
                        result_text = "✗"
                    vb_html += f'<div class="vb-line {css_class}">{outcome_text} {odds_text} {result_text}</div>'
                else:
                    vb_html += f'<div class="vb-line vb-pending">{outcome_text} {odds_text}</div>'

        html += f"""
            <tr>
                <td>{home}</td>
                <td class="score">{score}</td>
                <td>{away}</td>
                <td>{vb_html}</td>
            </tr>
        """

    html += """
        </tbody>
    </table>
    """

    return html, vb_wins, vb_total


# Content tabs
tab1, tab2, tab3 = st.tabs(["Results", "Season Overview", "Value Bets"])

with tab1:
    # Matchweek slider
    selected_mw = st.select_slider(
        "Matchweek",
        options=matchweeks,
        value=st.session_state.hist_mw,
        format_func=lambda x: f"MW {x}",
    )
    if selected_mw != st.session_state.hist_mw:
        st.session_state.hist_mw = selected_mw

    with st.spinner("Loading results..."):
        results = load_matchweek_results(selected_season, selected_mw)

    if not results:
        st.info("No matches found")
    else:
        # Summary metrics - focus on value bets only
        finished = [r for r in results if r["is_finished"]]

        # Render HTML table
        table_html, vb_wins, vb_total = render_results_table(results, teams)

        col1, col2, col3 = st.columns(3)
        col1.metric("Matches", len(results))
        col2.metric("Completed", len(finished))
        if vb_total > 0:
            col3.metric("Value Bets", f"{vb_wins}/{vb_total} won")

        # Display table
        st.html(table_html)

        # Value bet summary for this matchweek
        vb_results = []
        for r in finished:
            if r["value_bets"]:
                for vb in dedupe_value_bets(r["value_bets"]):
                    won = check_bet_won(vb["outcome"], r["home_score"], r["away_score"])
                    profit = 10 * (vb["odds"] - 1) if won else -10
                    vb_results.append({"won": won, "profit": profit})

        if vb_results:
            wins = sum(1 for v in vb_results if v["won"])
            total_profit = sum(v["profit"] for v in vb_results)
            st.caption(f"Value bets: {wins}/{len(vb_results)} won, £{total_profit:+.0f}")

with tab2:
    st.subheader(f"Value Bet Performance by Matchweek")

    with st.spinner("Loading season data..."):
        mw_stats = load_season_summary(selected_season)

    if not mw_stats:
        if not data_avail["has_odds"]:
            st.warning("No value bet data available - historical odds data not available for this season (pre-2000)")
        else:
            st.info("No value bets found for this season")
    else:
        total_bets = sum(s["bets"] for s in mw_stats.values())
        total_wins = sum(s["wins"] for s in mw_stats.values())
        total_profit = sum(s["profit"] for s in mw_stats.values())
        roi = total_profit / (total_bets * 10) * 100 if total_bets else 0

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Bets", total_bets)
        col2.metric("Win Rate", f"{100*total_wins/total_bets:.0f}%" if total_bets else "N/A")
        col3.metric("ROI", f"{roi:+.1f}%")
        col4.metric("Profit", f"£{total_profit:+.0f}")

        # Profit by matchweek chart
        chart_data = pd.DataFrame([
            {"MW": mw, "Profit": s["profit"]}
            for mw, s in sorted(mw_stats.items())
        ])
        st.bar_chart(chart_data.set_index("MW"))

with tab3:
    st.subheader(f"Value Bet Performance - {selected_season}")

    with st.spinner("Loading value bets..."):
        performance = load_season_value_bets(selected_season)

    if not performance:
        if not data_avail["has_odds"]:
            st.warning("No value bet data available - historical odds data not available for this season (pre-2000)")
        else:
            st.info("No value bets found for this season")
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
        bet_rows = []
        for p in performance[-50:]:  # Last 50 bets
            home = teams.get(p["home_team_id"], {}).get("short_name", "?")
            away = teams.get(p["away_team_id"], {}).get("short_name", "?")
            result_class = "vb-win" if p["won"] else "vb-loss"
            result_text = "✓" if p["won"] else "✗"
            pl_class = "vb-win" if p["profit"] > 0 else "vb-loss"
            bet_rows.append(f"""
                <tr>
                    <td>{p['matchweek']}</td>
                    <td>{home} v {away}</td>
                    <td>{format_outcome(p['outcome'])}</td>
                    <td>{p['odds']:.2f}</td>
                    <td>+{p['edge']:.0%}</td>
                    <td class="{result_class}">{result_text}</td>
                    <td class="{pl_class}">£{p['profit']:+.0f}</td>
                </tr>
            """)

        if bet_rows:
            bet_html = """
            <style>
            .bet-history-table {
                border-collapse: collapse;
                font-size: 14px;
                background-color: #0e1117;
            }
            .bet-history-table th {
                background-color: #262730;
                color: #fafafa;
                padding: 6px 10px;
                text-align: left;
                border-bottom: 2px solid #4a4a5a;
            }
            .bet-history-table td {
                padding: 5px 10px;
                border-bottom: 1px solid #3a3a4a;
                color: #fafafa;
                background-color: #0e1117;
            }
            .bet-history-table tr:hover td {
                background-color: #404050;
                color: #ffffff;
            }
            .bet-history-table .vb-win {
                color: #00cc66;
            }
            .bet-history-table .vb-loss {
                color: #ff4444;
            }
            .bet-history-table tr:hover .vb-win {
                color: #00ff77;
            }
            .bet-history-table tr:hover .vb-loss {
                color: #ff6666;
            }
            </style>
            <table class="bet-history-table">
                <thead>
                    <tr>
                        <th>MW</th>
                        <th>Match</th>
                        <th>Bet</th>
                        <th>Odds</th>
                        <th>Edge</th>
                        <th>Result</th>
                        <th>P/L</th>
                    </tr>
                </thead>
                <tbody>
            """ + "".join(bet_rows) + """
                </tbody>
            </table>
            """
            st.html(bet_html)
