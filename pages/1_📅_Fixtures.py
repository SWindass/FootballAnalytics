"""Upcoming fixtures dashboard with predictions and value bets."""
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
from datetime import datetime, timedelta, timezone
from sqlalchemy import select
from itertools import groupby

from app.db.database import SyncSessionLocal
from app.db.models import Match, MatchAnalysis, MatchStatus, Team, TeamStats, OddsHistory, ValueBet
from app.core.config import get_settings
from auth import require_auth, show_user_info, get_current_user
from pwa import inject_pwa_tags

settings = get_settings()

st.set_page_config(page_title="Fixtures", page_icon="📅", layout="wide")

# PWA support
inject_pwa_tags()

# Auth check - viewers and admins can access
require_auth(allowed_roles=["viewer", "admin"])
show_user_info()


# --- Data Loading ---

@st.cache_data(ttl=60)
def load_teams():
    with SyncSessionLocal() as session:
        teams = list(session.execute(select(Team)).scalars().all())
        return {t.id: {"name": t.name, "short_name": t.short_name, "crest_url": t.crest_url} for t in teams}


@st.cache_data(ttl=60)
def load_team_forms():
    from sqlalchemy import func
    with SyncSessionLocal() as session:
        subq = (
            select(TeamStats.team_id, func.max(TeamStats.matchweek).label("max_mw"))
            .where(TeamStats.season == settings.current_season)
            .group_by(TeamStats.team_id)
            .subquery()
        )
        stmt = (
            select(TeamStats)
            .join(subq, (TeamStats.team_id == subq.c.team_id) & (TeamStats.matchweek == subq.c.max_mw))
            .where(TeamStats.season == settings.current_season)
        )
        stats = list(session.execute(stmt).scalars().all())
        return {s.team_id: s.form for s in stats if s.form}


@st.cache_data(ttl=60)
def get_matchweeks():
    with SyncSessionLocal() as session:
        stmt = (
            select(Match.matchweek)
            .where(Match.season == settings.current_season)
            .distinct()
            .order_by(Match.matchweek)
        )
        return [r for r in session.execute(stmt).scalars().all()]


@st.cache_data(ttl=60)
def get_current_matchweek():
    with SyncSessionLocal() as session:
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

        stmt = (
            select(Match.matchweek)
            .where(Match.season == settings.current_season)
            .order_by(Match.matchweek.desc())
            .limit(1)
        )
        return session.execute(stmt).scalar_one_or_none() or 1


@st.cache_data(ttl=60)
def load_matchweek_fixtures(matchweek: int):
    with SyncSessionLocal() as session:
        stmt = (
            select(Match)
            .where(Match.season == settings.current_season)
            .where(Match.matchweek == matchweek)
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

            from sqlalchemy.orm import joinedload
            value_bets = list(session.execute(
                select(ValueBet)
                .options(joinedload(ValueBet.strategy))
                .where(ValueBet.match_id == match.id)
                .where(ValueBet.is_active == True)
                .order_by(ValueBet.edge.desc())
            ).scalars().all())

            status = match.status if hasattr(match.status, 'value') else match.status
            is_finished = status == MatchStatus.FINISHED if not isinstance(status, str) else status.upper() == "FINISHED"

            fixtures.append({
                "id": match.id,
                "season": match.season,
                "matchweek": match.matchweek,
                "kickoff": match.kickoff_time,
                "home_team_id": match.home_team_id,
                "away_team_id": match.away_team_id,
                "home_score": match.home_score,
                "away_score": match.away_score,
                "is_finished": is_finished,
                "analysis": analysis,
                "odds": odds,
                "value_bets": value_bets,
            })

        return fixtures


def dedupe_value_bets(value_bets):
    """Keep best odds per outcome."""
    best = {}
    for vb in value_bets:
        if vb.outcome not in best or float(vb.odds) > float(best[vb.outcome].odds):
            best[vb.outcome] = vb
    return list(best.values())


def format_outcome(outcome: str) -> str:
    return {
        "home_win": "Home", "draw": "Draw", "away_win": "Away",
        "over_2_5": "Over 2.5", "under_2_5": "Under 2.5",
        "btts_yes": "BTTS Yes", "btts_no": "BTTS No",
    }.get(outcome, outcome)


def style_form(form: str) -> str:
    if not form:
        return ""
    return "".join(["🟢" if c == "W" else "🟡" if c == "D" else "🔴" for c in form])


# --- Load Data ---

teams = load_teams()
team_forms = load_team_forms()
matchweeks = get_matchweeks()
current_mw = get_current_matchweek()


# --- Sidebar ---

with st.sidebar:
    st.subheader("Matchweek")

    if matchweeks:
        selected_mw = st.selectbox(
            "Select matchweek",
            matchweeks,
            index=matchweeks.index(current_mw) if current_mw in matchweeks else 0,
            format_func=lambda x: f"MW {x}" + (" (current)" if x == current_mw else ""),
            label_visibility="collapsed"
        )
    else:
        selected_mw = current_mw

    # Admin-only: Update Scores button
    user = get_current_user()
    if user and user.get("role") == "admin":
        st.divider()
        if st.button("🔄 Update Scores", use_container_width=True):
            with st.spinner("Updating scores..."):
                try:
                    from batch.jobs.results_update import run_results_update
                    result = run_results_update()
                    st.success(f"Updated {result['matches_updated']} matches")
                    if result.get('bets_resolved', 0) > 0:
                        st.info(f"Resolved {result['bets_resolved']} value bets")
                    # Clear cached data to show fresh results
                    st.cache_data.clear()
                    st.rerun()
                except Exception as e:
                    st.error(f"Update failed: {e}")

    st.divider()
    st.caption("**Legend**")
    st.caption("🟢 Win  🟡 Draw  🔴 Loss")

    # Version at bottom of sidebar
    st.markdown("---")
    st.caption("v1.0.3")


# --- Load fixtures for selected matchweek ---

fixtures = load_matchweek_fixtures(selected_mw)

if not fixtures:
    st.info("No fixtures found for this matchweek.")
    st.stop()


# --- Custom CSS for BBC-style layout ---

st.markdown("""
<style>
.block-container {
    padding-top: 2.5rem;
    padding-left: 1rem;
    padding-right: 1rem;
}

/* Date header */
.date-header {
    font-size: 14px;
    font-weight: 600;
    color: #aaa;
    padding: 20px 0 10px 0;
    border-bottom: 1px solid #444;
    margin-bottom: 12px;
}

/* Team name buttons */
[data-testid="stHorizontalBlock"] button {
    background: transparent !important;
    border: none !important;
    color: #4da6ff !important;
    font-size: 15px !important;
    font-weight: 500 !important;
    padding: 8px 4px !important;
    min-height: 44px !important;
    width: 100% !important;
    display: flex !important;
    align-items: center !important;
}

/* Home team - right aligned */
[data-testid="stHorizontalBlock"] > div:nth-child(1) button {
    justify-content: flex-end !important;
    text-align: right !important;
}

/* Away team - left aligned */
[data-testid="stHorizontalBlock"] > div:nth-child(5) button {
    justify-content: flex-start !important;
    text-align: left !important;
}
[data-testid="stHorizontalBlock"] button:hover {
    background: transparent !important;
    color: #ffcc00 !important;
}
[data-testid="stHorizontalBlock"] button:focus {
    box-shadow: none !important;
}

/* Score button - centered, larger, with background */
[data-testid="stHorizontalBlock"] > div:nth-child(3) button {
    font-size: 22px !important;
    font-weight: 700 !important;
    background: #ffffff !important;
    color: #000000 !important;
    border-radius: 6px !important;
    padding: 8px 12px !important;
}
[data-testid="stHorizontalBlock"] > div:nth-child(3) button:hover {
    background: #ffcc00 !important;
    color: #000000 !important;
}

/* Force columns to stay on one line - never wrap */
[data-testid="stHorizontalBlock"] {
    flex-wrap: nowrap !important;
    gap: 4px !important;
    display: flex !important;
    flex-direction: row !important;
    align-items: center !important;
}

[data-testid="stHorizontalBlock"] > div {
    flex-shrink: 1 !important;
    min-width: 0 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}

/* Center images vertically */
[data-testid="stImage"] {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}

/* Center score box vertically */
.score-box {
    margin: 0 !important;
}

[data-testid="stHorizontalBlock"] > div:nth-child(3) {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}

[data-testid="stHorizontalBlock"] > div:nth-child(3) > div {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 100% !important;
}

/* Mobile responsive */
@media (max-width: 768px) {
    .block-container {
        padding-left: 0.5rem;
        padding-right: 0.5rem;
    }

    .date-header {
        font-size: 13px;
        padding: 16px 0 8px 0;
    }

    [data-testid="stHorizontalBlock"] button {
        font-size: 13px !important;
        padding: 6px 2px !important;
        min-height: 40px !important;
    }

    [data-testid="stHorizontalBlock"] > div {
        padding: 0 !important;
        min-width: 0 !important;
    }
}


/* Very small screens */
@media (max-width: 480px) {
    h1 {
        font-size: 1.3rem !important;
    }

    [data-testid="stHorizontalBlock"] button {
        font-size: 12px !important;
    }
}

/* Value bet button - green gradient style (primary buttons) */
button[kind="primary"] {
    background: linear-gradient(135deg, #0a4d2e 0%, #0d6b3d 100%) !important;
    color: #00ff88 !important;
    border: 1px solid #00ff88 !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    box-shadow: 0 2px 8px rgba(0, 255, 136, 0.2) !important;
}
button[kind="primary"]:hover {
    background: linear-gradient(135deg, #0d6b3d 0%, #0f8a4d 100%) !important;
    color: #ffffff !important;
    border-color: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)


# --- Main Content ---

st.title(f"Matchweek {selected_mw}")

# Group fixtures by date
fixtures_by_date = {}
for f in fixtures:
    date_key = f["kickoff"].strftime("%A %d %B %Y")
    if date_key not in fixtures_by_date:
        fixtures_by_date[date_key] = []
    fixtures_by_date[date_key].append(f)


# Render fixtures grouped by date
for date_str, day_fixtures in fixtures_by_date.items():
    st.markdown(f"<div class='date-header'>{date_str}</div>", unsafe_allow_html=True)

    for f in day_fixtures:
        home = teams.get(f["home_team_id"], {})
        away = teams.get(f["away_team_id"], {})
        home_name = home.get("short_name", "?")
        away_name = away.get("short_name", "?")
        home_crest = home.get("crest_url", "")
        away_crest = away.get("crest_url", "")
        vbs = dedupe_value_bets(f["value_bets"])

        if f["is_finished"]:
            score_text = f"{f['home_score']} - {f['away_score']}"
            score_bg = "#1a472a"
        else:
            score_text = f["kickoff"].strftime("%H:%M")
            score_bg = "rgba(255,255,255,0.1)"

        match_id = f["id"]

        # Value bet text
        vb_text = ""
        if vbs and not f["is_finished"]:
            vb = vbs[0]
            vb_text = f"{format_outcome(vb.outcome)} +{float(vb.edge):.0%}"

        # Fixture row - 5 columns: home, crest, score, crest, away
        c1, c2, c3, c4, c5 = st.columns([1.7, 0.3, 0.8, 0.3, 1.7])

        with c1:
            if st.button(home_name, key=f"home_{match_id}", use_container_width=True):
                st.session_state["selected_match_id"] = match_id
                st.switch_page("pages/8_🔍_Match_Details.py")

        with c2:
            if home_crest:
                st.image(home_crest, width=28)

        with c3:
            if st.button(score_text, key=f"score_{match_id}", use_container_width=True):
                st.session_state["selected_match_id"] = match_id
                st.switch_page("pages/8_🔍_Match_Details.py")

        with c4:
            if away_crest:
                st.image(away_crest, width=28)

        with c5:
            if st.button(away_name, key=f"away_{match_id}", use_container_width=True):
                st.session_state["selected_match_id"] = match_id
                st.switch_page("pages/8_🔍_Match_Details.py")


        # Value bet - clickable button that navigates to details
        if vb_text:
            vb_col1, vb_col2, vb_col3 = st.columns([1, 2, 1])
            with vb_col2:
                if st.button(f"💰 {vb_text}", key=f"vb_{match_id}", use_container_width=True, type="primary"):
                    st.session_state["selected_match_id"] = match_id
                    st.switch_page("pages/8_🔍_Match_Details.py")

        # Separator after each fixture
        st.markdown("<hr style='margin:0;border:none;border-top:1px solid #333;'>", unsafe_allow_html=True)


# Footer
st.divider()
st.caption(f"Showing {len(fixtures)} fixtures for Matchweek {selected_mw} • Season {settings.current_season}")
