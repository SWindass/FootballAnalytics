"""Upcoming fixtures dashboard with predictions and value bets."""

import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta, timezone
from sqlalchemy import select

from app.db.database import SyncSessionLocal
from app.db.models import Match, MatchAnalysis, MatchStatus, Team, TeamStats, OddsHistory, ValueBet, EloRating
from app.core.config import get_settings

settings = get_settings()

st.set_page_config(page_title="Fixtures", page_icon="📅", layout="wide")

# --- Helper Functions ---

def decimal_to_fraction(decimal_odds: float) -> str:
    """Convert decimal odds to fractional odds."""
    from fractions import Fraction
    if decimal_odds <= 1:
        return "N/A"
    frac = Fraction(decimal_odds - 1).limit_denominator(100)
    if frac.denominator == 1:
        return f"{frac.numerator}/1"
    return f"{frac.numerator}/{frac.denominator}"


def refresh_results_quick():
    """Quick refresh - fetch latest scores only."""
    try:
        import asyncio
        from batch.data_sources.football_data_org import FootballDataClient, parse_match

        client = FootballDataClient()
        updated = 0

        with SyncSessionLocal() as session:
            date_from = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d")
            date_to = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            season_year = settings.current_season.split("-")[0]

            results = asyncio.run(client.get_matches(
                season=season_year, status="FINISHED",
                date_from=date_from, date_to=date_to,
            ))

            for result in results:
                parsed = parse_match(result)
                match = session.execute(
                    select(Match).where(Match.external_id == parsed["external_id"])
                ).scalar_one_or_none()

                if match and (match.home_score != parsed.get("home_score")):
                    match.status = parsed["status"]
                    match.home_score = parsed.get("home_score")
                    match.away_score = parsed.get("away_score")
                    updated += 1

            session.commit()
        return updated
    except Exception as e:
        return f"Error: {e}"


def refresh_odds_and_predictions():
    """Refresh odds from bookmakers and recalculate predictions."""
    try:
        from batch.jobs.odds_refresh import run_odds_refresh
        from batch.jobs.weekly_analysis import run_weekly_analysis

        odds_result = run_odds_refresh()
        analysis_result = run_weekly_analysis()

        return {
            "odds_stored": odds_result.get("odds_stored", 0),
            "value_bets": odds_result.get("value_bets_found", 0),
            "analyses": analysis_result.get("analyses_created", 0),
        }
    except Exception as e:
        return f"Error: {e}"


# --- Data Loading ---

@st.cache_data(ttl=60)
def load_teams():
    with SyncSessionLocal() as session:
        teams = list(session.execute(select(Team)).scalars().all())
        return {t.id: {"name": t.name, "short_name": t.short_name} for t in teams}


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


@st.cache_data(ttl=300)
def load_elo_history(team_id: int, season: str):
    with SyncSessionLocal() as session:
        stmt = (
            select(EloRating)
            .where(EloRating.team_id == team_id)
            .where(EloRating.season == season)
            .order_by(EloRating.matchweek)
        )
        ratings = list(session.execute(stmt).scalars().all())
        return [{"matchweek": r.matchweek, "rating": float(r.rating)} for r in ratings]


# --- Load Data ---

teams = load_teams()
team_forms = load_team_forms()
current_mw = get_current_matchweek()
fixtures = load_matchweek_fixtures(current_mw)

# --- Sidebar ---

with st.sidebar:
    st.subheader("Actions")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Scores", use_container_width=True):
            with st.spinner("Fetching..."):
                result = refresh_results_quick()
                if isinstance(result, int) and result > 0:
                    st.success(f"Updated {result}")
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.info("No updates")
    with col2:
        if st.button("💰 Odds", use_container_width=True):
            with st.spinner("Refreshing..."):
                result = refresh_odds_and_predictions()
                if isinstance(result, dict):
                    st.success(f"Done")
                    st.cache_data.clear()
                    st.rerun()

    st.divider()
    st.caption("**Legend**")
    st.caption("H/D/A = Home/Draw/Away probability")
    st.caption("💰 = Value bet identified")


# --- Main Content ---

if not fixtures:
    st.info("No fixtures found for this matchweek.")
    st.stop()

st.title(f"Matchweek {current_mw}")


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
        "over_2_5": "O2.5", "under_2_5": "U2.5",
        "btts_yes": "BTTS Y", "btts_no": "BTTS N",
    }.get(outcome, outcome)


# Build table data
table_rows = []
for f in fixtures:
    home = teams.get(f["home_team_id"], {}).get("short_name", "?")
    away = teams.get(f["away_team_id"], {}).get("short_name", "?")
    analysis = f["analysis"]
    odds = f["odds"]
    vbs = dedupe_value_bets(f["value_bets"])

    # Time/Score
    if f["is_finished"]:
        score = f"{f['home_score']} - {f['away_score']}"
        time_str = "FT"
    else:
        score = "vs"
        time_str = f["kickoff"].strftime("%a %H:%M")

    # Prediction
    if analysis and analysis.consensus_home_prob:
        h_prob = float(analysis.consensus_home_prob)
        d_prob = float(analysis.consensus_draw_prob)
        a_prob = float(analysis.consensus_away_prob)
        pred = f"{h_prob:.0%}/{d_prob:.0%}/{a_prob:.0%}"
    else:
        pred = "-"

    # Odds
    if odds:
        odds_str = f"{float(odds.home_odds):.2f} / {float(odds.draw_odds):.2f} / {float(odds.away_odds):.2f}"
    else:
        odds_str = "-"

    # Value bet
    if vbs:
        vb = vbs[0]
        vb_str = f"{format_outcome(vb.outcome)} +{float(vb.edge):.0%}"
        vb_class = "vb-yes"
    else:
        vb_str = ""
        vb_class = ""

    table_rows.append({
        "id": f["id"],
        "time": time_str,
        "home": home,
        "score": score,
        "away": away,
        "pred": pred,
        "odds": odds_str,
        "vb": vb_str,
        "vb_class": vb_class,
        "fixture": f,
    })


# Render HTML table
css = """
<style>
.fixtures-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 14px;
    background-color: #0e1117;
    cursor: pointer;
}
.fixtures-table th {
    background-color: #262730;
    color: #fafafa;
    padding: 10px 12px;
    text-align: left;
    border-bottom: 2px solid #4a4a5a;
    font-weight: 600;
}
.fixtures-table td {
    padding: 12px;
    border-bottom: 1px solid #2a2a3a;
    color: #fafafa;
    background-color: #0e1117;
}
.fixtures-table tr:hover td {
    background-color: #1a1a2e;
}
.fixtures-table .time {
    color: #888;
    font-size: 12px;
    width: 80px;
}
.fixtures-table .team {
    font-weight: 500;
}
.fixtures-table .score {
    text-align: center;
    font-weight: bold;
    width: 60px;
}
.fixtures-table .pred {
    color: #aaa;
    font-size: 12px;
}
.fixtures-table .odds {
    color: #888;
    font-size: 12px;
}
.fixtures-table .vb-yes {
    color: #00cc66;
    font-weight: 600;
}
</style>
"""

html = css + """
<table class="fixtures-table">
    <thead>
        <tr>
            <th>Time</th>
            <th>Home</th>
            <th style="text-align:center"></th>
            <th>Away</th>
            <th>Prediction</th>
            <th>Odds (H/D/A)</th>
            <th>Value</th>
        </tr>
    </thead>
    <tbody>
"""

for row in table_rows:
    html += f"""
        <tr>
            <td class="time">{row['time']}</td>
            <td class="team">{row['home']}</td>
            <td class="score">{row['score']}</td>
            <td class="team">{row['away']}</td>
            <td class="pred">{row['pred']}</td>
            <td class="odds">{row['odds']}</td>
            <td class="{row['vb_class']}">{row['vb']}</td>
        </tr>
    """

html += """
    </tbody>
</table>
"""

st.html(html)

# Match selector
st.markdown("---")
match_options = [f"{teams.get(f['home_team_id'], {}).get('short_name', '?')} vs {teams.get(f['away_team_id'], {}).get('short_name', '?')}" for f in fixtures]

selected_idx = st.selectbox(
    "Select match for details",
    range(len(fixtures)),
    format_func=lambda i: match_options[i],
    label_visibility="collapsed"
)

# --- Match Details ---

fixture = fixtures[selected_idx]
home_name = teams.get(fixture["home_team_id"], {}).get("short_name", "?")
away_name = teams.get(fixture["away_team_id"], {}).get("short_name", "?")
analysis = fixture["analysis"]
odds = fixture["odds"]
value_bets = dedupe_value_bets(fixture["value_bets"])

st.subheader(f"{home_name} vs {away_name}")

# Value bet callout
if value_bets and not fixture["is_finished"]:
    vb = value_bets[0]
    outcome_map = {
        "home_win": f"Back {home_name}", "draw": "Back the Draw", "away_win": f"Back {away_name}",
        "over_2_5": "Over 2.5 Goals", "under_2_5": "Under 2.5 Goals",
        "btts_yes": "Both Teams to Score", "btts_no": "Both Teams NOT to Score",
    }
    outcome_full = outcome_map.get(vb.outcome, vb.outcome.replace("_", " ").title())
    st.success(f"💰 **{outcome_full}** @ {decimal_to_fraction(float(vb.odds))} ({float(vb.odds):.2f}) — Edge: **+{float(vb.edge):.1%}** • Kelly: {float(vb.kelly_stake):.1%}")

# Three columns: Predictions, Odds, Form
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Predictions**")
    if analysis and analysis.consensus_home_prob:
        pred_data = []
        if analysis.elo_home_prob:
            pred_data.append({"Model": "ELO", "H": f"{float(analysis.elo_home_prob):.0%}", "D": f"{float(analysis.elo_draw_prob):.0%}", "A": f"{float(analysis.elo_away_prob):.0%}"})
        if analysis.poisson_home_prob:
            pred_data.append({"Model": "Poisson", "H": f"{float(analysis.poisson_home_prob):.0%}", "D": f"{float(analysis.poisson_draw_prob):.0%}", "A": f"{float(analysis.poisson_away_prob):.0%}"})
        if analysis.xgboost_home_prob:
            pred_data.append({"Model": "XGBoost", "H": f"{float(analysis.xgboost_home_prob):.0%}", "D": f"{float(analysis.xgboost_draw_prob):.0%}", "A": f"{float(analysis.xgboost_away_prob):.0%}"})
        pred_data.append({"Model": "**Consensus**", "H": f"**{float(analysis.consensus_home_prob):.0%}**", "D": f"**{float(analysis.consensus_draw_prob):.0%}**", "A": f"**{float(analysis.consensus_away_prob):.0%}**"})
        st.dataframe(pred_data, use_container_width=True, hide_index=True)

        if analysis.predicted_home_goals:
            st.caption(f"Expected Goals: {float(analysis.predicted_home_goals):.1f} - {float(analysis.predicted_away_goals):.1f}")
    else:
        st.caption("No predictions available")

with col2:
    st.markdown("**Odds**")
    if odds:
        h, d, a = float(odds.home_odds), float(odds.draw_odds), float(odds.away_odds)
        odds_data = [
            {"": "Decimal", "Home": f"{h:.2f}", "Draw": f"{d:.2f}", "Away": f"{a:.2f}"},
            {"": "Fractional", "Home": decimal_to_fraction(h), "Draw": decimal_to_fraction(d), "Away": decimal_to_fraction(a)},
            {"": "Implied %", "Home": f"{1/h:.0%}", "Draw": f"{1/d:.0%}", "Away": f"{1/a:.0%}"},
        ]
        st.dataframe(odds_data, use_container_width=True, hide_index=True)
    else:
        st.caption("No odds available")

with col3:
    st.markdown("**Form (Last 5)**")
    home_form = team_forms.get(fixture["home_team_id"], "")
    away_form = team_forms.get(fixture["away_team_id"], "")

    def style_form(form: str) -> str:
        styled = ""
        for c in form:
            if c == "W":
                styled += "🟢"
            elif c == "D":
                styled += "🟡"
            elif c == "L":
                styled += "🔴"
        return styled

    form_data = [
        {"Team": home_name, "Form": style_form(home_form), "": home_form},
        {"Team": away_name, "Form": style_form(away_form), "": away_form},
    ]
    st.dataframe(form_data, use_container_width=True, hide_index=True)

# ELO Chart
home_elo = load_elo_history(fixture["home_team_id"], fixture["season"])
away_elo = load_elo_history(fixture["away_team_id"], fixture["season"])

if home_elo or away_elo:
    st.markdown("**ELO Ratings**")
    fig = go.Figure()

    if home_elo:
        fig.add_trace(go.Scatter(
            x=[e["matchweek"] for e in home_elo],
            y=[e["rating"] for e in home_elo],
            mode='lines+markers',
            name=home_name,
            line=dict(color='#2E86AB', width=2),
            marker=dict(size=5),
        ))

    if away_elo:
        fig.add_trace(go.Scatter(
            x=[e["matchweek"] for e in away_elo],
            y=[e["rating"] for e in away_elo],
            mode='lines+markers',
            name=away_name,
            line=dict(color='#E94F37', width=2),
            marker=dict(size=5),
        ))

    fig.add_hline(y=1500, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_layout(
        height=250,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        xaxis=dict(title="Matchweek", dtick=5),
        yaxis=dict(title="ELO Rating"),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    st.plotly_chart(fig, use_container_width=True)

# AI Synopsis
if analysis and analysis.narrative:
    with st.expander("AI Synopsis", expanded=False):
        st.markdown(analysis.narrative)
