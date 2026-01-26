"""Upcoming fixtures dashboard with predictions and value bets."""

import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta, timezone
from sqlalchemy import select

from app.db.database import SyncSessionLocal
from app.db.models import Match, MatchAnalysis, MatchStatus, Team, TeamStats, OddsHistory, ValueBet, EloRating
from app.core.config import get_settings

settings = get_settings()


def refresh_results_quick():
    """Quick refresh - fetch latest scores only (no xG, ELO, or retraining)."""
    try:
        import asyncio
        from batch.data_sources.football_data_org import FootballDataClient, parse_match
        from app.db.models import Match

        client = FootballDataClient()
        updated = 0

        with SyncSessionLocal() as session:
            # Fetch recent results (last 7 days)
            date_from = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d")
            date_to = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            season_year = settings.current_season.split("-")[0]

            results = asyncio.run(client.get_matches(
                season=season_year,
                status="FINISHED",
                date_from=date_from,
                date_to=date_to,
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
                    match.home_ht_score = parsed.get("home_ht_score")
                    match.away_ht_score = parsed.get("away_ht_score")
                    updated += 1

            session.commit()

        return updated
    except Exception as e:
        return f"Error: {e}"


def refresh_results_full():
    """Full refresh - scores, xG, ELO, stats, and model retraining."""
    try:
        from batch.jobs.results_update import run_results_update
        result = run_results_update()
        return result.get("matches_updated", 0)
    except Exception as e:
        return f"Error: {e}"


def refresh_odds_and_predictions():
    """Refresh odds from bookmakers and recalculate predictions."""
    try:
        from batch.jobs.odds_refresh import run_odds_refresh
        from batch.jobs.weekly_analysis import run_weekly_analysis

        # Step 1: Refresh odds
        odds_result = run_odds_refresh()
        odds_stored = odds_result.get("odds_stored", 0)
        value_bets = odds_result.get("value_bets_found", 0)

        # Step 2: Recalculate predictions with new market data
        analysis_result = run_weekly_analysis()
        analyses = analysis_result.get("analyses_created", 0)

        return {
            "odds_stored": odds_stored,
            "value_bets": value_bets,
            "analyses": analyses,
        }
    except Exception as e:
        return f"Error: {e}"


def refresh_all_narratives(matchweek: int):
    """Regenerate AI narratives for all matches in a matchweek."""
    try:
        import asyncio
        from batch.ai.narrative_generator import NarrativeGenerator
        from app.db.models import Match, MatchAnalysis, Team

        generator = NarrativeGenerator()
        updated = 0

        with SyncSessionLocal() as session:
            # Get all matches for matchweek
            stmt = (
                select(Match)
                .where(Match.season == settings.current_season)
                .where(Match.matchweek == matchweek)
            )
            matches = list(session.execute(stmt).scalars().all())

            for match in matches:
                analysis = session.execute(
                    select(MatchAnalysis).where(MatchAnalysis.match_id == match.id)
                ).scalar_one_or_none()

                if not analysis:
                    continue

                # Load teams
                home_team = session.get(Team, match.home_team_id)
                away_team = session.get(Team, match.away_team_id)

                # Build data for narrative generator
                match_data = {
                    "home_team": home_team.short_name,
                    "away_team": away_team.short_name,
                    "kickoff_time": match.kickoff_time,
                    "venue": home_team.venue or "TBC",
                }

                # Get probabilities
                elo_probs = (
                    float(analysis.elo_home_prob or 0.4),
                    float(analysis.elo_draw_prob or 0.27),
                    float(analysis.elo_away_prob or 0.33),
                )
                poisson_probs = (
                    float(analysis.poisson_home_prob or 0.4),
                    float(analysis.poisson_draw_prob or 0.27),
                    float(analysis.poisson_away_prob or 0.33),
                )

                # Get market probs from features
                market_probs = None
                if analysis.features:
                    mkt_home = analysis.features.get("market_home_prob")
                    mkt_draw = analysis.features.get("market_draw_prob")
                    mkt_away = analysis.features.get("market_away_prob")
                    if mkt_home:
                        market_probs = (mkt_home, mkt_draw, mkt_away)

                # Build consensus predictions
                predictions = {
                    "home_win": float(analysis.consensus_home_prob or 0.4),
                    "draw": float(analysis.consensus_draw_prob or 0.27),
                    "away_win": float(analysis.consensus_away_prob or 0.33),
                    "predicted_score": f"{float(analysis.predicted_home_goals or 1.5):.1f}-{float(analysis.predicted_away_goals or 1.0):.1f}",
                }

                # Build confidence data
                confidence = float(analysis.confidence or 0)
                confidence_data = {
                    "confidence": confidence,
                    "models_agree": confidence > 0,
                    "elo_probs": elo_probs,
                    "poisson_probs": poisson_probs,
                    "market_probs": market_probs or (0.4, 0.27, 0.33),
                }

                # Build odds data
                odds_data = None
                if market_probs:
                    odds_data = {
                        "home_odds": 1 / market_probs[0] if market_probs[0] > 0 else 0,
                        "draw_odds": 1 / market_probs[1] if market_probs[1] > 0 else 0,
                        "away_odds": 1 / market_probs[2] if market_probs[2] > 0 else 0,
                    }

                # Generate narrative
                narrative = asyncio.run(
                    generator.generate_match_preview(
                        match_data=match_data,
                        home_stats={},
                        away_stats={},
                        predictions=predictions,
                        h2h_history=None,
                        confidence_data=confidence_data,
                        odds=odds_data,
                    )
                )

                # Save to database
                analysis.narrative = narrative
                analysis.narrative_generated_at = datetime.now(timezone.utc)
                updated += 1

            session.commit()

        return updated

    except Exception as e:
        return f"Error: {e}"


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
def load_team_forms():
    """Load current form (last 5 results) for all teams."""
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
    """Get the current matchweek number."""
    with SyncSessionLocal() as session:
        from app.core.config import get_settings
        settings = get_settings()

        # Find the matchweek with the most recent/upcoming matches
        now = datetime.now(timezone.utc)

        # First try to find a matchweek with scheduled matches
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

        # Otherwise get the latest matchweek
        stmt = (
            select(Match.matchweek)
            .where(Match.season == settings.current_season)
            .order_by(Match.matchweek.desc())
            .limit(1)
        )
        return session.execute(stmt).scalar_one_or_none() or 1


@st.cache_data(ttl=60)
def load_matchweek_fixtures(matchweek: int):
    """Load all fixtures for a matchweek (completed and upcoming)."""
    with SyncSessionLocal() as session:
        from app.core.config import get_settings
        settings = get_settings()

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

            # Load ALL active value bets for this match
            # Strategy filtering is done when value bets are created
            from sqlalchemy.orm import joinedload
            value_bets = list(session.execute(
                select(ValueBet)
                .options(joinedload(ValueBet.strategy))
                .where(ValueBet.match_id == match.id)
                .where(ValueBet.is_active == True)
                .order_by(ValueBet.edge.desc())
            ).scalars().all())

            # Determine status
            status = match.status if hasattr(match.status, 'value') else match.status
            if isinstance(status, str):
                is_finished = status.upper() == "FINISHED"
            else:
                is_finished = status == MatchStatus.FINISHED

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


# Load data
teams = load_teams()
team_forms = load_team_forms()

# Get current matchweek early for sidebar button
current_mw = get_current_matchweek()

# Sidebar
with st.sidebar:
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Scores", use_container_width=True, help="Quick refresh - scores only"):
            with st.spinner("Fetching scores..."):
                result = refresh_results_quick()
                if isinstance(result, int):
                    if result > 0:
                        st.success(f"Updated {result}")
                        st.cache_data.clear()
                        st.rerun()
                    else:
                        st.info("No new results")
                else:
                    st.error(result)
    with col2:
        if st.button("🔄 Full", use_container_width=True, help="Full refresh - scores, xG, ELO, stats"):
            with st.spinner("Full refresh (this takes a while)..."):
                result = refresh_results_full()
                if isinstance(result, int):
                    if result > 0:
                        st.success(f"Updated {result}")
                        st.cache_data.clear()
                        st.rerun()
                    else:
                        st.info("No new results")
                else:
                    st.error(result)

    if st.button("🤖 Refresh AI Synopsis", use_container_width=True):
        with st.spinner(f"Regenerating narratives for MW{current_mw}..."):
            result = refresh_all_narratives(current_mw)
            if isinstance(result, int):
                st.success(f"Updated {result} synopses")
                st.cache_data.clear()
                st.rerun()
            else:
                st.error(result)

    if st.button("💰 Refresh Odds & Predictions", use_container_width=True):
        with st.spinner("Fetching odds and recalculating predictions..."):
            result = refresh_odds_and_predictions()
            if isinstance(result, dict):
                st.success(f"Odds: {result['odds_stored']} | Value bets: {result['value_bets']} | Predictions: {result['analyses']}")
                st.cache_data.clear()
                st.rerun()
            else:
                st.error(result)

    st.divider()
    with st.expander("Glossary", expanded=False):
        st.markdown("""
**Edge**
The advantage you have over the bookmaker. Calculated as:
`Model Probability - Implied Probability`
E.g., if our model says 60% but odds imply 50%, edge = 10%

---
**Kelly Stake**
Optimal bet size based on your edge and the odds. Formula:
`(Edge × Odds - 1) / (Odds - 1)`
We use 0.25 fractional Kelly (25% of full Kelly) to reduce variance.

---
**Implied Probability**
What the bookmaker's odds suggest the true probability is:
`1 / Decimal Odds`
E.g., odds of 2.00 imply 50% probability

---
**ELO Rating**
A strength rating system (like chess). Higher = stronger team.
Average is ~1500. Top teams: 1700+, Relegation zone: <1400

---
**xG (Expected Goals)**
Statistical measure of chance quality. An xG of 1.5 means the chances created would typically result in 1.5 goals.

---
**Poisson Model**
Predicts goals using average scoring rates. Assumes goals follow a Poisson distribution.

---
**BTTS**
Both Teams To Score - a bet that both teams will score at least one goal.

---
**Over/Under 2.5**
A bet on whether the total goals will be over or under 2.5 (i.e., 3+ goals or 0-2 goals).
        """)

    with st.expander("Models & Strategy"):
        st.markdown("""
**Consensus** combines three models:
- **ELO** (35%): Team strength ratings
- **Poisson** (40%): Goal distribution model
- **XGBoost** (25%): ML classifier

**Value Bet Strategies** (backtest-validated):

*Away Wins (5-12% edge)*
- Backtest: +20% ROI, 51.6% win rate
- Higher edges (>12%) are overconfident

*Home Wins (filtered)*
- Only odds < 1.70 (short favorites)
- Only edge >= 10%
- Only reliable teams (tracked automatically)
- Backtest: +21% ROI, 83% win rate
        """)

# Load fixtures for current matchweek
fixtures = load_matchweek_fixtures(current_mw)

if not fixtures:
    st.info("No fixtures found for this matchweek.")
    st.stop()

st.title(f"Fixtures - Match Week {current_mw}")

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

        is_finished = fixture.get("is_finished", False)

        # Compact match row
        c1, c2, c3, c4, c5 = st.columns([2.5, 2, 0.3, 2, 0.8])

        with c1:
            home_form = team_forms.get(fixture["home_team_id"], "")
            away_form = team_forms.get(fixture["away_team_id"], "")
            form_str = f"{home_form} v {away_form}" if home_form or away_form else ""

            if is_finished:
                home_score = fixture.get("home_score", 0)
                away_score = fixture.get("away_score", 0)
                st.markdown(f"**{home_name}** {home_score} - {away_score} **{away_name}**")
                st.caption(f"FT • {kickoff_time}" + (f" • Form: {form_str}" if form_str else ""))
            else:
                # Add VALUE BET indicator if value bet exists
                if value_bets:
                    st.markdown(f"**{home_name}** vs **{away_name}** :green-background[💰 VALUE]")
                else:
                    st.markdown(f"**{home_name}** vs **{away_name}**")
                st.caption(kickoff_time + (f" • Form: {form_str}" if form_str else ""))

        with c2:
            st.caption("Prediction")
            if analysis and analysis.consensus_home_prob:
                prob_fig = render_probability_bar(
                    float(analysis.consensus_home_prob),
                    float(analysis.consensus_draw_prob),
                    float(analysis.consensus_away_prob)
                )
                st.plotly_chart(prob_fig, use_container_width=True, key=f"prob_{fixture['id']}")
            if analysis and analysis.predicted_home_goals:
                st.markdown(f"**xG: {float(analysis.predicted_home_goals):.1f} - {float(analysis.predicted_away_goals):.1f}**")

        with c3:
            st.markdown("<div style='border-left: 2px solid #ddd; height: 100px; margin: 0 auto; width: 1px;'></div>", unsafe_allow_html=True)

        with c4:
            if is_finished:
                # Show result summary for finished matches
                home_score = fixture.get("home_score", 0)
                away_score = fixture.get("away_score", 0)
                if home_score > away_score:
                    result = f"{home_name} Win"
                elif away_score > home_score:
                    result = f"{away_name} Win"
                else:
                    result = "Draw"
                st.caption("Result")
                st.markdown(f"**{result}**")
            elif odds:
                h, d, a = float(odds.home_odds), float(odds.draw_odds), float(odds.away_odds)
                st.caption("Best Odds")
                odds_df = {
                    "": ["Frac", "Dec"],
                    "Home": [decimal_to_fraction(h), f"{h:.2f}"],
                    "Draw": [decimal_to_fraction(d), f"{d:.2f}"],
                    "Away": [decimal_to_fraction(a), f"{a:.2f}"],
                }
                st.dataframe(odds_df, hide_index=True, height=107)

        with c5:
            if is_finished:
                # Show if prediction was correct
                if analysis and analysis.consensus_home_prob:
                    home_score = fixture.get("home_score", 0)
                    away_score = fixture.get("away_score", 0)
                    home_prob = float(analysis.consensus_home_prob)
                    draw_prob = float(analysis.consensus_draw_prob)
                    away_prob = float(analysis.consensus_away_prob)

                    predicted = "home" if home_prob > draw_prob and home_prob > away_prob else ("away" if away_prob > draw_prob else "draw")
                    actual = "home" if home_score > away_score else ("away" if away_score > home_score else "draw")

                    if predicted == actual:
                        st.markdown("✅")
                    else:
                        st.markdown("❌")

        # Deduplicate value bets - keep best odds per outcome
        unique_value_bets = {}
        for vb in value_bets:
            outcome = vb.outcome
            if outcome not in unique_value_bets or float(vb.odds) > float(unique_value_bets[outcome].odds):
                unique_value_bets[outcome] = vb
        value_bets_deduped = list(unique_value_bets.values())

        # Show prominent value bet callout for upcoming matches
        if not is_finished and value_bets_deduped:
            best = value_bets_deduped[0]
            outcome_map = {
                "home_win": f"Back {home_name}",
                "draw": "Back the Draw",
                "away_win": f"Back {away_name}",
                "over_2_5": "Over 2.5 Goals",
                "under_2_5": "Under 2.5 Goals",
                "btts_yes": "Both Teams to Score",
                "btts_no": "Both Teams NOT to Score",
            }
            outcome_full = outcome_map.get(best.outcome, best.outcome.replace("_", " ").title())
            odds_dec = float(best.odds)
            odds_frac = decimal_to_fraction(odds_dec)
            edge_pct = float(best.edge)
            kelly = float(best.kelly_stake)

            st.success(f"💰 **VALUE BET: {outcome_full}** @ {odds_frac} ({odds_dec:.2f}) — Edge: **+{edge_pct:.1%}** • Kelly stake: {kelly:.1%}")

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

                    if value_bets_deduped:
                        st.markdown("**💰 Value Bets:**")
                        for vb in value_bets_deduped:
                            outcome = vb.outcome.replace("_", " ").title()
                            odds_dec = float(vb.odds)
                            odds_frac = decimal_to_fraction(odds_dec)
                            strategy_name = ""
                            if hasattr(vb, 'strategy') and vb.strategy:
                                strategy_name = f" [{vb.strategy.name.replace('_', ' ').title()}]"
                            st.markdown(f"- **{outcome}** @ {odds_frac} ({odds_dec:.2f}) — **+{float(vb.edge):.1%} edge** • Kelly: {float(vb.kelly_stake):.1%}{strategy_name}")

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
                    st.markdown("**AI Synopsis**")
                    st.markdown(analysis.narrative)

    st.divider()

# Value bets summary
if any(f["value_bets"] for f in fixtures):
    st.markdown("### 💰 Value Bets")
    st.success(
        "Two backtest-validated strategies:\n\n"
        "**Away wins** (5-12% edge): +20% ROI, 51.6% win rate\n\n"
        "**Home wins** (odds < 1.70, edge ≥ 10%, reliable teams): +21% ROI, 83% win rate"
    )

    # Deduplicate: keep best odds per match+outcome
    best_bets = {}  # (match_id, outcome) -> best value bet
    for f in fixtures:
        for vb in f["value_bets"]:
            key = (f["id"], vb.outcome)
            if key not in best_bets or float(vb.odds) > float(best_bets[key]["vb"].odds):
                best_bets[key] = {"fixture": f, "vb": vb}

    unique = []
    for (match_id, outcome), data in best_bets.items():
        f = data["fixture"]
        vb = data["vb"]
        home = teams.get(f["home_team_id"], {}).get("short_name", "?")
        away = teams.get(f["away_team_id"], {}).get("short_name", "?")
        odds_dec = float(vb.odds)
        bet_type = {"home_win": "Home Win", "away_win": "Away Win", "draw": "Draw"}.get(vb.outcome, vb.outcome)
        strategy_name = ""
        if hasattr(vb, 'strategy') and vb.strategy:
            strategy_name = vb.strategy.name.replace('_', ' ').title()
        unique.append({
            "Match": f"{home} v {away}",
            "Kick-off": f["kickoff"].strftime("%a %H:%M"),
            "Bet": bet_type,
            "Odds": f"{decimal_to_fraction(odds_dec)} ({odds_dec:.2f})",
            "Edge": f"+{float(vb.edge):.1%}",
            "Kelly": f"{float(vb.kelly_stake):.1%}",
            "Strategy": strategy_name or "-",
        })

    unique.sort(key=lambda x: float(x["Edge"].lstrip("+").rstrip("%")), reverse=True)

    if unique:
        st.dataframe(unique, use_container_width=True, hide_index=True)
        total_kelly = sum(float(vb["Kelly"].rstrip("%")) for vb in unique)
        st.markdown(f"**Total recommended stake: {total_kelly:.1f}%** of bankroll across {len(unique)} bets")
