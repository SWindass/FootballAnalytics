"""Match Details Page - Shows full analysis for a single match."""
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
import plotly.graph_objects as go
from datetime import datetime, timezone
from sqlalchemy import select, or_, and_
from sqlalchemy.orm import joinedload

from app.db.database import SyncSessionLocal
from app.db.models import Match, MatchAnalysis, MatchStatus, Team, TeamStats, OddsHistory, ValueBet, EloRating
from app.core.config import get_settings
from auth import require_auth, show_user_info
from pwa import inject_pwa_tags

settings = get_settings()

st.set_page_config(page_title="Match Details", page_icon="🔍", layout="wide")

# PWA support
inject_pwa_tags()

# Mobile responsive CSS
st.markdown("""
<style>
/* Edge section - left on desktop, centered on mobile */
.edge-section {
    text-align: left;
}
.edge-section .edge-values {
    justify-content: flex-start;
}

/* Center all content on mobile */
@media (max-width: 768px) {
    .edge-section {
        text-align: center !important;
    }
    .edge-section .edge-values {
        justify-content: center !important;
    }
    .block-container {
        padding-left: 0.5rem;
        padding-right: 0.5rem;
    }

    /* Center column content */
    [data-testid="stHorizontalBlock"] > div {
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
        text-align: center !important;
    }

    /* Center images/crests - aggressive targeting */
    [data-testid="stImage"],
    [data-testid="stImage"] > div,
    [data-testid="stImage"] > div > div,
    [data-testid="stImage"] > div > div > img {
        display: block !important;
        margin-left: auto !important;
        margin-right: auto !important;
        text-align: center !important;
    }

    [data-testid="stHorizontalBlock"] > div:first-child,
    [data-testid="stHorizontalBlock"] > div:last-child {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
    }

    img {
        margin-left: auto !important;
        margin-right: auto !important;
        display: block !important;
    }

    /* Center headings and text */
    h1, h2, h3, h4, h5, h6, p {
        text-align: center !important;
    }

    /* Center markdown content */
    [data-testid="stMarkdown"] {
        text-align: center !important;
    }

    /* Center captions */
    [data-testid="stCaptionContainer"] {
        text-align: center !important;
    }

    /* Center metrics */
    [data-testid="stMetric"] {
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
    }
}
</style>
""", unsafe_allow_html=True)

# Auth check - viewers and admins can access
require_auth(allowed_roles=["viewer", "admin"])
show_user_info()


def decimal_to_fraction(decimal_odds: float) -> str:
    """Convert decimal odds to fractional odds."""
    from fractions import Fraction
    if decimal_odds <= 1:
        return "N/A"
    frac = Fraction(decimal_odds - 1).limit_denominator(100)
    if frac.denominator == 1:
        return f"{frac.numerator}/1"
    return f"{frac.numerator}/{frac.denominator}"


# Get match ID from session state or query params
match_id = st.session_state.get("selected_match_id") or st.query_params.get("id")

if not match_id:
    st.warning("No match selected. Please select a match from the Fixtures page.")
    st.stop()

try:
    match_id = int(match_id)
except (ValueError, TypeError):
    st.error("Invalid match ID")
    st.stop()


# Load match data
@st.cache_data(ttl=60)
def load_match_details(mid: int):
    with SyncSessionLocal() as session:
        match = session.execute(
            select(Match).where(Match.id == mid)
        ).scalar_one_or_none()

        if not match:
            return None

        analysis = session.execute(
            select(MatchAnalysis).where(MatchAnalysis.match_id == mid)
        ).scalar_one_or_none()

        odds = session.execute(
            select(OddsHistory)
            .where(OddsHistory.match_id == mid)
            .order_by(OddsHistory.recorded_at.desc())
            .limit(1)
        ).scalar_one_or_none()

        value_bets = list(session.execute(
            select(ValueBet)
            .options(joinedload(ValueBet.strategy))
            .where(ValueBet.match_id == mid)
            .where(ValueBet.is_active == True)
            .order_by(ValueBet.edge.desc())
        ).scalars().all())

        home_team = session.get(Team, match.home_team_id)
        away_team = session.get(Team, match.away_team_id)

        # Get team stats
        from sqlalchemy import func
        home_stats = session.execute(
            select(TeamStats)
            .where(TeamStats.team_id == match.home_team_id)
            .where(TeamStats.season == match.season)
            .order_by(TeamStats.matchweek.desc())
            .limit(1)
        ).scalar_one_or_none()

        away_stats = session.execute(
            select(TeamStats)
            .where(TeamStats.team_id == match.away_team_id)
            .where(TeamStats.season == match.season)
            .order_by(TeamStats.matchweek.desc())
            .limit(1)
        ).scalar_one_or_none()

        # Get ELO history
        home_elo = list(session.execute(
            select(EloRating)
            .where(EloRating.team_id == match.home_team_id)
            .where(EloRating.season == match.season)
            .order_by(EloRating.matchweek)
        ).scalars().all())

        away_elo = list(session.execute(
            select(EloRating)
            .where(EloRating.team_id == match.away_team_id)
            .where(EloRating.season == match.season)
            .order_by(EloRating.matchweek)
        ).scalars().all())

        # Get head-to-head history (last 10 meetings)
        h2h_matches = list(session.execute(
            select(Match)
            .where(Match.status == MatchStatus.FINISHED)
            .where(
                or_(
                    and_(Match.home_team_id == match.home_team_id, Match.away_team_id == match.away_team_id),
                    and_(Match.home_team_id == match.away_team_id, Match.away_team_id == match.home_team_id),
                )
            )
            .where(Match.id != match.id)
            .order_by(Match.kickoff_time.desc())
            .limit(10)
        ).scalars().all())

        return {
            "match": match,
            "analysis": analysis,
            "odds": odds,
            "value_bets": value_bets,
            "home_team": home_team,
            "away_team": away_team,
            "home_stats": home_stats,
            "away_stats": away_stats,
            "home_elo": [{"matchweek": r.matchweek, "rating": float(r.rating)} for r in home_elo],
            "away_elo": [{"matchweek": r.matchweek, "rating": float(r.rating)} for r in away_elo],
            "h2h_matches": h2h_matches,
        }


data = load_match_details(match_id)

if not data:
    st.error("Match not found")
    st.stop()

match = data["match"]
analysis = data["analysis"]
odds = data["odds"]
value_bets = data["value_bets"]
home_team = data["home_team"]
away_team = data["away_team"]
home_stats = data["home_stats"]
away_stats = data["away_stats"]
home_elo = data["home_elo"]
away_elo = data["away_elo"]
h2h_matches = data["h2h_matches"]

home_name = home_team.short_name if home_team else "Home"
away_name = away_team.short_name if away_team else "Away"
home_crest = home_team.crest_url if home_team else None
away_crest = away_team.crest_url if away_team else None
home_id = match.home_team_id
away_id = match.away_team_id
is_finished = match.status.value == "FINISHED" if hasattr(match.status, 'value') else str(match.status).upper() == "FINISHED"


# --- Header ---

# Build form icons
home_form_icons = ""
if home_stats and home_stats.form:
    home_form_icons = "".join(["🟢" if c == "W" else "🟡" if c == "D" else "🔴" for c in home_stats.form[::-1]])

away_form_icons = ""
if away_stats and away_stats.form:
    away_form_icons = "".join(["🟢" if c == "W" else "🟡" if c == "D" else "🔴" for c in away_stats.form[::-1]])

# Score/vs text
if is_finished:
    score_text = f"{match.home_score} - {match.away_score}"
    sub_text = "Full Time"
else:
    score_text = "vs"
    sub_text = match.kickoff_time.strftime("%a %d %b, %H:%M")

# xG values
home_xg = f"{float(analysis.predicted_home_goals):.1f}" if analysis and analysis.predicted_home_goals else ""
away_xg = f"{float(analysis.predicted_away_goals):.1f}" if analysis and analysis.predicted_away_goals else ""

# Consensus probabilities
home_prob = float(analysis.consensus_home_prob) * 100 if analysis and analysis.consensus_home_prob else 0
draw_prob = float(analysis.consensus_draw_prob) * 100 if analysis and analysis.consensus_draw_prob else 0
away_prob = float(analysis.consensus_away_prob) * 100 if analysis and analysis.consensus_away_prob else 0
has_probs = home_prob > 0 or draw_prob > 0 or away_prob > 0

# Pure HTML header with row-based alignment
st.markdown(f"""
<style>
.match-header-table, .match-header-table tr, .match-header-table td {{
    border: none !important;
    border-collapse: collapse !important;
    border-spacing: 0 !important;
    background: transparent !important;
}}
</style>
<table class="match-header-table" style="width:100%; border-collapse:collapse; text-align:center; border:none;">
    <tr>
        <td style="width:40%; vertical-align:middle; padding:8px; border:none;">
            {'<img src="' + home_crest + '" width="50">' if home_crest else ''}
        </td>
        <td style="width:20%; vertical-align:middle; border:none;">
            <div style="font-size:1.8em; font-weight:700;">{score_text}</div>
        </td>
        <td style="width:40%; vertical-align:middle; padding:8px; border:none;">
            {'<img src="' + away_crest + '" width="50">' if away_crest else ''}
        </td>
    </tr>
    <tr>
        <td style="vertical-align:top; padding:4px; border:none;">
            <div style="font-size:1.1em; font-weight:600;">{home_team.name if home_team else 'Home'}</div>
        </td>
        <td style="vertical-align:top; border:none;">
            <div style="font-size:0.85em; color:gray;">{sub_text}</div>
        </td>
        <td style="vertical-align:top; padding:4px; border:none;">
            <div style="font-size:1.1em; font-weight:600;">{away_team.name if away_team else 'Away'}</div>
        </td>
    </tr>
    <tr>
        <td style="padding:4px; border:none;">
            <div style="font-size:0.9em;">{home_form_icons}</div>
        </td>
        <td style="border:none;"></td>
        <td style="padding:4px; border:none;">
            <div style="font-size:0.9em;">{away_form_icons}</div>
        </td>
    </tr>
    {f'''<tr>
        <td colspan="3" style="padding:8px 4px; border:none;">
            <div style="font-size:1.4em; font-weight:600; color:#888;">{home_xg} - {away_xg}</div>
            <div style="font-size:0.75em; color:#666;">xG</div>
        </td>
    </tr>''' if home_xg and away_xg else ''}
    {f'''<tr>
        <td colspan="3" style="padding:12px 4px 4px 4px; border:none;">
            <div style="display:flex; height:24px; border-radius:4px; overflow:hidden; margin-bottom:4px;">
                <div style="width:{home_prob}%; background:#2E7D32; display:flex; align-items:center; justify-content:center;">
                    <span style="color:white; font-size:0.75em; font-weight:bold;">{home_prob:.0f}%</span>
                </div>
                <div style="width:{draw_prob}%; background:#666; display:flex; align-items:center; justify-content:center;">
                    <span style="color:white; font-size:0.75em; font-weight:bold;">{draw_prob:.0f}%</span>
                </div>
                <div style="width:{away_prob}%; background:#C62828; display:flex; align-items:center; justify-content:center;">
                    <span style="color:white; font-size:0.75em; font-weight:bold;">{away_prob:.0f}%</span>
                </div>
            </div>
            <div style="display:flex; justify-content:space-between; font-size:0.7em; color:#888;">
                <span>Home</span>
                <span>Draw</span>
                <span>Away</span>
            </div>
        </td>
    </tr>''' if has_probs else ''}
</table>
""", unsafe_allow_html=True)

# --- Value Bets ---
if value_bets and not is_finished:
    st.markdown("<h3 style='text-align:center;'>💰 Value Bet</h3>", unsafe_allow_html=True)

    # Style the expander with green background and good contrast
    st.markdown("""
    <style>
    /* Expander container - always green */
    div[data-testid="stExpander"] > details {
        background: linear-gradient(135deg, #1a472a 0%, #2d5a3d 100%) !important;
        border: 1px solid #4ade80 !important;
        border-radius: 8px !important;
    }
    /* Header text - white and bold */
    div[data-testid="stExpander"] > details > summary {
        color: #fff !important;
        font-weight: bold !important;
        background: transparent !important;
    }
    div[data-testid="stExpander"] > details > summary:hover {
        color: #4ade80 !important;
    }
    /* Expanded content area */
    div[data-testid="stExpander"] > details > div {
        background: #1a472a !important;
        color: #fff !important;
    }
    /* All text inside expander */
    div[data-testid="stExpander"] p,
    div[data-testid="stExpander"] span,
    div[data-testid="stExpander"] li,
    div[data-testid="stExpander"] strong {
        color: #fff !important;
    }
    /* Headers inside */
    div[data-testid="stExpander"] h1,
    div[data-testid="stExpander"] h2,
    div[data-testid="stExpander"] h3,
    div[data-testid="stExpander"] h4,
    div[data-testid="stExpander"] h5 {
        color: #4ade80 !important;
    }
    /* Italic/warning text */
    div[data-testid="stExpander"] em {
        color: #fbbf24 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Dedupe - keep best odds per outcome
    best_bets = {}
    for vb in value_bets:
        if vb.outcome not in best_bets or float(vb.odds) > float(best_bets[vb.outcome].odds):
            best_bets[vb.outcome] = vb
    deduped_bets = list(best_bets.values())

    for idx, vb in enumerate(deduped_bets):
        outcome_map = {
            "home_win": f"Back {home_name}", "draw": "Back the Draw", "away_win": f"Back {away_name}",
            "over_2_5": "Over 2.5 Goals", "under_2_5": "Under 2.5 Goals",
            "btts_yes": "Both Teams to Score", "btts_no": "Both Teams NOT to Score",
        }
        outcome_full = outcome_map.get(vb.outcome, vb.outcome.replace("_", " ").title())

        # Generate AI explanation for this bet
        edge_pct = float(vb.edge) * 100
        kelly_pct = float(vb.kelly_stake) * 100
        odds_val = float(vb.odds)
        implied_prob = (1 / odds_val) * 100
        model_prob = implied_prob + edge_pct

        # Stake recommendation based on Kelly
        if kelly_pct < 2:
            stake_advice = "a small stake (1-2% of bankroll)"
        elif kelly_pct < 5:
            stake_advice = "a moderate stake (2-4% of bankroll)"
        else:
            stake_advice = "a confident stake (3-5% of bankroll)"

        explanation = f"""
**What is this Value Bet?**

Our models predict **{outcome_full}** has a **{model_prob:.0f}%** chance of happening, but the bookmaker odds of **{odds_val:.2f}** imply only a **{implied_prob:.0f}%** chance.

This **{edge_pct:.1f}% edge** means we believe the true probability is higher than what the market is pricing.

**How to use it:**

1. **Find these odds** ({odds_val:.2f} or better) at your preferred bookmaker
2. **Stake**: The Kelly Criterion suggests {stake_advice}
3. **Expected value**: For every £10 bet, you'd expect to profit £{edge_pct/10:.2f} on average over many bets

**Why trust this?**

This bet was identified by combining ELO ratings, Poisson models, and historical patterns. Our backtested strategies have shown **+30% ROI** over 5 seasons.

⚠️ *Gambling involves risk. Never bet more than you can afford to lose. Past performance doesn't guarantee future results.*
"""

        with st.expander(f"🎯 {outcome_full} — Odds: {odds_val:.2f} | Edge: +{edge_pct:.1f}% | Kelly: {kelly_pct:.1f}%"):
            st.markdown(f"""
            <div style='background:linear-gradient(135deg, #1a472a 0%, #2d5a3d 100%); padding:12px 16px; border-radius:8px; margin-bottom:10px; border:1px solid #3d7a4d;'>
                <div style='display:flex; gap:15px; flex-wrap:wrap; justify-content:center;'>
                    <span style='color:#aaa;'>Odds: <strong style='color:#fff;'>{odds_val:.2f}</strong></span>
                    <span style='color:#aaa;'>Edge: <strong style='color:#4ade80;'>+{edge_pct:.1f}%</strong></span>
                    <span style='color:#aaa;'>Kelly: <strong style='color:#fff;'>{kelly_pct:.1f}%</strong></span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(explanation)

# --- ELO Chart ---
if home_elo or away_elo:
    st.markdown("<h3 style='text-align:center;'>ELO Rating</h3>", unsafe_allow_html=True)

    # EPL team colours (home kit primary, away kit, is_striped)
    # Keys must match team.short_name from database
    TEAM_COLOURS = {
        "Arsenal": ("#EF0107", "#063672", False),       # Red home, navy away
        "Aston Villa": ("#670E36", "#95BFE5", False),   # Claret home, light blue away
        "Bournemouth": ("#DA291C", "#1E1E1E", False),   # Red/black home
        "Brentford": ("#E30613", "#FBB800", True),      # Red/white stripes home
        "Brighton": ("#0057B8", "#FFCD00", True),       # Blue/white stripes home
        "Burnley": ("#6C1D45", "#99D6EA", False),       # Claret home, light blue away
        "Chelsea": ("#034694", "#DBA111", False),       # Blue home, gold away
        "Crystal Palace": ("#1B458F", "#C4122E", True), # Blue/red stripes home
        "Everton": ("#003399", "#FFFFFF", False),       # Blue home, white away
        "Fulham": ("#FFFFFF", "#CC0000", False),        # White home, red away
        "Ipswich": ("#0044AA", "#FFFFFF", False),       # Blue home, white away
        "Ipswich Town": ("#0044AA", "#FFFFFF", False),  # Alternative name
        "Leeds": ("#FFFFFF", "#1D428A", False),         # White home, blue away
        "Leeds United": ("#FFFFFF", "#1D428A", False),  # Alternative name
        "Leicester": ("#003090", "#FDBE11", False),     # Blue home, gold away
        "Leicester City": ("#003090", "#FDBE11", False),# Alternative name
        "Liverpool": ("#C8102E", "#00B2A9", False),     # Red home, teal away
        "Luton": ("#F78F1E", "#002D62", False),         # Orange home, navy away
        "Luton Town": ("#F78F1E", "#002D62", False),    # Alternative name
        "Man City": ("#6CABDD", "#1C2C5B", False),      # Sky blue home, navy away
        "Man United": ("#DA291C", "#FBE122", False),    # Red home, yellow away
        "Manchester City": ("#6CABDD", "#1C2C5B", False), # Alternative name
        "Manchester United": ("#DA291C", "#FBE122", False), # Alternative name
        "Newcastle": ("#FFFFFF", "#241F20", True),      # Black/white stripes home
        "Newcastle United": ("#FFFFFF", "#241F20", True), # Alternative name
        "Nott'm Forest": ("#E53233", "#FFFFFF", False), # Red home, white away
        "Nottingham Forest": ("#E53233", "#FFFFFF", False), # Alternative name
        "Nottingham": ("#E53233", "#FFFFFF", False),   # Alternative name
        "Forest": ("#E53233", "#FFFFFF", False),       # Alternative name
        "Sheffield United": ("#EE2737", "#FFFFFF", True), # Red/white stripes home
        "Sheffield Utd": ("#EE2737", "#FFFFFF", True),  # Alternative name
        "Southampton": ("#D71920", "#FFFFFF", True),    # Red/white stripes home
        "Sunderland": ("#EB172B", "#000000", True),    # Red/white stripes home
        "Tottenham": ("#FFFFFF", "#132257", False),     # White home, navy away
        "Watford": ("#FBEE23", "#ED2127", False),       # Yellow home, red away
        "West Brom": ("#122F67", "#FFFFFF", True),      # Navy/white stripes home
        "West Ham": ("#7A263A", "#1BB1E7", False),      # Claret home, blue away
        "Wolves": ("#FDB913", "#231F20", False),        # Gold home, black away
        "Wolverhampton": ("#FDB913", "#231F20", False), # Alternative name
    }

    def get_luminance(hex_color):
        """Calculate luminance of a hex color (0=black, 1=white)."""
        hex_color = hex_color.lstrip('#')
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        return (0.299 * r + 0.587 * g + 0.114 * b) / 255

    def is_visible(hex_color):
        """Check if color is visible on both light and dark backgrounds."""
        lum = get_luminance(hex_color)
        return 0.1 <= lum <= 0.9  # Allow darker/lighter colours through

    MID_GREY = "#888888"

    # Get team colours and stripe info
    home_data = TEAM_COLOURS.get(home_name, ("#2E86AB", "#2E86AB", False))
    away_data = TEAM_COLOURS.get(away_name, ("#E94F37", "#E94F37", False))

    # Use team colour if visible, otherwise grey
    home_colour = home_data[0] if is_visible(home_data[0]) else MID_GREY
    away_colour = away_data[0] if is_visible(away_data[0]) else MID_GREY

    # If colours are the same (both grey, or same team colour), use away team's away colour
    if home_colour == away_colour:
        away_colour = away_data[1] if is_visible(away_data[1]) else "#E94F37"

    home_striped = home_data[2]
    away_striped = away_data[2]

    fig = go.Figure()

    if home_elo:
        fig.add_trace(go.Scatter(
            x=[e["matchweek"] for e in home_elo],
            y=[e["rating"] for e in home_elo],
            mode='lines+markers',
            name=f"{home_name} {'(stripes)' if home_striped else ''}",
            line=dict(color=home_colour, width=3, dash='dashdot' if home_striped else 'solid'),
            marker=dict(size=6),
        ))

    if away_elo:
        fig.add_trace(go.Scatter(
            x=[e["matchweek"] for e in away_elo],
            y=[e["rating"] for e in away_elo],
            mode='lines+markers',
            name=f"{away_name} {'(stripes)' if away_striped else ''}",
            line=dict(color=away_colour, width=3, dash='dashdot' if away_striped else 'solid'),
            marker=dict(size=6),
        ))

    fig.add_hline(y=1500, line_dash="dash", line_color="gray", opacity=0.5,
                  annotation_text="League Average", annotation_position="right")

    fig.update_layout(
        height=300,
        margin=dict(l=0, r=0, t=20, b=60),
        legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5),
        xaxis=dict(title="Matchweek", dtick=2),
        yaxis=dict(title="ELO Rating"),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        hovermode='x unified',
    )

    st.plotly_chart(fig, use_container_width=True, config={
        'staticPlot': True,
        'displayModeBar': False
    })

st.divider()


# --- AI Analysis ---
if analysis and analysis.narrative:
    st.subheader("🤖 AI Analysis")
    st.markdown(analysis.narrative)
    st.divider()


# --- Predictions ---
st.subheader("📊 Predictions")

col1, col2 = st.columns(2)

with col1:
    if analysis and analysis.consensus_home_prob:
        st.markdown("**Model Probabilities**")

        pred_data = []
        if analysis.elo_home_prob:
            pred_data.append({
                "Model": "ELO Rating",
                "Home": f"{float(analysis.elo_home_prob):.0%}",
                "Draw": f"{float(analysis.elo_draw_prob):.0%}",
                "Away": f"{float(analysis.elo_away_prob):.0%}"
            })
        if analysis.poisson_home_prob:
            pred_data.append({
                "Model": "Poisson",
                "Home": f"{float(analysis.poisson_home_prob):.0%}",
                "Draw": f"{float(analysis.poisson_draw_prob):.0%}",
                "Away": f"{float(analysis.poisson_away_prob):.0%}"
            })
        # Dixon-Coles model (improved Poisson with goal correlation)
        if hasattr(analysis, 'dixon_coles_home_prob') and analysis.dixon_coles_home_prob:
            pred_data.append({
                "Model": "Dixon-Coles",
                "Home": f"{float(analysis.dixon_coles_home_prob):.0%}",
                "Draw": f"{float(analysis.dixon_coles_draw_prob):.0%}",
                "Away": f"{float(analysis.dixon_coles_away_prob):.0%}"
            })
        # Pi Rating model
        if hasattr(analysis, 'pi_rating_home_prob') and analysis.pi_rating_home_prob:
            pred_data.append({
                "Model": "Pi Rating",
                "Home": f"{float(analysis.pi_rating_home_prob):.0%}",
                "Draw": f"{float(analysis.pi_rating_draw_prob):.0%}",
                "Away": f"{float(analysis.pi_rating_away_prob):.0%}"
            })
        if analysis.xgboost_home_prob:
            pred_data.append({
                "Model": "XGBoost",
                "Home": f"{float(analysis.xgboost_home_prob):.0%}",
                "Draw": f"{float(analysis.xgboost_draw_prob):.0%}",
                "Away": f"{float(analysis.xgboost_away_prob):.0%}"
            })
        pred_data.append({
            "Model": "**Consensus**",
            "Home": f"**{float(analysis.consensus_home_prob):.0%}**",
            "Draw": f"**{float(analysis.consensus_draw_prob):.0%}**",
            "Away": f"**{float(analysis.consensus_away_prob):.0%}**"
        })

        st.dataframe(pred_data, use_container_width=True, hide_index=True)
    else:
        st.info("No predictions available for this match")

with col2:
    if odds:
        st.markdown("**Bookmaker Odds**")

        h, d, a = float(odds.home_odds), float(odds.draw_odds), float(odds.away_odds)

        odds_data = [
            {"": "Decimal", "Home": f"{h:.2f}", "Draw": f"{d:.2f}", "Away": f"{a:.2f}"},
            {"": "Fractional", "Home": decimal_to_fraction(h), "Draw": decimal_to_fraction(d), "Away": decimal_to_fraction(a)},
            {"": "Implied %", "Home": f"{1/h:.0%}", "Draw": f"{1/d:.0%}", "Away": f"{1/a:.0%}"},
        ]
        st.dataframe(odds_data, use_container_width=True, hide_index=True)
    else:
        st.info("No odds available for this match")

# Edge comparison - below the columns
if odds and analysis and analysis.consensus_home_prob:
    h, d, a = float(odds.home_odds), float(odds.draw_odds), float(odds.away_odds)
    home_edge = float(analysis.consensus_home_prob) - (1/h)
    draw_edge = float(analysis.consensus_draw_prob) - (1/d)
    away_edge = float(analysis.consensus_away_prob) - (1/a)

    st.markdown(f"""
    <div class='edge-section'>
        <p style='font-weight:bold;margin-bottom:5px;'>Edge (Model vs Market)</p>
        <div class='edge-values' style='display:flex;gap:20px;text-align:center;'>
            <div><span style='color:gray;font-size:0.8em;'>Home</span><br><strong>{home_edge:+.1%}</strong></div>
            <div><span style='color:gray;font-size:0.8em;'>Draw</span><br><strong>{draw_edge:+.1%}</strong></div>
            <div><span style='color:gray;font-size:0.8em;'>Away</span><br><strong>{away_edge:+.1%}</strong></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.divider()


# --- Head-to-Head History ---
if h2h_matches:
    st.subheader("Head-to-Head History")

    # Calculate H2H stats
    home_wins = 0
    away_wins = 0
    draws = 0

    for m in h2h_matches:
        if m.home_score == m.away_score:
            draws += 1
        elif m.home_team_id == home_id:
            if m.home_score > m.away_score:
                home_wins += 1
            else:
                away_wins += 1
        else:  # Teams swapped
            if m.home_score > m.away_score:
                away_wins += 1
            else:
                home_wins += 1

    # Summary stats
    st.markdown(f"""
    <div style='display:flex;justify-content:center;gap:30px;text-align:center;margin-bottom:15px;'>
        <div><span style='font-size:1.5em;font-weight:bold;'>{home_wins}</span><br><span style='color:gray;font-size:0.85em;'>{home_name}</span></div>
        <div><span style='font-size:1.5em;font-weight:bold;'>{draws}</span><br><span style='color:gray;font-size:0.85em;'>Draws</span></div>
        <div><span style='font-size:1.5em;font-weight:bold;'>{away_wins}</span><br><span style='color:gray;font-size:0.85em;'>{away_name}</span></div>
    </div>
    """, unsafe_allow_html=True)

    # Match list
    h2h_html = ""
    for m in h2h_matches:
        m_home = home_name if m.home_team_id == home_id else away_name
        m_away = away_name if m.away_team_id == away_id else home_name
        date_str = m.kickoff_time.strftime("%d %b %y")

        home_bold = "font-weight:bold;" if m.home_score > m.away_score else ""
        away_bold = "font-weight:bold;" if m.away_score > m.home_score else ""

        h2h_html += f"""
        <div style='display:flex;align-items:center;padding:8px 0;border-bottom:1px solid #333;'>
            <span style='color:gray;font-size:0.8em;width:70px;'>{date_str}</span>
            <span style='flex:1;text-align:right;{home_bold}'>{m_home}</span>
            <span style='width:50px;text-align:center;font-weight:bold;'>{m.home_score} - {m.away_score}</span>
            <span style='flex:1;text-align:left;{away_bold}'>{m_away}</span>
        </div>
        """

    st.markdown(h2h_html, unsafe_allow_html=True)

st.divider()
