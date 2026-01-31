"""Betting Performance Dashboard - Track P/L and results."""
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
import plotly.express as px
from datetime import datetime, timedelta
from decimal import Decimal
from sqlalchemy import select, func

from app.db.database import SyncSessionLocal
from app.db.models import Match, ValueBet, Team, MatchStatus
from auth import require_auth, show_user_info
from pwa import inject_pwa_tags

st.set_page_config(page_title="Betting Performance", page_icon="💰", layout="wide")

# PWA support
inject_pwa_tags()

# Auth check - admin only
require_auth(allowed_roles=["admin"])
show_user_info()

STANDARD_STAKE = 10.0  # Assumed stake per bet


@st.cache_data(ttl=60)
def load_betting_data():
    """Load all betting data."""
    with SyncSessionLocal() as session:
        # Get all value bets
        bets = list(session.execute(
            select(ValueBet).order_by(ValueBet.created_at)
        ).scalars().all())

        teams = {t.id: t.short_name for t in session.execute(select(Team)).scalars().all()}

        data = []
        for bet in bets:
            match = session.get(Match, bet.match_id)
            home = teams.get(match.home_team_id, "?")
            away = teams.get(match.away_team_id, "?")

            data.append({
                "id": bet.id,
                "match": f"{home} vs {away}",
                "home_team": home,
                "away_team": away,
                "kickoff": match.kickoff_time,
                "outcome": bet.outcome,
                "bookmaker": bet.bookmaker,
                "odds": float(bet.odds),
                "model_prob": float(bet.model_probability),
                "implied_prob": float(bet.implied_probability),
                "edge": float(bet.edge),
                "kelly_stake": float(bet.kelly_stake),
                "is_active": bet.is_active,
                "result": bet.result,
                "profit_loss": float(bet.profit_loss) if bet.profit_loss else None,
                "created_at": bet.created_at,
                "season": match.season,
                "score": f"{match.home_score}-{match.away_score}" if match.home_score is not None else None,
            })

        return data


# Main content
st.title("💰 Betting Performance")
st.markdown("Track value bet results and profit/loss over time.")

data = load_betting_data()

if not data:
    st.info("No value bets recorded yet. Run `python batch/jobs/odds_refresh.py` to find value bets.")
    st.stop()

# Separate active and settled bets
active_bets = [b for b in data if b["is_active"]]
settled_bets = [b for b in data if b["result"] is not None]
pending_bets = [b for b in data if not b["is_active"] and b["result"] is None]

# Summary metrics
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Bets", len(data))
col2.metric("Active", len(active_bets))
col3.metric("Settled", len(settled_bets))

if settled_bets:
    total_profit = sum(b["profit_loss"] or 0 for b in settled_bets)
    col4.metric("Total P/L", f"${total_profit:+.2f}")
else:
    col4.metric("Total P/L", "$0.00")

st.divider()

# Tabs for different views
tab1, tab2, tab3 = st.tabs(["Active Bets", "Settled Results", "Analytics"])

# ============ TAB 1: Active Bets ============
with tab1:
    st.subheader(f"Active Value Bets ({len(active_bets)})")

    if active_bets:
        # Group by match (show best bet per match)
        seen_matches = set()
        display_bets = []

        for bet in sorted(active_bets, key=lambda x: x["edge"], reverse=True):
            if bet["match"] not in seen_matches:
                seen_matches.add(bet["match"])
                display_bets.append(bet)

        for bet in display_bets:
            col1, col2, col3, col4 = st.columns([3, 2, 1, 1])

            with col1:
                st.markdown(f"**{bet['match']}**")
                st.caption(f"Kickoff: {bet['kickoff'].strftime('%a %d %b %H:%M')}")

            with col2:
                outcome_display = bet['outcome'].replace('_', ' ').title()
                st.markdown(f"{outcome_display} @ **{bet['odds']:.2f}**")
                st.caption(f"via {bet['bookmaker']}")

            with col3:
                st.metric("Edge", f"{bet['edge']:.1%}")

            with col4:
                st.metric("Kelly", f"{bet['kelly_stake']:.1%}")

            st.divider()

        # Settle button
        st.markdown("---")
        if st.button("🔄 Settle Finished Bets", type="primary"):
            from batch.jobs.settle_bets import settle_bets
            result = settle_bets()
            if result["settled"] > 0:
                st.success(f"Settled {result['settled']} bets: {result['won']} won, {result['lost']} lost | P/L: ${result['total_profit']:+.2f}")
                st.cache_data.clear()
                st.rerun()
            else:
                st.info("No bets to settle - matches not finished yet")
    else:
        st.info("No active value bets. Run odds refresh to find new opportunities.")

# ============ TAB 2: Settled Results ============
with tab2:
    st.subheader(f"Settled Bets ({len(settled_bets)})")

    if settled_bets:
        # Summary stats
        won = sum(1 for b in settled_bets if b["result"] == "won")
        lost = sum(1 for b in settled_bets if b["result"] == "lost")
        total_profit = sum(b["profit_loss"] or 0 for b in settled_bets)
        win_rate = won / len(settled_bets) * 100 if settled_bets else 0
        roi = total_profit / (len(settled_bets) * STANDARD_STAKE) * 100 if settled_bets else 0

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Win Rate", f"{win_rate:.1f}%")
        col2.metric("Won", won)
        col3.metric("Lost", lost)
        col4.metric("ROI", f"{roi:+.1f}%")

        st.divider()

        # Results table
        results_data = []
        for bet in sorted(settled_bets, key=lambda x: x["created_at"], reverse=True):
            results_data.append({
                "Date": bet["created_at"].strftime("%Y-%m-%d"),
                "Match": bet["match"],
                "Score": bet["score"] or "N/A",
                "Bet": bet["outcome"].replace("_", " ").title(),
                "Odds": f"{bet['odds']:.2f}",
                "Result": "✅ Won" if bet["result"] == "won" else "❌ Lost",
                "P/L": f"${bet['profit_loss']:+.2f}" if bet["profit_loss"] else "$0.00",
            })

        st.dataframe(results_data, use_container_width=True, hide_index=True)

        # Cumulative P/L chart
        st.subheader("Cumulative Profit/Loss")

        sorted_bets = sorted(settled_bets, key=lambda x: x["created_at"])
        cumulative = []
        running_total = 0
        for bet in sorted_bets:
            running_total += bet["profit_loss"] or 0
            cumulative.append({
                "date": bet["created_at"],
                "profit": running_total,
                "match": bet["match"],
            })

        if cumulative:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[c["date"] for c in cumulative],
                y=[c["profit"] for c in cumulative],
                mode='lines+markers',
                name='Cumulative P/L',
                line=dict(color='#2E86AB', width=3),
                fill='tozeroy',
                fillcolor='rgba(46, 134, 171, 0.2)',
                hovertemplate='%{text}<br>P/L: $%{y:.2f}<extra></extra>',
                text=[c["match"] for c in cumulative],
            ))
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            fig.update_layout(
                title="Cumulative Profit/Loss Over Time",
                xaxis_title="Date",
                yaxis_title="Profit/Loss ($)",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No settled bets yet. Bets will be settled automatically after matches finish.")

        if pending_bets:
            st.warning(f"{len(pending_bets)} bets are pending settlement. Click 'Settle Finished Bets' in Active Bets tab.")

# ============ TAB 3: Analytics ============
with tab3:
    st.subheader("Betting Analytics")

    if settled_bets:
        col1, col2 = st.columns(2)

        # By outcome type
        with col1:
            st.markdown("**Performance by Bet Type**")

            by_outcome = {}
            for bet in settled_bets:
                outcome = bet["outcome"]
                if outcome not in by_outcome:
                    by_outcome[outcome] = {"bets": 0, "won": 0, "profit": 0}
                by_outcome[outcome]["bets"] += 1
                if bet["result"] == "won":
                    by_outcome[outcome]["won"] += 1
                by_outcome[outcome]["profit"] += bet["profit_loss"] or 0

            outcome_data = []
            for outcome, stats in by_outcome.items():
                win_rate = stats["won"] / stats["bets"] * 100 if stats["bets"] > 0 else 0
                outcome_data.append({
                    "Bet Type": outcome.replace("_", " ").title(),
                    "Bets": stats["bets"],
                    "Won": stats["won"],
                    "Win %": f"{win_rate:.1f}%",
                    "Profit": f"${stats['profit']:+.2f}",
                })

            st.dataframe(outcome_data, use_container_width=True, hide_index=True)

        # By bookmaker
        with col2:
            st.markdown("**Performance by Bookmaker**")

            by_bookie = {}
            for bet in settled_bets:
                bookie = bet["bookmaker"]
                if bookie not in by_bookie:
                    by_bookie[bookie] = {"bets": 0, "won": 0, "profit": 0}
                by_bookie[bookie]["bets"] += 1
                if bet["result"] == "won":
                    by_bookie[bookie]["won"] += 1
                by_bookie[bookie]["profit"] += bet["profit_loss"] or 0

            bookie_data = []
            for bookie, stats in sorted(by_bookie.items(), key=lambda x: x[1]["profit"], reverse=True)[:10]:
                win_rate = stats["won"] / stats["bets"] * 100 if stats["bets"] > 0 else 0
                bookie_data.append({
                    "Bookmaker": bookie,
                    "Bets": stats["bets"],
                    "Win %": f"{win_rate:.1f}%",
                    "Profit": f"${stats['profit']:+.2f}",
                })

            st.dataframe(bookie_data, use_container_width=True, hide_index=True)

        # Edge vs Actual Win Rate
        st.markdown("**Edge Analysis**")
        st.markdown("Comparing predicted edge to actual results:")

        edge_buckets = {
            "3-5%": {"bets": 0, "won": 0},
            "5-10%": {"bets": 0, "won": 0},
            "10-15%": {"bets": 0, "won": 0},
            "15%+": {"bets": 0, "won": 0},
        }

        for bet in settled_bets:
            edge = bet["edge"] * 100
            if 3 <= edge < 5:
                bucket = "3-5%"
            elif 5 <= edge < 10:
                bucket = "5-10%"
            elif 10 <= edge < 15:
                bucket = "10-15%"
            else:
                bucket = "15%+"

            edge_buckets[bucket]["bets"] += 1
            if bet["result"] == "won":
                edge_buckets[bucket]["won"] += 1

        edge_data = []
        for bucket, stats in edge_buckets.items():
            if stats["bets"] > 0:
                win_rate = stats["won"] / stats["bets"] * 100
                edge_data.append({
                    "Edge Range": bucket,
                    "Bets": stats["bets"],
                    "Won": stats["won"],
                    "Win Rate": f"{win_rate:.1f}%",
                })

        if edge_data:
            st.dataframe(edge_data, use_container_width=True, hide_index=True)

    else:
        st.info("Analytics will be available once bets are settled.")

# Sidebar info
with st.sidebar:
    st.markdown("### Quick Actions")

    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("### Commands")
    st.code("""
# Settle finished bets
python batch/jobs/settle_bets.py --settle

# View performance report
python batch/jobs/settle_bets.py --report

# Refresh odds
python batch/jobs/odds_refresh.py
    """, language="bash")

    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    **Stake Size**: $10 flat per bet

    **Settlement**: Bets are settled
    when matches finish. Click
    'Settle Finished Bets' or run
    the settle job manually.

    **ROI Calculation**:
    `Profit / Total Staked × 100`
    """)
