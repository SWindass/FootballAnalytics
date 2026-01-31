"""Strategy Monitoring dashboard - track and manage betting strategies."""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st
from datetime import datetime
from decimal import Decimal

from sqlalchemy import select, func, desc
from app.db.database import SyncSessionLocal
from app.db.models import (
    BettingStrategy,
    StrategyMonitoringSnapshot,
    StrategyOptimizationRun,
    ValueBet,
    Match,
    MatchStatus,
    StrategyStatus,
)
from auth import require_auth, show_user_info
from pwa import inject_pwa_tags

st.set_page_config(page_title="Strategy Monitoring", page_icon="📈", layout="wide")

# PWA support
inject_pwa_tags()

# Auth check - admin only
require_auth(allowed_roles=["admin"])
show_user_info()

st.title("📈 Strategy Monitoring")


def get_strategies():
    """Load all betting strategies with stats."""
    with SyncSessionLocal() as session:
        stmt = select(BettingStrategy).order_by(BettingStrategy.name)
        return list(session.execute(stmt).scalars().all())


def get_recent_snapshots(strategy_id: int, limit: int = 10):
    """Get recent monitoring snapshots for a strategy."""
    with SyncSessionLocal() as session:
        stmt = (
            select(StrategyMonitoringSnapshot)
            .where(StrategyMonitoringSnapshot.strategy_id == strategy_id)
            .order_by(desc(StrategyMonitoringSnapshot.snapshot_date))
            .limit(limit)
        )
        return list(session.execute(stmt).scalars().all())


def get_strategy_bets(strategy_id: int):
    """Get settled bets for a strategy."""
    with SyncSessionLocal() as session:
        stmt = (
            select(ValueBet)
            .join(Match, ValueBet.match_id == Match.id)
            .where(ValueBet.strategy_id == strategy_id)
            .where(ValueBet.result.in_(["won", "lost"]))
            .where(Match.status == MatchStatus.FINISHED)
            .order_by(desc(Match.kickoff_time))
        )
        return list(session.execute(stmt).scalars().all())


def run_monitoring():
    """Run the weekly monitoring job."""
    from batch.jobs.strategy_monitoring_weekly import run_weekly_monitoring
    return run_weekly_monitoring()


def run_optimization(strategy_name: str = None, n_trials: int = 50):
    """Run the optimization job."""
    from batch.jobs.strategy_optimization_monthly import run_monthly_optimization
    return run_monthly_optimization(n_trials=n_trials, strategy_name=strategy_name)


# Load strategies
strategies = get_strategies()

if not strategies:
    st.warning("No strategies found. Run `python -m batch.jobs.seed_strategies` to create them.")
    st.stop()

# Strategy overview cards
st.header("Strategy Overview")

cols = st.columns(len(strategies))
for i, strategy in enumerate(strategies):
    with cols[i]:
        # Status indicator
        status_color = {
            StrategyStatus.ACTIVE: "🟢",
            StrategyStatus.PAUSED: "🟡",
            StrategyStatus.DISABLED: "🔴",
        }.get(strategy.status, "⚪")

        st.subheader(f"{status_color} {strategy.name.replace('_', ' ').title()}")

        # Key metrics
        if strategy.total_bets > 0:
            win_rate = strategy.total_wins / strategy.total_bets
            st.metric("Win Rate", f"{win_rate:.1%}")
        else:
            st.metric("Win Rate", "N/A")

        if strategy.historical_roi:
            roi_delta = None
            if strategy.rolling_50_roi:
                roi_delta = f"{float(strategy.rolling_50_roi) - float(strategy.historical_roi):.1%}"
            st.metric(
                "Historical ROI",
                f"{float(strategy.historical_roi):.1%}",
                delta=roi_delta,
                delta_color="normal"
            )
        else:
            st.metric("Historical ROI", "N/A")

        st.metric("Total Bets", strategy.total_bets)

        if strategy.rolling_50_roi:
            st.metric("Rolling 50 ROI", f"{float(strategy.rolling_50_roi):.1%}")

        status_val = strategy.status.value if hasattr(strategy.status, 'value') else strategy.status
        st.caption(f"Status: {status_val}")
        if strategy.status_reason:
            st.caption(f"Reason: {strategy.status_reason[:50]}...")

st.divider()

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Performance", "🔧 Run Monitoring", "⚙️ Optimization", "📜 History"])

with tab1:
    st.header("Detailed Performance")

    selected_strategy = st.selectbox(
        "Select Strategy",
        strategies,
        format_func=lambda s: s.name.replace("_", " ").title()
    )

    if selected_strategy:
        # Strategy parameters
        with st.expander("Strategy Parameters"):
            st.json(selected_strategy.parameters)
            st.caption(selected_strategy.description)

        # Get bets for this strategy
        bets = get_strategy_bets(selected_strategy.id)

        if bets:
            # Performance metrics
            col1, col2, col3, col4 = st.columns(4)

            n_bets = len(bets)
            n_wins = sum(1 for b in bets if b.result == "won")
            total_profit = sum(float(b.profit_loss or 0) for b in bets)
            roi = total_profit / n_bets if n_bets > 0 else 0

            with col1:
                st.metric("Total Bets", n_bets)
            with col2:
                st.metric("Wins", n_wins)
            with col3:
                st.metric("Win Rate", f"{n_wins/n_bets:.1%}" if n_bets > 0 else "N/A")
            with col4:
                st.metric("ROI", f"{roi:.1%}", delta=f"£{total_profit:.2f} profit")

            # Recent bets table
            st.subheader("Recent Bets")

            recent_bets = bets[:20]
            bet_data = []
            for bet in recent_bets:
                bet_data.append({
                    "Date": bet.created_at.strftime("%Y-%m-%d"),
                    "Outcome": (bet.outcome.value if hasattr(bet.outcome, 'value') else bet.outcome).replace("_", " ").title(),
                    "Odds": f"{float(bet.odds):.2f}",
                    "Edge": f"{float(bet.edge):.1%}",
                    "Result": bet.result.title() if bet.result else "Pending",
                    "P/L": f"£{float(bet.profit_loss):.2f}" if bet.profit_loss else "-",
                })

            st.dataframe(bet_data, use_container_width=True)

            # Rolling ROI chart
            st.subheader("Cumulative Profit")

            import plotly.graph_objects as go

            # Calculate cumulative profit
            sorted_bets = sorted(bets, key=lambda b: b.created_at)
            cumulative = []
            running_total = 0
            for bet in sorted_bets:
                running_total += float(bet.profit_loss or 0)
                cumulative.append({
                    "date": bet.created_at,
                    "profit": running_total,
                })

            if cumulative:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=[c["date"] for c in cumulative],
                    y=[c["profit"] for c in cumulative],
                    mode='lines',
                    name='Cumulative Profit',
                    line=dict(color='#2E86AB', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(46, 134, 171, 0.2)',
                ))

                fig.add_hline(y=0, line_dash="dash", line_color="gray")

                fig.update_layout(
                    title="Cumulative Profit Over Time",
                    xaxis_title="Date",
                    yaxis_title="Profit (£)",
                    height=400,
                )

                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No settled bets found for this strategy.")

with tab2:
    st.header("Run Monitoring")

    st.markdown("""
    ### Weekly Monitoring
    This job creates performance snapshots for each strategy, checks for drift,
    and auto-disables strategies with sustained negative ROI.

    **What it does:**
    - Creates weekly performance snapshots
    - Calculates rolling 30 and 50 bet metrics
    - Runs drift detection (Z-score + CUSUM)
    - Auto-disables strategies if rolling 50 ROI < 0%
    - Triggers alerts for concerning patterns
    """)

    col1, col2 = st.columns([1, 2])

    with col1:
        if st.button("🔄 Run Weekly Monitoring", type="primary", use_container_width=True):
            with st.spinner("Running monitoring..."):
                try:
                    result = run_monitoring()
                    st.success("Monitoring complete!")
                    st.json(result)
                    st.rerun()
                except Exception as e:
                    st.error(f"Monitoring failed: {e}")

    with col2:
        st.info("💡 This runs automatically every Monday at 9AM UTC via Azure Functions.")

    # Show recent snapshots
    st.subheader("Recent Snapshots")

    for strategy in strategies:
        with st.expander(f"{strategy.name.replace('_', ' ').title()}"):
            snapshots = get_recent_snapshots(strategy.id, limit=5)

            if snapshots:
                snapshot_data = []
                for snap in snapshots:
                    snapshot_data.append({
                        "Date": snap.snapshot_date.strftime("%Y-%m-%d %H:%M"),
                        "Type": snap.snapshot_type.value if hasattr(snap.snapshot_type, 'value') else snap.snapshot_type,
                        "Rolling 30 ROI": f"{float(snap.rolling_30_roi):.1%}" if snap.rolling_30_roi else "N/A",
                        "Rolling 50 ROI": f"{float(snap.rolling_50_roi):.1%}" if snap.rolling_50_roi else "N/A",
                        "Z-Score": f"{float(snap.z_score):.2f}" if snap.z_score else "N/A",
                        "Drift": "⚠️ Yes" if snap.is_drift_detected else "✅ No",
                        "Alert": snap.alert_type if snap.alert_triggered else "-",
                    })

                st.dataframe(snapshot_data, use_container_width=True)
            else:
                st.caption("No snapshots yet. Run monitoring to create them.")

with tab3:
    st.header("Parameter Optimization")

    st.markdown("""
    ### Bayesian Optimization
    Uses Optuna to find optimal strategy parameters based on historical data.

    **Process:**
    1. Loads 2 years of historical match data
    2. Runs N trials with different parameter combinations
    3. Applies new parameters only if ROI improves by ≥2%
    """)

    col1, col2 = st.columns(2)

    with col1:
        opt_strategy = st.selectbox(
            "Strategy to Optimize",
            [None] + strategies,
            format_func=lambda s: "All Strategies" if s is None else s.name.replace("_", " ").title()
        )

    with col2:
        n_trials = st.slider("Number of Trials", min_value=10, max_value=200, value=50, step=10)

    st.warning("⚠️ Optimization can take several minutes depending on the number of trials.")

    if st.button("🚀 Run Optimization", type="primary"):
        strategy_name = opt_strategy.name if opt_strategy else None
        with st.spinner(f"Running optimization ({n_trials} trials)..."):
            try:
                result = run_optimization(strategy_name=strategy_name, n_trials=n_trials)
                st.success("Optimization complete!")
                st.json(result)
                st.rerun()
            except Exception as e:
                st.error(f"Optimization failed: {e}")

    # Show optimization history
    st.subheader("Optimization History")

    with SyncSessionLocal() as session:
        stmt = (
            select(StrategyOptimizationRun)
            .order_by(desc(StrategyOptimizationRun.run_date))
            .limit(10)
        )
        opt_runs = list(session.execute(stmt).scalars().all())

    if opt_runs:
        opt_data = []
        for run in opt_runs:
            strategy = next((s for s in strategies if s.id == run.strategy_id), None)
            opt_data.append({
                "Date": run.run_date.strftime("%Y-%m-%d"),
                "Strategy": strategy.name if strategy else f"ID:{run.strategy_id}",
                "Trials": run.n_trials,
                "ROI Before": f"{float(run.backtest_roi_before):.1%}" if run.backtest_roi_before else "N/A",
                "ROI After": f"{float(run.backtest_roi_after):.1%}" if run.backtest_roi_after else "N/A",
                "Applied": "✅" if run.was_applied else "❌",
                "Reason": run.not_applied_reason[:30] + "..." if run.not_applied_reason and len(run.not_applied_reason) > 30 else (run.not_applied_reason or "-"),
            })

        st.dataframe(opt_data, use_container_width=True)
    else:
        st.info("No optimization runs yet.")

with tab4:
    st.header("Strategy History")

    for strategy in strategies:
        with st.expander(f"{strategy.name.replace('_', ' ').title()}", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Configuration**")
                st.write(f"Outcome Type: `{strategy.outcome_type}`")
                st.write(f"Status: `{strategy.status.value if hasattr(strategy.status, 'value') else strategy.status}`")
                st.write(f"Optimization Version: `{strategy.optimization_version}`")

                if strategy.last_optimized_at:
                    st.write(f"Last Optimized: {strategy.last_optimized_at.strftime('%Y-%m-%d')}")
                if strategy.last_backtest_at:
                    st.write(f"Last Backtest: {strategy.last_backtest_at.strftime('%Y-%m-%d')}")

            with col2:
                st.markdown("**Performance**")
                st.write(f"Total Bets: {strategy.total_bets}")
                st.write(f"Total Wins: {strategy.total_wins}")
                st.write(f"Total Profit: £{float(strategy.total_profit):.2f}")
                st.write(f"Losing Streak: {strategy.consecutive_losing_streak}")

            st.markdown("**Parameters**")
            st.json(strategy.parameters)

            if strategy.status_reason:
                st.warning(f"Status Reason: {strategy.status_reason}")
