"""Model Training dashboard - view and retrain prediction models."""
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
import json
import plotly.graph_objects as go
from datetime import datetime
from pathlib import Path

from app.db.database import SyncSessionLocal
from app.db.models import Match, MatchAnalysis, MatchStatus
from sqlalchemy import select, func, case
from auth import require_auth, show_user_info
from pwa import inject_pwa_tags

# Model paths
MODEL_DIR = Path(__file__).parent.parent / "batch" / "models" / "saved"
CONSENSUS_META_PATH = MODEL_DIR / "consensus_stacker_meta.json"
NEURAL_META_PATH = MODEL_DIR / "neural_stacker_meta.json"

st.set_page_config(page_title="Model Training", page_icon="🤖", layout="wide")

# PWA support
inject_pwa_tags()

# Auth check - admin only
require_auth(allowed_roles=["admin"])
show_user_info()

st.title("🤖 Model Training")


def load_model_metadata(path: Path) -> dict:
    """Load model metadata from JSON file."""
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}






# Main content
tab1, tab2, tab3 = st.tabs(["📊 Current Performance", "🔧 Training", "📈 Diagnostics"])

with tab1:
    st.header("Model Performance")

    # Load model metadata
    consensus_meta = load_model_metadata(CONSENSUS_META_PATH)
    neural_meta = load_model_metadata(NEURAL_META_PATH)

    # Model info cards
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Consensus Stacker")
        if consensus_meta:
            val_acc = consensus_meta.get("val_accuracy", 0)
            trained_at = consensus_meta.get("trained_at", "Unknown")
            epoch = consensus_meta.get("epoch", 0)

            st.metric("Validation Accuracy", f"{val_acc*100:.1f}%")
            st.caption(f"Best epoch: {epoch}")

            if trained_at != "Unknown":
                try:
                    dt = datetime.fromisoformat(trained_at.replace("Z", "+00:00"))
                    st.caption(f"Trained: {dt.strftime('%Y-%m-%d %H:%M')}")
                except:
                    st.caption(f"Trained: {trained_at}")
        else:
            st.warning("No trained model found")

    with col2:
        st.subheader("Neural Stacker")
        if neural_meta:
            # Neural stacker uses "val_acc" key instead of "val_accuracy"
            val_acc = neural_meta.get("val_accuracy") or neural_meta.get("val_acc", 0)
            trained_at = neural_meta.get("trained_at") or neural_meta.get("saved_at", "Unknown")

            st.metric("Validation Accuracy", f"{val_acc*100:.1f}%")

            if trained_at != "Unknown":
                try:
                    dt = datetime.fromisoformat(trained_at.replace("Z", "+00:00"))
                    st.caption(f"Trained: {dt.strftime('%Y-%m-%d %H:%M')}")
                except:
                    st.caption(f"Trained: {trained_at}")
        else:
            st.warning("No trained model found")

    st.divider()

    # Value Bet Performance Summary
    st.subheader("Value Bet Performance")

    st.info("📊 For detailed value bet performance analysis, see the **Historical Results** page.")

    # Quick summary from value bets
    with SyncSessionLocal() as vb_session:
        from app.db.models import ValueBet

        # Get summary stats
        stmt = (
            select(
                func.count(ValueBet.id),
                func.sum(case((ValueBet.result == 'won', 1), else_=0)),
                func.sum(ValueBet.profit_loss)
            )
            .join(Match, ValueBet.match_id == Match.id)
            .where(Match.status == MatchStatus.FINISHED)
            .where(ValueBet.result.isnot(None))
        )
        result = vb_session.execute(stmt).first()
        total_bets, wins, profit = result if result else (0, 0, 0)

        if total_bets and total_bets > 0:
            wins = wins or 0
            profit = float(profit) if profit else 0

            col1, col2, col3 = st.columns(3)
            col1.metric("Resolved Bets", f"{total_bets:,}")
            col2.metric("Win Rate", f"{wins/total_bets*100:.0f}%")
            col3.metric("Total P/L", f"£{profit:+,.0f}")
        else:
            st.caption("No resolved value bets yet.")

with tab2:
    # Generate Analysis for upcoming matches
    st.header("Generate Match Analysis")

    st.markdown("""
    Generate **predictions and AI narratives** for upcoming matches:

    - ELO and Poisson probability predictions
    - Consensus model predictions
    - AI-generated match previews
    - Value bet detection
    """)

    if st.button("🔮 Generate Analysis", type="primary", use_container_width=True):
        with st.spinner("Generating analysis for upcoming matches..."):
            try:
                from batch.jobs.weekly_analysis import run_weekly_analysis
                result = run_weekly_analysis()

                if result.get("status") == "skipped":
                    st.warning(f"Skipped: {result.get('reason', 'Unknown reason')}")
                else:
                    st.success(f"Generated analysis for {result.get('analyses_created', 0)} matches in MW {result.get('matchweek', '?')}")
                    st.rerun()
            except Exception as e:
                st.error(f"Analysis generation failed: {e}")

    st.divider()

    st.header("Retrain Everything")

    st.markdown("""
    This will perform a **full system refresh**:

    1. **Update Scores** - Fetch latest match results
    2. **Update xG Data** - Fetch expected goals from Understat
    3. **Recalculate ELO** - Rebuild all ELO ratings from scratch
    4. **Update Team Stats** - Recalculate form, goals, clean sheets etc.
    5. **Resolve Value Bets** - Mark finished bets as won/lost
    6. **Train Models** - Retrain both Consensus and Neural stackers
    """)

    st.warning("⚠️ This process takes several minutes. Only run when needed.")

    # Training stats
    with SyncSessionLocal() as session:
        finished_count = session.execute(
            select(func.count(Match.id))
            .where(Match.status == MatchStatus.FINISHED)
        ).scalar()

        with_analysis = session.execute(
            select(func.count(MatchAnalysis.id))
            .join(Match, Match.id == MatchAnalysis.match_id)
            .where(Match.status == MatchStatus.FINISHED)
            .where(MatchAnalysis.elo_home_prob.isnot(None))
            .where(MatchAnalysis.poisson_home_prob.isnot(None))
        ).scalar()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Finished Matches", f"{finished_count:,}")
    with col2:
        st.metric("With Complete Predictions", f"{with_analysis:,}")

    st.divider()

    # Single retrain button
    if st.button("🔄 Retrain Everything", type="primary", use_container_width=True):
        with st.spinner("Running full recalculation... This may take several minutes."):
            try:
                from batch.jobs.results_update import run_full_recalculation
                result = run_full_recalculation()

                st.success("Full recalculation complete!")

                # Show summary
                col1, col2, col3 = st.columns(3)
                col1.metric("Scores Updated", result.get("matches_updated", 0))
                col2.metric("ELO Matchweeks", result.get("elo_recalculated", 0))
                col3.metric("Bets Resolved", result.get("bets_resolved", 0))

                # Show model training results
                models = result.get("models_trained", {})
                if models:
                    st.subheader("Model Training Results")
                    for model_name, model_result in models.items():
                        if "error" in model_result:
                            st.error(f"{model_name}: {model_result['error']}")
                        else:
                            st.json({model_name: model_result})

                st.rerun()
            except Exception as e:
                st.error(f"Recalculation failed: {e}")

    st.divider()

    st.markdown("""
    ### When to Retrain
    - After a matchweek completes (to incorporate new results)
    - If model accuracy seems degraded
    - After making changes to feature engineering
    """)

with tab3:
    st.header("Model Diagnostics")

    # Feature importance / model info
    st.subheader("Consensus Stacker Features")

    features = [
        ("ELO Predictions", "elo_home_prob, elo_draw_prob, elo_away_prob"),
        ("Poisson Predictions", "poisson_home_prob, poisson_draw_prob, poisson_away_prob"),
        ("Market Odds", "market_home_prob, market_draw_prob, market_away_prob"),
        ("Average Predictions", "avg_home_prob, avg_draw_prob, avg_away_prob"),
        ("Disagreement Metrics", "home_std, draw_std, away_std"),
        ("Agreement Signals", "max_agreement, favorite_agreement, prediction_entropy"),
        ("Draw Boost", "disagreement_draw_boost (when models disagree)"),
        ("Strength Features", "elo_diff, market_favorite_strength, model_confidence_gap"),
    ]

    for name, desc in features:
        with st.expander(name):
            st.code(desc)

    st.divider()

    # Outcome distribution
    st.subheader("Historical Outcome Distribution")

    with SyncSessionLocal() as session:
        stmt = (
            select(Match)
            .where(Match.status == MatchStatus.FINISHED)
            .where(Match.home_score.isnot(None))
        )
        matches = session.execute(stmt).scalars().all()

        home_wins = sum(1 for m in matches if m.home_score > m.away_score)
        draws = sum(1 for m in matches if m.home_score == m.away_score)
        away_wins = sum(1 for m in matches if m.home_score < m.away_score)
        total = len(matches)

    if total > 0:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Home Wins", f"{home_wins/total*100:.1f}%", help=f"{home_wins:,} matches")
        with col2:
            st.metric("Draws", f"{draws/total*100:.1f}%", help=f"{draws:,} matches")
        with col3:
            st.metric("Away Wins", f"{away_wins/total*100:.1f}%", help=f"{away_wins:,} matches")

        # Pie chart
        fig = go.Figure(data=[go.Pie(
            labels=['Home Win', 'Draw', 'Away Win'],
            values=[home_wins, draws, away_wins],
            hole=0.4,
            marker_colors=['#2E86AB', '#A23B72', '#F18F01']
        )])

        fig.update_layout(
            title=f"Outcome Distribution ({total:,} matches)",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

        st.caption("""
        Note: Home advantage is real - home teams win ~45% of matches in EPL.
        A good model should predict home wins more often than away wins.
        """)
