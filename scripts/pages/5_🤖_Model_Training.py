"""Model Training dashboard - view and retrain prediction models."""

import streamlit as st
import json
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

from app.db.database import SyncSessionLocal
from app.db.models import Match, MatchAnalysis, MatchStatus
from sqlalchemy import select, func
from auth import require_auth, show_user_info
from pwa import inject_pwa_tags

# Model paths
MODEL_DIR = Path(__file__).parent.parent.parent / "batch" / "models" / "saved"
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


def get_historical_accuracy():
    """Calculate historical accuracy of predictions vs actual results."""
    with SyncSessionLocal() as session:
        # Get finished matches with predictions
        stmt = (
            select(Match, MatchAnalysis)
            .join(MatchAnalysis, Match.id == MatchAnalysis.match_id)
            .where(Match.status == MatchStatus.FINISHED)
            .where(Match.home_score.isnot(None))
            .where(MatchAnalysis.consensus_home_prob.isnot(None))
            .order_by(Match.kickoff_time.desc())
        )
        results = session.execute(stmt).all()

        if not results:
            return None

        # Calculate accuracy
        correct = 0
        total = 0
        by_confidence = {"high": {"correct": 0, "total": 0}, "medium": {"correct": 0, "total": 0}, "low": {"correct": 0, "total": 0}}
        by_agreement = {"agree": {"correct": 0, "total": 0}, "disagree": {"correct": 0, "total": 0}}
        recent_results = []

        for match, analysis in results:
            home_prob = float(analysis.consensus_home_prob)
            draw_prob = float(analysis.consensus_draw_prob)
            away_prob = float(analysis.consensus_away_prob)

            # Determine predicted outcome
            probs = [home_prob, draw_prob, away_prob]
            predicted = np.argmax(probs)  # 0=home, 1=draw, 2=away

            # Determine actual outcome
            if match.home_score > match.away_score:
                actual = 0  # Home win
            elif match.home_score < match.away_score:
                actual = 2  # Away win
            else:
                actual = 1  # Draw

            is_correct = predicted == actual
            if is_correct:
                correct += 1
            total += 1

            # Confidence buckets
            confidence = float(analysis.confidence) if analysis.confidence else 0.33
            if confidence >= 0.5:
                bucket = "high"
            elif confidence >= 0.4:
                bucket = "medium"
            else:
                bucket = "low"

            by_confidence[bucket]["total"] += 1
            if is_correct:
                by_confidence[bucket]["correct"] += 1

            # Model agreement - check if ELO, Poisson, and Market all predict same outcome
            elo_pred = None
            poisson_pred = None
            market_pred = None

            if analysis.elo_home_prob and analysis.elo_draw_prob and analysis.elo_away_prob:
                elo_probs = [float(analysis.elo_home_prob), float(analysis.elo_draw_prob), float(analysis.elo_away_prob)]
                elo_pred = np.argmax(elo_probs)

            if analysis.poisson_home_prob and analysis.poisson_draw_prob and analysis.poisson_away_prob:
                poisson_probs = [float(analysis.poisson_home_prob), float(analysis.poisson_draw_prob), float(analysis.poisson_away_prob)]
                poisson_pred = np.argmax(poisson_probs)

            if analysis.features:
                hist_odds = analysis.features.get("historical_odds", {})
                if hist_odds and hist_odds.get("implied_home_prob"):
                    market_probs = [hist_odds.get("implied_home_prob"), hist_odds.get("implied_draw_prob"), hist_odds.get("implied_away_prob")]
                    market_pred = np.argmax(market_probs)

            # Only count if we have all three models
            if elo_pred is not None and poisson_pred is not None and market_pred is not None:
                models_agree = (elo_pred == poisson_pred == market_pred)
                bucket = "agree" if models_agree else "disagree"
                by_agreement[bucket]["total"] += 1
                if is_correct:
                    by_agreement[bucket]["correct"] += 1

            # Store recent for chart
            if len(recent_results) < 100:
                recent_results.append({
                    "date": match.kickoff_time,
                    "predicted": predicted,
                    "actual": actual,
                    "correct": is_correct,
                    "confidence": confidence,
                })

        return {
            "total_accuracy": correct / total if total > 0 else 0,
            "total_matches": total,
            "correct": correct,
            "by_confidence": by_confidence,
            "by_agreement": by_agreement,
            "recent": recent_results,
        }


def train_consensus_model():
    """Train the consensus stacker model."""
    from batch.models.consensus_stacker import ConsensusStacker

    stacker = ConsensusStacker()
    result = stacker.train(epochs=100, batch_size=64)
    return result


def train_neural_stacker():
    """Train the neural stacker model."""
    from batch.models.neural_stacker import NeuralStacker

    stacker = NeuralStacker()
    result = stacker.train(epochs=50)
    return result


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

    # Historical accuracy
    st.subheader("Historical Prediction Accuracy")

    with st.spinner("Calculating historical accuracy..."):
        accuracy_data = get_historical_accuracy()

    if accuracy_data:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Overall Accuracy",
                f"{accuracy_data['total_accuracy']*100:.1f}%",
                help="Percentage of matches where predicted outcome matched actual"
            )

        with col2:
            st.metric("Total Predictions", f"{accuracy_data['total_matches']:,}")

        with col3:
            st.metric("Correct", f"{accuracy_data['correct']:,}")

        with col4:
            baseline = 100/3  # Random would be 33.3%
            improvement = accuracy_data['total_accuracy']*100 - baseline
            st.metric(
                "vs Random",
                f"+{improvement:.1f}pp",
                help="Improvement over random guessing (33.3%)"
            )

        # Accuracy by confidence
        st.subheader("Accuracy by Confidence Level")

        conf_data = accuracy_data["by_confidence"]
        conf_df = []
        for level, data in conf_data.items():
            if data["total"] > 0:
                acc = data["correct"] / data["total"]
                conf_df.append({
                    "Confidence": level.title(),
                    "Accuracy": acc,
                    "Matches": data["total"],
                })

        if conf_df:
            col1, col2, col3 = st.columns(3)

            for i, row in enumerate(conf_df):
                with [col1, col2, col3][i]:
                    st.metric(
                        f"{row['Confidence']} Confidence",
                        f"{row['Accuracy']*100:.1f}%",
                        help=f"Based on {row['Matches']} matches"
                    )

        # Accuracy by model agreement
        st.subheader("Accuracy by Model Agreement")

        agree_data = accuracy_data["by_agreement"]
        col1, col2 = st.columns(2)

        with col1:
            if agree_data["agree"]["total"] > 0:
                acc = agree_data["agree"]["correct"] / agree_data["agree"]["total"]
                st.metric(
                    "Models Agree",
                    f"{acc*100:.1f}%",
                    help=f"When ELO, Poisson, and Market all pick same favorite ({agree_data['agree']['total']} matches)"
                )
            else:
                st.metric("Models Agree", "N/A")

        with col2:
            if agree_data["disagree"]["total"] > 0:
                acc = agree_data["disagree"]["correct"] / agree_data["disagree"]["total"]
                st.metric(
                    "Models Disagree",
                    f"{acc*100:.1f}%",
                    help=f"When models pick different favorites ({agree_data['disagree']['total']} matches)"
                )
            else:
                st.metric("Models Disagree", "N/A")

        # Accuracy by matchweek chart
        st.subheader("Prediction Accuracy by Matchweek")

        # Query accuracy grouped by matchweek
        with SyncSessionLocal() as mw_session:
            from collections import defaultdict

            stmt = (
                select(Match, MatchAnalysis)
                .join(MatchAnalysis, Match.id == MatchAnalysis.match_id)
                .where(Match.status == MatchStatus.FINISHED)
                .where(Match.home_score.isnot(None))
                .where(MatchAnalysis.consensus_home_prob.isnot(None))
                .order_by(Match.season, Match.matchweek)
            )
            mw_results = mw_session.execute(stmt).all()

            # Group by season and matchweek - track both model and market odds
            mw_stats = defaultdict(lambda: {
                "model_correct": 0, "model_total": 0,
                "market_correct": 0, "market_total": 0
            })

            for match, analysis in mw_results:
                key = (match.season, match.matchweek)

                # Determine actual outcome
                if match.home_score > match.away_score:
                    actual = 0
                elif match.home_score < match.away_score:
                    actual = 2
                else:
                    actual = 1

                # Model prediction
                home_prob = float(analysis.consensus_home_prob)
                draw_prob = float(analysis.consensus_draw_prob)
                away_prob = float(analysis.consensus_away_prob)
                model_predicted = np.argmax([home_prob, draw_prob, away_prob])

                mw_stats[key]["model_total"] += 1
                if model_predicted == actual:
                    mw_stats[key]["model_correct"] += 1

                # Market odds prediction (if available)
                if analysis.features:
                    hist_odds = analysis.features.get("historical_odds", {})
                    if hist_odds and hist_odds.get("implied_home_prob"):
                        market_probs = [
                            hist_odds.get("implied_home_prob", 0),
                            hist_odds.get("implied_draw_prob", 0),
                            hist_odds.get("implied_away_prob", 0)
                        ]
                        market_predicted = np.argmax(market_probs)

                        mw_stats[key]["market_total"] += 1
                        if market_predicted == actual:
                            mw_stats[key]["market_correct"] += 1

            # Filter to complete matchweeks (10 matches) and recent seasons
            complete_mws = [
                (k, v) for k, v in sorted(mw_stats.items())
                if v["model_total"] >= 8  # Allow slightly incomplete weeks
            ]

            # Take last 40 matchweeks for display
            recent_mws = complete_mws[-40:]

            if recent_mws:
                x_labels = [f"{k[0][-5:]}\nMW{k[1]}" for k, v in recent_mws]
                model_values = [v["model_correct"] / v["model_total"] * 100 for k, v in recent_mws]
                market_values = [
                    v["market_correct"] / v["market_total"] * 100 if v["market_total"] > 0 else None
                    for k, v in recent_mws
                ]

                fig = go.Figure()

                # Model accuracy as bars
                fig.add_trace(go.Bar(
                    x=list(range(len(x_labels))),
                    y=model_values,
                    name="Model",
                    marker_color=['#E74C3C' if y < 33.3 else '#2E86AB' for y in model_values],
                    text=[f"{y:.0f}%" for y in model_values],
                    textposition='outside',
                ))

                # Market odds accuracy as line overlay
                valid_market = [(i, v) for i, v in enumerate(market_values) if v is not None]
                if valid_market:
                    fig.add_trace(go.Scatter(
                        x=[i for i, v in valid_market],
                        y=[v for i, v in valid_market],
                        name="Market Odds",
                        mode='lines+markers',
                        line=dict(color='#F18F01', width=3),
                        marker=dict(size=8),
                    ))

                fig.add_hline(y=33.3, line_dash="dash", line_color="gray",
                             annotation_text="Random (33.3%)")

                fig.update_layout(
                    title="Model vs Market Odds Accuracy by Matchweek",
                    xaxis_title="Season / Matchweek",
                    yaxis_title="Accuracy %",
                    yaxis=dict(range=[0, 100]),
                    xaxis=dict(
                        tickmode='array',
                        tickvals=list(range(len(x_labels))),
                        ticktext=x_labels,
                        tickangle=45,
                    ),
                    height=450,
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )

                st.plotly_chart(fig, use_container_width=True)

                st.caption("Blue/red bars = Model accuracy | Orange line = Market odds accuracy | Red bars = below random (33.3%)")
            else:
                st.info("Not enough completed matchweeks to display")
    else:
        st.info("No historical predictions found to analyze")

with tab2:
    st.header("Model Training")

    st.markdown("""
    ### Consensus Stacker
    The consensus stacker learns to combine predictions from multiple models (ELO, Poisson, Market odds)
    and boost confidence when models agree. It uses a neural network with agreement features.

    **Training data**: All finished matches with complete predictions
    """)

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

    # Training buttons
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Train Consensus Stacker")
        if st.button("🚀 Train Consensus Model", type="primary", key="train_consensus"):
            with st.spinner("Training consensus stacker (this may take a few minutes)..."):
                try:
                    result = train_consensus_model()
                    st.success(f"Training complete! Refreshing stats...")
                    st.json(result)
                    st.rerun()  # Refresh page to show updated stats
                except Exception as e:
                    st.error(f"Training failed: {e}")

    with col2:
        st.subheader("Train Neural Stacker")
        if st.button("🚀 Train Neural Model", type="primary", key="train_neural"):
            with st.spinner("Training neural stacker (this may take a few minutes)..."):
                try:
                    result = train_neural_stacker()
                    st.success(f"Training complete! Refreshing stats...")
                    st.json(result)
                    st.rerun()  # Refresh page to show updated stats
                except Exception as e:
                    st.error(f"Training failed: {e}")

    st.divider()

    st.markdown("""
    ### Training Notes
    - Models are trained using time-series cross-validation (last 20% for validation)
    - Training typically takes 1-3 minutes depending on data size
    - The best model (by validation accuracy) is automatically saved
    - After training, predictions will use the new model automatically
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
