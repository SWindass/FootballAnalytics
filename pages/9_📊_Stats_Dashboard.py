"""Comprehensive Model Performance Statistics Dashboard."""
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
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from decimal import Decimal
from sqlalchemy import select, func, text
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app.db.database import SyncSessionLocal
from app.db.models import Match, MatchAnalysis, MatchStatus, Team, ValueBet, BettingStrategy
from app.core.config import get_settings
from auth import require_auth, show_user_info
from pwa import inject_pwa_tags

settings = get_settings()

st.set_page_config(page_title="Stats Dashboard", page_icon="📊", layout="wide")

# PWA support
inject_pwa_tags()

# Auth check
require_auth(allowed_roles=["viewer", "admin"])
show_user_info()


# --- Custom CSS ---
st.markdown("""
<style>
.metric-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border-radius: 12px;
    padding: 20px;
    margin: 10px 0;
    border: 1px solid #0f3460;
}
.metric-value {
    font-size: 32px;
    font-weight: 700;
    color: #00ff88;
}
.metric-label {
    font-size: 14px;
    color: #888;
    margin-top: 5px;
}
.model-best {
    background-color: rgba(0, 255, 136, 0.2);
    color: #00ff88;
    padding: 2px 6px;
    border-radius: 4px;
}
.model-worst {
    background-color: rgba(255, 100, 100, 0.2);
    color: #ff6464;
    padding: 2px 6px;
    border-radius: 4px;
}
.status-live {
    color: #00ff88;
}
.status-active {
    color: #4da6ff;
}
.calibration-good {
    color: #00ff88;
}
.calibration-bad {
    color: #ff6464;
}
div[data-testid="stMetricValue"] {
    font-size: 28px;
}
</style>
""", unsafe_allow_html=True)


# --- Data Loading Functions ---

@st.cache_data(ttl=300)
def load_finished_matches_with_predictions():
    """Load all finished matches with their predictions."""
    with SyncSessionLocal() as session:
        query = text("""
            SELECT
                m.id, m.season, m.matchweek, m.kickoff_time,
                m.home_team_id, m.away_team_id,
                m.home_score, m.away_score,
                ht.name as home_team, ht.short_name as home_short,
                at.name as away_team, at.short_name as away_short,
                ma.elo_home_prob, ma.elo_draw_prob, ma.elo_away_prob,
                ma.poisson_home_prob, ma.poisson_draw_prob, ma.poisson_away_prob,
                ma.dixon_coles_home_prob, ma.dixon_coles_draw_prob, ma.dixon_coles_away_prob,
                ma.pi_rating_home_prob, ma.pi_rating_draw_prob, ma.pi_rating_away_prob,
                ma.xgboost_home_prob, ma.xgboost_draw_prob, ma.xgboost_away_prob,
                ma.consensus_home_prob, ma.consensus_draw_prob, ma.consensus_away_prob,
                ma.predicted_home_goals, ma.predicted_away_goals
            FROM matches m
            JOIN match_analyses ma ON m.id = ma.match_id
            JOIN teams ht ON m.home_team_id = ht.id
            JOIN teams at ON m.away_team_id = at.id
            WHERE m.status = 'finished'
            AND ma.consensus_home_prob IS NOT NULL
            ORDER BY m.kickoff_time DESC
        """)
        result = session.execute(query).fetchall()

        columns = [
            'id', 'season', 'matchweek', 'kickoff_time',
            'home_team_id', 'away_team_id', 'home_score', 'away_score',
            'home_team', 'home_short', 'away_team', 'away_short',
            'elo_home', 'elo_draw', 'elo_away',
            'poisson_home', 'poisson_draw', 'poisson_away',
            'dixon_home', 'dixon_draw', 'dixon_away',
            'pi_home', 'pi_draw', 'pi_away',
            'xgb_home', 'xgb_draw', 'xgb_away',
            'consensus_home', 'consensus_draw', 'consensus_away',
            'pred_home_goals', 'pred_away_goals'
        ]

        df = pd.DataFrame(result, columns=columns)

        # Convert Decimals to floats
        for col in df.columns:
            if df[col].dtype == object:
                try:
                    df[col] = df[col].apply(lambda x: float(x) if isinstance(x, Decimal) else x)
                except:
                    pass

        # Add actual result column
        def get_actual_result(row):
            if row['home_score'] > row['away_score']:
                return 'home'
            elif row['home_score'] < row['away_score']:
                return 'away'
            else:
                return 'draw'

        df['actual_result'] = df.apply(get_actual_result, axis=1)

        return df


@st.cache_data(ttl=300)
def load_value_bets_performance():
    """Load value bet performance data."""
    with SyncSessionLocal() as session:
        query = text("""
            SELECT
                vb.id, vb.match_id, vb.outcome, vb.bookmaker,
                vb.model_probability, vb.implied_probability, vb.edge,
                vb.odds, vb.result, vb.profit_loss, vb.created_at,
                bs.name as strategy_name,
                m.season, m.kickoff_time,
                ht.short_name as home_team, at.short_name as away_team
            FROM value_bets vb
            LEFT JOIN betting_strategies bs ON vb.strategy_id = bs.id
            JOIN matches m ON vb.match_id = m.id
            JOIN teams ht ON m.home_team_id = ht.id
            JOIN teams at ON m.away_team_id = at.id
            WHERE vb.result IS NOT NULL
            ORDER BY m.kickoff_time DESC
        """)
        result = session.execute(query).fetchall()

        columns = [
            'id', 'match_id', 'outcome', 'bookmaker',
            'model_prob', 'implied_prob', 'edge',
            'odds', 'result', 'profit_loss', 'created_at',
            'strategy_name', 'season', 'kickoff_time',
            'home_team', 'away_team'
        ]

        df = pd.DataFrame(result, columns=columns)

        for col in ['model_prob', 'implied_prob', 'edge', 'odds', 'profit_loss']:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: float(x) if isinstance(x, Decimal) else x)

        return df


@st.cache_data(ttl=300)
def load_strategies():
    """Load betting strategy performance."""
    with SyncSessionLocal() as session:
        query = text("""
            SELECT
                id, name, outcome_type, status,
                total_bets, total_wins, total_profit,
                historical_roi, rolling_50_roi,
                consecutive_losing_streak
            FROM betting_strategies
            ORDER BY historical_roi DESC NULLS LAST
        """)
        result = session.execute(query).fetchall()

        columns = [
            'id', 'name', 'outcome_type', 'status',
            'total_bets', 'total_wins', 'total_profit',
            'historical_roi', 'rolling_50_roi',
            'consecutive_losing_streak'
        ]

        df = pd.DataFrame(result, columns=columns)

        for col in ['total_profit', 'historical_roi', 'rolling_50_roi']:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: float(x) if isinstance(x, Decimal) else x)

        return df


# --- Metric Calculation Functions ---

def calculate_model_metrics(df, model_prefix, season_filter=None):
    """Calculate accuracy and other metrics for a model."""
    if season_filter:
        df = df[df['season'] == season_filter]

    if len(df) == 0:
        return None

    home_col = f'{model_prefix}_home'
    draw_col = f'{model_prefix}_draw'
    away_col = f'{model_prefix}_away'

    # Check if columns exist and have data
    if home_col not in df.columns or df[home_col].isna().all():
        return None

    # Get predicted result (highest probability)
    def get_predicted(row):
        probs = {
            'home': row[home_col] if pd.notna(row[home_col]) else 0,
            'draw': row[draw_col] if pd.notna(row[draw_col]) else 0,
            'away': row[away_col] if pd.notna(row[away_col]) else 0
        }
        return max(probs, key=probs.get)

    df = df.copy()
    df['predicted'] = df.apply(get_predicted, axis=1)

    # Overall accuracy
    correct = (df['predicted'] == df['actual_result']).sum()
    total = len(df)
    accuracy = correct / total if total > 0 else 0

    # Accuracy by outcome
    home_matches = df[df['actual_result'] == 'home']
    draw_matches = df[df['actual_result'] == 'draw']
    away_matches = df[df['actual_result'] == 'away']

    home_acc = (home_matches['predicted'] == 'home').mean() if len(home_matches) > 0 else 0
    draw_acc = (draw_matches['predicted'] == 'draw').mean() if len(draw_matches) > 0 else 0
    away_acc = (away_matches['predicted'] == 'away').mean() if len(away_matches) > 0 else 0

    # Brier score (lower is better)
    brier_scores = []
    for _, row in df.iterrows():
        actual_home = 1 if row['actual_result'] == 'home' else 0
        actual_draw = 1 if row['actual_result'] == 'draw' else 0
        actual_away = 1 if row['actual_result'] == 'away' else 0

        pred_home = row[home_col] if pd.notna(row[home_col]) else 0.33
        pred_draw = row[draw_col] if pd.notna(row[draw_col]) else 0.33
        pred_away = row[away_col] if pd.notna(row[away_col]) else 0.33

        brier = ((pred_home - actual_home) ** 2 +
                 (pred_draw - actual_draw) ** 2 +
                 (pred_away - actual_away) ** 2)
        brier_scores.append(brier)

    brier_score = np.mean(brier_scores) if brier_scores else 1.0

    # Log loss
    log_losses = []
    eps = 1e-15
    for _, row in df.iterrows():
        pred_home = max(min(row[home_col] if pd.notna(row[home_col]) else 0.33, 1-eps), eps)
        pred_draw = max(min(row[draw_col] if pd.notna(row[draw_col]) else 0.33, 1-eps), eps)
        pred_away = max(min(row[away_col] if pd.notna(row[away_col]) else 0.33, 1-eps), eps)

        if row['actual_result'] == 'home':
            log_losses.append(-np.log(pred_home))
        elif row['actual_result'] == 'draw':
            log_losses.append(-np.log(pred_draw))
        else:
            log_losses.append(-np.log(pred_away))

    log_loss = np.mean(log_losses) if log_losses else 2.0

    return {
        'accuracy': accuracy,
        'total': total,
        'correct': correct,
        'home_accuracy': home_acc,
        'draw_accuracy': draw_acc,
        'away_accuracy': away_acc,
        'brier_score': brier_score,
        'log_loss': log_loss,
    }


def calculate_confusion_matrix(df, model_prefix='consensus'):
    """Calculate confusion matrix for predictions."""
    home_col = f'{model_prefix}_home'
    draw_col = f'{model_prefix}_draw'
    away_col = f'{model_prefix}_away'

    if home_col not in df.columns or df[home_col].isna().all():
        return None

    df = df.copy()

    def get_predicted(row):
        probs = {
            'home': row[home_col] if pd.notna(row[home_col]) else 0,
            'draw': row[draw_col] if pd.notna(row[draw_col]) else 0,
            'away': row[away_col] if pd.notna(row[away_col]) else 0
        }
        return max(probs, key=probs.get)

    df['predicted'] = df.apply(get_predicted, axis=1)

    # Create confusion matrix
    matrix = np.zeros((3, 3), dtype=int)
    labels = ['home', 'draw', 'away']

    for _, row in df.iterrows():
        actual_idx = labels.index(row['actual_result'])
        pred_idx = labels.index(row['predicted'])
        matrix[actual_idx][pred_idx] += 1

    return matrix, labels


def calculate_calibration(df, model_prefix, outcome, bins=10):
    """Calculate calibration data for a specific outcome."""
    prob_col = f'{model_prefix}_{outcome}'

    if prob_col not in df.columns or df[prob_col].isna().all():
        return None

    df = df.copy()
    df = df[df[prob_col].notna()]

    if len(df) == 0:
        return None

    # Create bins
    df['prob_bin'] = pd.cut(df[prob_col], bins=bins, labels=False)
    df['actual'] = (df['actual_result'] == outcome).astype(int)

    calibration_data = []
    for bin_idx in range(bins):
        bin_df = df[df['prob_bin'] == bin_idx]
        if len(bin_df) >= 5:  # Only include bins with enough data
            mean_pred = bin_df[prob_col].mean()
            mean_actual = bin_df['actual'].mean()
            count = len(bin_df)
            calibration_data.append({
                'predicted': mean_pred,
                'actual': mean_actual,
                'count': count
            })

    return pd.DataFrame(calibration_data)


# --- Load Data ---

df = load_finished_matches_with_predictions()
value_bets_df = load_value_bets_performance()
strategies_df = load_strategies()

# Get available seasons
seasons = sorted(df['season'].unique().tolist(), reverse=True)
current_season = settings.current_season


# --- Sidebar Filters ---

with st.sidebar:
    st.subheader("Filters")

    filter_options = ["All Time", "Current Season", "Last 100", "Last 50"]
    date_filter = st.selectbox("Time Period", filter_options, index=1)

    if date_filter == "Current Season":
        filtered_df = df[df['season'] == current_season]
    elif date_filter == "Last 100":
        filtered_df = df.head(100)
    elif date_filter == "Last 50":
        filtered_df = df.head(50)
    else:
        filtered_df = df

    st.divider()

    model_options = ["consensus", "elo", "poisson", "dixon", "pi", "xgb"]
    model_names = {
        "consensus": "Ensemble",
        "elo": "ELO Rating",
        "poisson": "Poisson",
        "dixon": "Dixon-Coles",
        "pi": "Pi Rating",
        "xgb": "XGBoost"
    }

    selected_model = st.selectbox(
        "Primary Model",
        model_options,
        format_func=lambda x: model_names.get(x, x)
    )

    st.divider()
    st.caption(f"Data: {len(filtered_df)} matches")
    st.caption(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")


# --- Main Dashboard ---

st.title("📊 Model Performance Dashboard")

# --- PART 1: Overall Performance Summary ---

st.subheader("Performance Summary")

# Calculate metrics for all models
model_metrics = {}
for model in model_options:
    metrics = calculate_model_metrics(filtered_df, model)
    if metrics:
        model_metrics[model] = metrics

# Main metrics
if 'consensus' in model_metrics:
    metrics = model_metrics['consensus']

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Overall Accuracy",
            f"{metrics['accuracy']:.1%}",
            help="Percentage of correct predictions"
        )

    with col2:
        st.metric(
            "Brier Score",
            f"{metrics['brier_score']:.4f}",
            help="Lower is better. Perfect = 0, Random = 0.67"
        )

    with col3:
        st.metric(
            "Log Loss",
            f"{metrics['log_loss']:.4f}",
            help="Lower is better. Measures prediction confidence"
        )

    with col4:
        st.metric(
            "Predictions",
            f"{metrics['total']:,}",
            help="Total finished matches analyzed"
        )

    # Accuracy by outcome
    st.markdown("##### Accuracy by Outcome")
    col1, col2, col3 = st.columns(3)

    with col1:
        acc = metrics['home_accuracy']
        st.metric("Home Win", f"{acc:.1%}")
        st.progress(acc)

    with col2:
        acc = metrics['draw_accuracy']
        st.metric("Draw", f"{acc:.1%}")
        st.progress(acc)

    with col3:
        acc = metrics['away_accuracy']
        st.metric("Away Win", f"{acc:.1%}")
        st.progress(acc)


# --- PART 2: Model Comparison Table ---

st.divider()
st.subheader("Model Comparison")

comparison_data = []
for model, metrics in model_metrics.items():
    comparison_data.append({
        'Model': model_names.get(model, model),
        'Accuracy': metrics['accuracy'],
        'Brier Score': metrics['brier_score'],
        'Log Loss': metrics['log_loss'],
        'Home Acc': metrics['home_accuracy'],
        'Draw Acc': metrics['draw_accuracy'],
        'Away Acc': metrics['away_accuracy'],
        'Matches': metrics['total']
    })

if comparison_data:
    comp_df = pd.DataFrame(comparison_data)

    # Find best values
    best_acc = comp_df['Accuracy'].max()
    best_brier = comp_df['Brier Score'].min()
    best_logloss = comp_df['Log Loss'].min()

    # Style the dataframe
    def highlight_best(val, col):
        if col == 'Accuracy' and val == best_acc:
            return 'background-color: rgba(0, 255, 136, 0.3)'
        elif col == 'Brier Score' and val == best_brier:
            return 'background-color: rgba(0, 255, 136, 0.3)'
        elif col == 'Log Loss' and val == best_logloss:
            return 'background-color: rgba(0, 255, 136, 0.3)'
        return ''

    # Format display
    comp_df_display = comp_df.copy()
    comp_df_display['Accuracy'] = comp_df_display['Accuracy'].apply(lambda x: f"{x:.1%}")
    comp_df_display['Brier Score'] = comp_df_display['Brier Score'].apply(lambda x: f"{x:.4f}")
    comp_df_display['Log Loss'] = comp_df_display['Log Loss'].apply(lambda x: f"{x:.4f}")
    comp_df_display['Home Acc'] = comp_df_display['Home Acc'].apply(lambda x: f"{x:.1%}")
    comp_df_display['Draw Acc'] = comp_df_display['Draw Acc'].apply(lambda x: f"{x:.1%}")
    comp_df_display['Away Acc'] = comp_df_display['Away Acc'].apply(lambda x: f"{x:.1%}")

    st.dataframe(comp_df_display, use_container_width=True, hide_index=True)


# --- PART 3: Performance Over Time ---

st.divider()
st.subheader("Performance Over Time")

# Calculate accuracy by season
season_performance = []
for season in seasons:
    season_df = df[df['season'] == season]
    for model in ['consensus', 'elo', 'poisson', 'dixon', 'pi', 'xgb']:
        metrics = calculate_model_metrics(season_df, model)
        if metrics and metrics['total'] >= 20:
            season_performance.append({
                'Season': season,
                'Model': model_names.get(model, model),
                'Accuracy': metrics['accuracy'],
                'Brier': metrics['brier_score'],
                'Matches': metrics['total']
            })

if season_performance:
    perf_df = pd.DataFrame(season_performance)

    # Line chart
    fig = px.line(
        perf_df,
        x='Season',
        y='Accuracy',
        color='Model',
        markers=True,
        title='Model Accuracy by Season'
    )

    fig.add_hline(y=0.6, line_dash="dash", line_color="gray",
                  annotation_text="60% threshold")

    fig.update_layout(
        yaxis_tickformat='.0%',
        yaxis_range=[0.4, 0.75],
        template='plotly_dark',
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)


# --- PART 4: Confusion Matrix ---

st.divider()
st.subheader("Prediction Breakdown")

cm_result = calculate_confusion_matrix(filtered_df, selected_model)

if cm_result:
    matrix, labels = cm_result

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=['Pred: Home', 'Pred: Draw', 'Pred: Away'],
        y=['Actual: Home', 'Actual: Draw', 'Actual: Away'],
        text=matrix,
        texttemplate="%{text}",
        textfont={"size": 16},
        colorscale='Greens',
        showscale=False
    ))

    fig.update_layout(
        title=f'Confusion Matrix - {model_names.get(selected_model, selected_model)}',
        xaxis_title='Predicted',
        yaxis_title='Actual',
        template='plotly_dark',
        height=400
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("##### Matrix Analysis")

        # Calculate metrics from matrix
        total = matrix.sum()
        correct = np.trace(matrix)

        st.markdown(f"**Total Predictions:** {total:,}")
        st.markdown(f"**Correct:** {correct:,} ({correct/total:.1%})")

        # Per-class metrics
        st.markdown("---")
        for i, label in enumerate(labels):
            class_correct = matrix[i][i]
            class_total = matrix[i].sum()
            if class_total > 0:
                st.markdown(f"**{label.title()}:** {class_correct}/{class_total} ({class_correct/class_total:.1%})")


# --- PART 5: Calibration Analysis ---

st.divider()
st.subheader("Calibration Analysis")

tab1, tab2, tab3 = st.tabs(["Home Win", "Draw", "Away Win"])

for tab, outcome in [(tab1, 'home'), (tab2, 'draw'), (tab3, 'away')]:
    with tab:
        calib_df = calculate_calibration(filtered_df, selected_model, outcome, bins=10)

        if calib_df is not None and len(calib_df) > 0:
            fig = go.Figure()

            # Perfect calibration line
            fig.add_trace(go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                name='Perfect Calibration',
                line=dict(dash='dash', color='gray')
            ))

            # Actual calibration
            fig.add_trace(go.Scatter(
                x=calib_df['predicted'],
                y=calib_df['actual'],
                mode='lines+markers',
                name='Actual',
                marker=dict(size=calib_df['count'] / calib_df['count'].max() * 20 + 5),
                text=calib_df['count'].apply(lambda x: f'n={x}'),
                hovertemplate='Predicted: %{x:.1%}<br>Actual: %{y:.1%}<br>%{text}'
            ))

            fig.update_layout(
                title=f'{outcome.title()} Probability Calibration',
                xaxis_title='Predicted Probability',
                yaxis_title='Actual Frequency',
                xaxis_tickformat='.0%',
                yaxis_tickformat='.0%',
                template='plotly_dark',
                height=350,
                showlegend=True
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"Not enough data for {outcome} calibration analysis.")


# --- PART 6: Edge Analysis (Betting) ---

st.divider()
st.subheader("Value Bet Analysis")

if len(value_bets_df) > 0:
    col1, col2 = st.columns(2)

    with col1:
        # Edge distribution
        fig = px.histogram(
            value_bets_df,
            x='edge',
            nbins=20,
            title='Edge Distribution',
            color_discrete_sequence=['#00ff88']
        )
        fig.update_layout(
            xaxis_title='Edge',
            yaxis_title='Count',
            xaxis_tickformat='.0%',
            template='plotly_dark',
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Win rate by edge bucket
        value_bets_df['edge_bucket'] = pd.cut(
            value_bets_df['edge'],
            bins=[-1, 0, 0.05, 0.10, 0.15, 1],
            labels=['<0%', '0-5%', '5-10%', '10-15%', '>15%']
        )

        bucket_stats = value_bets_df.groupby('edge_bucket', observed=True).agg({
            'id': 'count',
            'result': lambda x: (x == 'won').sum() / len(x) if len(x) > 0 else 0,
            'profit_loss': 'sum'
        }).reset_index()
        bucket_stats.columns = ['Edge Bucket', 'Count', 'Win Rate', 'Profit']

        fig = px.bar(
            bucket_stats,
            x='Edge Bucket',
            y='Win Rate',
            text=bucket_stats['Win Rate'].apply(lambda x: f'{x:.0%}'),
            title='Win Rate by Edge',
            color_discrete_sequence=['#4da6ff']
        )
        fig.update_layout(
            yaxis_tickformat='.0%',
            template='plotly_dark',
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)

    # Summary stats
    st.markdown("##### Betting Performance Summary")

    total_bets = len(value_bets_df)
    won_bets = (value_bets_df['result'] == 'won').sum()
    total_profit = value_bets_df['profit_loss'].sum()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Bets", f"{total_bets:,}")
    with col2:
        st.metric("Win Rate", f"{won_bets/total_bets:.1%}" if total_bets > 0 else "N/A")
    with col3:
        st.metric("Total Profit", f"£{total_profit:,.2f}" if pd.notna(total_profit) else "N/A")
    with col4:
        avg_edge = value_bets_df['edge'].mean()
        st.metric("Avg Edge", f"{avg_edge:.1%}" if pd.notna(avg_edge) else "N/A")

    # By outcome
    st.markdown("##### Performance by Outcome")

    outcome_stats = value_bets_df.groupby('outcome').agg({
        'id': 'count',
        'result': lambda x: (x == 'won').sum(),
        'profit_loss': 'sum',
        'edge': 'mean',
        'odds': 'mean'
    }).reset_index()
    outcome_stats.columns = ['Outcome', 'Bets', 'Wins', 'Profit', 'Avg Edge', 'Avg Odds']
    outcome_stats['Win Rate'] = outcome_stats['Wins'] / outcome_stats['Bets']

    outcome_display = outcome_stats.copy()
    outcome_display['Win Rate'] = outcome_display['Win Rate'].apply(lambda x: f"{x:.1%}")
    outcome_display['Avg Edge'] = outcome_display['Avg Edge'].apply(lambda x: f"{x:.1%}")
    outcome_display['Avg Odds'] = outcome_display['Avg Odds'].apply(lambda x: f"{x:.2f}")
    outcome_display['Profit'] = outcome_display['Profit'].apply(lambda x: f"£{x:,.2f}")

    st.dataframe(outcome_display[['Outcome', 'Bets', 'Win Rate', 'Avg Edge', 'Avg Odds', 'Profit']],
                 use_container_width=True, hide_index=True)

else:
    st.info("No resolved value bets available yet.")


# --- PART 7: Recent Form ---

st.divider()
st.subheader("Recent Predictions")

recent_df = filtered_df.head(20).copy()

if len(recent_df) > 0:
    # Get predicted result
    def get_prediction_info(row):
        probs = {
            'home': (row['consensus_home'], 'H'),
            'draw': (row['consensus_draw'], 'D'),
            'away': (row['consensus_away'], 'A')
        }
        max_key = max(probs, key=lambda k: probs[k][0] if pd.notna(probs[k][0]) else 0)
        return probs[max_key][1], probs[max_key][0]

    recent_df['pred'], recent_df['conf'] = zip(*recent_df.apply(get_prediction_info, axis=1))
    recent_df['actual'] = recent_df['actual_result'].map({'home': 'H', 'draw': 'D', 'away': 'A'})
    recent_df['correct'] = recent_df['pred'] == recent_df['actual']

    # Display table
    display_df = recent_df[['kickoff_time', 'home_short', 'away_short', 'home_score', 'away_score',
                            'pred', 'conf', 'actual', 'correct']].copy()
    display_df.columns = ['Date', 'Home', 'Away', 'H', 'A', 'Pred', 'Conf', 'Actual', 'Correct']
    display_df['Date'] = pd.to_datetime(display_df['Date']).dt.strftime('%Y-%m-%d')
    display_df['Conf'] = display_df['Conf'].apply(lambda x: f"{x:.0%}" if pd.notna(x) else "-")
    display_df['Score'] = display_df['H'].astype(str) + '-' + display_df['A'].astype(str)
    display_df['Correct'] = display_df['Correct'].map({True: '✓', False: '✗'})

    st.dataframe(
        display_df[['Date', 'Home', 'Away', 'Score', 'Pred', 'Conf', 'Actual', 'Correct']],
        use_container_width=True,
        hide_index=True
    )

    # Running stats
    col1, col2, col3, col4 = st.columns(4)

    correct_count = recent_df['correct'].sum()
    total_count = len(recent_df)

    with col1:
        st.metric(f"L{total_count} Accuracy", f"{correct_count}/{total_count} ({correct_count/total_count:.0%})")

    # High confidence predictions
    high_conf = recent_df[recent_df['conf'] >= 0.55]
    high_correct = high_conf['correct'].sum() if len(high_conf) > 0 else 0

    with col2:
        if len(high_conf) > 0:
            st.metric("High Conf (>55%)", f"{high_correct}/{len(high_conf)} ({high_correct/len(high_conf):.0%})")
        else:
            st.metric("High Conf (>55%)", "N/A")

    # Low confidence
    low_conf = recent_df[recent_df['conf'] < 0.45]
    low_correct = low_conf['correct'].sum() if len(low_conf) > 0 else 0

    with col3:
        if len(low_conf) > 0:
            st.metric("Low Conf (<45%)", f"{low_correct}/{len(low_conf)} ({low_correct/len(low_conf):.0%})")
        else:
            st.metric("Low Conf (<45%)", "N/A")

    # Draws
    draw_preds = recent_df[recent_df['pred'] == 'D']
    draw_correct = (draw_preds['actual'] == 'D').sum() if len(draw_preds) > 0 else 0

    with col4:
        if len(draw_preds) > 0:
            st.metric("Draw Preds", f"{draw_correct}/{len(draw_preds)} ({draw_correct/len(draw_preds):.0%})")
        else:
            st.metric("Draw Preds", "N/A")


# --- PART 8: Strategy Performance ---

st.divider()
st.subheader("Betting Strategy Performance")

if len(strategies_df) > 0:
    strat_display = strategies_df.copy()
    strat_display['Win Rate'] = (strat_display['total_wins'] / strat_display['total_bets']).fillna(0)
    strat_display['ROI'] = strat_display['historical_roi'].fillna(0)

    strat_display['Win Rate'] = strat_display['Win Rate'].apply(lambda x: f"{x:.1%}")
    strat_display['ROI'] = strat_display['ROI'].apply(lambda x: f"{x:.1%}")
    strat_display['total_profit'] = strat_display['total_profit'].apply(lambda x: f"£{x:,.2f}" if pd.notna(x) else "£0.00")

    # Status icons
    strat_display['Status'] = strat_display['status'].map({
        'active': '✓ Active',
        'paused': '⏸ Paused',
        'disabled': '✗ Disabled'
    })

    st.dataframe(
        strat_display[['name', 'outcome_type', 'total_bets', 'Win Rate', 'ROI', 'total_profit', 'Status']].rename(columns={
            'name': 'Strategy',
            'outcome_type': 'Type',
            'total_bets': 'Bets',
            'total_profit': 'Profit'
        }),
        use_container_width=True,
        hide_index=True
    )
else:
    st.info("No betting strategies configured yet.")


# --- PART 9: Bankroll Simulation ---

st.divider()
st.subheader("Bankroll Simulation")

if len(value_bets_df) > 0:
    # Simulate bankroll growth
    starting_bankroll = 1000
    stake_pct = 0.02  # 2% of bankroll

    sorted_bets = value_bets_df.sort_values('kickoff_time')
    bankroll_history = [starting_bankroll]
    dates = [sorted_bets['kickoff_time'].iloc[0] - timedelta(days=1)]

    current_bankroll = starting_bankroll
    for _, bet in sorted_bets.iterrows():
        stake = current_bankroll * stake_pct
        if bet['result'] == 'won':
            profit = stake * (float(bet['odds']) - 1)
        else:
            profit = -stake
        current_bankroll += profit
        bankroll_history.append(current_bankroll)
        dates.append(bet['kickoff_time'])

    sim_df = pd.DataFrame({
        'Date': dates,
        'Bankroll': bankroll_history
    })

    col1, col2 = st.columns([2, 1])

    with col1:
        fig = px.line(
            sim_df,
            x='Date',
            y='Bankroll',
            title='Bankroll Growth (2% Stakes)',
        )
        fig.add_hline(y=starting_bankroll, line_dash="dash", line_color="gray")
        fig.update_layout(
            yaxis_title='Bankroll (£)',
            template='plotly_dark',
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("##### Simulation Stats")

        final_bankroll = bankroll_history[-1]
        profit = final_bankroll - starting_bankroll
        roi_pct = (profit / starting_bankroll) * 100

        st.metric("Starting", f"£{starting_bankroll:,.0f}")
        st.metric("Current", f"£{final_bankroll:,.0f}")
        st.metric("Profit", f"£{profit:,.0f}", delta=f"{roi_pct:+.1f}%")
        st.metric("Bets Placed", f"{len(sorted_bets):,}")

        # Streaks
        results = sorted_bets['result'].tolist()
        max_win_streak = 0
        max_lose_streak = 0
        current_streak = 0
        last_result = None

        for r in results:
            if r == last_result:
                current_streak += 1
            else:
                current_streak = 1

            if r == 'won':
                max_win_streak = max(max_win_streak, current_streak)
            else:
                max_lose_streak = max(max_lose_streak, current_streak)

            last_result = r

        st.markdown(f"**Win Streak:** {max_win_streak}")
        st.markdown(f"**Lose Streak:** {max_lose_streak}")

else:
    st.info("No betting data available for simulation.")


# --- Footer ---

st.divider()
st.caption(f"Dashboard refreshes every 5 minutes | Data from {len(df):,} total matches | Season {current_season}")
