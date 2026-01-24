"""Interactive Poisson model dashboard."""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from sqlalchemy import select

from app.db.database import SyncSessionLocal
from app.db.models import Match, MatchAnalysis, MatchStatus, Team
from batch.jobs.calculate_poisson import calculate_team_strengths_for_season


# Team colors for consistent styling
TEAM_COLORS = {
    "Liverpool": "#C8102E",
    "Man City": "#6CABDD",
    "Arsenal": "#EF0107",
    "Chelsea": "#034694",
    "Tottenham": "#132257",
    "Man United": "#DA291C",
    "Aston Villa": "#95BFE5",
    "Newcastle": "#241F20",
    "Brighton Hove": "#0057B8",
    "West Ham": "#7A263A",
    "Crystal Palace": "#1B458F",
    "Fulham": "#CC0000",
    "Brentford": "#E30613",
    "Nottingham": "#DD0000",
    "Bournemouth": "#B50E12",
    "Wolverhampton": "#FDB913",
    "Everton": "#003399",
    "Leicester": "#003090",
    "Southampton": "#D71920",
    "Ipswich": "#0033A0",
}


@st.cache_data(ttl=60)
def load_teams():
    """Load teams from database."""
    with SyncSessionLocal() as session:
        teams = list(session.execute(select(Team)).scalars().all())
        return {t.id: {"name": t.name, "short_name": t.short_name} for t in teams}


@st.cache_data(ttl=60)
def load_seasons():
    """Get available seasons."""
    with SyncSessionLocal() as session:
        stmt = select(Match.season).distinct().order_by(Match.season)
        return [s for (s,) in session.execute(stmt).all()]


@st.cache_data(ttl=60)
def load_matchweeks(season: str):
    """Get matchweeks with finished matches for a season."""
    with SyncSessionLocal() as session:
        stmt = (
            select(Match.matchweek)
            .where(Match.season == season)
            .where(Match.status == MatchStatus.FINISHED)
            .distinct()
            .order_by(Match.matchweek)
        )
        return [mw for (mw,) in session.execute(stmt).all()]


@st.cache_data(ttl=60)
def get_team_strengths(season: str, up_to_matchweek: int = None):
    """Get team strengths for a season."""
    return calculate_team_strengths_for_season(season, up_to_matchweek)


@st.cache_data(ttl=60)
def load_predictions(season: str):
    """Load Poisson predictions and results."""
    with SyncSessionLocal() as session:
        stmt = (
            select(Match, MatchAnalysis)
            .join(MatchAnalysis, Match.id == MatchAnalysis.match_id)
            .where(Match.season == season)
            .where(Match.status == MatchStatus.FINISHED)
            .where(MatchAnalysis.poisson_home_prob.isnot(None))
            .order_by(Match.matchweek)
        )
        results = list(session.execute(stmt).all())

        data = []
        for match, analysis in results:
            # Determine actual result
            if match.home_score > match.away_score:
                actual = "home"
            elif match.home_score < match.away_score:
                actual = "away"
            else:
                actual = "draw"

            # Determine predicted result
            probs = {
                "home": float(analysis.poisson_home_prob),
                "draw": float(analysis.poisson_draw_prob),
                "away": float(analysis.poisson_away_prob),
            }
            predicted = max(probs, key=probs.get)

            data.append({
                "matchweek": match.matchweek,
                "home_team_id": match.home_team_id,
                "away_team_id": match.away_team_id,
                "home_score": match.home_score,
                "away_score": match.away_score,
                "actual": actual,
                "predicted": predicted,
                "correct": actual == predicted,
                "home_prob": probs["home"],
                "draw_prob": probs["draw"],
                "away_prob": probs["away"],
                "predicted_home_goals": float(analysis.predicted_home_goals) if analysis.predicted_home_goals else None,
                "predicted_away_goals": float(analysis.predicted_away_goals) if analysis.predicted_away_goals else None,
            })

        return data


def get_team_color(short_name: str) -> str:
    """Get color for a team."""
    return TEAM_COLORS.get(short_name, "#888888")


def main():
    st.set_page_config(
        page_title="Poisson Model Dashboard",
        page_icon="",
        layout="wide",
    )

    st.title("Poisson Model Dashboard")
    st.markdown("Team attack/defense strengths and prediction analysis using the Poisson distribution model.")

    # Load data
    teams = load_teams()
    seasons = load_seasons()

    if not seasons:
        st.warning("No seasons found in database.")
        return

    # Sidebar
    st.sidebar.header("Settings")
    selected_season = st.sidebar.selectbox(
        "Season",
        seasons,
        index=len(seasons) - 1,
    )

    matchweeks = load_matchweeks(selected_season)
    if not matchweeks:
        st.warning(f"No finished matches for season {selected_season}")
        return

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Team Strengths", "Strength Progression", "Prediction Accuracy", "Model Comparison"])

    # ============ TAB 1: Team Strengths Scatter ============
    with tab1:
        st.subheader("Attack vs Defense Strength")

        # Matchweek selector
        col1, col2 = st.columns([1, 3])
        with col1:
            mw_option = st.radio(
                "Data up to",
                ["Current", "Select Matchweek"],
                key="strength_mw_option",
            )

        with col2:
            if mw_option == "Select Matchweek":
                selected_mw = st.slider(
                    "Matchweek",
                    min_value=min(matchweeks),
                    max_value=max(matchweeks),
                    value=max(matchweeks),
                    key="strength_mw_slider",
                )
            else:
                selected_mw = None

        # Get strengths
        strengths = get_team_strengths(selected_season, selected_mw)

        if not strengths:
            st.info("Not enough match data to calculate strengths yet.")
        else:
            # Build scatter plot
            fig = go.Figure()

            for team_id, stats in strengths.items():
                team_info = teams.get(team_id, {})
                team_name = team_info.get("short_name", f"Team {team_id}")
                color = get_team_color(team_name)

                fig.add_trace(go.Scatter(
                    x=[stats["attack"]],
                    y=[stats["defense"]],
                    mode='markers+text',
                    name=team_name,
                    text=[team_name],
                    textposition="top center",
                    marker=dict(size=15, color=color, line=dict(width=2, color='white')),
                    hovertemplate=(
                        f"<b>{team_name}</b><br>"
                        f"Attack: {stats['attack']:.3f}<br>"
                        f"Defense: {stats['defense']:.3f}<br>"
                        f"Games: {stats['games']}<br>"
                        "<extra></extra>"
                    ),
                    showlegend=False,
                ))

            # Reference lines
            fig.add_hline(y=1.0, line_dash="dash", line_color="gray", opacity=0.5)
            fig.add_vline(x=1.0, line_dash="dash", line_color="gray", opacity=0.5)

            # Quadrant annotations
            fig.add_annotation(x=1.3, y=0.7, text="Strong Attack<br>Strong Defense", showarrow=False, opacity=0.5)
            fig.add_annotation(x=0.7, y=0.7, text="Weak Attack<br>Strong Defense", showarrow=False, opacity=0.5)
            fig.add_annotation(x=1.3, y=1.3, text="Strong Attack<br>Weak Defense", showarrow=False, opacity=0.5)
            fig.add_annotation(x=0.7, y=1.3, text="Weak Attack<br>Weak Defense", showarrow=False, opacity=0.5)

            fig.update_layout(
                title=f"Team Strengths - {selected_season}" + (f" (up to MW{selected_mw})" if selected_mw else ""),
                xaxis_title="Attack Strength (>1 = above league avg)",
                yaxis_title="Defense Strength (>1 = concedes more)",
                height=600,
            )

            st.plotly_chart(fig, use_container_width=True)

            # Table
            st.subheader("Strength Rankings")
            table_data = []
            for team_id, stats in strengths.items():
                team_info = teams.get(team_id, {})
                table_data.append({
                    "Team": team_info.get("short_name", f"Team {team_id}"),
                    "Attack": round(stats["attack"], 3),
                    "Defense": round(stats["defense"], 3),
                    "Games": stats["games"],
                    "Net": round(stats["attack"] - stats["defense"], 3),
                })

            table_data.sort(key=lambda x: x["Net"], reverse=True)
            st.dataframe(table_data, use_container_width=True, hide_index=True)

    # ============ TAB 2: Strength Progression ============
    with tab2:
        st.subheader("How Team Strengths Evolve")

        if len(matchweeks) < 5:
            st.info("Need at least 5 matchweeks to show progression.")
        else:
            # Team selector
            team_names = sorted([teams[tid]["short_name"] for tid in strengths.keys()])

            # Default to top 6 by attack
            sorted_by_attack = sorted(strengths.items(), key=lambda x: x[1]["attack"], reverse=True)
            default_teams = [teams[t[0]]["short_name"] for t in sorted_by_attack[:6]]

            selected_teams = st.multiselect(
                "Select Teams",
                team_names,
                default=default_teams,
                key="progression_teams",
            )

            if selected_teams:
                # Calculate strengths at each checkpoint
                checkpoints = [mw for mw in matchweeks if mw >= 5]

                team_name_to_id = {teams[tid]["short_name"]: tid for tid in teams}

                attack_traces = []
                defense_traces = []

                for team_name in selected_teams:
                    team_id = team_name_to_id.get(team_name)
                    if not team_id:
                        continue

                    attack_data = []
                    defense_data = []

                    for mw in checkpoints:
                        mw_strengths = get_team_strengths(selected_season, mw + 1)
                        if team_id in mw_strengths:
                            attack_data.append((mw, mw_strengths[team_id]["attack"]))
                            defense_data.append((mw, mw_strengths[team_id]["defense"]))

                    if attack_data:
                        mws, attacks = zip(*attack_data)
                        attack_traces.append((team_name, mws, attacks))

                    if defense_data:
                        mws, defenses = zip(*defense_data)
                        defense_traces.append((team_name, mws, defenses))

                # Create subplots
                col1, col2 = st.columns(2)

                with col1:
                    fig_attack = go.Figure()
                    for team_name, mws, values in attack_traces:
                        fig_attack.add_trace(go.Scatter(
                            x=mws,
                            y=values,
                            mode='lines+markers',
                            name=team_name,
                            line=dict(color=get_team_color(team_name), width=2),
                            marker=dict(size=6),
                        ))
                    fig_attack.add_hline(y=1.0, line_dash="dash", line_color="gray", opacity=0.5)
                    fig_attack.update_layout(
                        title="Attack Strength Over Time",
                        xaxis_title="Matchweek",
                        yaxis_title="Attack Strength",
                        height=400,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    )
                    st.plotly_chart(fig_attack, use_container_width=True)

                with col2:
                    fig_defense = go.Figure()
                    for team_name, mws, values in defense_traces:
                        fig_defense.add_trace(go.Scatter(
                            x=mws,
                            y=values,
                            mode='lines+markers',
                            name=team_name,
                            line=dict(color=get_team_color(team_name), width=2),
                            marker=dict(size=6),
                        ))
                    fig_defense.add_hline(y=1.0, line_dash="dash", line_color="gray", opacity=0.5)
                    fig_defense.update_layout(
                        title="Defense Strength Over Time",
                        xaxis_title="Matchweek",
                        yaxis_title="Defense Strength (lower = better)",
                        height=400,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    )
                    st.plotly_chart(fig_defense, use_container_width=True)

    # ============ TAB 3: Prediction Accuracy ============
    with tab3:
        st.subheader("Model Performance")

        predictions = load_predictions(selected_season)

        if not predictions:
            st.info("No Poisson predictions found. Run `calculate_poisson.py --backfill` first.")
        else:
            # Calculate cumulative accuracy
            matchweek_stats = {}
            for p in predictions:
                mw = p["matchweek"]
                if mw not in matchweek_stats:
                    matchweek_stats[mw] = {"correct": 0, "total": 0}
                matchweek_stats[mw]["total"] += 1
                if p["correct"]:
                    matchweek_stats[mw]["correct"] += 1

            mws = sorted(matchweek_stats.keys())
            cumulative_correct = 0
            cumulative_total = 0
            accuracies = []

            for mw in mws:
                cumulative_correct += matchweek_stats[mw]["correct"]
                cumulative_total += matchweek_stats[mw]["total"]
                accuracies.append(cumulative_correct / cumulative_total * 100)

            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Predictions", cumulative_total)
            col2.metric("Correct", cumulative_correct)
            col3.metric("Accuracy", f"{accuracies[-1]:.1f}%")
            col4.metric("vs Random", f"+{accuracies[-1] - 33.3:.1f}%")

            # Accuracy chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=mws,
                y=accuracies,
                mode='lines+markers',
                name='Cumulative Accuracy',
                line=dict(color='#2E86AB', width=3),
                fill='tozeroy',
                fillcolor='rgba(46, 134, 171, 0.2)',
            ))
            fig.add_hline(y=33.3, line_dash="dash", line_color="red", opacity=0.7,
                          annotation_text="Random Guess (33.3%)")

            fig.update_layout(
                title="Cumulative Prediction Accuracy",
                xaxis_title="Matchweek",
                yaxis_title="Accuracy (%)",
                yaxis_range=[0, 70],
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Per-matchweek breakdown
            st.subheader("Per-Matchweek Breakdown")
            mw_data = []
            for mw in mws:
                stats = matchweek_stats[mw]
                mw_data.append({
                    "Matchweek": mw,
                    "Matches": stats["total"],
                    "Correct": stats["correct"],
                    "Accuracy": f"{stats['correct'] / stats['total'] * 100:.1f}%",
                })
            st.dataframe(mw_data, use_container_width=True, hide_index=True)

    # ============ TAB 4: Model Comparison ============
    with tab4:
        st.subheader("ELO vs Poisson vs XGBoost vs Consensus")

        # Load all predictions with model data
        with SyncSessionLocal() as session:
            stmt = (
                select(Match, MatchAnalysis)
                .join(MatchAnalysis, Match.id == MatchAnalysis.match_id)
                .where(Match.season == selected_season)
                .where(Match.status == MatchStatus.FINISHED)
                .where(MatchAnalysis.consensus_home_prob.isnot(None))
                .order_by(Match.matchweek)
            )
            results = list(session.execute(stmt).all())

        if not results:
            st.info("No consensus predictions found. Run `calculate_consensus.py --backfill` first.")
        else:
            # Calculate accuracy for each model
            model_stats = {
                "ELO": {"correct": 0, "total": 0},
                "Poisson": {"correct": 0, "total": 0},
                "XGBoost": {"correct": 0, "total": 0},
                "Consensus": {"correct": 0, "total": 0},
            }

            comparison_data = []
            for match, analysis in results:
                # Actual result
                if match.home_score > match.away_score:
                    actual = "home"
                elif match.home_score < match.away_score:
                    actual = "away"
                else:
                    actual = "draw"

                # Initialize predictions
                elo_pred = None
                poisson_pred = None
                xgboost_pred = None
                consensus_pred = None

                # ELO prediction
                if analysis.elo_home_prob:
                    elo_probs = {"home": float(analysis.elo_home_prob), "draw": float(analysis.elo_draw_prob), "away": float(analysis.elo_away_prob)}
                    elo_pred = max(elo_probs, key=elo_probs.get)
                    model_stats["ELO"]["total"] += 1
                    if elo_pred == actual:
                        model_stats["ELO"]["correct"] += 1

                # Poisson prediction
                if analysis.poisson_home_prob:
                    poisson_probs = {"home": float(analysis.poisson_home_prob), "draw": float(analysis.poisson_draw_prob), "away": float(analysis.poisson_away_prob)}
                    poisson_pred = max(poisson_probs, key=poisson_probs.get)
                    model_stats["Poisson"]["total"] += 1
                    if poisson_pred == actual:
                        model_stats["Poisson"]["correct"] += 1

                # XGBoost prediction
                if analysis.xgboost_home_prob:
                    xgboost_probs = {"home": float(analysis.xgboost_home_prob), "draw": float(analysis.xgboost_draw_prob), "away": float(analysis.xgboost_away_prob)}
                    xgboost_pred = max(xgboost_probs, key=xgboost_probs.get)
                    model_stats["XGBoost"]["total"] += 1
                    if xgboost_pred == actual:
                        model_stats["XGBoost"]["correct"] += 1

                # Consensus prediction
                if analysis.consensus_home_prob:
                    consensus_probs = {"home": float(analysis.consensus_home_prob), "draw": float(analysis.consensus_draw_prob), "away": float(analysis.consensus_away_prob)}
                    consensus_pred = max(consensus_probs, key=consensus_probs.get)
                    model_stats["Consensus"]["total"] += 1
                    if consensus_pred == actual:
                        model_stats["Consensus"]["correct"] += 1

                home_name = teams.get(match.home_team_id, {}).get("short_name", "?")
                away_name = teams.get(match.away_team_id, {}).get("short_name", "?")

                comparison_data.append({
                    "matchweek": match.matchweek,
                    "match": f"{home_name} vs {away_name}",
                    "result": f"{match.home_score}-{match.away_score}",
                    "actual": actual,
                    "elo_pred": elo_pred if elo_pred else "N/A",
                    "poisson_pred": poisson_pred if poisson_pred else "N/A",
                    "xgboost_pred": xgboost_pred if xgboost_pred else "N/A",
                    "consensus_pred": consensus_pred if consensus_pred else "N/A",
                })

            # Summary metrics
            st.markdown("### Model Accuracy Comparison")
            col1, col2, col3, col4 = st.columns(4)

            for col, (model, stats) in zip([col1, col2, col3, col4], model_stats.items()):
                if stats["total"] > 0:
                    acc = stats["correct"] / stats["total"] * 100
                    col.metric(
                        model,
                        f"{acc:.1f}%",
                        f"{stats['correct']}/{stats['total']} correct",
                    )

            # Bar chart comparison
            fig = go.Figure()
            models = []
            accuracies = []
            colors = ["#E94F37", "#2E86AB", "#F4A261", "#8AC926"]

            for model, stats in model_stats.items():
                if stats["total"] > 0:
                    models.append(model)
                    accuracies.append(stats["correct"] / stats["total"] * 100)

            fig.add_trace(go.Bar(
                x=models,
                y=accuracies,
                marker_color=colors[:len(models)],
                text=[f"{a:.1f}%" for a in accuracies],
                textposition='auto',
            ))
            fig.add_hline(y=33.3, line_dash="dash", line_color="gray",
                          annotation_text="Random (33.3%)")
            fig.update_layout(
                title="Model Accuracy Comparison",
                yaxis_title="Accuracy (%)",
                yaxis_range=[0, 70],
                height=400,
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Detailed predictions table
            st.markdown("### Recent Predictions")
            # Show last 20 matches
            recent = comparison_data[-20:][::-1]
            display_data = []
            for r in recent:
                display_data.append({
                    "MW": r["matchweek"],
                    "Match": r["match"],
                    "Result": r["result"],
                    "ELO": "✓" if r["elo_pred"] == r["actual"] else "✗",
                    "Poisson": "✓" if r["poisson_pred"] == r["actual"] else "✗",
                    "XGBoost": "✓" if r["xgboost_pred"] == r["actual"] else ("N/A" if r["xgboost_pred"] == "N/A" else "✗"),
                    "Consensus": "✓" if r["consensus_pred"] == r["actual"] else "✗",
                })
            st.dataframe(display_data, use_container_width=True, hide_index=True)

    # Info box
    with st.sidebar.expander("About Poisson Model"):
        st.markdown("""
        **Poisson Distribution Model**

        Predicts match outcomes based on:
        - **Attack Strength**: Goals scored vs league avg
        - **Defense Strength**: Goals conceded vs league avg
        - **Home Advantage**: +0.25 expected goals

        **Calculation**:
        ```
        Home xG = home_attack × away_defense × avg × 1.25
        Away xG = away_attack × home_defense × avg
        ```

        **Interpretation**:
        - Attack > 1.0 = scores more than average
        - Defense < 1.0 = concedes less than average
        - Defense > 1.0 = leaky defense
        """)


if __name__ == "__main__":
    main()
