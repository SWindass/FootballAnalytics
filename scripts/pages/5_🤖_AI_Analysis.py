"""AI Match Analysis - View AI-generated match previews."""

import streamlit as st
from datetime import datetime
from sqlalchemy import select

from app.db.database import SyncSessionLocal
from app.db.models import Match, MatchAnalysis, MatchStatus, Team

st.set_page_config(page_title="AI Analysis", page_icon="🤖", layout="wide")


@st.cache_data(ttl=300)
def load_matches_with_narratives():
    """Load matches that have AI narratives."""
    with SyncSessionLocal() as session:
        teams = {t.id: t for t in session.execute(select(Team)).scalars().all()}

        stmt = (
            select(Match, MatchAnalysis)
            .join(MatchAnalysis, Match.id == MatchAnalysis.match_id)
            .where(MatchAnalysis.narrative.isnot(None))
            .order_by(Match.kickoff_time.desc())
            .limit(50)
        )
        results = list(session.execute(stmt).all())

        matches = []
        for match, analysis in results:
            home = teams.get(match.home_team_id)
            away = teams.get(match.away_team_id)

            matches.append({
                "id": match.id,
                "home_team": home.short_name if home else "?",
                "away_team": away.short_name if away else "?",
                "home_crest": home.crest_url if home else None,
                "away_crest": away.crest_url if away else None,
                "kickoff": match.kickoff_time,
                "venue": home.venue if home else "TBC",
                "status": match.status.value if hasattr(match.status, 'value') else match.status,
                "score": f"{match.home_score}-{match.away_score}" if match.home_score is not None else None,
                "matchweek": match.matchweek,
                "season": match.season,
                "narrative": analysis.narrative,
                "narrative_generated_at": analysis.narrative_generated_at,
                "home_prob": float(analysis.consensus_home_prob) if analysis.consensus_home_prob else None,
                "draw_prob": float(analysis.consensus_draw_prob) if analysis.consensus_draw_prob else None,
                "away_prob": float(analysis.consensus_away_prob) if analysis.consensus_away_prob else None,
                "pred_home": float(analysis.predicted_home_goals) if analysis.predicted_home_goals else None,
                "pred_away": float(analysis.predicted_away_goals) if analysis.predicted_away_goals else None,
            })

        return matches


@st.cache_data(ttl=300)
def load_upcoming_without_narratives():
    """Load upcoming matches without narratives."""
    with SyncSessionLocal() as session:
        teams = {t.id: t for t in session.execute(select(Team)).scalars().all()}

        # Upcoming matches with analysis but no narrative
        stmt = (
            select(Match, MatchAnalysis)
            .join(MatchAnalysis, Match.id == MatchAnalysis.match_id)
            .where(Match.status == MatchStatus.SCHEDULED)
            .where(MatchAnalysis.narrative.is_(None))
            .order_by(Match.kickoff_time)
            .limit(20)
        )
        results = list(session.execute(stmt).all())

        matches = []
        for match, analysis in results:
            home = teams.get(match.home_team_id)
            away = teams.get(match.away_team_id)
            matches.append({
                "id": match.id,
                "home_team": home.short_name if home else "?",
                "away_team": away.short_name if away else "?",
                "kickoff": match.kickoff_time,
                "matchweek": match.matchweek,
            })

        return matches


# Main content
st.title("🤖 AI Match Analysis")
st.markdown("AI-generated match previews with form analysis, tactical insights, and betting angles.")

matches = load_matches_with_narratives()
pending = load_upcoming_without_narratives()

if not matches:
    st.info("No AI narratives generated yet. Run the narrative generator to create match previews.")

    if pending:
        st.warning(f"{len(pending)} matches have analyses but no narratives.")
        st.markdown("**Generate narratives:**")
        st.code("python batch/jobs/generate_narratives.py --pending", language="bash")

    st.stop()

# Filter options
col1, col2 = st.columns([2, 1])
with col1:
    matchweeks = sorted(set(m["matchweek"] for m in matches), reverse=True)
    selected_mw = st.selectbox("Matchweek", ["All"] + matchweeks, index=0)

with col2:
    status_filter = st.radio("Status", ["All", "Upcoming", "Finished"], horizontal=True)

# Apply filters
filtered = matches
if selected_mw != "All":
    filtered = [m for m in filtered if m["matchweek"] == selected_mw]
if status_filter == "Upcoming":
    filtered = [m for m in filtered if m["status"].upper() == "SCHEDULED"]
elif status_filter == "Finished":
    filtered = [m for m in filtered if m["status"].upper() == "FINISHED"]

st.divider()

# Display matches
for match in filtered:
    with st.container():
        # Header
        header_col1, header_col2, header_col3 = st.columns([3, 2, 1])

        with header_col1:
            st.markdown(f"### {match['home_team']} vs {match['away_team']}")
            st.caption(f"MW{match['matchweek']} | {match['kickoff'].strftime('%a %d %b %H:%M')} | {match['venue']}")

        with header_col2:
            if match["score"]:
                st.markdown(f"**Final Score: {match['score']}**")
            elif match["home_prob"]:
                st.markdown(f"**Prediction:** {match['home_prob']:.0%} - {match['draw_prob']:.0%} - {match['away_prob']:.0%}")

        with header_col3:
            if match["status"].upper() == "SCHEDULED":
                st.markdown("🟢 Upcoming")
            else:
                st.markdown("✅ Finished")

        # Narrative
        if match["narrative"]:
            st.markdown(match["narrative"])

            if match["narrative_generated_at"]:
                st.caption(f"Generated: {match['narrative_generated_at'].strftime('%Y-%m-%d %H:%M')}")

        st.divider()

# Sidebar
with st.sidebar:
    st.markdown("### Stats")
    st.metric("Narratives", len(matches))
    st.metric("Pending", len(pending))

    st.markdown("---")
    st.markdown("### Generate Narratives")

    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    st.markdown("**CLI Commands:**")
    st.code("""
# Generate for pending matches
python batch/jobs/generate_narratives.py --pending

# Generate for matchweek
python batch/jobs/generate_narratives.py --matchweek 20

# Generate for specific match
python batch/jobs/generate_narratives.py --match 123

# Regenerate all
python batch/jobs/generate_narratives.py --matchweek 20 --force
    """, language="bash")

    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    Narratives are generated using **Claude AI** and include:
    - Recent form analysis
    - Key tactical matchups
    - Statistical insights
    - Betting angle summary

    Narratives are generated during the weekly analysis job
    or can be generated on-demand using the CLI.
    """)
