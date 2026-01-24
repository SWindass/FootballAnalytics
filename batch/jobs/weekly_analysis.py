"""Weekly analysis batch job - runs Tuesday 5PM.

Generates predictions and narratives for the upcoming matchweek.
"""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional

import structlog
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.database import SyncSessionLocal
from app.db.models import EloRating, Match, MatchAnalysis, MatchStatus, Team, TeamStats
from batch.ai.narrative_generator import NarrativeGenerator
from batch.betting.value_calculator import ValueBetDetector, calculate_consensus_probabilities
from batch.models.elo import EloRatingSystem
from batch.models.poisson import PoissonModel, calculate_team_strengths
from batch.models.xgboost_model import MatchOutcomeClassifier, build_feature_dataframe

logger = structlog.get_logger()
settings = get_settings()


class WeeklyAnalysisJob:
    """Generates weekly match analyses."""

    def __init__(self, session: Optional[Session] = None):
        self.session = session or SyncSessionLocal()
        self.elo = EloRatingSystem()
        self.poisson = PoissonModel()
        self.classifier = MatchOutcomeClassifier()
        self.narrative_gen = NarrativeGenerator()
        self.value_detector = ValueBetDetector()

    def run(self) -> dict:
        """Execute the weekly analysis job.

        Returns:
            Summary of job results
        """
        logger.info("Starting weekly analysis job")
        start_time = datetime.utcnow()

        try:
            # 1. Find upcoming matchweek
            matchweek = self._find_next_matchweek()
            if not matchweek:
                logger.warning("No upcoming matchweek found")
                return {"status": "skipped", "reason": "No upcoming matches"}

            logger.info(f"Processing matchweek {matchweek}")

            # 2. Load teams and their current stats
            teams = self._load_teams()
            team_stats = self._load_team_stats(matchweek - 1)  # Previous matchweek stats

            # 3. Initialize ELO ratings
            self._initialize_elo_ratings(matchweek - 1)

            # 4. Calculate Poisson team strengths
            completed_matches = self._load_completed_matches()
            poisson_strengths = calculate_team_strengths(
                completed_matches,
                league_avg_scored=1.4,  # EPL average
                league_avg_conceded=1.4,
            )

            # 5. Get upcoming matches
            matches = self._load_matchweek_fixtures(matchweek)
            logger.info(f"Found {len(matches)} fixtures for matchweek {matchweek}")

            # 6. Generate predictions for each match
            analyses_created = 0
            for match in matches:
                try:
                    analysis = self._analyze_match(
                        match, teams, team_stats, poisson_strengths
                    )
                    if analysis:
                        analyses_created += 1
                except Exception as e:
                    logger.error(f"Failed to analyze match {match.id}", error=str(e))

            self.session.commit()

            duration = (datetime.utcnow() - start_time).total_seconds()
            logger.info(
                "Weekly analysis completed",
                matchweek=matchweek,
                matches=len(matches),
                analyses=analyses_created,
                duration_seconds=duration,
            )

            return {
                "status": "success",
                "matchweek": matchweek,
                "matches_processed": len(matches),
                "analyses_created": analyses_created,
                "duration_seconds": duration,
            }

        except Exception as e:
            logger.error("Weekly analysis job failed", error=str(e))
            self.session.rollback()
            raise

    def _find_next_matchweek(self) -> Optional[int]:
        """Find the next matchweek with scheduled matches."""
        now = datetime.utcnow()

        stmt = (
            select(Match)
            .where(Match.season == settings.current_season)
            .where(Match.status == MatchStatus.SCHEDULED)
            .where(Match.kickoff_time > now)
            .order_by(Match.kickoff_time)
            .limit(1)
        )
        result = self.session.execute(stmt)
        match = result.scalar_one_or_none()

        return match.matchweek if match else None

    def _load_teams(self) -> dict[int, Team]:
        """Load all teams indexed by ID."""
        stmt = select(Team)
        result = self.session.execute(stmt)
        teams = result.scalars().all()
        return {t.id: t for t in teams}

    def _load_team_stats(self, matchweek: int) -> dict[int, TeamStats]:
        """Load team stats for a specific matchweek."""
        stmt = (
            select(TeamStats)
            .where(TeamStats.season == settings.current_season)
            .where(TeamStats.matchweek == matchweek)
        )
        result = self.session.execute(stmt)
        stats = result.scalars().all()
        return {s.team_id: s for s in stats}

    def _initialize_elo_ratings(self, matchweek: int) -> None:
        """Initialize ELO system with current ratings."""
        stmt = (
            select(EloRating)
            .where(EloRating.season == settings.current_season)
            .where(EloRating.matchweek == matchweek)
        )
        result = self.session.execute(stmt)
        ratings = result.scalars().all()

        for rating in ratings:
            self.elo.set_rating(rating.team_id, float(rating.rating))

    def _load_completed_matches(self) -> list[dict]:
        """Load completed matches for the season."""
        stmt = (
            select(Match)
            .where(Match.season == settings.current_season)
            .where(Match.status == MatchStatus.FINISHED)
        )
        result = self.session.execute(stmt)
        matches = result.scalars().all()

        return [
            {
                "home_team_id": m.home_team_id,
                "away_team_id": m.away_team_id,
                "home_score": m.home_score,
                "away_score": m.away_score,
            }
            for m in matches
        ]

    def _load_matchweek_fixtures(self, matchweek: int) -> list[Match]:
        """Load fixtures for a matchweek."""
        stmt = (
            select(Match)
            .where(Match.season == settings.current_season)
            .where(Match.matchweek == matchweek)
            .where(Match.status == MatchStatus.SCHEDULED)
        )
        result = self.session.execute(stmt)
        return list(result.scalars().all())

    def _analyze_match(
        self,
        match: Match,
        teams: dict[int, Team],
        team_stats: dict[int, TeamStats],
        poisson_strengths: dict[int, tuple[float, float]],
    ) -> Optional[MatchAnalysis]:
        """Generate analysis for a single match."""
        home_id = match.home_team_id
        away_id = match.away_team_id

        # Get ELO predictions
        elo_probs = self.elo.match_probabilities(home_id, away_id)

        # Get Poisson predictions
        home_strength = poisson_strengths.get(home_id, (1.0, 1.0))
        away_strength = poisson_strengths.get(away_id, (1.0, 1.0))
        home_exp, away_exp = self.poisson.calculate_expected_goals(
            home_strength[0], home_strength[1],
            away_strength[0], away_strength[1],
        )
        poisson_probs = self.poisson.match_probabilities(home_exp, away_exp)
        over_2_5_prob, _ = self.poisson.over_under_probability(home_exp, away_exp)
        btts_prob, _ = self.poisson.btts_probability(home_exp, away_exp)

        # Calculate consensus
        consensus = calculate_consensus_probabilities(
            elo_probs, poisson_probs, None  # XGBoost requires trained model
        )

        # Create or update analysis
        existing = self.session.execute(
            select(MatchAnalysis).where(MatchAnalysis.match_id == match.id)
        ).scalar_one_or_none()

        if existing:
            analysis = existing
        else:
            analysis = MatchAnalysis(match_id=match.id)

        # Set predictions
        analysis.elo_home_prob = Decimal(str(round(elo_probs[0], 4)))
        analysis.elo_draw_prob = Decimal(str(round(elo_probs[1], 4)))
        analysis.elo_away_prob = Decimal(str(round(elo_probs[2], 4)))

        analysis.poisson_home_prob = Decimal(str(round(poisson_probs[0], 4)))
        analysis.poisson_draw_prob = Decimal(str(round(poisson_probs[1], 4)))
        analysis.poisson_away_prob = Decimal(str(round(poisson_probs[2], 4)))
        analysis.poisson_over_2_5_prob = Decimal(str(round(over_2_5_prob, 4)))
        analysis.poisson_btts_prob = Decimal(str(round(btts_prob, 4)))

        analysis.consensus_home_prob = Decimal(str(round(consensus[0], 4)))
        analysis.consensus_draw_prob = Decimal(str(round(consensus[1], 4)))
        analysis.consensus_away_prob = Decimal(str(round(consensus[2], 4)))

        analysis.predicted_home_goals = Decimal(str(round(home_exp, 2)))
        analysis.predicted_away_goals = Decimal(str(round(away_exp, 2)))

        # Store feature data
        analysis.features = {
            "home_elo": self.elo.get_rating(home_id),
            "away_elo": self.elo.get_rating(away_id),
            "home_attack": home_strength[0],
            "home_defense": home_strength[1],
            "away_attack": away_strength[0],
            "away_defense": away_strength[1],
        }

        if not existing:
            self.session.add(analysis)

        # Generate AI narrative
        try:
            narrative = self._generate_narrative(
                match, teams, team_stats, consensus, home_exp, away_exp
            )
            if narrative:
                analysis.narrative = narrative
                analysis.narrative_generated_at = datetime.utcnow()
                logger.info(f"Generated narrative for {teams[home_id].short_name} vs {teams[away_id].short_name}")
        except Exception as e:
            logger.warning(f"Failed to generate narrative for match {match.id}", error=str(e))

        return analysis

    def _generate_narrative(
        self,
        match: Match,
        teams: dict[int, Team],
        team_stats: dict[int, TeamStats],
        consensus: tuple[float, float, float],
        home_exp: float,
        away_exp: float,
    ) -> Optional[str]:
        """Generate AI narrative for a match."""
        home_team = teams[match.home_team_id]
        away_team = teams[match.away_team_id]
        home_stats = team_stats.get(match.home_team_id)
        away_stats = team_stats.get(match.away_team_id)

        # Build match data dict
        match_data = {
            "home_team": home_team.short_name,
            "away_team": away_team.short_name,
            "kickoff_time": match.kickoff_time,
            "venue": home_team.venue or "TBC",
        }

        # Build home stats dict
        home_stats_dict = {}
        if home_stats:
            home_stats_dict = {
                "form": home_stats.form or "N/A",
                "position": "N/A",  # Would need league table query
                "goals_scored": home_stats.goals_scored,
                "goals_conceded": home_stats.goals_conceded,
                "avg_xg_for": float(home_stats.avg_xg_for) if home_stats.avg_xg_for else None,
                "home_wins": home_stats.home_wins,
                "home_draws": home_stats.home_draws,
                "home_losses": home_stats.home_losses,
                "injuries": [],  # Would need injury data
            }

        # Build away stats dict
        away_stats_dict = {}
        if away_stats:
            away_stats_dict = {
                "form": away_stats.form or "N/A",
                "position": "N/A",
                "goals_scored": away_stats.goals_scored,
                "goals_conceded": away_stats.goals_conceded,
                "avg_xg_for": float(away_stats.avg_xg_for) if away_stats.avg_xg_for else None,
                "away_wins": away_stats.away_wins,
                "away_draws": away_stats.away_draws,
                "away_losses": away_stats.away_losses,
                "injuries": [],
            }

        # Build predictions dict
        predictions = {
            "home_win": consensus[0],
            "draw": consensus[1],
            "away_win": consensus[2],
            "predicted_score": f"{home_exp:.1f}-{away_exp:.1f}",
        }

        # Get H2H history
        h2h = self._get_head_to_head(match.home_team_id, match.away_team_id, teams)

        # Generate narrative (async)
        return asyncio.run(
            self.narrative_gen.generate_match_preview(
                match_data=match_data,
                home_stats=home_stats_dict,
                away_stats=away_stats_dict,
                predictions=predictions,
                h2h_history=h2h,
            )
        )

    def _get_head_to_head(
        self,
        home_id: int,
        away_id: int,
        teams: dict[int, Team],
    ) -> list[dict]:
        """Get head-to-head history between two teams."""
        stmt = (
            select(Match)
            .where(
                ((Match.home_team_id == home_id) & (Match.away_team_id == away_id)) |
                ((Match.home_team_id == away_id) & (Match.away_team_id == home_id))
            )
            .where(Match.status == MatchStatus.FINISHED)
            .order_by(Match.kickoff_time.desc())
            .limit(5)
        )
        matches = list(self.session.execute(stmt).scalars().all())

        h2h = []
        for m in matches:
            h2h.append({
                "date": m.kickoff_time.strftime("%d %b %Y"),
                "home_team": teams[m.home_team_id].short_name,
                "away_team": teams[m.away_team_id].short_name,
                "home_score": m.home_score,
                "away_score": m.away_score,
            })
        return h2h


def run_weekly_analysis():
    """Entry point for weekly analysis job."""
    with SyncSessionLocal() as session:
        job = WeeklyAnalysisJob(session)
        return job.run()


if __name__ == "__main__":
    result = run_weekly_analysis()
    print(f"Job completed: {result}")
