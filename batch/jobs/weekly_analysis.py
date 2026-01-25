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
from app.db.models import EloRating, Match, MatchAnalysis, MatchStatus, Referee, Team, TeamStats
from batch.ai.narrative_generator import NarrativeGenerator
from batch.betting.value_calculator import ValueBetDetector, calculate_consensus_probabilities
from batch.data_sources.tipster_aggregator import TipsterAggregator
from batch.models.consensus_stacker import ConsensusStacker
from batch.models.elo import EloRatingSystem
from batch.models.neural_stacker import NeuralStacker
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
        self.neural_stacker = NeuralStacker()
        self._neural_stacker_available = self.neural_stacker.load_model()
        self.tipster_aggregator = TipsterAggregator()
        self._market_predictions: dict[tuple[str, str], tuple[float, float, float]] = {}
        self.consensus_stacker = ConsensusStacker()
        self._consensus_stacker_available = self.consensus_stacker.load_model()

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

            # 1b. Load market predictions (betting odds consensus)
            self._load_market_predictions()

            # 2. Load teams and their current stats
            teams = self._load_teams()
            team_stats = self._load_team_stats(matchweek - 1)  # Previous matchweek stats

            # 3. Initialize ELO ratings
            self._initialize_elo_ratings(matchweek - 1)
            elo_ratings = self._load_elo_ratings(matchweek - 1)

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

            # Log neural stacker status
            if self._neural_stacker_available:
                logger.info("Using neural stacker for consensus predictions")
            else:
                logger.info("Neural stacker not available, using weighted average")

            # 6. Generate predictions for each match
            analyses_created = 0
            for match in matches:
                try:
                    analysis = self._analyze_match(
                        match, teams, team_stats, elo_ratings, poisson_strengths
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

    def _calculate_agreement_confidence(
        self,
        elo_probs: tuple[float, float, float],
        poisson_probs: tuple[float, float, float],
        market_probs: Optional[tuple[float, float, float]],
    ) -> float:
        """Calculate confidence based on model agreement.

        When ELO, Poisson, and market odds all agree on the favorite,
        confidence is higher. Returns 0 if models disagree.

        Research shows:
        - All 3 agree: 56.8% accuracy
        - All 3 agree + 50%+ confidence: 69.6% accuracy
        - Models disagree: 40.9% accuracy
        """
        import numpy as np

        # Use defaults if no market data
        if not market_probs:
            market_probs = (0.4, 0.27, 0.33)

        # Which outcome does each model favor?
        elo_favorite = np.argmax(elo_probs)
        poisson_favorite = np.argmax(poisson_probs)
        market_favorite = np.argmax(market_probs)

        # Do all 3 agree?
        if not (elo_favorite == poisson_favorite == market_favorite):
            return 0.0  # No agreement

        # All agree - calculate agreement strength
        # (minimum probability across models for the agreed favorite)
        if elo_favorite == 0:  # Home
            agreement_strength = min(elo_probs[0], poisson_probs[0], market_probs[0])
        elif elo_favorite == 1:  # Draw
            agreement_strength = min(elo_probs[1], poisson_probs[1], market_probs[1])
        else:  # Away
            agreement_strength = min(elo_probs[2], poisson_probs[2], market_probs[2])

        return agreement_strength

    def _apply_disagreement_draw_boost(
        self,
        consensus: tuple[float, float, float],
        elo_probs: tuple[float, float, float],
        poisson_probs: tuple[float, float, float],
        market_probs: Optional[tuple[float, float, float]],
    ) -> tuple[float, float, float]:
        """Boost draw probability when models disagree.

        Analysis shows that when models disagree on the favorite,
        draws occur more frequently:
        - Models agree: 23.7% draw rate
        - Models disagree: 28.6% draw rate (+4.9pp)

        This adjustment improves calibration for uncertain matches.
        """
        import numpy as np

        # Use defaults if no market data
        if not market_probs:
            market_probs = (0.4, 0.27, 0.33)

        # Which outcome does each model favor?
        elo_favorite = np.argmax(elo_probs)
        poisson_favorite = np.argmax(poisson_probs)
        market_favorite = np.argmax(market_probs)

        # If all models agree, no adjustment needed
        if elo_favorite == poisson_favorite == market_favorite:
            return consensus

        # Models disagree - boost draw probability
        # Empirical boost: +5pp for draw when models disagree
        DRAW_BOOST = 0.05

        home_prob, draw_prob, away_prob = consensus

        # Add boost to draw
        new_draw = draw_prob + DRAW_BOOST

        # Subtract proportionally from home/away based on their relative strengths
        home_share = home_prob / (home_prob + away_prob) if (home_prob + away_prob) > 0 else 0.5
        away_share = 1 - home_share

        new_home = home_prob - (DRAW_BOOST * home_share)
        new_away = away_prob - (DRAW_BOOST * away_share)

        # Ensure probabilities stay valid
        new_home = max(0.05, new_home)
        new_away = max(0.05, new_away)
        new_draw = min(0.5, new_draw)  # Cap draw at 50%

        # Normalize
        total = new_home + new_draw + new_away
        return (new_home / total, new_draw / total, new_away / total)

    def _load_market_predictions(self) -> None:
        """Load market consensus predictions from betting odds.

        Fetches predictions based on betting odds from multiple bookmakers.
        These represent the "wisdom of crowds" and can improve our predictions.
        """
        try:
            predictions = asyncio.run(self.tipster_aggregator.market_predictor.fetch_predictions())

            for pred in predictions:
                # Key by team names (as they appear in our database)
                key = (pred.home_team, pred.away_team)
                self._market_predictions[key] = (pred.home_prob, pred.draw_prob, pred.away_prob)

            logger.info(f"Loaded {len(self._market_predictions)} market consensus predictions")
        except Exception as e:
            logger.warning(f"Failed to load market predictions: {e}")
            self._market_predictions = {}

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

    def _load_elo_ratings(self, matchweek: int) -> dict[int, EloRating]:
        """Load ELO ratings for a specific matchweek."""
        stmt = (
            select(EloRating)
            .where(EloRating.season == settings.current_season)
            .where(EloRating.matchweek == matchweek)
        )
        result = self.session.execute(stmt)
        ratings = result.scalars().all()
        return {r.team_id: r for r in ratings}

    def _calculate_rest_days(self, team_id: int, match_date: datetime) -> Optional[int]:
        """Calculate days since team's last match.

        Args:
            team_id: Team ID
            match_date: Date of upcoming match

        Returns:
            Number of days since last match, or None if no previous match
        """
        stmt = (
            select(Match)
            .where(
                (Match.home_team_id == team_id) | (Match.away_team_id == team_id)
            )
            .where(Match.kickoff_time < match_date)
            .where(Match.status == MatchStatus.FINISHED)
            .order_by(Match.kickoff_time.desc())
            .limit(1)
        )
        prev_match = self.session.execute(stmt).scalar_one_or_none()

        if not prev_match:
            return None

        delta = match_date - prev_match.kickoff_time
        return delta.days

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
        elo_ratings: dict[int, EloRating],
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

        # Look up market prediction if available
        home_team = teams[home_id]
        away_team = teams[away_id]
        market_key = (home_team.name, away_team.name)
        market_probs = self._market_predictions.get(market_key)

        # Calculate consensus using neural stacker if available
        if self._neural_stacker_available:
            # Create a temporary analysis object with predictions for the neural stacker
            temp_analysis = MatchAnalysis(match_id=match.id)
            temp_analysis.elo_home_prob = Decimal(str(round(elo_probs[0], 4)))
            temp_analysis.elo_draw_prob = Decimal(str(round(elo_probs[1], 4)))
            temp_analysis.elo_away_prob = Decimal(str(round(elo_probs[2], 4)))
            temp_analysis.poisson_home_prob = Decimal(str(round(poisson_probs[0], 4)))
            temp_analysis.poisson_draw_prob = Decimal(str(round(poisson_probs[1], 4)))
            temp_analysis.poisson_away_prob = Decimal(str(round(poisson_probs[2], 4)))
            temp_analysis.poisson_over_2_5_prob = Decimal(str(round(over_2_5_prob, 4)))
            temp_analysis.poisson_btts_prob = Decimal(str(round(btts_prob, 4)))

            # Get team context for neural stacker
            home_stats_obj = team_stats.get(home_id)
            away_stats_obj = team_stats.get(away_id)
            home_elo_obj = elo_ratings.get(home_id)
            away_elo_obj = elo_ratings.get(away_id)

            # Get referee if assigned
            referee_obj = None
            if match.referee_id:
                referee_obj = self.session.get(Referee, match.referee_id)

            # Calculate rest days for each team
            home_rest = self._calculate_rest_days(home_id, match.kickoff_time)
            away_rest = self._calculate_rest_days(away_id, match.kickoff_time)

            # Get neural network consensus
            neural_consensus = self.neural_stacker.predict(
                temp_analysis, home_stats_obj, away_stats_obj, home_elo_obj, away_elo_obj,
                referee_obj, home_rest, away_rest
            )

            # Blend with market consensus if available (40% market, 60% neural)
            if market_probs:
                consensus = (
                    0.6 * neural_consensus[0] + 0.4 * market_probs[0],
                    0.6 * neural_consensus[1] + 0.4 * market_probs[1],
                    0.6 * neural_consensus[2] + 0.4 * market_probs[2],
                )
                # Normalize
                total = sum(consensus)
                consensus = (consensus[0] / total, consensus[1] / total, consensus[2] / total)
                logger.debug(f"Blended consensus (60% neural + 40% market): {consensus}")
            else:
                consensus = neural_consensus
                logger.debug(f"Neural stacker consensus (no market data): {consensus}")

            # Apply disagreement draw boost
            consensus = self._apply_disagreement_draw_boost(
                consensus, elo_probs, poisson_probs, market_probs
            )
        else:
            # Fallback to weighted average, incorporating market if available
            if market_probs:
                # Blend ELO, Poisson, and market (30%, 30%, 40%)
                consensus = (
                    0.3 * elo_probs[0] + 0.3 * poisson_probs[0] + 0.4 * market_probs[0],
                    0.3 * elo_probs[1] + 0.3 * poisson_probs[1] + 0.4 * market_probs[1],
                    0.3 * elo_probs[2] + 0.3 * poisson_probs[2] + 0.4 * market_probs[2],
                )
                # Normalize
                total = sum(consensus)
                consensus = (consensus[0] / total, consensus[1] / total, consensus[2] / total)
            else:
                consensus = calculate_consensus_probabilities(
                    elo_probs, poisson_probs, None  # XGBoost requires trained model
                )

            # Apply disagreement draw boost
            consensus = self._apply_disagreement_draw_boost(
                consensus, elo_probs, poisson_probs, market_probs
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

        # Store market consensus if available
        if market_probs:
            analysis.features["market_home_prob"] = market_probs[0]
            analysis.features["market_draw_prob"] = market_probs[1]
            analysis.features["market_away_prob"] = market_probs[2]

        # Calculate agreement confidence
        agreement_confidence = self._calculate_agreement_confidence(
            elo_probs, poisson_probs, market_probs
        )
        analysis.confidence = Decimal(str(round(agreement_confidence, 3)))
        analysis.features["models_agree"] = agreement_confidence > 0

        if not existing:
            self.session.add(analysis)

        # Generate AI narrative
        try:
            narrative = self._generate_narrative(
                match, teams, team_stats, consensus, home_exp, away_exp,
                elo_probs, poisson_probs, market_probs, agreement_confidence
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
        elo_probs: tuple[float, float, float] = None,
        poisson_probs: tuple[float, float, float] = None,
        market_probs: tuple[float, float, float] = None,
        confidence: float = 0.0,
    ) -> Optional[str]:
        """Generate AI narrative for a match with confidence analysis."""
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

        # Build confidence data for AI narrative
        confidence_data = {
            "confidence": confidence,
            "models_agree": confidence > 0,
            "elo_probs": elo_probs or (0.4, 0.27, 0.33),
            "poisson_probs": poisson_probs or (0.4, 0.27, 0.33),
            "market_probs": market_probs or (0.4, 0.27, 0.33),
        }

        # Build odds dict for value edge analysis
        # Convert probabilities back to decimal odds for the narrative generator
        odds_data = None
        if market_probs:
            odds_data = {
                "home_odds": 1 / market_probs[0] if market_probs[0] > 0 else 0,
                "draw_odds": 1 / market_probs[1] if market_probs[1] > 0 else 0,
                "away_odds": 1 / market_probs[2] if market_probs[2] > 0 else 0,
            }

        # Generate narrative (async)
        return asyncio.run(
            self.narrative_gen.generate_match_preview(
                match_data=match_data,
                home_stats=home_stats_dict,
                away_stats=away_stats_dict,
                predictions=predictions,
                h2h_history=h2h,
                confidence_data=confidence_data,
                odds=odds_data,
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
