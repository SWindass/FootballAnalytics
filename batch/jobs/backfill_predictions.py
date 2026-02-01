"""Backfill historical predictions for training data.

This job generates ELO and Poisson predictions for historical matches
to expand the training dataset for the neural stacker.
"""

import argparse
from datetime import datetime
from decimal import Decimal

import structlog
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.database import SyncSessionLocal
from app.db.models import EloRating, Match, MatchAnalysis, MatchStatus
from batch.models.elo import EloRatingSystem
from batch.models.poisson import PoissonModel

logger = structlog.get_logger()


class HistoricalPredictionBackfill:
    """Backfills predictions for historical matches."""

    def __init__(self, session: Session | None = None):
        self.session = session or SyncSessionLocal()
        self.elo = EloRatingSystem()
        self.poisson = PoissonModel()

    def run(self, seasons: list[str] = None, force: bool = False) -> dict:
        """Run the backfill job.

        Args:
            seasons: List of seasons to process (e.g., ["2023-24", "2022-23"])
                     If None, processes all seasons with finished matches
            force: If True, regenerate predictions even if they exist

        Returns:
            Summary of backfill results
        """
        logger.info("Starting historical prediction backfill")
        start_time = datetime.utcnow()

        # Get seasons to process
        if seasons is None:
            seasons = self._get_all_seasons()

        total_created = 0
        total_updated = 0

        for season in sorted(seasons):
            created, updated = self._process_season(season, force)
            total_created += created
            total_updated += updated
            logger.info(f"Season {season}: created={created}, updated={updated}")

        self.session.commit()

        duration = (datetime.utcnow() - start_time).total_seconds()
        logger.info(
            "Backfill completed",
            seasons_processed=len(seasons),
            predictions_created=total_created,
            predictions_updated=total_updated,
            duration_seconds=round(duration, 1),
        )

        return {
            "status": "success",
            "seasons_processed": len(seasons),
            "predictions_created": total_created,
            "predictions_updated": total_updated,
            "duration_seconds": duration,
        }

    def _get_all_seasons(self) -> list[str]:
        """Get all seasons with finished matches."""
        stmt = (
            select(Match.season)
            .where(Match.status == MatchStatus.FINISHED)
            .group_by(Match.season)
            .order_by(Match.season)
        )
        return [row[0] for row in self.session.execute(stmt).all()]

    def _process_season(self, season: str, force: bool) -> tuple[int, int]:
        """Process all matches in a season.

        Returns:
            Tuple of (created_count, updated_count)
        """
        logger.info(f"Processing season {season}")

        # Reset ELO for this season (will carry over from previous if available)
        self._initialize_season_elo(season)

        # Get all finished matches in chronological order
        stmt = (
            select(Match)
            .where(Match.season == season)
            .where(Match.status == MatchStatus.FINISHED)
            .order_by(Match.matchweek, Match.kickoff_time)
        )
        matches = list(self.session.execute(stmt).scalars().all())

        created = 0
        updated = 0

        # Track team goals for Poisson calculation
        team_goals_for = {}  # team_id -> [goals]
        team_goals_against = {}  # team_id -> [goals]

        for match in matches:
            # Check if analysis exists
            existing = self.session.execute(
                select(MatchAnalysis).where(MatchAnalysis.match_id == match.id)
            ).scalar_one_or_none()

            if existing and not force:
                # Skip if exists and not forcing
                continue

            # Calculate ELO predictions BEFORE updating ratings
            elo_probs = self.elo.match_probabilities(
                match.home_team_id, match.away_team_id
            )

            # Calculate Poisson predictions
            poisson_probs, home_exp, away_exp, over_prob, btts_prob = self._calculate_poisson(
                match, team_goals_for, team_goals_against
            )

            # Create or update analysis
            if existing:
                analysis = existing
                updated += 1
            else:
                analysis = MatchAnalysis(match_id=match.id)
                created += 1

            # Set predictions
            analysis.elo_home_prob = Decimal(str(round(elo_probs[0], 4)))
            analysis.elo_draw_prob = Decimal(str(round(elo_probs[1], 4)))
            analysis.elo_away_prob = Decimal(str(round(elo_probs[2], 4)))

            analysis.poisson_home_prob = Decimal(str(round(poisson_probs[0], 4)))
            analysis.poisson_draw_prob = Decimal(str(round(poisson_probs[1], 4)))
            analysis.poisson_away_prob = Decimal(str(round(poisson_probs[2], 4)))
            analysis.poisson_over_2_5_prob = Decimal(str(round(over_prob, 4)))
            analysis.poisson_btts_prob = Decimal(str(round(btts_prob, 4)))

            analysis.predicted_home_goals = Decimal(str(round(home_exp, 2)))
            analysis.predicted_away_goals = Decimal(str(round(away_exp, 2)))

            # Simple consensus (weighted average)
            consensus_home = elo_probs[0] * 0.45 + poisson_probs[0] * 0.55
            consensus_draw = elo_probs[1] * 0.45 + poisson_probs[1] * 0.55
            consensus_away = elo_probs[2] * 0.45 + poisson_probs[2] * 0.55
            total = consensus_home + consensus_draw + consensus_away

            analysis.consensus_home_prob = Decimal(str(round(consensus_home / total, 4)))
            analysis.consensus_draw_prob = Decimal(str(round(consensus_draw / total, 4)))
            analysis.consensus_away_prob = Decimal(str(round(consensus_away / total, 4)))

            if not existing:
                self.session.add(analysis)

            # NOW update ELO ratings based on actual result
            if match.home_score is not None and match.away_score is not None:
                self.elo.update_ratings(
                    match.home_team_id,
                    match.away_team_id,
                    match.home_score,
                    match.away_score,
                )

                # Update goal tracking for Poisson
                if match.home_team_id not in team_goals_for:
                    team_goals_for[match.home_team_id] = []
                    team_goals_against[match.home_team_id] = []
                if match.away_team_id not in team_goals_for:
                    team_goals_for[match.away_team_id] = []
                    team_goals_against[match.away_team_id] = []

                team_goals_for[match.home_team_id].append(match.home_score)
                team_goals_against[match.home_team_id].append(match.away_score)
                team_goals_for[match.away_team_id].append(match.away_score)
                team_goals_against[match.away_team_id].append(match.home_score)

            # Commit periodically to avoid memory issues
            if (created + updated) % 100 == 0:
                self.session.flush()

        return created, updated

    def _initialize_season_elo(self, season: str):
        """Initialize ELO ratings for a season.

        Carries over ratings from previous season with regression to mean.
        """
        # Get previous season
        year = int(season.split("-")[0])
        prev_season = f"{year-1}-{str(year)[2:]}"

        # Try to get end-of-season ratings from previous season
        stmt = (
            select(EloRating)
            .where(EloRating.season == prev_season)
            .order_by(EloRating.matchweek.desc())
        )
        prev_ratings = {r.team_id: r.rating for r in self.session.execute(stmt).scalars().all()}

        # Get teams for this season
        stmt = (
            select(Match.home_team_id)
            .where(Match.season == season)
            .union(
                select(Match.away_team_id)
                .where(Match.season == season)
            )
        )
        team_ids = [row[0] for row in self.session.execute(stmt).all()]

        # Initialize ELO system
        self.elo = EloRatingSystem()

        for team_id in team_ids:
            if team_id in prev_ratings:
                # Carry over with regression to mean (30% towards 1500)
                prev_rating = float(prev_ratings[team_id])
                new_rating = prev_rating * 0.7 + 1500 * 0.3
                self.elo.set_rating(team_id, new_rating)
            else:
                # New team - start at 1500 (or slightly below for promoted teams)
                self.elo.set_rating(team_id, 1450)

    def _calculate_poisson(
        self,
        match: Match,
        team_goals_for: dict,
        team_goals_against: dict,
    ) -> tuple:
        """Calculate Poisson predictions for a match.

        Returns:
            Tuple of (probs, home_exp, away_exp, over_prob, btts_prob)
        """
        home_id = match.home_team_id
        away_id = match.away_team_id

        # Get team averages (or defaults if not enough data)
        league_avg = 1.4  # EPL average goals per game

        def get_avg(goals_list, default):
            if not goals_list or len(goals_list) < 3:
                return default
            return sum(goals_list[-10:]) / len(goals_list[-10:])  # Last 10 games

        home_attack = get_avg(team_goals_for.get(home_id, []), league_avg)
        home_defense = get_avg(team_goals_against.get(home_id, []), league_avg)
        away_attack = get_avg(team_goals_for.get(away_id, []), league_avg)
        away_defense = get_avg(team_goals_against.get(away_id, []), league_avg)

        # Normalize to strength values
        home_att_str = home_attack / league_avg if league_avg > 0 else 1.0
        home_def_str = home_defense / league_avg if league_avg > 0 else 1.0
        away_att_str = away_attack / league_avg if league_avg > 0 else 1.0
        away_def_str = away_defense / league_avg if league_avg > 0 else 1.0

        # Calculate expected goals
        home_exp, away_exp = self.poisson.calculate_expected_goals(
            home_att_str, home_def_str, away_att_str, away_def_str
        )

        # Get probabilities
        probs = self.poisson.match_probabilities(home_exp, away_exp)
        over_prob, _ = self.poisson.over_under_probability(home_exp, away_exp)
        btts_prob, _ = self.poisson.btts_probability(home_exp, away_exp)

        return probs, home_exp, away_exp, over_prob, btts_prob


def run_backfill(seasons: list[str] = None, force: bool = False):
    """Entry point for backfill job."""
    with SyncSessionLocal() as session:
        job = HistoricalPredictionBackfill(session)
        return job.run(seasons=seasons, force=force)


def main():
    parser = argparse.ArgumentParser(
        description="Backfill historical predictions for neural stacker training"
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        help="Specific seasons to process (e.g., 2023-24 2022-23)",
    )
    parser.add_argument(
        "--recent",
        type=int,
        default=0,
        help="Process last N seasons only",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate predictions even if they exist",
    )

    args = parser.parse_args()

    # Suppress SQLAlchemy logging
    import logging
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

    seasons = args.seasons
    if args.recent > 0:
        # Get last N seasons
        with SyncSessionLocal() as session:
            stmt = (
                select(Match.season)
                .where(Match.status == MatchStatus.FINISHED)
                .group_by(Match.season)
                .order_by(Match.season.desc())
                .limit(args.recent)
            )
            seasons = [row[0] for row in session.execute(stmt).all()]
            seasons.reverse()  # Process oldest first

    print("Historical Prediction Backfill")
    print("=" * 40)
    if seasons:
        print(f"Seasons: {', '.join(seasons)}")
    else:
        print("Seasons: ALL")
    print(f"Force regenerate: {args.force}")
    print()

    result = run_backfill(seasons=seasons, force=args.force)

    print("\nBackfill complete!")
    print(f"  Seasons processed: {result['seasons_processed']}")
    print(f"  Predictions created: {result['predictions_created']}")
    print(f"  Predictions updated: {result['predictions_updated']}")
    print(f"  Duration: {result['duration_seconds']:.1f}s")


if __name__ == "__main__":
    main()
