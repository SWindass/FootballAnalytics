"""Results update batch job - runs every 6 hours.

Updates match results, recalculates ELO ratings, and updates team stats.
"""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional

import structlog
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.database import SyncSessionLocal
from app.db.models import EloRating, Match, MatchStatus, Team, TeamStats, ValueBet
from batch.data_sources.football_data_org import FootballDataClient, parse_match
from batch.data_sources.understat import UnderstatScraper, parse_understat_match
from batch.models.elo import EloRatingSystem

logger = structlog.get_logger()
settings = get_settings()


class ResultsUpdateJob:
    """Updates match results and recalculates statistics."""

    def __init__(self, session: Optional[Session] = None):
        self.session = session or SyncSessionLocal()
        self.football_client = FootballDataClient()
        self.elo = EloRatingSystem()

    def run(self) -> dict:
        """Execute the results update job.

        Returns:
            Summary of job results
        """
        logger.info("Starting results update job")
        start_time = datetime.utcnow()

        try:
            # 1. Fetch recent results from API
            recent_results = asyncio.run(self._fetch_recent_results())
            logger.info(f"Fetched {len(recent_results)} results from API")

            # 2. Update matches in database
            updated_count = self._update_match_results(recent_results)
            logger.info(f"Updated {updated_count} match results")

            # 3. Fetch xG data for finished matches
            xg_updated = asyncio.run(self._update_xg_data())
            logger.info(f"Updated xG for {xg_updated} matches")

            # 4. Recalculate ELO ratings
            elo_updated = self._recalculate_elo_ratings()

            # 5. Update team statistics
            stats_updated = self._update_team_stats()

            # 6. Update value bet results
            bets_resolved = self._resolve_value_bets()

            self.session.commit()

            duration = (datetime.utcnow() - start_time).total_seconds()
            logger.info(
                "Results update completed",
                matches_updated=updated_count,
                xg_updated=xg_updated,
                elo_updated=elo_updated,
                stats_updated=stats_updated,
                bets_resolved=bets_resolved,
                duration_seconds=duration,
            )

            return {
                "status": "success",
                "matches_updated": updated_count,
                "xg_updated": xg_updated,
                "elo_recalculated": elo_updated,
                "stats_updated": stats_updated,
                "bets_resolved": bets_resolved,
                "duration_seconds": duration,
            }

        except Exception as e:
            logger.error("Results update job failed", error=str(e))
            self.session.rollback()
            raise

    async def _fetch_recent_results(self) -> list[dict]:
        """Fetch recent match results from API."""
        # Get matches from last 7 days
        date_from = (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%d")
        date_to = datetime.utcnow().strftime("%Y-%m-%d")

        # Extract year from season (e.g., "2025-26" -> "2025")
        season_year = settings.current_season.split("-")[0]

        matches = await self.football_client.get_matches(
            season=season_year,
            status="FINISHED",
            date_from=date_from,
            date_to=date_to,
        )

        return [parse_match(m) for m in matches]

    def _update_match_results(self, results: list[dict]) -> int:
        """Update match results in database."""
        updated = 0

        for result in results:
            # Find match by external ID
            stmt = select(Match).where(Match.external_id == result["external_id"])
            match = self.session.execute(stmt).scalar_one_or_none()

            if not match:
                continue

            # Update only if status changed or scores added
            if match.status != result["status"] or match.home_score != result.get("home_score"):
                match.status = result["status"]
                match.home_score = result.get("home_score")
                match.away_score = result.get("away_score")
                match.home_ht_score = result.get("home_ht_score")
                match.away_ht_score = result.get("away_ht_score")
                match.updated_at = datetime.utcnow()
                updated += 1

        return updated

    async def _update_xg_data(self) -> int:
        """Fetch and update xG data from Understat."""
        scraper = UnderstatScraper()
        try:
            # Get all finished matches without xG
            stmt = (
                select(Match)
                .where(Match.season == settings.current_season)
                .where(Match.status == MatchStatus.FINISHED)
                .where(Match.home_xg.is_(None))
            )
            matches_needing_xg = list(self.session.execute(stmt).scalars().all())

            if not matches_needing_xg:
                return 0

            # Fetch Understat data
            understat_matches = await scraper.get_league_matches("2024")

            updated = 0
            for us_match in understat_matches:
                parsed = parse_understat_match(us_match)
                if not parsed["is_finished"] or parsed["home_xg"] is None:
                    continue

                # Try to match with our fixture
                for match in matches_needing_xg:
                    home_team = self.session.get(Team, match.home_team_id)
                    away_team = self.session.get(Team, match.away_team_id)

                    if not home_team or not away_team:
                        continue

                    # Simple name matching
                    if (
                        parsed["home_team"].lower() in home_team.name.lower()
                        or home_team.name.lower() in parsed["home_team"].lower()
                    ) and (
                        parsed["away_team"].lower() in away_team.name.lower()
                        or away_team.name.lower() in parsed["away_team"].lower()
                    ):
                        # Check date is close
                        time_diff = abs((parsed["datetime"] - match.kickoff_time).total_seconds())
                        if time_diff < 86400:  # Within 24 hours
                            match.home_xg = parsed["home_xg"]
                            match.away_xg = parsed["away_xg"]
                            updated += 1
                            break

            return updated
        finally:
            await scraper.close()

    def _recalculate_elo_ratings(self) -> int:
        """Recalculate ELO ratings based on results."""
        # Get completed matches in order
        stmt = (
            select(Match)
            .where(Match.season == settings.current_season)
            .where(Match.status == MatchStatus.FINISHED)
            .order_by(Match.matchweek, Match.kickoff_time)
        )
        matches = list(self.session.execute(stmt).scalars().all())

        if not matches:
            return 0

        # Initialize ELO system
        self.elo = EloRatingSystem()

        # Track which matchweeks have new ratings
        updated_matchweeks = set()

        for match in matches:
            if match.home_score is None or match.away_score is None:
                continue

            # Update ratings
            home_change, away_change = self.elo.update_ratings(
                match.home_team_id,
                match.away_team_id,
                match.home_score,
                match.away_score,
            )

            updated_matchweeks.add(match.matchweek)

        # Save updated ratings
        for matchweek in updated_matchweeks:
            for team_id, rating in self.elo.ratings.items():
                existing = self.session.execute(
                    select(EloRating)
                    .where(EloRating.team_id == team_id)
                    .where(EloRating.season == settings.current_season)
                    .where(EloRating.matchweek == matchweek)
                ).scalar_one_or_none()

                if existing:
                    existing.rating = Decimal(str(round(rating, 2)))
                else:
                    new_rating = EloRating(
                        team_id=team_id,
                        season=settings.current_season,
                        matchweek=matchweek,
                        rating=Decimal(str(round(rating, 2))),
                    )
                    self.session.add(new_rating)

        return len(updated_matchweeks)

    def _update_team_stats(self) -> int:
        """Update team statistics based on results."""
        # Get latest matchweek with results
        stmt = (
            select(Match.matchweek)
            .where(Match.season == settings.current_season)
            .where(Match.status == MatchStatus.FINISHED)
            .order_by(Match.matchweek.desc())
            .limit(1)
        )
        result = self.session.execute(stmt)
        latest_mw = result.scalar_one_or_none()

        if not latest_mw:
            return 0

        # Calculate stats for each team
        teams = list(self.session.execute(select(Team)).scalars().all())
        updated = 0

        for team in teams:
            stats = self._calculate_team_stats(team.id, latest_mw)
            if stats:
                updated += 1

        return updated

    def _calculate_team_stats(self, team_id: int, matchweek: int) -> Optional[TeamStats]:
        """Calculate statistics for a team up to a matchweek."""
        # Get all matches for team up to this matchweek
        stmt = (
            select(Match)
            .where(Match.season == settings.current_season)
            .where(Match.matchweek <= matchweek)
            .where(Match.status == MatchStatus.FINISHED)
            .where((Match.home_team_id == team_id) | (Match.away_team_id == team_id))
            .order_by(Match.matchweek.desc())
        )
        matches = list(self.session.execute(stmt).scalars().all())

        if not matches:
            return None

        # Calculate statistics
        goals_scored = 0
        goals_conceded = 0
        home_wins, home_draws, home_losses = 0, 0, 0
        away_wins, away_draws, away_losses = 0, 0, 0
        clean_sheets = 0
        failed_to_score = 0
        xg_for = Decimal("0")
        xg_against = Decimal("0")
        form_results = []

        for match in matches:
            is_home = match.home_team_id == team_id
            team_goals = match.home_score if is_home else match.away_score
            opp_goals = match.away_score if is_home else match.home_score

            if team_goals is None or opp_goals is None:
                continue

            goals_scored += team_goals
            goals_conceded += opp_goals

            if team_goals > opp_goals:
                result = "W"
                if is_home:
                    home_wins += 1
                else:
                    away_wins += 1
            elif team_goals < opp_goals:
                result = "L"
                if is_home:
                    home_losses += 1
                else:
                    away_losses += 1
            else:
                result = "D"
                if is_home:
                    home_draws += 1
                else:
                    away_draws += 1

            if opp_goals == 0:
                clean_sheets += 1
            if team_goals == 0:
                failed_to_score += 1

            # xG
            team_xg = match.home_xg if is_home else match.away_xg
            opp_xg = match.away_xg if is_home else match.home_xg
            if team_xg:
                xg_for += team_xg
            if opp_xg:
                xg_against += opp_xg

            if len(form_results) < 5:
                form_results.append(result)

        total_games = len([m for m in matches if m.home_score is not None])
        form = "".join(form_results)
        form_points = sum(3 if r == "W" else 1 if r == "D" else 0 for r in form_results)

        # Check for existing stats
        existing = self.session.execute(
            select(TeamStats)
            .where(TeamStats.team_id == team_id)
            .where(TeamStats.season == settings.current_season)
            .where(TeamStats.matchweek == matchweek)
        ).scalar_one_or_none()

        if existing:
            stats = existing
        else:
            stats = TeamStats(
                team_id=team_id,
                season=settings.current_season,
                matchweek=matchweek,
            )

        stats.form = form
        stats.form_points = form_points
        stats.goals_scored = goals_scored
        stats.goals_conceded = goals_conceded
        stats.avg_goals_scored = Decimal(str(round(goals_scored / total_games, 2))) if total_games else None
        stats.avg_goals_conceded = Decimal(str(round(goals_conceded / total_games, 2))) if total_games else None
        stats.home_wins = home_wins
        stats.home_draws = home_draws
        stats.home_losses = home_losses
        stats.away_wins = away_wins
        stats.away_draws = away_draws
        stats.away_losses = away_losses
        stats.clean_sheets = clean_sheets
        stats.failed_to_score = failed_to_score
        stats.xg_for = xg_for
        stats.xg_against = xg_against
        stats.avg_xg_for = Decimal(str(round(float(xg_for) / total_games, 2))) if total_games and xg_for else None
        stats.avg_xg_against = Decimal(str(round(float(xg_against) / total_games, 2))) if total_games and xg_against else None

        if not existing:
            self.session.add(stats)

        return stats

    def _resolve_value_bets(self) -> int:
        """Update results for value bets on finished matches."""
        stmt = (
            select(ValueBet)
            .join(Match)
            .where(ValueBet.is_active == True)
            .where(Match.status == MatchStatus.FINISHED)
        )
        active_bets = list(self.session.execute(stmt).scalars().all())

        resolved = 0
        for bet in active_bets:
            match = self.session.get(Match, bet.match_id)
            if not match or match.home_score is None:
                continue

            # Determine actual outcome
            if match.home_score > match.away_score:
                actual = "home_win"
            elif match.home_score < match.away_score:
                actual = "away_win"
            else:
                actual = "draw"

            total_goals = match.home_score + match.away_score
            btts = match.home_score > 0 and match.away_score > 0

            # Check if bet won
            won = False
            if bet.outcome == actual:
                won = True
            elif bet.outcome == "over_2_5" and total_goals > 2.5:
                won = True
            elif bet.outcome == "under_2_5" and total_goals < 2.5:
                won = True
            elif bet.outcome == "btts_yes" and btts:
                won = True
            elif bet.outcome == "btts_no" and not btts:
                won = True

            bet.is_active = False
            bet.result = "won" if won else "lost"
            bet.profit_loss = (
                Decimal(str(float(bet.recommended_stake) * (float(bet.odds) - 1)))
                if won
                else -bet.recommended_stake
            )
            resolved += 1

        return resolved


def run_results_update():
    """Entry point for results update job."""
    with SyncSessionLocal() as session:
        job = ResultsUpdateJob(session)
        return job.run()


if __name__ == "__main__":
    result = run_results_update()
    print(f"Job completed: {result}")
