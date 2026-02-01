"""Update team context data - injuries, managers, etc.

This job runs before weekly analysis to ensure all contextual data is fresh.
"""

import asyncio
from datetime import datetime

import structlog
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.database import SyncSessionLocal
from app.db.models import Manager, ManagerTenure, Match, MatchStatus, Referee, Team, TeamStats
from batch.data_sources.football_data_org import FootballDataClient, parse_match
from batch.data_sources.injuries import get_injuries_for_team

logger = structlog.get_logger()
settings = get_settings()


class TeamContextUpdateJob:
    """Updates team context data including injuries and manager info."""

    def __init__(self, session: Session | None = None):
        self.session = session or SyncSessionLocal()
        self.football_client = FootballDataClient()

    def run(self) -> dict:
        """Execute the team context update job."""
        logger.info("Starting team context update job")
        start_time = datetime.utcnow()

        try:
            # 1. Update injury data for all teams
            injuries_updated = asyncio.run(self._update_injuries())

            # 2. Update manager data from API
            managers_updated = asyncio.run(self._update_managers())

            # 3. Update referee data from recent matches
            referees_updated = asyncio.run(self._update_referees())

            self.session.commit()

            duration = (datetime.utcnow() - start_time).total_seconds()
            logger.info(
                "Team context update completed",
                injuries_updated=injuries_updated,
                managers_updated=managers_updated,
                referees_updated=referees_updated,
                duration_seconds=duration,
            )

            return {
                "status": "success",
                "injuries_updated": injuries_updated,
                "managers_updated": managers_updated,
                "referees_updated": referees_updated,
                "duration_seconds": duration,
            }

        except Exception as e:
            logger.error("Team context update failed", error=str(e))
            self.session.rollback()
            raise

    async def _update_injuries(self) -> int:
        """Update injury data for all teams."""
        teams = list(self.session.execute(select(Team)).scalars().all())
        updated = 0

        # Get current matchweek
        stmt = (
            select(Match.matchweek)
            .where(Match.season == settings.current_season)
            .where(Match.status == MatchStatus.SCHEDULED)
            .order_by(Match.kickoff_time)
            .limit(1)
        )
        current_mw = self.session.execute(stmt).scalar_one_or_none() or 1

        for team in teams:
            try:
                injury_data = await get_injuries_for_team(team.name)

                # Find or create team stats for current matchweek
                stats = self.session.execute(
                    select(TeamStats)
                    .where(TeamStats.team_id == team.id)
                    .where(TeamStats.season == settings.current_season)
                    .where(TeamStats.matchweek == current_mw - 1)
                ).scalar_one_or_none()

                if stats:
                    stats.injuries = injury_data
                    stats.injury_count = injury_data.get("count", 0)
                    stats.key_players_out = injury_data.get("key_players_out", 0)
                    updated += 1
                    logger.debug(f"Updated injuries for {team.short_name}: {injury_data['count']} injured")

            except Exception as e:
                logger.warning(f"Failed to update injuries for {team.name}: {e}")

        return updated

    async def _update_managers(self) -> int:
        """Update manager data from football-data.org API."""
        teams = list(self.session.execute(select(Team)).scalars().all())
        updated = 0

        # Get current matchweek
        stmt = (
            select(Match.matchweek)
            .where(Match.season == settings.current_season)
            .where(Match.status == MatchStatus.SCHEDULED)
            .order_by(Match.kickoff_time)
            .limit(1)
        )
        current_mw = self.session.execute(stmt).scalar_one_or_none() or 1

        # Fetch recent matches which include coach data
        try:
            matches = await self.football_client.get_matches(
                season=settings.current_season.split("-")[0],
                status="FINISHED",
            )

            # Build team -> coach mapping from recent data
            team_coaches = {}
            for match_data in matches:
                parsed = parse_match(match_data)

                if parsed.get("home_coach"):
                    home_team_ext_id = parsed["home_team_external_id"]
                    team_coaches[home_team_ext_id] = parsed["home_coach"]

                if parsed.get("away_coach"):
                    away_team_ext_id = parsed["away_team_external_id"]
                    team_coaches[away_team_ext_id] = parsed["away_coach"]

            # Update manager records
            for team in teams:
                if team.external_id not in team_coaches:
                    continue

                coach_data = team_coaches[team.external_id]

                # Find or create manager
                manager = self.session.execute(
                    select(Manager).where(Manager.external_id == coach_data["external_id"])
                ).scalar_one_or_none()

                if not manager:
                    manager = Manager(
                        external_id=coach_data["external_id"],
                        name=coach_data["name"],
                        nationality=coach_data.get("nationality"),
                    )
                    self.session.add(manager)
                    self.session.flush()
                    logger.info(f"Created new manager: {coach_data['name']}")

                # Check/update tenure
                current_tenure = self.session.execute(
                    select(ManagerTenure)
                    .where(ManagerTenure.team_id == team.id)
                    .where(ManagerTenure.end_date.is_(None))
                ).scalar_one_or_none()

                if current_tenure and current_tenure.manager_id != manager.id:
                    # Manager changed!
                    current_tenure.end_date = datetime.utcnow()
                    logger.info(f"Manager change at {team.short_name}: ended tenure")

                    # Create new tenure
                    new_tenure = ManagerTenure(
                        manager_id=manager.id,
                        team_id=team.id,
                        start_date=datetime.utcnow(),
                    )
                    self.session.add(new_tenure)
                    updated += 1

                elif not current_tenure:
                    # No current tenure - create one
                    new_tenure = ManagerTenure(
                        manager_id=manager.id,
                        team_id=team.id,
                        start_date=datetime.utcnow(),
                    )
                    self.session.add(new_tenure)
                    updated += 1

                # Update team stats with manager games
                stats = self.session.execute(
                    select(TeamStats)
                    .where(TeamStats.team_id == team.id)
                    .where(TeamStats.season == settings.current_season)
                    .where(TeamStats.matchweek == current_mw - 1)
                ).scalar_one_or_none()

                if stats:
                    tenure = self.session.execute(
                        select(ManagerTenure)
                        .where(ManagerTenure.team_id == team.id)
                        .where(ManagerTenure.end_date.is_(None))
                    ).scalar_one_or_none()

                    if tenure:
                        stats.manager_games = tenure.matches_managed
                        stats.is_new_manager = tenure.matches_managed < 5

        except Exception as e:
            logger.warning(f"Failed to update managers: {e}")

        return updated

    async def _update_referees(self) -> int:
        """Update referee data from recent matches."""
        updated = 0

        try:
            # Fetch recent matches with referee data
            matches = await self.football_client.get_matches(
                season=settings.current_season.split("-")[0],
            )

            for match_data in matches:
                parsed = parse_match(match_data)

                if not parsed.get("referee"):
                    continue

                ref_data = parsed["referee"]

                # Find or create referee
                referee = self.session.execute(
                    select(Referee).where(Referee.external_id == ref_data["external_id"])
                ).scalar_one_or_none()

                if not referee:
                    referee = Referee(
                        external_id=ref_data["external_id"],
                        name=ref_data["name"],
                        nationality=ref_data.get("nationality"),
                    )
                    self.session.add(referee)
                    self.session.flush()
                    logger.info(f"Created new referee: {ref_data['name']}")
                    updated += 1

                # Update match with referee_id
                match = self.session.execute(
                    select(Match).where(Match.external_id == parsed["external_id"])
                ).scalar_one_or_none()

                if match and not match.referee_id:
                    match.referee_id = referee.id

        except Exception as e:
            logger.warning(f"Failed to update referees: {e}")

        # Calculate referee statistics
        self._calculate_referee_stats()

        return updated

    def _calculate_referee_stats(self):
        """Calculate referee statistics from historical matches."""
        referees = list(self.session.execute(select(Referee)).scalars().all())

        for referee in referees:
            # Get all matches officiated by this referee
            matches = list(self.session.execute(
                select(Match)
                .where(Match.referee_id == referee.id)
                .where(Match.status == MatchStatus.FINISHED)
            ).scalars().all())

            if not matches:
                continue

            total = len(matches)
            home_wins = sum(1 for m in matches if m.home_score > m.away_score)

            referee.matches_officiated = total
            referee.home_win_pct = round((home_wins / total) * 100, 2) if total > 0 else None

            logger.debug(f"Referee {referee.name}: {total} matches, {referee.home_win_pct}% home wins")


def run_team_context_update():
    """Entry point for team context update job."""
    with SyncSessionLocal() as session:
        job = TeamContextUpdateJob(session)
        return job.run()


if __name__ == "__main__":
    result = run_team_context_update()
    print(f"Job completed: {result}")
