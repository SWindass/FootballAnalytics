"""Fetch European competition matches for EPL teams.

This job fetches Champions League (and Europa League when available) matches
for EPL teams to enable accurate rest day calculations.

Uses TeamFixture table to track fixtures even when opponent isn't in our DB.
"""

import argparse
import asyncio
from datetime import datetime

import structlog
from sqlalchemy import and_, select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.database import SyncSessionLocal
from app.db.models import Team, TeamFixture
from batch.data_sources.football_data_org import FootballDataClient

logger = structlog.get_logger()
settings = get_settings()


class EuropeanMatchesFetcher:
    """Fetches European competition matches for EPL teams.

    Populates TeamFixture table for rest day calculations.
    """

    def __init__(self, session: Session | None = None):
        self.session = session or SyncSessionLocal()
        self.client = FootballDataClient()

    def run(self, season: str = None, competitions: list[str] = None) -> dict:
        """Fetch European matches.

        Args:
            season: Season to fetch (e.g., "2024" for 2024-25)
            competitions: List of competition codes (default: ["CL"])

        Returns:
            Summary of fetched fixtures
        """
        logger.info("Starting European matches fetch")
        start_time = datetime.utcnow()

        if season is None:
            # Extract year from current season
            season = settings.current_season.split("-")[0]

        if competitions is None:
            competitions = ["CL"]  # Champions League only (free tier)

        # Get EPL teams with their external IDs
        teams = self._get_epl_teams()
        team_ext_ids = {t.external_id: t for t in teams}

        logger.info(f"Fetching matches for {len(teams)} EPL teams")

        total_fetched = 0
        total_created = 0

        for comp in competitions:
            fetched, created = asyncio.run(
                self._fetch_competition_fixtures(comp, season, team_ext_ids)
            )
            total_fetched += fetched
            total_created += created

        self.session.commit()

        duration = (datetime.utcnow() - start_time).total_seconds()
        logger.info(
            "European matches fetch completed",
            fixtures_fetched=total_fetched,
            fixtures_created=total_created,
            duration_seconds=round(duration, 1),
        )

        return {
            "status": "success",
            "fixtures_fetched": total_fetched,
            "fixtures_created": total_created,
            "duration_seconds": duration,
        }

    def _get_epl_teams(self) -> list[Team]:
        """Get all EPL teams."""
        return list(self.session.execute(select(Team)).scalars().all())

    async def _fetch_competition_fixtures(
        self,
        competition: str,
        season: str,
        team_ext_ids: dict[int, Team],
    ) -> tuple[int, int]:
        """Fetch fixtures for a competition.

        Returns:
            Tuple of (fixtures_fetched, fixtures_created)
        """
        logger.info(f"Fetching {competition} fixtures for season {season}")

        try:
            matches = await self.client.get_competition_matches(
                competition=competition,
                season=season,
            )
        except Exception as e:
            logger.warning(f"Failed to fetch {competition}: {e}")
            return 0, 0

        fetched = 0
        created = 0

        for match_data in matches:
            home_ext_id = match_data.get("homeTeam", {}).get("id")
            away_ext_id = match_data.get("awayTeam", {}).get("id")
            home_team_name = match_data.get("homeTeam", {}).get("name", "Unknown")
            away_team_name = match_data.get("awayTeam", {}).get("name", "Unknown")

            # Get EPL teams involved (if any)
            home_team = team_ext_ids.get(home_ext_id)
            away_team = team_ext_ids.get(away_ext_id)

            if not home_team and not away_team:
                # Neither team is in EPL - skip
                continue

            # Parse kickoff time
            utc_date_str = match_data.get("utcDate", "")
            try:
                kickoff_time = datetime.fromisoformat(utc_date_str.replace("Z", "+00:00"))
            except ValueError:
                logger.warning(f"Invalid date format: {utc_date_str}")
                continue

            fetched += 1

            # Create fixture for home team if EPL
            if home_team:
                fixture_created = self._create_fixture(
                    team=home_team,
                    kickoff_time=kickoff_time,
                    competition=competition,
                    opponent_name=away_team_name,
                    is_home=True,
                )
                if fixture_created:
                    created += 1

            # Create fixture for away team if EPL
            if away_team:
                fixture_created = self._create_fixture(
                    team=away_team,
                    kickoff_time=kickoff_time,
                    competition=competition,
                    opponent_name=home_team_name,
                    is_home=False,
                )
                if fixture_created:
                    created += 1

        logger.info(f"{competition}: matches_fetched={fetched}, fixtures_created={created}")
        return fetched, created

    def _create_fixture(
        self,
        team: Team,
        kickoff_time: datetime,
        competition: str,
        opponent_name: str,
        is_home: bool,
    ) -> bool:
        """Create a TeamFixture record if it doesn't exist.

        Returns:
            True if created, False if already exists
        """
        # Check if fixture already exists
        existing = self.session.execute(
            select(TeamFixture).where(
                and_(
                    TeamFixture.team_id == team.id,
                    TeamFixture.kickoff_time == kickoff_time,
                    TeamFixture.competition == competition,
                )
            )
        ).scalar_one_or_none()

        if existing:
            return False

        fixture = TeamFixture(
            team_id=team.id,
            kickoff_time=kickoff_time,
            competition=competition,
            opponent_name=opponent_name,
            is_home=is_home,
        )
        self.session.add(fixture)
        return True


def run_european_fixtures_fetch(season: str = None, competitions: list[str] = None):
    """Entry point for European fixtures fetch."""
    with SyncSessionLocal() as session:
        job = EuropeanMatchesFetcher(session)
        return job.run(season=season, competitions=competitions)


def main():
    parser = argparse.ArgumentParser(
        description="Fetch European competition fixtures for EPL teams"
    )
    parser.add_argument(
        "--season",
        help="Season year (e.g., 2024 for 2024-25)",
    )
    parser.add_argument(
        "--competitions",
        nargs="+",
        default=["CL"],
        help="Competition codes to fetch (default: CL)",
    )
    parser.add_argument(
        "--all-seasons",
        action="store_true",
        help="Fetch all recent seasons (2020-present)",
    )

    args = parser.parse_args()

    # Suppress SQLAlchemy logging
    import logging
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

    print("European Fixtures Fetcher")
    print("=" * 40)

    if args.all_seasons:
        # Fetch multiple seasons
        seasons = ["2020", "2021", "2022", "2023", "2024", "2025"]
        total_created = 0
        for season in seasons:
            print(f"\nFetching season {season}...")
            result = run_european_fixtures_fetch(
                season=season,
                competitions=args.competitions,
            )
            total_created += result["fixtures_created"]
            print(f"  Created: {result['fixtures_created']}")
        print(f"\nTotal fixtures created: {total_created}")
    else:
        result = run_european_fixtures_fetch(
            season=args.season,
            competitions=args.competitions,
        )
        print("\nFetch complete!")
        print(f"  Fixtures fetched: {result['fixtures_fetched']}")
        print(f"  Fixtures created: {result['fixtures_created']}")
        print(f"  Duration: {result['duration_seconds']:.1f}s")


if __name__ == "__main__":
    main()
