"""Sync all matches to TeamFixture table for rest day calculations.

This job ensures TeamFixture contains ALL fixtures (EPL + European)
for accurate rest day calculations.
"""

import argparse
from datetime import datetime
from typing import Optional

import structlog
from sqlalchemy import select, and_
from sqlalchemy.orm import Session

from app.db.database import SyncSessionLocal
from app.db.models import Match, Team, TeamFixture

logger = structlog.get_logger()


def sync_matches_to_fixtures(session: Session) -> dict:
    """Sync all matches from Match table to TeamFixture table.

    Returns:
        Summary of synced fixtures
    """
    logger.info("Starting match to fixture sync")

    # Get all matches
    matches = list(session.execute(select(Match)).scalars().all())

    # Get teams for name lookup
    teams = {t.id: t for t in session.execute(select(Team)).scalars().all()}

    created = 0
    skipped = 0

    for match in matches:
        home_team = teams.get(match.home_team_id)
        away_team = teams.get(match.away_team_id)

        if not home_team or not away_team:
            continue

        # Create fixture for home team
        if _create_fixture_if_not_exists(
            session,
            team_id=match.home_team_id,
            kickoff_time=match.kickoff_time,
            competition=match.competition or "PL",
            opponent_name=away_team.name,
            is_home=True,
            match_id=match.id,
        ):
            created += 1
        else:
            skipped += 1

        # Create fixture for away team
        if _create_fixture_if_not_exists(
            session,
            team_id=match.away_team_id,
            kickoff_time=match.kickoff_time,
            competition=match.competition or "PL",
            opponent_name=home_team.name,
            is_home=False,
            match_id=match.id,
        ):
            created += 1
        else:
            skipped += 1

    session.commit()

    logger.info(
        "Match to fixture sync completed",
        fixtures_created=created,
        fixtures_skipped=skipped,
    )

    return {
        "status": "success",
        "fixtures_created": created,
        "fixtures_skipped": skipped,
    }


def _create_fixture_if_not_exists(
    session: Session,
    team_id: int,
    kickoff_time: datetime,
    competition: str,
    opponent_name: str,
    is_home: bool,
    match_id: Optional[int] = None,
) -> bool:
    """Create a TeamFixture record if it doesn't exist.

    Returns:
        True if created, False if already exists
    """
    # Check if fixture already exists
    existing = session.execute(
        select(TeamFixture).where(
            and_(
                TeamFixture.team_id == team_id,
                TeamFixture.kickoff_time == kickoff_time,
                TeamFixture.competition == competition,
            )
        )
    ).scalar_one_or_none()

    if existing:
        # Update match_id if missing
        if existing.match_id is None and match_id is not None:
            existing.match_id = match_id
        return False

    fixture = TeamFixture(
        team_id=team_id,
        kickoff_time=kickoff_time,
        competition=competition,
        opponent_name=opponent_name,
        is_home=is_home,
        match_id=match_id,
    )
    session.add(fixture)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Sync matches to TeamFixture table"
    )
    args = parser.parse_args()

    # Suppress SQLAlchemy logging
    import logging
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

    print("Match to Fixture Sync")
    print("=" * 40)

    with SyncSessionLocal() as session:
        result = sync_matches_to_fixtures(session)

    print(f"\nSync complete!")
    print(f"  Fixtures created: {result['fixtures_created']}")
    print(f"  Fixtures skipped: {result['fixtures_skipped']}")


if __name__ == "__main__":
    main()
