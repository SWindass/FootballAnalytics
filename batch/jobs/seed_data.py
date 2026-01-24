"""Seed script to populate initial data from APIs."""

import asyncio
from datetime import datetime

import structlog
from sqlalchemy import select

from app.core.config import get_settings
from app.db.database import SyncSessionLocal
from app.db.models import Match, Team
from batch.data_sources.football_data_org import FootballDataClient, parse_match, parse_team

logger = structlog.get_logger()
settings = get_settings()


async def seed_teams(session) -> int:
    """Fetch and seed EPL teams."""
    client = FootballDataClient()

    logger.info("Fetching teams from football-data.org...")
    teams_data = await client.get_teams()

    created = 0
    for team_data in teams_data:
        parsed = parse_team(team_data)

        # Check if team exists
        existing = session.execute(
            select(Team).where(Team.external_id == parsed["external_id"])
        ).scalar_one_or_none()

        if existing:
            logger.debug(f"Team {parsed['name']} already exists")
            continue

        team = Team(
            external_id=parsed["external_id"],
            name=parsed["name"],
            short_name=parsed["short_name"],
            tla=parsed["tla"],
            crest_url=parsed.get("crest_url"),
            venue=parsed.get("venue"),
            founded=parsed.get("founded"),
        )
        session.add(team)
        created += 1
        logger.info(f"Added team: {parsed['name']}")

    session.commit()
    return created


async def seed_matches(session) -> int:
    """Fetch and seed EPL matches for current season."""
    client = FootballDataClient()

    # Build team lookup
    teams = session.execute(select(Team)).scalars().all()
    team_lookup = {t.external_id: t.id for t in teams}

    if not team_lookup:
        logger.error("No teams in database. Run seed_teams first.")
        return 0

    logger.info("Fetching matches from football-data.org...")
    matches_data = await client.get_matches(season="2024")

    created = 0
    for match_data in matches_data:
        parsed = parse_match(match_data)

        # Check if match exists
        existing = session.execute(
            select(Match).where(Match.external_id == parsed["external_id"])
        ).scalar_one_or_none()

        if existing:
            continue

        home_team_id = team_lookup.get(parsed["home_team_external_id"])
        away_team_id = team_lookup.get(parsed["away_team_external_id"])

        if not home_team_id or not away_team_id:
            logger.warning(f"Could not find teams for match {parsed['external_id']}")
            continue

        match = Match(
            external_id=parsed["external_id"],
            season=parsed["season"],
            matchweek=parsed["matchweek"],
            kickoff_time=parsed["kickoff_time"],
            status=parsed["status"],
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            home_score=parsed.get("home_score"),
            away_score=parsed.get("away_score"),
            home_ht_score=parsed.get("home_ht_score"),
            away_ht_score=parsed.get("away_ht_score"),
        )
        session.add(match)
        created += 1

    session.commit()
    logger.info(f"Added {created} matches")
    return created


async def run_seed():
    """Run the full seed process."""
    with SyncSessionLocal() as session:
        print("=" * 50)
        print("Seeding FootballAnalytics Database")
        print("=" * 50)

        # Seed teams
        print("\n1. Fetching EPL teams...")
        teams_created = await seed_teams(session)
        print(f"   Created {teams_created} teams")

        # Seed matches
        print("\n2. Fetching matches...")
        matches_created = await seed_matches(session)
        print(f"   Created {matches_created} matches")

        # Summary
        team_count = session.execute(select(Team)).scalars().all()
        match_count = session.execute(select(Match)).scalars().all()

        print("\n" + "=" * 50)
        print("Seed Complete!")
        print(f"  Teams in database: {len(team_count)}")
        print(f"  Matches in database: {len(match_count)}")
        print("=" * 50)


if __name__ == "__main__":
    asyncio.run(run_seed())
