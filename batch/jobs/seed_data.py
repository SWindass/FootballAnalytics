"""Seed script to populate initial data from APIs."""

import argparse
import asyncio

import structlog
from sqlalchemy import select

from app.core.config import get_settings
from app.db.database import SyncSessionLocal
from app.db.models import Match, Team
from batch.data_sources.football_data_org import FootballDataClient, parse_match, parse_team

logger = structlog.get_logger()
settings = get_settings()


async def seed_teams(session, season: str = "2024") -> int:
    """Fetch and seed EPL teams for a given season.

    Args:
        session: Database session
        season: Season year (e.g., "2024" for 2024-25)
    """
    client = FootballDataClient()

    logger.info(f"Fetching teams from football-data.org for {season} season...")

    # Get teams from matches for the season (includes promoted teams)
    matches_data = await client.get_matches(season=season)

    # Extract unique teams from matches
    team_ids = set()
    for match in matches_data:
        team_ids.add(match["homeTeam"]["id"])
        team_ids.add(match["awayTeam"]["id"])

    # Also get current teams list for metadata
    teams_data = await client.get_teams()
    teams_by_id = {t["id"]: t for t in teams_data}

    created = 0
    for team_id in team_ids:
        # Check if team exists
        existing = session.execute(
            select(Team).where(Team.external_id == team_id)
        ).scalar_one_or_none()

        if existing:
            continue

        # Get team data - try from current season, fallback to match data
        team_data = teams_by_id.get(team_id)
        if team_data:
            parsed = parse_team(team_data)
        else:
            # Team not in current season (relegated), find from match data
            for match in matches_data:
                if match["homeTeam"]["id"] == team_id:
                    team_info = match["homeTeam"]
                    break
                elif match["awayTeam"]["id"] == team_id:
                    team_info = match["awayTeam"]
                    break
            else:
                continue

            parsed = {
                "external_id": team_info["id"],
                "name": team_info["name"],
                "short_name": team_info.get("shortName", team_info["name"]),
                "tla": team_info.get("tla", team_info["name"][:3].upper()),
                "crest_url": team_info.get("crest"),
            }

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


async def seed_matches(session, season: str = "2024") -> int:
    """Fetch and seed EPL matches for a season.

    Args:
        session: Database session
        season: Season year (e.g., "2024" for 2024-25, "2023" for 2023-24)
    """
    client = FootballDataClient()

    # Build team lookup
    teams = session.execute(select(Team)).scalars().all()
    team_lookup = {t.external_id: t.id for t in teams}

    if not team_lookup:
        logger.error("No teams in database. Run seed_teams first.")
        return 0

    logger.info(f"Fetching matches from football-data.org for {season} season...")
    matches_data = await client.get_matches(season=season)

    created = 0
    skipped_no_team = 0
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
            skipped_no_team += 1
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
    logger.info(f"Added {created} matches (skipped {skipped_no_team} with unknown teams)")
    return created


async def run_seed(seasons: list[str] = None):
    """Run the seed process for specified seasons.

    Args:
        seasons: List of season years (e.g., ["2023", "2024", "2025"])
    """
    if seasons is None:
        seasons = ["2024"]

    with SyncSessionLocal() as session:
        print("=" * 50)
        print("Seeding FootballAnalytics Database")
        print("=" * 50)

        total_teams = 0
        total_matches = 0

        for season in seasons:
            season_name = f"{season}-{str(int(season)+1)[-2:]}"
            print(f"\n--- Season {season_name} ---")

            # Seed teams for this season
            print(f"1. Fetching teams...")
            teams_created = await seed_teams(session, season)
            print(f"   Created {teams_created} teams")
            total_teams += teams_created

            # Seed matches
            print(f"2. Fetching matches...")
            matches_created = await seed_matches(session, season)
            print(f"   Created {matches_created} matches")
            total_matches += matches_created

        # Summary
        team_count = len(session.execute(select(Team)).scalars().all())
        match_count = len(session.execute(select(Match)).scalars().all())

        print("\n" + "=" * 50)
        print("Seed Complete!")
        print(f"  New teams added: {total_teams}")
        print(f"  New matches added: {total_matches}")
        print(f"  Total teams in database: {team_count}")
        print(f"  Total matches in database: {match_count}")
        print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed EPL data from football-data.org")
    parser.add_argument(
        "--seasons",
        nargs="+",
        default=["2024"],
        help="Season years to seed (e.g., 2023 2024 2025 for 23-24, 24-25, 25-26)",
    )

    args = parser.parse_args()
    asyncio.run(run_seed(args.seasons))
