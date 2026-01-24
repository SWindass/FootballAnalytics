"""Fetch xG data from Understat and update matches."""

import argparse
import asyncio
from decimal import Decimal

import structlog
from sqlalchemy import select, update

from app.db.database import SyncSessionLocal
from app.db.models import Match, MatchStatus, Team
from batch.data_sources.understat import UnderstatScraper, parse_understat_match

logger = structlog.get_logger()


def convert_season_format(season: str) -> str:
    """Convert '2024-25' to '2024' for Understat."""
    return season.split("-")[0]


async def fetch_xg_for_season(season: str) -> dict:
    """Fetch xG data from Understat for a season.

    Args:
        season: Season in format '2024-25'

    Returns:
        Summary dict
    """
    understat_season = convert_season_format(season)
    print(f"Fetching xG data for {season} (Understat: {understat_season})...")

    scraper = UnderstatScraper()

    try:
        # Fetch matches from Understat
        matches_data = await scraper.get_league_matches(understat_season)
        print(f"  Found {len(matches_data)} matches on Understat")

        if not matches_data:
            return {"status": "no_data", "season": season}

        # Parse Understat data
        understat_matches = []
        for m in matches_data:
            try:
                parsed = parse_understat_match(m)
                if parsed["is_finished"] and parsed["home_xg"] is not None:
                    understat_matches.append(parsed)
            except (KeyError, ValueError) as e:
                continue

        print(f"  Parsed {len(understat_matches)} finished matches with xG")

        # Match to database fixtures and update
        updated = 0
        not_found = 0

        with SyncSessionLocal() as session:
            # Get teams for name matching
            teams = {t.name.lower(): t.id for t in session.execute(select(Team)).scalars().all()}
            # Also add short names
            for t in session.execute(select(Team)).scalars().all():
                teams[t.short_name.lower()] = t.id

            # Get all fixtures for this season
            stmt = (
                select(Match)
                .where(Match.season == season)
                .where(Match.status == MatchStatus.FINISHED)
            )
            db_matches = list(session.execute(stmt).scalars().all())
            print(f"  Found {len(db_matches)} finished matches in database")

            # Build lookup by teams and approximate date
            for us_match in understat_matches:
                # Find matching database match
                matched = False
                us_home = us_match["home_team"].lower()
                us_away = us_match["away_team"].lower()
                us_date = us_match["datetime"].date()

                for db_match in db_matches:
                    # Skip if already has xG
                    if db_match.home_xg is not None:
                        continue

                    # Get team names
                    home_team = session.get(Team, db_match.home_team_id)
                    away_team = session.get(Team, db_match.away_team_id)

                    if not home_team or not away_team:
                        continue

                    db_home = home_team.name.lower()
                    db_away = away_team.name.lower()
                    db_date = db_match.kickoff_time.date()

                    # Check if teams match (fuzzy)
                    home_match = (
                        us_home in db_home or db_home in us_home or
                        us_home in home_team.short_name.lower() or
                        home_team.short_name.lower() in us_home
                    )
                    away_match = (
                        us_away in db_away or db_away in us_away or
                        us_away in away_team.short_name.lower() or
                        away_team.short_name.lower() in us_away
                    )

                    # Check date (within 1 day)
                    date_diff = abs((us_date - db_date).days)

                    if home_match and away_match and date_diff <= 1:
                        # Update xG
                        db_match.home_xg = us_match["home_xg"]
                        db_match.away_xg = us_match["away_xg"]
                        updated += 1
                        matched = True
                        break

                if not matched:
                    not_found += 1

            session.commit()

        print(f"  Updated {updated} matches with xG data")
        print(f"  Could not match {not_found} Understat matches")

        return {
            "status": "success",
            "season": season,
            "understat_matches": len(understat_matches),
            "updated": updated,
            "not_found": not_found,
        }

    finally:
        await scraper.close()


def check_xg_coverage():
    """Print xG data coverage stats."""
    with SyncSessionLocal() as session:
        from sqlalchemy import func, text

        result = session.execute(text("""
            SELECT season,
                   COUNT(*) as total,
                   COUNT(home_xg) as with_xg
            FROM matches
            WHERE status = 'finished'
            GROUP BY season
            ORDER BY season DESC
            LIMIT 10
        """))

        print("\nxG Data Coverage (recent seasons):")
        print("-" * 40)
        for row in result:
            pct = row[2] / row[1] * 100 if row[1] > 0 else 0
            print(f"  {row[0]}: {row[2]}/{row[1]} ({pct:.0f}%)")


async def main(seasons: list[str]):
    """Fetch xG data for multiple seasons."""
    for season in seasons:
        result = await fetch_xg_for_season(season)
        print(f"Result: {result}\n")

    check_xg_coverage()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch xG data from Understat")
    parser.add_argument(
        "--seasons",
        type=str,
        default="2024-25,2025-26",
        help="Comma-separated seasons to fetch",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only check current xG coverage",
    )

    args = parser.parse_args()

    if args.check:
        check_xg_coverage()
    else:
        seasons = [s.strip() for s in args.seasons.split(",")]
        asyncio.run(main(seasons))
