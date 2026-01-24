"""Calculate ELO ratings from historical match data."""

from decimal import Decimal

import structlog
from sqlalchemy import select

from app.core.config import get_settings
from app.db.database import SyncSessionLocal
from app.db.models import EloRating, Match, MatchStatus, Team
from batch.models.elo import EloRatingSystem

logger = structlog.get_logger()
settings = get_settings()


def calculate_elo_ratings(season: str = "2024-25") -> dict:
    """Calculate ELO ratings for all finished matches in a season.

    Args:
        season: Season to calculate ratings for

    Returns:
        Summary of results
    """
    with SyncSessionLocal() as session:
        # Get all finished matches in chronological order
        stmt = (
            select(Match)
            .where(Match.season == season)
            .where(Match.status == MatchStatus.FINISHED)
            .order_by(Match.matchweek, Match.kickoff_time)
        )
        matches = list(session.execute(stmt).scalars().all())

        if not matches:
            print(f"No finished matches found for season {season}")
            return {"status": "no_matches"}

        print(f"Processing {len(matches)} matches for season {season}")

        # Initialize ELO system
        elo = EloRatingSystem()

        # Track ratings per matchweek for each team
        matchweek_ratings: dict[int, dict[int, float]] = {}  # matchweek -> {team_id -> rating}

        for match in matches:
            if match.home_score is None or match.away_score is None:
                continue

            # Update ratings
            elo.update_ratings(
                match.home_team_id,
                match.away_team_id,
                match.home_score,
                match.away_score,
            )

            # Store ratings after this matchweek
            if match.matchweek not in matchweek_ratings:
                matchweek_ratings[match.matchweek] = {}

            # Copy current ratings for both teams
            matchweek_ratings[match.matchweek][match.home_team_id] = elo.get_rating(match.home_team_id)
            matchweek_ratings[match.matchweek][match.away_team_id] = elo.get_rating(match.away_team_id)

        # Save to database - one rating per team per matchweek
        ratings_saved = 0
        for matchweek, team_ratings in matchweek_ratings.items():
            for team_id, rating in team_ratings.items():
                existing = session.execute(
                    select(EloRating)
                    .where(EloRating.team_id == team_id)
                    .where(EloRating.season == season)
                    .where(EloRating.matchweek == matchweek)
                ).scalar_one_or_none()

                if existing:
                    existing.rating = Decimal(str(round(rating, 2)))
                else:
                    new_rating = EloRating(
                        team_id=team_id,
                        season=season,
                        matchweek=matchweek,
                        rating=Decimal(str(round(rating, 2))),
                    )
                    session.add(new_rating)
                    ratings_saved += 1

        session.commit()

        # Print final standings
        print("\n" + "=" * 60)
        print(f"Final ELO Ratings - {season} Season")
        print("=" * 60)

        # Get team names
        teams = {t.id: t.name for t in session.execute(select(Team)).scalars().all()}

        # Sort by final rating
        final_ratings = [(team_id, rating) for team_id, rating in elo.ratings.items()]
        final_ratings.sort(key=lambda x: x[1], reverse=True)

        for rank, (team_id, rating) in enumerate(final_ratings, 1):
            team_name = teams.get(team_id, f"Team {team_id}")
            change = rating - 1500  # Change from initial
            sign = "+" if change >= 0 else ""
            print(f"{rank:2}. {team_name:<30} {rating:7.1f} ({sign}{change:.1f})")

        print("=" * 60)
        print(f"\nSaved {ratings_saved} new ELO ratings to database")

        return {
            "status": "success",
            "matches_processed": len(matches),
            "ratings_saved": ratings_saved,
            "matchweeks": len(matchweek_ratings),
        }


if __name__ == "__main__":
    result = calculate_elo_ratings()
    print(f"\nResult: {result}")
