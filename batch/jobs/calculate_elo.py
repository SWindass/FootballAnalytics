"""Calculate ELO ratings from historical match data.

Supports multi-season calculation with ratings carrying over between seasons
(with configurable regression toward the mean).
"""

import argparse
from decimal import Decimal
from typing import Optional

import structlog
from sqlalchemy import select

from app.core.config import get_settings
from app.db.database import SyncSessionLocal
from app.db.models import EloRating, Match, MatchStatus, Team
from batch.models.elo import EloRatingSystem

logger = structlog.get_logger()
settings = get_settings()


def get_previous_season(season: str) -> Optional[str]:
    """Get the previous season string (e.g., '2024-25' -> '2023-24')."""
    try:
        start_year = int(season.split("-")[0])
        return f"{start_year - 1}-{str(start_year)[-2:]}"
    except (ValueError, IndexError):
        return None


def calculate_elo_ratings(
    season: str = "2024-25",
    regression_factor: float = 0.33,
    use_previous_season: bool = True,
) -> dict:
    """Calculate ELO ratings for all finished matches in a season.

    Args:
        season: Season to calculate ratings for
        regression_factor: How much to regress toward 1500 at season start (0-1).
                          0 = keep previous rating, 1 = reset to 1500
        use_previous_season: Whether to use previous season's final ratings

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

        # Try to load previous season's final ratings
        initial_ratings = {}
        if use_previous_season:
            prev_season = get_previous_season(season)
            if prev_season:
                print(f"Looking for previous season ratings ({prev_season})...")
                # Get each team's latest rating from previous season
                # (not all teams have the same max matchweek)
                from sqlalchemy import func

                # Subquery to find max matchweek per team
                max_mw_subq = (
                    select(
                        EloRating.team_id,
                        func.max(EloRating.matchweek).label("max_mw")
                    )
                    .where(EloRating.season == prev_season)
                    .group_by(EloRating.team_id)
                    .subquery()
                )

                # Join to get ratings at each team's max matchweek
                stmt = (
                    select(EloRating)
                    .join(
                        max_mw_subq,
                        (EloRating.team_id == max_mw_subq.c.team_id) &
                        (EloRating.matchweek == max_mw_subq.c.max_mw)
                    )
                    .where(EloRating.season == prev_season)
                )
                prev_ratings = list(session.execute(stmt).scalars().all())

                if prev_ratings:
                    print(f"Found {len(prev_ratings)} teams from {prev_season}")
                    for r in prev_ratings:
                        # Apply regression toward mean
                        prev_rating = float(r.rating)
                        regressed = prev_rating + regression_factor * (1500 - prev_rating)
                        initial_ratings[r.team_id] = regressed
                        elo.set_rating(r.team_id, regressed)

                    print(f"Applied {regression_factor:.0%} regression toward 1500")
                else:
                    print(f"No previous season data found, starting from 1500")
            else:
                print("Could not determine previous season, starting from 1500")

        # Get all teams in this season (both home and away)
        season_team_ids = set()
        for match in matches:
            season_team_ids.add(match.home_team_id)
            season_team_ids.add(match.away_team_id)

        # Initialize ratings for teams that don't have previous season data
        for team_id in season_team_ids:
            if team_id not in elo.ratings:
                elo.set_rating(team_id, 1500)

        # Group matches by matchweek
        matches_by_mw: dict[int, list] = {}
        for match in matches:
            if match.matchweek not in matches_by_mw:
                matches_by_mw[match.matchweek] = []
            matches_by_mw[match.matchweek].append(match)

        # Track ratings per matchweek for ALL teams
        matchweek_ratings: dict[int, dict[int, float]] = {}  # matchweek -> {team_id -> rating}
        previous_ratings: dict[int, float] = {}  # Track previous matchweek ratings for change calculation

        for matchweek in sorted(matches_by_mw.keys()):
            mw_matches = matches_by_mw[matchweek]

            # Process all matches in this matchweek
            for match in mw_matches:
                if match.home_score is None or match.away_score is None:
                    continue

                # Update ratings
                elo.update_ratings(
                    match.home_team_id,
                    match.away_team_id,
                    match.home_score,
                    match.away_score,
                )

            # Store ratings for ALL teams after this matchweek
            matchweek_ratings[matchweek] = {}
            for team_id in season_team_ids:
                matchweek_ratings[matchweek][team_id] = elo.get_rating(team_id)

        # Save to database - one rating per team per matchweek
        ratings_saved = 0
        sorted_matchweeks = sorted(matchweek_ratings.keys())

        for matchweek in sorted_matchweeks:
            team_ratings = matchweek_ratings[matchweek]

            # Get previous matchweek's ratings for calculating change
            prev_mw_idx = sorted_matchweeks.index(matchweek) - 1
            if prev_mw_idx >= 0:
                prev_mw = sorted_matchweeks[prev_mw_idx]
                prev_ratings = matchweek_ratings.get(prev_mw, {})
            else:
                # First matchweek - use initial ratings
                prev_ratings = initial_ratings

            for team_id, rating in team_ratings.items():
                # Calculate change from previous matchweek
                prev_rating = prev_ratings.get(team_id, 1500)
                rating_change = rating - prev_rating

                existing = session.execute(
                    select(EloRating)
                    .where(EloRating.team_id == team_id)
                    .where(EloRating.season == season)
                    .where(EloRating.matchweek == matchweek)
                ).scalar_one_or_none()

                if existing:
                    existing.rating = Decimal(str(round(rating, 2)))
                    existing.rating_change = Decimal(str(round(rating_change, 2)))
                else:
                    new_rating = EloRating(
                        team_id=team_id,
                        season=season,
                        matchweek=matchweek,
                        rating=Decimal(str(round(rating, 2))),
                        rating_change=Decimal(str(round(rating_change, 2))),
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
            start_rating = initial_ratings.get(team_id, 1500)
            change = rating - start_rating
            sign = "+" if change >= 0 else ""
            print(f"{rank:2}. {team_name:<30} {rating:7.1f} ({sign}{change:.1f})")

        print("=" * 60)
        print(f"\nSaved {ratings_saved} new ELO ratings to database")

        return {
            "status": "success",
            "matches_processed": len(matches),
            "ratings_saved": ratings_saved,
            "matchweeks": len(matchweek_ratings),
            "used_previous_season": bool(initial_ratings),
        }


def calculate_all_seasons(regression_factor: float = 0.33) -> dict:
    """Calculate ELO ratings for all seasons in chronological order.

    This ensures proper carry-over between seasons.
    """
    with SyncSessionLocal() as session:
        # Get all unique seasons
        stmt = select(Match.season).distinct().order_by(Match.season)
        seasons = [s for (s,) in session.execute(stmt).all()]

    if not seasons:
        print("No seasons found in database")
        return {"status": "no_seasons"}

    print(f"Found {len(seasons)} seasons: {', '.join(seasons)}")
    print(f"Regression factor: {regression_factor:.0%}\n")

    results = {}
    for i, season in enumerate(seasons):
        print(f"\n{'='*60}")
        print(f"Processing season {i+1}/{len(seasons)}: {season}")
        print("=" * 60)

        # First season starts fresh, subsequent seasons use previous
        use_prev = i > 0
        result = calculate_elo_ratings(
            season=season,
            regression_factor=regression_factor,
            use_previous_season=use_prev,
        )
        results[season] = result

    return {"status": "success", "seasons": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate ELO ratings")
    parser.add_argument(
        "--season",
        type=str,
        default=None,
        help="Season to calculate (e.g., '2024-25'). If not specified, calculates all seasons.",
    )
    parser.add_argument(
        "--regression",
        type=float,
        default=0.33,
        help="Regression factor toward 1500 at season start (0-1, default: 0.33)",
    )
    parser.add_argument(
        "--no-carryover",
        action="store_true",
        help="Don't use previous season ratings (start fresh at 1500)",
    )

    args = parser.parse_args()

    if args.season:
        result = calculate_elo_ratings(
            season=args.season,
            regression_factor=args.regression,
            use_previous_season=not args.no_carryover,
        )
    else:
        result = calculate_all_seasons(regression_factor=args.regression)

    print(f"\nResult: {result}")
