"""Backfill historical TeamStats from match results.

Computes form, goals, home/away splits, clean sheets, etc. for each team
at each matchweek based on their historical match results.

Run with: PYTHONPATH=. python batch/jobs/backfill_team_stats.py
"""

import argparse
from collections import defaultdict
from decimal import Decimal

import structlog
from sqlalchemy import delete, select

from app.db.database import SyncSessionLocal
from app.db.models import Match, MatchStatus, Team, TeamStats

logger = structlog.get_logger()


def compute_form_string(results: list[str], max_len: int = 5) -> str:
    """Convert list of results (W/D/L) to form string.

    Args:
        results: List of 'W', 'D', 'L' from oldest to newest
        max_len: Maximum length of form string

    Returns:
        Form string like "WDLWW" (most recent last)
    """
    recent = results[-max_len:] if len(results) > max_len else results
    return "".join(recent)


def compute_form_points(form: str) -> int:
    """Calculate points from form string (W=3, D=1, L=0)."""
    points = {"W": 3, "D": 1, "L": 0}
    return sum(points.get(r, 0) for r in form)


def backfill_team_stats(
    season: str | None = None,
    dry_run: bool = False,
    clear_existing: bool = False,
) -> dict:
    """Backfill TeamStats from historical match results.

    Args:
        season: Specific season to process, or None for all seasons
        dry_run: If True, don't commit changes
        clear_existing: If True, delete existing TeamStats before backfilling

    Returns:
        Summary of records created
    """
    with SyncSessionLocal() as session:
        # Get all teams
        teams = {t.id: t for t in session.execute(select(Team)).scalars().all()}
        logger.info(f"Found {len(teams)} teams")

        # Get all finished matches
        stmt = (
            select(Match)
            .where(Match.status == MatchStatus.FINISHED)
            .where(Match.home_score.isnot(None))
            .order_by(Match.season, Match.matchweek, Match.kickoff_time)
        )
        if season:
            stmt = stmt.where(Match.season == season)

        matches = list(session.execute(stmt).scalars().all())
        logger.info(f"Found {len(matches)} finished matches")

        if not matches:
            return {"created": 0, "seasons": []}

        # Optionally clear existing stats
        if clear_existing and not dry_run:
            if season:
                session.execute(delete(TeamStats).where(TeamStats.season == season))
            else:
                session.execute(delete(TeamStats))
            session.commit()
            logger.info("Cleared existing TeamStats")

        # Group matches by season
        matches_by_season = defaultdict(list)
        for m in matches:
            matches_by_season[m.season].append(m)

        # Process each season
        total_created = 0
        seasons_processed = []

        for season_name in sorted(matches_by_season.keys()):
            season_matches = matches_by_season[season_name]

            # Find max matchweek
            max_mw = max(m.matchweek for m in season_matches)

            # Track cumulative stats for each team
            team_cumulative = defaultdict(lambda: {
                "matches_played": 0,
                "results": [],  # List of (is_home, result, goals_for, goals_against)
                "goals_scored": 0,
                "goals_conceded": 0,
                "home_wins": 0, "home_draws": 0, "home_losses": 0,
                "away_wins": 0, "away_draws": 0, "away_losses": 0,
                "clean_sheets": 0,
                "failed_to_score": 0,
            })

            # Process matches in order
            matches_by_mw = defaultdict(list)
            for m in season_matches:
                matches_by_mw[m.matchweek].append(m)

            created_this_season = 0

            for mw in range(1, max_mw + 1):
                # First, create TeamStats for all teams BEFORE processing this matchweek's results
                # This represents the state going INTO this matchweek
                for team_id, _team in teams.items():
                    stats = team_cumulative[team_id]

                    # Skip if team has no matches yet this season
                    if stats["matches_played"] == 0 and mw > 1:
                        # Check if team will play this season
                        team_in_season = any(
                            m.home_team_id == team_id or m.away_team_id == team_id
                            for m in season_matches
                        )
                        if not team_in_season:
                            continue

                    # Compute form
                    results_only = [r[1] for r in stats["results"]]
                    form = compute_form_string(results_only)
                    form_points = compute_form_points(form)

                    # Compute averages
                    mp = stats["matches_played"]
                    avg_scored = stats["goals_scored"] / mp if mp > 0 else 0
                    avg_conceded = stats["goals_conceded"] / mp if mp > 0 else 0

                    # Create or update TeamStats
                    existing = session.execute(
                        select(TeamStats)
                        .where(TeamStats.team_id == team_id)
                        .where(TeamStats.season == season_name)
                        .where(TeamStats.matchweek == mw)
                    ).scalar_one_or_none()

                    if existing:
                        ts = existing
                    else:
                        ts = TeamStats(
                            team_id=team_id,
                            season=season_name,
                            matchweek=mw,
                        )

                    ts.form = form if form else None
                    ts.form_points = form_points
                    ts.goals_scored = stats["goals_scored"]
                    ts.goals_conceded = stats["goals_conceded"]
                    ts.avg_goals_scored = Decimal(str(round(avg_scored, 2)))
                    ts.avg_goals_conceded = Decimal(str(round(avg_conceded, 2)))
                    ts.home_wins = stats["home_wins"]
                    ts.home_draws = stats["home_draws"]
                    ts.home_losses = stats["home_losses"]
                    ts.away_wins = stats["away_wins"]
                    ts.away_draws = stats["away_draws"]
                    ts.away_losses = stats["away_losses"]
                    ts.clean_sheets = stats["clean_sheets"]
                    ts.failed_to_score = stats["failed_to_score"]

                    if not existing and not dry_run:
                        session.add(ts)
                        created_this_season += 1

                # Now process this matchweek's results to update cumulative stats
                for match in matches_by_mw.get(mw, []):
                    home_id = match.home_team_id
                    away_id = match.away_team_id
                    home_score = match.home_score
                    away_score = match.away_score

                    # Home team result
                    if home_score > away_score:
                        home_result, away_result = "W", "L"
                        team_cumulative[home_id]["home_wins"] += 1
                        team_cumulative[away_id]["away_losses"] += 1
                    elif home_score < away_score:
                        home_result, away_result = "L", "W"
                        team_cumulative[home_id]["home_losses"] += 1
                        team_cumulative[away_id]["away_wins"] += 1
                    else:
                        home_result, away_result = "D", "D"
                        team_cumulative[home_id]["home_draws"] += 1
                        team_cumulative[away_id]["away_draws"] += 1

                    # Update home team
                    team_cumulative[home_id]["matches_played"] += 1
                    team_cumulative[home_id]["results"].append((True, home_result, home_score, away_score))
                    team_cumulative[home_id]["goals_scored"] += home_score
                    team_cumulative[home_id]["goals_conceded"] += away_score
                    if away_score == 0:
                        team_cumulative[home_id]["clean_sheets"] += 1
                    if home_score == 0:
                        team_cumulative[home_id]["failed_to_score"] += 1

                    # Update away team
                    team_cumulative[away_id]["matches_played"] += 1
                    team_cumulative[away_id]["results"].append((False, away_result, away_score, home_score))
                    team_cumulative[away_id]["goals_scored"] += away_score
                    team_cumulative[away_id]["goals_conceded"] += home_score
                    if home_score == 0:
                        team_cumulative[away_id]["clean_sheets"] += 1
                    if away_score == 0:
                        team_cumulative[away_id]["failed_to_score"] += 1

                # Commit periodically
                if not dry_run and mw % 10 == 0:
                    session.commit()

            if not dry_run:
                session.commit()

            total_created += created_this_season
            seasons_processed.append(season_name)
            logger.info(f"Season {season_name}: created {created_this_season} TeamStats records")

        return {
            "created": total_created,
            "seasons": seasons_processed,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Backfill TeamStats from historical match results"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be created without making changes",
    )
    parser.add_argument(
        "--season",
        type=str,
        default=None,
        help="Season to process (e.g., '2024-25'). If not specified, process ALL seasons.",
    )
    parser.add_argument(
        "--clear-existing",
        action="store_true",
        help="Clear existing TeamStats before backfilling",
    )
    args = parser.parse_args()

    result = backfill_team_stats(
        season=args.season,
        dry_run=args.dry_run,
        clear_existing=args.clear_existing,
    )

    print(f"\nResult: {result}")


if __name__ == "__main__":
    main()
