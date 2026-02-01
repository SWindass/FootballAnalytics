"""Calculate Poisson predictions from historical match data.

Calculates team attack/defense strengths and generates match outcome
probabilities using the Poisson distribution model.
"""

import argparse
from decimal import Decimal

import structlog
from sqlalchemy import select

from app.core.config import get_settings
from app.db.database import SyncSessionLocal
from app.db.models import Match, MatchAnalysis, MatchStatus, Team
from batch.models.poisson import (
    PoissonModel,
    calculate_home_away_strengths,
    calculate_team_strengths,
)

logger = structlog.get_logger()
settings = get_settings()


def get_league_averages(matches: list[dict]) -> tuple[float, float]:
    """Calculate league average goals scored/conceded per team per game.

    Args:
        matches: List of completed match dicts with home_score and away_score

    Returns:
        Tuple of (avg_scored, avg_conceded) - should be equal for the league
    """
    total_goals = 0
    total_games = 0

    for match in matches:
        if match.get("home_score") is not None and match.get("away_score") is not None:
            total_goals += match["home_score"] + match["away_score"]
            total_games += 1

    if total_games == 0:
        return 1.375, 1.375  # Default EPL average (2.75 goals/game / 2 teams)

    # Each game has 2 teams, so divide by 2 to get per-team average
    avg_per_team = total_goals / (total_games * 2)
    return avg_per_team, avg_per_team


def calculate_team_strengths_for_season(
    season: str,
    up_to_matchweek: int | None = None,
) -> dict:
    """Calculate attack/defense strengths for all teams in a season.

    Args:
        season: Season to calculate for (e.g., "2024-25")
        up_to_matchweek: Only include matches up to this matchweek (exclusive).
                        If None, includes all finished matches.

    Returns:
        Dict with team_id -> {"attack": float, "defense": float, "games": int}
    """
    with SyncSessionLocal() as session:
        # Build query for finished matches
        stmt = (
            select(Match)
            .where(Match.season == season)
            .where(Match.status == MatchStatus.FINISHED)
        )

        if up_to_matchweek is not None:
            stmt = stmt.where(Match.matchweek < up_to_matchweek)

        stmt = stmt.order_by(Match.matchweek, Match.kickoff_time)
        matches = list(session.execute(stmt).scalars().all())

        if not matches:
            return {}

        # Convert to dict format for calculate_team_strengths
        match_dicts = [
            {
                "home_team_id": m.home_team_id,
                "away_team_id": m.away_team_id,
                "home_score": m.home_score,
                "away_score": m.away_score,
            }
            for m in matches
        ]

        # Calculate league averages
        avg_scored, avg_conceded = get_league_averages(match_dicts)

        # Calculate team strengths
        strengths = calculate_team_strengths(match_dicts, avg_scored, avg_conceded)

        # Count games per team
        team_games: dict[int, int] = {}
        for m in match_dicts:
            team_games[m["home_team_id"]] = team_games.get(m["home_team_id"], 0) + 1
            team_games[m["away_team_id"]] = team_games.get(m["away_team_id"], 0) + 1

        # Format result
        result = {}
        for team_id, (attack, defense) in strengths.items():
            result[team_id] = {
                "attack": attack,
                "defense": defense,
                "games": team_games.get(team_id, 0),
            }

        return result


def calculate_home_away_strengths_for_season(
    season: str,
    up_to_matchweek: int | None = None,
) -> dict:
    """Calculate home/away specific attack/defense strengths.

    Args:
        season: Season to calculate for
        up_to_matchweek: Only include matches before this matchweek

    Returns:
        Dict with team_id -> {home_attack, home_defense, away_attack, away_defense, ...}
    """
    with SyncSessionLocal() as session:
        stmt = (
            select(Match)
            .where(Match.season == season)
            .where(Match.status == MatchStatus.FINISHED)
        )

        if up_to_matchweek is not None:
            stmt = stmt.where(Match.matchweek < up_to_matchweek)

        matches = list(session.execute(stmt).scalars().all())

        if not matches:
            return {}

        match_dicts = [
            {
                "home_team_id": m.home_team_id,
                "away_team_id": m.away_team_id,
                "home_score": m.home_score,
                "away_score": m.away_score,
            }
            for m in matches
        ]

        return calculate_home_away_strengths(match_dicts)


def calculate_poisson_predictions(
    season: str,
    matchweek: int | None = None,
    backfill: bool = False,
) -> dict:
    """Calculate Poisson predictions for matches.

    Args:
        season: Season to calculate for (e.g., "2024-25")
        matchweek: Specific matchweek to calculate. If None and backfill=True,
                  calculates all matchweeks.
        backfill: If True, recalculate predictions for all past matchweeks.

    Returns:
        Summary of results
    """
    with SyncSessionLocal() as session:
        # Get team names for output
        teams = {t.id: t for t in session.execute(select(Team)).scalars().all()}

        # Determine which matchweeks to process
        if matchweek is not None:
            matchweeks_to_process = [matchweek]
        elif backfill:
            # Get all matchweeks with matches
            stmt = (
                select(Match.matchweek)
                .where(Match.season == season)
                .distinct()
                .order_by(Match.matchweek)
            )
            matchweeks_to_process = [mw for (mw,) in session.execute(stmt).all()]
        else:
            # Get the latest matchweek
            stmt = (
                select(Match.matchweek)
                .where(Match.season == season)
                .order_by(Match.matchweek.desc())
                .limit(1)
            )
            result = session.execute(stmt).first()
            if result:
                matchweeks_to_process = [result[0]]
            else:
                print(f"No matches found for season {season}")
                return {"status": "no_matches"}

        print(f"Processing matchweeks: {matchweeks_to_process}")

        model = PoissonModel()
        predictions_created = 0
        predictions_updated = 0

        for mw in matchweeks_to_process:
            print(f"\n{'='*60}")
            print(f"Matchweek {mw}")
            print("=" * 60)

            # Calculate home/away specific strengths (more accurate)
            ha_strengths = calculate_home_away_strengths_for_season(season, up_to_matchweek=mw)

            if not ha_strengths:
                print("  No prior matches to calculate strengths, using defaults")

            # Get matches for this matchweek
            stmt = (
                select(Match)
                .where(Match.season == season)
                .where(Match.matchweek == mw)
                .order_by(Match.kickoff_time)
            )
            matches = list(session.execute(stmt).scalars().all())

            for match in matches:
                home_team = teams.get(match.home_team_id)
                away_team = teams.get(match.away_team_id)
                home_name = home_team.short_name if home_team else f"Team {match.home_team_id}"
                away_name = away_team.short_name if away_team else f"Team {match.away_team_id}"

                # Default strengths
                default_ha = {
                    "home_attack": 1.0, "home_defense": 1.0,
                    "away_attack": 1.0, "away_defense": 1.0,
                    "home_games": 0, "away_games": 0,
                }

                # Get home/away specific strengths
                home_ha = ha_strengths.get(match.home_team_id, default_ha)
                away_ha = ha_strengths.get(match.away_team_id, default_ha)

                # Use home team's HOME attack vs away team's AWAY defense
                # And away team's AWAY attack vs home team's HOME defense
                home_xg, away_xg = model.calculate_expected_goals(
                    home_attack=home_ha["home_attack"],
                    home_defense=home_ha["home_defense"],
                    away_attack=away_ha["away_attack"],
                    away_defense=away_ha["away_defense"],
                )

                # Calculate match probabilities
                home_prob, draw_prob, away_prob = model.match_probabilities(home_xg, away_xg)

                # Calculate over/under and BTTS
                over_prob, _ = model.over_under_probability(home_xg, away_xg, line=2.5)
                btts_prob, _ = model.btts_probability(home_xg, away_xg)

                # Get or create MatchAnalysis
                existing = session.execute(
                    select(MatchAnalysis).where(MatchAnalysis.match_id == match.id)
                ).scalar_one_or_none()

                # Prepare features dict with home/away splits
                features = {
                    "home_attack_strength": round(home_ha["home_attack"], 4),
                    "home_defense_strength": round(home_ha["home_defense"], 4),
                    "away_attack_strength": round(away_ha["away_attack"], 4),
                    "away_defense_strength": round(away_ha["away_defense"], 4),
                    "home_games_played": home_ha["home_games"] + home_ha["away_games"],
                    "away_games_played": away_ha["home_games"] + away_ha["away_games"],
                    "home_expected_goals": round(home_xg, 2),
                    "away_expected_goals": round(away_xg, 2),
                }

                if existing:
                    # Update existing analysis
                    existing.poisson_home_prob = Decimal(str(round(home_prob, 4)))
                    existing.poisson_draw_prob = Decimal(str(round(draw_prob, 4)))
                    existing.poisson_away_prob = Decimal(str(round(away_prob, 4)))
                    existing.poisson_over_2_5_prob = Decimal(str(round(over_prob, 4)))
                    existing.poisson_btts_prob = Decimal(str(round(btts_prob, 4)))
                    existing.predicted_home_goals = Decimal(str(round(home_xg, 2)))
                    existing.predicted_away_goals = Decimal(str(round(away_xg, 2)))
                    # Merge features with existing
                    if existing.features:
                        existing.features.update(features)
                    else:
                        existing.features = features
                    predictions_updated += 1
                else:
                    # Create new analysis
                    analysis = MatchAnalysis(
                        match_id=match.id,
                        poisson_home_prob=Decimal(str(round(home_prob, 4))),
                        poisson_draw_prob=Decimal(str(round(draw_prob, 4))),
                        poisson_away_prob=Decimal(str(round(away_prob, 4))),
                        poisson_over_2_5_prob=Decimal(str(round(over_prob, 4))),
                        poisson_btts_prob=Decimal(str(round(btts_prob, 4))),
                        predicted_home_goals=Decimal(str(round(home_xg, 2))),
                        predicted_away_goals=Decimal(str(round(away_xg, 2))),
                        features=features,
                    )
                    session.add(analysis)
                    predictions_created += 1

                # Print match prediction
                print(
                    f"  {home_name:>20} vs {away_name:<20} | "
                    f"H:{home_prob:.1%} D:{draw_prob:.1%} A:{away_prob:.1%} | "
                    f"xG: {home_xg:.2f}-{away_xg:.2f}"
                )

        session.commit()

        # Print summary
        print(f"\n{'='*60}")
        print("Summary")
        print("=" * 60)
        print(f"Season: {season}")
        print(f"Matchweeks processed: {len(matchweeks_to_process)}")
        print(f"Predictions created: {predictions_created}")
        print(f"Predictions updated: {predictions_updated}")

        return {
            "status": "success",
            "season": season,
            "matchweeks_processed": len(matchweeks_to_process),
            "predictions_created": predictions_created,
            "predictions_updated": predictions_updated,
        }


def print_team_strengths(season: str, matchweek: int | None = None) -> None:
    """Print team strengths table for a season.

    Args:
        season: Season to show strengths for
        matchweek: Calculate strengths up to this matchweek (exclusive).
                  If None, uses all completed matches.
    """
    strengths = calculate_team_strengths_for_season(season, up_to_matchweek=matchweek)

    if not strengths:
        print(f"No data available for season {season}")
        return

    with SyncSessionLocal() as session:
        teams = {t.id: t for t in session.execute(select(Team)).scalars().all()}

    print(f"\n{'='*70}")
    label = f"Team Strengths - {season}"
    if matchweek:
        label += f" (up to MW{matchweek})"
    print(label)
    print("=" * 70)
    print(f"{'Team':<25} {'Attack':>10} {'Defense':>10} {'Games':>8}")
    print("-" * 70)

    # Sort by attack strength descending
    sorted_strengths = sorted(
        strengths.items(),
        key=lambda x: x[1]["attack"],
        reverse=True,
    )

    for team_id, stats in sorted_strengths:
        team = teams.get(team_id)
        team_name = team.name if team else f"Team {team_id}"
        print(
            f"{team_name:<25} {stats['attack']:>10.3f} {stats['defense']:>10.3f} {stats['games']:>8}"
        )

    print("=" * 70)
    print("Attack > 1.0 = scores more than league average")
    print("Defense > 1.0 = concedes more than league average")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate Poisson predictions")
    parser.add_argument(
        "--season",
        type=str,
        default="2024-25",
        help="Season to calculate (e.g., '2024-25')",
    )
    parser.add_argument(
        "--matchweek",
        type=int,
        default=None,
        help="Specific matchweek to calculate predictions for",
    )
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="Recalculate predictions for all matchweeks in the season",
    )
    parser.add_argument(
        "--strengths-only",
        action="store_true",
        help="Only print team strengths, don't calculate predictions",
    )

    args = parser.parse_args()

    if args.strengths_only:
        print_team_strengths(args.season, args.matchweek)
    else:
        result = calculate_poisson_predictions(
            season=args.season,
            matchweek=args.matchweek,
            backfill=args.backfill,
        )
        print(f"\nResult: {result}")
