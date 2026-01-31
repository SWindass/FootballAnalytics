"""Load historical betting odds from Football-Data.co.uk.

Football-Data.co.uk provides free historical betting odds data for EPL
from 1993-present. This module downloads and parses the CSV files,
matching them to our existing matches in the database.

Data includes odds from multiple bookmakers:
- Bet365, Betway, Pinnacle, William Hill, 1xBet
- Average and Maximum odds across all bookmakers
- Over/Under 2.5 goals
- Asian Handicap

Source: https://www.football-data.co.uk/englandm.php
"""

import csv
import io
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Optional

import httpx
import structlog
from sqlalchemy import select, update
from sqlalchemy.orm.attributes import flag_modified

from app.db.database import SyncSessionLocal
from app.db.models import Match, MatchStatus, Team

logger = structlog.get_logger()

# Base URL for Football-Data.co.uk CSV files
BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/E0.csv"

# Map our season format (2024-25) to Football-Data format (2425)
def season_to_fd_format(season: str) -> str:
    """Convert '2024-25' to '2425'."""
    parts = season.split("-")
    return parts[0][2:] + parts[1]


def fd_season_to_our_format(fd_season: str) -> str:
    """Convert '2425' to '2024-25'."""
    year1 = int(fd_season[:2])
    year2 = int(fd_season[2:])
    # Handle century rollover
    if year1 > 90:
        full_year1 = 1900 + year1
    else:
        full_year1 = 2000 + year1
    return f"{full_year1}-{fd_season[2:]}"


@dataclass
class HistoricalOdds:
    """Historical betting odds for a match."""
    date: datetime
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    result: str  # H, D, A

    # Average odds (most useful for training)
    avg_home_odds: float
    avg_draw_odds: float
    avg_away_odds: float

    # Max odds
    max_home_odds: Optional[float] = None
    max_draw_odds: Optional[float] = None
    max_away_odds: Optional[float] = None

    # Bet365 odds (most liquid bookmaker)
    b365_home_odds: Optional[float] = None
    b365_draw_odds: Optional[float] = None
    b365_away_odds: Optional[float] = None

    # Over/Under 2.5
    avg_over_2_5_odds: Optional[float] = None
    avg_under_2_5_odds: Optional[float] = None

    # Implied probabilities (calculated from average odds, normalized)
    implied_home_prob: float = 0.0
    implied_draw_prob: float = 0.0
    implied_away_prob: float = 0.0

    def __post_init__(self):
        """Calculate implied probabilities from odds."""
        if self.avg_home_odds and self.avg_draw_odds and self.avg_away_odds:
            # Raw implied probabilities (will sum to > 1 due to overround)
            raw_home = 1 / self.avg_home_odds
            raw_draw = 1 / self.avg_draw_odds
            raw_away = 1 / self.avg_away_odds

            # Normalize to sum to 1
            total = raw_home + raw_draw + raw_away
            self.implied_home_prob = raw_home / total
            self.implied_draw_prob = raw_draw / total
            self.implied_away_prob = raw_away / total


# Team name mapping: Football-Data.co.uk -> Our database names
TEAM_MAPPING = {
    "Man United": "Manchester United FC",
    "Man City": "Manchester City FC",
    "Tottenham": "Tottenham Hotspur FC",
    "Newcastle": "Newcastle United FC",
    "West Ham": "West Ham United FC",
    "Wolves": "Wolverhampton Wanderers FC",
    "Brighton": "Brighton & Hove Albion FC",
    "Nott'm Forest": "Nottingham Forest FC",
    "Nottingham Forest": "Nottingham Forest FC",
    "Leicester": "Leicester City FC",
    "Crystal Palace": "Crystal Palace FC",
    "Bournemouth": "AFC Bournemouth",
    "Ipswich": "Ipswich Town FC",
    "Arsenal": "Arsenal FC",
    "Chelsea": "Chelsea FC",
    "Liverpool": "Liverpool FC",
    "Everton": "Everton FC",
    "Fulham": "Fulham FC",
    "Brentford": "Brentford FC",
    "Aston Villa": "Aston Villa FC",
    "Southampton": "Southampton FC",
    "Luton": "Luton Town FC",
    "Sheffield United": "Sheffield United FC",
    "Burnley": "Burnley FC",
    "Leeds": "Leeds United FC",
    "West Brom": "West Bromwich Albion FC",
    "Watford": "Watford FC",
    "Norwich": "Norwich City FC",
    "Swansea": "Swansea City AFC",
    "Stoke": "Stoke City FC",
    "Sunderland": "Sunderland AFC",
    "Hull": "Hull City AFC",
    "Middlesbrough": "Middlesbrough FC",
    "QPR": "Queens Park Rangers FC",
    "Reading": "Reading FC",
    "Wigan": "Wigan Athletic FC",
    "Bolton": "Bolton Wanderers FC",
    "Blackburn": "Blackburn Rovers FC",
    "Birmingham": "Birmingham City FC",
    "Blackpool": "Blackpool FC",
    "Cardiff": "Cardiff City FC",
    "Huddersfield": "Huddersfield Town AFC",
    "Charlton": "Charlton Athletic FC",
    "Portsmouth": "Portsmouth FC",
    "Derby": "Derby County FC",
    "Coventry": "Coventry City FC",
    "Sheffield Weds": "Sheffield Wednesday FC",
    "Bradford": "Bradford City AFC",
    "Wimbledon": "Wimbledon FC",
    "Oldham": "Oldham Athletic AFC",
    "Swindon": "Swindon Town FC",
    "Barnsley": "Barnsley FC",
}


def normalize_team_name(name: str) -> str:
    """Normalize team name to match our database."""
    return TEAM_MAPPING.get(name, name + " FC")


def parse_float(value: str) -> Optional[float]:
    """Parse float from CSV, handling empty strings."""
    if not value or value.strip() == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def download_season_odds(season: str) -> list[HistoricalOdds]:
    """Download and parse odds for a season.

    Args:
        season: Season in our format, e.g., "2024-25"

    Returns:
        List of HistoricalOdds for all matches in the season
    """
    fd_season = season_to_fd_format(season)
    url = BASE_URL.format(season=fd_season)

    logger.info(f"Downloading odds for season {season} from {url}")

    try:
        response = httpx.get(url, timeout=30.0, follow_redirects=True)
        response.raise_for_status()
    except Exception as e:
        logger.error(f"Failed to download odds for {season}: {e}")
        return []

    # Parse CSV
    odds_list = []
    content = response.text

    # Handle BOM if present
    if content.startswith('\ufeff'):
        content = content[1:]

    reader = csv.DictReader(io.StringIO(content))

    for row in reader:
        try:
            # Parse date (format varies: DD/MM/YYYY or DD/MM/YY)
            date_str = row.get("Date", "")
            try:
                match_date = datetime.strptime(date_str, "%d/%m/%Y")
            except ValueError:
                match_date = datetime.strptime(date_str, "%d/%m/%y")

            # Get scores
            home_score = int(row.get("FTHG", 0))
            away_score = int(row.get("FTAG", 0))
            result = row.get("FTR", "")

            # Get average odds (most useful)
            # Column names changed: older files use BbAvH/D/A, newer use AvgH/D/A
            avg_home = parse_float(row.get("AvgH", "")) or parse_float(row.get("BbAvH", ""))
            avg_draw = parse_float(row.get("AvgD", "")) or parse_float(row.get("BbAvD", ""))
            avg_away = parse_float(row.get("AvgA", "")) or parse_float(row.get("BbAvA", ""))

            # Fall back to Bet365 if no average available
            if not avg_home or not avg_draw or not avg_away:
                avg_home = parse_float(row.get("B365H", ""))
                avg_draw = parse_float(row.get("B365D", ""))
                avg_away = parse_float(row.get("B365A", ""))

            # Fall back to William Hill for older seasons (2000-02)
            if not avg_home or not avg_draw or not avg_away:
                avg_home = parse_float(row.get("WHH", ""))
                avg_draw = parse_float(row.get("WHD", ""))
                avg_away = parse_float(row.get("WHA", ""))

            # Fall back to Ladbrokes for very old seasons
            if not avg_home or not avg_draw or not avg_away:
                avg_home = parse_float(row.get("LBH", ""))
                avg_draw = parse_float(row.get("LBD", ""))
                avg_away = parse_float(row.get("LBA", ""))

            # Skip if still no odds available
            if not avg_home or not avg_draw or not avg_away:
                continue

            # Max odds (column names vary by era)
            max_home = parse_float(row.get("MaxH", "")) or parse_float(row.get("BbMxH", ""))
            max_draw = parse_float(row.get("MaxD", "")) or parse_float(row.get("BbMxD", ""))
            max_away = parse_float(row.get("MaxA", "")) or parse_float(row.get("BbMxA", ""))

            # Over/Under 2.5 odds
            avg_over = parse_float(row.get("Avg>2.5", "")) or parse_float(row.get("BbAv>2.5", ""))
            avg_under = parse_float(row.get("Avg<2.5", "")) or parse_float(row.get("BbAv<2.5", ""))

            odds = HistoricalOdds(
                date=match_date,
                home_team=row.get("HomeTeam", ""),
                away_team=row.get("AwayTeam", ""),
                home_score=home_score,
                away_score=away_score,
                result=result,
                avg_home_odds=avg_home,
                avg_draw_odds=avg_draw,
                avg_away_odds=avg_away,
                max_home_odds=max_home,
                max_draw_odds=max_draw,
                max_away_odds=max_away,
                b365_home_odds=parse_float(row.get("B365H", "")),
                b365_draw_odds=parse_float(row.get("B365D", "")),
                b365_away_odds=parse_float(row.get("B365A", "")),
                avg_over_2_5_odds=avg_over,
                avg_under_2_5_odds=avg_under,
            )
            odds_list.append(odds)

        except Exception as e:
            logger.warning(f"Failed to parse row: {e}")
            continue

    logger.info(f"Parsed {len(odds_list)} matches with odds for {season}")
    return odds_list


def match_odds_to_database(odds_list: list[HistoricalOdds], season: str) -> dict:
    """Match historical odds to our database matches.

    Args:
        odds_list: List of historical odds
        season: Season string

    Returns:
        Summary of matching results
    """
    with SyncSessionLocal() as session:
        # Load all teams for name matching
        teams = list(session.execute(select(Team)).scalars().all())
        team_name_to_id = {}
        for team in teams:
            team_name_to_id[team.name] = team.id
            team_name_to_id[team.short_name] = team.id

        # Load all matches for the season
        matches = list(session.execute(
            select(Match)
            .where(Match.season == season)
            .where(Match.status == MatchStatus.FINISHED)
        ).scalars().all())

        # Create lookup by (home_team_id, away_team_id, date)
        match_lookup = {}
        for m in matches:
            # Use date only (not time) for matching
            date_key = m.kickoff_time.date()
            match_lookup[(m.home_team_id, m.away_team_id, date_key)] = m

        matched = 0
        unmatched = 0
        updated = 0

        for odds in odds_list:
            # Normalize team names and look up IDs
            home_name = normalize_team_name(odds.home_team)
            away_name = normalize_team_name(odds.away_team)

            home_id = team_name_to_id.get(home_name)
            away_id = team_name_to_id.get(away_name)

            if not home_id or not away_id:
                logger.debug(f"Team not found: {odds.home_team} ({home_name}) or {odds.away_team} ({away_name})")
                unmatched += 1
                continue

            # Look up match
            date_key = odds.date.date()
            match = match_lookup.get((home_id, away_id, date_key))

            if not match:
                logger.debug(f"Match not found: {odds.home_team} vs {odds.away_team} on {date_key}")
                unmatched += 1
                continue

            matched += 1

            # Store odds in match features (or a dedicated column if we add one)
            # For now, we'll update the features JSON if there's a match_analysis
            from app.db.models import MatchAnalysis

            analysis = session.execute(
                select(MatchAnalysis).where(MatchAnalysis.match_id == match.id)
            ).scalar_one_or_none()

            if analysis:
                # Create a new dict to ensure SQLAlchemy detects the change
                features = dict(analysis.features or {})
                features["historical_odds"] = {
                    "avg_home_odds": odds.avg_home_odds,
                    "avg_draw_odds": odds.avg_draw_odds,
                    "avg_away_odds": odds.avg_away_odds,
                    "implied_home_prob": odds.implied_home_prob,
                    "implied_draw_prob": odds.implied_draw_prob,
                    "implied_away_prob": odds.implied_away_prob,
                    "b365_home_odds": odds.b365_home_odds,
                    "b365_draw_odds": odds.b365_draw_odds,
                    "b365_away_odds": odds.b365_away_odds,
                    "avg_over_2_5_odds": odds.avg_over_2_5_odds,
                    "avg_under_2_5_odds": odds.avg_under_2_5_odds,
                }
                analysis.features = features
                flag_modified(analysis, "features")
                updated += 1

        session.commit()

        summary = {
            "season": season,
            "total_odds": len(odds_list),
            "matched": matched,
            "unmatched": unmatched,
            "updated_analyses": updated,
        }
        logger.info(f"Matching complete: {summary}")
        return summary


def load_all_historical_odds(start_season: str = "2012-13", end_season: str = "2024-25"):
    """Load historical odds for multiple seasons.

    Args:
        start_season: First season to load
        end_season: Last season to load
    """
    # Generate list of seasons
    start_year = int(start_season.split("-")[0])
    end_year = int(end_season.split("-")[0])

    all_summaries = []

    for year in range(start_year, end_year + 1):
        season = f"{year}-{str(year + 1)[2:]}"
        odds = download_season_odds(season)
        if odds:
            summary = match_odds_to_database(odds, season)
            all_summaries.append(summary)

    # Print overall summary
    total_matched = sum(s["matched"] for s in all_summaries)
    total_updated = sum(s["updated_analyses"] for s in all_summaries)

    print(f"\n{'='*60}")
    print("HISTORICAL ODDS LOADING COMPLETE")
    print(f"{'='*60}")
    print(f"Seasons processed: {len(all_summaries)}")
    print(f"Total matches matched: {total_matched}")
    print(f"Total analyses updated: {total_updated}")
    print(f"{'='*60}")

    return all_summaries


def get_odds_as_training_data() -> list[dict]:
    """Extract historical odds as training data.

    Returns list of dicts with match features including implied probabilities
    from historical odds.
    """
    with SyncSessionLocal() as session:
        from app.db.models import MatchAnalysis

        analyses = list(session.execute(
            select(MatchAnalysis)
            .where(MatchAnalysis.features.isnot(None))
        ).scalars().all())

        training_data = []

        for analysis in analyses:
            features = analysis.features or {}
            hist_odds = features.get("historical_odds")

            if not hist_odds:
                continue

            # Get match for actual result
            match = session.get(Match, analysis.match_id)
            if not match or match.status != MatchStatus.FINISHED:
                continue

            # Determine actual result
            if match.home_score > match.away_score:
                actual = 0  # Home win
            elif match.home_score == match.away_score:
                actual = 1  # Draw
            else:
                actual = 2  # Away win

            training_data.append({
                "match_id": match.id,
                "season": match.season,
                "matchweek": match.matchweek,
                "actual_result": actual,
                # Model predictions
                "elo_home_prob": float(analysis.elo_home_prob) if analysis.elo_home_prob else None,
                "elo_draw_prob": float(analysis.elo_draw_prob) if analysis.elo_draw_prob else None,
                "elo_away_prob": float(analysis.elo_away_prob) if analysis.elo_away_prob else None,
                "poisson_home_prob": float(analysis.poisson_home_prob) if analysis.poisson_home_prob else None,
                "poisson_draw_prob": float(analysis.poisson_draw_prob) if analysis.poisson_draw_prob else None,
                "poisson_away_prob": float(analysis.poisson_away_prob) if analysis.poisson_away_prob else None,
                # Market odds (implied probabilities)
                "market_home_prob": hist_odds["implied_home_prob"],
                "market_draw_prob": hist_odds["implied_draw_prob"],
                "market_away_prob": hist_odds["implied_away_prob"],
                # Raw odds for value analysis
                "avg_home_odds": hist_odds["avg_home_odds"],
                "avg_draw_odds": hist_odds["avg_draw_odds"],
                "avg_away_odds": hist_odds["avg_away_odds"],
            })

        logger.info(f"Extracted {len(training_data)} matches with historical odds for training")
        return training_data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load historical betting odds")
    parser.add_argument("--season", help="Load single season (e.g., 2024-25)")
    parser.add_argument("--all", action="store_true", help="Load all seasons 2012-2024")
    parser.add_argument("--extract", action="store_true", help="Extract training data with odds")

    args = parser.parse_args()

    if args.extract:
        data = get_odds_as_training_data()
        print(f"Extracted {len(data)} training samples with historical odds")
        if data:
            print("\nSample:")
            import json
            print(json.dumps(data[0], indent=2))
    elif args.season:
        odds = download_season_odds(args.season)
        if odds:
            summary = match_odds_to_database(odds, args.season)
            print(f"Summary: {summary}")
    elif args.all:
        load_all_historical_odds()
    else:
        # Default: show sample
        print("Downloading sample from 2024-25 season...")
        odds = download_season_odds("2024-25")
        print(f"\nSample odds (first 3 matches):")
        for o in odds[:3]:
            print(f"\n{o.date.strftime('%Y-%m-%d')} {o.home_team} vs {o.away_team}")
            print(f"  Result: {o.home_score}-{o.away_score} ({o.result})")
            print(f"  Avg Odds: H={o.avg_home_odds:.2f} D={o.avg_draw_odds:.2f} A={o.avg_away_odds:.2f}")
            print(f"  Implied:  H={o.implied_home_prob:.1%} D={o.implied_draw_prob:.1%} A={o.implied_away_prob:.1%}")
