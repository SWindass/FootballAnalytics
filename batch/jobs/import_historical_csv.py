"""Import historical EPL data from Kaggle CSV files.

Supports common Kaggle EPL dataset formats:
- https://www.kaggle.com/datasets/evangower/premier-league-matches-19922022
- https://www.kaggle.com/datasets/saife245/english-premier-league
- football-data.co.uk CSV format

Usage:
    python import_historical_csv.py path/to/matches.csv
    python import_historical_csv.py path/to/folder/  # imports all CSVs
"""

import argparse
import csv
from datetime import datetime
from pathlib import Path

import structlog
from sqlalchemy import select

from app.db.database import SyncSessionLocal
from app.db.models import Match, MatchStatus, Team

logger = structlog.get_logger()


# Common column name mappings (normalize different CSV formats)
COLUMN_MAPPINGS = {
    # Date columns
    "date": ["date", "Date", "match_date", "MatchDate", "datetime"],
    # Team columns
    "home_team": ["HomeTeam", "Home", "home_team", "home", "HT", "home_team_name"],
    "away_team": ["AwayTeam", "Away", "away_team", "away", "AT", "away_team_name"],
    # Score columns
    "home_goals": ["FTHG", "HomeGoals", "home_goals", "HG", "home_score", "GH"],
    "away_goals": ["FTAG", "AwayGoals", "away_goals", "AG", "away_score", "GA"],
    # Half-time scores (optional)
    "home_ht_goals": ["HTHG", "HTHomeGoals", "home_ht_goals", "HHG"],
    "away_ht_goals": ["HTAG", "HTAwayGoals", "away_ht_goals", "HAG"],
    # Season (optional - can be derived from date)
    "season": ["Season", "season", "Year", "year"],
    # Matchweek/round (optional)
    "matchweek": ["Wk", "Round", "Matchweek", "matchweek", "MW", "round"],
}

# Team name normalization (common variations -> standard name)
TEAM_NAME_MAP = {
    # Current teams
    "arsenal": "Arsenal FC",
    "aston villa": "Aston Villa FC",
    "bournemouth": "AFC Bournemouth",
    "afc bournemouth": "AFC Bournemouth",
    "brentford": "Brentford FC",
    "brighton": "Brighton & Hove Albion FC",
    "brighton and hove albion": "Brighton & Hove Albion FC",
    "brighton & hove albion": "Brighton & Hove Albion FC",
    "chelsea": "Chelsea FC",
    "crystal palace": "Crystal Palace FC",
    "everton": "Everton FC",
    "fulham": "Fulham FC",
    "ipswich": "Ipswich Town FC",
    "ipswich town": "Ipswich Town FC",
    "leicester": "Leicester City FC",
    "leicester city": "Leicester City FC",
    "liverpool": "Liverpool FC",
    "man city": "Manchester City FC",
    "manchester city": "Manchester City FC",
    "man united": "Manchester United FC",
    "manchester united": "Manchester United FC",
    "newcastle": "Newcastle United FC",
    "newcastle united": "Newcastle United FC",
    "nottingham forest": "Nottingham Forest FC",
    "nott'm forest": "Nottingham Forest FC",
    "southampton": "Southampton FC",
    "tottenham": "Tottenham Hotspur FC",
    "tottenham hotspur": "Tottenham Hotspur FC",
    "spurs": "Tottenham Hotspur FC",
    "west ham": "West Ham United FC",
    "west ham united": "West Ham United FC",
    "wolves": "Wolverhampton Wanderers FC",
    "wolverhampton": "Wolverhampton Wanderers FC",
    "wolverhampton wanderers": "Wolverhampton Wanderers FC",
    # Historic teams (relegated/promoted over the years)
    "barnsley": "Barnsley FC",
    "birmingham": "Birmingham City FC",
    "birmingham city": "Birmingham City FC",
    "blackburn": "Blackburn Rovers FC",
    "blackburn rovers": "Blackburn Rovers FC",
    "blackpool": "Blackpool FC",
    "bolton": "Bolton Wanderers FC",
    "bolton wanderers": "Bolton Wanderers FC",
    "bradford": "Bradford City AFC",
    "bradford city": "Bradford City AFC",
    "burnley": "Burnley FC",
    "cardiff": "Cardiff City FC",
    "cardiff city": "Cardiff City FC",
    "charlton": "Charlton Athletic FC",
    "charlton athletic": "Charlton Athletic FC",
    "coventry": "Coventry City FC",
    "coventry city": "Coventry City FC",
    "derby": "Derby County FC",
    "derby county": "Derby County FC",
    "huddersfield": "Huddersfield Town AFC",
    "huddersfield town": "Huddersfield Town AFC",
    "hull": "Hull City AFC",
    "hull city": "Hull City AFC",
    "leeds": "Leeds United FC",
    "leeds united": "Leeds United FC",
    "luton": "Luton Town FC",
    "luton town": "Luton Town FC",
    "middlesbrough": "Middlesbrough FC",
    "middlesboro": "Middlesbrough FC",
    "norwich": "Norwich City FC",
    "norwich city": "Norwich City FC",
    "oldham": "Oldham Athletic AFC",
    "oldham athletic": "Oldham Athletic AFC",
    "portsmouth": "Portsmouth FC",
    "qpr": "Queens Park Rangers FC",
    "queens park rangers": "Queens Park Rangers FC",
    "reading": "Reading FC",
    "sheffield united": "Sheffield United FC",
    "sheffield weds": "Sheffield Wednesday FC",
    "sheffield wednesday": "Sheffield Wednesday FC",
    "stoke": "Stoke City FC",
    "stoke city": "Stoke City FC",
    "sunderland": "Sunderland AFC",
    "swansea": "Swansea City AFC",
    "swansea city": "Swansea City AFC",
    "swindon": "Swindon Town FC",
    "swindon town": "Swindon Town FC",
    "watford": "Watford FC",
    "west brom": "West Bromwich Albion FC",
    "west bromwich albion": "West Bromwich Albion FC",
    "wigan": "Wigan Athletic FC",
    "wigan athletic": "Wigan Athletic FC",
    "wimbledon": "Wimbledon FC",
}


def normalize_team_name(name: str) -> str:
    """Normalize team name to standard format."""
    if not name:
        return name

    # Clean and lowercase for lookup
    clean = name.strip().lower()

    # Direct match
    if clean in TEAM_NAME_MAP:
        return TEAM_NAME_MAP[clean]

    # Try partial match
    for key, value in TEAM_NAME_MAP.items():
        if key in clean or clean in key:
            return value

    # Return original with title case if no match
    return name.strip().title() + " FC" if "FC" not in name else name.strip()


def find_column(headers: list[str], column_type: str) -> str | None:
    """Find the actual column name for a given column type."""
    possible_names = COLUMN_MAPPINGS.get(column_type, [])
    for name in possible_names:
        if name in headers:
            return name
    return None


def parse_date(date_str: str) -> datetime | None:
    """Parse date from various formats."""
    if not date_str:
        return None

    formats = [
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%d/%m/%y",
        "%Y-%m-%d %H:%M:%S",
        "%d-%m-%Y",
        "%m/%d/%Y",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue

    logger.warning(f"Could not parse date: {date_str}")
    return None


def get_season_from_date(dt: datetime) -> str:
    """Derive season string from match date (e.g., '2023-24')."""
    # EPL season runs Aug-May
    if dt.month >= 8:  # Aug-Dec = first year of season
        start_year = dt.year
    else:  # Jan-Jul = second year of season
        start_year = dt.year - 1

    return f"{start_year}-{str(start_year + 1)[-2:]}"


def calculate_matchweek(dt: datetime, season_matches: dict) -> int:
    """Estimate matchweek from date within season.

    Uses match count (10 per matchweek) rather than unique dates.
    Assumes CSV rows are in chronological order.
    """
    season = get_season_from_date(dt)

    if season not in season_matches:
        season_matches[season] = 0

    # Count this match
    match_idx = season_matches[season]
    season_matches[season] += 1

    # 10 matches per matchweek
    matchweek = (match_idx // 10) + 1

    return min(matchweek, 38)


def import_csv_file(filepath: Path, session, team_cache: dict, dry_run: bool = False) -> dict:
    """Import a single CSV file."""
    logger.info(f"Processing {filepath}")

    stats = {
        "file": str(filepath),
        "rows_processed": 0,
        "matches_created": 0,
        "matches_skipped": 0,
        "teams_created": 0,
        "errors": [],
    }

    # Try different encodings
    encodings = ["utf-8-sig", "utf-8", "latin-1", "cp1252"]
    content = None

    for encoding in encodings:
        try:
            with open(filepath, encoding=encoding) as f:
                content = f.read()
                break
        except UnicodeDecodeError:
            continue

    if content is None:
        stats["errors"].append("Could not decode file with any encoding")
        return stats

    # Process as string
    lines = content.splitlines()
    if not lines:
        stats["errors"].append("Empty file")
        return stats

    first_line = lines[0]

    # Determine delimiter (default to comma)
    if ";" in first_line and "," not in first_line:
        delimiter = ";"
    elif "\t" in first_line and "," not in first_line:
        delimiter = "\t"
    else:
        delimiter = ","

    reader = csv.DictReader(lines, delimiter=delimiter)
    headers = reader.fieldnames

    if not headers:
        stats["errors"].append("No headers found")
        return stats

    # Find column mappings
    date_col = find_column(headers, "date")
    home_col = find_column(headers, "home_team")
    away_col = find_column(headers, "away_team")
    home_goals_col = find_column(headers, "home_goals")
    away_goals_col = find_column(headers, "away_goals")
    home_ht_col = find_column(headers, "home_ht_goals")
    away_ht_col = find_column(headers, "away_ht_goals")
    season_col = find_column(headers, "season")
    matchweek_col = find_column(headers, "matchweek")

    if not all([date_col, home_col, away_col, home_goals_col, away_goals_col]):
        missing = []
        if not date_col:
            missing.append("date")
        if not home_col:
            missing.append("home_team")
        if not away_col:
            missing.append("away_team")
        if not home_goals_col:
            missing.append("home_goals")
        if not away_goals_col:
            missing.append("away_goals")
        stats["errors"].append(f"Missing required columns: {missing}")
        stats["errors"].append(f"Available columns: {headers}")
        return stats

    logger.info(f"Column mapping: date={date_col}, home={home_col}, away={away_col}")

    season_matches = {}  # For matchweek calculation

    for row in reader:
        stats["rows_processed"] += 1

        try:
            # Parse date
            match_date = parse_date(row.get(date_col, ""))
            if not match_date:
                continue

            # Get/normalize team names
            home_name = normalize_team_name(row.get(home_col, ""))
            away_name = normalize_team_name(row.get(away_col, ""))

            if not home_name or not away_name:
                continue

            # Get or create teams
            for team_name in [home_name, away_name]:
                if team_name not in team_cache:
                    existing = session.execute(
                        select(Team).where(Team.name == team_name)
                    ).scalar_one_or_none()

                    if existing:
                        team_cache[team_name] = existing.id
                    elif not dry_run:
                        # Create new team
                        short_name = team_name.replace(" FC", "").replace(" AFC", "")
                        if len(short_name) > 15:
                            short_name = short_name.split()[0]

                        team = Team(
                            external_id=-(abs(hash(team_name)) % 100000),  # Negative to avoid API collision
                            name=team_name,
                            short_name=short_name,
                            tla=team_name[:3].upper(),
                        )
                        session.add(team)
                        session.flush()
                        team_cache[team_name] = team.id
                        stats["teams_created"] += 1
                        logger.info(f"Created team: {team_name}")
                    else:
                        team_cache[team_name] = None

            home_team_id = team_cache.get(home_name)
            away_team_id = team_cache.get(away_name)

            if not home_team_id or not away_team_id:
                continue

            # Get season
            if season_col and row.get(season_col):
                season_raw = row[season_col]
                # Handle various formats: "2023-24", "2023/24", "2023"
                if "/" in season_raw:
                    season = season_raw.replace("/", "-")
                elif len(season_raw) == 4:
                    season = f"{season_raw}-{str(int(season_raw)+1)[-2:]}"
                else:
                    season = season_raw
            else:
                season = get_season_from_date(match_date)

            # Get matchweek
            if matchweek_col and row.get(matchweek_col):
                try:
                    matchweek = int(row[matchweek_col])
                except ValueError:
                    matchweek = calculate_matchweek(match_date, season_matches)
            else:
                matchweek = calculate_matchweek(match_date, season_matches)

            # Parse scores
            try:
                home_goals = int(row.get(home_goals_col, 0))
                away_goals = int(row.get(away_goals_col, 0))
            except (ValueError, TypeError):
                continue

            home_ht = None
            away_ht = None
            if home_ht_col and away_ht_col:
                try:
                    home_ht = int(row.get(home_ht_col, ""))
                    away_ht = int(row.get(away_ht_col, ""))
                except (ValueError, TypeError):
                    pass

            # Create unique external ID from date and teams
            # Use negative range to avoid collision with API data (which uses positive IDs)
            external_id = -(abs(hash(f"{match_date.isoformat()}-{home_name}-{away_name}")) % 100000000)

            # Check if match exists
            existing_match = session.execute(
                select(Match).where(
                    Match.season == season,
                    Match.home_team_id == home_team_id,
                    Match.away_team_id == away_team_id,
                    Match.matchweek == matchweek,
                )
            ).scalar_one_or_none()

            if existing_match:
                stats["matches_skipped"] += 1
                continue

            if not dry_run:
                match = Match(
                    external_id=external_id,
                    season=season,
                    matchweek=matchweek,
                    kickoff_time=match_date,
                    status=MatchStatus.FINISHED,
                    home_team_id=home_team_id,
                    away_team_id=away_team_id,
                    home_score=home_goals,
                    away_score=away_goals,
                    home_ht_score=home_ht,
                    away_ht_score=away_ht,
                )
                session.add(match)

            stats["matches_created"] += 1

        except Exception as e:
            stats["errors"].append(f"Row {stats['rows_processed']}: {str(e)}")
            continue

    if not dry_run:
        session.commit()

    return stats


def import_historical_data(
    path: str,
    dry_run: bool = False,
) -> dict:
    """Import historical data from CSV file or folder.

    Args:
        path: Path to CSV file or folder containing CSVs
        dry_run: If True, don't actually write to database
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")

    # Find CSV files
    if path.is_file():
        csv_files = [path]
    else:
        csv_files = sorted(path.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {path}")

    logger.info(f"Found {len(csv_files)} CSV file(s) to import")

    results = {
        "files_processed": 0,
        "total_matches_created": 0,
        "total_teams_created": 0,
        "dry_run": dry_run,
        "file_results": [],
    }

    with SyncSessionLocal() as session:
        # Build team cache from existing teams
        existing_teams = session.execute(select(Team)).scalars().all()
        team_cache = {t.name: t.id for t in existing_teams}
        logger.info(f"Loaded {len(team_cache)} existing teams")

        for csv_file in csv_files:
            file_stats = import_csv_file(csv_file, session, team_cache, dry_run)
            results["file_results"].append(file_stats)
            results["files_processed"] += 1
            results["total_matches_created"] += file_stats["matches_created"]
            results["total_teams_created"] += file_stats["teams_created"]

            logger.info(
                f"  {csv_file.name}: {file_stats['matches_created']} matches, "
                f"{file_stats['teams_created']} teams, "
                f"{file_stats['matches_skipped']} skipped"
            )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Import historical EPL data from Kaggle CSV files"
    )
    parser.add_argument(
        "path",
        help="Path to CSV file or folder containing CSVs",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse files but don't write to database",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("EPL Historical Data Importer")
    print("=" * 60)

    if args.dry_run:
        print("DRY RUN MODE - no data will be written")

    print()

    try:
        results = import_historical_data(args.path, dry_run=args.dry_run)

        print()
        print("=" * 60)
        print("Import Complete!")
        print(f"  Files processed: {results['files_processed']}")
        print(f"  Matches created: {results['total_matches_created']}")
        print(f"  Teams created: {results['total_teams_created']}")
        print("=" * 60)

        # Show any errors
        for file_result in results["file_results"]:
            if file_result["errors"]:
                print(f"\nErrors in {file_result['file']}:")
                for error in file_result["errors"][:5]:
                    print(f"  - {error}")
                if len(file_result["errors"]) > 5:
                    print(f"  ... and {len(file_result['errors']) - 5} more")

    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
