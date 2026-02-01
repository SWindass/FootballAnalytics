"""Import historical FPL data from vaastav/Fantasy-Premier-League GitHub repo.

Downloads and imports match-by-match player data from 2016-17 to present.
"""

import asyncio
import io
from datetime import datetime
from decimal import Decimal
from typing import Optional

import httpx
import pandas as pd
import structlog
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.database import SyncSessionLocal
from app.db.models import Player, PlayerMatchPerformance, Team

logger = structlog.get_logger()
settings = get_settings()

# GitHub raw content base URL
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"

# Seasons to import (oldest to newest)
SEASONS = [
    "2016-17",
    "2017-18",
    "2018-19",
    "2019-20",
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24",
    "2024-25",
]

# Map FPL team names to our team names (handles name changes over years)
FPL_TEAM_NAME_MAP = {
    # Current teams
    "Arsenal": "Arsenal FC",
    "Aston Villa": "Aston Villa FC",
    "Bournemouth": "AFC Bournemouth",
    "Brentford": "Brentford FC",
    "Brighton": "Brighton & Hove Albion FC",
    "Chelsea": "Chelsea FC",
    "Crystal Palace": "Crystal Palace FC",
    "Everton": "Everton FC",
    "Fulham": "Fulham FC",
    "Ipswich": "Ipswich Town FC",
    "Leicester": "Leicester City FC",
    "Liverpool": "Liverpool FC",
    "Man City": "Manchester City FC",
    "Man Utd": "Manchester United FC",
    "Newcastle": "Newcastle United FC",
    "Nott'm Forest": "Nottingham Forest FC",
    "Southampton": "Southampton FC",
    "Spurs": "Tottenham Hotspur FC",
    "West Ham": "West Ham United FC",
    "Wolves": "Wolverhampton Wanderers FC",
    # Historical teams (relegated/promoted over years)
    "Burnley": "Burnley FC",
    "Cardiff": "Cardiff City FC",
    "Huddersfield": "Huddersfield Town AFC",
    "Leeds": "Leeds United FC",
    "Luton": "Luton Town FC",
    "Middlesbrough": "Middlesbrough FC",
    "Norwich": "Norwich City FC",
    "QPR": "Queens Park Rangers FC",
    "Sheffield Utd": "Sheffield United FC",
    "Stoke": "Stoke City FC",
    "Sunderland": "Sunderland AFC",
    "Swansea": "Swansea City AFC",
    "Watford": "Watford FC",
    "West Brom": "West Bromwich Albion FC",
}


class FPLHistoricalImporter:
    """Imports historical FPL data from GitHub."""

    def __init__(self, session: Optional[Session] = None):
        self.session = session or SyncSessionLocal()
        self._team_cache: dict[str, int] = {}  # team name -> our team_id
        self._player_cache: dict[str, int] = {}  # "name_season" -> player_id

    def run(self, seasons: list[str] = None) -> dict:
        """Import historical FPL data.

        Args:
            seasons: List of seasons to import, e.g. ["2022-23", "2023-24"]
                    If None, imports all available seasons.
        """
        seasons = seasons or SEASONS
        logger.info(f"Starting historical FPL import for {len(seasons)} seasons")
        start_time = datetime.utcnow()

        try:
            # Build team mapping
            self._build_team_mapping()

            total_records = 0
            for season in seasons:
                count = asyncio.run(self._import_season(season))
                total_records += count
                self.session.commit()
                logger.info(f"Imported {count} records for {season}")

            duration = (datetime.utcnow() - start_time).total_seconds()
            logger.info(
                "Historical FPL import completed",
                total_records=total_records,
                seasons=len(seasons),
                duration_seconds=duration,
            )

            return {
                "status": "success",
                "total_records": total_records,
                "seasons_imported": len(seasons),
                "duration_seconds": duration,
            }

        except Exception as e:
            logger.error("Historical FPL import failed", error=str(e))
            self.session.rollback()
            raise

    def _build_team_mapping(self):
        """Build mapping from team names to our team IDs."""
        teams = list(self.session.execute(select(Team)).scalars().all())

        for team in teams:
            self._team_cache[team.name] = team.id
            # Also map by short name variations
            for fpl_name, our_name in FPL_TEAM_NAME_MAP.items():
                if our_name == team.name:
                    self._team_cache[fpl_name] = team.id

        logger.info(f"Built team mapping with {len(self._team_cache)} entries")

    async def _import_season(self, season: str) -> int:
        """Import all gameweek data for a season."""
        url = f"{GITHUB_RAW_BASE}/{season}/gws/merged_gw.csv"

        logger.info(f"Downloading {season} data from {url}")

        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=60)
            if response.status_code == 404:
                logger.warning(f"No merged_gw.csv found for {season}, trying individual GWs")
                return await self._import_season_individual_gws(season, client)

            response.raise_for_status()
            csv_content = response.text

        # Parse CSV
        df = pd.read_csv(io.StringIO(csv_content))

        return self._process_dataframe(df, season)

    async def _import_season_individual_gws(self, season: str, client: httpx.AsyncClient) -> int:
        """Fallback: import individual gameweek files."""
        all_data = []

        for gw in range(1, 39):
            url = f"{GITHUB_RAW_BASE}/{season}/gws/gw{gw}.csv"
            try:
                response = await client.get(url, timeout=30)
                if response.status_code == 200:
                    df = pd.read_csv(io.StringIO(response.text))
                    df['GW'] = gw  # Add gameweek column if missing
                    all_data.append(df)
            except Exception as e:
                logger.debug(f"GW{gw} not available for {season}: {e}")
                continue

        if not all_data:
            logger.warning(f"No gameweek data found for {season}")
            return 0

        combined_df = pd.concat(all_data, ignore_index=True)
        return self._process_dataframe(combined_df, season)

    def _process_dataframe(self, df: pd.DataFrame, season: str) -> int:
        """Process a dataframe of gameweek data."""
        records_imported = 0

        # Normalize column names (they vary between seasons)
        df.columns = df.columns.str.lower().str.strip()

        # Map common column name variations
        column_map = {
            'round': 'gw',
            'gameweek': 'gw',
            'opponent_team': 'opponent',
            'was_home': 'is_home',
            'ict_index': 'ict',
        }
        df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})

        # Group by player and gameweek
        required_cols = ['name', 'gw']
        if not all(col in df.columns for col in required_cols):
            logger.warning(f"Missing required columns in {season}. Columns: {df.columns.tolist()}")
            return 0

        # Deduplicate: keep first occurrence of each player/gameweek combo
        df = df.drop_duplicates(subset=['name', 'gw'], keep='first')
        logger.info(f"Processing {len(df)} unique player/gameweek records for {season}")

        # Track combinations we've processed in this batch
        seen_in_batch: set[tuple[int, int]] = set()  # (player_id, gameweek)

        # Convert to list of dicts for easier processing
        records = df.to_dict('records')

        for row in records:
            try:
                player_name = str(row.get('name', '')).strip()
                gw_val = row.get('gw')
                # Handle NaN, None, empty string
                if pd.isna(gw_val) or gw_val is None or gw_val == '':
                    continue
                gameweek = int(float(gw_val))  # float() handles string decimals

                if not player_name or gameweek == 0:
                    continue

                # Get or create player
                player_id = self._get_or_create_player(player_name, row, season)
                if not player_id:
                    continue

                # Check if we've already processed this in the current batch
                batch_key = (player_id, gameweek)
                if batch_key in seen_in_batch:
                    continue
                seen_in_batch.add(batch_key)

                # Check if performance already exists in database
                existing = self.session.execute(
                    select(PlayerMatchPerformance)
                    .where(PlayerMatchPerformance.player_id == player_id)
                    .where(PlayerMatchPerformance.season == season)
                    .where(PlayerMatchPerformance.gameweek == gameweek)
                ).scalar_one_or_none()

                if existing:
                    continue  # Skip duplicates

                # Helper to safely get int values
                def safe_int(val, default=0):
                    try:
                        return int(val) if val is not None and val != '' else default
                    except (ValueError, TypeError):
                        return default

                def safe_decimal(val, default='0'):
                    try:
                        return Decimal(str(val)) if val is not None and val != '' else Decimal(default)
                    except:
                        return Decimal(default)

                # Create performance record
                was_home_val = row.get('is_home', row.get('was_home', True))
                if isinstance(was_home_val, str):
                    was_home_val = was_home_val.lower() == 'true'

                perf = PlayerMatchPerformance(
                    player_id=player_id,
                    season=season,
                    gameweek=gameweek,
                    was_home=bool(was_home_val),
                    minutes=safe_int(row.get('minutes')),
                    total_points=safe_int(row.get('total_points')),
                    bonus=safe_int(row.get('bonus')),
                    bps=safe_int(row.get('bps')),
                    goals_scored=safe_int(row.get('goals_scored')),
                    assists=safe_int(row.get('assists')),
                    clean_sheets=safe_int(row.get('clean_sheets')),
                    goals_conceded=safe_int(row.get('goals_conceded')),
                    influence=safe_decimal(row.get('influence')),
                    creativity=safe_decimal(row.get('creativity')),
                    threat=safe_decimal(row.get('threat')),
                    ict_index=safe_decimal(row.get('ict', row.get('ict_index'))),
                    expected_goals=safe_decimal(row.get('expected_goals', row.get('xg'))),
                    expected_assists=safe_decimal(row.get('expected_assists', row.get('xa'))),
                    expected_goal_involvements=safe_decimal(row.get('expected_goal_involvements', row.get('xgi'))),
                    value=safe_decimal(safe_int(row.get('value')) / 10) if row.get('value') else Decimal('0'),
                    selected=safe_int(row.get('selected', row.get('transfers_balance'))),
                )

                # Try to map opponent team
                opponent = row.get('opponent', row.get('opponent_team', ''))
                if opponent and str(opponent) in self._team_cache:
                    perf.opponent_team_id = self._team_cache[str(opponent)]

                self.session.add(perf)
                records_imported += 1

                # Commit in batches
                if records_imported % 1000 == 0:
                    self.session.flush()
                    logger.info(f"Processed {records_imported} records for {season}...")

            except Exception as e:
                logger.debug(f"Error processing row: {e}")
                continue

        return records_imported

    def _get_or_create_player(self, name: str, row: dict, season: str) -> Optional[int]:
        """Get or create a player record."""
        cache_key = f"{name}_{season}"

        if cache_key in self._player_cache:
            return self._player_cache[cache_key]

        # Try to find existing player by name
        existing = self.session.execute(
            select(Player).where(Player.name == name)
        ).scalar_one_or_none()

        if existing:
            self._player_cache[cache_key] = existing.id
            return existing.id

        # Also try by web_name
        web_name = name.split()[-1] if ' ' in name else name
        existing = self.session.execute(
            select(Player).where(Player.web_name == web_name)
        ).scalar_one_or_none()

        if existing:
            self._player_cache[cache_key] = existing.id
            return existing.id

        # Create new player (historical, may not be in current FPL)
        # Use negative fpl_id for historical players without current FPL presence
        max_fpl_id = self.session.execute(
            text("SELECT COALESCE(MIN(fpl_id), 0) FROM players WHERE fpl_id < 0")
        ).scalar() or 0
        new_fpl_id = min(max_fpl_id - 1, -1)

        # Determine position from row if available
        position = "MID"  # Default
        pos_val = row.get('position', row.get('element_type', ''))
        if pos_val:
            pos_map = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD',
                      'GKP': 'GKP', 'DEF': 'DEF', 'MID': 'MID', 'FWD': 'FWD',
                      'GK': 'GKP', 'Goalkeeper': 'GKP', 'Defender': 'DEF',
                      'Midfielder': 'MID', 'Forward': 'FWD'}
            position = pos_map.get(pos_val, pos_map.get(str(pos_val), 'MID'))

        player = Player(
            fpl_id=new_fpl_id,
            name=name,
            web_name=web_name,
            position=position,
            price=Decimal(str((row.get('value', 50) or 50) / 10)),
            total_points=0,
            status='u',  # Unavailable (historical)
        )

        # Try to map team
        team_name = row.get('team', row.get('team_name', ''))
        if team_name and str(team_name) in self._team_cache:
            player.team_id = self._team_cache[str(team_name)]

        self.session.add(player)
        self.session.flush()

        self._player_cache[cache_key] = player.id
        return player.id


def run_historical_import(seasons: list[str] = None):
    """Entry point for historical FPL import."""
    with SyncSessionLocal() as session:
        importer = FPLHistoricalImporter(session)
        return importer.run(seasons)


if __name__ == "__main__":
    import sys

    # Allow specifying seasons via command line
    seasons = sys.argv[1:] if len(sys.argv) > 1 else None

    result = run_historical_import(seasons)
    print(f"Import completed: {result}")
