"""Synchronize player team mappings from historical FPL data.

This script updates team_id for all players using the FPL CSV data,
ensuring every player has their correct team assignment.
"""

import asyncio
import io
from collections import defaultdict
from datetime import datetime

import httpx
import pandas as pd
import structlog
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.database import SyncSessionLocal
from app.db.models import Player, Team

logger = structlog.get_logger()

# GitHub raw content base URL
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"

# Seasons to process (newest first for most recent team assignment)
SEASONS = [
    "2024-25",
    "2023-24",
    "2022-23",
    "2021-22",
    "2020-21",
    "2019-20",
    "2018-19",
    "2017-18",
    "2016-17",
]

# Map FPL team names to our team names
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
    # Historical teams
    "Burnley": "Burnley FC",
    "Cardiff": "Cardiff City FC",
    "Huddersfield": "Huddersfield Town AFC",
    "Hull": "Hull City AFC",
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


class PlayerTeamSyncer:
    """Synchronizes player team mappings from FPL CSV data."""

    def __init__(self, session: Session | None = None):
        self.session = session or SyncSessionLocal()
        self._team_cache: dict[str, int] = {}  # FPL team name -> our team_id
        self._player_teams: dict[str, str] = {}  # player name -> FPL team name

    def run(self) -> dict:
        """Sync all player-team mappings."""
        logger.info("Starting player-team sync")
        start_time = datetime.utcnow()

        try:
            # Build team mapping from database
            self._build_team_mapping()

            # Get current state
            players_before = self._count_players_with_teams()

            # Download FPL data and build player-team mapping
            asyncio.run(self._collect_player_teams())

            # Update players in database
            updated_count = self._update_player_teams()

            # Get final state
            players_after = self._count_players_with_teams()

            duration = (datetime.utcnow() - start_time).total_seconds()

            result = {
                "status": "success",
                "players_before": players_before["with_team"],
                "players_after": players_after["with_team"],
                "players_updated": updated_count,
                "total_players": players_after["total"],
                "coverage_before": f"{players_before['with_team']/players_before['total']*100:.1f}%",
                "coverage_after": f"{players_after['with_team']/players_after['total']*100:.1f}%",
                "duration_seconds": duration,
            }

            logger.info("Player-team sync completed", **result)
            return result

        except Exception as e:
            logger.error("Player-team sync failed", error=str(e))
            self.session.rollback()
            raise

    def _build_team_mapping(self):
        """Build mapping from FPL team names to our team IDs."""
        teams = list(self.session.execute(select(Team)).scalars().all())

        for team in teams:
            self._team_cache[team.name] = team.id
            # Also map by FPL short names
            for fpl_name, our_name in FPL_TEAM_NAME_MAP.items():
                if our_name == team.name:
                    self._team_cache[fpl_name] = team.id

        logger.info(f"Built team mapping with {len(self._team_cache)} entries")

    def _count_players_with_teams(self) -> dict:
        """Count players with and without team assignments."""
        total = self.session.execute(
            select(Player)
        ).scalars().all()

        with_team = [p for p in total if p.team_id is not None]

        return {
            "total": len(total),
            "with_team": len(with_team),
            "without_team": len(total) - len(with_team),
        }

    async def _collect_player_teams(self):
        """Download FPL CSVs and collect player-team mappings."""
        async with httpx.AsyncClient() as client:
            for season in SEASONS:
                # First try merged_gw.csv with 'team' column
                url = f"{GITHUB_RAW_BASE}/{season}/gws/merged_gw.csv"
                logger.info(f"Downloading {season} data from {url}")

                try:
                    response = await client.get(url, timeout=60)
                    if response.status_code == 404:
                        # Try players_raw.csv instead
                        await self._collect_from_players_raw(season, client)
                        continue

                    response.raise_for_status()

                    # Check if 'team' column exists
                    df = pd.read_csv(io.StringIO(response.text))
                    df.columns = df.columns.str.lower().str.strip()

                    if 'team' in df.columns:
                        self._process_csv(response.text, season)
                    else:
                        # Fall back to players_raw.csv for older seasons
                        logger.info(f"No 'team' column in merged_gw.csv for {season}, trying players_raw.csv")
                        await self._collect_from_players_raw(season, client)

                except Exception as e:
                    logger.warning(f"Failed to process {season}: {e}")
                    continue

        logger.info(f"Collected team mappings for {len(self._player_teams)} players")

    async def _collect_from_players_raw(self, season: str, client: httpx.AsyncClient):
        """Collect player-team mappings from players_raw.csv and master_team_list.csv."""
        try:
            # Get master team list mapping for this season
            master_url = f"{GITHUB_RAW_BASE}/master_team_list.csv"
            master_response = await client.get(master_url, timeout=30)
            if master_response.status_code != 200:
                logger.warning("No master_team_list.csv found")
                return

            master_df = pd.read_csv(io.StringIO(master_response.text))
            master_df.columns = master_df.columns.str.lower().str.strip()

            # Build FPL team_id -> team_name mapping for this season
            fpl_team_map = {}
            season_data = master_df[master_df['season'] == season]
            for _, row in season_data.iterrows():
                team_id = int(row['team'])
                team_name = str(row['team_name'])
                fpl_team_map[team_id] = team_name

            if not fpl_team_map:
                logger.warning(f"No team mappings found in master_team_list for {season}")
                return

            # Get players
            players_url = f"{GITHUB_RAW_BASE}/{season}/players_raw.csv"
            players_response = await client.get(players_url, timeout=60)
            if players_response.status_code != 200:
                logger.warning(f"No players_raw.csv found for {season}")
                return

            players_df = pd.read_csv(io.StringIO(players_response.text))
            players_df.columns = players_df.columns.str.lower().str.strip()

            if 'first_name' not in players_df.columns or 'second_name' not in players_df.columns:
                logger.warning(f"Missing name columns in players_raw.csv for {season}")
                return

            count_before = len(self._player_teams)

            for _, row in players_df.iterrows():
                first_name = str(row.get('first_name', '')).strip()
                second_name = str(row.get('second_name', '')).strip()
                player_name = f"{first_name} {second_name}".strip()

                team_id = row.get('team')
                if pd.isna(team_id):
                    continue

                team_name = fpl_team_map.get(int(team_id))

                if player_name and team_name and player_name not in self._player_teams:
                    self._player_teams[player_name] = team_name

            count_added = len(self._player_teams) - count_before
            logger.info(f"Processed {season} via players_raw.csv: added {count_added} mappings, total now {len(self._player_teams)}")

        except Exception as e:
            logger.warning(f"Failed to process players_raw.csv for {season}: {e}")

    def _process_csv(self, csv_content: str, season: str):
        """Process CSV and extract player-team mappings."""
        df = pd.read_csv(io.StringIO(csv_content))
        df.columns = df.columns.str.lower().str.strip()

        if 'name' not in df.columns or 'team' not in df.columns:
            logger.warning(f"Missing required columns in {season}")
            return

        # Get unique player-team pairs (use most recent appearance)
        for _, row in df[['name', 'team']].drop_duplicates().iterrows():
            player_name = str(row['name']).strip()
            team_name = str(row['team']).strip()

            if player_name and team_name and team_name != 'nan':
                # Only update if we don't have a mapping yet
                # (we process newest seasons first, so keep that)
                if player_name not in self._player_teams:
                    self._player_teams[player_name] = team_name

        logger.info(f"Processed {season}: {len(self._player_teams)} total player mappings")

    def _normalize_name(self, name: str) -> str:
        """Normalize player name for matching (handle underscores, case, trailing numbers)."""
        import re
        # Replace underscores with spaces
        name = name.replace('_', ' ').strip()
        # Remove trailing numbers (e.g., "Aaron Cresswell 402" -> "Aaron Cresswell")
        name = re.sub(r'\s+\d+$', '', name)
        return name.lower()

    def _strip_accents(self, name: str) -> str:
        """Remove accents and special characters for fuzzy matching."""
        import re
        import unicodedata
        # First, remove corrupted characters like '�' (replacement character)
        name = name.replace('�', '').replace('\ufffd', '')
        # Normalize unicode and decompose accented characters
        try:
            nfkd = unicodedata.normalize('NFKD', name)
            # Remove combining marks (accents)
            ascii_name = nfkd.encode('ASCII', 'ignore').decode('ASCII')
        except:
            ascii_name = name
        # Also remove any remaining special/corrupt characters
        ascii_name = re.sub(r'[^\w\s-]', '', ascii_name)
        # Remove multiple spaces and trailing spaces
        ascii_name = ' '.join(ascii_name.split())
        return ascii_name.lower().strip()

    def _update_player_teams(self) -> int:
        """Update team_id for all players based on collected mappings."""
        # Get all players
        players = list(self.session.execute(select(Player)).scalars().all())
        updated = 0
        missing_teams = set()

        # Build normalized lookup for CSV player-team mappings
        normalized_player_teams = {}
        accent_stripped_teams = {}  # For fuzzy matching
        for csv_name, team in self._player_teams.items():
            normalized_player_teams[self._normalize_name(csv_name)] = team
            accent_stripped_teams[self._strip_accents(csv_name)] = team

        # Also build web_name (last name) to team mapping, but only if unique
        web_name_counts = defaultdict(list)
        for csv_name, team in self._player_teams.items():
            parts = csv_name.split()
            if len(parts) > 1:
                web_name = parts[-1].lower()
                web_name_counts[web_name].append(team)
                # Also add accent-stripped web_name
                stripped_web_name = self._strip_accents(parts[-1])
                web_name_counts[stripped_web_name].append(team)

        # Only use web_name mappings that are unique
        unique_web_name_teams = {}
        for web_name, teams in web_name_counts.items():
            if len(set(teams)) == 1:  # All same team
                unique_web_name_teams[web_name] = teams[0]

        for player in players:
            # Skip if already has team
            if player.team_id is not None:
                continue

            # Try normalized name match (full name)
            norm_name = self._normalize_name(player.name)
            team_name = normalized_player_teams.get(norm_name)

            # Also try web_name (normalized) against full CSV names
            if not team_name:
                norm_web_name = self._normalize_name(player.web_name)
                team_name = normalized_player_teams.get(norm_web_name)

            # Try accent-stripped matching (for encoding issues)
            if not team_name:
                stripped_name = self._strip_accents(self._normalize_name(player.name))
                team_name = accent_stripped_teams.get(stripped_name)

            # Try unique web_name mapping (only if player's last name is unique)
            if not team_name:
                parts = player.name.replace('_', ' ').split()
                if len(parts) > 1:
                    last_name = parts[-1].lower()
                    team_name = unique_web_name_teams.get(last_name)
                    # Also try accent-stripped last name
                    if not team_name:
                        stripped_last = self._strip_accents(last_name)
                        team_name = unique_web_name_teams.get(stripped_last)

            # For corrupted names, try first name + partial last name matching
            if not team_name and '�' in player.name:
                parts = player.name.replace('_', ' ').split()
                if len(parts) >= 2:
                    first_name = parts[0].lower()
                    # Remove numbers from last part
                    import re
                    last_part = re.sub(r'\d+$', '', parts[-1]).lower()
                    # Strip the corruption char
                    last_part = last_part.replace('�', '')
                    # Find matches with same first name and last name starting with same letters
                    for csv_name, csv_team in self._player_teams.items():
                        csv_parts = csv_name.split()
                        if len(csv_parts) >= 2:
                            csv_first = csv_parts[0].lower()
                            csv_last = self._strip_accents(csv_parts[-1])
                            # Match if first name matches and last name starts similarly
                            if csv_first == first_name and len(last_part) >= 3:
                                if csv_last.startswith(last_part[:3]) or last_part.startswith(csv_last[:3]):
                                    team_name = csv_team
                                    break

            if team_name:
                # Map to our team_id
                team_id = self._team_cache.get(team_name)
                if team_id:
                    player.team_id = team_id
                    updated += 1
                else:
                    missing_teams.add(team_name)

        if missing_teams:
            logger.warning(f"Teams not found in database: {missing_teams}")

        self.session.commit()
        logger.info(f"Updated {updated} players with team assignments")
        return updated


def run_player_team_sync():
    """Entry point for player-team sync."""
    with SyncSessionLocal() as session:
        syncer = PlayerTeamSyncer(session)
        return syncer.run()


if __name__ == "__main__":
    result = run_player_team_sync()
    print("\nSync completed:")
    print(f"  Players with teams before: {result['players_before']} ({result['coverage_before']})")
    print(f"  Players with teams after:  {result['players_after']} ({result['coverage_after']})")
    print(f"  Players updated: {result['players_updated']}")
