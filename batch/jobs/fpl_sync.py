"""FPL data sync job.

Fetches player data from Fantasy Premier League API and syncs to database.
Run weekly to keep player stats, form, and availability up to date.
"""

import asyncio
from datetime import datetime
from decimal import Decimal
from typing import Optional

import structlog
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.database import SyncSessionLocal
from app.db.models import Player, PlayerMatchPerformance, Team
from batch.data_sources.fpl import FPLClient, parse_fpl_player, parse_fpl_match_performance

logger = structlog.get_logger()
settings = get_settings()


# Mapping of FPL team names to our team names (for matching)
FPL_TEAM_NAME_MAP = {
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
}


class FPLSyncJob:
    """Syncs FPL player data to database."""

    def __init__(self, session: Optional[Session] = None):
        self.session = session or SyncSessionLocal()
        self.client = FPLClient()
        self._team_cache: dict[int, int] = {}  # FPL team_id -> our team_id
        self._fpl_teams: dict[int, str] = {}  # FPL team_id -> team name

    def run(self) -> dict:
        """Execute the FPL sync job."""
        logger.info("Starting FPL sync job")
        start_time = datetime.utcnow()

        try:
            # 1. Build team mapping
            self._build_team_mapping()

            # 2. Sync all players
            players_synced = asyncio.run(self._sync_players())
            logger.info(f"Synced {players_synced} players")

            # 3. Sync recent match performances (optional - can be slow)
            # performances_synced = asyncio.run(self._sync_performances())

            self.session.commit()

            duration = (datetime.utcnow() - start_time).total_seconds()
            logger.info(
                "FPL sync completed",
                players_synced=players_synced,
                duration_seconds=duration,
            )

            return {
                "status": "success",
                "players_synced": players_synced,
                "duration_seconds": duration,
            }

        except Exception as e:
            logger.error("FPL sync failed", error=str(e))
            self.session.rollback()
            raise

    def _build_team_mapping(self):
        """Build mapping between FPL team IDs and our team IDs."""
        # Get FPL teams
        fpl_teams = asyncio.run(self.client.get_all_teams())
        self._fpl_teams = {t["id"]: t["name"] for t in fpl_teams}

        # Get our teams
        our_teams = list(self.session.execute(select(Team)).scalars().all())

        # Build mapping
        for fpl_id, fpl_name in self._fpl_teams.items():
            our_name = FPL_TEAM_NAME_MAP.get(fpl_name)
            if our_name:
                for team in our_teams:
                    if team.name == our_name:
                        self._team_cache[fpl_id] = team.id
                        # Update team's fpl_id if not set
                        if team.fpl_id != fpl_id:
                            team.fpl_id = fpl_id
                        break

        logger.info(f"Mapped {len(self._team_cache)} teams between FPL and our database")

    async def _sync_players(self) -> int:
        """Sync all players from FPL."""
        players = await self.client.get_all_players()
        synced = 0

        for player_data in players:
            parsed = parse_fpl_player(player_data, self._fpl_teams)

            # Find existing player or create new
            existing = self.session.execute(
                select(Player).where(Player.fpl_id == parsed["fpl_id"])
            ).scalar_one_or_none()

            if existing:
                player = existing
            else:
                player = Player(fpl_id=parsed["fpl_id"])
                self.session.add(player)

            # Update fields
            player.name = parsed["name"]
            player.web_name = parsed["web_name"]
            player.team_id = self._team_cache.get(parsed["team_fpl_id"])
            player.position = parsed["position"]
            player.price = Decimal(str(parsed["price"]))
            player.selected_by_percent = Decimal(str(parsed["selected_by_percent"]))
            player.total_points = parsed["total_points"]
            player.points_per_game = Decimal(str(parsed["points_per_game"]))
            player.minutes = parsed["minutes"]
            player.starts = parsed["starts"]
            player.goals_scored = parsed["goals_scored"]
            player.assists = parsed["assists"]
            player.clean_sheets = parsed["clean_sheets"]
            player.goals_conceded = parsed["goals_conceded"]
            player.form = Decimal(str(parsed["form"]))
            player.influence = Decimal(str(parsed["influence"]))
            player.creativity = Decimal(str(parsed["creativity"]))
            player.threat = Decimal(str(parsed["threat"]))
            player.ict_index = Decimal(str(parsed["ict_index"]))
            player.expected_goals = Decimal(str(parsed["expected_goals"]))
            player.expected_assists = Decimal(str(parsed["expected_assists"]))
            player.expected_goal_involvements = Decimal(str(parsed["expected_goal_involvements"]))
            player.expected_goals_conceded = Decimal(str(parsed["expected_goals_conceded"]))
            player.status = parsed["status"]
            player.chance_of_playing = parsed["chance_of_playing"]
            player.news = parsed["news"]

            synced += 1

        return synced

    async def _sync_performances(self, top_n_players: int = 100) -> int:
        """Sync match performances for top players.

        This is slower as it requires individual API calls per player.
        Only sync top N players by total points to save API calls.
        """
        # Get top players by points
        stmt = (
            select(Player)
            .order_by(Player.total_points.desc())
            .limit(top_n_players)
        )
        top_players = list(self.session.execute(stmt).scalars().all())

        synced = 0
        for player in top_players:
            try:
                history = await self.client.get_player_history(player.fpl_id)

                for entry in history:
                    parsed = parse_fpl_match_performance(entry)

                    # Check if already exists
                    existing = self.session.execute(
                        select(PlayerMatchPerformance)
                        .where(PlayerMatchPerformance.player_id == player.id)
                        .where(PlayerMatchPerformance.season == settings.current_season)
                        .where(PlayerMatchPerformance.gameweek == parsed["gameweek"])
                    ).scalar_one_or_none()

                    if existing:
                        perf = existing
                    else:
                        perf = PlayerMatchPerformance(
                            player_id=player.id,
                            season=settings.current_season,
                            gameweek=parsed["gameweek"],
                        )
                        self.session.add(perf)

                    # Update fields
                    perf.opponent_team_id = self._team_cache.get(parsed["opponent_team_fpl_id"])
                    perf.was_home = parsed["was_home"]
                    perf.minutes = parsed["minutes"]
                    perf.total_points = parsed["total_points"]
                    perf.bonus = parsed["bonus"]
                    perf.bps = parsed["bps"]
                    perf.goals_scored = parsed["goals_scored"]
                    perf.assists = parsed["assists"]
                    perf.clean_sheets = parsed["clean_sheets"]
                    perf.goals_conceded = parsed["goals_conceded"]
                    perf.influence = Decimal(str(parsed["influence"]))
                    perf.creativity = Decimal(str(parsed["creativity"]))
                    perf.threat = Decimal(str(parsed["threat"]))
                    perf.ict_index = Decimal(str(parsed["ict_index"]))
                    perf.expected_goals = Decimal(str(parsed["expected_goals"]))
                    perf.expected_assists = Decimal(str(parsed["expected_assists"]))
                    perf.expected_goal_involvements = Decimal(str(parsed["expected_goal_involvements"]))
                    perf.value = Decimal(str(parsed["value"]))
                    perf.selected = parsed["selected"]

                    synced += 1

                # Small delay between player requests
                await asyncio.sleep(0.5)

            except Exception as e:
                logger.warning(f"Failed to sync performances for {player.web_name}: {e}")
                continue

        return synced


def run_fpl_sync():
    """Entry point for FPL sync job."""
    with SyncSessionLocal() as session:
        job = FPLSyncJob(session)
        return job.run()


if __name__ == "__main__":
    result = run_fpl_sync()
    print(f"Job completed: {result}")
