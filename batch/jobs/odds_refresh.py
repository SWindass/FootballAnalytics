"""Odds refresh batch job - runs Saturday 8AM.

Captures final odds before weekend matches and detects value bets.
"""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional

import structlog
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.database import SyncSessionLocal
from app.db.models import Match, MatchAnalysis, MatchStatus, OddsHistory, Team, ValueBet, TeamStats, EloRating
from batch.betting.value_detector import ValueDetector, ValueDetectorConfig
from batch.data_sources.the_odds_api import OddsApiClient, match_team_names, parse_odds

logger = structlog.get_logger()
settings = get_settings()


class OddsRefreshJob:
    """Refreshes odds and detects value bets."""

    def __init__(self, session: Optional[Session] = None):
        self.session = session or SyncSessionLocal()
        self.odds_client = OddsApiClient()
        # Use optimized config: away wins with 5-12% edge (backtest shows +20% ROI)
        self.value_detector = ValueDetector(ValueDetectorConfig(
            min_edge=0.05,
            max_edge=0.12,
            allowed_outcomes=["away_win"],  # Focus on profitable outcome
            min_confidence=0.50,
            min_models_agreeing=1,
        ))

    def run(self) -> dict:
        """Execute the odds refresh job.

        Returns:
            Summary of job results
        """
        logger.info("Starting odds refresh job")
        start_time = datetime.utcnow()

        try:
            # 1. Fetch current odds from API
            odds_data = asyncio.run(self._fetch_odds())
            logger.info(f"Fetched odds for {len(odds_data)} events")

            # 2. Load teams for name matching
            teams = self._load_teams()

            # 3. Match odds to fixtures and store
            odds_stored = self._store_odds(odds_data, teams)
            self.session.flush()  # Ensure odds are visible for value bet detection
            logger.info(f"Stored odds for {odds_stored} matches")

            # 4. Detect value bets
            value_bets_found = self._detect_value_bets()
            logger.info(f"Found {value_bets_found} value bets")

            self.session.commit()

            duration = (datetime.utcnow() - start_time).total_seconds()
            api_remaining = self.odds_client.requests_remaining

            logger.info(
                "Odds refresh completed",
                events_fetched=len(odds_data),
                odds_stored=odds_stored,
                value_bets=value_bets_found,
                api_remaining=api_remaining,
                duration_seconds=duration,
            )

            return {
                "status": "success",
                "events_fetched": len(odds_data),
                "odds_stored": odds_stored,
                "value_bets_found": value_bets_found,
                "api_requests_remaining": api_remaining,
                "duration_seconds": duration,
            }

        except Exception as e:
            logger.error("Odds refresh job failed", error=str(e))
            self.session.rollback()
            raise

    async def _fetch_odds(self) -> list[dict]:
        """Fetch odds from The Odds API."""
        return await self.odds_client.get_odds(
            markets="h2h,totals",
            regions="uk",
        )

    def _load_teams(self) -> list[dict]:
        """Load teams for name matching."""
        stmt = select(Team)
        result = self.session.execute(stmt)
        teams = result.scalars().all()
        return [{"id": t.id, "name": t.name, "short_name": t.short_name} for t in teams]

    def _store_odds(self, odds_data: list[dict], teams: list[dict]) -> int:
        """Store odds in database."""
        stored = 0
        now = datetime.utcnow()

        for event in odds_data:
            # Parse all odds records for this event
            odds_records = parse_odds(event)

            if not odds_records:
                continue

            # Match to our fixture
            home_team = event.get("home_team", "")
            away_team = event.get("away_team", "")
            home_id, away_id = match_team_names(home_team, away_team, teams)

            if not home_id or not away_id:
                logger.debug(f"Could not match teams: {home_team} vs {away_team}")
                continue

            # Find fixture
            commence = event.get("commence_time")
            if commence:
                commence = datetime.fromisoformat(commence.replace("Z", "+00:00"))

            stmt = (
                select(Match)
                .where(Match.home_team_id == home_id)
                .where(Match.away_team_id == away_id)
                .where(Match.season == settings.current_season)
                .where(Match.status == MatchStatus.SCHEDULED)
            )
            match = self.session.execute(stmt).scalar_one_or_none()

            if not match:
                logger.debug(f"No fixture found for {home_team} vs {away_team}")
                continue

            # Store each bookmaker's odds
            for record in odds_records:
                odds_entry = OddsHistory(
                    match_id=match.id,
                    bookmaker=record["bookmaker"],
                    market="1x2",
                    recorded_at=now,
                    home_odds=record.get("home_odds"),
                    draw_odds=record.get("draw_odds"),
                    away_odds=record.get("away_odds"),
                    over_2_5_odds=record.get("over_2_5_odds"),
                    under_2_5_odds=record.get("under_2_5_odds"),
                )
                self.session.add(odds_entry)

            stored += 1

        return stored

    def _detect_value_bets(self) -> int:
        """Detect value betting opportunities.

        Uses optimized ValueDetector configured for away wins with 5-12% edge,
        which backtest shows has +20% ROI.
        """
        # Get upcoming matches with analysis
        now = datetime.utcnow()
        stmt = (
            select(Match)
            .where(Match.season == settings.current_season)
            .where(Match.status == MatchStatus.SCHEDULED)
            .where(Match.kickoff_time > now)
            .where(Match.kickoff_time < now + timedelta(days=7))
        )
        matches = list(self.session.execute(stmt).scalars().all())

        value_bets_created = 0

        for match in matches:
            # Get analysis
            analysis = self.session.execute(
                select(MatchAnalysis).where(MatchAnalysis.match_id == match.id)
            ).scalar_one_or_none()

            if not analysis or not analysis.consensus_home_prob:
                continue

            # Get latest odds
            stmt = (
                select(OddsHistory)
                .where(OddsHistory.match_id == match.id)
                .order_by(OddsHistory.recorded_at.desc())
            )
            odds_records = list(self.session.execute(stmt).scalars().all())

            # Get team stats for previous matchweek
            home_stats = self.session.execute(
                select(TeamStats)
                .where(TeamStats.team_id == match.home_team_id)
                .where(TeamStats.season == match.season)
                .where(TeamStats.matchweek == match.matchweek - 1)
            ).scalar_one_or_none()

            away_stats = self.session.execute(
                select(TeamStats)
                .where(TeamStats.team_id == match.away_team_id)
                .where(TeamStats.season == match.season)
                .where(TeamStats.matchweek == match.matchweek - 1)
            ).scalar_one_or_none()

            # Get ELO ratings
            home_elo = self.session.execute(
                select(EloRating)
                .where(EloRating.team_id == match.home_team_id)
                .where(EloRating.season == match.season)
                .where(EloRating.matchweek == match.matchweek - 1)
            ).scalar_one_or_none()

            away_elo = self.session.execute(
                select(EloRating)
                .where(EloRating.team_id == match.away_team_id)
                .where(EloRating.season == match.season)
                .where(EloRating.matchweek == match.matchweek - 1)
            ).scalar_one_or_none()

            # Detect value bets using optimized detector
            value_bets = self.value_detector.find_value_bets(
                match_id=match.id,
                analysis=analysis,
                home_stats=home_stats,
                away_stats=away_stats,
                home_elo=home_elo,
                away_elo=away_elo,
                odds_history=odds_records,
            )

            # Deactivate existing value bets for this match
            existing = list(self.session.execute(
                select(ValueBet).where(ValueBet.match_id == match.id).where(ValueBet.is_active == True)
            ).scalars().all())
            for eb in existing:
                eb.is_active = False

            # Store new value bets
            for vb in value_bets:
                new_bet = ValueBet(
                    match_id=vb.match_id,
                    outcome=vb.outcome,
                    bookmaker=vb.bookmaker,
                    model_probability=Decimal(str(round(vb.model_prob, 4))),
                    implied_probability=Decimal(str(round(vb.market_prob, 4))),
                    edge=Decimal(str(round(vb.edge, 4))),
                    odds=Decimal(str(round(vb.odds, 2))),
                    kelly_stake=Decimal(str(round(vb.kelly_stake, 4))),
                    recommended_stake=Decimal(str(round(vb.recommended_stake, 4))),
                    is_active=True,
                )
                self.session.add(new_bet)
                value_bets_created += 1

            if value_bets:
                logger.info(
                    f"Found {len(value_bets)} value bet(s) for match {match.id}",
                    outcomes=[vb.outcome for vb in value_bets],
                    edges=[f"{vb.edge:.1%}" for vb in value_bets],
                )

        return value_bets_created


def run_odds_refresh():
    """Entry point for odds refresh job."""
    with SyncSessionLocal() as session:
        job = OddsRefreshJob(session)
        return job.run()


if __name__ == "__main__":
    result = run_odds_refresh()
    print(f"Job completed: {result}")
