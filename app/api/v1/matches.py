from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload, selectinload

from app.api.v1.matchweek import build_match_detail
from app.core.config import get_settings
from app.db.database import get_async_session
from app.db.models import Match, Team
from app.schemas.match import MatchDetailResponse, MatchResponse

router = APIRouter()
settings = get_settings()


@router.get("/match/{match_id}", response_model=MatchDetailResponse)
async def get_match(
    match_id: int,
    session: AsyncSession = Depends(get_async_session),
):
    """Get detailed information for a specific match."""
    stmt = (
        select(Match)
        .where(Match.id == match_id)
        .options(
            joinedload(Match.home_team),
            joinedload(Match.away_team),
            joinedload(Match.analysis),
            selectinload(Match.odds_history),
            selectinload(Match.value_bets),
        )
    )
    result = await session.execute(stmt)
    match = result.unique().scalar_one_or_none()

    if not match:
        raise HTTPException(status_code=404, detail=f"Match {match_id} not found")

    return build_match_detail(match)


@router.get("/matches", response_model=list[MatchResponse])
async def list_matches(
    season: Optional[str] = Query(default=None),
    matchweek: Optional[int] = Query(default=None, ge=1, le=38),
    team_id: Optional[int] = Query(default=None),
    status: Optional[str] = Query(default=None),
    limit: int = Query(default=50, le=100),
    offset: int = Query(default=0, ge=0),
    session: AsyncSession = Depends(get_async_session),
):
    """List matches with optional filters."""
    target_season = season or settings.current_season

    stmt = (
        select(Match)
        .where(Match.season == target_season)
        .options(
            joinedload(Match.home_team),
            joinedload(Match.away_team),
        )
    )

    if matchweek:
        stmt = stmt.where(Match.matchweek == matchweek)

    if team_id:
        stmt = stmt.where((Match.home_team_id == team_id) | (Match.away_team_id == team_id))

    if status:
        stmt = stmt.where(Match.status == status)

    stmt = stmt.order_by(Match.kickoff_time.desc()).offset(offset).limit(limit)

    result = await session.execute(stmt)
    matches = result.unique().scalars().all()

    return [MatchResponse.model_validate(m) for m in matches]


@router.get("/matches/upcoming", response_model=list[MatchResponse])
async def get_upcoming_matches(
    days: int = Query(default=7, ge=1, le=30),
    session: AsyncSession = Depends(get_async_session),
):
    """Get upcoming matches within the specified number of days."""
    from datetime import timedelta

    now = datetime.utcnow()
    end_date = now + timedelta(days=days)

    stmt = (
        select(Match)
        .where(Match.season == settings.current_season)
        .where(Match.status == "scheduled")
        .where(Match.kickoff_time >= now)
        .where(Match.kickoff_time <= end_date)
        .options(
            joinedload(Match.home_team),
            joinedload(Match.away_team),
        )
        .order_by(Match.kickoff_time)
    )

    result = await session.execute(stmt)
    matches = result.unique().scalars().all()

    return [MatchResponse.model_validate(m) for m in matches]


@router.get("/matches/head-to-head", response_model=list[MatchResponse])
async def get_head_to_head(
    team1_id: int = Query(...),
    team2_id: int = Query(...),
    limit: int = Query(default=10, le=20),
    session: AsyncSession = Depends(get_async_session),
):
    """Get head-to-head history between two teams."""
    if team1_id == team2_id:
        raise HTTPException(status_code=400, detail="Teams must be different")

    stmt = (
        select(Match)
        .where(
            ((Match.home_team_id == team1_id) & (Match.away_team_id == team2_id))
            | ((Match.home_team_id == team2_id) & (Match.away_team_id == team1_id))
        )
        .where(Match.status == "finished")
        .options(
            joinedload(Match.home_team),
            joinedload(Match.away_team),
        )
        .order_by(Match.kickoff_time.desc())
        .limit(limit)
    )

    result = await session.execute(stmt)
    matches = result.unique().scalars().all()

    return [MatchResponse.model_validate(m) for m in matches]
