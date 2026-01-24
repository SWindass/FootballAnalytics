from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from app.core.config import get_settings
from app.db.database import get_async_session
from app.db.models import Match, MatchStatus, ValueBet
from app.schemas.team import TeamResponse
from app.schemas.value_bet import ValueBetMatchInfo, ValueBetResponse, ValueBetsListResponse

router = APIRouter()
settings = get_settings()


@router.get("/value-bets", response_model=ValueBetsListResponse)
async def get_value_bets(
    active_only: bool = Query(default=True),
    min_edge: Optional[float] = Query(default=None, ge=0.0, le=0.5),
    outcome: Optional[str] = Query(default=None),
    limit: int = Query(default=50, le=100),
    session: AsyncSession = Depends(get_async_session),
):
    """Get current value betting opportunities."""
    stmt = (
        select(ValueBet)
        .join(ValueBet.match)
        .where(Match.season == settings.current_season)
        .options(
            joinedload(ValueBet.match).joinedload(Match.home_team),
            joinedload(ValueBet.match).joinedload(Match.away_team),
        )
    )

    if active_only:
        stmt = stmt.where(ValueBet.is_active == True)
        stmt = stmt.where(Match.status == MatchStatus.SCHEDULED)

    if min_edge is not None:
        stmt = stmt.where(ValueBet.edge >= min_edge)

    if outcome:
        stmt = stmt.where(ValueBet.outcome == outcome)

    stmt = stmt.order_by(ValueBet.edge.desc()).limit(limit)

    result = await session.execute(stmt)
    value_bets = result.unique().scalars().all()

    response_bets = []
    for vb in value_bets:
        match_info = ValueBetMatchInfo(
            id=vb.match.id,
            home_team=TeamResponse.model_validate(vb.match.home_team),
            away_team=TeamResponse.model_validate(vb.match.away_team),
            kickoff_time=vb.match.kickoff_time,
            matchweek=vb.match.matchweek,
        )
        response_bets.append(
            ValueBetResponse(
                id=vb.id,
                match=match_info,
                outcome=vb.outcome,
                bookmaker=vb.bookmaker,
                model_probability=vb.model_probability,
                implied_probability=vb.implied_probability,
                edge=vb.edge,
                odds=vb.odds,
                kelly_stake=vb.kelly_stake,
                recommended_stake=vb.recommended_stake,
                is_active=vb.is_active,
                result=vb.result,
                created_at=vb.created_at,
            )
        )

    # Count totals
    count_stmt = select(ValueBet).join(ValueBet.match).where(Match.season == settings.current_season)
    result = await session.execute(count_stmt)
    all_bets = result.scalars().all()

    return ValueBetsListResponse(
        value_bets=response_bets,
        total_count=len(all_bets),
        active_count=sum(1 for b in all_bets if b.is_active),
        generated_at=datetime.utcnow(),
    )


@router.get("/value-bets/{bet_id}", response_model=ValueBetResponse)
async def get_value_bet(
    bet_id: int,
    session: AsyncSession = Depends(get_async_session),
):
    """Get a specific value bet by ID."""
    stmt = (
        select(ValueBet)
        .where(ValueBet.id == bet_id)
        .options(
            joinedload(ValueBet.match).joinedload(Match.home_team),
            joinedload(ValueBet.match).joinedload(Match.away_team),
        )
    )

    result = await session.execute(stmt)
    vb = result.unique().scalar_one_or_none()

    if not vb:
        raise HTTPException(status_code=404, detail=f"Value bet {bet_id} not found")

    match_info = ValueBetMatchInfo(
        id=vb.match.id,
        home_team=TeamResponse.model_validate(vb.match.home_team),
        away_team=TeamResponse.model_validate(vb.match.away_team),
        kickoff_time=vb.match.kickoff_time,
        matchweek=vb.match.matchweek,
    )

    return ValueBetResponse(
        id=vb.id,
        match=match_info,
        outcome=vb.outcome,
        bookmaker=vb.bookmaker,
        model_probability=vb.model_probability,
        implied_probability=vb.implied_probability,
        edge=vb.edge,
        odds=vb.odds,
        kelly_stake=vb.kelly_stake,
        recommended_stake=vb.recommended_stake,
        is_active=vb.is_active,
        result=vb.result,
        created_at=vb.created_at,
    )


@router.get("/value-bets/match/{match_id}", response_model=list[ValueBetResponse])
async def get_match_value_bets(
    match_id: int,
    session: AsyncSession = Depends(get_async_session),
):
    """Get all value bets for a specific match."""
    stmt = (
        select(ValueBet)
        .where(ValueBet.match_id == match_id)
        .options(
            joinedload(ValueBet.match).joinedload(Match.home_team),
            joinedload(ValueBet.match).joinedload(Match.away_team),
        )
        .order_by(ValueBet.edge.desc())
    )

    result = await session.execute(stmt)
    value_bets = result.unique().scalars().all()

    response_bets = []
    for vb in value_bets:
        match_info = ValueBetMatchInfo(
            id=vb.match.id,
            home_team=TeamResponse.model_validate(vb.match.home_team),
            away_team=TeamResponse.model_validate(vb.match.away_team),
            kickoff_time=vb.match.kickoff_time,
            matchweek=vb.match.matchweek,
        )
        response_bets.append(
            ValueBetResponse(
                id=vb.id,
                match=match_info,
                outcome=vb.outcome,
                bookmaker=vb.bookmaker,
                model_probability=vb.model_probability,
                implied_probability=vb.implied_probability,
                edge=vb.edge,
                odds=vb.odds,
                kelly_stake=vb.kelly_stake,
                recommended_stake=vb.recommended_stake,
                is_active=vb.is_active,
                result=vb.result,
                created_at=vb.created_at,
            )
        )

    return response_bets
