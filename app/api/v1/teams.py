from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from app.core.config import get_settings
from app.db.database import get_async_session
from app.db.models import EloRating, Team, TeamStats
from app.schemas.team import TeamResponse, TeamStatsResponse

router = APIRouter()
settings = get_settings()


@router.get("/teams", response_model=list[TeamResponse])
async def list_teams(
    session: AsyncSession = Depends(get_async_session),
):
    """Get all EPL teams."""
    stmt = select(Team).order_by(Team.name)
    result = await session.execute(stmt)
    teams = result.scalars().all()

    return [TeamResponse.model_validate(t) for t in teams]


@router.get("/team/{team_id}", response_model=TeamResponse)
async def get_team(
    team_id: int,
    session: AsyncSession = Depends(get_async_session),
):
    """Get a team by ID."""
    stmt = select(Team).where(Team.id == team_id)
    result = await session.execute(stmt)
    team = result.scalar_one_or_none()

    if not team:
        raise HTTPException(status_code=404, detail=f"Team {team_id} not found")

    return TeamResponse.model_validate(team)


@router.get("/team/{team_id}/stats", response_model=TeamStatsResponse)
async def get_team_stats(
    team_id: int,
    season: Optional[str] = Query(default=None),
    matchweek: Optional[int] = Query(default=None, ge=1, le=38),
    session: AsyncSession = Depends(get_async_session),
):
    """Get team statistics for a specific season and matchweek."""
    target_season = season or settings.current_season

    # Get team
    team_stmt = select(Team).where(Team.id == team_id)
    result = await session.execute(team_stmt)
    team = result.scalar_one_or_none()

    if not team:
        raise HTTPException(status_code=404, detail=f"Team {team_id} not found")

    # Get latest stats if matchweek not specified
    stats_stmt = (
        select(TeamStats)
        .where(TeamStats.team_id == team_id)
        .where(TeamStats.season == target_season)
    )

    if matchweek:
        stats_stmt = stats_stmt.where(TeamStats.matchweek == matchweek)
    else:
        stats_stmt = stats_stmt.order_by(TeamStats.matchweek.desc()).limit(1)

    result = await session.execute(stats_stmt)
    stats = result.scalar_one_or_none()

    if not stats:
        raise HTTPException(
            status_code=404,
            detail=f"Stats not found for team {team_id} in season {target_season}",
        )

    # Get current ELO rating
    elo_stmt = (
        select(EloRating)
        .where(EloRating.team_id == team_id)
        .where(EloRating.season == target_season)
        .order_by(EloRating.matchweek.desc())
        .limit(1)
    )
    result = await session.execute(elo_stmt)
    elo = result.scalar_one_or_none()

    # Format home/away records
    home_record = f"{stats.home_wins}-{stats.home_draws}-{stats.home_losses}"
    away_record = f"{stats.away_wins}-{stats.away_draws}-{stats.away_losses}"

    # Parse injuries
    injuries_list = None
    if stats.injuries:
        injuries_list = stats.injuries.get("players", []) if isinstance(stats.injuries, dict) else []

    return TeamStatsResponse(
        team=TeamResponse.model_validate(team),
        season=stats.season,
        matchweek=stats.matchweek,
        form=stats.form,
        form_points=stats.form_points,
        goals_scored=stats.goals_scored,
        goals_conceded=stats.goals_conceded,
        avg_goals_scored=stats.avg_goals_scored,
        avg_goals_conceded=stats.avg_goals_conceded,
        xg_for=stats.xg_for,
        xg_against=stats.xg_against,
        home_record=home_record,
        away_record=away_record,
        clean_sheets=stats.clean_sheets,
        failed_to_score=stats.failed_to_score,
        elo_rating=elo.rating if elo else None,
        injuries=injuries_list,
    )


@router.get("/team/name/{team_name}", response_model=TeamResponse)
async def get_team_by_name(
    team_name: str,
    session: AsyncSession = Depends(get_async_session),
):
    """Get a team by name (case-insensitive partial match)."""
    stmt = select(Team).where(Team.name.ilike(f"%{team_name}%"))
    result = await session.execute(stmt)
    teams = result.scalars().all()

    if not teams:
        raise HTTPException(status_code=404, detail=f"Team matching '{team_name}' not found")

    if len(teams) > 1:
        team_names = [t.name for t in teams]
        raise HTTPException(
            status_code=400,
            detail=f"Multiple teams match '{team_name}': {team_names}. Please be more specific.",
        )

    return TeamResponse.model_validate(teams[0])


@router.get("/team/{team_id}/elo-history", response_model=list[dict])
async def get_team_elo_history(
    team_id: int,
    season: Optional[str] = Query(default=None),
    session: AsyncSession = Depends(get_async_session),
):
    """Get ELO rating history for a team."""
    target_season = season or settings.current_season

    # Verify team exists
    team_stmt = select(Team).where(Team.id == team_id)
    result = await session.execute(team_stmt)
    team = result.scalar_one_or_none()

    if not team:
        raise HTTPException(status_code=404, detail=f"Team {team_id} not found")

    # Get ELO history
    stmt = (
        select(EloRating)
        .where(EloRating.team_id == team_id)
        .where(EloRating.season == target_season)
        .order_by(EloRating.matchweek)
    )

    result = await session.execute(stmt)
    elo_records = result.scalars().all()

    return [
        {
            "matchweek": e.matchweek,
            "rating": float(e.rating),
            "change": float(e.rating_change) if e.rating_change else 0,
        }
        for e in elo_records
    ]
