from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload, selectinload

from app.core.config import get_settings
from app.db.database import get_async_session
from app.db.models import Match, MatchAnalysis, MatchStatus, OddsHistory, Team, TeamStats, ValueBet
from app.schemas.match import (
    AdditionalPredictions,
    MatchAnalysisResponse,
    MatchDetailResponse,
    MatchOdds,
    MatchweekResponse,
    ModelPredictions,
    PredictionProbabilities,
)
from app.schemas.team import TeamFormResponse, TeamResponse

router = APIRouter()
settings = get_settings()


def build_predictions(analysis: Optional[MatchAnalysis]) -> Optional[MatchAnalysisResponse]:
    """Build predictions response from analysis model."""
    if not analysis:
        return None

    return MatchAnalysisResponse(
        predictions=ModelPredictions(
            elo=PredictionProbabilities(
                home_win=analysis.elo_home_prob,
                draw=analysis.elo_draw_prob,
                away_win=analysis.elo_away_prob,
            )
            if analysis.elo_home_prob
            else None,
            poisson=PredictionProbabilities(
                home_win=analysis.poisson_home_prob,
                draw=analysis.poisson_draw_prob,
                away_win=analysis.poisson_away_prob,
            )
            if analysis.poisson_home_prob
            else None,
            xgboost=PredictionProbabilities(
                home_win=analysis.xgboost_home_prob,
                draw=analysis.xgboost_draw_prob,
                away_win=analysis.xgboost_away_prob,
            )
            if analysis.xgboost_home_prob
            else None,
            consensus=PredictionProbabilities(
                home_win=analysis.consensus_home_prob,
                draw=analysis.consensus_draw_prob,
                away_win=analysis.consensus_away_prob,
            )
            if analysis.consensus_home_prob
            else None,
        ),
        additional=AdditionalPredictions(
            over_2_5_prob=analysis.poisson_over_2_5_prob,
            btts_prob=analysis.poisson_btts_prob,
            predicted_home_goals=analysis.predicted_home_goals,
            predicted_away_goals=analysis.predicted_away_goals,
        ),
        confidence=analysis.confidence,
        narrative=analysis.narrative,
        narrative_generated_at=analysis.narrative_generated_at,
    )


def build_match_detail(
    match: Match,
    team_forms: dict[int, str] = None,
) -> MatchDetailResponse:
    """Build detailed match response.

    Args:
        match: Match object
        team_forms: Dict mapping team_id to form string (e.g., "WWDLW")
    """
    team_forms = team_forms or {}

    # Get latest odds per bookmaker
    latest_odds = []
    if match.odds_history:
        # Group by bookmaker and get latest
        bookmaker_odds: dict[str, OddsHistory] = {}
        for odds in sorted(match.odds_history, key=lambda x: x.recorded_at, reverse=True):
            if odds.bookmaker not in bookmaker_odds:
                bookmaker_odds[odds.bookmaker] = odds
        latest_odds = [
            MatchOdds(
                bookmaker=o.bookmaker,
                home_odds=o.home_odds,
                draw_odds=o.draw_odds,
                away_odds=o.away_odds,
                over_2_5_odds=o.over_2_5_odds,
                under_2_5_odds=o.under_2_5_odds,
                btts_yes_odds=o.btts_yes_odds,
                btts_no_odds=o.btts_no_odds,
                recorded_at=o.recorded_at,
            )
            for o in bookmaker_odds.values()
        ]

    # Build team responses with form
    home_team_data = TeamResponse.model_validate(match.home_team)
    home_team_data.form = team_forms.get(match.home_team_id)

    away_team_data = TeamResponse.model_validate(match.away_team)
    away_team_data.form = team_forms.get(match.away_team_id)

    return MatchDetailResponse(
        id=match.id,
        external_id=match.external_id,
        season=match.season,
        matchweek=match.matchweek,
        kickoff_time=match.kickoff_time,
        status=match.status,
        home_team=home_team_data,
        away_team=away_team_data,
        home_score=match.home_score,
        away_score=match.away_score,
        home_xg=match.home_xg,
        away_xg=match.away_xg,
        analysis=build_predictions(match.analysis),
        latest_odds=latest_odds,
        value_bets=[],  # Will be populated separately
    )


@router.get("/matchweek/current", response_model=MatchweekResponse)
async def get_current_matchweek(
    session: AsyncSession = Depends(get_async_session),
):
    """Get the current matchweek with all matches and analyses."""
    # Find the current/next matchweek
    now = datetime.utcnow()

    # First try to find an ongoing matchweek (has both finished and scheduled matches)
    # or the next upcoming matchweek
    stmt = (
        select(Match)
        .where(Match.season == settings.current_season)
        .where(Match.status.in_([MatchStatus.SCHEDULED, MatchStatus.IN_PLAY]))
        .order_by(Match.kickoff_time)
        .limit(1)
    )
    result = await session.execute(stmt)
    upcoming_match = result.scalar_one_or_none()

    if not upcoming_match:
        # No upcoming matches, get the latest matchweek
        stmt = (
            select(Match)
            .where(Match.season == settings.current_season)
            .order_by(Match.matchweek.desc())
            .limit(1)
        )
        result = await session.execute(stmt)
        latest_match = result.scalar_one_or_none()
        if not latest_match:
            raise HTTPException(status_code=404, detail="No matches found")
        current_matchweek = latest_match.matchweek
    else:
        current_matchweek = upcoming_match.matchweek

    # Fetch all matches for this matchweek with relationships
    stmt = (
        select(Match)
        .where(Match.season == settings.current_season)
        .where(Match.matchweek == current_matchweek)
        .options(
            joinedload(Match.home_team),
            joinedload(Match.away_team),
            joinedload(Match.analysis),
            selectinload(Match.odds_history),
            selectinload(Match.value_bets),
        )
        .order_by(Match.kickoff_time)
    )
    result = await session.execute(stmt)
    matches = result.unique().scalars().all()

    # Fetch team form data (latest available stats for each team)
    team_ids = set()
    for m in matches:
        team_ids.add(m.home_team_id)
        team_ids.add(m.away_team_id)

    # Get the most recent stats for each team
    from sqlalchemy import func
    subq = (
        select(
            TeamStats.team_id,
            func.max(TeamStats.matchweek).label("max_mw")
        )
        .where(TeamStats.season == settings.current_season)
        .where(TeamStats.team_id.in_(team_ids))
        .group_by(TeamStats.team_id)
        .subquery()
    )
    stmt = (
        select(TeamStats)
        .join(subq, (TeamStats.team_id == subq.c.team_id) & (TeamStats.matchweek == subq.c.max_mw))
        .where(TeamStats.season == settings.current_season)
    )
    result = await session.execute(stmt)
    team_stats = result.scalars().all()
    team_forms = {ts.team_id: ts.form for ts in team_stats if ts.form}

    match_details = [build_match_detail(m, team_forms) for m in matches]
    matches_with_value = sum(1 for m in matches if m.value_bets)

    return MatchweekResponse(
        season=settings.current_season,
        matchweek=current_matchweek,
        matches=match_details,
        total_matches=len(matches),
        matches_with_value_bets=matches_with_value,
        generated_at=datetime.utcnow(),
    )


@router.get("/matchweek/{matchweek}", response_model=MatchweekResponse)
async def get_matchweek(
    matchweek: int,
    season: Optional[str] = Query(default=None),
    session: AsyncSession = Depends(get_async_session),
):
    """Get a specific matchweek."""
    target_season = season or settings.current_season

    if not 1 <= matchweek <= 38:
        raise HTTPException(status_code=400, detail="Matchweek must be between 1 and 38")

    stmt = (
        select(Match)
        .where(Match.season == target_season)
        .where(Match.matchweek == matchweek)
        .options(
            joinedload(Match.home_team),
            joinedload(Match.away_team),
            joinedload(Match.analysis),
            selectinload(Match.odds_history),
            selectinload(Match.value_bets),
        )
        .order_by(Match.kickoff_time)
    )
    result = await session.execute(stmt)
    matches = result.unique().scalars().all()

    if not matches:
        raise HTTPException(status_code=404, detail=f"Matchweek {matchweek} not found")

    # Fetch team form data (latest available stats for each team)
    team_ids = set()
    for m in matches:
        team_ids.add(m.home_team_id)
        team_ids.add(m.away_team_id)

    # Get the most recent stats for each team up to this matchweek
    from sqlalchemy import func
    subq = (
        select(
            TeamStats.team_id,
            func.max(TeamStats.matchweek).label("max_mw")
        )
        .where(TeamStats.season == target_season)
        .where(TeamStats.team_id.in_(team_ids))
        .where(TeamStats.matchweek <= matchweek)
        .group_by(TeamStats.team_id)
        .subquery()
    )
    stmt = (
        select(TeamStats)
        .join(subq, (TeamStats.team_id == subq.c.team_id) & (TeamStats.matchweek == subq.c.max_mw))
        .where(TeamStats.season == target_season)
    )
    result = await session.execute(stmt)
    team_stats = result.scalars().all()
    team_forms = {ts.team_id: ts.form for ts in team_stats if ts.form}

    match_details = [build_match_detail(m, team_forms) for m in matches]
    matches_with_value = sum(1 for m in matches if m.value_bets)

    return MatchweekResponse(
        season=target_season,
        matchweek=matchweek,
        matches=match_details,
        total_matches=len(matches),
        matches_with_value_bets=matches_with_value,
        generated_at=datetime.utcnow(),
    )
