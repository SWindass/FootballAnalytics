from datetime import datetime
from decimal import Decimal
from typing import Optional

from pydantic import BaseModel, ConfigDict, computed_field

from app.schemas.team import TeamFormResponse, TeamResponse


class PredictionProbabilities(BaseModel):
    """Match outcome probabilities."""

    home_win: Optional[Decimal] = None
    draw: Optional[Decimal] = None
    away_win: Optional[Decimal] = None


class ModelPredictions(BaseModel):
    """Predictions from individual models."""

    elo: Optional[PredictionProbabilities] = None
    poisson: Optional[PredictionProbabilities] = None
    xgboost: Optional[PredictionProbabilities] = None
    consensus: Optional[PredictionProbabilities] = None


class AdditionalPredictions(BaseModel):
    """Additional betting market predictions."""

    over_2_5_prob: Optional[Decimal] = None
    btts_prob: Optional[Decimal] = None
    predicted_home_goals: Optional[Decimal] = None
    predicted_away_goals: Optional[Decimal] = None


class MatchOdds(BaseModel):
    """Current odds for a match."""

    bookmaker: str
    home_odds: Optional[Decimal] = None
    draw_odds: Optional[Decimal] = None
    away_odds: Optional[Decimal] = None
    over_2_5_odds: Optional[Decimal] = None
    under_2_5_odds: Optional[Decimal] = None
    btts_yes_odds: Optional[Decimal] = None
    btts_no_odds: Optional[Decimal] = None
    recorded_at: datetime


class MatchAnalysisResponse(BaseModel):
    """Match analysis with predictions and narrative."""

    model_config = ConfigDict(from_attributes=True)

    predictions: ModelPredictions
    additional: AdditionalPredictions
    confidence: Optional[Decimal] = None
    narrative: Optional[str] = None
    narrative_generated_at: Optional[datetime] = None


class MatchBase(BaseModel):
    """Base match schema."""

    season: str
    matchweek: int
    kickoff_time: datetime
    status: str


class MatchResponse(MatchBase):
    """Match response schema."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    external_id: int
    home_team: TeamResponse
    away_team: TeamResponse
    home_score: Optional[int] = None
    away_score: Optional[int] = None
    home_xg: Optional[Decimal] = None
    away_xg: Optional[Decimal] = None

    @computed_field
    @property
    def is_finished(self) -> bool:
        return self.status == "finished"


class MatchDetailResponse(MatchResponse):
    """Detailed match response with analysis."""

    home_team_form: Optional[TeamFormResponse] = None
    away_team_form: Optional[TeamFormResponse] = None
    analysis: Optional[MatchAnalysisResponse] = None
    latest_odds: Optional[list[MatchOdds]] = None
    value_bets: Optional[list["ValueBetResponse"]] = None


class MatchweekResponse(BaseModel):
    """Matchweek summary response."""

    season: str
    matchweek: int
    matches: list[MatchDetailResponse]
    total_matches: int
    matches_with_value_bets: int
    generated_at: datetime


# Import here to avoid circular import
from app.schemas.value_bet import ValueBetResponse  # noqa: E402

MatchDetailResponse.model_rebuild()
