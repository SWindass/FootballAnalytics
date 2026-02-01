from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, computed_field

from app.schemas.team import TeamFormResponse, TeamResponse


class PredictionProbabilities(BaseModel):
    """Match outcome probabilities."""

    home_win: Decimal | None = None
    draw: Decimal | None = None
    away_win: Decimal | None = None


class ModelPredictions(BaseModel):
    """Predictions from individual models."""

    elo: PredictionProbabilities | None = None
    poisson: PredictionProbabilities | None = None
    xgboost: PredictionProbabilities | None = None
    consensus: PredictionProbabilities | None = None


class AdditionalPredictions(BaseModel):
    """Additional betting market predictions."""

    over_2_5_prob: Decimal | None = None
    btts_prob: Decimal | None = None
    predicted_home_goals: Decimal | None = None
    predicted_away_goals: Decimal | None = None


class MatchOdds(BaseModel):
    """Current odds for a match."""

    bookmaker: str
    home_odds: Decimal | None = None
    draw_odds: Decimal | None = None
    away_odds: Decimal | None = None
    over_2_5_odds: Decimal | None = None
    under_2_5_odds: Decimal | None = None
    btts_yes_odds: Decimal | None = None
    btts_no_odds: Decimal | None = None
    recorded_at: datetime


class MatchAnalysisResponse(BaseModel):
    """Match analysis with predictions and narrative."""

    model_config = ConfigDict(from_attributes=True)

    predictions: ModelPredictions
    additional: AdditionalPredictions
    confidence: Decimal | None = None
    narrative: str | None = None
    narrative_generated_at: datetime | None = None


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
    home_score: int | None = None
    away_score: int | None = None
    home_xg: Decimal | None = None
    away_xg: Decimal | None = None

    @computed_field
    @property
    def is_finished(self) -> bool:
        return self.status == "finished"


class MatchDetailResponse(MatchResponse):
    """Detailed match response with analysis."""

    home_team_form: TeamFormResponse | None = None
    away_team_form: TeamFormResponse | None = None
    analysis: MatchAnalysisResponse | None = None
    latest_odds: list[MatchOdds] | None = None
    value_bets: list["ValueBetResponse"] | None = None


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
