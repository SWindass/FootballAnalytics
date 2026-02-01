from decimal import Decimal

from pydantic import BaseModel, ConfigDict


class TeamBase(BaseModel):
    """Base team schema."""

    name: str
    short_name: str
    tla: str


class TeamResponse(TeamBase):
    """Team response schema."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    external_id: int
    crest_url: str | None = None
    venue: str | None = None
    form: str | None = None  # Last 5 results e.g. "WWDLW"


class TeamStatsResponse(BaseModel):
    """Team statistics response schema."""

    model_config = ConfigDict(from_attributes=True)

    team: TeamResponse
    season: str
    matchweek: int
    form: str | None = None
    form_points: int | None = None
    goals_scored: int
    goals_conceded: int
    avg_goals_scored: Decimal | None = None
    avg_goals_conceded: Decimal | None = None
    xg_for: Decimal | None = None
    xg_against: Decimal | None = None
    home_record: str  # "W-D-L" format
    away_record: str
    clean_sheets: int
    failed_to_score: int
    elo_rating: Decimal | None = None
    injuries: list[str] | None = None


class TeamFormResponse(BaseModel):
    """Simplified team form for match context."""

    model_config = ConfigDict(from_attributes=True)

    team: TeamResponse
    form: str | None = None
    elo_rating: Decimal | None = None
    avg_goals_scored: Decimal | None = None
    avg_goals_conceded: Decimal | None = None
