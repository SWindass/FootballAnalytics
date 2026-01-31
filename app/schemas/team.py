from datetime import datetime
from decimal import Decimal
from typing import Optional

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
    crest_url: Optional[str] = None
    venue: Optional[str] = None
    form: Optional[str] = None  # Last 5 results e.g. "WWDLW"


class TeamStatsResponse(BaseModel):
    """Team statistics response schema."""

    model_config = ConfigDict(from_attributes=True)

    team: TeamResponse
    season: str
    matchweek: int
    form: Optional[str] = None
    form_points: Optional[int] = None
    goals_scored: int
    goals_conceded: int
    avg_goals_scored: Optional[Decimal] = None
    avg_goals_conceded: Optional[Decimal] = None
    xg_for: Optional[Decimal] = None
    xg_against: Optional[Decimal] = None
    home_record: str  # "W-D-L" format
    away_record: str
    clean_sheets: int
    failed_to_score: int
    elo_rating: Optional[Decimal] = None
    injuries: Optional[list[str]] = None


class TeamFormResponse(BaseModel):
    """Simplified team form for match context."""

    model_config = ConfigDict(from_attributes=True)

    team: TeamResponse
    form: Optional[str] = None
    elo_rating: Optional[Decimal] = None
    avg_goals_scored: Optional[Decimal] = None
    avg_goals_conceded: Optional[Decimal] = None
