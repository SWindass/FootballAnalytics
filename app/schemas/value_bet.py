from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, computed_field

from app.schemas.team import TeamResponse


class ValueBetMatchInfo(BaseModel):
    """Simplified match info for value bet display."""

    id: int
    home_team: TeamResponse
    away_team: TeamResponse
    kickoff_time: datetime
    matchweek: int


class ValueBetResponse(BaseModel):
    """Value bet response schema."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    match: ValueBetMatchInfo | None = None
    outcome: str
    bookmaker: str
    model_probability: Decimal
    implied_probability: Decimal
    edge: Decimal
    odds: Decimal
    kelly_stake: Decimal
    recommended_stake: Decimal
    is_active: bool
    result: str | None = None
    created_at: datetime

    @computed_field
    @property
    def edge_percentage(self) -> str:
        """Edge as a percentage string."""
        return f"{float(self.edge) * 100:.1f}%"

    @computed_field
    @property
    def outcome_display(self) -> str:
        """Human-readable outcome display."""
        outcome_map = {
            "home_win": "Home Win",
            "draw": "Draw",
            "away_win": "Away Win",
            "over_2_5": "Over 2.5 Goals",
            "under_2_5": "Under 2.5 Goals",
            "btts_yes": "Both Teams to Score",
            "btts_no": "Clean Sheet",
        }
        return outcome_map.get(self.outcome, self.outcome)


class ValueBetsListResponse(BaseModel):
    """Response for list of value bets."""

    value_bets: list[ValueBetResponse]
    total_count: int
    active_count: int
    generated_at: datetime
