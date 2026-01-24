from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base


class MatchStatus(str, Enum):
    """Match status enumeration."""

    SCHEDULED = "scheduled"
    IN_PLAY = "in_play"
    PAUSED = "paused"
    FINISHED = "finished"
    POSTPONED = "postponed"
    CANCELLED = "cancelled"


class BetOutcome(str, Enum):
    """Bet outcome types."""

    HOME_WIN = "home_win"
    DRAW = "draw"
    AWAY_WIN = "away_win"
    OVER_2_5 = "over_2_5"
    UNDER_2_5 = "under_2_5"
    BTTS_YES = "btts_yes"
    BTTS_NO = "btts_no"


class Team(Base):
    """EPL team reference data."""

    __tablename__ = "teams"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    external_id: Mapped[int] = mapped_column(Integer, unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    short_name: Mapped[str] = mapped_column(String(50), nullable=False)
    tla: Mapped[str] = mapped_column(String(3), nullable=False)  # Three-letter abbreviation
    crest_url: Mapped[Optional[str]] = mapped_column(String(255))
    venue: Mapped[Optional[str]] = mapped_column(String(100))
    founded: Mapped[Optional[int]] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Relationships
    home_matches: Mapped[list["Match"]] = relationship(
        "Match", foreign_keys="Match.home_team_id", back_populates="home_team"
    )
    away_matches: Mapped[list["Match"]] = relationship(
        "Match", foreign_keys="Match.away_team_id", back_populates="away_team"
    )
    elo_ratings: Mapped[list["EloRating"]] = relationship("EloRating", back_populates="team")
    team_stats: Mapped[list["TeamStats"]] = relationship("TeamStats", back_populates="team")

    def __repr__(self) -> str:
        return f"<Team {self.name}>"


class Match(Base):
    """Match fixtures and results."""

    __tablename__ = "matches"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    external_id: Mapped[int] = mapped_column(Integer, unique=True, nullable=False)
    season: Mapped[str] = mapped_column(String(10), nullable=False)  # e.g., "2024-25"
    matchweek: Mapped[int] = mapped_column(Integer, nullable=False)
    kickoff_time: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    status: Mapped[MatchStatus] = mapped_column(String(20), default=MatchStatus.SCHEDULED)

    # Teams
    home_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), nullable=False)
    away_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), nullable=False)

    # Results (nullable until match finished)
    home_score: Mapped[Optional[int]] = mapped_column(Integer)
    away_score: Mapped[Optional[int]] = mapped_column(Integer)
    home_ht_score: Mapped[Optional[int]] = mapped_column(Integer)
    away_ht_score: Mapped[Optional[int]] = mapped_column(Integer)

    # xG data from Understat (nullable until available)
    home_xg: Mapped[Optional[Decimal]] = mapped_column(Numeric(5, 2))
    away_xg: Mapped[Optional[Decimal]] = mapped_column(Numeric(5, 2))

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Relationships
    home_team: Mapped["Team"] = relationship("Team", foreign_keys=[home_team_id])
    away_team: Mapped["Team"] = relationship("Team", foreign_keys=[away_team_id])
    analysis: Mapped[Optional["MatchAnalysis"]] = relationship(
        "MatchAnalysis", back_populates="match", uselist=False
    )
    odds_history: Mapped[list["OddsHistory"]] = relationship(
        "OddsHistory", back_populates="match"
    )
    value_bets: Mapped[list["ValueBet"]] = relationship("ValueBet", back_populates="match")

    __table_args__ = (
        Index("ix_matches_season_matchweek", "season", "matchweek"),
        Index("ix_matches_kickoff_time", "kickoff_time"),
        CheckConstraint("home_team_id != away_team_id", name="ck_different_teams"),
    )

    def __repr__(self) -> str:
        return f"<Match {self.id}: MW{self.matchweek}>"


class EloRating(Base):
    """Team ELO ratings tracked by matchweek."""

    __tablename__ = "elo_ratings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), nullable=False)
    season: Mapped[str] = mapped_column(String(10), nullable=False)
    matchweek: Mapped[int] = mapped_column(Integer, nullable=False)
    rating: Mapped[Decimal] = mapped_column(Numeric(7, 2), nullable=False)
    rating_change: Mapped[Optional[Decimal]] = mapped_column(Numeric(6, 2))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    team: Mapped["Team"] = relationship("Team", back_populates="elo_ratings")

    __table_args__ = (
        UniqueConstraint("team_id", "season", "matchweek", name="uq_team_season_matchweek"),
        Index("ix_elo_ratings_team_season", "team_id", "season"),
    )

    def __repr__(self) -> str:
        return f"<EloRating {self.team_id}: {self.rating}>"


class TeamStats(Base):
    """Team statistics and form data."""

    __tablename__ = "team_stats"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), nullable=False)
    season: Mapped[str] = mapped_column(String(10), nullable=False)
    matchweek: Mapped[int] = mapped_column(Integer, nullable=False)

    # Form (last 5 games)
    form: Mapped[Optional[str]] = mapped_column(String(5))  # e.g., "WDLWW"
    form_points: Mapped[Optional[int]] = mapped_column(Integer)

    # Goals
    goals_scored: Mapped[int] = mapped_column(Integer, default=0)
    goals_conceded: Mapped[int] = mapped_column(Integer, default=0)
    avg_goals_scored: Mapped[Optional[Decimal]] = mapped_column(Numeric(4, 2))
    avg_goals_conceded: Mapped[Optional[Decimal]] = mapped_column(Numeric(4, 2))

    # xG stats
    xg_for: Mapped[Optional[Decimal]] = mapped_column(Numeric(6, 2))
    xg_against: Mapped[Optional[Decimal]] = mapped_column(Numeric(6, 2))
    avg_xg_for: Mapped[Optional[Decimal]] = mapped_column(Numeric(4, 2))
    avg_xg_against: Mapped[Optional[Decimal]] = mapped_column(Numeric(4, 2))

    # Home/Away splits
    home_wins: Mapped[int] = mapped_column(Integer, default=0)
    home_draws: Mapped[int] = mapped_column(Integer, default=0)
    home_losses: Mapped[int] = mapped_column(Integer, default=0)
    away_wins: Mapped[int] = mapped_column(Integer, default=0)
    away_draws: Mapped[int] = mapped_column(Integer, default=0)
    away_losses: Mapped[int] = mapped_column(Integer, default=0)

    # Clean sheets
    clean_sheets: Mapped[int] = mapped_column(Integer, default=0)
    failed_to_score: Mapped[int] = mapped_column(Integer, default=0)

    # Injuries (JSON list of player names)
    injuries: Mapped[Optional[dict]] = mapped_column(JSONB)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Relationships
    team: Mapped["Team"] = relationship("Team", back_populates="team_stats")

    __table_args__ = (
        UniqueConstraint("team_id", "season", "matchweek", name="uq_team_stats_season_matchweek"),
        Index("ix_team_stats_team_season", "team_id", "season"),
    )

    def __repr__(self) -> str:
        return f"<TeamStats {self.team_id} MW{self.matchweek}>"


class MatchAnalysis(Base):
    """Pre-computed match predictions and AI narratives."""

    __tablename__ = "match_analyses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"), unique=True, nullable=False)

    # Model predictions (probabilities as decimals, e.g., 0.45 = 45%)
    elo_home_prob: Mapped[Optional[Decimal]] = mapped_column(Numeric(5, 4))
    elo_draw_prob: Mapped[Optional[Decimal]] = mapped_column(Numeric(5, 4))
    elo_away_prob: Mapped[Optional[Decimal]] = mapped_column(Numeric(5, 4))

    poisson_home_prob: Mapped[Optional[Decimal]] = mapped_column(Numeric(5, 4))
    poisson_draw_prob: Mapped[Optional[Decimal]] = mapped_column(Numeric(5, 4))
    poisson_away_prob: Mapped[Optional[Decimal]] = mapped_column(Numeric(5, 4))
    poisson_over_2_5_prob: Mapped[Optional[Decimal]] = mapped_column(Numeric(5, 4))
    poisson_btts_prob: Mapped[Optional[Decimal]] = mapped_column(Numeric(5, 4))

    xgboost_home_prob: Mapped[Optional[Decimal]] = mapped_column(Numeric(5, 4))
    xgboost_draw_prob: Mapped[Optional[Decimal]] = mapped_column(Numeric(5, 4))
    xgboost_away_prob: Mapped[Optional[Decimal]] = mapped_column(Numeric(5, 4))

    # Consensus predictions (weighted average)
    consensus_home_prob: Mapped[Optional[Decimal]] = mapped_column(Numeric(5, 4))
    consensus_draw_prob: Mapped[Optional[Decimal]] = mapped_column(Numeric(5, 4))
    consensus_away_prob: Mapped[Optional[Decimal]] = mapped_column(Numeric(5, 4))

    # Predicted score
    predicted_home_goals: Mapped[Optional[Decimal]] = mapped_column(Numeric(4, 2))
    predicted_away_goals: Mapped[Optional[Decimal]] = mapped_column(Numeric(4, 2))

    # AI-generated narrative
    narrative: Mapped[Optional[str]] = mapped_column(Text)
    narrative_generated_at: Mapped[Optional[datetime]] = mapped_column(DateTime)

    # Feature data used for predictions (stored for debugging/transparency)
    features: Mapped[Optional[dict]] = mapped_column(JSONB)

    # Confidence score (0-1)
    confidence: Mapped[Optional[Decimal]] = mapped_column(Numeric(4, 3))

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Relationships
    match: Mapped["Match"] = relationship("Match", back_populates="analysis")

    def __repr__(self) -> str:
        return f"<MatchAnalysis match_id={self.match_id}>"


class OddsHistory(Base):
    """Historical odds tracking."""

    __tablename__ = "odds_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"), nullable=False)
    bookmaker: Mapped[str] = mapped_column(String(50), nullable=False)
    market: Mapped[str] = mapped_column(String(50), nullable=False)  # e.g., "1x2", "over_under"
    recorded_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Odds values
    home_odds: Mapped[Optional[Decimal]] = mapped_column(Numeric(6, 2))
    draw_odds: Mapped[Optional[Decimal]] = mapped_column(Numeric(6, 2))
    away_odds: Mapped[Optional[Decimal]] = mapped_column(Numeric(6, 2))
    over_2_5_odds: Mapped[Optional[Decimal]] = mapped_column(Numeric(6, 2))
    under_2_5_odds: Mapped[Optional[Decimal]] = mapped_column(Numeric(6, 2))
    btts_yes_odds: Mapped[Optional[Decimal]] = mapped_column(Numeric(6, 2))
    btts_no_odds: Mapped[Optional[Decimal]] = mapped_column(Numeric(6, 2))

    # Relationships
    match: Mapped["Match"] = relationship("Match", back_populates="odds_history")

    __table_args__ = (
        Index("ix_odds_history_match_recorded", "match_id", "recorded_at"),
        Index("ix_odds_history_bookmaker", "bookmaker"),
    )

    def __repr__(self) -> str:
        return f"<OddsHistory match={self.match_id} {self.bookmaker}>"


class ValueBet(Base):
    """Detected value betting opportunities."""

    __tablename__ = "value_bets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"), nullable=False)
    outcome: Mapped[BetOutcome] = mapped_column(String(20), nullable=False)
    bookmaker: Mapped[str] = mapped_column(String(50), nullable=False)

    # Probability comparison
    model_probability: Mapped[Decimal] = mapped_column(Numeric(5, 4), nullable=False)
    implied_probability: Mapped[Decimal] = mapped_column(Numeric(5, 4), nullable=False)
    edge: Mapped[Decimal] = mapped_column(Numeric(5, 4), nullable=False)

    # Odds and stake
    odds: Mapped[Decimal] = mapped_column(Numeric(6, 2), nullable=False)
    kelly_stake: Mapped[Decimal] = mapped_column(Numeric(5, 4), nullable=False)
    recommended_stake: Mapped[Decimal] = mapped_column(Numeric(5, 4), nullable=False)

    # Status tracking
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    result: Mapped[Optional[str]] = mapped_column(String(10))  # "won", "lost", "void"
    profit_loss: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 2))

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Relationships
    match: Mapped["Match"] = relationship("Match", back_populates="value_bets")

    __table_args__ = (
        Index("ix_value_bets_match_active", "match_id", "is_active"),
        Index("ix_value_bets_created", "created_at"),
        CheckConstraint("edge > 0", name="ck_positive_edge"),
    )

    def __repr__(self) -> str:
        return f"<ValueBet match={self.match_id} {self.outcome} edge={self.edge}>"
