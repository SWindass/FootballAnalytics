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


class Competition(str, Enum):
    """Competition types."""

    PREMIER_LEAGUE = "PL"
    CHAMPIONS_LEAGUE = "CL"
    EUROPA_LEAGUE = "EL"
    FA_CUP = "FAC"
    LEAGUE_CUP = "ELC"  # EFL Cup / Carabao Cup
    COMMUNITY_SHIELD = "CS"


class BetOutcome(str, Enum):
    """Bet outcome types."""

    HOME_WIN = "home_win"
    DRAW = "draw"
    AWAY_WIN = "away_win"
    OVER_2_5 = "over_2_5"
    UNDER_2_5 = "under_2_5"
    BTTS_YES = "btts_yes"
    BTTS_NO = "btts_no"


class StrategyStatus(str, Enum):
    """Betting strategy status."""

    ACTIVE = "active"
    PAUSED = "paused"
    DISABLED = "disabled"


class SnapshotType(str, Enum):
    """Strategy monitoring snapshot types."""

    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"


class Referee(Base):
    """Referee data and historical statistics."""

    __tablename__ = "referees"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    external_id: Mapped[Optional[int]] = mapped_column(Integer, unique=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    nationality: Mapped[Optional[str]] = mapped_column(String(50))

    # Career statistics (updated periodically)
    matches_officiated: Mapped[int] = mapped_column(Integer, default=0)
    avg_fouls_per_game: Mapped[Optional[Decimal]] = mapped_column(Numeric(4, 2))
    avg_yellow_cards: Mapped[Optional[Decimal]] = mapped_column(Numeric(4, 2))
    avg_red_cards: Mapped[Optional[Decimal]] = mapped_column(Numeric(4, 2))
    penalties_awarded: Mapped[int] = mapped_column(Integer, default=0)
    home_win_pct: Mapped[Optional[Decimal]] = mapped_column(Numeric(5, 2))  # Home win % in their games

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Relationships
    matches: Mapped[list["Match"]] = relationship("Match", back_populates="referee")

    def __repr__(self) -> str:
        return f"<Referee {self.name}>"


class Manager(Base):
    """Manager/coach data and statistics."""

    __tablename__ = "managers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    external_id: Mapped[Optional[int]] = mapped_column(Integer, unique=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    nationality: Mapped[Optional[str]] = mapped_column(String(50))
    date_of_birth: Mapped[Optional[datetime]] = mapped_column(DateTime)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Relationships
    tenures: Mapped[list["ManagerTenure"]] = relationship("ManagerTenure", back_populates="manager")

    def __repr__(self) -> str:
        return f"<Manager {self.name}>"


class ManagerTenure(Base):
    """Track manager's tenure at each club."""

    __tablename__ = "manager_tenures"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    manager_id: Mapped[int] = mapped_column(ForeignKey("managers.id"), nullable=False)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), nullable=False)
    start_date: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    end_date: Mapped[Optional[datetime]] = mapped_column(DateTime)  # NULL = current manager
    is_interim: Mapped[bool] = mapped_column(Boolean, default=False)

    # Stats during tenure
    matches_managed: Mapped[int] = mapped_column(Integer, default=0)
    wins: Mapped[int] = mapped_column(Integer, default=0)
    draws: Mapped[int] = mapped_column(Integer, default=0)
    losses: Mapped[int] = mapped_column(Integer, default=0)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Relationships
    manager: Mapped["Manager"] = relationship("Manager", back_populates="tenures")
    team: Mapped["Team"] = relationship("Team", back_populates="manager_tenures")

    __table_args__ = (
        Index("ix_manager_tenures_team_current", "team_id", "end_date"),
    )

    def __repr__(self) -> str:
        return f"<ManagerTenure {self.manager_id} @ {self.team_id}>"


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
    manager_tenures: Mapped[list["ManagerTenure"]] = relationship("ManagerTenure", back_populates="team")

    def __repr__(self) -> str:
        return f"<Team {self.name}>"


class Match(Base):
    """Match fixtures and results."""

    __tablename__ = "matches"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    external_id: Mapped[int] = mapped_column(Integer, unique=True, nullable=False)
    competition: Mapped[str] = mapped_column(String(10), default=Competition.PREMIER_LEAGUE)
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

    # Referee assignment (nullable until assigned)
    referee_id: Mapped[Optional[int]] = mapped_column(ForeignKey("referees.id"))

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Relationships
    home_team: Mapped["Team"] = relationship("Team", foreign_keys=[home_team_id])
    away_team: Mapped["Team"] = relationship("Team", foreign_keys=[away_team_id])
    referee: Mapped[Optional["Referee"]] = relationship("Referee", back_populates="matches")
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

    # Injuries (JSON with player details)
    # Format: {"players": [{"name": "...", "type": "...", "return_date": "..."}], "count": N}
    injuries: Mapped[Optional[dict]] = mapped_column(JSONB)
    injury_count: Mapped[int] = mapped_column(Integer, default=0)  # Quick access to count
    key_players_out: Mapped[int] = mapped_column(Integer, default=0)  # Star players injured

    # Manager info at this point in time
    manager_games: Mapped[int] = mapped_column(Integer, default=0)  # Games since manager started
    is_new_manager: Mapped[bool] = mapped_column(Boolean, default=False)  # < 5 games

    # Recent transfers/signings impact (subjective 0-10 scale)
    transfer_impact: Mapped[Optional[int]] = mapped_column(Integer)  # Significant new signings

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


class TeamFixture(Base):
    """Lightweight fixture tracking for rest day calculation.

    Tracks all matches a team plays (including cup/European games)
    even when the opponent isn't in our teams table.
    """

    __tablename__ = "team_fixtures"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), nullable=False)
    kickoff_time: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    competition: Mapped[str] = mapped_column(String(10), nullable=False)
    opponent_name: Mapped[Optional[str]] = mapped_column(String(100))  # For reference
    is_home: Mapped[bool] = mapped_column(Boolean, default=True)
    match_id: Mapped[Optional[int]] = mapped_column(ForeignKey("matches.id"))  # Link to full match if exists

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    team: Mapped["Team"] = relationship("Team")
    match: Mapped[Optional["Match"]] = relationship("Match")

    __table_args__ = (
        Index("ix_team_fixtures_team_kickoff", "team_id", "kickoff_time"),
        UniqueConstraint("team_id", "kickoff_time", "competition", name="uq_team_fixture"),
    )

    def __repr__(self) -> str:
        return f"<TeamFixture {self.team_id} {self.kickoff_time}>"


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

    # Strategy link (for monitoring)
    strategy_id: Mapped[Optional[int]] = mapped_column(ForeignKey("betting_strategies.id"))

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Relationships
    match: Mapped["Match"] = relationship("Match", back_populates="value_bets")
    strategy: Mapped[Optional["BettingStrategy"]] = relationship(
        "BettingStrategy", back_populates="value_bets"
    )

    __table_args__ = (
        Index("ix_value_bets_match_active", "match_id", "is_active"),
        Index("ix_value_bets_created", "created_at"),
        Index("ix_value_bets_strategy_id", "strategy_id"),
        # Note: ck_positive_edge constraint removed to allow home win strategy
        # which uses negative edge (form 12+ with market seeing more value)
    )

    def __repr__(self) -> str:
        return f"<ValueBet match={self.match_id} {self.outcome} edge={self.edge}>"


class BettingStrategy(Base):
    """Betting strategy configuration and performance tracking."""

    __tablename__ = "betting_strategies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    outcome_type: Mapped[str] = mapped_column(String(20), nullable=False)  # home_win, away_win

    # Strategy parameters (edge ranges, odds ranges, form filters)
    parameters: Mapped[dict] = mapped_column(JSONB, nullable=False)

    # Status
    status: Mapped[StrategyStatus] = mapped_column(String(20), default=StrategyStatus.ACTIVE)
    status_reason: Mapped[Optional[str]] = mapped_column(Text)

    # Performance metrics
    total_bets: Mapped[int] = mapped_column(Integer, default=0)
    total_wins: Mapped[int] = mapped_column(Integer, default=0)
    total_profit: Mapped[Decimal] = mapped_column(Numeric(10, 2), default=Decimal("0"))
    historical_roi: Mapped[Optional[Decimal]] = mapped_column(Numeric(6, 4))
    rolling_50_roi: Mapped[Optional[Decimal]] = mapped_column(Numeric(6, 4))
    consecutive_losing_streak: Mapped[int] = mapped_column(Integer, default=0)

    # Optimization tracking
    last_optimized_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    last_backtest_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    optimization_version: Mapped[int] = mapped_column(Integer, default=1)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Relationships
    value_bets: Mapped[list["ValueBet"]] = relationship("ValueBet", back_populates="strategy")
    snapshots: Mapped[list["StrategyMonitoringSnapshot"]] = relationship(
        "StrategyMonitoringSnapshot", back_populates="strategy"
    )
    optimization_runs: Mapped[list["StrategyOptimizationRun"]] = relationship(
        "StrategyOptimizationRun", back_populates="strategy"
    )

    __table_args__ = (Index("ix_betting_strategies_status", "status"),)

    def __repr__(self) -> str:
        return f"<BettingStrategy {self.name} status={self.status}>"


class StrategyMonitoringSnapshot(Base):
    """Point-in-time performance snapshots for strategy monitoring."""

    __tablename__ = "strategy_monitoring_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    strategy_id: Mapped[int] = mapped_column(ForeignKey("betting_strategies.id"), nullable=False)
    snapshot_date: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    snapshot_type: Mapped[SnapshotType] = mapped_column(String(20), nullable=False)

    # Rolling 30-bet metrics
    rolling_30_bets: Mapped[int] = mapped_column(Integer, default=0)
    rolling_30_wins: Mapped[int] = mapped_column(Integer, default=0)
    rolling_30_roi: Mapped[Optional[Decimal]] = mapped_column(Numeric(6, 4))
    rolling_30_profit: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 2))

    # Rolling 50-bet metrics
    rolling_50_bets: Mapped[int] = mapped_column(Integer, default=0)
    rolling_50_wins: Mapped[int] = mapped_column(Integer, default=0)
    rolling_50_roi: Mapped[Optional[Decimal]] = mapped_column(Numeric(6, 4))
    rolling_50_profit: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 2))

    # Cumulative metrics
    cumulative_bets: Mapped[int] = mapped_column(Integer, default=0)
    cumulative_roi: Mapped[Optional[Decimal]] = mapped_column(Numeric(6, 4))

    # Drift detection statistics
    z_score: Mapped[Optional[Decimal]] = mapped_column(Numeric(6, 4))
    cusum_statistic: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 4))
    is_drift_detected: Mapped[bool] = mapped_column(Boolean, default=False)

    # Alerting
    alert_triggered: Mapped[bool] = mapped_column(Boolean, default=False)
    alert_type: Mapped[Optional[str]] = mapped_column(String(50))

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    strategy: Mapped["BettingStrategy"] = relationship("BettingStrategy", back_populates="snapshots")

    __table_args__ = (
        Index("ix_strategy_snapshots_strategy_date", "strategy_id", "snapshot_date"),
        Index("ix_strategy_snapshots_type", "snapshot_type"),
    )

    def __repr__(self) -> str:
        return f"<StrategySnapshot strategy={self.strategy_id} date={self.snapshot_date}>"


class StrategyOptimizationRun(Base):
    """Track optimization runs and parameter changes."""

    __tablename__ = "strategy_optimization_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    strategy_id: Mapped[int] = mapped_column(ForeignKey("betting_strategies.id"), nullable=False)
    run_date: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    run_type: Mapped[str] = mapped_column(String(20), nullable=False)  # monthly, quarterly, manual

    # Data window used
    data_start: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    data_end: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    n_matches_used: Mapped[int] = mapped_column(Integer, nullable=False)

    # Parameters
    parameters_before: Mapped[dict] = mapped_column(JSONB, nullable=False)
    parameters_after: Mapped[dict] = mapped_column(JSONB, nullable=False)

    # Optimization results
    n_trials: Mapped[int] = mapped_column(Integer, nullable=False)
    best_roi_found: Mapped[Optional[Decimal]] = mapped_column(Numeric(6, 4))
    backtest_roi_before: Mapped[Optional[Decimal]] = mapped_column(Numeric(6, 4))
    backtest_roi_after: Mapped[Optional[Decimal]] = mapped_column(Numeric(6, 4))

    # Application status
    was_applied: Mapped[bool] = mapped_column(Boolean, default=False)
    applied_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    not_applied_reason: Mapped[Optional[str]] = mapped_column(Text)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    strategy: Mapped["BettingStrategy"] = relationship(
        "BettingStrategy", back_populates="optimization_runs"
    )

    __table_args__ = (Index("ix_optimization_runs_strategy_date", "strategy_id", "run_date"),)

    def __repr__(self) -> str:
        return f"<OptimizationRun strategy={self.strategy_id} date={self.run_date}>"
