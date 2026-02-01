"""Strategy classifier for matching bets to strategies.

Classifies value bets into strategies based on outcome type, edge ranges,
odds ranges, and form filters defined in the strategy parameters.
"""


import structlog
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import BettingStrategy, Match, StrategyStatus, TeamStats, ValueBet

logger = structlog.get_logger()


class StrategyClassifier:
    """Classifies value bets into betting strategies."""

    def __init__(self, session: Session):
        self.session = session
        self._strategies: dict[int, BettingStrategy] = {}
        self._load_strategies()

    def _load_strategies(self) -> None:
        """Load all active strategies from database."""
        stmt = select(BettingStrategy).where(
            BettingStrategy.status.in_([StrategyStatus.ACTIVE, StrategyStatus.PAUSED])
        )
        strategies = list(self.session.execute(stmt).scalars().all())
        self._strategies = {s.id: s for s in strategies}
        logger.info(f"Loaded {len(self._strategies)} strategies for classification")

    def classify_bet(
        self,
        value_bet: ValueBet,
        home_form: int | None = None,
    ) -> int | None:
        """Classify a value bet into a strategy.

        Args:
            value_bet: The value bet to classify
            home_form: Home team's form points (from last 5 games, 0-15)

        Returns:
            Strategy ID if bet matches a strategy, None otherwise
        """
        outcome = str(value_bet.outcome.value) if hasattr(value_bet.outcome, 'value') else str(value_bet.outcome)
        edge = float(value_bet.edge)
        odds = float(value_bet.odds)

        for strategy_id, strategy in self._strategies.items():
            if self._matches_strategy(strategy, outcome, edge, odds, home_form):
                return strategy_id

        return None

    def _matches_strategy(
        self,
        strategy: BettingStrategy,
        outcome: str,
        edge: float,
        odds: float,
        home_form: int | None,
    ) -> bool:
        """Check if a bet matches a strategy's criteria.

        Args:
            strategy: The strategy to check against
            outcome: Bet outcome type (home_win, away_win, etc.)
            edge: Model probability - market implied probability
            odds: Decimal betting odds
            home_form: Home team form points (0-15)

        Returns:
            True if bet matches strategy criteria
        """
        params = strategy.parameters

        # Check outcome type
        if outcome != strategy.outcome_type:
            return False

        # Check odds range
        min_odds = params.get("min_odds", 1.01)
        max_odds = params.get("max_odds", 100.0)
        if odds < min_odds or odds > max_odds:
            return False

        # Strategy-specific checks
        if outcome == "away_win":
            # Away win strategy: positive edge required (5-12%)
            min_edge = params.get("min_edge", 0.05)
            max_edge = params.get("max_edge", 0.12)
            if edge < min_edge or edge > max_edge:
                return False

            # Enhanced: exclude when home team form is 4-6 (poor but not terrible)
            exclude_home_form_min = params.get("exclude_home_form_min")
            exclude_home_form_max = params.get("exclude_home_form_max")
            if (
                exclude_home_form_min is not None
                and exclude_home_form_max is not None
                and home_form is not None
            ):
                if exclude_home_form_min <= home_form <= exclude_home_form_max:
                    return False

        elif outcome == "home_win":
            # Home win strategy: negative edge AND hot streak required
            max_edge = params.get("max_edge", 0.0)
            min_form = params.get("min_form", 12)

            if edge > max_edge:  # Edge must be <= max_edge (typically negative)
                return False
            if home_form is None or home_form < min_form:
                return False

        return True

    def get_home_form_for_match(self, match: Match) -> int | None:
        """Get home team's form points for a match.

        Args:
            match: The match to look up form for

        Returns:
            Home team form points (0-15), or None if not available
        """
        # Get team stats from previous matchweek
        stmt = (
            select(TeamStats)
            .where(TeamStats.team_id == match.home_team_id)
            .where(TeamStats.season == match.season)
            .where(TeamStats.matchweek == match.matchweek - 1)
        )
        stats = self.session.execute(stmt).scalar_one_or_none()
        return stats.form_points if stats else None

    def classify_and_assign(self, value_bet: ValueBet) -> int | None:
        """Classify a bet and assign strategy_id.

        This is a convenience method that gets home form and classifies the bet.

        Args:
            value_bet: The value bet to classify and update

        Returns:
            Strategy ID if assigned, None otherwise
        """
        match = value_bet.match
        home_form = self.get_home_form_for_match(match)
        strategy_id = self.classify_bet(value_bet, home_form)

        if strategy_id:
            value_bet.strategy_id = strategy_id

        return strategy_id
