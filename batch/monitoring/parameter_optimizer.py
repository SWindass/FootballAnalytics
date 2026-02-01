"""Bayesian parameter optimization for betting strategies.

Uses Optuna for hyperparameter optimization with TPE sampler.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal

import structlog

try:
    import optuna
    from optuna.samplers import TPESampler

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from sqlalchemy import and_, select
from sqlalchemy.orm import Session

from app.db.models import (
    BettingStrategy,
    Match,
    MatchAnalysis,
    MatchStatus,
    OddsHistory,
    StrategyOptimizationRun,
    TeamStats,
)

logger = structlog.get_logger()


@dataclass
class OptimizationResult:
    """Result of a parameter optimization run."""

    parameters_before: dict
    parameters_after: dict
    n_trials: int
    best_roi_found: float
    backtest_roi_before: float
    backtest_roi_after: float
    improvement: float
    should_apply: bool
    not_applied_reason: str | None = None


class ParameterOptimizer:
    """Optimizes betting strategy parameters using Bayesian optimization."""

    # Minimum improvement required to apply new parameters
    MIN_IMPROVEMENT_PCT = 0.02  # 2% improvement required

    # Default optimization settings
    DEFAULT_N_TRIALS = 100
    DEFAULT_LOOKBACK_YEARS = 2

    def __init__(self, session: Session):
        self.session = session
        if not OPTUNA_AVAILABLE:
            logger.warning("Optuna not installed. Parameter optimization disabled.")

    def optimize_strategy(
        self,
        strategy: BettingStrategy,
        n_trials: int = DEFAULT_N_TRIALS,
        lookback_years: float = DEFAULT_LOOKBACK_YEARS,
    ) -> OptimizationResult | None:
        """Optimize parameters for a single strategy.

        Args:
            strategy: The strategy to optimize
            n_trials: Number of optimization trials
            lookback_years: Years of historical data to use

        Returns:
            OptimizationResult or None if optimization failed
        """
        if not OPTUNA_AVAILABLE:
            logger.error("Cannot optimize: optuna not installed")
            return None

        logger.info(f"Starting optimization for strategy {strategy.name}")

        # Get historical data
        data_end = datetime.utcnow()
        data_start = data_end - timedelta(days=int(lookback_years * 365))

        matches_data = self._load_historical_data(strategy, data_start, data_end)
        if len(matches_data) < 100:
            logger.warning(
                f"Insufficient data for optimization: {len(matches_data)} matches"
            )
            return None

        logger.info(f"Loaded {len(matches_data)} matches for optimization")

        # Store original parameters
        parameters_before = dict(strategy.parameters)

        # Calculate baseline ROI
        backtest_roi_before = self._backtest_strategy(
            matches_data, parameters_before, strategy.outcome_type
        )

        # Create Optuna study
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=42),
        )

        # Define objective function
        def objective(trial: optuna.Trial) -> float:
            params = self._suggest_parameters(trial, strategy.outcome_type)
            roi = self._backtest_strategy(matches_data, params, strategy.outcome_type)
            return roi

        # Run optimization
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        # Get best parameters
        best_params = self._get_best_params(study, strategy.outcome_type)
        backtest_roi_after = study.best_value
        improvement = backtest_roi_after - backtest_roi_before

        # Determine if we should apply
        should_apply = improvement >= self.MIN_IMPROVEMENT_PCT
        not_applied_reason = None
        if not should_apply:
            not_applied_reason = (
                f"Improvement of {improvement:.2%} is below threshold of {self.MIN_IMPROVEMENT_PCT:.2%}"
            )

        result = OptimizationResult(
            parameters_before=parameters_before,
            parameters_after=best_params,
            n_trials=n_trials,
            best_roi_found=backtest_roi_after,
            backtest_roi_before=backtest_roi_before,
            backtest_roi_after=backtest_roi_after,
            improvement=improvement,
            should_apply=should_apply,
            not_applied_reason=not_applied_reason,
        )

        # Record the optimization run
        self._record_optimization_run(strategy, result, data_start, data_end, len(matches_data))

        logger.info(
            f"Optimization complete for {strategy.name}",
            roi_before=backtest_roi_before,
            roi_after=backtest_roi_after,
            improvement=improvement,
            should_apply=should_apply,
        )

        return result

    def apply_optimization(
        self, strategy: BettingStrategy, result: OptimizationResult
    ) -> None:
        """Apply optimized parameters to a strategy.

        Args:
            strategy: The strategy to update
            result: The optimization result to apply
        """
        strategy.parameters = result.parameters_after
        strategy.last_optimized_at = datetime.utcnow()
        strategy.optimization_version += 1

        logger.info(
            f"Applied new parameters to {strategy.name}",
            new_params=result.parameters_after,
            version=strategy.optimization_version,
        )

    def _load_historical_data(
        self,
        strategy: BettingStrategy,
        start_date: datetime,
        end_date: datetime,
    ) -> list[dict]:
        """Load historical match data for backtesting."""
        stmt = (
            select(Match, MatchAnalysis)
            .join(MatchAnalysis, Match.id == MatchAnalysis.match_id)
            .where(
                and_(
                    Match.status == MatchStatus.FINISHED,
                    Match.kickoff_time >= start_date,
                    Match.kickoff_time <= end_date,
                    MatchAnalysis.consensus_home_prob.isnot(None),
                )
            )
            .order_by(Match.kickoff_time)
        )
        results = list(self.session.execute(stmt).all())

        # Load odds and stats
        match_ids = [m.id for m, _ in results]

        # Load odds
        odds_stmt = select(OddsHistory).where(OddsHistory.match_id.in_(match_ids))
        odds_data = list(self.session.execute(odds_stmt).scalars().all())
        odds_lookup = {}
        for oh in odds_data:
            if oh.match_id not in odds_lookup:
                odds_lookup[oh.match_id] = {"home_odds": None, "away_odds": None}
            if oh.home_odds:
                if (
                    odds_lookup[oh.match_id]["home_odds"] is None
                    or float(oh.home_odds) > odds_lookup[oh.match_id]["home_odds"]
                ):
                    odds_lookup[oh.match_id]["home_odds"] = float(oh.home_odds)
            if oh.away_odds:
                if (
                    odds_lookup[oh.match_id]["away_odds"] is None
                    or float(oh.away_odds) > odds_lookup[oh.match_id]["away_odds"]
                ):
                    odds_lookup[oh.match_id]["away_odds"] = float(oh.away_odds)

        # Load team stats for home form
        seasons = {m.season for m, _ in results}
        stats_stmt = select(TeamStats).where(TeamStats.season.in_(seasons))
        stats_data = list(self.session.execute(stats_stmt).scalars().all())
        stats_lookup = {(ts.team_id, ts.season, ts.matchweek): ts for ts in stats_data}

        # Build data list
        data = []
        for match, analysis in results:
            odds = odds_lookup.get(match.id, {})
            home_stats = stats_lookup.get(
                (match.home_team_id, match.season, match.matchweek - 1)
            )

            # Get odds from features if not in OddsHistory
            if not odds.get("home_odds") and analysis.features:
                hist_odds = analysis.features.get("historical_odds", {})
                odds["home_odds"] = hist_odds.get("avg_home_odds")
                odds["away_odds"] = hist_odds.get("avg_away_odds")

            data.append(
                {
                    "match_id": match.id,
                    "home_score": match.home_score,
                    "away_score": match.away_score,
                    "consensus_home_prob": float(analysis.consensus_home_prob),
                    "consensus_away_prob": float(analysis.consensus_away_prob),
                    "home_odds": odds.get("home_odds"),
                    "away_odds": odds.get("away_odds"),
                    "home_form": home_stats.form_points if home_stats else 0,
                }
            )

        return data

    def _suggest_parameters(
        self, trial: optuna.Trial, outcome_type: str
    ) -> dict:
        """Suggest parameters for an Optuna trial."""
        if outcome_type == "away_win":
            return {
                "min_edge": trial.suggest_float("min_edge", 0.02, 0.10),
                "max_edge": trial.suggest_float("max_edge", 0.08, 0.20),
                "min_odds": trial.suggest_float("min_odds", 1.3, 2.0),
                "max_odds": trial.suggest_float("max_odds", 5.0, 12.0),
                "exclude_home_form_min": trial.suggest_int("exclude_home_form_min", 3, 6),
                "exclude_home_form_max": trial.suggest_int("exclude_home_form_max", 5, 8),
            }
        elif outcome_type == "home_win":
            return {
                "max_edge": trial.suggest_float("max_edge", -0.10, 0.05),
                "min_form": trial.suggest_int("min_form", 10, 14),
                "min_odds": trial.suggest_float("min_odds", 1.01, 1.5),
                "max_odds": trial.suggest_float("max_odds", 5.0, 15.0),
            }
        else:
            raise ValueError(f"Unknown outcome type: {outcome_type}")

    def _get_best_params(self, study: optuna.Study, outcome_type: str) -> dict:
        """Extract best parameters from study."""
        best = study.best_params

        if outcome_type == "away_win":
            return {
                "min_edge": best["min_edge"],
                "max_edge": best["max_edge"],
                "min_odds": best["min_odds"],
                "max_odds": best["max_odds"],
                "exclude_home_form_min": best["exclude_home_form_min"],
                "exclude_home_form_max": best["exclude_home_form_max"],
            }
        elif outcome_type == "home_win":
            return {
                "max_edge": best["max_edge"],
                "min_form": best["min_form"],
                "min_odds": best["min_odds"],
                "max_odds": best["max_odds"],
            }
        else:
            return best

    def _backtest_strategy(
        self, data: list[dict], params: dict, outcome_type: str
    ) -> float:
        """Backtest a strategy with given parameters.

        Returns ROI as a decimal (e.g., 0.15 = 15% ROI).
        """
        total_profit = 0.0
        n_bets = 0

        for match in data:
            # Check if this match qualifies
            if outcome_type == "away_win":
                if not self._qualifies_away_win(match, params):
                    continue
                odds = match.get("away_odds")
                won = (
                    match["away_score"] is not None
                    and match["home_score"] is not None
                    and match["away_score"] > match["home_score"]
                )
            elif outcome_type == "home_win":
                if not self._qualifies_home_win(match, params):
                    continue
                odds = match.get("home_odds")
                won = (
                    match["home_score"] is not None
                    and match["away_score"] is not None
                    and match["home_score"] > match["away_score"]
                )
            else:
                continue

            if odds is None:
                continue

            n_bets += 1
            if won:
                total_profit += odds - 1  # Profit on 1 unit stake
            else:
                total_profit -= 1  # Lost 1 unit stake

        if n_bets == 0:
            return -1.0  # Penalize strategies with no bets

        return total_profit / n_bets

    def _qualifies_away_win(self, match: dict, params: dict) -> bool:
        """Check if match qualifies for away win strategy."""
        away_odds = match.get("away_odds")
        if not away_odds:
            return False

        market_prob = 1 / away_odds
        model_prob = match["consensus_away_prob"]
        edge = model_prob - market_prob

        # Check edge range
        if edge < params.get("min_edge", 0.05):
            return False
        if edge > params.get("max_edge", 0.12):
            return False

        # Check odds range
        if away_odds < params.get("min_odds", 1.5):
            return False
        if away_odds > params.get("max_odds", 8.0):
            return False

        # Check home form exclusion
        home_form = match.get("home_form", 0)
        exclude_min = params.get("exclude_home_form_min")
        exclude_max = params.get("exclude_home_form_max")
        if exclude_min is not None and exclude_max is not None:
            if exclude_min <= home_form <= exclude_max:
                return False

        return True

    def _qualifies_home_win(self, match: dict, params: dict) -> bool:
        """Check if match qualifies for home win strategy."""
        home_odds = match.get("home_odds")
        if not home_odds:
            return False

        market_prob = 1 / home_odds
        model_prob = match["consensus_home_prob"]
        edge = model_prob - market_prob

        # Home win requires negative edge (market sees more value)
        if edge > params.get("max_edge", 0.0):
            return False

        # Check form requirement
        home_form = match.get("home_form", 0)
        if home_form < params.get("min_form", 12):
            return False

        # Check odds range
        if home_odds < params.get("min_odds", 1.01):
            return False
        if home_odds > params.get("max_odds", 10.0):
            return False

        return True

    def _record_optimization_run(
        self,
        strategy: BettingStrategy,
        result: OptimizationResult,
        data_start: datetime,
        data_end: datetime,
        n_matches: int,
    ) -> None:
        """Record an optimization run in the database."""
        run = StrategyOptimizationRun(
            strategy_id=strategy.id,
            run_date=datetime.utcnow(),
            run_type="monthly",
            data_start=data_start,
            data_end=data_end,
            n_matches_used=n_matches,
            parameters_before=result.parameters_before,
            parameters_after=result.parameters_after,
            n_trials=result.n_trials,
            best_roi_found=Decimal(str(round(result.best_roi_found, 4))),
            backtest_roi_before=Decimal(str(round(result.backtest_roi_before, 4))),
            backtest_roi_after=Decimal(str(round(result.backtest_roi_after, 4))),
            was_applied=False,  # Will be updated if applied
            not_applied_reason=result.not_applied_reason,
        )
        self.session.add(run)
