"""ROI backtesting for value bet detection.

This module validates the value bet detection system by simulating
historical betting and calculating key metrics like ROI and profit per bet.

Success criteria:
- ROI > 0% (profitable)
- Win rate > implied probability from odds
- Consistent across seasons
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Optional

import numpy as np
import structlog
from sqlalchemy import select

from app.db.database import SyncSessionLocal
from app.db.models import (
    Match, MatchAnalysis, MatchStatus, EloRating, TeamStats,
    OddsHistory, Team,
)
from batch.betting.value_detector import ValueDetector, ValueDetectorConfig, ValueBetOpportunity
from batch.models.meta_model import MetaModel

logger = structlog.get_logger()


@dataclass
class BetResult:
    """Result of a simulated bet."""

    match_id: int
    match_date: datetime
    season: str
    outcome: str
    bookmaker: str

    # Bet details
    stake: float
    odds: float
    model_prob: float
    market_prob: float
    edge: float
    confidence: float

    # Result
    won: bool
    profit: float  # stake * (odds - 1) if won, -stake if lost

    # Meta-model
    meta_confidence: Optional[float] = None


@dataclass
class BacktestResult:
    """Results from a backtest run."""

    # Overview
    total_bets: int
    total_staked: float
    total_return: float
    net_profit: float

    # Key metrics
    roi: float  # (return - staked) / staked
    profit_per_bet: float
    win_rate: float
    avg_odds: float

    # By outcome
    home_win_bets: int = 0
    home_win_roi: float = 0.0
    draw_bets: int = 0
    draw_roi: float = 0.0
    away_win_bets: int = 0
    away_win_roi: float = 0.0

    # By season
    season_results: dict = field(default_factory=dict)

    # Individual bets (for analysis)
    bets: list[BetResult] = field(default_factory=list)

    def summary(self) -> str:
        """Generate summary report."""
        lines = [
            "=" * 60,
            "BACKTEST RESULTS",
            "=" * 60,
            "",
            f"Total bets placed: {self.total_bets}",
            f"Total staked: £{self.total_staked:.2f}",
            f"Total return: £{self.total_return:.2f}",
            f"Net profit: £{self.net_profit:.2f}",
            "",
            f"ROI: {self.roi:.1%}",
            f"Profit per bet: £{self.profit_per_bet:.2f}",
            f"Win rate: {self.win_rate:.1%}",
            f"Average odds: {self.avg_odds:.2f}",
            "",
            "By Outcome:",
            f"  Home wins: {self.home_win_bets} bets, ROI: {self.home_win_roi:.1%}",
            f"  Draws: {self.draw_bets} bets, ROI: {self.draw_roi:.1%}",
            f"  Away wins: {self.away_win_bets} bets, ROI: {self.away_win_roi:.1%}",
            "",
            "By Season:",
        ]

        for season, data in sorted(self.season_results.items()):
            lines.append(f"  {season}: {data['bets']} bets, ROI: {data['roi']:.1%}")

        return "\n".join(lines)


class Backtester:
    """Backtest value bet detection on historical data."""

    def __init__(
        self,
        detector: Optional[ValueDetector] = None,
        meta_model: Optional[MetaModel] = None,
        use_meta_model: bool = True,
        meta_min_confidence: float = 0.55,
        base_stake: float = 10.0,
        use_kelly: bool = True,
    ):
        """Initialize backtester.

        Args:
            detector: Value detector instance
            meta_model: Meta-model for filtering bets
            use_meta_model: Whether to use meta-model for filtering
            meta_min_confidence: Minimum meta-model confidence to place bet
            base_stake: Base stake amount
            use_kelly: Whether to use Kelly sizing (otherwise flat stake)
        """
        self.detector = detector or ValueDetector()
        self.meta_model = meta_model
        self.use_meta_model = use_meta_model
        self.meta_min_confidence = meta_min_confidence
        self.base_stake = base_stake
        self.use_kelly = use_kelly

        if use_meta_model and meta_model is None:
            self.meta_model = MetaModel()
            self.meta_model.load_model()

    def run_backtest(
        self,
        start_season: str = "2020-21",
        end_season: str = "2025-26",
    ) -> BacktestResult:
        """Run backtest on historical matches.

        Args:
            start_season: First season to include
            end_season: Last season to include

        Returns:
            BacktestResult with all metrics
        """
        with SyncSessionLocal() as session:
            logger.info(f"Running backtest from {start_season} to {end_season}")

            # Load all data
            matches_query = (
                select(Match, MatchAnalysis)
                .join(MatchAnalysis, Match.id == MatchAnalysis.match_id)
                .where(Match.status == MatchStatus.FINISHED)
                .where(MatchAnalysis.consensus_home_prob.isnot(None))
                .where(Match.season >= start_season)
                .where(Match.season <= end_season)
                .order_by(Match.kickoff_time)
            )
            matches = list(session.execute(matches_query).all())
            logger.info(f"Found {len(matches)} matches to backtest")

            # Bulk load supporting data
            all_team_stats = list(session.execute(select(TeamStats)).scalars().all())
            team_stats_lookup = {
                (ts.team_id, ts.season, ts.matchweek): ts for ts in all_team_stats
            }

            all_elo_ratings = list(session.execute(select(EloRating)).scalars().all())
            elo_lookup = {
                (er.team_id, er.season, er.matchweek): er for er in all_elo_ratings
            }

            all_odds = list(session.execute(select(OddsHistory)).scalars().all())
            odds_lookup = {}
            for oh in all_odds:
                if oh.match_id not in odds_lookup:
                    odds_lookup[oh.match_id] = []
                odds_lookup[oh.match_id].append(oh)

            all_teams = list(session.execute(select(Team)).scalars().all())
            team_lookup = {t.id: t for t in all_teams}

            # Track results
            bets = []
            skipped_no_odds = 0
            skipped_meta_model = 0

            for match, analysis in matches:
                # Get supporting data
                home_stats = team_stats_lookup.get(
                    (match.home_team_id, match.season, match.matchweek - 1)
                )
                away_stats = team_stats_lookup.get(
                    (match.away_team_id, match.season, match.matchweek - 1)
                )
                home_elo = elo_lookup.get(
                    (match.home_team_id, match.season, match.matchweek - 1)
                )
                away_elo = elo_lookup.get(
                    (match.away_team_id, match.season, match.matchweek - 1)
                )
                odds_history = odds_lookup.get(match.id, [])
                home_team = team_lookup.get(match.home_team_id)
                away_team = team_lookup.get(match.away_team_id)

                # Check if we have odds data (either from odds_history table or analysis.features)
                has_odds = bool(odds_history)
                if not has_odds and analysis.features:
                    hist_odds = analysis.features.get("historical_odds", {})
                    has_odds = bool(hist_odds.get("b365_home_odds") or hist_odds.get("avg_home_odds"))

                if not has_odds:
                    skipped_no_odds += 1
                    continue

                # Find value bets
                value_bets = self.detector.find_value_bets(
                    match_id=match.id,
                    analysis=analysis,
                    home_stats=home_stats,
                    away_stats=away_stats,
                    home_elo=home_elo,
                    away_elo=away_elo,
                    odds_history=odds_history,
                )

                # Determine actual result
                if match.home_score > match.away_score:
                    actual = "home_win"
                elif match.home_score == match.away_score:
                    actual = "draw"
                else:
                    actual = "away_win"

                # Process each value bet
                for vb in value_bets:
                    # Optionally filter with meta-model
                    meta_confidence = None
                    if self.use_meta_model and self.meta_model:
                        meta_confidence = self.meta_model.predict(
                            match, analysis, vb.outcome,
                            home_stats, away_stats,
                            home_elo, away_elo,
                            home_team, away_team,
                        )

                        if meta_confidence is not None and meta_confidence < self.meta_min_confidence:
                            skipped_meta_model += 1
                            continue

                    # Calculate stake
                    if self.use_kelly:
                        stake = self.base_stake * (vb.recommended_stake / 0.05)  # Scale by recommended
                        stake = max(self.base_stake * 0.5, min(stake, self.base_stake * 3))
                    else:
                        stake = self.base_stake

                    # Determine outcome
                    won = vb.outcome == actual
                    profit = stake * (vb.odds - 1) if won else -stake

                    bet_result = BetResult(
                        match_id=match.id,
                        match_date=match.kickoff_time,
                        season=match.season,
                        outcome=vb.outcome,
                        bookmaker=vb.bookmaker,
                        stake=stake,
                        odds=vb.odds,
                        model_prob=vb.model_prob,
                        market_prob=vb.market_prob,
                        edge=vb.edge,
                        confidence=vb.confidence,
                        won=won,
                        profit=profit,
                        meta_confidence=meta_confidence,
                    )
                    bets.append(bet_result)

            logger.info(f"Placed {len(bets)} bets")
            logger.info(f"Skipped {skipped_no_odds} matches without odds")
            if self.use_meta_model:
                logger.info(f"Filtered {skipped_meta_model} bets by meta-model")

            return self._calculate_results(bets)

    def _calculate_results(self, bets: list[BetResult]) -> BacktestResult:
        """Calculate backtest results from individual bets."""
        if not bets:
            return BacktestResult(
                total_bets=0,
                total_staked=0.0,
                total_return=0.0,
                net_profit=0.0,
                roi=0.0,
                profit_per_bet=0.0,
                win_rate=0.0,
                avg_odds=0.0,
                bets=[],
            )

        total_staked = sum(b.stake for b in bets)
        total_return = sum(b.stake + b.profit for b in bets)
        net_profit = total_return - total_staked
        wins = sum(1 for b in bets if b.won)

        # By outcome
        home_bets = [b for b in bets if b.outcome == "home_win"]
        draw_bets = [b for b in bets if b.outcome == "draw"]
        away_bets = [b for b in bets if b.outcome == "away_win"]

        def calculate_roi(bet_list):
            if not bet_list:
                return 0.0
            staked = sum(b.stake for b in bet_list)
            returned = sum(b.stake + b.profit for b in bet_list)
            return (returned - staked) / staked if staked > 0 else 0.0

        # By season
        seasons = {}
        for bet in bets:
            if bet.season not in seasons:
                seasons[bet.season] = []
            seasons[bet.season].append(bet)

        season_results = {}
        for season, season_bets in seasons.items():
            season_results[season] = {
                "bets": len(season_bets),
                "wins": sum(1 for b in season_bets if b.won),
                "roi": calculate_roi(season_bets),
                "profit": sum(b.profit for b in season_bets),
            }

        return BacktestResult(
            total_bets=len(bets),
            total_staked=total_staked,
            total_return=total_return,
            net_profit=net_profit,
            roi=net_profit / total_staked if total_staked > 0 else 0.0,
            profit_per_bet=net_profit / len(bets) if bets else 0.0,
            win_rate=wins / len(bets) if bets else 0.0,
            avg_odds=sum(b.odds for b in bets) / len(bets) if bets else 0.0,
            home_win_bets=len(home_bets),
            home_win_roi=calculate_roi(home_bets),
            draw_bets=len(draw_bets),
            draw_roi=calculate_roi(draw_bets),
            away_win_bets=len(away_bets),
            away_win_roi=calculate_roi(away_bets),
            season_results=season_results,
            bets=bets,
        )


def run_backtest_comparison() -> dict:
    """Run backtest with different configurations for comparison.

    Returns:
        Dict with results for each configuration
    """
    results = {}

    # 1. Base detector (no meta-model)
    logger.info("Running backtest: Base detector only")
    base_detector = ValueDetector(ValueDetectorConfig(
        min_edge=0.05,
        min_confidence=0.55,
    ))
    backtester = Backtester(
        detector=base_detector,
        use_meta_model=False,
    )
    results["base"] = backtester.run_backtest()

    # 2. With meta-model filtering
    logger.info("Running backtest: With meta-model")
    meta_backtester = Backtester(
        detector=base_detector,
        use_meta_model=True,
        meta_min_confidence=0.55,
    )
    results["meta_filtered"] = meta_backtester.run_backtest()

    # 3. Higher edge threshold
    logger.info("Running backtest: High edge threshold")
    high_edge_detector = ValueDetector(ValueDetectorConfig(
        min_edge=0.08,
        min_confidence=0.60,
    ))
    high_edge_backtester = Backtester(
        detector=high_edge_detector,
        use_meta_model=False,
    )
    results["high_edge"] = high_edge_backtester.run_backtest()

    # 4. Draw-focused
    logger.info("Running backtest: Draw focused")
    draw_detector = ValueDetector(ValueDetectorConfig(
        min_edge=0.05,
        min_confidence=0.50,
    ))

    class DrawOnlyBacktester(Backtester):
        def run_backtest(self, *args, **kwargs):
            result = super().run_backtest(*args, **kwargs)
            # Filter to only draw bets
            draw_bets = [b for b in result.bets if b.outcome == "draw"]
            return self._calculate_results(draw_bets)

    draw_backtester = DrawOnlyBacktester(
        detector=draw_detector,
        use_meta_model=False,
    )
    results["draw_only"] = draw_backtester.run_backtest()

    return results


def print_comparison(results: dict):
    """Print comparison of backtest results."""
    print("\n" + "=" * 80)
    print("BACKTEST COMPARISON")
    print("=" * 80)

    headers = ["Config", "Bets", "ROI", "Win Rate", "Profit/Bet", "Net Profit"]
    print(f"\n{headers[0]:<20} {headers[1]:>8} {headers[2]:>10} {headers[3]:>10} {headers[4]:>12} {headers[5]:>12}")
    print("-" * 80)

    for name, result in results.items():
        print(
            f"{name:<20} "
            f"{result.total_bets:>8} "
            f"{result.roi:>9.1%} "
            f"{result.win_rate:>9.1%} "
            f"£{result.profit_per_bet:>10.2f} "
            f"£{result.net_profit:>10.2f}"
        )

    print("\n" + "=" * 80)

    # Detailed results for best performing
    best_config = max(results.keys(), key=lambda k: results[k].roi)
    print(f"\nBest configuration: {best_config}")
    print(results[best_config].summary())


if __name__ == "__main__":
    import logging
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

    print("Running Value Bet Backtest...")

    # Simple backtest first
    backtester = Backtester(use_meta_model=False)
    result = backtester.run_backtest()
    print(result.summary())

    # Optionally run comparison
    # results = run_backtest_comparison()
    # print_comparison(results)
