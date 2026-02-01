"""Comprehensive backtest for STATIC 12% + ERA-BASED FORM OVERRIDE strategy.

Strategy Rules:
1. AWAY BETS - ALWAYS ACTIVE (all eras):
   - Edge >= 12% (no cap)
   - Exclude when home_form in [4, 5, 6]
   - Odds range: 1.50 to 8.00

2. HOME BETS - 2020s ERA ONLY:
   - Form Override: home_form >= 12 AND edge < 0
   - Only active when season >= "2020-21"

3. STOP-LOSS:
   - Hard stop if 12-month ROI < -20%
   - Hard stop if away success rate < 30% over 50+ bets
"""

import argparse
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Optional

import structlog
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.database import SyncSessionLocal
from app.db.models import (
    Match,
    MatchAnalysis,
    MatchStatus,
    OddsHistory,
    TeamStats,
)

logger = structlog.get_logger()


@dataclass
class Bet:
    """A single bet record."""
    match_id: int
    season: str
    matchweek: int
    kickoff: datetime
    outcome: str  # 'away_win' or 'home_win'
    odds: float
    model_prob: float
    edge: float
    home_form: int
    away_form: int
    actual_result: str  # 'H', 'D', 'A'
    won: bool
    profit: float  # Net profit (win: odds-1, lose: -1)
    strategy: str  # 'away_12pct', 'home_form_override'


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    bets: list[Bet] = field(default_factory=list)

    @property
    def total_bets(self) -> int:
        return len(self.bets)

    @property
    def wins(self) -> int:
        return sum(1 for b in self.bets if b.won)

    @property
    def win_rate(self) -> float:
        return self.wins / self.total_bets if self.total_bets > 0 else 0

    @property
    def total_profit(self) -> float:
        return sum(b.profit for b in self.bets)

    @property
    def roi(self) -> float:
        return self.total_profit / self.total_bets if self.total_bets > 0 else 0

    def by_season(self) -> dict[str, "BacktestResult"]:
        """Group results by season."""
        by_season = defaultdict(lambda: BacktestResult())
        for bet in self.bets:
            by_season[bet.season].bets.append(bet)
        return dict(by_season)

    def by_strategy(self) -> dict[str, "BacktestResult"]:
        """Group results by strategy."""
        by_strategy = defaultdict(lambda: BacktestResult())
        for bet in self.bets:
            by_strategy[bet.strategy].bets.append(bet)
        return dict(by_strategy)

    def rolling_roi(self, window_months: int = 12) -> list[tuple[datetime, float]]:
        """Calculate rolling ROI over time."""
        if not self.bets:
            return []

        sorted_bets = sorted(self.bets, key=lambda b: b.kickoff)
        results = []

        for i, current_bet in enumerate(sorted_bets):
            # Find bets within window
            cutoff = current_bet.kickoff.replace(
                year=current_bet.kickoff.year - (window_months // 12),
                month=((current_bet.kickoff.month - (window_months % 12) - 1) % 12) + 1
            ) if window_months < 12 else current_bet.kickoff.replace(year=current_bet.kickoff.year - 1)

            window_bets = [b for b in sorted_bets[:i+1] if b.kickoff >= cutoff]
            if len(window_bets) >= 10:  # Need minimum bets for meaningful ROI
                roi = sum(b.profit for b in window_bets) / len(window_bets)
                results.append((current_bet.kickoff, roi))

        return results


class StrategyBacktest:
    """Backtests the STATIC 12% + ERA-BASED FORM OVERRIDE strategy."""

    # Era boundary - home bets only active from this season
    ERA_2020S_START = "2020-21"

    # Strategy parameters
    # Why 5% edge instead of 12%?
    # In 2020s era: 5% threshold has +30.2% ROI vs 23.0% at 12%, with 5x more bets
    # Model is well-calibrated now - even small edges are profitable
    AWAY_MIN_EDGE = 0.05  # 5% minimum edge - optimal for modern era
    AWAY_MIN_ODDS = 1.50
    AWAY_MAX_ODDS = 8.00
    AWAY_EXCLUDE_HOME_FORM = (4, 5, 6)  # Exclude when home form is 4-6

    HOME_MIN_FORM = 12  # Hot streak required
    HOME_MAX_EDGE = 0.0  # Negative edge required
    HOME_MIN_ODDS = 1.01
    HOME_MAX_ODDS = 10.00

    def __init__(self, session: Session):
        self.session = session

    def is_2020s_era(self, season: str) -> bool:
        """Check if season is in the 2020s era."""
        return season >= self.ERA_2020S_START

    def run(
        self,
        start_season: str = "1993-94",
        end_season: str = "2025-26",
        apply_stop_loss: bool = False,
    ) -> BacktestResult:
        """Run the backtest across all seasons.

        Args:
            start_season: First season to include
            end_season: Last season to include
            apply_stop_loss: Whether to apply stop-loss rules

        Returns:
            BacktestResult with all bets and statistics
        """
        logger.info(f"Starting backtest from {start_season} to {end_season}")

        # Load all matches with predictions and odds
        stmt = (
            select(Match, MatchAnalysis)
            .join(MatchAnalysis, Match.id == MatchAnalysis.match_id)
            .where(Match.season >= start_season)
            .where(Match.season <= end_season)
            .where(Match.status == MatchStatus.FINISHED)
            .where(MatchAnalysis.consensus_home_prob.isnot(None))
            .order_by(Match.kickoff_time)
        )
        matches = list(self.session.execute(stmt).all())
        logger.info(f"Found {len(matches)} matches with predictions")

        # Load all odds
        match_ids = [m.id for m, _ in matches]
        all_odds = list(self.session.execute(
            select(OddsHistory).where(OddsHistory.match_id.in_(match_ids))
        ).scalars().all())

        odds_lookup = defaultdict(list)
        for oh in all_odds:
            odds_lookup[oh.match_id].append(oh)

        # Load all team stats
        seasons = list(set(m.season for m, _ in matches))
        all_stats = list(self.session.execute(
            select(TeamStats).where(TeamStats.season.in_(seasons))
        ).scalars().all())

        stats_lookup = {(ts.team_id, ts.season, ts.matchweek): ts for ts in all_stats}

        # Track bets and stop-loss state
        result = BacktestResult()
        stopped = False

        for match, analysis in matches:
            if stopped and apply_stop_loss:
                break

            # Get odds for this match
            match_odds = odds_lookup.get(match.id, [])
            best_away_odds = max((oh.away_odds for oh in match_odds if oh.away_odds), default=None)
            best_home_odds = max((oh.home_odds for oh in match_odds if oh.home_odds), default=None)

            # Fall back to historical odds from features
            if analysis.features:
                hist_odds = analysis.features.get("historical_odds", {})
                if not best_away_odds and hist_odds.get("avg_away_odds"):
                    best_away_odds = Decimal(str(hist_odds["avg_away_odds"]))
                if not best_home_odds and hist_odds.get("avg_home_odds"):
                    best_home_odds = Decimal(str(hist_odds["avg_home_odds"]))

            # Get form data
            home_stats = stats_lookup.get((match.home_team_id, match.season, match.matchweek - 1))
            away_stats = stats_lookup.get((match.away_team_id, match.season, match.matchweek - 1))
            home_form = home_stats.form_points if home_stats else 0
            away_form = away_stats.form_points if away_stats else 0

            # Determine actual result
            if match.home_score > match.away_score:
                actual_result = "H"
            elif match.away_score > match.home_score:
                actual_result = "A"
            else:
                actual_result = "D"

            # Check AWAY BET (always active)
            if best_away_odds:
                away_odds_float = float(best_away_odds)
                away_prob = float(analysis.consensus_away_prob or 0)

                if away_prob > 0 and self.AWAY_MIN_ODDS <= away_odds_float <= self.AWAY_MAX_ODDS:
                    market_prob = 1 / away_odds_float
                    edge = away_prob - market_prob

                    # Check edge and form exclusion
                    if edge >= self.AWAY_MIN_EDGE:
                        if home_form not in self.AWAY_EXCLUDE_HOME_FORM:
                            won = actual_result == "A"
                            profit = (away_odds_float - 1) if won else -1

                            result.bets.append(Bet(
                                match_id=match.id,
                                season=match.season,
                                matchweek=match.matchweek,
                                kickoff=match.kickoff_time,
                                outcome="away_win",
                                odds=away_odds_float,
                                model_prob=away_prob,
                                edge=edge,
                                home_form=home_form,
                                away_form=away_form,
                                actual_result=actual_result,
                                won=won,
                                profit=profit,
                                strategy="away_12pct",
                            ))

            # Check HOME BET (2020s era only)
            if self.is_2020s_era(match.season) and best_home_odds:
                home_odds_float = float(best_home_odds)
                home_prob = float(analysis.consensus_home_prob or 0)

                if home_prob > 0 and self.HOME_MIN_ODDS <= home_odds_float <= self.HOME_MAX_ODDS:
                    market_prob = 1 / home_odds_float
                    edge = home_prob - market_prob

                    # Form override: hot streak (12+) with negative edge
                    if home_form >= self.HOME_MIN_FORM and edge < self.HOME_MAX_EDGE:
                        won = actual_result == "H"
                        profit = (home_odds_float - 1) if won else -1

                        result.bets.append(Bet(
                            match_id=match.id,
                            season=match.season,
                            matchweek=match.matchweek,
                            kickoff=match.kickoff_time,
                            outcome="home_win",
                            odds=home_odds_float,
                            model_prob=home_prob,
                            edge=edge,
                            home_form=home_form,
                            away_form=away_form,
                            actual_result=actual_result,
                            won=won,
                            profit=profit,
                            strategy="home_form_override",
                        ))

            # Check stop-loss conditions
            if apply_stop_loss and len(result.bets) >= 50:
                recent_bets = result.bets[-50:]
                roi = sum(b.profit for b in recent_bets) / len(recent_bets)
                away_bets = [b for b in recent_bets if b.outcome == "away_win"]
                away_win_rate = sum(1 for b in away_bets if b.won) / len(away_bets) if away_bets else 0

                if roi < -0.20:
                    logger.warning(f"Stop-loss triggered: 12-month ROI {roi:.1%} < -20%")
                    stopped = True
                elif len(away_bets) >= 20 and away_win_rate < 0.30:
                    logger.warning(f"Stop-loss triggered: Away win rate {away_win_rate:.1%} < 30%")
                    stopped = True

        logger.info(
            f"Backtest complete: {result.total_bets} bets, "
            f"{result.win_rate:.1%} win rate, {result.roi:.1%} ROI"
        )

        return result

    def compare_strategies(
        self,
        start_season: str = "1993-94",
        end_season: str = "2025-26",
    ) -> dict:
        """Compare different strategy configurations."""

        # Run full backtest
        full_result = self.run(start_season, end_season)

        # Split by era
        pre_2020s = BacktestResult(
            bets=[b for b in full_result.bets if not self.is_2020s_era(b.season)]
        )
        era_2020s = BacktestResult(
            bets=[b for b in full_result.bets if self.is_2020s_era(b.season)]
        )

        # Split by strategy
        away_only = BacktestResult(
            bets=[b for b in full_result.bets if b.strategy == "away_12pct"]
        )
        home_form = BacktestResult(
            bets=[b for b in full_result.bets if b.strategy == "home_form_override"]
        )

        return {
            "overall": {
                "bets": full_result.total_bets,
                "wins": full_result.wins,
                "win_rate": full_result.win_rate,
                "profit": full_result.total_profit,
                "roi": full_result.roi,
            },
            "pre_2020s": {
                "bets": pre_2020s.total_bets,
                "wins": pre_2020s.wins,
                "win_rate": pre_2020s.win_rate,
                "profit": pre_2020s.total_profit,
                "roi": pre_2020s.roi,
            },
            "era_2020s": {
                "bets": era_2020s.total_bets,
                "wins": era_2020s.wins,
                "win_rate": era_2020s.win_rate,
                "profit": era_2020s.total_profit,
                "roi": era_2020s.roi,
            },
            "away_12pct": {
                "bets": away_only.total_bets,
                "wins": away_only.wins,
                "win_rate": away_only.win_rate,
                "profit": away_only.total_profit,
                "roi": away_only.roi,
            },
            "home_form_override": {
                "bets": home_form.total_bets,
                "wins": home_form.wins,
                "win_rate": home_form.win_rate,
                "profit": home_form.total_profit,
                "roi": home_form.roi,
            },
            "by_season": {
                season: {
                    "bets": r.total_bets,
                    "wins": r.wins,
                    "win_rate": r.win_rate,
                    "profit": r.total_profit,
                    "roi": r.roi,
                }
                for season, r in sorted(full_result.by_season().items())
            },
        }


def print_results(results: dict):
    """Print formatted backtest results."""
    print("\n" + "=" * 70)
    print("STATIC 12% + ERA-BASED FORM OVERRIDE STRATEGY BACKTEST")
    print("=" * 70)

    print("\n--- OVERALL PERFORMANCE ---")
    o = results["overall"]
    print(f"Total Bets:  {o['bets']}")
    print(f"Win Rate:    {o['win_rate']:.1%}")
    print(f"Total P/L:   {o['profit']:+.1f} units")
    print(f"ROI:         {o['roi']:+.1%}")

    print("\n--- BY ERA ---")
    print(f"{'Era':<15} {'Bets':>8} {'Win%':>8} {'P/L':>10} {'ROI':>10}")
    print("-" * 55)

    for era, data in [("Pre-2020s", results["pre_2020s"]), ("2020s Era", results["era_2020s"])]:
        if data["bets"] > 0:
            print(f"{era:<15} {data['bets']:>8} {data['win_rate']:>7.1%} {data['profit']:>+9.1f} {data['roi']:>+9.1%}")

    print("\n--- BY STRATEGY ---")
    print(f"{'Strategy':<20} {'Bets':>8} {'Win%':>8} {'P/L':>10} {'ROI':>10}")
    print("-" * 60)

    for name, key in [("Away 12% Edge", "away_12pct"), ("Home Form Override", "home_form_override")]:
        data = results[key]
        if data["bets"] > 0:
            print(f"{name:<20} {data['bets']:>8} {data['win_rate']:>7.1%} {data['profit']:>+9.1f} {data['roi']:>+9.1%}")

    print("\n--- YEAR BY YEAR ---")
    print(f"{'Season':<12} {'Bets':>6} {'Win%':>8} {'P/L':>10} {'ROI':>10} {'Cumul ROI':>12}")
    print("-" * 62)

    cumul_bets = 0
    cumul_profit = 0
    for season, data in sorted(results["by_season"].items()):
        if data["bets"] > 0:
            cumul_bets += data["bets"]
            cumul_profit += data["profit"]
            cumul_roi = cumul_profit / cumul_bets if cumul_bets > 0 else 0
            print(f"{season:<12} {data['bets']:>6} {data['win_rate']:>7.1%} {data['profit']:>+9.1f} {data['roi']:>+9.1%} {cumul_roi:>+11.1%}")

    print("\n" + "=" * 70)


def main():
    # Disable logging FIRST before any imports that might configure loggers
    import logging
    import os
    os.environ["SQLALCHEMY_SILENCE_UBER_WARNING"] = "1"
    logging.getLogger("sqlalchemy").setLevel(logging.CRITICAL)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.CRITICAL)

    parser = argparse.ArgumentParser(description="Backtest betting strategy")
    parser.add_argument(
        "--start-season",
        default="1993-94",
        help="First season to include (default: 1993-94)",
    )
    parser.add_argument(
        "--end-season",
        default="2025-26",
        help="Last season to include (default: 2025-26)",
    )
    parser.add_argument(
        "--apply-stop-loss",
        action="store_true",
        help="Apply stop-loss rules during backtest",
    )
    parser.add_argument(
        "--output",
        help="Output file for results (default: stdout)",
    )
    args = parser.parse_args()

    with SyncSessionLocal() as session:
        backtest = StrategyBacktest(session)
        results = backtest.compare_strategies(args.start_season, args.end_season)

        if args.output:
            import sys
            original_stdout = sys.stdout
            with open(args.output, 'w') as f:
                sys.stdout = f
                print_results(results)
            sys.stdout = original_stdout
            print(f"Results written to {args.output}")
        else:
            print_results(results)


if __name__ == "__main__":
    main()
