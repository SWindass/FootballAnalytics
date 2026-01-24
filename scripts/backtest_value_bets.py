"""Backtest value betting strategy on historical data.

Simulates what would have happened if we bet on high-confidence predictions
using Kelly criterion stake sizing.
"""

import argparse
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Optional

from sqlalchemy import select

from app.db.database import SyncSessionLocal
from app.db.models import Match, MatchAnalysis, MatchStatus, Team


@dataclass
class BacktestConfig:
    """Configuration for backtest."""
    min_confidence: float = 0.60  # Minimum model confidence to bet
    min_edge: float = 0.05  # Minimum edge over implied odds
    bookmaker_margin: float = 0.05  # Typical bookmaker overround
    kelly_fraction: float = 0.25  # Fractional Kelly
    initial_bankroll: float = 1000.0  # Starting bankroll
    min_odds: float = 1.5  # Minimum odds to bet
    max_odds: float = 10.0  # Maximum odds to bet


@dataclass
class Bet:
    """A simulated bet."""
    match_id: int
    season: str
    matchweek: int
    home_team: str
    away_team: str
    outcome: str  # 'home', 'draw', 'away'
    model_prob: float
    implied_prob: float
    odds: float
    edge: float
    stake: float
    stake_pct: float
    won: bool
    profit: float
    bankroll_after: float


def calculate_implied_odds(prob: float, margin: float) -> float:
    """Calculate bookmaker odds from true probability with margin."""
    # Fair odds would be 1/prob
    # Bookmaker adds margin, reducing the odds
    fair_odds = 1 / prob
    # Apply margin (bookmaker takes ~5% edge)
    return fair_odds * (1 - margin)


def calculate_kelly_stake(prob: float, odds: float, fraction: float = 0.25) -> float:
    """Calculate Kelly criterion stake as fraction of bankroll."""
    # Kelly formula: f = (bp - q) / b
    # where b = odds - 1, p = win prob, q = lose prob
    b = odds - 1
    p = prob
    q = 1 - p

    kelly = (b * p - q) / b

    # Apply fractional Kelly and cap at 10%
    stake = max(0, min(kelly * fraction, 0.10))
    return stake


def run_backtest(config: BacktestConfig, seasons: list[str] = None, verbose: bool = False):
    """Run backtest on historical data."""

    with SyncSessionLocal() as session:
        # Load teams
        teams = {t.id: t.short_name for t in session.execute(select(Team)).scalars().all()}

        # Build query for finished matches with predictions
        stmt = (
            select(Match, MatchAnalysis)
            .join(MatchAnalysis, Match.id == MatchAnalysis.match_id)
            .where(Match.status == MatchStatus.FINISHED)
            .where(MatchAnalysis.consensus_home_prob.isnot(None))
            .order_by(Match.season, Match.matchweek, Match.kickoff_time)
        )

        if seasons:
            stmt = stmt.where(Match.season.in_(seasons))

        results = list(session.execute(stmt).all())

    print(f"Loaded {len(results)} matches with predictions")
    print(f"Config: min_confidence={config.min_confidence:.0%}, min_edge={config.min_edge:.0%}, kelly_fraction={config.kelly_fraction}")
    print("=" * 80)

    # Track results
    bets: list[Bet] = []
    bankroll = config.initial_bankroll

    # Stats by confidence level
    confidence_buckets = {
        "50-55%": {"bets": 0, "wins": 0, "profit": 0},
        "55-60%": {"bets": 0, "wins": 0, "profit": 0},
        "60-65%": {"bets": 0, "wins": 0, "profit": 0},
        "65-70%": {"bets": 0, "wins": 0, "profit": 0},
        "70-75%": {"bets": 0, "wins": 0, "profit": 0},
        "75%+": {"bets": 0, "wins": 0, "profit": 0},
    }

    for match, analysis in results:
        home_team = teams.get(match.home_team_id, "?")
        away_team = teams.get(match.away_team_id, "?")

        # Determine actual result
        if match.home_score > match.away_score:
            actual = "home"
        elif match.home_score < match.away_score:
            actual = "away"
        else:
            actual = "draw"

        # Get consensus probabilities
        probs = {
            "home": float(analysis.consensus_home_prob),
            "draw": float(analysis.consensus_draw_prob),
            "away": float(analysis.consensus_away_prob),
        }

        # Check each outcome for value
        for outcome, model_prob in probs.items():
            # Track by confidence bucket (for analysis)
            if 0.50 <= model_prob < 0.55:
                bucket = "50-55%"
            elif 0.55 <= model_prob < 0.60:
                bucket = "55-60%"
            elif 0.60 <= model_prob < 0.65:
                bucket = "60-65%"
            elif 0.65 <= model_prob < 0.70:
                bucket = "65-70%"
            elif 0.70 <= model_prob < 0.75:
                bucket = "70-75%"
            elif model_prob >= 0.75:
                bucket = "75%+"
            else:
                continue

            # Skip if below confidence threshold
            if model_prob < config.min_confidence:
                continue

            # Calculate simulated bookmaker odds
            # Assume market is roughly efficient, so implied prob ≈ true prob
            # But bookmaker adds margin
            implied_prob = model_prob * (1 + config.bookmaker_margin)  # Market slightly underestimates
            implied_prob = min(implied_prob, 0.95)  # Cap at 95%

            # For simulation: assume we can get odds slightly better than implied
            # This represents finding value at different bookmakers
            odds = 1 / (model_prob - config.min_edge)  # Odds we'd need for min_edge

            # Check odds within range
            if odds < config.min_odds or odds > config.max_odds:
                continue

            # Calculate edge
            edge = model_prob - (1 / odds)

            if edge < config.min_edge:
                continue

            # Calculate stake
            stake_pct = calculate_kelly_stake(model_prob, odds, config.kelly_fraction)
            stake = bankroll * stake_pct

            if stake <= 0:
                continue

            # Determine if bet won
            won = (outcome == actual)

            # Calculate profit
            if won:
                profit = stake * (odds - 1)
            else:
                profit = -stake

            bankroll += profit

            # Record bet
            bet = Bet(
                match_id=match.id,
                season=match.season,
                matchweek=match.matchweek,
                home_team=home_team,
                away_team=away_team,
                outcome=outcome,
                model_prob=model_prob,
                implied_prob=implied_prob,
                odds=odds,
                edge=edge,
                stake=stake,
                stake_pct=stake_pct,
                won=won,
                profit=profit,
                bankroll_after=bankroll,
            )
            bets.append(bet)

            # Update bucket stats
            confidence_buckets[bucket]["bets"] += 1
            if won:
                confidence_buckets[bucket]["wins"] += 1
            confidence_buckets[bucket]["profit"] += profit

            if verbose:
                result_str = "✓" if won else "✗"
                print(f"{match.season} MW{match.matchweek}: {home_team} vs {away_team} | "
                      f"{outcome.upper()} @ {odds:.2f} | Model: {model_prob:.1%} | "
                      f"Stake: ${stake:.2f} ({stake_pct:.1%}) | {result_str} ${profit:+.2f} | "
                      f"Bankroll: ${bankroll:.2f}")

    # Summary
    print("\n" + "=" * 80)
    print("BACKTEST RESULTS")
    print("=" * 80)

    if not bets:
        print("No bets placed with current configuration.")
        return

    total_bets = len(bets)
    wins = sum(1 for b in bets if b.won)
    losses = total_bets - wins
    win_rate = wins / total_bets

    total_staked = sum(b.stake for b in bets)
    total_profit = sum(b.profit for b in bets)
    roi = (total_profit / total_staked) * 100 if total_staked > 0 else 0

    final_bankroll = bets[-1].bankroll_after
    total_return = ((final_bankroll - config.initial_bankroll) / config.initial_bankroll) * 100

    print(f"\nTotal Bets: {total_bets}")
    print(f"Wins: {wins} ({win_rate:.1%})")
    print(f"Losses: {losses}")
    print(f"\nTotal Staked: ${total_staked:.2f}")
    print(f"Total Profit: ${total_profit:+.2f}")
    print(f"ROI: {roi:+.1f}%")
    print(f"\nInitial Bankroll: ${config.initial_bankroll:.2f}")
    print(f"Final Bankroll: ${final_bankroll:.2f}")
    print(f"Total Return: {total_return:+.1f}%")

    # Breakdown by confidence
    print("\n" + "-" * 80)
    print("BREAKDOWN BY CONFIDENCE LEVEL")
    print("-" * 80)
    print(f"{'Confidence':<12} {'Bets':>8} {'Wins':>8} {'Win%':>8} {'Profit':>12} {'ROI':>10}")
    print("-" * 80)

    for bucket, stats in confidence_buckets.items():
        if stats["bets"] > 0:
            bucket_win_rate = stats["wins"] / stats["bets"] * 100
            # Estimate staked (rough)
            bucket_roi = "N/A"
            print(f"{bucket:<12} {stats['bets']:>8} {stats['wins']:>8} {bucket_win_rate:>7.1f}% ${stats['profit']:>+10.2f}")

    # Breakdown by season
    print("\n" + "-" * 80)
    print("BREAKDOWN BY SEASON")
    print("-" * 80)

    seasons_seen = sorted(set(b.season for b in bets))
    print(f"{'Season':<12} {'Bets':>8} {'Wins':>8} {'Win%':>8} {'Profit':>12}")
    print("-" * 80)

    for season in seasons_seen:
        season_bets = [b for b in bets if b.season == season]
        season_wins = sum(1 for b in season_bets if b.won)
        season_profit = sum(b.profit for b in season_bets)
        season_win_rate = season_wins / len(season_bets) * 100 if season_bets else 0
        print(f"{season:<12} {len(season_bets):>8} {season_wins:>8} {season_win_rate:>7.1f}% ${season_profit:>+10.2f}")

    # Breakdown by bet type
    print("\n" + "-" * 80)
    print("BREAKDOWN BY BET TYPE")
    print("-" * 80)

    for outcome in ["home", "draw", "away"]:
        outcome_bets = [b for b in bets if b.outcome == outcome]
        if outcome_bets:
            outcome_wins = sum(1 for b in outcome_bets if b.won)
            outcome_profit = sum(b.profit for b in outcome_bets)
            outcome_win_rate = outcome_wins / len(outcome_bets) * 100
            print(f"{outcome.upper():<12} {len(outcome_bets):>8} {outcome_wins:>8} {outcome_win_rate:>7.1f}% ${outcome_profit:>+10.2f}")

    # Max drawdown
    peak = config.initial_bankroll
    max_drawdown = 0
    for bet in bets:
        if bet.bankroll_after > peak:
            peak = bet.bankroll_after
        drawdown = (peak - bet.bankroll_after) / peak
        max_drawdown = max(max_drawdown, drawdown)

    print(f"\nMax Drawdown: {max_drawdown:.1%}")

    # Longest losing streak
    current_streak = 0
    max_streak = 0
    for bet in bets:
        if not bet.won:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0

    print(f"Longest Losing Streak: {max_streak} bets")

    return bets


def main():
    parser = argparse.ArgumentParser(description="Backtest value betting strategy")
    parser.add_argument("--seasons", nargs="+", help="Seasons to include (e.g., 2023-24 2024-25)")
    parser.add_argument("--min-confidence", type=float, default=0.60, help="Minimum confidence threshold")
    parser.add_argument("--min-edge", type=float, default=0.05, help="Minimum edge threshold")
    parser.add_argument("--kelly-fraction", type=float, default=0.25, help="Kelly fraction")
    parser.add_argument("--bankroll", type=float, default=1000.0, help="Initial bankroll")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show individual bets")

    args = parser.parse_args()

    config = BacktestConfig(
        min_confidence=args.min_confidence,
        min_edge=args.min_edge,
        kelly_fraction=args.kelly_fraction,
        initial_bankroll=args.bankroll,
    )

    run_backtest(config, args.seasons, args.verbose)


if __name__ == "__main__":
    main()
