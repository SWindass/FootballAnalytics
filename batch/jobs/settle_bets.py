"""Settle value bets after matches finish.

Runs periodically to:
1. Find active value bets where match is finished
2. Determine win/loss based on actual result
3. Calculate profit/loss
4. Update bet records
"""

from datetime import datetime
from decimal import Decimal
from typing import Optional

import structlog
from sqlalchemy import select, and_
from sqlalchemy.orm import Session

from app.db.database import SyncSessionLocal
from app.db.models import Match, MatchStatus, Team, ValueBet

logger = structlog.get_logger()

# Standard stake for P/L calculation (percentage of bankroll)
STANDARD_STAKE = Decimal("10.00")  # $10 flat stake for tracking


def determine_bet_result(
    outcome: str,
    home_score: int,
    away_score: int,
    home_xg: Optional[Decimal] = None,
    away_xg: Optional[Decimal] = None,
) -> str:
    """Determine if a bet won or lost based on match result.

    Args:
        outcome: Bet outcome type ('home_win', 'away_win', 'draw', 'over_2_5', etc.)
        home_score: Final home score
        away_score: Final away score
        home_xg: Home expected goals (for xG-based markets)
        away_xg: Away expected goals (for xG-based markets)

    Returns:
        'won', 'lost', or 'void'
    """
    total_goals = home_score + away_score

    if outcome == "home_win":
        return "won" if home_score > away_score else "lost"

    elif outcome == "away_win":
        return "won" if away_score > home_score else "lost"

    elif outcome == "draw":
        return "won" if home_score == away_score else "lost"

    elif outcome == "over_2_5":
        return "won" if total_goals > 2.5 else "lost"

    elif outcome == "under_2_5":
        return "won" if total_goals < 2.5 else "lost"

    elif outcome == "over_1_5":
        return "won" if total_goals > 1.5 else "lost"

    elif outcome == "under_1_5":
        return "won" if total_goals < 1.5 else "lost"

    elif outcome == "btts_yes":
        return "won" if home_score > 0 and away_score > 0 else "lost"

    elif outcome == "btts_no":
        return "won" if home_score == 0 or away_score == 0 else "lost"

    else:
        logger.warning(f"Unknown outcome type: {outcome}")
        return "void"


def calculate_profit_loss(result: str, odds: Decimal, stake: Decimal = STANDARD_STAKE) -> Decimal:
    """Calculate profit/loss for a bet.

    Args:
        result: 'won', 'lost', or 'void'
        odds: Decimal odds
        stake: Stake amount

    Returns:
        Profit (positive) or loss (negative)
    """
    if result == "won":
        return stake * (odds - 1)
    elif result == "lost":
        return -stake
    else:  # void
        return Decimal("0.00")


def settle_bets(session: Optional[Session] = None) -> dict:
    """Settle all active value bets where match is finished.

    Returns:
        Summary of settled bets
    """
    close_session = False
    if session is None:
        session = SyncSessionLocal()
        close_session = True

    try:
        # Find active value bets with finished matches
        stmt = (
            select(ValueBet)
            .join(Match, ValueBet.match_id == Match.id)
            .where(ValueBet.is_active == True)
            .where(ValueBet.result.is_(None))
            .where(Match.status == MatchStatus.FINISHED)
        )

        active_bets = list(session.execute(stmt).scalars().all())

        logger.info(f"Found {len(active_bets)} bets to settle")

        settled = 0
        won = 0
        lost = 0
        void = 0
        total_profit = Decimal("0.00")

        teams_cache = {}

        for bet in active_bets:
            match = session.get(Match, bet.match_id)

            if match.home_score is None or match.away_score is None:
                logger.warning(f"Match {match.id} marked finished but no score")
                continue

            # Get team names for logging
            if match.home_team_id not in teams_cache:
                home_team = session.get(Team, match.home_team_id)
                away_team = session.get(Team, match.away_team_id)
                teams_cache[match.home_team_id] = home_team.short_name
                teams_cache[match.away_team_id] = away_team.short_name

            home_name = teams_cache[match.home_team_id]
            away_name = teams_cache[match.away_team_id]

            # Determine result
            result = determine_bet_result(
                bet.outcome,
                match.home_score,
                match.away_score,
                match.home_xg,
                match.away_xg,
            )

            # Calculate P/L
            profit_loss = calculate_profit_loss(result, bet.odds)

            # Update bet
            bet.result = result
            bet.profit_loss = profit_loss
            bet.is_active = False

            settled += 1
            total_profit += profit_loss

            if result == "won":
                won += 1
                logger.info(
                    f"WON: {home_name} vs {away_name} | {bet.outcome} @ {bet.odds} | "
                    f"Score: {match.home_score}-{match.away_score} | P/L: ${profit_loss:+.2f}"
                )
            elif result == "lost":
                lost += 1
                logger.info(
                    f"LOST: {home_name} vs {away_name} | {bet.outcome} @ {bet.odds} | "
                    f"Score: {match.home_score}-{match.away_score} | P/L: ${profit_loss:+.2f}"
                )
            else:
                void += 1
                logger.info(f"VOID: {home_name} vs {away_name} | {bet.outcome}")

        session.commit()

        summary = {
            "settled": settled,
            "won": won,
            "lost": lost,
            "void": void,
            "total_profit": float(total_profit),
            "win_rate": won / settled * 100 if settled > 0 else 0,
        }

        logger.info(
            "Bet settlement complete",
            **summary
        )

        return summary

    finally:
        if close_session:
            session.close()


def get_betting_performance(session: Optional[Session] = None) -> dict:
    """Get overall betting performance stats.

    Returns:
        Performance summary
    """
    close_session = False
    if session is None:
        session = SyncSessionLocal()
        close_session = True

    try:
        # Get all settled bets
        stmt = select(ValueBet).where(ValueBet.result.isnot(None))
        settled_bets = list(session.execute(stmt).scalars().all())

        if not settled_bets:
            return {"message": "No settled bets yet"}

        total_bets = len(settled_bets)
        won = sum(1 for b in settled_bets if b.result == "won")
        lost = sum(1 for b in settled_bets if b.result == "lost")
        void = sum(1 for b in settled_bets if b.result == "void")

        total_profit = sum(b.profit_loss or 0 for b in settled_bets)
        total_staked = STANDARD_STAKE * (won + lost)  # Don't count void

        roi = (total_profit / total_staked * 100) if total_staked > 0 else 0

        # By outcome type
        by_outcome = {}
        for bet in settled_bets:
            if bet.outcome not in by_outcome:
                by_outcome[bet.outcome] = {"bets": 0, "won": 0, "profit": Decimal("0")}
            by_outcome[bet.outcome]["bets"] += 1
            if bet.result == "won":
                by_outcome[bet.outcome]["won"] += 1
            by_outcome[bet.outcome]["profit"] += bet.profit_loss or 0

        # By month
        by_month = {}
        for bet in settled_bets:
            month_key = bet.created_at.strftime("%Y-%m")
            if month_key not in by_month:
                by_month[month_key] = {"bets": 0, "won": 0, "profit": Decimal("0")}
            by_month[month_key]["bets"] += 1
            if bet.result == "won":
                by_month[month_key]["won"] += 1
            by_month[month_key]["profit"] += bet.profit_loss or 0

        return {
            "total_bets": total_bets,
            "won": won,
            "lost": lost,
            "void": void,
            "win_rate": won / (won + lost) * 100 if (won + lost) > 0 else 0,
            "total_profit": float(total_profit),
            "total_staked": float(total_staked),
            "roi": float(roi),
            "by_outcome": {k: {"bets": v["bets"], "won": v["won"], "profit": float(v["profit"])}
                          for k, v in by_outcome.items()},
            "by_month": {k: {"bets": v["bets"], "won": v["won"], "profit": float(v["profit"])}
                        for k, v in sorted(by_month.items())},
        }

    finally:
        if close_session:
            session.close()


def print_performance_report():
    """Print a formatted performance report."""
    stats = get_betting_performance()

    if "message" in stats:
        print(stats["message"])
        return

    print("=" * 70)
    print("BETTING PERFORMANCE REPORT")
    print("=" * 70)

    print(f"\nOverall Stats:")
    print(f"  Total Bets: {stats['total_bets']}")
    print(f"  Won: {stats['won']} | Lost: {stats['lost']} | Void: {stats['void']}")
    print(f"  Win Rate: {stats['win_rate']:.1f}%")
    print(f"  Total Staked: ${stats['total_staked']:.2f}")
    print(f"  Total Profit: ${stats['total_profit']:+.2f}")
    print(f"  ROI: {stats['roi']:+.1f}%")

    print(f"\nBy Outcome Type:")
    print(f"  {'Outcome':<15} {'Bets':>6} {'Won':>6} {'Win%':>8} {'Profit':>10}")
    print(f"  {'-'*50}")
    for outcome, data in stats['by_outcome'].items():
        win_rate = data['won'] / data['bets'] * 100 if data['bets'] > 0 else 0
        print(f"  {outcome:<15} {data['bets']:>6} {data['won']:>6} {win_rate:>7.1f}% ${data['profit']:>+9.2f}")

    print(f"\nBy Month:")
    print(f"  {'Month':<10} {'Bets':>6} {'Won':>6} {'Win%':>8} {'Profit':>10}")
    print(f"  {'-'*45}")
    for month, data in stats['by_month'].items():
        win_rate = data['won'] / data['bets'] * 100 if data['bets'] > 0 else 0
        print(f"  {month:<10} {data['bets']:>6} {data['won']:>6} {win_rate:>7.1f}% ${data['profit']:>+9.2f}")


if __name__ == "__main__":
    import argparse
    import logging

    parser = argparse.ArgumentParser(description="Settle value bets and track performance")
    parser.add_argument("--settle", action="store_true", help="Settle pending bets")
    parser.add_argument("--report", action="store_true", help="Show performance report")

    args = parser.parse_args()

    # Reduce SQLAlchemy noise
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

    if args.settle or (not args.settle and not args.report):
        result = settle_bets()
        print(f"\nSettled {result['settled']} bets: {result['won']} won, {result['lost']} lost")
        print(f"Session P/L: ${result['total_profit']:+.2f}")

    if args.report:
        print()
        print_performance_report()
