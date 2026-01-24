"""Calibrated backtest - uses actual historical win rates.

Key insight: Model is overconfident. At 70%+ confidence, actual win rate is ~60%.
So we should only bet when odds imply less than 60%.
"""

from sqlalchemy import select
from app.db.database import SyncSessionLocal
from app.db.models import Match, MatchAnalysis, MatchStatus, Team


def run_calibrated_backtest():
    """Run backtest using calibrated probabilities."""

    with SyncSessionLocal() as session:
        teams = {t.id: t.short_name for t in session.execute(select(Team)).scalars().all()}

        stmt = (
            select(Match, MatchAnalysis)
            .join(MatchAnalysis, Match.id == MatchAnalysis.match_id)
            .where(Match.status == MatchStatus.FINISHED)
            .where(MatchAnalysis.consensus_home_prob.isnot(None))
            .order_by(Match.season, Match.matchweek)
        )
        results = list(session.execute(stmt).all())

    print(f"Analyzing {len(results)} matches")
    print("=" * 90)

    # Calibration map: model confidence -> actual win rate (from previous analysis)
    # These are the REAL probabilities based on historical data
    calibration = {
        0.60: 0.566,  # Model says 60%, actually wins 56.6%
        0.65: 0.585,  # Model says 65%, actually wins 58.5%
        0.70: 0.607,  # Model says 70%, actually wins 60.7%
        0.75: 0.667,  # Model says 75%, actually wins 66.7%
    }

    print("\nCALIBRATION TABLE:")
    print(f"{'Model Confidence':<20} {'Actual Win Rate':<20} {'Min Odds for Profit':<20}")
    print("-" * 60)
    for model_conf, actual in calibration.items():
        min_odds = 1 / actual
        print(f"{model_conf:.0%}                  {actual:.1%}                 {min_odds:.2f}")

    # Now simulate betting with realistic odds
    # Assume we can shop around and find odds close to "true" market odds
    # Market odds typically have 5-8% margin

    print("\n" + "=" * 90)
    print("STRATEGY: Bet when model >= 70% (actual 60.7% win rate)")
    print("Only bet if available odds >= 1.65 (implied prob <= 60.6%)")
    print("=" * 90)

    # Simulate various odds scenarios
    odds_scenarios = [
        ("Fair odds (no margin)", 0.00),
        ("2% edge over actual", -0.02),  # Odds imply 58.7% when we win 60.7%
        ("5% bookmaker margin", 0.05),
        ("3% bookmaker margin", 0.03),
    ]

    stake = 10
    threshold = 0.70
    actual_win_rate = 0.607

    for scenario_name, margin in odds_scenarios:
        bets = 0
        wins = 0
        total_profit = 0

        for match, analysis in results:
            if match.home_score > match.away_score:
                actual = "home"
            elif match.home_score < match.away_score:
                actual = "away"
            else:
                actual = "draw"

            probs = {
                "home": float(analysis.consensus_home_prob),
                "draw": float(analysis.consensus_draw_prob),
                "away": float(analysis.consensus_away_prob),
            }

            best_outcome = max(probs, key=probs.get)
            best_prob = probs[best_outcome]

            if best_prob >= threshold:
                # Simulate odds: based on actual win rate + margin
                # If margin is negative, we're getting BETTER than fair odds
                implied_prob = actual_win_rate + margin
                if implied_prob >= 1 or implied_prob <= 0:
                    continue
                odds = 1 / implied_prob

                bets += 1
                if best_outcome == actual:
                    wins += 1
                    profit = stake * (odds - 1)
                else:
                    profit = -stake
                total_profit += profit

        if bets > 0:
            win_rate = wins / bets * 100
            roi = total_profit / (bets * stake) * 100
            print(f"{scenario_name:<30}: {bets} bets, {wins} wins ({win_rate:.1f}%), Profit: ${total_profit:+.2f} (ROI: {roi:+.1f}%)")

    # Key insight test: what if we only bet HOME wins at high confidence?
    print("\n" + "=" * 90)
    print("HOME WINS ONLY AT VARIOUS THRESHOLDS")
    print("(Home favorites are more predictable)")
    print("=" * 90)

    for threshold in [0.60, 0.65, 0.70, 0.75]:
        bets = 0
        wins = 0

        for match, analysis in results:
            actual = "home" if match.home_score > match.away_score else ("away" if match.home_score < match.away_score else "draw")
            home_prob = float(analysis.consensus_home_prob)

            if home_prob >= threshold:
                bets += 1
                if actual == "home":
                    wins += 1

        if bets > 0:
            win_rate = wins / bets * 100
            min_odds = 1 / (wins / bets)  # Breakeven odds
            print(f"Home win >= {threshold:.0%}: {bets} bets, {wins} wins ({win_rate:.1f}%), need odds >= {min_odds:.2f} to profit")

    # Test away wins
    print("\n" + "=" * 90)
    print("AWAY WINS ONLY AT VARIOUS THRESHOLDS")
    print("=" * 90)

    for threshold in [0.50, 0.55, 0.60]:
        bets = 0
        wins = 0

        for match, analysis in results:
            actual = "home" if match.home_score > match.away_score else ("away" if match.home_score < match.away_score else "draw")
            away_prob = float(analysis.consensus_away_prob)

            if away_prob >= threshold:
                bets += 1
                if actual == "away":
                    wins += 1

        if bets > 0:
            win_rate = wins / bets * 100
            min_odds = 1 / (wins / bets)
            print(f"Away win >= {threshold:.0%}: {bets} bets, {wins} wins ({win_rate:.1f}%), need odds >= {min_odds:.2f} to profit")

    # Final recommendation
    print("\n" + "=" * 90)
    print("PROFITABLE STRATEGY SIMULATION")
    print("=" * 90)
    print("\nAssumptions:")
    print("- Bet home wins when model >= 70% confident")
    print("- Actual win rate at this level: ~59%")
    print("- Shop for odds >= 1.70 (implied 58.8%)")
    print("- This gives ~0.2% edge per bet\n")

    # Simulate with realistic shopping
    stake = 10
    bankroll = 1000
    bets_placed = 0
    total_wins = 0

    for match, analysis in results:
        actual = "home" if match.home_score > match.away_score else ("away" if match.home_score < match.away_score else "draw")
        home_prob = float(analysis.consensus_home_prob)

        # Only bet strong home favorites
        if home_prob >= 0.70:
            # Assume we can find odds of 1.70 by shopping (realistic for heavy favorites)
            odds = 1.70
            bets_placed += 1

            if actual == "home":
                total_wins += 1
                bankroll += stake * (odds - 1)
            else:
                bankroll -= stake

    if bets_placed > 0:
        win_rate = total_wins / bets_placed * 100
        total_return = (bankroll - 1000) / 1000 * 100
        print(f"Results: {bets_placed} bets, {total_wins} wins ({win_rate:.1f}%)")
        print(f"Final bankroll: ${bankroll:.2f} ({total_return:+.1f}%)")

        # What odds would we have needed?
        breakeven_odds = bets_placed / total_wins if total_wins > 0 else 999
        print(f"Breakeven odds: {breakeven_odds:.2f}")


if __name__ == "__main__":
    import logging
    logging.disable(logging.CRITICAL)
    run_calibrated_backtest()
