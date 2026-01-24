"""Simple backtest - flat stakes at fair odds.

This tests if our model's predictions are accurate enough to be profitable
if we could bet at fair odds (1/probability).
"""

from sqlalchemy import select
from app.db.database import SyncSessionLocal
from app.db.models import Match, MatchAnalysis, MatchStatus, Team


def run_backtest():
    """Run simple flat-stake backtest."""

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

    print(f"Analyzing {len(results)} matches with predictions")
    print("=" * 90)

    # Test different confidence thresholds
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]

    print(f"\n{'Threshold':<12} {'Bets':>8} {'Wins':>8} {'Win%':>8} {'Exp Win%':>10} {'Edge':>8} {'Profit':>12} {'ROI':>10}")
    print("-" * 90)

    for threshold in thresholds:
        bets = 0
        wins = 0
        total_profit = 0
        total_expected_wins = 0
        stake = 10  # Flat $10 stake

        for match, analysis in results:
            # Determine actual result
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

            # Find highest confidence prediction
            best_outcome = max(probs, key=probs.get)
            best_prob = probs[best_outcome]

            if best_prob >= threshold:
                bets += 1
                total_expected_wins += best_prob

                # Fair odds = 1/probability
                fair_odds = 1 / best_prob

                if best_outcome == actual:
                    wins += 1
                    profit = stake * (fair_odds - 1)
                else:
                    profit = -stake

                total_profit += profit

        if bets > 0:
            win_rate = wins / bets * 100
            expected_win_rate = total_expected_wins / bets * 100
            edge = win_rate - expected_win_rate
            roi = total_profit / (bets * stake) * 100
            print(f"{threshold:.0%}+          {bets:>8} {wins:>8} {win_rate:>7.1f}% {expected_win_rate:>9.1f}% {edge:>+7.1f}% ${total_profit:>+10.2f} {roi:>+9.1f}%")

    # Now test with bookmaker margin
    print("\n" + "=" * 90)
    print("WITH 5% BOOKMAKER MARGIN (more realistic)")
    print("=" * 90)
    print(f"\n{'Threshold':<12} {'Bets':>8} {'Wins':>8} {'Win%':>8} {'Profit':>12} {'ROI':>10}")
    print("-" * 90)

    margin = 0.05  # 5% bookmaker edge

    for threshold in thresholds:
        bets = 0
        wins = 0
        total_profit = 0
        stake = 10

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
                bets += 1

                # Bookmaker odds with margin
                implied_prob_with_margin = best_prob + margin
                if implied_prob_with_margin >= 1:
                    continue
                bookie_odds = 1 / implied_prob_with_margin

                if best_outcome == actual:
                    wins += 1
                    profit = stake * (bookie_odds - 1)
                else:
                    profit = -stake

                total_profit += profit

        if bets > 0:
            win_rate = wins / bets * 100
            roi = total_profit / (bets * stake) * 100
            print(f"{threshold:.0%}+          {bets:>8} {wins:>8} {win_rate:>7.1f}% ${total_profit:>+10.2f} {roi:>+9.1f}%")

    # Test betting AGAINST low confidence predictions (contrarian)
    print("\n" + "=" * 90)
    print("DETAILED BREAKDOWN AT 60% THRESHOLD")
    print("=" * 90)

    by_outcome = {"home": {"bets": 0, "wins": 0, "profit": 0},
                  "draw": {"bets": 0, "wins": 0, "profit": 0},
                  "away": {"bets": 0, "wins": 0, "profit": 0}}

    by_season = {}
    stake = 10

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

        if best_prob >= 0.60:
            fair_odds = 1 / best_prob
            won = best_outcome == actual
            profit = stake * (fair_odds - 1) if won else -stake

            by_outcome[best_outcome]["bets"] += 1
            if won:
                by_outcome[best_outcome]["wins"] += 1
            by_outcome[best_outcome]["profit"] += profit

            if match.season not in by_season:
                by_season[match.season] = {"bets": 0, "wins": 0, "profit": 0}
            by_season[match.season]["bets"] += 1
            if won:
                by_season[match.season]["wins"] += 1
            by_season[match.season]["profit"] += profit

    print(f"\n{'Outcome':<12} {'Bets':>8} {'Wins':>8} {'Win%':>8} {'Profit':>12}")
    print("-" * 50)
    for outcome in ["home", "draw", "away"]:
        stats = by_outcome[outcome]
        if stats["bets"] > 0:
            wr = stats["wins"] / stats["bets"] * 100
            print(f"{outcome.upper():<12} {stats['bets']:>8} {stats['wins']:>8} {wr:>7.1f}% ${stats['profit']:>+10.2f}")

    print(f"\n{'Season':<12} {'Bets':>8} {'Wins':>8} {'Win%':>8} {'Profit':>12}")
    print("-" * 50)
    for season in sorted(by_season.keys()):
        stats = by_season[season]
        if stats["bets"] > 0:
            wr = stats["wins"] / stats["bets"] * 100
            print(f"{season:<12} {stats['bets']:>8} {stats['wins']:>8} {wr:>7.1f}% ${stats['profit']:>+10.2f}")

    # Value where model disagrees with implied
    print("\n" + "=" * 90)
    print("TESTING VALUE: Where model prob > 60% AND at least 10% higher than draw")
    print("=" * 90)

    for threshold in [0.60, 0.65, 0.70]:
        bets = 0
        wins = 0
        total_profit = 0
        stake = 10

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

            # Only bet if there's a clear favorite (not close to draw)
            best_outcome = max(probs, key=probs.get)
            best_prob = probs[best_outcome]
            second_prob = sorted(probs.values())[-2]

            if best_prob >= threshold and (best_prob - second_prob) >= 0.10:
                bets += 1
                fair_odds = 1 / best_prob

                if best_outcome == actual:
                    wins += 1
                    profit = stake * (fair_odds - 1)
                else:
                    profit = -stake

                total_profit += profit

        if bets > 0:
            win_rate = wins / bets * 100
            roi = total_profit / (bets * stake) * 100
            print(f"{threshold:.0%}+ (clear favorite): {bets:>5} bets, {wins:>5} wins ({win_rate:.1f}%), Profit: ${total_profit:+.2f} (ROI: {roi:+.1f}%)")


if __name__ == "__main__":
    import logging
    logging.disable(logging.CRITICAL)
    run_backtest()
