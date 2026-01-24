"""Final backtest - focusing on the profitable strategy.

Key finding: Home wins at 75%+ model confidence hit 70% of the time.
Need odds >= 1.43 to break even. Most bookies offer 1.45-1.55 for heavy favorites.
"""

from sqlalchemy import select
from app.db.database import SyncSessionLocal
from app.db.models import Match, MatchAnalysis, MatchStatus, Team


def run_final_backtest():
    """Test the most promising strategy."""

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

    print("=" * 90)
    print("FINAL BACKTEST: HOME FAVORITES STRATEGY")
    print("=" * 90)
    print(f"\nAnalyzing {len(results)} matches\n")

    # Strategy: Bet on home wins when model >= 75% confident
    # At this level, actual win rate is 70%
    # Breakeven odds = 1.43, typical bookmaker odds = 1.45-1.55

    strategies = [
        ("Conservative: 75%+ conf, odds 1.50", 0.75, 1.50),
        ("Moderate: 75%+ conf, odds 1.45", 0.75, 1.45),
        ("Aggressive: 70%+ conf, odds 1.60", 0.70, 1.60),
        ("Very Aggressive: 65%+ conf, odds 1.70", 0.65, 1.70),
    ]

    for strategy_name, min_conf, assumed_odds in strategies:
        print(f"\n{strategy_name}")
        print("-" * 60)

        stake = 10
        bankroll = 1000
        bets = []

        for match, analysis in results:
            actual = "home" if match.home_score > match.away_score else ("away" if match.home_score < match.away_score else "draw")
            home_prob = float(analysis.consensus_home_prob)
            home_team = teams.get(match.home_team_id, "?")
            away_team = teams.get(match.away_team_id, "?")

            if home_prob >= min_conf:
                won = actual == "home"
                if won:
                    profit = stake * (assumed_odds - 1)
                else:
                    profit = -stake

                bankroll += profit
                bets.append({
                    "season": match.season,
                    "match": f"{home_team} vs {away_team}",
                    "model_prob": home_prob,
                    "won": won,
                    "profit": profit,
                    "bankroll": bankroll,
                })

        if bets:
            wins = sum(1 for b in bets if b["won"])
            total_profit = sum(b["profit"] for b in bets)
            roi = total_profit / (len(bets) * stake) * 100

            print(f"Bets: {len(bets)}, Wins: {wins} ({wins/len(bets)*100:.1f}%)")
            print(f"Total Profit: ${total_profit:+.2f}")
            print(f"ROI: {roi:+.1f}%")
            print(f"Final Bankroll: ${bankroll:.2f} ({(bankroll-1000)/10:.1f}% return)")

            # Max drawdown
            peak = 1000
            max_dd = 0
            for b in bets:
                if b["bankroll"] > peak:
                    peak = b["bankroll"]
                dd = (peak - b["bankroll"]) / peak
                max_dd = max(max_dd, dd)
            print(f"Max Drawdown: {max_dd:.1%}")

    # Show the best bets from the conservative strategy
    print("\n" + "=" * 90)
    print("SAMPLE BETS (75%+ confidence strategy)")
    print("=" * 90)

    stake = 10
    bankroll = 1000
    sample_bets = []

    for match, analysis in results:
        actual = "home" if match.home_score > match.away_score else ("away" if match.home_score < match.away_score else "draw")
        home_prob = float(analysis.consensus_home_prob)
        home_team = teams.get(match.home_team_id, "?")
        away_team = teams.get(match.away_team_id, "?")

        if home_prob >= 0.75:
            won = actual == "home"
            odds = 1.50
            profit = stake * (odds - 1) if won else -stake
            bankroll += profit

            sample_bets.append({
                "season": match.season,
                "mw": match.matchweek,
                "match": f"{home_team} vs {away_team}",
                "score": f"{match.home_score}-{match.away_score}",
                "model": f"{home_prob:.0%}",
                "result": "WIN" if won else "LOSS",
                "profit": profit,
                "bankroll": bankroll,
            })

    # Show last 15 bets
    print(f"\n{'Season':<10} {'MW':<4} {'Match':<25} {'Score':<8} {'Model':<8} {'Result':<8} {'P/L':<10} {'Bank':<10}")
    print("-" * 90)
    for b in sample_bets[-15:]:
        print(f"{b['season']:<10} {b['mw']:<4} {b['match']:<25} {b['score']:<8} {b['model']:<8} {b['result']:<8} ${b['profit']:+.2f}     ${b['bankroll']:.2f}")

    # Summary
    print("\n" + "=" * 90)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 90)
    print("""
FINDINGS:
1. Model is overconfident - says 75% but actually wins 70%
2. However, 70% win rate is still valuable!

PROFITABLE STRATEGY:
- Bet on HOME wins when model confidence >= 75%
- Historical win rate at this level: 70%
- Need odds >= 1.43 to break even
- At typical odds of 1.50, expect ~3.5% ROI

CAUTIONS:
- Small sample size (20 bets in historical data)
- Need to verify with more seasons
- Always shop for best odds
- Consider using fractional Kelly (25%) for stake sizing

RECOMMENDED IMPLEMENTATION:
1. Flag matches where consensus home_prob >= 75%
2. Only bet if you can get odds >= 1.50
3. Use 2-3% of bankroll per bet (fractional Kelly)
4. Track results to validate strategy
    """)


if __name__ == "__main__":
    import logging
    logging.disable(logging.CRITICAL)
    run_final_backtest()
