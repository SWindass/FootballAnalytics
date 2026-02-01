"""Era detection and regime monitoring for betting strategies.

STATIC 12% + ERA-BASED FORM OVERRIDE STRATEGY

The strategy uses era-based detection rather than regime detection because:
1. Regime detection has a bootstrap problem - can't build ROI history without betting
2. Era-based approach is simpler and has been validated to work historically

Era Rules:
- Pre-2020s: Only away bets with 12%+ edge
- 2020s Era (2020-21+): Add home form override (form 12+ with negative edge)

Stop-Loss Triggers:
- Hard stop if 12-month ROI < -20%
- Hard stop if away success rate < 30% over 50+ bets
"""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Optional

from sqlalchemy import select, func, desc
from sqlalchemy.orm import Session

from app.db.models import Match, MatchStatus


# Era boundary
ERA_2020S_START = "2020-21"

# Stop-loss thresholds
STOP_LOSS_ROI_THRESHOLD = -0.20  # -20%
STOP_LOSS_WIN_RATE_THRESHOLD = 0.30  # 30%
STOP_LOSS_MIN_BETS = 50


@dataclass
class EraStatus:
    """Current era status for betting strategy."""
    current_season: str
    is_2020s_era: bool
    away_strategy_active: bool  # Always true
    home_form_override_active: bool  # Only in 2020s
    stop_loss_triggered: bool
    stop_loss_reason: Optional[str]

    # Performance metrics
    total_bets: int
    total_wins: int
    win_rate: float
    total_profit: float
    roi: float

    # Away strategy metrics
    away_bets: int
    away_wins: int
    away_win_rate: float
    away_roi: float

    # Home form override metrics (2020s only)
    home_bets: int
    home_wins: int
    home_win_rate: float
    home_roi: float

    # Rolling metrics
    rolling_12m_roi: Optional[float]
    rolling_50_win_rate: Optional[float]


def is_2020s_era(season: str) -> bool:
    """Check if a season is in the 2020s era."""
    return season >= ERA_2020S_START


def get_current_season() -> str:
    """Get the current season string based on today's date."""
    today = datetime.now()
    year = today.year
    month = today.month

    # Season starts in August
    if month >= 8:
        return f"{year}-{str(year + 1)[2:]}"
    else:
        return f"{year - 1}-{str(year)[2:]}"


def calculate_era_status(session: Session) -> EraStatus:
    """Calculate the current era status and performance metrics.

    Uses the backtest approach to calculate bets from Match/MatchAnalysis/OddsHistory
    rather than relying on the ValueBet table which may not be populated.
    """
    from collections import defaultdict
    from app.db.models import MatchAnalysis, OddsHistory, TeamStats

    current_season = get_current_season()

    # Strategy parameters (matching strategy_backtest.py)
    # 5% edge optimal for 2020s: +30.2% ROI, 23 bets/yr vs 12%: +23% ROI, 6 bets/yr
    AWAY_MIN_EDGE = 0.05
    AWAY_MIN_ODDS = 1.50
    AWAY_MAX_ODDS = 8.00
    AWAY_EXCLUDE_HOME_FORM = (4, 5, 6)
    HOME_MIN_FORM = 12
    HOME_MAX_EDGE = 0.0
    HOME_MIN_ODDS = 1.01
    HOME_MAX_ODDS = 10.00

    # Load all matches with predictions
    stmt = (
        select(Match, MatchAnalysis)
        .join(MatchAnalysis, Match.id == MatchAnalysis.match_id)
        .where(Match.status == MatchStatus.FINISHED)
        .where(MatchAnalysis.consensus_home_prob.isnot(None))
        .order_by(Match.kickoff_time)
    )
    matches = list(session.execute(stmt).all())

    if not matches:
        return EraStatus(
            current_season=current_season,
            is_2020s_era=is_2020s_era(current_season),
            away_strategy_active=True,
            home_form_override_active=is_2020s_era(current_season),
            stop_loss_triggered=False,
            stop_loss_reason=None,
            total_bets=0, total_wins=0, win_rate=0, total_profit=0, roi=0,
            away_bets=0, away_wins=0, away_win_rate=0, away_roi=0,
            home_bets=0, home_wins=0, home_win_rate=0, home_roi=0,
            rolling_12m_roi=None, rolling_50_win_rate=None,
        )

    # Load odds
    match_ids = [m.id for m, _ in matches]
    all_odds = list(session.execute(
        select(OddsHistory).where(OddsHistory.match_id.in_(match_ids))
    ).scalars().all())
    odds_lookup = defaultdict(list)
    for oh in all_odds:
        odds_lookup[oh.match_id].append(oh)

    # Load team stats
    seasons = list(set(m.season for m, _ in matches))
    all_stats = list(session.execute(
        select(TeamStats).where(TeamStats.season.in_(seasons))
    ).scalars().all())
    stats_lookup = {(ts.team_id, ts.season, ts.matchweek): ts for ts in all_stats}

    # Calculate bets using strategy rules
    class BetResult:
        def __init__(self, kickoff, outcome, won, profit):
            self.kickoff = kickoff
            self.outcome = outcome
            self.won = won
            self.profit = profit

    results = []

    for match, analysis in matches:
        match_odds = odds_lookup.get(match.id, [])
        best_away_odds = max((float(oh.away_odds) for oh in match_odds if oh.away_odds), default=None)
        best_home_odds = max((float(oh.home_odds) for oh in match_odds if oh.home_odds), default=None)

        # Fall back to historical odds
        if analysis.features:
            hist_odds = analysis.features.get("historical_odds", {})
            if not best_away_odds and hist_odds.get("avg_away_odds"):
                best_away_odds = hist_odds["avg_away_odds"]
            if not best_home_odds and hist_odds.get("avg_home_odds"):
                best_home_odds = hist_odds["avg_home_odds"]

        # Get form data
        home_stats = stats_lookup.get((match.home_team_id, match.season, match.matchweek - 1))
        home_form = home_stats.form_points if home_stats else 0

        # Determine result
        if match.home_score > match.away_score:
            actual = "H"
        elif match.away_score > match.home_score:
            actual = "A"
        else:
            actual = "D"

        # Check AWAY BET
        if best_away_odds and AWAY_MIN_ODDS <= best_away_odds <= AWAY_MAX_ODDS:
            away_prob = float(analysis.consensus_away_prob or 0)
            if away_prob > 0:
                market_prob = 1 / best_away_odds
                edge = away_prob - market_prob
                if edge >= AWAY_MIN_EDGE and home_form not in AWAY_EXCLUDE_HOME_FORM:
                    won = actual == "A"
                    profit = (best_away_odds - 1) if won else -1
                    results.append(BetResult(match.kickoff_time, "away_win", won, profit))

        # Check HOME BET (2020s only)
        if is_2020s_era(match.season) and best_home_odds and HOME_MIN_ODDS <= best_home_odds <= HOME_MAX_ODDS:
            home_prob = float(analysis.consensus_home_prob or 0)
            if home_prob > 0:
                market_prob = 1 / best_home_odds
                edge = home_prob - market_prob
                if home_form >= HOME_MIN_FORM and edge < HOME_MAX_EDGE:
                    won = actual == "H"
                    profit = (best_home_odds - 1) if won else -1
                    results.append(BetResult(match.kickoff_time, "home_win", won, profit))

    # Return empty if no bets after processing
    if not results:
        return EraStatus(
            current_season=current_season,
            is_2020s_era=is_2020s_era(current_season),
            away_strategy_active=True,
            home_form_override_active=is_2020s_era(current_season),
            stop_loss_triggered=False,
            stop_loss_reason=None,
            total_bets=0, total_wins=0, win_rate=0, total_profit=0, roi=0,
            away_bets=0, away_wins=0, away_win_rate=0, away_roi=0,
            home_bets=0, home_wins=0, home_win_rate=0, home_roi=0,
            rolling_12m_roi=None, rolling_50_win_rate=None,
        )

    # Calculate overall metrics
    total_bets = len(results)
    total_wins = sum(1 for r in results if r.won)
    total_profit = sum(r.profit for r in results)
    win_rate = total_wins / total_bets if total_bets > 0 else 0
    roi = total_profit / total_bets if total_bets > 0 else 0

    # Split by outcome type
    away_results = [r for r in results if r.outcome == "away_win"]
    home_results = [r for r in results if r.outcome == "home_win"]

    away_bets = len(away_results)
    away_wins = sum(1 for r in away_results if r.won)
    away_profit = sum(r.profit for r in away_results)
    away_win_rate = away_wins / away_bets if away_bets > 0 else 0
    away_roi = away_profit / away_bets if away_bets > 0 else 0

    home_bets = len(home_results)
    home_wins = sum(1 for r in home_results if r.won)
    home_profit = sum(r.profit for r in home_results)
    home_win_rate = home_wins / home_bets if home_bets > 0 else 0
    home_roi = home_profit / home_bets if home_bets > 0 else 0

    # Calculate rolling metrics
    rolling_12m_roi = None
    rolling_50_win_rate = None

    # Last 12 months
    cutoff_date = datetime.now().replace(year=datetime.now().year - 1)
    recent_bets = [r for r in results if r.kickoff >= cutoff_date]
    if recent_bets:
        recent_profit = sum(r.profit for r in recent_bets)
        rolling_12m_roi = recent_profit / len(recent_bets)

    # Last 50 bets
    if len(results) >= 50:
        last_50 = results[-50:]
        last_50_wins = sum(1 for r in last_50 if r.won)
        rolling_50_win_rate = last_50_wins / 50

    # Check stop-loss conditions
    stop_loss_triggered = False
    stop_loss_reason = None

    if rolling_12m_roi is not None and rolling_12m_roi < STOP_LOSS_ROI_THRESHOLD:
        stop_loss_triggered = True
        stop_loss_reason = f"12-month ROI ({rolling_12m_roi:.1%}) below threshold ({STOP_LOSS_ROI_THRESHOLD:.0%})"

    if len(away_results) >= STOP_LOSS_MIN_BETS:
        last_50_away = away_results[-50:]
        last_50_away_wins = sum(1 for r in last_50_away if r.won)
        away_50_win_rate = last_50_away_wins / len(last_50_away)
        if away_50_win_rate < STOP_LOSS_WIN_RATE_THRESHOLD:
            stop_loss_triggered = True
            stop_loss_reason = f"Away win rate ({away_50_win_rate:.1%}) below threshold ({STOP_LOSS_WIN_RATE_THRESHOLD:.0%})"

    return EraStatus(
        current_season=current_season,
        is_2020s_era=is_2020s_era(current_season),
        away_strategy_active=not stop_loss_triggered,
        home_form_override_active=is_2020s_era(current_season) and not stop_loss_triggered,
        stop_loss_triggered=stop_loss_triggered,
        stop_loss_reason=stop_loss_reason,
        total_bets=total_bets,
        total_wins=total_wins,
        win_rate=win_rate,
        total_profit=total_profit,
        roi=roi,
        away_bets=away_bets,
        away_wins=away_wins,
        away_win_rate=away_win_rate,
        away_roi=away_roi,
        home_bets=home_bets,
        home_wins=home_wins,
        home_win_rate=home_win_rate,
        home_roi=home_roi,
        rolling_12m_roi=rolling_12m_roi,
        rolling_50_win_rate=rolling_50_win_rate,
    )


def run_5_year_simulation(
    session: Session,
    scenario: str = "regime_continues",
) -> dict:
    """
    Run a 5-year forward simulation with different scenarios.

    Scenarios:
    - regime_continues: 2020s performance continues
    - regime_ends_year_2: Returns to pre-2020 performance after 2 years
    - regime_ends_year_4: Returns to pre-2020 performance after 4 years

    Uses backtest results (from strategy_backtest.py) to project future results.
    """
    from batch.jobs.strategy_backtest import StrategyBacktest

    # Run backtest to get historical performance
    backtest = StrategyBacktest(session)
    backtest_result = backtest.run()

    # Split results by era
    era_2020s_start = "2020-21"

    pre_2020s_bets = [b for b in backtest_result.bets if b.season < era_2020s_start]
    era_2020s_bets = [b for b in backtest_result.bets if b.season >= era_2020s_start]

    # Pre-2020s: only away bets (home form override wasn't active)
    pre_2020s_away = [b for b in pre_2020s_bets if b.outcome == "away_win"]
    pre_2020s_seasons = set(b.season for b in pre_2020s_away)
    n_pre_2020s_seasons = len(pre_2020s_seasons) or 1
    pre_2020s_away_roi = sum(b.profit for b in pre_2020s_away) / len(pre_2020s_away) if pre_2020s_away else 0
    pre_2020s_bets_per_year = len(pre_2020s_away) / n_pre_2020s_seasons

    # 2020s: both away and home
    era_2020s_away = [b for b in era_2020s_bets if b.outcome == "away_win"]
    era_2020s_home = [b for b in era_2020s_bets if b.outcome == "home_win"]
    era_2020s_seasons = set(b.season for b in era_2020s_bets)
    n_era_2020s_seasons = len(era_2020s_seasons) or 1

    era_2020s_away_roi = sum(b.profit for b in era_2020s_away) / len(era_2020s_away) if era_2020s_away else 0
    era_2020s_away_bets_per_year = len(era_2020s_away) / n_era_2020s_seasons

    era_2020s_home_roi = sum(b.profit for b in era_2020s_home) / len(era_2020s_home) if era_2020s_home else 0
    era_2020s_home_bets_per_year = len(era_2020s_home) / n_era_2020s_seasons

    # Run simulation
    years = 5
    results = {
        "scenario": scenario,
        "years": [],
        "total_bets": 0,
        "total_profit": 0,
        "total_roi": 0,
        "assumptions": {
            "pre_2020s_away_roi": pre_2020s_away_roi,
            "pre_2020s_bets_per_year": pre_2020s_bets_per_year,
            "era_2020s_away_roi": era_2020s_away_roi,
            "era_2020s_away_bets_per_year": era_2020s_away_bets_per_year,
            "era_2020s_home_roi": era_2020s_home_roi,
            "era_2020s_home_bets_per_year": era_2020s_home_bets_per_year,
        },
    }

    regime_ends_year = {
        "regime_continues": 999,  # Never ends
        "regime_ends_year_2": 2,
        "regime_ends_year_4": 4,
    }.get(scenario, 999)

    cumulative_bets = 0
    cumulative_profit = 0

    for year in range(1, years + 1):
        is_2020s_regime = year <= regime_ends_year

        if is_2020s_regime:
            # 2020s performance continues
            away_bets = era_2020s_away_bets_per_year
            away_roi = era_2020s_away_roi
            home_bets = era_2020s_home_bets_per_year
            home_roi = era_2020s_home_roi
        else:
            # Reverts to pre-2020s performance (away only)
            away_bets = pre_2020s_bets_per_year
            away_roi = pre_2020s_away_roi
            home_bets = 0
            home_roi = 0

        year_bets = away_bets + home_bets
        year_profit = (away_bets * away_roi) + (home_bets * home_roi)
        year_roi = year_profit / year_bets if year_bets > 0 else 0

        cumulative_bets += year_bets
        cumulative_profit += year_profit

        results["years"].append({
            "year": year,
            "is_2020s_regime": is_2020s_regime,
            "away_bets": round(away_bets, 1),
            "home_bets": round(home_bets, 1),
            "total_bets": round(year_bets, 1),
            "profit": round(year_profit, 1),
            "roi": year_roi,
            "cumulative_bets": round(cumulative_bets, 1),
            "cumulative_profit": round(cumulative_profit, 1),
            "cumulative_roi": cumulative_profit / cumulative_bets if cumulative_bets > 0 else 0,
        })

    results["total_bets"] = round(cumulative_bets, 1)
    results["total_profit"] = round(cumulative_profit, 1)
    results["total_roi"] = cumulative_profit / cumulative_bets if cumulative_bets > 0 else 0

    return results
