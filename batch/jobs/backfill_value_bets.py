"""Backfill value bets for historical matches.

This job generates value bets for historical matches that have:
- Model predictions (consensus probabilities)
- Historical odds data

This allows viewing value bet performance across all historical seasons.
"""

import argparse
from datetime import datetime
from decimal import Decimal

import structlog
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.db.database import SyncSessionLocal
from app.db.models import (
    BetOutcome,
    Match,
    MatchAnalysis,
    MatchStatus,
    OddsHistory,
    TeamStats,
    ValueBet,
)

logger = structlog.get_logger()


class ValueBetBackfill:
    """Backfills value bets for historical matches."""

    def __init__(self, session: Session | None = None):
        self.session = session or SyncSessionLocal()

        # STATIC 5% + ERA-BASED FORM OVERRIDE STRATEGY
        #
        # Why 5% instead of 12%?
        # -----------------------
        # Analysis of 2020s era data showed lower thresholds are MORE profitable:
        #
        #   | Edge | Bets/Yr | Win%  | ROI    | Profit |
        #   |------|---------|-------|--------|--------|
        #   | 5%   | 23.4    | 55.6% | +30.2% | +35.4  |
        #   | 12%  | 6.4     | 59.4% | +23.0% | +7.4   |
        #
        # The 5% threshold has HIGHER ROI and 5x more profit because:
        # 1. Model is well-calibrated in 2020s - even small edges are real
        # 2. Volume compounds - more bets × decent ROI = more profit
        # 3. 55.6% win rate at 5% edge proves the model finds true value
        #
        # Historical (pre-2020s) required 12% because model was less calibrated.
        # Modern era benefits from lower threshold + higher volume.
        #
        # Strategy 1: Away wins with 5%+ edge (always active)
        #             2020s: 117 bets, 55.6% win, +30.2% ROI
        # Strategy 2: Home wins with form 12+ and negative edge (2020s only)
        #             94 bets, 67.0% win, +30.4% ROI
        # Strategy 3: Over 2.5 goals with 10-12% edge (2020s only)
        #             97 bets, 62.9% win, +6.6% ROI
        self.era_2020s_start = "2020-21"
        self.strategies = {
            "away_win": {
                "min_edge": 0.05,  # 5% minimum edge - optimal for 2020s era
                "min_odds": 1.50,
                "max_odds": 8.00,
                # Exclude when home team form is 4-6 (poor but not terrible)
                "exclude_home_form_min": 4,
                "exclude_home_form_max": 6,
            },
            "home_win": {
                # Home wins require negative edge (market sees more value than model)
                # AND home team on hot streak (form 12+)
                # ONLY active in 2020s era
                "max_edge": 0.0,  # Negative edge required (edge < 0)
                "min_form": 12,   # 12+ form points from last 5 games
                "min_odds": 1.01,
                "max_odds": 10.00,
                "era_only": True,  # Only active from 2020-21 season
            },
            "over_2_5": {
                # Over 2.5 goals with tight 10-12% edge (Poisson model)
                # Backtest: 97 bets, 62.9% win rate, +6.6% ROI
                # ONLY active in 2020s era
                "min_edge": 0.10,  # 10% minimum edge
                "max_edge": 0.12,  # 12% maximum edge (higher is overconfident)
                "min_odds": 1.50,
                "max_odds": 3.00,
                "era_only": True,  # Only active from 2020-21 season
            },
        }

    def run(
        self,
        seasons: list[str] = None,
        start_season: str = "2000-01",
        force: bool = False,
    ) -> dict:
        """Run the backfill job.

        Args:
            seasons: List of specific seasons to process
            start_season: Earliest season to process (default: 2000-01)
            force: If True, delete existing value bets and regenerate

        Returns:
            Summary of backfill results
        """
        logger.info("Starting value bet backfill")
        start_time = datetime.utcnow()

        # Get seasons to process
        if seasons is None:
            seasons = self._get_seasons_with_odds(start_season)

        total_created = 0
        total_matches = 0

        for season in sorted(seasons):
            created, matches = self._process_season(season, force)
            total_created += created
            total_matches += matches
            logger.info(f"Season {season}: {created} value bets from {matches} matches")

        self.session.commit()

        duration = (datetime.utcnow() - start_time).total_seconds()
        logger.info(
            "Value bet backfill completed",
            seasons_processed=len(seasons),
            value_bets_created=total_created,
            matches_processed=total_matches,
            duration_seconds=round(duration, 1),
        )

        return {
            "status": "success",
            "seasons_processed": len(seasons),
            "value_bets_created": total_created,
            "matches_processed": total_matches,
            "duration_seconds": duration,
        }

    def _get_seasons_with_odds(self, start_season: str) -> list[str]:
        """Get all seasons with historical odds data."""
        stmt = (
            select(Match.season)
            .distinct()
            .where(Match.season >= start_season)
            .order_by(Match.season)
        )
        return [s for (s,) in self.session.execute(stmt).all()]

    def _process_season(self, season: str, force: bool) -> tuple[int, int]:
        """Process a single season.

        Returns:
            Tuple of (value_bets_created, matches_processed)
        """
        # Load all matches with analysis for this season
        stmt = (
            select(Match, MatchAnalysis)
            .join(MatchAnalysis, Match.id == MatchAnalysis.match_id)
            .where(Match.season == season)
            .where(Match.status == MatchStatus.FINISHED)
            .where(MatchAnalysis.consensus_home_prob.isnot(None))
            .order_by(Match.kickoff_time)
        )
        matches = list(self.session.execute(stmt).all())

        if not matches:
            return 0, 0

        # If force, delete existing value bets for this season
        # Only delete the outcomes we're regenerating
        if force:
            match_ids = [m.id for m, _ in matches]
            self.session.execute(
                delete(ValueBet)
                .where(ValueBet.match_id.in_(match_ids))
                .where(ValueBet.outcome.in_(["away_win", "home_win", "over_2_5"]))
            )
            self.session.flush()

        # Bulk load supporting data
        match_ids = [m.id for m, _ in matches]

        # Load odds history
        all_odds = list(self.session.execute(
            select(OddsHistory)
            .where(OddsHistory.match_id.in_(match_ids))
        ).scalars().all())
        odds_lookup = {}
        for oh in all_odds:
            if oh.match_id not in odds_lookup:
                odds_lookup[oh.match_id] = []
            odds_lookup[oh.match_id].append(oh)

        # Load team stats for form data (needed for home win strategy)
        all_stats = list(self.session.execute(
            select(TeamStats)
            .where(TeamStats.season == season)
        ).scalars().all())
        stats_lookup = {(ts.team_id, ts.matchweek): ts for ts in all_stats}

        # Check existing value bets (to avoid duplicates if not forcing)
        existing_vb = set()
        if not force:
            existing = self.session.execute(
                select(ValueBet.match_id, ValueBet.outcome)
                .where(ValueBet.match_id.in_(match_ids))
            ).all()
            existing_vb = {(mid, outcome) for mid, outcome in existing}

        created = 0
        for match, analysis in matches:
            odds_history = odds_lookup.get(match.id, [])

            # Get odds - prefer OddsHistory table, fall back to features
            best_odds = {}

            for oh in odds_history:
                if oh.away_odds:
                    if "away_win" not in best_odds or float(oh.away_odds) > best_odds["away_win"]["odds"]:
                        best_odds["away_win"] = {"odds": float(oh.away_odds), "bookmaker": oh.bookmaker}
                if oh.home_odds:
                    if "home_win" not in best_odds or float(oh.home_odds) > best_odds["home_win"]["odds"]:
                        best_odds["home_win"] = {"odds": float(oh.home_odds), "bookmaker": oh.bookmaker}
                if oh.over_2_5_odds:
                    if "over_2_5" not in best_odds or float(oh.over_2_5_odds) > best_odds["over_2_5"]["odds"]:
                        best_odds["over_2_5"] = {"odds": float(oh.over_2_5_odds), "bookmaker": oh.bookmaker}

            # Fall back to historical_odds from features
            if analysis.features:
                hist_odds = analysis.features.get("historical_odds", {})
                if "away_win" not in best_odds and hist_odds.get("avg_away_odds"):
                    best_odds["away_win"] = {"odds": hist_odds["avg_away_odds"], "bookmaker": "historical"}
                if "home_win" not in best_odds and hist_odds.get("avg_home_odds"):
                    best_odds["home_win"] = {"odds": hist_odds["avg_home_odds"], "bookmaker": "historical"}
                if "over_2_5" not in best_odds and hist_odds.get("avg_over_2_5_odds"):
                    best_odds["over_2_5"] = {"odds": hist_odds["avg_over_2_5_odds"], "bookmaker": "historical"}

            if not best_odds:
                continue

            # Get home team form for home_win strategy
            home_stats = stats_lookup.get((match.home_team_id, match.matchweek - 1))
            home_form = home_stats.form_points if home_stats else 0

            # Check each strategy
            for outcome, strategy in self.strategies.items():
                if outcome not in best_odds:
                    continue

                odds_data = best_odds[outcome]
                decimal_odds = odds_data["odds"]
                bookmaker = odds_data["bookmaker"]

                # Get model probability
                if outcome == "away_win":
                    model_prob = float(analysis.consensus_away_prob or 0)
                elif outcome == "home_win":
                    model_prob = float(analysis.consensus_home_prob or 0)
                elif outcome == "draw":
                    model_prob = float(analysis.consensus_draw_prob or 0)
                elif outcome == "over_2_5":
                    model_prob = float(analysis.poisson_over_2_5_prob or 0)
                else:
                    continue

                if model_prob == 0:
                    continue

                # Check odds range
                if decimal_odds < strategy["min_odds"] or decimal_odds > strategy["max_odds"]:
                    continue

                # Calculate edge
                market_prob = 1 / decimal_odds
                edge = model_prob - market_prob

                # Strategy-specific checks
                if outcome == "away_win":
                    # Away win: 12%+ edge required (no cap)
                    if edge < strategy["min_edge"]:
                        continue
                    # Exclude when home team form is 4-6 (poor but not terrible)
                    exclude_min = strategy.get("exclude_home_form_min")
                    exclude_max = strategy.get("exclude_home_form_max")
                    if exclude_min and exclude_max and exclude_min <= home_form <= exclude_max:
                        continue
                elif outcome == "home_win":
                    # Home win: negative edge AND form 12+ required
                    # ONLY active in 2020s era
                    if strategy.get("era_only") and match.season < self.era_2020s_start:
                        continue
                    if edge > strategy["max_edge"]:  # Edge must be negative
                        continue
                    if home_form < strategy["min_form"]:  # Need hot streak
                        continue
                elif outcome == "over_2_5":
                    # Over 2.5 goals: 10-12% edge required (tight band)
                    # ONLY active in 2020s era
                    if strategy.get("era_only") and match.season < self.era_2020s_start:
                        continue
                    if edge < strategy["min_edge"]:  # Need at least 10% edge
                        continue
                    if edge > strategy["max_edge"]:  # Cap at 12% (higher is overconfident)
                        continue
                else:
                    continue

                # Skip if already exists
                if (match.id, outcome) in existing_vb:
                    continue

                # Map outcome string to BetOutcome enum
                outcome_map = {
                    "home_win": BetOutcome.HOME_WIN,
                    "draw": BetOutcome.DRAW,
                    "away_win": BetOutcome.AWAY_WIN,
                    "over_2_5": BetOutcome.OVER_2_5,
                }

                bet_outcome = outcome_map.get(outcome)
                if not bet_outcome:
                    continue

                # Calculate Kelly stake
                b = decimal_odds - 1
                kelly_full = (b * model_prob - (1 - model_prob)) / b if b > 0 else 0
                kelly_stake = max(0, min(kelly_full * 0.25, 0.05))  # 25% Kelly, max 5%

                value_bet = ValueBet(
                    match_id=match.id,
                    outcome=bet_outcome,
                    bookmaker=bookmaker,
                    odds=Decimal(str(round(decimal_odds, 2))),
                    model_probability=Decimal(str(round(model_prob, 4))),
                    implied_probability=Decimal(str(round(market_prob, 4))),
                    edge=Decimal(str(round(edge, 4))),
                    kelly_stake=Decimal(str(round(kelly_stake, 4))),
                    recommended_stake=Decimal(str(round(kelly_stake, 4))),
                    is_active=False,  # Historical bets are not active
                )
                self.session.add(value_bet)
                created += 1
                existing_vb.add((match.id, outcome))

            # Commit in batches
            if created % 500 == 0 and created > 0:
                self.session.flush()

        return created, len(matches)


def main():
    parser = argparse.ArgumentParser(description="Backfill value bets for historical matches")
    parser.add_argument(
        "--seasons",
        nargs="+",
        help="Specific seasons to process (e.g., 2023-24 2022-23)",
    )
    parser.add_argument(
        "--start-season",
        default="2000-01",
        help="Earliest season to process (default: 2000-01)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete existing value bets and regenerate",
    )
    args = parser.parse_args()

    import logging
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

    with SyncSessionLocal() as session:
        backfill = ValueBetBackfill(session)
        result = backfill.run(
            seasons=args.seasons,
            start_season=args.start_season,
            force=args.force,
        )
        print(f"Backfill complete: {result['value_bets_created']} value bets created")


if __name__ == "__main__":
    main()
