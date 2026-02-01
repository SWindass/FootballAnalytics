"""Aggregate predictions from external tipster/prediction sources.

This module fetches predictions from various external sources and aggregates
them with our internal models. Research shows aggregated "wisdom of crowds"
predictions contain information not captured by individual models.

Key insight: Betting odds represent the market's collective wisdom and are
extremely efficient. Using them as an input signal (rather than just as
a comparison benchmark) can improve predictions.

Sources:
- Betting odds (via The Odds API) - Market consensus across bookmakers
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta

import structlog

from batch.data_sources.the_odds_api import OddsApiClient

logger = structlog.get_logger()


@dataclass
class ExternalPrediction:
    """Prediction from an external source."""
    source: str
    home_team: str
    away_team: str
    home_prob: float
    draw_prob: float
    away_prob: float
    match_date: datetime | None = None
    confidence: float | None = None
    raw_data: dict | None = None


def decimal_odds_to_probability(odds: float) -> float:
    """Convert decimal odds to implied probability.

    Decimal odds of 2.0 = 50% probability (1/2.0)
    Decimal odds of 3.0 = 33.3% probability (1/3.0)
    """
    if odds <= 1.0:
        return 1.0
    return 1.0 / odds


def normalize_probabilities(home: float, draw: float, away: float) -> tuple[float, float, float]:
    """Normalize probabilities to sum to 1.0.

    Removes bookmaker margin (overround) from implied probabilities.
    """
    total = home + draw + away
    if total <= 0:
        return (0.4, 0.25, 0.35)  # Default fallback
    return (home / total, draw / total, away / total)


class MarketConsensusPredictor:
    """Extracts predictions from betting market odds.

    Betting odds represent the "wisdom of crowds" - the aggregate opinion
    of thousands of bettors and bookmakers. Converting odds to probabilities
    gives us a market consensus prediction.

    Research shows that betting markets are highly efficient and hard to beat,
    making them a valuable external signal for prediction models.
    """

    # Team name mapping: Odds API -> Our database
    TEAM_MAPPING = {
        "Arsenal": "Arsenal FC",
        "Aston Villa": "Aston Villa FC",
        "AFC Bournemouth": "AFC Bournemouth",
        "Brentford": "Brentford FC",
        "Brighton and Hove Albion": "Brighton & Hove Albion FC",
        "Burnley": "Burnley FC",
        "Chelsea": "Chelsea FC",
        "Crystal Palace": "Crystal Palace FC",
        "Everton": "Everton FC",
        "Fulham": "Fulham FC",
        "Ipswich Town": "Ipswich Town FC",
        "Leeds United": "Leeds United FC",
        "Leicester City": "Leicester City FC",
        "Liverpool": "Liverpool FC",
        "Luton Town": "Luton Town FC",
        "Manchester City": "Manchester City FC",
        "Manchester United": "Manchester United FC",
        "Newcastle United": "Newcastle United FC",
        "Nottingham Forest": "Nottingham Forest FC",
        "Sheffield United": "Sheffield United FC",
        "Southampton": "Southampton FC",
        "Tottenham Hotspur": "Tottenham Hotspur FC",
        "West Ham United": "West Ham United FC",
        "Wolverhampton Wanderers": "Wolverhampton Wanderers FC",
    }

    def __init__(self):
        self.odds_client = OddsApiClient()
        self._cache: dict[str, list[ExternalPrediction]] = {}
        self._cache_time: datetime | None = None
        self._cache_duration = timedelta(hours=1)  # Odds change frequently

    def _normalize_team_name(self, name: str) -> str:
        """Normalize team name to match our database."""
        return self.TEAM_MAPPING.get(name, name)

    async def fetch_predictions(self) -> list[ExternalPrediction]:
        """Fetch market consensus predictions from betting odds.

        Fetches odds from multiple bookmakers and averages them to get
        a market consensus prediction.

        Returns:
            List of predictions for upcoming EPL matches
        """
        # Check cache
        if (self._cache_time and
            datetime.now() - self._cache_time < self._cache_duration and
            "epl" in self._cache):
            logger.info("Using cached market consensus predictions")
            return self._cache["epl"]

        try:
            logger.info("Fetching betting odds for market consensus")
            odds_data = await self.odds_client.get_odds(
                markets="h2h",
                regions="uk,eu",
            )

            predictions = []

            for event in odds_data:
                home_team = event.get("home_team", "")
                away_team = event.get("away_team", "")
                commence_time = event.get("commence_time", "")

                # Parse commence time
                try:
                    match_date = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
                except (ValueError, AttributeError):
                    match_date = None

                # Collect odds from all bookmakers
                home_odds_list = []
                draw_odds_list = []
                away_odds_list = []

                for bookmaker in event.get("bookmakers", []):
                    for market in bookmaker.get("markets", []):
                        if market["key"] == "h2h":
                            for outcome in market["outcomes"]:
                                price = outcome.get("price", 0)
                                if price <= 1.0:
                                    continue

                                if outcome["name"] == home_team:
                                    home_odds_list.append(price)
                                elif outcome["name"] == away_team:
                                    away_odds_list.append(price)
                                elif outcome["name"] == "Draw":
                                    draw_odds_list.append(price)

                # Need odds from at least 3 bookmakers for reliable consensus
                if len(home_odds_list) < 3 or len(draw_odds_list) < 3 or len(away_odds_list) < 3:
                    continue

                # Average odds across bookmakers
                avg_home_odds = sum(home_odds_list) / len(home_odds_list)
                avg_draw_odds = sum(draw_odds_list) / len(draw_odds_list)
                avg_away_odds = sum(away_odds_list) / len(away_odds_list)

                # Convert to probabilities
                home_prob = decimal_odds_to_probability(avg_home_odds)
                draw_prob = decimal_odds_to_probability(avg_draw_odds)
                away_prob = decimal_odds_to_probability(avg_away_odds)

                # Normalize to remove overround
                home_prob, draw_prob, away_prob = normalize_probabilities(
                    home_prob, draw_prob, away_prob
                )

                # Confidence based on number of bookmakers and odds variance
                odds_std = (
                    (max(home_odds_list) - min(home_odds_list)) +
                    (max(draw_odds_list) - min(draw_odds_list)) +
                    (max(away_odds_list) - min(away_odds_list))
                ) / 3
                # Lower variance = higher confidence (scale 0.5 to 1.0)
                confidence = max(0.5, 1.0 - odds_std / 2.0)

                predictions.append(ExternalPrediction(
                    source="MarketConsensus",
                    home_team=self._normalize_team_name(home_team),
                    away_team=self._normalize_team_name(away_team),
                    home_prob=home_prob,
                    draw_prob=draw_prob,
                    away_prob=away_prob,
                    match_date=match_date,
                    confidence=confidence,
                    raw_data={
                        "avg_home_odds": avg_home_odds,
                        "avg_draw_odds": avg_draw_odds,
                        "avg_away_odds": avg_away_odds,
                        "num_bookmakers": len(home_odds_list),
                    }
                ))

            # Cache results
            self._cache["epl"] = predictions
            self._cache_time = datetime.now()

            logger.info(f"Fetched {len(predictions)} market consensus predictions")
            return predictions

        except Exception as e:
            logger.error(f"Failed to fetch market consensus predictions: {e}")
            return []

    def fetch_predictions_sync(self) -> list[ExternalPrediction]:
        """Synchronous wrapper for fetch_predictions."""
        return asyncio.run(self.fetch_predictions())


class TipsterAggregator:
    """Aggregates predictions from multiple sources.

    Combines predictions using weighted averaging, where weights
    can be adjusted based on historical accuracy of each source.

    Default weights give significant influence to market consensus,
    as betting markets have proven extremely efficient.
    """

    DEFAULT_WEIGHTS = {
        "MarketConsensus": 0.4,  # Betting odds are highly efficient
        "Internal_ELO": 0.35,    # Our tuned ELO model
        "Internal_Poisson": 0.25,  # Poisson goal model
    }

    def __init__(self):
        self.market_predictor = MarketConsensusPredictor()
        self.weights = self.DEFAULT_WEIGHTS.copy()
        self._market_predictions: list[ExternalPrediction] | None = None

    async def load_market_predictions(self) -> None:
        """Pre-load market predictions to avoid repeated API calls."""
        self._market_predictions = await self.market_predictor.fetch_predictions()

    def get_aggregated_prediction(
        self,
        home_team: str,
        away_team: str,
        internal_elo_probs: tuple[float, float, float] | None = None,
        internal_poisson_probs: tuple[float, float, float] | None = None,
    ) -> tuple[float, float, float]:
        """Get aggregated prediction combining external and internal sources.

        Args:
            home_team: Home team name (as in our database)
            away_team: Away team name (as in our database)
            internal_elo_probs: Our ELO model probabilities (home, draw, away)
            internal_poisson_probs: Our Poisson model probabilities

        Returns:
            Tuple of (home_prob, draw_prob, away_prob)
        """
        predictions = []
        weights = []

        # Use cached market predictions if available
        market_preds = self._market_predictions
        if market_preds is None:
            market_preds = self.market_predictor.fetch_predictions_sync()

        # Find matching market prediction
        for pred in market_preds:
            if pred.home_team == home_team and pred.away_team == away_team:
                predictions.append((pred.home_prob, pred.draw_prob, pred.away_prob))
                # Weight by confidence if available
                base_weight = self.weights.get("MarketConsensus", 0.4)
                confidence = pred.confidence or 1.0
                weights.append(base_weight * confidence)
                break

        # Add internal predictions
        if internal_elo_probs:
            predictions.append(internal_elo_probs)
            weights.append(self.weights.get("Internal_ELO", 0.35))

        if internal_poisson_probs:
            predictions.append(internal_poisson_probs)
            weights.append(self.weights.get("Internal_Poisson", 0.25))

        if not predictions:
            # Fall back to prior distribution if no predictions available
            return (0.40, 0.27, 0.33)  # EPL historical rates

        # Weighted average
        total_weight = sum(weights)
        home_prob = sum(p[0] * w for p, w in zip(predictions, weights, strict=False)) / total_weight
        draw_prob = sum(p[1] * w for p, w in zip(predictions, weights, strict=False)) / total_weight
        away_prob = sum(p[2] * w for p, w in zip(predictions, weights, strict=False)) / total_weight

        # Normalize to ensure sum = 1
        total = home_prob + draw_prob + away_prob
        return (home_prob / total, draw_prob / total, away_prob / total)


async def test_market_consensus():
    """Test market consensus predictor."""
    predictor = MarketConsensusPredictor()
    predictions = await predictor.fetch_predictions()

    print(f"Fetched {len(predictions)} market consensus predictions:")
    print("-" * 80)

    for pred in predictions[:10]:
        date_str = pred.match_date.strftime('%Y-%m-%d %H:%M') if pred.match_date else 'TBD'
        print(f"{date_str:18} {pred.home_team:25} vs {pred.away_team:25}")
        print(f"    Home: {pred.home_prob:.1%}  Draw: {pred.draw_prob:.1%}  Away: {pred.away_prob:.1%}")
        if pred.raw_data:
            print(f"    Odds: H={pred.raw_data['avg_home_odds']:.2f}  "
                  f"D={pred.raw_data['avg_draw_odds']:.2f}  "
                  f"A={pred.raw_data['avg_away_odds']:.2f}  "
                  f"(from {pred.raw_data['num_bookmakers']} bookmakers)")
        print()

    if len(predictions) > 10:
        print(f"... and {len(predictions) - 10} more")


if __name__ == "__main__":
    asyncio.run(test_market_consensus())
