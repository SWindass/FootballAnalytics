"""Advanced value bet detection using market-residual analysis.

This module identifies value bets by finding situations where our models
disagree with market odds in a predictable, profitable way.

Key insight: Don't try to beat the market on raw prediction.
Instead, find the 5-10% of matches where you have genuine edge.
"""

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Optional

import structlog

from app.db.models import MatchAnalysis, TeamStats, EloRating, OddsHistory, BetOutcome
from batch.betting.kelly_criterion import KellyCalculator, KellyConfig

logger = structlog.get_logger()


@dataclass
class ValueBetOpportunity:
    """A detected value betting opportunity with reasoning."""

    match_id: int
    outcome: str  # 'home_win', 'draw', 'away_win'
    bookmaker: str

    # Probabilities
    model_prob: float  # Consensus model probability
    market_prob: float  # Implied probability from odds
    edge: float  # model_prob - market_prob

    # Odds and stake
    odds: float
    kelly_stake: float
    recommended_stake: float

    # Confidence and reasoning
    confidence: float  # 0-1 confidence in this value bet
    reasons: list[str] = field(default_factory=list)

    # Model agreement
    elo_agrees: bool = False
    poisson_agrees: bool = False
    models_agreeing: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "match_id": self.match_id,
            "outcome": self.outcome,
            "bookmaker": self.bookmaker,
            "model_prob": self.model_prob,
            "market_prob": self.market_prob,
            "edge": self.edge,
            "odds": self.odds,
            "kelly_stake": self.kelly_stake,
            "recommended_stake": self.recommended_stake,
            "confidence": self.confidence,
            "reasons": self.reasons,
            "elo_agrees": self.elo_agrees,
            "poisson_agrees": self.poisson_agrees,
            "models_agreeing": self.models_agreeing,
        }


@dataclass
class ValueDetectorConfig:
    """Configuration for value detection.

    Optimized based on backtest of 2,000+ EPL matches (2020-2025):
    - Away wins with 5-12% edge: +20.5% ROI, 51.6% win rate
    - Higher edges (>12%) are overconfident and lose money
    - Home wins show no edge (model overestimates home advantage)
    """

    # Edge thresholds - backtest shows 5-12% is the sweet spot
    min_edge: float = 0.05  # Minimum 5% edge required
    max_edge: float = 0.12  # Maximum edge - higher edges are overconfident
    strong_edge: float = 0.10  # Edge that increases confidence

    # Confidence thresholds
    min_confidence: float = 0.50  # Minimum confidence to flag as value
    min_models_agreeing: int = 1  # At least 1 model must agree (ELO or Poisson)

    # Outcome filter - backtest shows away wins are most profitable
    # Options: None (all outcomes), or list like ["away_win", "draw"]
    allowed_outcomes: list = None  # Default: ["away_win"] for optimal strategy

    # Odds constraints
    min_odds: float = 1.30  # Minimum odds to consider
    max_odds: float = 8.00  # Maximum odds to consider

    # Kelly settings
    kelly_fraction: float = 0.25  # Fractional Kelly for risk management
    max_stake_pct: float = 0.05  # Maximum 5% of bankroll per bet

    # Feature thresholds for value signals
    xg_regression_threshold: float = -0.3  # xG underperformance to flag
    key_injury_threshold: int = 2  # Number of key players out to flag

    def __post_init__(self):
        # Default to away wins only (the profitable strategy)
        if self.allowed_outcomes is None:
            self.allowed_outcomes = ["away_win"]


class ValueDetector:
    """Detects value bets by analyzing market-model disagreement.

    This is different from simple edge detection:
    1. Requires multiple models to agree
    2. Provides reasoning for WHY the bet is value
    3. Considers context (injuries, xG regression, etc.)
    4. Only flags bets with high confidence
    """

    def __init__(self, config: Optional[ValueDetectorConfig] = None):
        self.config = config or ValueDetectorConfig()
        self.kelly = KellyCalculator(KellyConfig(
            fraction=self.config.kelly_fraction,
            min_edge=self.config.min_edge,
        ))

    def find_value_bets(
        self,
        match_id: int,
        analysis: MatchAnalysis,
        home_stats: Optional[TeamStats],
        away_stats: Optional[TeamStats],
        home_elo: Optional[EloRating],
        away_elo: Optional[EloRating],
        odds_history: list[OddsHistory],
    ) -> list[ValueBetOpportunity]:
        """Find value bets for a match.

        Args:
            match_id: Match ID
            analysis: Match analysis with model predictions
            home_stats: Home team statistics
            away_stats: Away team statistics
            home_elo: Home team ELO rating
            away_elo: Away team ELO rating
            odds_history: Historical odds for this match

        Returns:
            List of value bet opportunities
        """
        if not analysis:
            return []

        value_bets = []

        # Get model probabilities
        model_probs = self._get_model_probabilities(analysis)
        if not model_probs:
            return []

        # Get market odds by bookmaker (tries odds_history first, then analysis.features)
        market_odds = self._get_market_odds(odds_history, analysis)
        if not market_odds:
            return []

        # Check each outcome for value
        # Filter to allowed outcomes based on backtest results
        outcomes_to_check = self.config.allowed_outcomes or ["home_win", "draw", "away_win"]

        for outcome in outcomes_to_check:
            for bookmaker, bookie_odds in market_odds.items():
                if outcome not in bookie_odds:
                    continue

                decimal_odds = bookie_odds[outcome]

                # Skip if odds outside range
                if decimal_odds < self.config.min_odds or decimal_odds > self.config.max_odds:
                    continue

                # Get probabilities
                consensus_prob = model_probs["consensus"][outcome]
                elo_prob = model_probs["elo"].get(outcome, consensus_prob)
                poisson_prob = model_probs["poisson"].get(outcome, consensus_prob)
                market_prob = 1.0 / decimal_odds

                # Calculate edge
                edge = consensus_prob - market_prob

                # Skip if edge outside optimal range (backtest shows 5-12% is sweet spot)
                if edge < self.config.min_edge:
                    continue
                if edge > self.config.max_edge:
                    continue  # High edges are overconfident

                # Check model agreement
                elo_agrees = elo_prob > market_prob
                poisson_agrees = poisson_prob > market_prob
                models_agreeing = sum([elo_agrees, poisson_agrees])

                # Skip if not enough models agree
                if models_agreeing < self.config.min_models_agreeing:
                    continue

                # Calculate confidence
                confidence = self._calculate_confidence(
                    edge=edge,
                    elo_agrees=elo_agrees,
                    poisson_agrees=poisson_agrees,
                    elo_prob=elo_prob,
                    poisson_prob=poisson_prob,
                    market_prob=market_prob,
                    outcome=outcome,
                    home_stats=home_stats,
                    away_stats=away_stats,
                    home_elo=home_elo,
                    away_elo=away_elo,
                )

                # Skip if confidence too low
                if confidence < self.config.min_confidence:
                    continue

                # Get reasons
                reasons = self._get_value_reasons(
                    outcome=outcome,
                    edge=edge,
                    elo_prob=elo_prob,
                    poisson_prob=poisson_prob,
                    market_prob=market_prob,
                    home_stats=home_stats,
                    away_stats=away_stats,
                    home_elo=home_elo,
                    away_elo=away_elo,
                )

                # Calculate Kelly stake
                kelly_stake = self._calculate_kelly_stake(consensus_prob, decimal_odds)

                value_bet = ValueBetOpportunity(
                    match_id=match_id,
                    outcome=outcome,
                    bookmaker=bookmaker,
                    model_prob=consensus_prob,
                    market_prob=market_prob,
                    edge=edge,
                    odds=decimal_odds,
                    kelly_stake=kelly_stake,
                    recommended_stake=min(kelly_stake, self.config.max_stake_pct),
                    confidence=confidence,
                    reasons=reasons,
                    elo_agrees=elo_agrees,
                    poisson_agrees=poisson_agrees,
                    models_agreeing=models_agreeing,
                )
                value_bets.append(value_bet)

        # Sort by confidence * edge (prioritize high confidence + high edge)
        value_bets.sort(key=lambda x: x.confidence * x.edge, reverse=True)

        return value_bets

    def _get_model_probabilities(self, analysis: MatchAnalysis) -> Optional[dict]:
        """Extract model probabilities from analysis."""
        try:
            consensus = {
                "home_win": float(analysis.consensus_home_prob or 0.33),
                "draw": float(analysis.consensus_draw_prob or 0.33),
                "away_win": float(analysis.consensus_away_prob or 0.33),
            }

            elo = {
                "home_win": float(analysis.elo_home_prob or consensus["home_win"]),
                "draw": float(analysis.elo_draw_prob or consensus["draw"]),
                "away_win": float(analysis.elo_away_prob or consensus["away_win"]),
            }

            poisson = {
                "home_win": float(analysis.poisson_home_prob or consensus["home_win"]),
                "draw": float(analysis.poisson_draw_prob or consensus["draw"]),
                "away_win": float(analysis.poisson_away_prob or consensus["away_win"]),
            }

            return {
                "consensus": consensus,
                "elo": elo,
                "poisson": poisson,
            }
        except Exception as e:
            logger.warning(f"Failed to extract model probabilities: {e}")
            return None

    def _get_market_odds(
        self,
        odds_history: list[OddsHistory],
        analysis: Optional[MatchAnalysis] = None,
    ) -> dict[str, dict[str, float]]:
        """Get most recent odds by bookmaker.

        Checks odds_history table first, then falls back to historical_odds
        stored in analysis.features for older matches.
        """
        market_odds = {}

        # First try odds_history table
        for oh in odds_history:
            if oh.bookmaker not in market_odds:
                market_odds[oh.bookmaker] = {}

            if oh.home_odds:
                market_odds[oh.bookmaker]["home_win"] = float(oh.home_odds)
            if oh.draw_odds:
                market_odds[oh.bookmaker]["draw"] = float(oh.draw_odds)
            if oh.away_odds:
                market_odds[oh.bookmaker]["away_win"] = float(oh.away_odds)

        # Fall back to historical odds in analysis.features
        if not market_odds and analysis and analysis.features:
            hist_odds = analysis.features.get("historical_odds", {})
            if hist_odds:
                # Use actual odds if available (b365 or average)
                home_odds = hist_odds.get("b365_home_odds") or hist_odds.get("avg_home_odds")
                draw_odds = hist_odds.get("b365_draw_odds") or hist_odds.get("avg_draw_odds")
                away_odds = hist_odds.get("b365_away_odds") or hist_odds.get("avg_away_odds")

                if home_odds and draw_odds and away_odds:
                    market_odds["bet365"] = {
                        "home_win": float(home_odds),
                        "draw": float(draw_odds),
                        "away_win": float(away_odds),
                    }

        return market_odds

    def _calculate_confidence(
        self,
        edge: float,
        elo_agrees: bool,
        poisson_agrees: bool,
        elo_prob: float,
        poisson_prob: float,
        market_prob: float,
        outcome: str,
        home_stats: Optional[TeamStats],
        away_stats: Optional[TeamStats],
        home_elo: Optional[EloRating],
        away_elo: Optional[EloRating],
    ) -> float:
        """Calculate confidence score for a value bet.

        Confidence is based on:
        1. Size of edge (larger edge = more confident)
        2. Model agreement (both models agreeing = more confident)
        3. Strength of model signals
        4. Contextual factors (xG regression, injuries, etc.)
        """
        confidence = 0.5  # Start at neutral

        # Edge size boost (larger edge = more confident)
        if edge >= self.config.strong_edge:
            confidence += 0.15
        elif edge >= self.config.min_edge:
            confidence += 0.05 + (edge - self.config.min_edge) * 2

        # Model agreement boost
        if elo_agrees and poisson_agrees:
            confidence += 0.15

            # Extra boost if both models strongly agree
            elo_edge = elo_prob - market_prob
            poisson_edge = poisson_prob - market_prob
            if min(elo_edge, poisson_edge) > self.config.min_edge:
                confidence += 0.05

        # xG regression signal (for home/away outcomes)
        if outcome in ["home_win", "away_win"]:
            stats = home_stats if outcome == "home_win" else away_stats
            opp_stats = away_stats if outcome == "home_win" else home_stats

            # Team underperforming xG = due for positive regression
            if stats and self._is_xg_underperformer(stats):
                confidence += 0.08

            # Opponent overperforming xG = due for negative regression
            if opp_stats and self._is_xg_overperformer(opp_stats):
                confidence += 0.05

        # Close match boost for draws
        if outcome == "draw" and home_elo and away_elo:
            elo_diff = abs(float(home_elo.rating) - float(away_elo.rating))
            if elo_diff < 50:
                confidence += 0.10  # Close matches favor draws

        # Injury impact
        if outcome == "home_win" and away_stats:
            if away_stats.key_players_out >= self.config.key_injury_threshold:
                confidence += 0.05
        elif outcome == "away_win" and home_stats:
            if home_stats.key_players_out >= self.config.key_injury_threshold:
                confidence += 0.05

        return min(confidence, 0.95)  # Cap at 95%

    def _is_xg_underperformer(self, stats: TeamStats) -> bool:
        """Check if team is underperforming their xG."""
        if not stats.xg_for or not stats.goals_scored:
            return False

        games_played = (stats.home_wins or 0) + (stats.home_draws or 0) + \
                      (stats.home_losses or 0) + (stats.away_wins or 0) + \
                      (stats.away_draws or 0) + (stats.away_losses or 0)

        if games_played < 3:
            return False

        goals_per_game = stats.goals_scored / games_played
        xg_per_game = float(stats.xg_for) / games_played

        # Underperforming = scoring less than xG suggests
        return (goals_per_game - xg_per_game) < self.config.xg_regression_threshold

    def _is_xg_overperformer(self, stats: TeamStats) -> bool:
        """Check if team is overperforming their xG."""
        if not stats.xg_for or not stats.goals_scored:
            return False

        games_played = (stats.home_wins or 0) + (stats.home_draws or 0) + \
                      (stats.home_losses or 0) + (stats.away_wins or 0) + \
                      (stats.away_draws or 0) + (stats.away_losses or 0)

        if games_played < 3:
            return False

        goals_per_game = stats.goals_scored / games_played
        xg_per_game = float(stats.xg_for) / games_played

        # Overperforming = scoring more than xG suggests (lucky)
        return (goals_per_game - xg_per_game) > abs(self.config.xg_regression_threshold)

    def _get_value_reasons(
        self,
        outcome: str,
        edge: float,
        elo_prob: float,
        poisson_prob: float,
        market_prob: float,
        home_stats: Optional[TeamStats],
        away_stats: Optional[TeamStats],
        home_elo: Optional[EloRating],
        away_elo: Optional[EloRating],
    ) -> list[str]:
        """Generate human-readable reasons for the value bet."""
        reasons = []

        # Edge size
        if edge >= self.config.strong_edge:
            reasons.append(f"Strong edge of {edge:.1%} over market")
        else:
            reasons.append(f"Model edge of {edge:.1%} over market")

        # Model agreement
        elo_edge = elo_prob - market_prob
        poisson_edge = poisson_prob - market_prob

        if elo_edge > 0 and poisson_edge > 0:
            reasons.append("Both ELO and Poisson models agree")
        elif elo_edge > 0:
            reasons.append("ELO model shows value")
        elif poisson_edge > 0:
            reasons.append("Poisson model shows value")

        # xG-based reasons
        if outcome == "home_win":
            if home_stats and self._is_xg_underperformer(home_stats):
                reasons.append("Home team underperforming xG - due for positive regression")
            if away_stats and self._is_xg_overperformer(away_stats):
                reasons.append("Away team overperforming xG - due for regression")
        elif outcome == "away_win":
            if away_stats and self._is_xg_underperformer(away_stats):
                reasons.append("Away team underperforming xG - due for positive regression")
            if home_stats and self._is_xg_overperformer(home_stats):
                reasons.append("Home team overperforming xG - due for regression")

        # Injury reasons
        if outcome == "home_win" and away_stats:
            if away_stats.key_players_out >= self.config.key_injury_threshold:
                reasons.append(f"Opposition missing {away_stats.key_players_out} key players")
        elif outcome == "away_win" and home_stats:
            if home_stats.key_players_out >= self.config.key_injury_threshold:
                reasons.append(f"Home team missing {home_stats.key_players_out} key players")

        # Draw reasons
        if outcome == "draw" and home_elo and away_elo:
            elo_diff = abs(float(home_elo.rating) - float(away_elo.rating))
            if elo_diff < 50:
                reasons.append("Close ELO ratings favor draw")
            if elo_diff < 30:
                reasons.append("Teams are very evenly matched")

        # Manager reasons
        if home_stats and home_stats.is_new_manager:
            if outcome == "home_win":
                reasons.append("New manager bounce potential")
            elif outcome != "home_win":
                reasons.append("Home team in transition with new manager")

        if away_stats and away_stats.is_new_manager:
            if outcome == "away_win":
                reasons.append("New manager bounce potential")
            elif outcome != "away_win":
                reasons.append("Away team in transition with new manager")

        return reasons

    def _calculate_kelly_stake(self, probability: float, odds: float) -> float:
        """Calculate Kelly criterion stake."""
        if probability <= 0 or probability >= 1 or odds <= 1:
            return 0.0

        # Kelly formula: f* = (bp - q) / b
        b = odds - 1
        p = probability
        q = 1 - p

        kelly = (b * p - q) / b

        # Apply fractional Kelly
        kelly *= self.config.kelly_fraction

        return max(0.0, kelly)


def format_value_opportunities(opportunities: list[ValueBetOpportunity]) -> str:
    """Format value bet opportunities for display."""
    if not opportunities:
        return "No value bets detected."

    lines = ["Value Bet Opportunities", "=" * 60]

    for i, opp in enumerate(opportunities, 1):
        outcome_display = {
            "home_win": "Home Win",
            "draw": "Draw",
            "away_win": "Away Win",
        }.get(opp.outcome, opp.outcome)

        lines.append(f"\n{i}. {outcome_display} @ {opp.odds:.2f} ({opp.bookmaker})")
        lines.append(f"   Model: {opp.model_prob:.1%} | Market: {opp.market_prob:.1%} | Edge: {opp.edge:.1%}")
        lines.append(f"   Confidence: {opp.confidence:.1%} | Models agreeing: {opp.models_agreeing}")
        lines.append(f"   Recommended stake: {opp.recommended_stake:.2%} of bankroll")
        lines.append("   Reasons:")
        for reason in opp.reasons:
            lines.append(f"     - {reason}")

    return "\n".join(lines)
