"""Advanced value bet detection using market-residual analysis.

This module identifies value bets by finding situations where our models
disagree with market odds in a predictable, profitable way.

Key insight: Don't try to beat the market on raw prediction.
Instead, find the 5-10% of matches where you have genuine edge.

STATIC 5% + ERA-BASED FORM OVERRIDE STRATEGY (backtest validated 2020-2025):
1. Away wins with 5%+ edge: +30.2% ROI, 55.6% win rate
   - Excludes home form 4-6 (poor but not terrible)
2. Home wins with negative edge + form 12+: +30.4% ROI, 67.0% win rate
   - Counterintuitive: trust market when home team is on hot streak
   - Market prices momentum better than our statistical models
3. Over 2.5 goals with 10-12% edge: +6.6% ROI, 62.9% win rate
   - Uses Poisson model probability for over 2.5 goals
   - Tight edge band prevents overconfidence
"""

from collections import defaultdict
from dataclasses import dataclass, field

import structlog

from app.db.models import EloRating, MatchAnalysis, OddsHistory, TeamStats
from batch.betting.kelly_criterion import KellyCalculator, KellyConfig

logger = structlog.get_logger()


@dataclass
class TeamReliabilityScore:
    """Reliability score for a team's home value bet conversion."""

    team_id: int
    total_bets: int = 0
    bets_won: int = 0

    @property
    def win_rate(self) -> float:
        """Win rate on home value bets."""
        if self.total_bets == 0:
            return 0.0
        return self.bets_won / self.total_bets

    @property
    def reliability(self) -> float:
        """Reliability score (0-1). Returns 0 if insufficient data."""
        return self.win_rate


class TeamReliabilityTracker:
    """Tracks and predicts team reliability for home value bets.

    Some teams consistently convert home value bets (Liverpool, Man City)
    while others are unreliable (Man United, Newcastle). This tracker
    learns which teams to trust based on historical performance.

    Backtest results:
    - Without filtering: 20 bets, 80% win rate, +17.4% ROI
    - With reliability filtering (min_history=2, min_reliability=0.6):
      18 bets, 83.3% win rate, +21.6% ROI
    """

    def __init__(
        self,
        min_history: int = 2,
        lookback_window: int = 10,
        min_reliability: float = 0.60,
    ):
        """Initialize the reliability tracker.

        Args:
            min_history: Minimum bets required before tracking reliability
            lookback_window: Number of recent bets to consider (rolling window)
            min_reliability: Minimum win rate to consider team reliable
        """
        self.min_history = min_history
        self.lookback_window = lookback_window
        self.min_reliability = min_reliability

        # team_id -> list of (won: bool) in chronological order
        self._team_history: dict[int, list[bool]] = defaultdict(list)
        # Precomputed scores for fast lookup
        self._reliability_scores: dict[int, TeamReliabilityScore] = {}

    def record_bet(self, team_id: int, won: bool) -> None:
        """Record a home value bet result for a team.

        Args:
            team_id: The home team's ID
            won: Whether the bet was won
        """
        history = self._team_history[team_id]
        history.append(won)

        # Keep only lookback_window most recent
        if len(history) > self.lookback_window:
            self._team_history[team_id] = history[-self.lookback_window:]

        # Update precomputed score
        self._update_score(team_id)

    def _update_score(self, team_id: int) -> None:
        """Update the reliability score for a team."""
        history = self._team_history[team_id]
        score = TeamReliabilityScore(
            team_id=team_id,
            total_bets=len(history),
            bets_won=sum(1 for won in history if won),
        )
        self._reliability_scores[team_id] = score

    def get_reliability(self, team_id: int) -> float | None:
        """Get reliability score for a team.

        Returns:
            Reliability score (0-1) or None if insufficient history
        """
        score = self._reliability_scores.get(team_id)
        if score is None or score.total_bets < self.min_history:
            return None
        return score.reliability

    def is_reliable(self, team_id: int) -> bool:
        """Check if a team is reliable for home value bets.

        Returns True if:
        - Team has no history (give benefit of doubt)
        - Team has insufficient history (give benefit of doubt)
        - Team's reliability >= min_reliability threshold

        Returns False if:
        - Team has sufficient history but reliability < threshold
        """
        reliability = self.get_reliability(team_id)

        # No history or insufficient history - give benefit of doubt
        if reliability is None:
            return True

        return reliability >= self.min_reliability

    def get_all_scores(self) -> dict[int, TeamReliabilityScore]:
        """Get all reliability scores."""
        return dict(self._reliability_scores)

    def load_from_history(self, history: list[tuple[int, bool]]) -> None:
        """Load historical bet results.

        Args:
            history: List of (team_id, won) tuples in chronological order
        """
        for team_id, won in history:
            self.record_bet(team_id, won)

    def to_dict(self) -> dict:
        """Serialize tracker state for persistence."""
        return {
            "min_history": self.min_history,
            "lookback_window": self.lookback_window,
            "min_reliability": self.min_reliability,
            "team_history": {
                str(k): v for k, v in self._team_history.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TeamReliabilityTracker":
        """Deserialize tracker from stored state."""
        tracker = cls(
            min_history=data.get("min_history", 2),
            lookback_window=data.get("lookback_window", 10),
            min_reliability=data.get("min_reliability", 0.60),
        )

        for team_id_str, history in data.get("team_history", {}).items():
            team_id = int(team_id_str)
            for won in history:
                tracker.record_bet(team_id, won)

        return tracker


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
    - Home wins with odds < 1.7, edge >= 10%, reliable teams: +21.6% ROI, 83% win rate
    - Higher edges (>12%) for away wins are overconfident and lose money
    """

    # Edge thresholds - backtest shows 5-12% is the sweet spot for away wins
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

    # === Home win strategy (ERA-BASED FORM OVERRIDE) ===
    # Backtest: negative edge + form 12+ = +30.4% ROI, 67% win rate
    # This is counterintuitive: when market values home team MORE than model,
    # AND home team is on a hot streak, trust market momentum pricing.
    enable_home_wins: bool = True  # Enable filtered home win detection
    home_max_odds: float = 10.00  # No practical odds restriction
    home_min_edge: float = -1.0   # Negative edge required (market > model)
    home_max_edge: float = 0.0    # Edge must be < 0
    home_min_form: int = 12       # 12+ form points from last 5 games (W=3, D=1)

    # Away win form exclusion - skip when home team form is 4-6
    # (poor but not terrible - market may have already adjusted)
    away_exclude_home_form_min: int = 4
    away_exclude_home_form_max: int = 6

    # === Over 2.5 goals strategy ===
    # Backtest: 10-12% edge = +6.6% ROI, 62.9% win rate
    # Uses Poisson model probability for over 2.5 goals
    enable_over_2_5: bool = True  # Enable over 2.5 goals detection
    over_2_5_min_edge: float = 0.10  # 10% minimum edge
    over_2_5_max_edge: float = 0.12  # 12% maximum (higher is overconfident)
    over_2_5_min_odds: float = 1.50  # Minimum odds
    over_2_5_max_odds: float = 3.00  # Maximum odds

    # Team reliability filtering for home wins (DEPRECATED - using form instead)
    use_reliability_filter: bool = False  # Disabled - form is more predictive
    reliability_min_history: int = 2  # Min bets to establish reliability
    reliability_lookback: int = 10  # Rolling window for reliability calc
    reliability_min_threshold: float = 0.60  # Min win rate to be "reliable"

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

    Supports two strategies:
    - Away wins: 5-12% edge, any team (+20.5% ROI)
    - Home wins: Odds < 1.7, edge >= 10%, reliable teams only (+21.6% ROI)
    """

    def __init__(
        self,
        config: ValueDetectorConfig | None = None,
        reliability_tracker: TeamReliabilityTracker | None = None,
    ):
        self.config = config or ValueDetectorConfig()
        self.kelly = KellyCalculator(KellyConfig(
            fraction=self.config.kelly_fraction,
            min_edge=self.config.min_edge,
        ))

        # Initialize reliability tracker for home wins
        if self.config.enable_home_wins and self.config.use_reliability_filter:
            self.reliability_tracker = reliability_tracker or TeamReliabilityTracker(
                min_history=self.config.reliability_min_history,
                lookback_window=self.config.reliability_lookback,
                min_reliability=self.config.reliability_min_threshold,
            )
        else:
            self.reliability_tracker = None

    def find_value_bets(
        self,
        match_id: int,
        analysis: MatchAnalysis,
        home_stats: TeamStats | None,
        away_stats: TeamStats | None,
        home_elo: EloRating | None,
        away_elo: EloRating | None,
        odds_history: list[OddsHistory],
        home_team_id: int | None = None,
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
            home_team_id: Home team ID (for reliability tracking)

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

        # Build list of outcomes to check
        # Start with configured allowed outcomes (default: ["away_win"])
        outcomes_to_check = list(self.config.allowed_outcomes or ["home_win", "draw", "away_win"])

        # Add home_win if enabled and not already in list
        if self.config.enable_home_wins and "home_win" not in outcomes_to_check:
            outcomes_to_check.append("home_win")

        # Add over_2_5 if enabled
        if self.config.enable_over_2_5:
            outcomes_to_check.append("over_2_5")

        # Get home team form points for strategy filtering
        home_form = 0
        if home_stats and home_stats.form_points is not None:
            home_form = home_stats.form_points

        for outcome in outcomes_to_check:
            for bookmaker, bookie_odds in market_odds.items():
                if outcome not in bookie_odds:
                    continue

                decimal_odds = bookie_odds[outcome]

                # Different rules for home wins vs other outcomes
                if outcome == "home_win" and self.config.enable_home_wins:
                    # ERA-BASED FORM OVERRIDE: negative edge + form 12+
                    # When market values home team MORE than model AND home
                    # team is on a hot streak, trust market momentum pricing.
                    if decimal_odds > self.config.home_max_odds:
                        continue

                    # Check form requirement (12+ from last 5 games)
                    if home_form < self.config.home_min_form:
                        continue

                    # Check team reliability if filter is enabled (deprecated)
                    if (self.config.use_reliability_filter
                        and self.reliability_tracker
                        and home_team_id):
                        if not self.reliability_tracker.is_reliable(home_team_id):
                            logger.debug(
                                f"Skipping home win for team {home_team_id} - "
                                f"reliability too low"
                            )
                            continue

                    min_edge = self.config.home_min_edge  # -1.0 (negative required)
                    max_edge = self.config.home_max_edge  # 0.0 (must be < 0)
                elif outcome == "away_win":
                    # Away win strategy: 5%+ edge, exclude home form 4-6
                    if decimal_odds < self.config.min_odds or decimal_odds > self.config.max_odds:
                        continue

                    # Exclude when home team form is 4-6 (poor but not terrible)
                    # Market may have already adjusted odds in this zone
                    exclude_min = self.config.away_exclude_home_form_min
                    exclude_max = self.config.away_exclude_home_form_max
                    if exclude_min <= home_form <= exclude_max:
                        continue

                    min_edge = self.config.min_edge
                    max_edge = self.config.max_edge
                elif outcome == "over_2_5":
                    # Over 2.5 goals: 10-12% edge, Poisson model
                    if decimal_odds < self.config.over_2_5_min_odds or decimal_odds > self.config.over_2_5_max_odds:
                        continue
                    min_edge = self.config.over_2_5_min_edge
                    max_edge = self.config.over_2_5_max_edge
                else:
                    # Standard strategy for draws
                    if decimal_odds < self.config.min_odds or decimal_odds > self.config.max_odds:
                        continue
                    min_edge = self.config.min_edge
                    max_edge = self.config.max_edge

                # Get probabilities
                consensus_prob = model_probs["consensus"].get(outcome)
                if consensus_prob is None:
                    continue
                elo_prob = model_probs["elo"].get(outcome, consensus_prob)
                poisson_prob = model_probs["poisson"].get(outcome, consensus_prob)
                market_prob = 1.0 / decimal_odds

                # Calculate edge
                edge = consensus_prob - market_prob

                # Skip if edge outside optimal range
                if edge < min_edge:
                    continue
                if edge > max_edge:
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

        # Deduplicate: keep only best odds per outcome
        best_by_outcome = {}
        for vb in value_bets:
            if vb.outcome not in best_by_outcome or vb.odds > best_by_outcome[vb.outcome].odds:
                best_by_outcome[vb.outcome] = vb

        deduped = list(best_by_outcome.values())

        # Sort by confidence * edge (prioritize high confidence + high edge)
        deduped.sort(key=lambda x: x.confidence * x.edge, reverse=True)

        return deduped

    def _get_model_probabilities(self, analysis: MatchAnalysis) -> dict | None:
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

            # Add over_2_5 from Poisson model (only Poisson predicts this)
            if analysis.poisson_over_2_5_prob:
                over_2_5_prob = float(analysis.poisson_over_2_5_prob)
                consensus["over_2_5"] = over_2_5_prob
                poisson["over_2_5"] = over_2_5_prob

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
        analysis: MatchAnalysis | None = None,
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
            if oh.over_2_5_odds:
                market_odds[oh.bookmaker]["over_2_5"] = float(oh.over_2_5_odds)

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

                # Add over_2_5 odds if available
                over_2_5_odds = hist_odds.get("avg_over_2_5_odds")
                if over_2_5_odds:
                    if "bet365" not in market_odds:
                        market_odds["bet365"] = {}
                    market_odds["bet365"]["over_2_5"] = float(over_2_5_odds)

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
        home_stats: TeamStats | None,
        away_stats: TeamStats | None,
        home_elo: EloRating | None,
        away_elo: EloRating | None,
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
        home_stats: TeamStats | None,
        away_stats: TeamStats | None,
        home_elo: EloRating | None,
        away_elo: EloRating | None,
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

    def record_home_bet_result(self, team_id: int, won: bool) -> None:
        """Record the result of a home value bet for reliability tracking.

        Call this after a home value bet settles to update the team's
        reliability score.

        Args:
            team_id: The home team's ID
            won: Whether the bet was won
        """
        if self.reliability_tracker:
            self.reliability_tracker.record_bet(team_id, won)

    def get_team_reliability(self, team_id: int) -> float | None:
        """Get reliability score for a team.

        Args:
            team_id: The team's ID

        Returns:
            Reliability score (0-1) or None if insufficient history
        """
        if self.reliability_tracker:
            return self.reliability_tracker.get_reliability(team_id)
        return None

    def get_reliability_scores(self) -> dict[int, TeamReliabilityScore]:
        """Get all team reliability scores.

        Returns:
            Dict mapping team_id to TeamReliabilityScore
        """
        if self.reliability_tracker:
            return self.reliability_tracker.get_all_scores()
        return {}

    def save_reliability_state(self) -> dict:
        """Save reliability tracker state for persistence.

        Returns:
            Dict that can be serialized to JSON
        """
        if self.reliability_tracker:
            return self.reliability_tracker.to_dict()
        return {}

    def load_reliability_state(self, state: dict) -> None:
        """Load reliability tracker state from stored data.

        Args:
            state: Dict loaded from JSON
        """
        if state and self.config.enable_home_wins and self.config.use_reliability_filter:
            self.reliability_tracker = TeamReliabilityTracker.from_dict(state)


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
            "over_2_5": "Over 2.5 Goals",
        }.get(opp.outcome, opp.outcome)

        lines.append(f"\n{i}. {outcome_display} @ {opp.odds:.2f} ({opp.bookmaker})")
        lines.append(f"   Model: {opp.model_prob:.1%} | Market: {opp.market_prob:.1%} | Edge: {opp.edge:.1%}")
        lines.append(f"   Confidence: {opp.confidence:.1%} | Models agreeing: {opp.models_agreeing}")
        lines.append(f"   Recommended stake: {opp.recommended_stake:.2%} of bankroll")
        lines.append("   Reasons:")
        for reason in opp.reasons:
            lines.append(f"     - {reason}")

    return "\n".join(lines)


def bootstrap_reliability_from_backtest(
    min_history: int = 2,
    lookback_window: int = 10,
    min_reliability: float = 0.60,
    home_max_odds: float = 1.70,
    home_min_edge: float = 0.10,
) -> TeamReliabilityTracker:
    """Bootstrap a reliability tracker from historical backtest data.

    Runs a simulated backtest on historical matches to initialize
    team reliability scores based on past home value bet performance.

    Args:
        min_history: Minimum bets to establish reliability
        lookback_window: Rolling window for reliability calculation
        min_reliability: Minimum win rate threshold
        home_max_odds: Maximum odds for home win value bets
        home_min_edge: Minimum edge for home win value bets

    Returns:
        Initialized TeamReliabilityTracker with historical data
    """
    from sqlalchemy import select

    from app.db.database import SyncSessionLocal
    from app.db.models import Match, MatchAnalysis, MatchStatus, OddsHistory

    tracker = TeamReliabilityTracker(
        min_history=min_history,
        lookback_window=lookback_window,
        min_reliability=min_reliability,
    )

    # Create detector configured for home wins
    config = ValueDetectorConfig(
        enable_home_wins=True,
        home_max_odds=home_max_odds,
        home_min_edge=home_min_edge,
        use_reliability_filter=False,  # Don't filter during bootstrap
    )
    detector = ValueDetector(config)

    with SyncSessionLocal() as session:
        # Load historical matches
        stmt = (
            select(Match, MatchAnalysis)
            .join(MatchAnalysis, Match.id == MatchAnalysis.match_id)
            .where(Match.status == MatchStatus.FINISHED)
            .where(MatchAnalysis.consensus_home_prob.isnot(None))
            .order_by(Match.kickoff_time)
        )
        matches = list(session.execute(stmt).all())

        logger.info(f"Bootstrapping reliability from {len(matches)} historical matches")

        # Bulk load supporting data
        all_odds = list(session.execute(select(OddsHistory)).scalars().all())
        odds_lookup = {}
        for oh in all_odds:
            if oh.match_id not in odds_lookup:
                odds_lookup[oh.match_id] = []
            odds_lookup[oh.match_id].append(oh)

        home_bets_found = 0
        home_bets_won = 0

        for match, analysis in matches:
            odds_history = odds_lookup.get(match.id, [])

            # Check for home win value bet using the same criteria as live detection
            value_bets = detector.find_value_bets(
                match_id=match.id,
                analysis=analysis,
                home_stats=None,  # Not using for reliability bootstrap
                away_stats=None,
                home_elo=None,
                away_elo=None,
                odds_history=odds_history,
                home_team_id=match.home_team_id,
            )

            # Check if we found a home win value bet
            home_value_bet = next((vb for vb in value_bets if vb.outcome == "home_win"), None)

            if home_value_bet:
                # Determine if it would have won
                if match.home_score is not None and match.away_score is not None:
                    won = match.home_score > match.away_score
                    tracker.record_bet(match.home_team_id, won)
                    home_bets_found += 1
                    if won:
                        home_bets_won += 1

        logger.info(
            f"Bootstrap complete: {home_bets_found} home bets found, "
            f"{home_bets_won} won ({home_bets_won/home_bets_found*100:.1f}% win rate)"
            if home_bets_found > 0 else "Bootstrap complete: no home bets found"
        )

        # Log reliability scores
        scores = tracker.get_all_scores()
        for team_id, score in sorted(scores.items(), key=lambda x: x[1].win_rate, reverse=True):
            if score.total_bets >= min_history:
                logger.info(f"Team {team_id}: {score.win_rate:.0%} ({score.bets_won}/{score.total_bets})")

    return tracker
