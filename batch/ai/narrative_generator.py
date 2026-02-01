"""AI narrative generation using Claude API."""

from typing import Any

import anthropic
import structlog

from app.core.config import get_settings

logger = structlog.get_logger()
settings = get_settings()


MATCH_PREVIEW_PROMPT = """You are a football analyst writing a match preview for the English Premier League.

Write a concise, engaging match preview (200-250 words) for the following fixture:

**{home_team} vs {away_team}**
- Kickoff: {kickoff_time}
- Venue: {venue}

**Home Team ({home_team}) Form:**
- Recent form: {home_form}
- League position: {home_position}
- Goals scored/conceded: {home_goals_for}/{home_goals_against}
- Key stats: {home_stats}

**Away Team ({away_team}) Form:**
- Recent form: {away_form}
- League position: {away_position}
- Goals scored/conceded: {away_goals_for}/{away_goals_against}
- Key stats: {away_stats}

**Head-to-Head (Last 5 meetings):**
{h2h_summary}

**Model Predictions:**
- Consensus: Home {home_prob}% | Draw {draw_prob}% | Away {away_prob}%
- Predicted score: {predicted_score}

**Prediction Confidence Analysis:**
{confidence_analysis}

**Value Edge Analysis:**
{value_edge_analysis}

**Notable absences:**
- {home_team}: {home_injuries}
- {away_team}: {away_injuries}

Write the preview with these sections:
1. **Match Overview** - Recent form, context, and what's at stake
2. **Key Tactical Battle** - The matchup that will decide this game
3. **Prediction Confidence** - IMPORTANT: Clearly state whether this is a HIGH CONFIDENCE or UNCERTAIN prediction based on the confidence analysis above. If models agree, emphasize this gives us more certainty. If they disagree, warn that this is a difficult match to call.
4. **Betting Angle** - Whether value exists and how confident we are in the prediction

Keep the tone professional. Be honest about uncertainty - if models disagree, say so clearly."""


class NarrativeGenerator:
    """Generates AI match narratives using Claude API."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or settings.anthropic_api_key
        self.client = anthropic.Anthropic(api_key=self.api_key) if self.api_key else None

    async def generate_match_preview(
        self,
        match_data: dict[str, Any],
        home_stats: dict[str, Any],
        away_stats: dict[str, Any],
        predictions: dict[str, float],
        h2h_history: list[dict] | None = None,
        value_bets: list[dict] | None = None,
        odds: dict[str, float] | None = None,
        confidence_data: dict[str, Any] | None = None,
    ) -> str:
        """Generate a match preview narrative.

        Args:
            match_data: Match information (teams, kickoff, venue)
            home_stats: Home team statistics
            away_stats: Away team statistics
            predictions: Model predictions (home_prob, draw_prob, away_prob)
            h2h_history: Optional head-to-head history
            value_bets: Optional list of detected value bets for this match
            odds: Optional current bookmaker odds (home_odds, draw_odds, away_odds)
            confidence_data: Optional confidence analysis with model agreement info

        Returns:
            Generated narrative text
        """
        if not self.client:
            logger.warning("Anthropic API key not configured, returning placeholder")
            return self._generate_placeholder(match_data, predictions)

        # Format H2H summary
        h2h_summary = self._format_h2h(h2h_history) if h2h_history else "No recent meetings"

        # Format injuries
        home_injuries = ", ".join(home_stats.get("injuries", [])) or "None reported"
        away_injuries = ", ".join(away_stats.get("injuries", [])) or "None reported"

        # Format value edge analysis
        value_edge_analysis = self._format_value_edge(predictions, odds, value_bets)

        # Format confidence analysis
        confidence_analysis = self._format_confidence_analysis(confidence_data)

        # Build prompt
        prompt = MATCH_PREVIEW_PROMPT.format(
            home_team=match_data["home_team"],
            away_team=match_data["away_team"],
            kickoff_time=match_data["kickoff_time"].strftime("%A %d %B, %H:%M"),
            venue=match_data.get("venue", "TBC"),
            home_form=home_stats.get("form", "N/A"),
            home_position=home_stats.get("position", "N/A"),
            home_goals_for=home_stats.get("goals_scored", 0),
            home_goals_against=home_stats.get("goals_conceded", 0),
            home_stats=self._format_team_stats(home_stats),
            away_form=away_stats.get("form", "N/A"),
            away_position=away_stats.get("position", "N/A"),
            away_goals_for=away_stats.get("goals_scored", 0),
            away_goals_against=away_stats.get("goals_conceded", 0),
            away_stats=self._format_team_stats(away_stats),
            h2h_summary=h2h_summary,
            home_prob=round(predictions.get("home_win", 0.33) * 100, 1),
            draw_prob=round(predictions.get("draw", 0.33) * 100, 1),
            away_prob=round(predictions.get("away_win", 0.33) * 100, 1),
            predicted_score=predictions.get("predicted_score", "N/A"),
            confidence_analysis=confidence_analysis,
            value_edge_analysis=value_edge_analysis,
            home_injuries=home_injuries,
            away_injuries=away_injuries,
        )

        try:
            message = self.client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
            )
            return message.content[0].text
        except Exception as e:
            logger.error("Failed to generate narrative", error=str(e))
            return self._generate_placeholder(match_data, predictions)

    def _format_h2h(self, h2h_history: list[dict]) -> str:
        """Format head-to-head history."""
        if not h2h_history:
            return "No recent meetings"

        lines = []
        for match in h2h_history[:5]:
            date = match.get("date", "Unknown")
            home = match.get("home_team", "")
            away = match.get("away_team", "")
            score = f"{match.get('home_score', '?')}-{match.get('away_score', '?')}"
            lines.append(f"- {date}: {home} {score} {away}")

        return "\n".join(lines)

    def _format_team_stats(self, stats: dict) -> str:
        """Format team statistics for prompt."""
        parts = []

        if stats.get("avg_xg_for"):
            parts.append(f"xG: {stats['avg_xg_for']:.2f}")
        if stats.get("clean_sheets"):
            parts.append(f"Clean sheets: {stats['clean_sheets']}")
        if stats.get("home_wins") is not None:
            parts.append(f"Home record: {stats.get('home_wins', 0)}W-{stats.get('home_draws', 0)}D-{stats.get('home_losses', 0)}L")
        if stats.get("away_wins") is not None:
            parts.append(f"Away record: {stats.get('away_wins', 0)}W-{stats.get('away_draws', 0)}D-{stats.get('away_losses', 0)}L")

        return ", ".join(parts) if parts else "N/A"

    def _format_confidence_analysis(
        self,
        confidence_data: dict[str, Any] | None,
    ) -> str:
        """Format confidence analysis for the prompt.

        Args:
            confidence_data: Dict containing:
                - confidence: float (0-1), agreement strength
                - models_agree: bool, whether all models pick same favorite
                - elo_probs: tuple of ELO model probabilities
                - poisson_probs: tuple of Poisson model probabilities
                - market_probs: tuple of market odds probabilities

        Returns:
            Formatted string describing prediction confidence
        """
        if not confidence_data:
            return "Confidence data not available."

        confidence = confidence_data.get("confidence", 0)
        models_agree = confidence_data.get("models_agree", False)
        elo_probs = confidence_data.get("elo_probs", (0.4, 0.27, 0.33))
        poisson_probs = confidence_data.get("poisson_probs", (0.4, 0.27, 0.33))
        market_probs = confidence_data.get("market_probs", (0.4, 0.27, 0.33))

        lines = []

        # Individual model predictions
        lines.append("Individual Model Predictions:")
        lines.append(f"  - ELO Model:    Home {elo_probs[0]*100:.0f}% | Draw {elo_probs[1]*100:.0f}% | Away {elo_probs[2]*100:.0f}%")
        lines.append(f"  - Poisson Model: Home {poisson_probs[0]*100:.0f}% | Draw {poisson_probs[1]*100:.0f}% | Away {poisson_probs[2]*100:.0f}%")
        lines.append(f"  - Market Odds:   Home {market_probs[0]*100:.0f}% | Draw {market_probs[1]*100:.0f}% | Away {market_probs[2]*100:.0f}%")
        lines.append("")

        # Agreement status
        if models_agree:
            # Determine what they agree on
            outcomes = ["Home Win", "Draw", "Away Win"]
            elo_fav = elo_probs.index(max(elo_probs))
            agreed_outcome = outcomes[elo_fav]

            if confidence >= 0.5:
                lines.append(f"**HIGH CONFIDENCE PREDICTION** (Confidence: {confidence*100:.0f}%)")
                lines.append(f"All three models (ELO, Poisson, and Market) agree: {agreed_outcome} is the most likely outcome.")
                lines.append("Historical accuracy when models agree at this confidence level: ~65-70%")
                lines.append("This is a STRONG signal - consider this a reliable prediction.")
            elif confidence >= 0.4:
                lines.append(f"**MEDIUM CONFIDENCE PREDICTION** (Confidence: {confidence*100:.0f}%)")
                lines.append(f"All three models agree on {agreed_outcome}, but with moderate conviction.")
                lines.append("Historical accuracy when models agree: ~57%")
                lines.append("This is a reasonable prediction but not a certainty.")
            else:
                lines.append(f"**LOW CONFIDENCE PREDICTION** (Confidence: {confidence*100:.0f}%)")
                lines.append(f"Models technically agree on {agreed_outcome}, but with weak conviction.")
                lines.append("The probabilities are close - this could go any way.")
        else:
            lines.append("**MODELS DISAGREE - UNCERTAIN PREDICTION**")
            lines.append("The ELO, Poisson, and Market models do NOT agree on the most likely outcome.")
            lines.append("Historical accuracy when models disagree: only ~41%")
            lines.append("WARNING: This match is difficult to predict. Exercise caution with any bets.")

            # Show what each model favors
            outcomes = ["Home Win", "Draw", "Away Win"]
            elo_fav = outcomes[elo_probs.index(max(elo_probs))]
            poisson_fav = outcomes[poisson_probs.index(max(poisson_probs))]
            market_fav = outcomes[market_probs.index(max(market_probs))]

            lines.append(f"  - ELO favors: {elo_fav}")
            lines.append(f"  - Poisson favors: {poisson_fav}")
            lines.append(f"  - Market favors: {market_fav}")

        return "\n".join(lines)

    def _format_value_edge(
        self,
        predictions: dict[str, float],
        odds: dict[str, float] | None,
        value_bets: list[dict] | None,
    ) -> str:
        """Format value edge analysis for the prompt.

        Args:
            predictions: Model probabilities (home_win, draw, away_win)
            odds: Bookmaker odds (home_odds, draw_odds, away_odds)
            value_bets: Detected value bets for this match

        Returns:
            Formatted string describing value edges
        """
        lines = []

        # If we have value bets detected, highlight them
        if value_bets:
            for vb in value_bets:
                market = vb.get("market", "Unknown")
                edge = vb.get("edge", 0) * 100  # Convert to percentage
                model_prob = vb.get("model_prob", 0) * 100
                implied_prob = vb.get("implied_prob", 0) * 100
                odds_val = vb.get("odds", 0)
                lines.append(
                    f"- VALUE DETECTED: {market} @ {odds_val:.2f} odds "
                    f"(Model: {model_prob:.1f}%, Bookmaker: {implied_prob:.1f}%, Edge: +{edge:.1f}%)"
                )

        # Compare model vs odds even if no official value bet
        if odds and not value_bets:
            model_home = predictions.get("home_win", 0.33)
            model_draw = predictions.get("draw", 0.33)
            model_away = predictions.get("away_win", 0.33)

            home_odds = odds.get("home_odds", 0)
            draw_odds = odds.get("draw_odds", 0)
            away_odds = odds.get("away_odds", 0)

            if home_odds > 0:
                implied_home = 1 / home_odds
                edge_home = (model_home - implied_home) * 100
                if abs(edge_home) > 2:  # Only mention if > 2% difference
                    lines.append(f"- Home win: Model {model_home*100:.1f}% vs Implied {implied_home*100:.1f}% (Edge: {edge_home:+.1f}%)")

            if draw_odds > 0:
                implied_draw = 1 / draw_odds
                edge_draw = (model_draw - implied_draw) * 100
                if abs(edge_draw) > 2:
                    lines.append(f"- Draw: Model {model_draw*100:.1f}% vs Implied {implied_draw*100:.1f}% (Edge: {edge_draw:+.1f}%)")

            if away_odds > 0:
                implied_away = 1 / away_odds
                edge_away = (model_away - implied_away) * 100
                if abs(edge_away) > 2:
                    lines.append(f"- Away win: Model {model_away*100:.1f}% vs Implied {implied_away*100:.1f}% (Edge: {edge_away:+.1f}%)")

        if not lines:
            return "No significant edge detected - odds appear fairly priced."

        return "\n".join(lines)

    def _generate_placeholder(
        self,
        match_data: dict[str, Any],
        predictions: dict[str, float],
    ) -> str:
        """Generate a placeholder narrative when API is unavailable."""
        home = match_data["home_team"]
        away = match_data["away_team"]
        home_prob = round(predictions.get("home_win", 0.33) * 100, 1)
        draw_prob = round(predictions.get("draw", 0.33) * 100, 1)
        away_prob = round(predictions.get("away_win", 0.33) * 100, 1)

        return f"""{home} host {away} in this Premier League fixture.

Our model gives {home} a {home_prob}% chance of victory, with the draw at {draw_prob}% and {away} at {away_prob}%.

Check back later for a full AI-generated preview with form analysis, tactical insights, and betting angles."""


async def generate_batch_narratives(
    matches: list[dict],
    batch_size: int = 5,
) -> dict[int, str]:
    """Generate narratives for multiple matches.

    Args:
        matches: List of match dicts with all required data
        batch_size: Number of concurrent API calls

    Returns:
        Dict mapping match_id to narrative text
    """
    generator = NarrativeGenerator()
    narratives = {}

    for match in matches:
        try:
            narrative = await generator.generate_match_preview(
                match_data=match,
                home_stats=match.get("home_stats", {}),
                away_stats=match.get("away_stats", {}),
                predictions=match.get("predictions", {}),
                h2h_history=match.get("h2h_history"),
                value_bets=match.get("value_bets"),
                odds=match.get("odds"),
            )
            narratives[match["id"]] = narrative
        except Exception as e:
            logger.error(f"Failed to generate narrative for match {match['id']}", error=str(e))
            narratives[match["id"]] = generator._generate_placeholder(match, match.get("predictions", {}))

    return narratives
