"""AI narrative generation using Claude API."""

from datetime import datetime
from typing import Any, Optional

import anthropic
import structlog

from app.core.config import get_settings

logger = structlog.get_logger()
settings = get_settings()


MATCH_PREVIEW_PROMPT = """You are a football analyst writing a match preview for the English Premier League.

Write a concise, engaging match preview (150-200 words) for the following fixture:

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
- Home win: {home_prob}%
- Draw: {draw_prob}%
- Away win: {away_prob}%
- Predicted score: {predicted_score}

**Notable absences:**
- {home_team}: {home_injuries}
- {away_team}: {away_injuries}

Write the preview focusing on:
1. Recent form and momentum
2. Key tactical matchups
3. Statistical insights
4. A brief betting angle (mention the model's view and any value)

Keep the tone professional but engaging. Do not use clichés like "six-pointer" or "must-win game" unless truly appropriate."""


class NarrativeGenerator:
    """Generates AI match narratives using Claude API."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.anthropic_api_key
        self.client = anthropic.Anthropic(api_key=self.api_key) if self.api_key else None

    async def generate_match_preview(
        self,
        match_data: dict[str, Any],
        home_stats: dict[str, Any],
        away_stats: dict[str, Any],
        predictions: dict[str, float],
        h2h_history: Optional[list[dict]] = None,
    ) -> str:
        """Generate a match preview narrative.

        Args:
            match_data: Match information (teams, kickoff, venue)
            home_stats: Home team statistics
            away_stats: Away team statistics
            predictions: Model predictions (home_prob, draw_prob, away_prob)
            h2h_history: Optional head-to-head history

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
            )
            narratives[match["id"]] = narrative
        except Exception as e:
            logger.error(f"Failed to generate narrative for match {match['id']}", error=str(e))
            narratives[match["id"]] = generator._generate_placeholder(match, match.get("predictions", {}))

    return narratives
