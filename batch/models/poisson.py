"""Poisson distribution model for goal prediction."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.stats import poisson


@dataclass
class PoissonConfig:
    """Configuration for Poisson model."""

    home_advantage: float = 0.25  # Goals added to home team's expected
    max_goals: int = 10  # Maximum goals to consider in probability matrix
    league_avg_goals: float = 2.75  # Average goals per game in league


class PoissonModel:
    """Poisson distribution model for predicting match goals.

    Uses team attack/defense strengths relative to league average
    to estimate expected goals, then calculates outcome probabilities.
    """

    def __init__(self, config: Optional[PoissonConfig] = None):
        self.config = config or PoissonConfig()

    def calculate_expected_goals(
        self,
        home_attack: float,
        home_defense: float,
        away_attack: float,
        away_defense: float,
    ) -> tuple[float, float]:
        """Calculate expected goals for each team.

        Args:
            home_attack: Home team's attack strength (goals scored / league avg)
            home_defense: Home team's defense strength (goals conceded / league avg)
            away_attack: Away team's attack strength
            away_defense: Away team's defense strength

        Returns:
            Tuple of (home_expected_goals, away_expected_goals)
        """
        league_avg = self.config.league_avg_goals / 2  # Per team

        # Home team expected = home_attack * away_defense * league_avg + home_advantage
        home_expected = (
            home_attack * away_defense * league_avg + self.config.home_advantage
        )

        # Away team expected = away_attack * home_defense * league_avg
        away_expected = away_attack * home_defense * league_avg

        # Ensure positive values
        return max(0.1, home_expected), max(0.1, away_expected)

    def score_probability_matrix(
        self,
        home_expected: float,
        away_expected: float,
    ) -> np.ndarray:
        """Calculate probability matrix for all score combinations.

        Args:
            home_expected: Expected goals for home team
            away_expected: Expected goals for away team

        Returns:
            2D numpy array where [i,j] = P(home=i, away=j)
        """
        max_goals = self.config.max_goals

        # Calculate probabilities for each goal count
        home_probs = poisson.pmf(range(max_goals + 1), home_expected)
        away_probs = poisson.pmf(range(max_goals + 1), away_expected)

        # Create probability matrix
        prob_matrix = np.outer(home_probs, away_probs)

        return prob_matrix

    def match_probabilities(
        self,
        home_expected: float,
        away_expected: float,
    ) -> tuple[float, float, float]:
        """Calculate match outcome probabilities.

        Args:
            home_expected: Expected goals for home team
            away_expected: Expected goals for away team

        Returns:
            Tuple of (home_win_prob, draw_prob, away_win_prob)
        """
        prob_matrix = self.score_probability_matrix(home_expected, away_expected)

        # Sum probabilities for each outcome
        home_win = np.sum(np.tril(prob_matrix, -1))  # Below diagonal
        draw = np.sum(np.diag(prob_matrix))  # Diagonal
        away_win = np.sum(np.triu(prob_matrix, 1))  # Above diagonal

        # Normalize (should be very close to 1 already)
        total = home_win + draw + away_win
        return home_win / total, draw / total, away_win / total

    def over_under_probability(
        self,
        home_expected: float,
        away_expected: float,
        line: float = 2.5,
    ) -> tuple[float, float]:
        """Calculate over/under probability for a given line.

        Args:
            home_expected: Expected goals for home team
            away_expected: Expected goals for away team
            line: Goals line (e.g., 2.5)

        Returns:
            Tuple of (over_prob, under_prob)
        """
        prob_matrix = self.score_probability_matrix(home_expected, away_expected)

        under_prob = 0.0
        for i in range(int(line) + 1):
            for j in range(int(line) + 1 - i):
                under_prob += prob_matrix[i, j]

        over_prob = 1 - under_prob
        return over_prob, under_prob

    def btts_probability(
        self,
        home_expected: float,
        away_expected: float,
    ) -> tuple[float, float]:
        """Calculate both teams to score probability.

        Args:
            home_expected: Expected goals for home team
            away_expected: Expected goals for away team

        Returns:
            Tuple of (btts_yes_prob, btts_no_prob)
        """
        prob_matrix = self.score_probability_matrix(home_expected, away_expected)

        # BTTS No = P(home=0) + P(away=0) - P(0-0)
        btts_no = prob_matrix[0, :].sum() + prob_matrix[:, 0].sum() - prob_matrix[0, 0]
        btts_yes = 1 - btts_no

        return btts_yes, btts_no

    def most_likely_scores(
        self,
        home_expected: float,
        away_expected: float,
        top_n: int = 5,
    ) -> list[tuple[int, int, float]]:
        """Get most likely exact scores.

        Args:
            home_expected: Expected goals for home team
            away_expected: Expected goals for away team
            top_n: Number of scores to return

        Returns:
            List of (home_goals, away_goals, probability) tuples
        """
        prob_matrix = self.score_probability_matrix(home_expected, away_expected)

        # Flatten and get top indices
        flat_indices = np.argsort(prob_matrix.ravel())[::-1][:top_n]

        scores = []
        for idx in flat_indices:
            home_goals, away_goals = divmod(idx, self.config.max_goals + 1)
            prob = prob_matrix[home_goals, away_goals]
            scores.append((home_goals, away_goals, prob))

        return scores


def calculate_team_strengths(
    matches: list[dict],
    league_avg_scored: float,
    league_avg_conceded: float,
) -> dict[int, tuple[float, float]]:
    """Calculate attack and defense strengths for all teams.

    Args:
        matches: List of completed match dicts with:
            - home_team_id: int
            - away_team_id: int
            - home_score: int
            - away_score: int
        league_avg_scored: League average goals scored per team per game
        league_avg_conceded: League average goals conceded per team per game

    Returns:
        Dict mapping team_id to (attack_strength, defense_strength)
    """
    # Aggregate stats per team
    team_stats: dict[int, dict] = {}

    for match in matches:
        home_id = match["home_team_id"]
        away_id = match["away_team_id"]
        home_score = match.get("home_score")
        away_score = match.get("away_score")

        if home_score is None or away_score is None:
            continue

        # Initialize team stats if needed
        for team_id in [home_id, away_id]:
            if team_id not in team_stats:
                team_stats[team_id] = {
                    "scored": 0,
                    "conceded": 0,
                    "games": 0,
                }

        # Update home team
        team_stats[home_id]["scored"] += home_score
        team_stats[home_id]["conceded"] += away_score
        team_stats[home_id]["games"] += 1

        # Update away team
        team_stats[away_id]["scored"] += away_score
        team_stats[away_id]["conceded"] += home_score
        team_stats[away_id]["games"] += 1

    # Calculate strengths
    strengths = {}
    for team_id, stats in team_stats.items():
        if stats["games"] == 0:
            strengths[team_id] = (1.0, 1.0)
            continue

        avg_scored = stats["scored"] / stats["games"]
        avg_conceded = stats["conceded"] / stats["games"]

        attack = avg_scored / league_avg_scored if league_avg_scored > 0 else 1.0
        defense = avg_conceded / league_avg_conceded if league_avg_conceded > 0 else 1.0

        strengths[team_id] = (attack, defense)

    return strengths
