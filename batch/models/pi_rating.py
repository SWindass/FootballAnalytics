"""Pi Rating System for Football Teams.

Implementation based on Constantinou & Fenton (2013):
"Solving the problem of inadequate scoring rules for assessing probabilistic
football forecast models"

The Pi Rating system maintains separate home and away ratings for each team,
updating them based on match results using a diminishing returns function
to handle large goal differences appropriately.

Key features:
- Separate home/away ratings capture venue-specific performance
- Diminishing returns prevents outlier matches from over-influencing ratings
- Learning parameters (lambda, gamma) control adaptation speed
- Rating history tracking for analysis and visualization
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class TeamRating:
    """Stores a team's Pi ratings."""

    home_rating: float = 0.0
    away_rating: float = 0.0

    @property
    def overall(self) -> float:
        """Overall rating is the average of home and away ratings."""
        return (self.home_rating + self.away_rating) / 2

    def copy(self) -> "TeamRating":
        """Create a copy of this rating."""
        return TeamRating(home_rating=self.home_rating, away_rating=self.away_rating)


@dataclass
class RatingSnapshot:
    """A snapshot of a team's rating at a point in time."""

    date: datetime
    team: str
    home_rating: float
    away_rating: float
    overall_rating: float
    match_number: int = 0


@dataclass
class MatchPrediction:
    """Prediction for a match."""

    home_team: str
    away_team: str
    predicted_goal_diff: float  # Positive = home win expected
    home_rating: float
    away_rating: float
    confidence: float = 0.0  # Based on rating reliability


class PiRating:
    """Pi Rating system for football team strength estimation.

    The Pi Rating system maintains four ratings that get updated after each match:
    - Home team's home rating (RHH)
    - Home team's away rating (RHA)
    - Away team's home rating (RAH)
    - Away team's away rating (RAA)

    Parameters
    ----------
    lambda_param : float, default=0.06
        Learning rate for the team playing at home in the match.
        Controls how quickly ratings adapt to new results.

    gamma_param : float, default=0.6
        Learning rate multiplier for the team playing away.
        Away rating updates use lambda * gamma as their learning rate.

    initial_rating : float, default=0.0
        Starting rating for all teams (both home and away components).

    Example
    -------
    >>> pi = PiRating()
    >>> pi.update_ratings("Liverpool", "Arsenal", 3, 1, datetime(2024, 1, 1))
    >>> prediction = pi.predict_match("Liverpool", "Chelsea")
    >>> print(f"Expected goal diff: {prediction.predicted_goal_diff:.2f}")
    """

    def __init__(
        self,
        lambda_param: float = 0.06,
        gamma_param: float = 0.6,
        initial_rating: float = 0.0,
    ):
        self.lambda_param = lambda_param
        self.gamma_param = gamma_param
        self.initial_rating = initial_rating

        # Team ratings: team_name -> TeamRating
        self._ratings: dict[str, TeamRating] = {}

        # Rating history for analysis
        self._history: list[RatingSnapshot] = []

        # Match count for tracking
        self._match_count = 0

        # Prediction history for MSE calculation
        self._predictions: list[dict] = []

    def _get_or_create_rating(self, team: str) -> TeamRating:
        """Get a team's rating, creating it with initial values if needed."""
        if team not in self._ratings:
            self._ratings[team] = TeamRating(
                home_rating=self.initial_rating,
                away_rating=self.initial_rating,
            )
        return self._ratings[team]

    def calculate_expected_goal_diff(
        self, home_team: str, away_team: str
    ) -> float:
        """Calculate expected goal difference from ratings.

        The expected goal difference is calculated as:
        E[GD] = RHH - RAA

        Where:
        - RHH = Home team's home rating
        - RAA = Away team's away rating

        A positive value indicates expected home win.

        Parameters
        ----------
        home_team : str
            Name of the home team.
        away_team : str
            Name of the away team.

        Returns
        -------
        float
            Expected goal difference (positive = home advantage).
        """
        home_rating = self._get_or_create_rating(home_team)
        away_rating = self._get_or_create_rating(away_team)

        return home_rating.home_rating - away_rating.away_rating

    def predict_match(self, home_team: str, away_team: str) -> MatchPrediction:
        """Predict the outcome of a match.

        Parameters
        ----------
        home_team : str
            Name of the home team.
        away_team : str
            Name of the away team.

        Returns
        -------
        MatchPrediction
            Prediction object with expected goal difference and ratings.
        """
        home_r = self._get_or_create_rating(home_team)
        away_r = self._get_or_create_rating(away_team)

        predicted_gd = self.calculate_expected_goal_diff(home_team, away_team)

        # Confidence based on how many matches we've seen these teams play
        # (Simple heuristic - could be improved)
        confidence = min(1.0, self._match_count / 100)

        return MatchPrediction(
            home_team=home_team,
            away_team=away_team,
            predicted_goal_diff=predicted_gd,
            home_rating=home_r.overall,
            away_rating=away_r.overall,
            confidence=confidence,
        )

    @staticmethod
    def calculate_error(predicted_gd: float, actual_gd: float) -> float:
        """Calculate prediction error.

        Parameters
        ----------
        predicted_gd : float
            Predicted goal difference.
        actual_gd : float
            Actual goal difference.

        Returns
        -------
        float
            Error (actual - predicted).
        """
        return actual_gd - predicted_gd

    @staticmethod
    def apply_diminishing_returns(error: float) -> float:
        """Apply diminishing returns function to error.

        Uses sqrt(|e|) * sign(e) to reduce the impact of large goal
        differences. This prevents outlier results (e.g., 7-0) from
        having disproportionate influence on ratings.

        Parameters
        ----------
        error : float
            Raw prediction error.

        Returns
        -------
        float
            Adjusted error with diminishing returns applied.
        """
        if error == 0:
            return 0.0
        return np.sqrt(abs(error)) * np.sign(error)

    def update_ratings(
        self,
        home_team: str,
        away_team: str,
        home_goals: int,
        away_goals: int,
        match_date: Optional[datetime] = None,
        store_history: bool = True,
    ) -> tuple[float, float]:
        """Update ratings based on match result.

        Updates all four relevant ratings:
        - Home team's home rating (RHH)
        - Home team's away rating (RHA)
        - Away team's home rating (RAH)
        - Away team's away rating (RAA)

        The update formulas are:
        - RHH_new = RHH + λ * ψ(e)
        - RHA_new = RHA + λ * γ * ψ(e)
        - RAA_new = RAA - λ * ψ(e)
        - RAH_new = RAH - λ * γ * ψ(e)

        Where ψ(e) is the diminishing returns function.

        Parameters
        ----------
        home_team : str
            Name of the home team.
        away_team : str
            Name of the away team.
        home_goals : int
            Goals scored by home team.
        away_goals : int
            Goals scored by away team.
        match_date : datetime, optional
            Date of the match (for history tracking).
        store_history : bool, default=True
            Whether to store rating snapshots after update.

        Returns
        -------
        tuple[float, float]
            (predicted_goal_diff, actual_goal_diff)
        """
        # Get current ratings
        home_r = self._get_or_create_rating(home_team)
        away_r = self._get_or_create_rating(away_team)

        # Calculate prediction and error
        predicted_gd = self.calculate_expected_goal_diff(home_team, away_team)
        actual_gd = home_goals - away_goals
        error = self.calculate_error(predicted_gd, actual_gd)

        # Apply diminishing returns
        adjusted_error = self.apply_diminishing_returns(error)

        # Calculate update amounts
        home_venue_update = self.lambda_param * adjusted_error
        away_venue_update = self.lambda_param * self.gamma_param * adjusted_error

        # Update home team ratings (positive direction for positive error)
        home_r.home_rating += home_venue_update
        home_r.away_rating += away_venue_update

        # Update away team ratings (negative direction)
        away_r.away_rating -= home_venue_update
        away_r.home_rating -= away_venue_update

        self._match_count += 1

        # Store prediction for MSE calculation
        self._predictions.append({
            "date": match_date,
            "home_team": home_team,
            "away_team": away_team,
            "predicted_gd": predicted_gd,
            "actual_gd": actual_gd,
            "error": error,
            "squared_error": error ** 2,
        })

        # Store history snapshots
        if store_history and match_date is not None:
            self._history.append(RatingSnapshot(
                date=match_date,
                team=home_team,
                home_rating=home_r.home_rating,
                away_rating=home_r.away_rating,
                overall_rating=home_r.overall,
                match_number=self._match_count,
            ))
            self._history.append(RatingSnapshot(
                date=match_date,
                team=away_team,
                home_rating=away_r.home_rating,
                away_rating=away_r.away_rating,
                overall_rating=away_r.overall,
                match_number=self._match_count,
            ))

        return predicted_gd, actual_gd

    def process_matches(
        self,
        matches: pd.DataFrame,
        date_col: str = "date",
        home_team_col: str = "home_team",
        away_team_col: str = "away_team",
        home_goals_col: str = "home_goals",
        away_goals_col: str = "away_goals",
        sort_by_date: bool = True,
    ) -> "PiRating":
        """Process multiple matches to build ratings.

        Parameters
        ----------
        matches : pd.DataFrame
            DataFrame containing match data.
        date_col : str
            Column name for match date.
        home_team_col : str
            Column name for home team.
        away_team_col : str
            Column name for away team.
        home_goals_col : str
            Column name for home goals.
        away_goals_col : str
            Column name for away goals.
        sort_by_date : bool, default=True
            Whether to sort matches chronologically before processing.

        Returns
        -------
        PiRating
            Self, for method chaining.
        """
        df = matches.copy()

        if sort_by_date:
            df = df.sort_values(date_col)

        for _, row in df.iterrows():
            match_date = row[date_col]
            if isinstance(match_date, str):
                match_date = pd.to_datetime(match_date)

            self.update_ratings(
                home_team=row[home_team_col],
                away_team=row[away_team_col],
                home_goals=int(row[home_goals_col]),
                away_goals=int(row[away_goals_col]),
                match_date=match_date,
            )

        return self

    def get_ratings_dataframe(self) -> pd.DataFrame:
        """Get current ratings as a DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: team, home_rating, away_rating, overall_rating
        """
        data = []
        for team, rating in sorted(self._ratings.items(), key=lambda x: -x[1].overall):
            data.append({
                "team": team,
                "home_rating": round(rating.home_rating, 4),
                "away_rating": round(rating.away_rating, 4),
                "overall_rating": round(rating.overall, 4),
            })
        return pd.DataFrame(data)

    def get_history_dataframe(self) -> pd.DataFrame:
        """Get rating history as a DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: date, team, home_rating, away_rating,
            overall_rating, match_number
        """
        if not self._history:
            return pd.DataFrame()

        return pd.DataFrame([
            {
                "date": s.date,
                "team": s.team,
                "home_rating": round(s.home_rating, 4),
                "away_rating": round(s.away_rating, 4),
                "overall_rating": round(s.overall_rating, 4),
                "match_number": s.match_number,
            }
            for s in self._history
        ])

    def get_predictions_dataframe(self) -> pd.DataFrame:
        """Get prediction history as a DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with prediction details and errors.
        """
        if not self._predictions:
            return pd.DataFrame()
        return pd.DataFrame(self._predictions)

    def calculate_mse(self) -> float:
        """Calculate Mean Squared Error for all predictions.

        Returns
        -------
        float
            Mean squared error of predictions.
        """
        if not self._predictions:
            return 0.0
        return np.mean([p["squared_error"] for p in self._predictions])

    def calculate_mae(self) -> float:
        """Calculate Mean Absolute Error for all predictions.

        Returns
        -------
        float
            Mean absolute error of predictions.
        """
        if not self._predictions:
            return 0.0
        return np.mean([abs(p["error"]) for p in self._predictions])

    def get_team_rating(self, team: str) -> Optional[TeamRating]:
        """Get a specific team's current rating.

        Parameters
        ----------
        team : str
            Team name.

        Returns
        -------
        TeamRating or None
            Team's rating if found, None otherwise.
        """
        return self._ratings.get(team)

    def get_ratings_at_date(self, target_date: datetime) -> dict[str, TeamRating]:
        """Get all team ratings as of a specific date.

        Parameters
        ----------
        target_date : datetime
            Date to get ratings for.

        Returns
        -------
        dict[str, TeamRating]
            Dictionary mapping team names to their ratings at that date.
        """
        ratings: dict[str, TeamRating] = {}

        # Filter history up to target date
        relevant_history = [
            s for s in self._history
            if s.date is not None and s.date <= target_date
        ]

        # Get most recent rating for each team
        for snapshot in sorted(relevant_history, key=lambda x: x.date):
            ratings[snapshot.team] = TeamRating(
                home_rating=snapshot.home_rating,
                away_rating=snapshot.away_rating,
            )

        return ratings

    def reset(self) -> None:
        """Reset all ratings and history."""
        self._ratings.clear()
        self._history.clear()
        self._predictions.clear()
        self._match_count = 0

    def copy(self) -> "PiRating":
        """Create a deep copy of this PiRating instance."""
        new_pi = PiRating(
            lambda_param=self.lambda_param,
            gamma_param=self.gamma_param,
            initial_rating=self.initial_rating,
        )
        new_pi._ratings = {k: v.copy() for k, v in self._ratings.items()}
        new_pi._history = self._history.copy()
        new_pi._predictions = self._predictions.copy()
        new_pi._match_count = self._match_count
        return new_pi


def optimize_parameters(
    matches: pd.DataFrame,
    lambda_range: tuple[float, float] = (0.01, 0.15),
    gamma_range: tuple[float, float] = (0.3, 0.9),
    n_trials: int = 50,
    metric: str = "mse",
    date_col: str = "date",
    home_team_col: str = "home_team",
    away_team_col: str = "away_team",
    home_goals_col: str = "home_goals",
    away_goals_col: str = "away_goals",
) -> dict:
    """Find optimal lambda and gamma parameters using grid search.

    Parameters
    ----------
    matches : pd.DataFrame
        Historical match data.
    lambda_range : tuple[float, float]
        Range for lambda parameter search.
    gamma_range : tuple[float, float]
        Range for gamma parameter search.
    n_trials : int
        Number of parameter combinations to try.
    metric : str
        Metric to optimize ('mse' or 'mae').
    date_col, home_team_col, away_team_col, home_goals_col, away_goals_col : str
        Column names in the matches DataFrame.

    Returns
    -------
    dict
        Dictionary with optimal parameters and their performance.
    """
    best_result = {
        "lambda": 0.06,
        "gamma": 0.6,
        "mse": float("inf"),
        "mae": float("inf"),
    }

    # Generate parameter grid
    lambdas = np.linspace(lambda_range[0], lambda_range[1], int(np.sqrt(n_trials)))
    gammas = np.linspace(gamma_range[0], gamma_range[1], int(np.sqrt(n_trials)))

    results = []

    for lam in lambdas:
        for gam in gammas:
            pi = PiRating(lambda_param=lam, gamma_param=gam)
            pi.process_matches(
                matches,
                date_col=date_col,
                home_team_col=home_team_col,
                away_team_col=away_team_col,
                home_goals_col=home_goals_col,
                away_goals_col=away_goals_col,
            )

            mse = pi.calculate_mse()
            mae = pi.calculate_mae()

            results.append({
                "lambda": lam,
                "gamma": gam,
                "mse": mse,
                "mae": mae,
            })

            target_metric = mse if metric == "mse" else mae
            best_metric = best_result["mse"] if metric == "mse" else best_result["mae"]

            if target_metric < best_metric:
                best_result = {
                    "lambda": lam,
                    "gamma": gam,
                    "mse": mse,
                    "mae": mae,
                }

    return {
        "best_params": best_result,
        "all_results": pd.DataFrame(results),
    }


def plot_rating_history(
    pi_rating: PiRating,
    teams: list[str],
    figsize: tuple[int, int] = (12, 6),
    title: str = "Pi Rating Development Over Time",
):
    """Plot rating development over time for selected teams.

    Parameters
    ----------
    pi_rating : PiRating
        PiRating instance with history.
    teams : list[str]
        List of team names to plot.
    figsize : tuple[int, int]
        Figure size.
    title : str
        Plot title.

    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib figure object.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for plotting")

    history_df = pi_rating.get_history_dataframe()
    if history_df.empty:
        raise ValueError("No history data available")

    fig, ax = plt.subplots(figsize=figsize)

    for team in teams:
        team_history = history_df[history_df["team"] == team].sort_values("date")
        if not team_history.empty:
            ax.plot(
                team_history["date"],
                team_history["overall_rating"],
                label=team,
                linewidth=2,
            )

    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Date")
    ax.set_ylabel("Overall Rating")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# Example usage
if __name__ == "__main__":
    # Create sample match data
    sample_matches = pd.DataFrame([
        {"date": "2024-01-01", "home_team": "Liverpool", "away_team": "Arsenal", "home_goals": 3, "away_goals": 1},
        {"date": "2024-01-08", "home_team": "Chelsea", "away_team": "Liverpool", "home_goals": 1, "away_goals": 2},
        {"date": "2024-01-15", "home_team": "Arsenal", "away_team": "Chelsea", "home_goals": 2, "away_goals": 0},
        {"date": "2024-01-22", "home_team": "Liverpool", "away_team": "Chelsea", "home_goals": 4, "away_goals": 1},
        {"date": "2024-01-29", "home_team": "Arsenal", "away_team": "Liverpool", "home_goals": 1, "away_goals": 1},
        {"date": "2024-02-05", "home_team": "Chelsea", "away_team": "Arsenal", "home_goals": 0, "away_goals": 2},
    ])

    # Initialize and process matches
    pi = PiRating(lambda_param=0.06, gamma_param=0.6)
    pi.process_matches(sample_matches)

    # Display current ratings
    print("Current Team Ratings:")
    print("=" * 60)
    ratings_df = pi.get_ratings_dataframe()
    print(ratings_df.to_string(index=False))

    # Make a prediction
    print("\nMatch Prediction:")
    print("=" * 60)
    prediction = pi.predict_match("Liverpool", "Arsenal")
    print(f"Liverpool (H) vs Arsenal (A)")
    print(f"Expected goal difference: {prediction.predicted_goal_diff:.2f}")
    print(f"(Positive = Liverpool expected to win)")

    # Show prediction accuracy
    print(f"\nModel Performance:")
    print("=" * 60)
    print(f"Mean Squared Error: {pi.calculate_mse():.4f}")
    print(f"Mean Absolute Error: {pi.calculate_mae():.4f}")

    # Show rating history
    print("\nRating History (last 5 updates):")
    print("=" * 60)
    history_df = pi.get_history_dataframe()
    print(history_df.tail(10).to_string(index=False))
