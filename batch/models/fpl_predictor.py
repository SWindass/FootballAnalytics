"""FPL-based match predictor.

Uses Fantasy Premier League player data to:
1. Calculate team attack/defense strength
2. Estimate expected goals
3. Generate match outcome probabilities

Combines well with Pi Rating and ELO for ensemble predictions.
"""

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import poisson


@dataclass
class TeamFPLMetrics:
    """FPL-derived metrics for a team."""

    # Attack metrics (goals, assists, xG)
    attack_score: float = 0.0  # Weighted FPL points from attackers
    attack_form: float = 0.0  # Recent vs season average
    expected_goals: float = 0.0

    # Defense metrics (clean sheets, goals conceded)
    defense_score: float = 0.0  # Weighted FPL points from defenders
    defense_form: float = 0.0
    expected_goals_against: float = 0.0

    # Overall
    total_fpl_points: float = 0.0
    form_index: float = 1.0  # >1 = above average form

    # Missing player impact
    missing_impact: float = 0.0


@dataclass
class FPLMatchPrediction:
    """FPL-based match prediction."""

    home_xg: float
    away_xg: float
    home_prob: float
    draw_prob: float
    away_prob: float

    # Score matrix
    score_matrix: np.ndarray | None = None

    # Confidence based on data quality
    confidence: float = 0.0


class FPLPredictor:
    """Match predictor using FPL player statistics.

    Calculates team strength from aggregated player FPL data,
    then uses Poisson model to predict match outcomes.

    Parameters
    ----------
    form_window : int
        Number of recent gameweeks for form calculation.
    rho : float
        Dixon-Coles correlation parameter.
    league_avg_goals : float
        Average goals per team per match.
    """

    def __init__(
        self,
        form_window: int = 5,
        rho: float = -0.11,
        league_avg_goals: float = 1.35,
    ):
        self.form_window = form_window
        self.rho = rho
        self.league_avg_goals = league_avg_goals

        # Team FPL data: team -> {season -> {gameweek -> metrics}}
        self._team_data: dict[str, dict[str, dict[int, dict]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(dict))
        )

        # Player data cache: player_id -> recent performances
        self._player_cache: dict[int, list[dict]] = defaultdict(list)

        # League averages by season
        self._league_avgs: dict[str, dict] = {}

        # Position weights for attack/defense scoring
        self.position_weights = {
            "attack": {"FWD": 1.0, "MID": 0.7, "DEF": 0.2, "GKP": 0.0},
            "defense": {"FWD": 0.0, "MID": 0.3, "DEF": 1.0, "GKP": 0.8},
        }

    def load_fpl_data(self, performances_df: pd.DataFrame, players_df: pd.DataFrame):
        """Load FPL performance data.

        Parameters
        ----------
        performances_df : pd.DataFrame
            PlayerMatchPerformance data with columns:
            player_id, season, gameweek, total_points, goals_scored,
            assists, clean_sheets, minutes, influence, creativity, threat
        players_df : pd.DataFrame
            Player info with columns: id, team_id, position, name
        """
        # Build player -> team mapping
        player_teams = {}
        player_positions = {}
        player_names = {}

        for _, p in players_df.iterrows():
            player_teams[p["id"]] = p.get("team_name", "Unknown")
            player_positions[p["id"]] = p.get("position", "MID")
            player_names[p["id"]] = p.get("name", "Unknown")

        # Process performances
        for _, perf in performances_df.iterrows():
            player_id = perf["player_id"]
            team = player_teams.get(player_id, "Unknown")
            season = perf["season"]
            gw = perf["gameweek"]
            position = player_positions.get(player_id, "MID")

            if team == "Unknown":
                continue

            # Store performance
            perf_data = {
                "player_id": player_id,
                "player_name": player_names.get(player_id, "Unknown"),
                "position": position,
                "total_points": perf.get("total_points", 0) or 0,
                "goals_scored": perf.get("goals_scored", 0) or 0,
                "assists": perf.get("assists", 0) or 0,
                "clean_sheets": perf.get("clean_sheets", 0) or 0,
                "minutes": perf.get("minutes", 0) or 0,
                "influence": float(perf.get("influence", 0) or 0),
                "creativity": float(perf.get("creativity", 0) or 0),
                "threat": float(perf.get("threat", 0) or 0),
                "ict_index": float(perf.get("ict_index", 0) or 0),
                "expected_goals": float(perf.get("expected_goals", 0) or 0),
                "expected_assists": float(perf.get("expected_assists", 0) or 0),
            }

            # Add to team data
            if gw not in self._team_data[team][season]:
                self._team_data[team][season][gw] = {
                    "players": [],
                    "attack_score": 0,
                    "defense_score": 0,
                    "total_points": 0,
                    "goals": 0,
                    "xg": 0,
                }

            self._team_data[team][season][gw]["players"].append(perf_data)

            # Aggregate team metrics
            att_weight = self.position_weights["attack"].get(position, 0.5)
            def_weight = self.position_weights["defense"].get(position, 0.5)

            # Attack score: goals, assists, threat, xG
            attack_contrib = (
                perf_data["goals_scored"] * 6 +
                perf_data["assists"] * 3 +
                perf_data["threat"] * 0.01 +
                perf_data["expected_goals"] * 4
            ) * att_weight

            # Defense score: clean sheets, influence
            defense_contrib = (
                perf_data["clean_sheets"] * 4 +
                perf_data["influence"] * 0.01
            ) * def_weight

            self._team_data[team][season][gw]["attack_score"] += attack_contrib
            self._team_data[team][season][gw]["defense_score"] += defense_contrib
            self._team_data[team][season][gw]["total_points"] += perf_data["total_points"]
            self._team_data[team][season][gw]["goals"] += perf_data["goals_scored"]
            self._team_data[team][season][gw]["xg"] += perf_data["expected_goals"]

        # Calculate league averages
        self._calculate_league_averages()

        print(f"Loaded FPL data for {len(self._team_data)} teams")

    def _calculate_league_averages(self):
        """Calculate league average attack/defense scores by season."""
        season_data = defaultdict(lambda: {"attack": [], "defense": [], "goals": []})

        for _team, seasons in self._team_data.items():
            for season, gws in seasons.items():
                for _gw, data in gws.items():
                    season_data[season]["attack"].append(data["attack_score"])
                    season_data[season]["defense"].append(data["defense_score"])
                    season_data[season]["goals"].append(data["goals"])

        for season, data in season_data.items():
            self._league_avgs[season] = {
                "attack": np.mean(data["attack"]) if data["attack"] else 10,
                "defense": np.mean(data["defense"]) if data["defense"] else 5,
                "goals": np.mean(data["goals"]) if data["goals"] else 1.35,
            }

    def get_team_metrics(
        self,
        team: str,
        season: str,
        gameweek: int,
    ) -> TeamFPLMetrics:
        """Get FPL metrics for a team at a specific point.

        Uses data from previous gameweeks (not current) to avoid lookahead.

        Parameters
        ----------
        team : str
            Team name.
        season : str
            Season (e.g., "2023-24").
        gameweek : int
            Current gameweek (will use data from prior GWs).

        Returns
        -------
        TeamFPLMetrics
            Team's FPL-derived metrics.
        """
        team_seasons = self._team_data.get(team, {})
        season_data = team_seasons.get(season, {})

        if not season_data:
            # No data for this team/season
            return TeamFPLMetrics()

        # Get recent gameweeks (before current)
        recent_gws = [
            gw for gw in sorted(season_data.keys())
            if gw < gameweek
        ][-self.form_window:]

        if not recent_gws:
            # Try previous season
            return self._get_from_previous_season(team, season)

        # Calculate metrics from recent data
        attack_scores = []
        defense_scores = []
        total_points = []
        goals = []
        xg = []

        for gw in recent_gws:
            gw_data = season_data[gw]
            attack_scores.append(gw_data["attack_score"])
            defense_scores.append(gw_data["defense_score"])
            total_points.append(gw_data["total_points"])
            goals.append(gw_data["goals"])
            xg.append(gw_data["xg"])

        # Get season averages for form calculation
        all_gws = [gw for gw in season_data.keys() if gw < gameweek]
        season_attack = np.mean([season_data[gw]["attack_score"] for gw in all_gws]) if all_gws else 10
        season_defense = np.mean([season_data[gw]["defense_score"] for gw in all_gws]) if all_gws else 5

        recent_attack = np.mean(attack_scores)
        recent_defense = np.mean(defense_scores)

        # Form index: recent / season average
        attack_form = recent_attack / season_attack if season_attack > 0 else 1.0
        defense_form = recent_defense / season_defense if season_defense > 0 else 1.0

        # League averages
        league_avg = self._league_avgs.get(season, {"attack": 10, "defense": 5, "goals": 1.35})

        # Expected goals based on attack strength
        attack_ratio = recent_attack / league_avg["attack"] if league_avg["attack"] > 0 else 1.0
        exp_goals = self.league_avg_goals * attack_ratio

        # Expected goals against based on defense (inverse)
        defense_ratio = league_avg["defense"] / recent_defense if recent_defense > 0 else 1.0
        exp_goals_against = self.league_avg_goals * defense_ratio

        return TeamFPLMetrics(
            attack_score=recent_attack,
            attack_form=attack_form,
            expected_goals=min(3.0, max(0.5, exp_goals)),
            defense_score=recent_defense,
            defense_form=defense_form,
            expected_goals_against=min(3.0, max(0.5, exp_goals_against)),
            total_fpl_points=np.mean(total_points),
            form_index=(attack_form + defense_form) / 2,
        )

    def _get_from_previous_season(self, team: str, season: str) -> TeamFPLMetrics:
        """Get metrics from previous season when current unavailable."""
        # Try to find previous season
        try:
            year = int(season.split("-")[0])
            prev_season = f"{year-1}-{str(year)[-2:]}"
        except:
            return TeamFPLMetrics()

        team_seasons = self._team_data.get(team, {})
        season_data = team_seasons.get(prev_season, {})

        if not season_data:
            return TeamFPLMetrics()

        # Use last 10 GWs of previous season
        recent_gws = sorted(season_data.keys())[-10:]

        attack_scores = [season_data[gw]["attack_score"] for gw in recent_gws]
        defense_scores = [season_data[gw]["defense_score"] for gw in recent_gws]

        return TeamFPLMetrics(
            attack_score=np.mean(attack_scores),
            defense_score=np.mean(defense_scores),
            form_index=1.0,  # Reset form for new season
        )

    def predict_match(
        self,
        home_team: str,
        away_team: str,
        season: str,
        gameweek: int,
    ) -> FPLMatchPrediction:
        """Predict match using FPL-derived expected goals.

        Parameters
        ----------
        home_team, away_team : str
            Team names.
        season : str
            Season (e.g., "2023-24").
        gameweek : int
            Gameweek number.

        Returns
        -------
        FPLMatchPrediction
            Prediction with probabilities.
        """
        # Get team metrics
        home_metrics = self.get_team_metrics(home_team, season, gameweek)
        away_metrics = self.get_team_metrics(away_team, season, gameweek)

        # Calculate expected goals
        # Home xG = home attack vs away defense
        home_attack_strength = home_metrics.attack_score / 10 if home_metrics.attack_score else 1.0
        away_defense_weakness = 5 / away_metrics.defense_score if away_metrics.defense_score else 1.0

        # Away xG = away attack vs home defense
        away_attack_strength = away_metrics.attack_score / 10 if away_metrics.attack_score else 1.0
        home_defense_weakness = 5 / home_metrics.defense_score if home_metrics.defense_score else 1.0

        # Base xG with home advantage
        home_xg = self.league_avg_goals * home_attack_strength * away_defense_weakness * 1.1
        away_xg = self.league_avg_goals * away_attack_strength * home_defense_weakness * 0.9

        # Apply form adjustment
        home_xg *= home_metrics.form_index
        away_xg *= away_metrics.form_index

        # Clamp to reasonable range
        home_xg = min(3.5, max(0.5, home_xg))
        away_xg = min(3.0, max(0.3, away_xg))

        # Generate score matrix with Dixon-Coles
        matrix = self._calculate_score_matrix(home_xg, away_xg)

        # Calculate outcome probabilities
        home_prob, draw_prob, away_prob = self._matrix_to_probs(matrix)

        # Confidence based on data availability
        has_home_data = home_metrics.attack_score > 0
        has_away_data = away_metrics.attack_score > 0
        confidence = 0.5 * has_home_data + 0.5 * has_away_data

        return FPLMatchPrediction(
            home_xg=home_xg,
            away_xg=away_xg,
            home_prob=home_prob,
            draw_prob=draw_prob,
            away_prob=away_prob,
            score_matrix=matrix,
            confidence=confidence,
        )

    def _calculate_score_matrix(
        self,
        home_xg: float,
        away_xg: float,
        max_goals: int = 6,
    ) -> np.ndarray:
        """Calculate score probability matrix with Dixon-Coles."""
        matrix = np.zeros((max_goals + 1, max_goals + 1))

        for h in range(max_goals + 1):
            for a in range(max_goals + 1):
                base_prob = poisson.pmf(h, home_xg) * poisson.pmf(a, away_xg)
                correction = self._dixon_coles_correction(h, a, home_xg, away_xg)
                matrix[h, a] = base_prob * correction

        # Normalize
        matrix /= matrix.sum()
        return matrix

    def _dixon_coles_correction(
        self,
        home_goals: int,
        away_goals: int,
        home_xg: float,
        away_xg: float,
    ) -> float:
        """Apply Dixon-Coles low-score correlation."""
        if home_goals == 0 and away_goals == 0:
            return 1 - home_xg * away_xg * self.rho
        elif home_goals == 0 and away_goals == 1:
            return 1 + home_xg * self.rho
        elif home_goals == 1 and away_goals == 0:
            return 1 + away_xg * self.rho
        elif home_goals == 1 and away_goals == 1:
            return 1 - self.rho
        return 1.0

    def _matrix_to_probs(self, matrix: np.ndarray) -> tuple[float, float, float]:
        """Convert score matrix to outcome probabilities."""
        home_win = draw = away_win = 0.0

        for h in range(matrix.shape[0]):
            for a in range(matrix.shape[1]):
                if h > a:
                    home_win += matrix[h, a]
                elif h == a:
                    draw += matrix[h, a]
                else:
                    away_win += matrix[h, a]

        return home_win, draw, away_win

    def get_available_seasons(self) -> list[str]:
        """Get list of seasons with FPL data."""
        seasons = set()
        for team_data in self._team_data.values():
            seasons.update(team_data.keys())
        return sorted(seasons)

    def get_teams_in_season(self, season: str) -> list[str]:
        """Get teams with data in a season."""
        teams = []
        for team, seasons in self._team_data.items():
            if season in seasons:
                teams.append(team)
        return sorted(teams)
