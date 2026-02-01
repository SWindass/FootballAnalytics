"""
FPL-Enhanced Prediction Layer Evaluation.

Builds a sophisticated FPL-based predictor following:
- Phase 1: Granular FPL metrics (attack/defense, form)
- Phase 2: FPL-based xG with Dixon-Coles
- Phase 3: Four-model ensemble optimization
- Phase 4: Detailed analysis
"""

import warnings
from collections import defaultdict
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson
from sklearn.calibration import calibration_curve
from sklearn.metrics import log_loss

from app.db.database import SyncSessionLocal

warnings.filterwarnings("ignore")


@dataclass
class FPLTeamMetrics:
    """Detailed FPL metrics for a team."""
    # Overall strength
    total_fpl_points: float = 0.0

    # Attack metrics (FWD + MID)
    attack_fpl: float = 0.0
    attack_goals: float = 0.0
    attack_assists: float = 0.0
    attack_threat: float = 0.0
    attack_xg: float = 0.0

    # Defense metrics (GK + DEF)
    defense_fpl: float = 0.0
    defense_clean_sheets: float = 0.0
    defense_influence: float = 0.0

    # Form (recent vs season average)
    form_index: float = 1.0
    attack_form: float = 1.0
    defense_form: float = 1.0

    # Data quality
    player_count: int = 0
    minutes_coverage: float = 0.0


@dataclass
class FPLMatchPrediction:
    """FPL-based match prediction."""
    home_xg: float
    away_xg: float
    home_prob: float
    draw_prob: float
    away_prob: float
    confidence: float = 0.0
    score_matrix: np.ndarray | None = None


class EnhancedFPLPredictor:
    """
    Enhanced FPL predictor with attack/defense splits and form.
    """

    def __init__(
        self,
        form_window: int = 5,
        rho: float = -0.114,
        alpha: float = 1.35,  # Base xG multiplier
    ):
        self.form_window = form_window
        self.rho = rho
        self.alpha = alpha

        # Team data: team_name -> season -> gameweek -> metrics
        self._team_data: dict[str, dict[str, dict[int, FPLTeamMetrics]]] = defaultdict(
            lambda: defaultdict(dict)
        )

        # League averages by season
        self._league_avgs: dict[str, dict] = {}

        # Position weights
        self.attack_positions = {'FWD', 'MID'}
        self.defense_positions = {'GKP', 'DEF', 'GK'}

    def load_data(self, players_df: pd.DataFrame, performances_df: pd.DataFrame):
        """
        Load and process FPL data.

        Parameters
        ----------
        players_df : DataFrame with columns: id, name, position, team_name
        performances_df : DataFrame with match-level FPL data
        """
        # Build player lookup
        player_info = {}
        for _, p in players_df.iterrows():
            player_info[p['id']] = {
                'name': p['name'],
                'position': p.get('position', 'MID'),
                'team_name': p.get('team_name'),
            }

        # Process performances and aggregate by team/season/gameweek
        for _, perf in performances_df.iterrows():
            player_id = perf['player_id']
            info = player_info.get(player_id)

            if not info or not info['team_name']:
                continue

            team = info['team_name']
            position = info['position']
            season = perf['season']
            gw = perf['gameweek']

            # Get or create metrics for this team/season/gw
            if gw not in self._team_data[team][season]:
                self._team_data[team][season][gw] = FPLTeamMetrics()

            metrics = self._team_data[team][season][gw]

            # Extract performance data
            points = float(perf.get('total_points', 0) or 0)
            goals = float(perf.get('goals_scored', 0) or 0)
            assists = float(perf.get('assists', 0) or 0)
            clean_sheets = float(perf.get('clean_sheets', 0) or 0)
            minutes = float(perf.get('minutes', 0) or 0)
            influence = float(perf.get('influence', 0) or 0)
            float(perf.get('creativity', 0) or 0)
            threat = float(perf.get('threat', 0) or 0)
            xg = float(perf.get('expected_goals', 0) or 0)

            # Aggregate totals
            metrics.total_fpl_points += points
            metrics.player_count += 1
            metrics.minutes_coverage += minutes / 90.0

            # Split by position
            if position in self.attack_positions:
                metrics.attack_fpl += points
                metrics.attack_goals += goals
                metrics.attack_assists += assists
                metrics.attack_threat += threat
                metrics.attack_xg += xg
            elif position in self.defense_positions:
                metrics.defense_fpl += points
                metrics.defense_clean_sheets += clean_sheets
                metrics.defense_influence += influence

        # Calculate league averages
        self._calculate_league_averages()

        print(f"Loaded FPL data for {len(self._team_data)} teams")

    def _calculate_league_averages(self):
        """Calculate league average metrics by season."""
        season_data = defaultdict(lambda: {
            'attack_fpl': [], 'defense_fpl': [], 'total_fpl': [],
            'attack_xg': [], 'attack_threat': []
        })

        for _team, seasons in self._team_data.items():
            for season, gws in seasons.items():
                for _gw, metrics in gws.items():
                    if metrics.player_count > 0:
                        season_data[season]['attack_fpl'].append(metrics.attack_fpl)
                        season_data[season]['defense_fpl'].append(metrics.defense_fpl)
                        season_data[season]['total_fpl'].append(metrics.total_fpl_points)
                        season_data[season]['attack_xg'].append(metrics.attack_xg)
                        season_data[season]['attack_threat'].append(metrics.attack_threat)

        for season, data in season_data.items():
            self._league_avgs[season] = {
                'attack_fpl': np.mean(data['attack_fpl']) if data['attack_fpl'] else 20.0,
                'defense_fpl': np.mean(data['defense_fpl']) if data['defense_fpl'] else 15.0,
                'total_fpl': np.mean(data['total_fpl']) if data['total_fpl'] else 35.0,
                'attack_xg': np.mean(data['attack_xg']) if data['attack_xg'] else 1.0,
                'attack_threat': np.mean(data['attack_threat']) if data['attack_threat'] else 30.0,
            }

    def get_team_metrics(
        self,
        team: str,
        season: str,
        gameweek: int,
    ) -> FPLTeamMetrics:
        """
        Get FPL metrics for a team using data from previous gameweeks.

        Calculates form by comparing recent performance to season average.
        """
        team_seasons = self._team_data.get(team, {})
        season_data = team_seasons.get(season, {})

        if not season_data:
            return FPLTeamMetrics()

        # Get recent gameweeks (before current)
        available_gws = sorted([gw for gw in season_data.keys() if gw < gameweek])
        recent_gws = available_gws[-self.form_window:] if available_gws else []

        if not recent_gws:
            return FPLTeamMetrics()

        # Aggregate recent metrics
        recent_attack = []
        recent_defense = []
        recent_total = []
        recent_attack_xg = []

        for gw in recent_gws:
            m = season_data[gw]
            recent_attack.append(m.attack_fpl)
            recent_defense.append(m.defense_fpl)
            recent_total.append(m.total_fpl_points)
            recent_attack_xg.append(m.attack_xg)

        # Calculate season averages (all GWs before current)
        all_attack = [season_data[gw].attack_fpl for gw in available_gws]
        all_defense = [season_data[gw].defense_fpl for gw in available_gws]
        all_total = [season_data[gw].total_fpl_points for gw in available_gws]

        season_attack_avg = np.mean(all_attack) if all_attack else 1.0
        season_defense_avg = np.mean(all_defense) if all_defense else 1.0
        season_total_avg = np.mean(all_total) if all_total else 1.0

        recent_attack_avg = np.mean(recent_attack)
        recent_defense_avg = np.mean(recent_defense)
        recent_total_avg = np.mean(recent_total)

        # Calculate form indices
        attack_form = recent_attack_avg / season_attack_avg if season_attack_avg > 0 else 1.0
        defense_form = recent_defense_avg / season_defense_avg if season_defense_avg > 0 else 1.0
        form_index = recent_total_avg / season_total_avg if season_total_avg > 0 else 1.0

        # Get latest metrics for other values
        latest = season_data[recent_gws[-1]]

        return FPLTeamMetrics(
            total_fpl_points=recent_total_avg,
            attack_fpl=recent_attack_avg,
            attack_goals=np.mean([season_data[gw].attack_goals for gw in recent_gws]),
            attack_assists=np.mean([season_data[gw].attack_assists for gw in recent_gws]),
            attack_threat=np.mean([season_data[gw].attack_threat for gw in recent_gws]),
            attack_xg=np.mean(recent_attack_xg),
            defense_fpl=recent_defense_avg,
            defense_clean_sheets=np.mean([season_data[gw].defense_clean_sheets for gw in recent_gws]),
            defense_influence=np.mean([season_data[gw].defense_influence for gw in recent_gws]),
            form_index=form_index,
            attack_form=attack_form,
            defense_form=defense_form,
            player_count=latest.player_count,
            minutes_coverage=latest.minutes_coverage,
        )

    def predict_match(
        self,
        home_team: str,
        away_team: str,
        season: str,
        gameweek: int,
    ) -> FPLMatchPrediction:
        """
        Predict match using FPL-based expected goals.

        home_xG = α × (home_attack / league_avg) × (league_avg / away_defense) × home_form
        """
        home_metrics = self.get_team_metrics(home_team, season, gameweek)
        away_metrics = self.get_team_metrics(away_team, season, gameweek)

        league_avg = self._league_avgs.get(season, {
            'attack_fpl': 20.0, 'defense_fpl': 15.0, 'total_fpl': 35.0
        })

        # Calculate attack/defense ratios
        home_attack_ratio = home_metrics.attack_fpl / league_avg['attack_fpl'] if league_avg['attack_fpl'] > 0 else 1.0
        away_attack_ratio = away_metrics.attack_fpl / league_avg['attack_fpl'] if league_avg['attack_fpl'] > 0 else 1.0

        # Defense ratio (higher defense FPL = stronger, so inverse for xG against)
        home_defense_ratio = league_avg['defense_fpl'] / home_metrics.defense_fpl if home_metrics.defense_fpl > 0 else 1.0
        away_defense_ratio = league_avg['defense_fpl'] / away_metrics.defense_fpl if away_metrics.defense_fpl > 0 else 1.0

        # Calculate xG with form adjustment
        # Home advantage factor
        home_advantage = 1.12
        away_disadvantage = 0.88

        home_xg = (
            self.alpha *
            home_attack_ratio *
            away_defense_ratio *
            home_metrics.attack_form *
            home_advantage
        )

        away_xg = (
            self.alpha *
            away_attack_ratio *
            home_defense_ratio *
            away_metrics.attack_form *
            away_disadvantage
        )

        # Clamp to reasonable range
        home_xg = np.clip(home_xg, 0.4, 3.5)
        away_xg = np.clip(away_xg, 0.3, 3.0)

        # Generate probability matrix with Dixon-Coles
        matrix = self._calculate_score_matrix(home_xg, away_xg)

        # Extract probabilities
        home_prob, draw_prob, away_prob = self._matrix_to_probs(matrix)

        # Calculate confidence based on data quality
        has_home = home_metrics.player_count > 5
        has_away = away_metrics.player_count > 5
        confidence = 0.5 * has_home + 0.5 * has_away

        return FPLMatchPrediction(
            home_xg=home_xg,
            away_xg=away_xg,
            home_prob=home_prob,
            draw_prob=draw_prob,
            away_prob=away_prob,
            confidence=confidence,
            score_matrix=matrix,
        )

    def _calculate_score_matrix(
        self,
        home_xg: float,
        away_xg: float,
        max_goals: int = 6,
    ) -> np.ndarray:
        """Calculate score probability matrix with Dixon-Coles correction."""
        matrix = np.zeros((max_goals + 1, max_goals + 1))

        for h in range(max_goals + 1):
            for a in range(max_goals + 1):
                base_prob = poisson.pmf(h, home_xg) * poisson.pmf(a, away_xg)
                correction = self._dixon_coles_correction(h, a, home_xg, away_xg)
                matrix[h, a] = base_prob * correction

        # Normalize
        total = matrix.sum()
        if total > 0:
            matrix /= total

        return matrix

    def _dixon_coles_correction(
        self,
        home_goals: int,
        away_goals: int,
        home_xg: float,
        away_xg: float,
    ) -> float:
        """Apply Dixon-Coles low-score correlation correction."""
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

    def optimize_alpha(self, matches: list[dict]) -> float:
        """Optimize alpha parameter using historical matches."""

        def objective(alpha):
            self.alpha = alpha[0]
            total_brier = 0.0
            count = 0

            for m in matches:
                try:
                    pred = self.predict_match(
                        m['home_team'], m['away_team'],
                        m['season'], m['gameweek']
                    )

                    if pred.confidence < 0.5:
                        continue

                    # Calculate Brier score
                    actual = [0, 0, 0]
                    if m['home_score'] > m['away_score']:
                        actual[0] = 1
                    elif m['home_score'] == m['away_score']:
                        actual[1] = 1
                    else:
                        actual[2] = 1

                    brier = (
                        (pred.home_prob - actual[0])**2 +
                        (pred.draw_prob - actual[1])**2 +
                        (pred.away_prob - actual[2])**2
                    ) / 3

                    total_brier += brier
                    count += 1
                except:
                    continue

            return total_brier / count if count > 0 else 1.0

        result = minimize(
            objective,
            x0=[1.35],
            bounds=[(0.5, 2.5)],
            method='L-BFGS-B'
        )

        self.alpha = result.x[0]
        return self.alpha


def load_data():
    """Load all required data from database."""
    from sqlalchemy import text

    print("Loading data...")

    with SyncSessionLocal() as session:
        # Load players with team info
        print("  Loading players...")
        players_query = text("""
            SELECT p.id, p.name, p.position, p.team_id, t.name as team_name
            FROM players p
            LEFT JOIN teams t ON p.team_id = t.id
        """)
        players_result = session.execute(players_query)
        players_df = pd.DataFrame(players_result.fetchall())
        if not players_df.empty:
            players_df.columns = ['id', 'name', 'position', 'team_id', 'team_name']

        # Load performances
        print("  Loading performances...")
        perfs_query = text("""
            SELECT player_id, season, gameweek, total_points, goals_scored,
                   assists, clean_sheets, minutes, influence, creativity,
                   threat, ict_index, expected_goals, expected_assists
            FROM player_match_performances
        """)
        perfs_result = session.execute(perfs_query)
        perfs_df = pd.DataFrame(perfs_result.fetchall())
        if not perfs_df.empty:
            perfs_df.columns = [
                'player_id', 'season', 'gameweek', 'total_points', 'goals_scored',
                'assists', 'clean_sheets', 'minutes', 'influence', 'creativity',
                'threat', 'ict_index', 'expected_goals', 'expected_assists'
            ]

        # Get seasons with FPL data
        fpl_seasons = sorted(perfs_df['season'].unique()) if not perfs_df.empty else []
        print(f"  FPL seasons: {', '.join(fpl_seasons)}")

        # Load matches
        print("  Loading matches...")
        matches_query = text("""
            SELECT m.id, m.kickoff_time, m.matchweek, m.season,
                   m.home_team_id, m.away_team_id, m.home_score, m.away_score,
                   ht.name as home_team, at.name as away_team
            FROM matches m
            JOIN teams ht ON m.home_team_id = ht.id
            JOIN teams at ON m.away_team_id = at.id
            WHERE m.status = 'finished'
            AND m.home_score IS NOT NULL
            ORDER BY m.kickoff_time
        """)
        matches_result = session.execute(matches_query)
        matches_df = pd.DataFrame(matches_result.fetchall())
        if not matches_df.empty:
            matches_df.columns = [
                'id', 'kickoff_time', 'matchweek', 'season',
                'home_team_id', 'away_team_id', 'home_score', 'away_score',
                'home_team', 'away_team'
            ]

        # Filter to seasons with FPL data
        matches_df = matches_df[matches_df['season'].isin(fpl_seasons)]

        print(f"  Loaded {len(players_df)} players, {len(perfs_df)} performances, {len(matches_df)} matches")

        return players_df, perfs_df, matches_df


def build_base_predictions(matches_df: pd.DataFrame, warmup: int = 500) -> dict[int, dict]:
    """
    Build predictions from Pi+DC, ELO, and Pi Baseline models.
    Returns dict[match_id] -> {model predictions}
    """
    import math

    from batch.models.elo import EloConfig, EloRatingSystem
    from batch.models.pi_dixon_coles import PiDixonColesModel
    from batch.models.pi_rating import PiRating

    print("\nBuilding base model predictions...")

    # Initialize models
    elo = EloRatingSystem(EloConfig(k_factor=28.0, home_advantage=50.0))
    pi = PiRating(lambda_param=0.07, gamma_param=0.7)
    pidc = PiDixonColesModel(pi_lambda=0.07, pi_gamma=0.7, rho=-0.11)

    # Team ID mapping for ELO
    team_ids = {}
    next_id = 1

    # Sort matches by date
    matches_df = matches_df.sort_values('kickoff_time').reset_index(drop=True)

    predictions = {}

    for idx, row in matches_df.iterrows():
        home = row['home_team']
        away = row['away_team']
        match_id = row['id']

        # Ensure teams have IDs for ELO
        if home not in team_ids:
            team_ids[home] = next_id
            next_id += 1
        if away not in team_ids:
            team_ids[away] = next_id
            next_id += 1

        # Get predictions before updating
        try:
            # Pi+DC prediction
            pidc_pred = pidc.predict_match(home, away, apply_draw_model=pidc.draw_model_trained)

            # Pi baseline prediction (convert goal diff to probabilities)
            pi_gd = pi.calculate_expected_goal_diff(home, away)
            pi_h = 1 / (1 + math.exp(-pi_gd * 0.7))
            pi_a = 1 / (1 + math.exp(pi_gd * 0.7))
            pi_d = max(0, 0.28 - 0.1 * abs(pi_gd))
            pi_total = pi_h + pi_d + pi_a
            pi_h, pi_d, pi_a = pi_h/pi_total, pi_d/pi_total, pi_a/pi_total

            # ELO prediction
            elo_h, elo_d, elo_a = elo.match_probabilities(
                team_ids[home], team_ids[away]
            )

            # Only store predictions after warmup
            if idx >= warmup:
                predictions[match_id] = {
                    'elo': {
                        'home_prob': elo_h,
                        'draw_prob': elo_d,
                        'away_prob': elo_a,
                    },
                    'pi': {
                        'home_prob': pi_h,
                        'draw_prob': pi_d,
                        'away_prob': pi_a,
                    },
                    'pi_dc': {
                        'home_prob': pidc_pred.home_win,
                        'draw_prob': pidc_pred.draw,
                        'away_prob': pidc_pred.away_win,
                    },
                }
        except Exception:
            pass

        # Update models with actual result
        pidc.update_after_match(
            home, away, row['home_score'], row['away_score'],
            row['kickoff_time'], collect_training_data=True
        )
        pi.update_ratings(
            home, away, row['home_score'], row['away_score'],
            row['kickoff_time'], store_history=False
        )
        elo.update_ratings(
            team_ids[home], team_ids[away],
            row['home_score'], row['away_score']
        )

        if (idx + 1) % 500 == 0:
            print(f"  Processed {idx + 1}/{len(matches_df)} matches")

    print(f"  Generated predictions for {len(predictions)} matches")
    return predictions


def evaluate_models(
    matches_df: pd.DataFrame,
    base_preds: dict,
    fpl_predictor: EnhancedFPLPredictor,
    test_seasons: list[str],
) -> pd.DataFrame:
    """
    Evaluate all models and ensembles on test data.
    """
    results = []

    for _, row in matches_df.iterrows():
        if row['season'] not in test_seasons:
            continue

        match_id = row['id']
        if match_id not in base_preds:
            continue

        # Get base predictions
        base = base_preds[match_id]

        # Get FPL prediction
        fpl_pred = fpl_predictor.predict_match(
            row['home_team'], row['away_team'],
            row['season'], row['matchweek']
        )

        # Actual outcome
        if row['home_score'] > row['away_score']:
            actual = 'H'
        elif row['home_score'] == row['away_score']:
            actual = 'D'
        else:
            actual = 'A'

        results.append({
            'match_id': match_id,
            'season': row['season'],
            'gameweek': row['matchweek'],
            'home_team': row['home_team'],
            'away_team': row['away_team'],
            'home_score': row['home_score'],
            'away_score': row['away_score'],
            'actual': actual,
            # ELO
            'elo_home': base['elo']['home_prob'],
            'elo_draw': base['elo']['draw_prob'],
            'elo_away': base['elo']['away_prob'],
            # Pi
            'pi_home': base['pi']['home_prob'],
            'pi_draw': base['pi']['draw_prob'],
            'pi_away': base['pi']['away_prob'],
            # Pi+DC
            'pidc_home': base['pi_dc']['home_prob'],
            'pidc_draw': base['pi_dc']['draw_prob'],
            'pidc_away': base['pi_dc']['away_prob'],
            # FPL
            'fpl_home': fpl_pred.home_prob,
            'fpl_draw': fpl_pred.draw_prob,
            'fpl_away': fpl_pred.away_prob,
            'fpl_home_xg': fpl_pred.home_xg,
            'fpl_away_xg': fpl_pred.away_xg,
            'fpl_confidence': fpl_pred.confidence,
        })

    return pd.DataFrame(results)


def optimize_ensemble_weights(
    results_df: pd.DataFrame,
    models: list[str] = None,
) -> dict[str, float]:
    """
    Optimize ensemble weights using Brier score.
    """
    if models is None:
        models = ['pidc', 'elo', 'pi', 'fpl']
    n_models = len(models)

    def ensemble_brier(weights):
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize

        total_brier = 0.0

        for _, row in results_df.iterrows():
            # Weighted average predictions
            home_prob = sum(w * row[f'{m}_home'] for w, m in zip(weights, models, strict=False))
            draw_prob = sum(w * row[f'{m}_draw'] for w, m in zip(weights, models, strict=False))
            away_prob = sum(w * row[f'{m}_away'] for w, m in zip(weights, models, strict=False))

            # Normalize
            total = home_prob + draw_prob + away_prob
            home_prob /= total
            draw_prob /= total
            away_prob /= total

            # Actual
            actual = [0, 0, 0]
            if row['actual'] == 'H':
                actual[0] = 1
            elif row['actual'] == 'D':
                actual[1] = 1
            else:
                actual[2] = 1

            brier = (
                (home_prob - actual[0])**2 +
                (draw_prob - actual[1])**2 +
                (away_prob - actual[2])**2
            ) / 3

            total_brier += brier

        return total_brier / len(results_df)

    # Initial weights - equal for all models
    x0 = [1.0 / n_models] * n_models

    # Optimize
    result = minimize(
        ensemble_brier,
        x0=x0,
        bounds=[(0, 1)] * n_models,
        method='L-BFGS-B',
        options={'maxiter': 1000}
    )

    weights = np.array(result.x)
    weights = weights / weights.sum()

    return dict(zip(models, weights, strict=False))


def calculate_metrics(
    results_df: pd.DataFrame,
    model_prefix: str,
) -> dict:
    """Calculate accuracy, Brier score, log loss for a model."""

    y_true = []
    y_pred = []
    predictions = []

    for _, row in results_df.iterrows():
        probs = [
            row[f'{model_prefix}_home'],
            row[f'{model_prefix}_draw'],
            row[f'{model_prefix}_away'],
        ]

        # Normalize
        total = sum(probs)
        probs = [p / total for p in probs]

        y_pred.append(probs)

        if row['actual'] == 'H':
            y_true.append(0)
        elif row['actual'] == 'D':
            y_true.append(1)
        else:
            y_true.append(2)

        predictions.append(np.argmax(probs))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    predictions = np.array(predictions)

    # Accuracy
    accuracy = (predictions == y_true).mean()

    # Brier score (multiclass)
    y_true_onehot = np.zeros((len(y_true), 3))
    y_true_onehot[np.arange(len(y_true)), y_true] = 1
    brier = ((y_pred - y_true_onehot) ** 2).sum(axis=1).mean()

    # Log loss
    y_pred_clipped = np.clip(y_pred, 1e-10, 1 - 1e-10)
    logloss = log_loss(y_true, y_pred_clipped)

    # Draw metrics
    draw_pred = (predictions == 1)
    draw_actual = (y_true == 1)
    draw_precision = draw_pred[draw_actual].sum() / draw_pred.sum() if draw_pred.sum() > 0 else 0
    draw_recall = draw_pred[draw_actual].sum() / draw_actual.sum() if draw_actual.sum() > 0 else 0

    return {
        'accuracy': accuracy,
        'brier': brier,
        'logloss': logloss,
        'draw_precision': draw_precision,
        'draw_recall': draw_recall,
    }


def add_ensemble_predictions(
    results_df: pd.DataFrame,
    weights: dict[str, float],
    prefix: str = 'ens4',
) -> pd.DataFrame:
    """Add ensemble predictions to results dataframe."""
    df = results_df.copy()

    models = list(weights.keys())

    for outcome in ['home', 'draw', 'away']:
        df[f'{prefix}_{outcome}'] = sum(
            weights[m] * df[f'{m}_{outcome}']
            for m in models
        )

    # Normalize
    total = df[f'{prefix}_home'] + df[f'{prefix}_draw'] + df[f'{prefix}_away']
    df[f'{prefix}_home'] /= total
    df[f'{prefix}_draw'] /= total
    df[f'{prefix}_away'] /= total

    return df


def analyze_fpl_performance(results_df: pd.DataFrame):
    """
    Analyze when FPL predictions are most useful.
    """
    print("\n" + "=" * 80)
    print("FPL PERFORMANCE ANALYSIS")
    print("=" * 80)

    # Calculate FPL prediction error vs base models
    results_df = results_df.copy()

    for _, row in results_df.iterrows():
        actual_vec = [0, 0, 0]
        if row['actual'] == 'H':
            actual_vec[0] = 1
        elif row['actual'] == 'D':
            actual_vec[1] = 1
        else:
            actual_vec[2] = 1

    # FPL vs Pi+DC comparison
    fpl_better = 0
    pidc_better = 0

    for _, row in results_df.iterrows():
        actual_vec = [0, 0, 0]
        if row['actual'] == 'H':
            actual_vec[0] = 1
        elif row['actual'] == 'D':
            actual_vec[1] = 1
        else:
            actual_vec[2] = 1

        fpl_brier = (
            (row['fpl_home'] - actual_vec[0])**2 +
            (row['fpl_draw'] - actual_vec[1])**2 +
            (row['fpl_away'] - actual_vec[2])**2
        )

        pidc_brier = (
            (row['pidc_home'] - actual_vec[0])**2 +
            (row['pidc_draw'] - actual_vec[1])**2 +
            (row['pidc_away'] - actual_vec[2])**2
        )

        if fpl_brier < pidc_brier:
            fpl_better += 1
        elif pidc_brier < fpl_brier:
            pidc_better += 1

    print("\nMatch-by-match comparison (FPL vs Pi+DC):")
    print(f"  FPL better:  {fpl_better} ({fpl_better/len(results_df)*100:.1f}%)")
    print(f"  Pi+DC better: {pidc_better} ({pidc_better/len(results_df)*100:.1f}%)")
    print(f"  Equal:       {len(results_df) - fpl_better - pidc_better}")

    # FPL xG correlation with actual goals
    print("\nFPL xG correlation with actual goals:")
    home_xg_corr = np.corrcoef(results_df['fpl_home_xg'], results_df['home_score'])[0, 1]
    away_xg_corr = np.corrcoef(results_df['fpl_away_xg'], results_df['away_score'])[0, 1]
    total_xg = results_df['fpl_home_xg'] + results_df['fpl_away_xg']
    total_goals = results_df['home_score'] + results_df['away_score']
    total_xg_corr = np.corrcoef(total_xg, total_goals)[0, 1]

    print(f"  Home xG correlation:  {home_xg_corr:.3f}")
    print(f"  Away xG correlation:  {away_xg_corr:.3f}")
    print(f"  Total xG correlation: {total_xg_corr:.3f}")

    # Analyze by form differential
    print("\nPerformance by FPL confidence:")
    conf_bins = [(0, 0.5), (0.5, 0.75), (0.75, 1.0)]

    for low, high in conf_bins:
        mask = (results_df['fpl_confidence'] >= low) & (results_df['fpl_confidence'] < high)
        subset = results_df[mask]
        if len(subset) > 10:
            fpl_metrics = calculate_metrics(subset, 'fpl')
            pidc_metrics = calculate_metrics(subset, 'pidc')
            print(f"  Confidence [{low:.2f}, {high:.2f}): n={len(subset)}")
            print(f"    FPL Brier:  {fpl_metrics['brier']:.4f}")
            print(f"    Pi+DC Brier: {pidc_metrics['brier']:.4f}")


def plot_calibration(results_df: pd.DataFrame, save_path: str = None):
    """Plot calibration curves for all models."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    outcomes = ['home', 'draw', 'away']
    outcome_names = ['Home Win', 'Draw', 'Away Win']

    for ax, outcome, name in zip(axes, outcomes, outcome_names, strict=False):
        # Actual outcome
        y_true = (results_df['actual'] == outcome[0].upper()).astype(int)

        # Plot calibration for each model
        for model, color, label in [
            ('pidc', 'blue', 'Pi+DC'),
            ('elo', 'green', 'ELO'),
            ('fpl', 'red', 'FPL'),
            ('ens4', 'purple', 'Ensemble'),
        ]:
            if f'{model}_{outcome}' in results_df.columns:
                y_prob = results_df[f'{model}_{outcome}']

                try:
                    prob_true, prob_pred = calibration_curve(
                        y_true, y_prob, n_bins=10, strategy='uniform'
                    )
                    ax.plot(prob_pred, prob_true, 's-', color=color, label=label)
                except:
                    pass

        ax.plot([0, 1], [0, 1], 'k--', label='Perfect')
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title(f'{name} Calibration')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nCalibration plot saved to {save_path}")
    else:
        plt.show()


def main():
    # Load data
    players_df, perfs_df, matches_df = load_data()

    if matches_df.empty:
        print("No match data available!")
        return

    # Initialize FPL predictor
    print("\nInitializing Enhanced FPL Predictor...")
    fpl_predictor = EnhancedFPLPredictor()
    fpl_predictor.load_data(players_df, perfs_df)

    # Build base model predictions
    base_preds = build_base_predictions(matches_df)

    # Get all seasons
    all_seasons = sorted(matches_df['season'].unique())
    print(f"\nSeasons available: {', '.join(all_seasons)}")

    # Split: train on first 70%, test on last 30%
    n_seasons = len(all_seasons)
    train_seasons = all_seasons[:int(n_seasons * 0.7)]
    test_seasons = all_seasons[int(n_seasons * 0.7):]

    print(f"Train seasons: {', '.join(train_seasons)}")
    print(f"Test seasons:  {', '.join(test_seasons)}")

    # Prepare training data for alpha optimization
    train_matches = []
    for _, row in matches_df.iterrows():
        if row['season'] in train_seasons and row['id'] in base_preds:
            train_matches.append({
                'home_team': row['home_team'],
                'away_team': row['away_team'],
                'home_score': row['home_score'],
                'away_score': row['away_score'],
                'season': row['season'],
                'gameweek': row['matchweek'],
            })

    # Optimize FPL alpha parameter
    print("\nOptimizing FPL alpha parameter...")
    optimal_alpha = fpl_predictor.optimize_alpha(train_matches)
    print(f"Optimal alpha: {optimal_alpha:.3f}")

    # Evaluate on test set
    print("\nEvaluating models on test set...")
    results_df = evaluate_models(
        matches_df, base_preds, fpl_predictor, test_seasons
    )

    if results_df.empty:
        print("No test results!")
        return

    print(f"Test matches: {len(results_df)}")

    # Calculate individual model metrics
    print("\n" + "=" * 80)
    print("INDIVIDUAL MODEL PERFORMANCE")
    print("=" * 80)

    models = ['pidc', 'elo', 'pi', 'fpl']
    model_names = ['Pi+DC', 'ELO', 'Pi Baseline', 'FPL']

    print(f"\n{'Model':<15} {'Accuracy':>10} {'Brier':>10} {'Log Loss':>10} {'Draw Prec':>10} {'Draw Rec':>10}")
    print("-" * 75)

    for model, name in zip(models, model_names, strict=False):
        metrics = calculate_metrics(results_df, model)
        print(f"{name:<15} {metrics['accuracy']*100:>9.1f}% {metrics['brier']:>10.4f} "
              f"{metrics['logloss']:>10.4f} {metrics['draw_precision']*100:>9.1f}% "
              f"{metrics['draw_recall']*100:>9.1f}%")

    # Optimize 3-model ensemble (without FPL)
    print("\n" + "=" * 80)
    print("ENSEMBLE OPTIMIZATION")
    print("=" * 80)

    # 3-model ensemble
    train_results = evaluate_models(
        matches_df, base_preds, fpl_predictor, train_seasons
    )

    weights_3model = optimize_ensemble_weights(train_results, ['pidc', 'elo', 'pi'])
    print("\n3-Model Ensemble Weights (Pi+DC, ELO, Pi):")
    for m, w in weights_3model.items():
        print(f"  {m}: {w*100:.1f}%")

    # 4-model ensemble (with FPL)
    weights_4model = optimize_ensemble_weights(train_results, ['pidc', 'elo', 'pi', 'fpl'])
    print("\n4-Model Ensemble Weights (Pi+DC, ELO, Pi, FPL):")
    for m, w in weights_4model.items():
        print(f"  {m}: {w*100:.1f}%")

    # Add ensemble predictions to results
    results_df = add_ensemble_predictions(results_df, weights_3model, 'ens3')
    results_df = add_ensemble_predictions(results_df, weights_4model, 'ens4')

    # Compare ensembles
    print("\n" + "=" * 80)
    print("ENSEMBLE COMPARISON")
    print("=" * 80)

    ens3_metrics = calculate_metrics(results_df, 'ens3')
    ens4_metrics = calculate_metrics(results_df, 'ens4')

    print(f"\n{'Ensemble':<20} {'Accuracy':>10} {'Brier':>10} {'Log Loss':>10} {'Draw Prec':>10} {'Draw Rec':>10}")
    print("-" * 80)
    print(f"{'3-Model (no FPL)':<20} {ens3_metrics['accuracy']*100:>9.1f}% {ens3_metrics['brier']:>10.4f} "
          f"{ens3_metrics['logloss']:>10.4f} {ens3_metrics['draw_precision']*100:>9.1f}% "
          f"{ens3_metrics['draw_recall']*100:>9.1f}%")
    print(f"{'4-Model (with FPL)':<20} {ens4_metrics['accuracy']*100:>9.1f}% {ens4_metrics['brier']:>10.4f} "
          f"{ens4_metrics['logloss']:>10.4f} {ens4_metrics['draw_precision']*100:>9.1f}% "
          f"{ens4_metrics['draw_recall']*100:>9.1f}%")

    # Improvement
    brier_improvement = (ens3_metrics['brier'] - ens4_metrics['brier']) / ens3_metrics['brier'] * 100
    acc_improvement = (ens4_metrics['accuracy'] - ens3_metrics['accuracy']) * 100

    print("\nFPL Contribution:")
    print(f"  Brier improvement: {brier_improvement:+.2f}%")
    print(f"  Accuracy improvement: {acc_improvement:+.2f}pp")

    # Analyze FPL performance
    analyze_fpl_performance(results_df)

    # Season breakdown
    print("\n" + "=" * 80)
    print("SEASON BREAKDOWN")
    print("=" * 80)

    print(f"\n{'Season':<12} {'Matches':>8} {'3-Ens Acc':>12} {'4-Ens Acc':>12} {'FPL Weight':>12}")
    print("-" * 60)

    for season in test_seasons:
        season_df = results_df[results_df['season'] == season]
        if len(season_df) > 0:
            ens3_acc = calculate_metrics(season_df, 'ens3')['accuracy']
            ens4_acc = calculate_metrics(season_df, 'ens4')['accuracy']
            fpl_weight = weights_4model.get('fpl', 0) * 100
            print(f"{season:<12} {len(season_df):>8} {ens3_acc*100:>11.1f}% {ens4_acc*100:>11.1f}% {fpl_weight:>11.1f}%")

    # Final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    print(f"""
Best individual model: Pi+DC
  Accuracy: {calculate_metrics(results_df, 'pidc')['accuracy']*100:.1f}%
  Brier:    {calculate_metrics(results_df, 'pidc')['brier']:.4f}

3-Model Ensemble (Pi+DC + ELO + Pi):
  Accuracy: {ens3_metrics['accuracy']*100:.1f}%
  Brier:    {ens3_metrics['brier']:.4f}

4-Model Ensemble (+ FPL @ {weights_4model.get('fpl', 0)*100:.1f}%):
  Accuracy: {ens4_metrics['accuracy']*100:.1f}%
  Brier:    {ens4_metrics['brier']:.4f}

FPL Optimal Weight: {weights_4model.get('fpl', 0)*100:.1f}%
FPL-only Performance:
  Accuracy: {calculate_metrics(results_df, 'fpl')['accuracy']*100:.1f}%
  Brier:    {calculate_metrics(results_df, 'fpl')['brier']:.4f}
""")

    # Plot calibration
    try:
        plot_calibration(results_df, '/tmp/fpl_calibration.png')
    except Exception as e:
        print(f"Could not generate calibration plot: {e}")


if __name__ == "__main__":
    main()
