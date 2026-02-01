"""Ensemble predictor combining Pi+Dixon-Coles, Pi Baseline, ELO, and xG.

Implements multiple weighting strategies:
1. Fixed weights (optimized on training data)
2. Situational weights (adapt based on match characteristics)
3. Confidence-based weights (trust more confident predictions)
4. Recalibration-aware (adjusts based on recent match statistics)

Note: The seasonal recalibration addresses the 2025-26 performance issue
where elevated draw rates caused accuracy to drop from 54.6% to 47.6%.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from batch.models.seasonal_recalibration import (
    SeasonalRecalibration,
    ConservativeRecalibration,
    apply_draw_threshold_adjustment,
)


@dataclass
class EnsemblePrediction:
    """Ensemble prediction with component breakdown."""

    home_prob: float
    draw_prob: float
    away_prob: float

    # Component predictions
    pidc_probs: tuple[float, float, float] = (0, 0, 0)
    pi_probs: tuple[float, float, float] = (0, 0, 0)
    elo_probs: tuple[float, float, float] = (0, 0, 0)

    # Weights used
    weights: tuple[float, float, float] = (0.33, 0.33, 0.33)

    # Confidence metrics
    max_prob: float = 0.0
    entropy: float = 0.0
    agreement: float = 0.0  # How much models agree

    def get_prediction(self, draw_threshold: float = 0.26, aggressive_draws: bool = True) -> str:
        """Get predicted outcome with configurable draw threshold."""
        if aggressive_draws:
            # Predict draw if:
            # 1. Draw prob is high and home/away are close
            if self.draw_prob >= draw_threshold and abs(self.home_prob - self.away_prob) < 0.08:
                return "D"

            # 2. Models disagree and draw is competitive
            if self.agreement < 0.55 and self.draw_prob >= 0.24:
                return "D"

            # 3. Draw is very close to max probability
            max_prob = max(self.home_prob, self.away_prob)
            if self.draw_prob >= max_prob * 0.95 and self.draw_prob >= 0.25:
                return "D"

        if self.home_prob >= self.away_prob:
            return "H"
        return "A"

    def get_prediction_with_threshold(
        self,
        draw_threshold: float = 0.26,
        parity_threshold: float = 0.08,
    ) -> str:
        """Get prediction using the draw threshold adjustment algorithm.

        This method uses the recalibration module's threshold-based
        prediction which better handles draw prediction.
        """
        return apply_draw_threshold_adjustment(
            self.home_prob,
            self.draw_prob,
            self.away_prob,
            draw_threshold=draw_threshold,
            parity_threshold=parity_threshold,
        )


class EnsemblePredictor:
    """Ensemble of Pi+DC, Pi Baseline, and ELO models.

    Supports multiple weighting strategies:
    - 'fixed': Constant optimized weights
    - 'situational': Adjust weights based on match type
    - 'confidence': Weight by prediction confidence
    - 'stacking': Meta-model learns optimal combination
    - 'recalibrated': Apply seasonal recalibration to final probabilities
    """

    def __init__(
        self,
        weights: tuple[float, float, float] = (0.33, 0.33, 0.34),
        strategy: str = "fixed",
        enable_recalibration: bool = False,
        recalibration_window: int = 50,
    ):
        """Initialize ensemble.

        Parameters
        ----------
        weights : tuple[float, float, float]
            Weights for (Pi+DC, Pi Baseline, ELO).
        strategy : str
            Weighting strategy: 'fixed', 'situational', 'confidence'.
        enable_recalibration : bool
            If True, apply seasonal recalibration to adjust for
            changing league characteristics (draw rates, etc.)
        recalibration_window : int
            Number of recent matches to track for recalibration.
        """
        self.base_weights = np.array(weights)
        self.strategy = strategy

        # Learned parameters for situational weighting
        self.situational_params = {
            "even_match_pidc_boost": 0.15,  # Boost Pi+DC for even matches
            "decisive_elo_boost": 0.10,  # Boost ELO for decisive favorites
        }

        # Seasonal recalibration
        self.enable_recalibration = enable_recalibration
        if enable_recalibration:
            self.recalibrator = ConservativeRecalibration(window_size=recalibration_window)
        else:
            self.recalibrator = None

    def add_result(
        self,
        home_goals: int,
        away_goals: int,
        home_xg: float = 0.0,
        away_xg: float = 0.0,
    ) -> None:
        """Add a match result for recalibration tracking.

        Call this after each match to keep the recalibrator up to date.
        """
        if self.recalibrator:
            self.recalibrator.add_match(home_goals, away_goals, home_xg, away_xg)

    def get_recalibration_summary(self) -> str:
        """Get diagnostic summary of current recalibration state."""
        if self.recalibrator:
            return self.recalibrator.get_diagnostic_summary()
        return "Recalibration disabled"

    def combine_predictions(
        self,
        pidc_probs: tuple[float, float, float],
        pi_probs: tuple[float, float, float],
        elo_probs: tuple[float, float, float],
        match_features: Optional[dict] = None,
        apply_recalibration: Optional[bool] = None,
    ) -> EnsemblePrediction:
        """Combine predictions from three models.

        Parameters
        ----------
        pidc_probs : tuple
            (home, draw, away) from Pi+Dixon-Coles.
        pi_probs : tuple
            (home, draw, away) from Pi Baseline.
        elo_probs : tuple
            (home, draw, away) from ELO.
        match_features : dict, optional
            Features for situational weighting.
        apply_recalibration : bool, optional
            Override recalibration setting. If None, uses self.enable_recalibration.

        Returns
        -------
        EnsemblePrediction
            Combined prediction.
        """
        # Get weights based on strategy
        weights = self._get_weights(
            pidc_probs, pi_probs, elo_probs, match_features
        )

        # Stack predictions
        all_probs = np.array([pidc_probs, pi_probs, elo_probs])

        # Weighted combination
        combined = weights @ all_probs

        # Normalize
        combined = combined / combined.sum()

        # Apply seasonal recalibration if enabled
        should_recalibrate = apply_recalibration if apply_recalibration is not None else self.enable_recalibration
        if should_recalibrate and self.recalibrator:
            home, draw, away = self.recalibrator.adjust_probabilities(
                combined[0], combined[1], combined[2]
            )
            combined = np.array([home, draw, away])

        # Calculate confidence metrics
        max_prob = combined.max()
        entropy = -np.sum(combined * np.log(combined + 1e-10))

        # Agreement: how similar are the three predictions
        pairwise_diff = (
            np.abs(np.array(pidc_probs) - np.array(pi_probs)).sum() +
            np.abs(np.array(pidc_probs) - np.array(elo_probs)).sum() +
            np.abs(np.array(pi_probs) - np.array(elo_probs)).sum()
        ) / 6  # Normalize
        agreement = 1 - pairwise_diff

        return EnsemblePrediction(
            home_prob=combined[0],
            draw_prob=combined[1],
            away_prob=combined[2],
            pidc_probs=pidc_probs,
            pi_probs=pi_probs,
            elo_probs=elo_probs,
            weights=tuple(weights),
            max_prob=max_prob,
            entropy=entropy,
            agreement=agreement,
        )

    def _get_weights(
        self,
        pidc_probs: tuple,
        pi_probs: tuple,
        elo_probs: tuple,
        features: Optional[dict],
    ) -> np.ndarray:
        """Calculate weights based on strategy."""
        if self.strategy == "fixed":
            return self.base_weights

        elif self.strategy == "situational":
            return self._situational_weights(pidc_probs, pi_probs, elo_probs, features)

        elif self.strategy == "confidence":
            return self._confidence_weights(pidc_probs, pi_probs, elo_probs)

        else:
            return self.base_weights

    def _situational_weights(
        self,
        pidc_probs: tuple,
        pi_probs: tuple,
        elo_probs: tuple,
        features: Optional[dict],
    ) -> np.ndarray:
        """Adjust weights based on match characteristics."""
        weights = self.base_weights.copy()

        if features is None:
            return weights

        # For even matches (close ratings), boost Pi+DC (better at draws)
        rating_diff = abs(features.get("rating_diff", 0))
        if rating_diff < 0.3:
            boost = self.situational_params["even_match_pidc_boost"]
            weights[0] += boost
            weights[1] -= boost / 2
            weights[2] -= boost / 2

        # For decisive favorites, boost ELO (better calibrated for extremes)
        max_prob = max(max(pidc_probs), max(pi_probs), max(elo_probs))
        if max_prob > 0.55:
            boost = self.situational_params["decisive_elo_boost"]
            weights[2] += boost
            weights[0] -= boost / 2
            weights[1] -= boost / 2

        # Ensure non-negative and normalize
        weights = np.maximum(weights, 0.05)
        weights = weights / weights.sum()

        return weights

    def _confidence_weights(
        self,
        pidc_probs: tuple,
        pi_probs: tuple,
        elo_probs: tuple,
    ) -> np.ndarray:
        """Weight by prediction confidence (inverse entropy)."""
        def confidence(probs):
            probs = np.array(probs)
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            # Lower entropy = higher confidence
            return 1 / (1 + entropy)

        conf_pidc = confidence(pidc_probs)
        conf_pi = confidence(pi_probs)
        conf_elo = confidence(elo_probs)

        # Blend with base weights
        conf_weights = np.array([conf_pidc, conf_pi, conf_elo])
        conf_weights = conf_weights / conf_weights.sum()

        # 50% base weights, 50% confidence weights
        weights = 0.5 * self.base_weights + 0.5 * conf_weights

        return weights / weights.sum()


def optimize_ensemble_weights(
    results_df: pd.DataFrame,
    metric: str = "brier",
) -> dict:
    """Find optimal ensemble weights to minimize Brier score.

    Parameters
    ----------
    results_df : pd.DataFrame
        Backtest results with model probabilities.
    metric : str
        Metric to optimize: 'brier', 'log_loss', or 'accuracy'.

    Returns
    -------
    dict
        Optimal weights and performance metrics.
    """
    def objective(weights):
        w = weights / weights.sum()  # Normalize

        total_metric = 0
        for _, row in results_df.iterrows():
            # Combine probabilities
            home = (
                w[0] * row["pidc_home_prob"] +
                w[1] * row["pi_home_prob"] +
                w[2] * row["elo_home_prob"]
            )
            draw = (
                w[0] * row["pidc_draw_prob"] +
                w[1] * row["pi_draw_prob"] +
                w[2] * row["elo_draw_prob"]
            )
            away = (
                w[0] * row["pidc_away_prob"] +
                w[1] * row["pi_away_prob"] +
                w[2] * row["elo_away_prob"]
            )

            # Normalize
            total = home + draw + away
            home, draw, away = home/total, draw/total, away/total

            # Calculate metric
            actual = row["actual"]
            h_act = 1 if actual == "H" else 0
            d_act = 1 if actual == "D" else 0
            a_act = 1 if actual == "A" else 0

            if metric == "brier":
                total_metric += (home - h_act)**2 + (draw - d_act)**2 + (away - a_act)**2
            elif metric == "log_loss":
                epsilon = 1e-10
                if actual == "H":
                    total_metric -= np.log(max(home, epsilon))
                elif actual == "D":
                    total_metric -= np.log(max(draw, epsilon))
                else:
                    total_metric -= np.log(max(away, epsilon))
            elif metric == "accuracy":
                pred = "H" if home >= draw and home >= away else ("D" if draw >= away else "A")
                total_metric -= 1 if pred == actual else 0

        return total_metric / len(results_df)

    # Optimize with constraints
    from scipy.optimize import minimize

    # Initial weights
    x0 = np.array([0.33, 0.33, 0.34])

    # Bounds: each weight between 0.05 and 0.8
    bounds = [(0.05, 0.8), (0.05, 0.8), (0.05, 0.8)]

    # Constraint: weights sum to 1
    constraint = {"type": "eq", "fun": lambda w: w.sum() - 1}

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraint,
    )

    optimal_weights = result.x / result.x.sum()

    return {
        "weights": optimal_weights,
        "pidc_weight": optimal_weights[0],
        "pi_weight": optimal_weights[1],
        "elo_weight": optimal_weights[2],
        "optimal_metric": result.fun,
        "success": result.success,
    }


def evaluate_ensemble(
    results_df: pd.DataFrame,
    weights: tuple[float, float, float],
    strategy: str = "fixed",
) -> dict:
    """Evaluate ensemble performance.

    Parameters
    ----------
    results_df : pd.DataFrame
        Backtest results.
    weights : tuple
        Ensemble weights (pidc, pi, elo).
    strategy : str
        Weighting strategy.

    Returns
    -------
    dict
        Evaluation metrics.
    """
    ensemble = EnsemblePredictor(weights=weights, strategy=strategy)

    predictions = []
    actuals = []

    for _, row in results_df.iterrows():
        # Get component predictions
        pidc = (row["pidc_home_prob"], row["pidc_draw_prob"], row["pidc_away_prob"])
        pi = (row["pi_home_prob"], row["pi_draw_prob"], row["pi_away_prob"])
        elo = (row["elo_home_prob"], row["elo_draw_prob"], row["elo_away_prob"])

        # Features for situational weighting
        features = {
            "rating_diff": row.get("pidc_home_prob", 0.5) - row.get("pidc_away_prob", 0.5),
        }

        # Combine
        pred = ensemble.combine_predictions(pidc, pi, elo, features)
        predictions.append(pred)
        actuals.append(row["actual"])

    # Calculate metrics
    correct = 0
    brier = 0
    log_loss = 0
    draw_correct = 0
    draw_predicted = 0
    draw_actual = 0

    confusion = {"H": {"H": 0, "D": 0, "A": 0},
                 "D": {"H": 0, "D": 0, "A": 0},
                 "A": {"H": 0, "D": 0, "A": 0}}

    for pred, actual in zip(predictions, actuals):
        # Predicted outcome
        predicted = pred.get_prediction()

        # Confusion matrix
        confusion[actual][predicted] += 1

        # Accuracy
        if predicted == actual:
            correct += 1

        # Draw tracking
        if predicted == "D":
            draw_predicted += 1
            if actual == "D":
                draw_correct += 1
        if actual == "D":
            draw_actual += 1

        # Brier score
        h_act = 1 if actual == "H" else 0
        d_act = 1 if actual == "D" else 0
        a_act = 1 if actual == "A" else 0

        brier += (pred.home_prob - h_act)**2
        brier += (pred.draw_prob - d_act)**2
        brier += (pred.away_prob - a_act)**2

        # Log loss
        epsilon = 1e-10
        if actual == "H":
            log_loss -= np.log(max(pred.home_prob, epsilon))
        elif actual == "D":
            log_loss -= np.log(max(pred.draw_prob, epsilon))
        else:
            log_loss -= np.log(max(pred.away_prob, epsilon))

    n = len(predictions)

    return {
        "accuracy": correct / n,
        "brier_score": brier / n,
        "log_loss": log_loss / n,
        "draw_predictions": draw_predicted,
        "draw_correct": draw_correct,
        "draw_precision": draw_correct / draw_predicted if draw_predicted > 0 else 0,
        "draw_recall": draw_correct / draw_actual if draw_actual > 0 else 0,
        "confusion_matrix": confusion,
        "predictions": predictions,
    }


def analyze_ensemble_vs_individuals(
    results_df: pd.DataFrame,
    ensemble_weights: tuple[float, float, float],
) -> dict:
    """Analyze where ensemble beats/loses to individual models.

    Returns cases where:
    - Ensemble correct, all individuals wrong
    - Ensemble wrong, at least one individual correct
    - Agreement cases vs disagreement cases
    """
    ensemble = EnsemblePredictor(weights=ensemble_weights, strategy="fixed")

    analysis = {
        "ensemble_only_correct": [],
        "ensemble_wrong_individual_correct": [],
        "all_agree_correct": 0,
        "all_agree_wrong": 0,
        "disagreement_ensemble_correct": 0,
        "disagreement_ensemble_wrong": 0,
    }

    for idx, row in results_df.iterrows():
        actual = row["actual"]

        # Individual predictions
        pidc_pred = _get_prediction(row, "pidc")
        pi_pred = _get_prediction(row, "pi")
        elo_pred = _get_prediction(row, "elo")

        # Ensemble prediction
        pidc = (row["pidc_home_prob"], row["pidc_draw_prob"], row["pidc_away_prob"])
        pi = (row["pi_home_prob"], row["pi_draw_prob"], row["pi_away_prob"])
        elo = (row["elo_home_prob"], row["elo_draw_prob"], row["elo_away_prob"])
        ens_pred_obj = ensemble.combine_predictions(pidc, pi, elo)
        ens_pred = ens_pred_obj.get_prediction()

        # Check correctness
        pidc_correct = pidc_pred == actual
        pi_correct = pi_pred == actual
        elo_correct = elo_pred == actual
        ens_correct = ens_pred == actual

        # All agree?
        all_agree = (pidc_pred == pi_pred == elo_pred)

        if all_agree:
            if ens_correct:
                analysis["all_agree_correct"] += 1
            else:
                analysis["all_agree_wrong"] += 1
        else:
            if ens_correct:
                analysis["disagreement_ensemble_correct"] += 1
            else:
                analysis["disagreement_ensemble_wrong"] += 1

        # Ensemble uniquely correct
        if ens_correct and not (pidc_correct or pi_correct or elo_correct):
            analysis["ensemble_only_correct"].append({
                "match": f"{row['home_team']} vs {row['away_team']}",
                "actual": actual,
                "ens_pred": ens_pred,
                "pidc_pred": pidc_pred,
                "pi_pred": pi_pred,
                "elo_pred": elo_pred,
            })

        # Ensemble wrong but someone was right
        if not ens_correct and (pidc_correct or pi_correct or elo_correct):
            analysis["ensemble_wrong_individual_correct"].append({
                "match": f"{row['home_team']} vs {row['away_team']}",
                "actual": actual,
                "ens_pred": ens_pred,
                "pidc_pred": pidc_pred,
                "pidc_correct": pidc_correct,
                "pi_pred": pi_pred,
                "pi_correct": pi_correct,
                "elo_pred": elo_pred,
                "elo_correct": elo_correct,
            })

    return analysis


def _get_prediction(row, prefix):
    """Get predicted outcome from row."""
    h = row[f"{prefix}_home_prob"]
    d = row[f"{prefix}_draw_prob"]
    a = row[f"{prefix}_away_prob"]

    if h >= d and h >= a:
        return "H"
    elif d >= a:
        return "D"
    return "A"
