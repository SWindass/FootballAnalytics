"""Neural Network Stacker - learns optimal model combination.

Takes predictions from ELO, Poisson, and XGBoost models plus additional
features and learns the optimal way to combine them.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from typing import Optional
from datetime import datetime, timezone
from decimal import Decimal

import structlog
from sqlalchemy import select

from app.db.database import SyncSessionLocal
from app.db.models import Match, MatchAnalysis, MatchStatus, EloRating, TeamStats, Referee, TeamFixture

logger = structlog.get_logger()

# Model save path
MODEL_DIR = Path(__file__).parent / "saved"
MODEL_PATH = MODEL_DIR / "neural_stacker.pt"
METADATA_PATH = MODEL_DIR / "neural_stacker_meta.json"

# Model version - increment when features change
MODEL_VERSION = 9  # v9: 35 features + market odds from historical betting data


class MatchPredictorNet(nn.Module):
    """Neural network for match outcome prediction.

    Architecture:
    - Input: Model predictions + contextual features
    - Hidden layers with dropout for regularization
    - Output: Softmax probabilities for home/draw/away
    """

    def __init__(self, input_size: int = 15, hidden_sizes: list = [64, 32], dropout: float = 0.3):
        super().__init__()

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_size = hidden_size

        # Output layer - 3 classes (home, draw, away)
        layers.append(nn.Linear(prev_size, 3))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.network(x)
        return torch.softmax(logits, dim=1)


class NeuralStacker:
    """Manages training and inference for the neural stacker model."""

    # Feature names for documentation
    # Extended features include injuries, manager, referee, and congestion data
    FEATURE_NAMES = [
        # Model predictions (8 features)
        "elo_home_prob", "elo_draw_prob", "elo_away_prob",
        "poisson_home_prob", "poisson_draw_prob", "poisson_away_prob",
        "poisson_over_2_5", "poisson_btts",
        # Team strength (1 feature)
        "elo_diff",  # home_elo - away_elo, normalized
        # Form (2 features)
        "home_form_points",  # points from last 5 games, normalized
        "away_form_points",
        # Goals (4 features)
        "home_goals_avg",  # average goals scored, normalized
        "away_goals_avg",
        "home_conceded_avg",  # average goals conceded, normalized
        "away_conceded_avg",
        # Injuries (4 features)
        "home_injury_count",  # normalized 0-1
        "away_injury_count",
        "home_key_players_out",  # normalized 0-1
        "away_key_players_out",
        # Manager (2 features)
        "home_new_manager",  # 1 if new manager (< 5 games), else 0
        "away_new_manager",
        # Referee (1 feature)
        "referee_home_bias",  # referee's historical home win %, normalized
        # Head-to-head (3 features)
        "h2h_home_dominance",  # (home_wins - away_wins) / total_h2h, normalized
        "h2h_home_goals_avg",  # avg goals by home team in H2H
        "h2h_away_goals_avg",  # avg goals by away team in H2H
        # Home/Away specific form (3 features)
        "home_home_ppg",  # home team's points per game at home, normalized
        "away_away_ppg",  # away team's points per game away, normalized
        "venue_advantage",  # home_home_ppg - away_away_ppg
        # Recency-weighted form (4 features)
        "home_recent_form",  # exponential decay weighted form (last 5)
        "away_recent_form",
        "home_momentum",  # recent form - older form (positive = improving)
        "away_momentum",
        # Market odds - implied probabilities from betting markets (3 features)
        "market_home_prob",  # Market consensus home win probability
        "market_draw_prob",  # Market consensus draw probability
        "market_away_prob",  # Market consensus away win probability
    ]

    def __init__(self):
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = len(self.FEATURE_NAMES)

    def _ensure_model_dir(self):
        """Ensure model directory exists."""
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

    def build_model(self, hidden_sizes: list = [64, 32], dropout: float = 0.3):
        """Build a new model."""
        self.model = MatchPredictorNet(
            input_size=self.input_size,
            hidden_sizes=hidden_sizes,
            dropout=dropout
        ).to(self.device)
        return self.model

    def load_model(self) -> bool:
        """Load saved model if exists and version matches."""
        if not MODEL_PATH.exists():
            logger.warning("No saved model found")
            return False

        # Check version compatibility and get architecture
        hidden_sizes = [64, 32]  # Default architecture
        if METADATA_PATH.exists():
            try:
                with open(METADATA_PATH) as f:
                    metadata = json.load(f)
                saved_version = metadata.get("model_version", 1)
                if saved_version != MODEL_VERSION:
                    logger.warning(
                        f"Model version mismatch: saved={saved_version}, current={MODEL_VERSION}. "
                        "Please retrain the model."
                    )
                    return False
                # Get saved architecture if available
                hidden_sizes = metadata.get("hidden_sizes", [128, 64, 32])
            except Exception:
                pass

        try:
            self.model = MatchPredictorNet(
                input_size=self.input_size,
                hidden_sizes=hidden_sizes
            ).to(self.device)
            self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device, weights_only=True))
            self.model.eval()
            logger.info("Loaded neural stacker model")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def save_model(self, metadata: dict = None, hidden_sizes: list = None):
        """Save model and metadata."""
        self._ensure_model_dir()

        torch.save(self.model.state_dict(), MODEL_PATH)

        if metadata is None:
            metadata = {}
        metadata["saved_at"] = datetime.now(timezone.utc).isoformat()
        metadata["model_version"] = MODEL_VERSION
        metadata["input_size"] = self.input_size
        metadata["features"] = self.FEATURE_NAMES
        if hidden_sizes:
            metadata["hidden_sizes"] = hidden_sizes

        with open(METADATA_PATH, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved model to {MODEL_PATH}")

    def prepare_training_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Prepare training data from historical matches.

        OPTIMIZED: Batch loads all data upfront to avoid N+1 query problems.
        Previous version made ~14 queries per match (168,000+ total queries).
        This version makes ~10 bulk queries total.

        Returns:
            Tuple of (features, labels) as numpy arrays
        """
        with SyncSessionLocal() as session:
            logger.info("Loading data in bulk...")

            # 1. Get finished matches with analysis
            stmt = (
                select(Match, MatchAnalysis)
                .join(MatchAnalysis, Match.id == MatchAnalysis.match_id)
                .where(Match.status == MatchStatus.FINISHED)
                .where(MatchAnalysis.elo_home_prob.isnot(None))
                .where(MatchAnalysis.poisson_home_prob.isnot(None))
                .order_by(Match.kickoff_time)
            )
            results = list(session.execute(stmt).all())
            logger.info(f"Found {len(results)} matches with predictions")

            # 2. Bulk load all TeamStats into dict: (team_id, season, matchweek) -> TeamStats
            all_team_stats = list(session.execute(select(TeamStats)).scalars().all())
            team_stats_lookup = {
                (ts.team_id, ts.season, ts.matchweek): ts for ts in all_team_stats
            }
            logger.info(f"Loaded {len(all_team_stats)} team stats records")

            # 3. Bulk load all EloRatings into dict: (team_id, season, matchweek) -> EloRating
            all_elo_ratings = list(session.execute(select(EloRating)).scalars().all())
            elo_lookup = {
                (er.team_id, er.season, er.matchweek): er for er in all_elo_ratings
            }
            logger.info(f"Loaded {len(all_elo_ratings)} ELO ratings")

            # 4. Bulk load all Referees into dict: referee_id -> Referee
            all_referees = list(session.execute(select(Referee)).scalars().all())
            referee_lookup = {r.id: r for r in all_referees}
            logger.info(f"Loaded {len(all_referees)} referees")

            # 5. Bulk load all finished matches for H2H/venue/recency calculations
            all_finished_matches = list(
                session.execute(
                    select(Match)
                    .where(Match.status == MatchStatus.FINISHED)
                    .order_by(Match.kickoff_time)
                ).scalars().all()
            )
            logger.info(f"Loaded {len(all_finished_matches)} finished matches for feature calculation")

            # Build match lookup structures for efficient feature calculation
            # matches_by_teams[(team1_id, team2_id)] = list of matches (sorted by date)
            from collections import defaultdict
            h2h_matches = defaultdict(list)
            team_home_matches = defaultdict(list)  # team_id -> list of home matches
            team_away_matches = defaultdict(list)  # team_id -> list of away matches
            team_all_matches = defaultdict(list)   # team_id -> list of all matches

            for m in all_finished_matches:
                # H2H: store with canonical key (sorted team ids)
                key = tuple(sorted([m.home_team_id, m.away_team_id]))
                h2h_matches[key].append(m)

                # Venue-specific
                team_home_matches[m.home_team_id].append(m)
                team_away_matches[m.away_team_id].append(m)

                # All matches
                team_all_matches[m.home_team_id].append(m)
                team_all_matches[m.away_team_id].append(m)

            logger.info("Built match lookup structures, processing features...")

            features = []
            labels = []

            for i, (match, analysis) in enumerate(results):
                # Lookup team stats
                home_stats = team_stats_lookup.get(
                    (match.home_team_id, match.season, match.matchweek - 1)
                )
                away_stats = team_stats_lookup.get(
                    (match.away_team_id, match.season, match.matchweek - 1)
                )

                # Lookup ELO ratings
                home_elo = elo_lookup.get(
                    (match.home_team_id, match.season, match.matchweek - 1)
                )
                away_elo = elo_lookup.get(
                    (match.away_team_id, match.season, match.matchweek - 1)
                )

                # Lookup referee
                referee = referee_lookup.get(match.referee_id) if match.referee_id else None

                # Calculate H2H features from pre-loaded data
                h2h_features = self._calculate_h2h_from_cache(
                    h2h_matches, match.home_team_id, match.away_team_id, match.kickoff_time
                )

                # Calculate venue-specific form from pre-loaded data
                home_home_ppg = self._calculate_venue_form_from_cache(
                    team_home_matches[match.home_team_id],
                    match.home_team_id, is_home=True,
                    before_date=match.kickoff_time, season=match.season
                )
                away_away_ppg = self._calculate_venue_form_from_cache(
                    team_away_matches[match.away_team_id],
                    match.away_team_id, is_home=False,
                    before_date=match.kickoff_time, season=match.season
                )

                # Calculate recency-weighted form from pre-loaded data
                home_recency = self._calculate_recency_from_cache(
                    team_all_matches[match.home_team_id],
                    match.home_team_id, match.kickoff_time
                )
                away_recency = self._calculate_recency_from_cache(
                    team_all_matches[match.away_team_id],
                    match.away_team_id, match.kickoff_time
                )

                # Build feature vector (no rest days needed - removed those queries)
                feature = self._build_feature_vector(
                    analysis, home_stats, away_stats, home_elo, away_elo, referee,
                    None, None,  # rest days not used in current feature set
                    match.home_team_id, match.away_team_id,
                    None, None,  # prev fixtures not used
                    h2h_features, home_home_ppg, away_away_ppg,
                    home_recency, away_recency
                )

                if feature is not None:
                    features.append(feature)

                    # Label: 0=home win, 1=draw, 2=away win
                    if match.home_score > match.away_score:
                        labels.append(0)
                    elif match.home_score == match.away_score:
                        labels.append(1)
                    else:
                        labels.append(2)

                # Progress indicator
                if (i + 1) % 2000 == 0:
                    logger.info(f"Processed {i + 1}/{len(results)} matches")

            logger.info(f"Prepared {len(features)} training samples")

            return np.array(features, dtype=np.float32), np.array(labels, dtype=np.int64)

    def _calculate_h2h_from_cache(
        self,
        h2h_matches: dict,
        home_team_id: int,
        away_team_id: int,
        before_date: datetime,
        max_matches: int = 10,
    ) -> tuple[float, float, float]:
        """Calculate H2H features from pre-loaded matches."""
        key = tuple(sorted([home_team_id, away_team_id]))
        matches = h2h_matches.get(key, [])

        # Filter to matches before this date and limit
        matches = [m for m in matches if m.kickoff_time < before_date][-max_matches:]

        if len(matches) < 2:
            return 0.0, 0.0, 0.0

        home_wins = 0
        away_wins = 0
        home_goals = 0
        away_goals = 0

        for m in matches:
            if m.home_team_id == home_team_id:
                home_goals += m.home_score or 0
                away_goals += m.away_score or 0
                if m.home_score > m.away_score:
                    home_wins += 1
                elif m.home_score < m.away_score:
                    away_wins += 1
            else:
                home_goals += m.away_score or 0
                away_goals += m.home_score or 0
                if m.away_score > m.home_score:
                    home_wins += 1
                elif m.away_score < m.home_score:
                    away_wins += 1

        n = len(matches)
        h2h_home_dominance = (home_wins - away_wins) / n
        h2h_home_goals_avg = home_goals / n / 3
        h2h_away_goals_avg = away_goals / n / 3

        return h2h_home_dominance, h2h_home_goals_avg, h2h_away_goals_avg

    def _calculate_venue_form_from_cache(
        self,
        venue_matches: list,
        team_id: int,
        is_home: bool,
        before_date: datetime,
        season: str,
    ) -> float:
        """Calculate venue form from pre-loaded matches."""
        matches = [
            m for m in venue_matches
            if m.kickoff_time < before_date and m.season == season
        ]

        if len(matches) < 2:
            return 0.5

        points = 0
        for m in matches:
            if is_home:
                if m.home_score > m.away_score:
                    points += 3
                elif m.home_score == m.away_score:
                    points += 1
            else:
                if m.away_score > m.home_score:
                    points += 3
                elif m.home_score == m.away_score:
                    points += 1

        ppg = points / len(matches)
        return ppg / 3

    def _calculate_recency_from_cache(
        self,
        team_matches: list,
        team_id: int,
        before_date: datetime,
        n_matches: int = 10,
        decay: float = 0.8,
    ) -> tuple[float, float]:
        """Calculate recency-weighted form from pre-loaded matches."""
        # Filter and get most recent N matches
        matches = [m for m in team_matches if m.kickoff_time < before_date]
        matches = sorted(matches, key=lambda m: m.kickoff_time, reverse=True)[:n_matches]

        if len(matches) < 3:
            return 0.0, 0.0

        # Calculate points for each match
        points_list = []
        for m in matches:
            if m.home_team_id == team_id:
                pts = 3 if m.home_score > m.away_score else (1 if m.home_score == m.away_score else 0)
            else:
                pts = 3 if m.away_score > m.home_score else (1 if m.home_score == m.away_score else 0)
            points_list.append(pts)

        # Recency-weighted form
        recent_weighted = 0.0
        recent_weight_sum = 0.0
        for i, pts in enumerate(points_list[:5]):
            w = decay ** i
            recent_weighted += pts * w
            recent_weight_sum += w

        recent_form = (recent_weighted / recent_weight_sum / 3) if recent_weight_sum > 0 else 0.5

        # Momentum
        if len(points_list) >= 6:
            recent_3_avg = sum(points_list[:3]) / 3
            older_3_avg = sum(points_list[3:6]) / 3
            momentum = (recent_3_avg - older_3_avg) / 3
        else:
            momentum = 0.0

        recent_form_centered = (recent_form - 0.5) * 2
        return recent_form_centered, momentum

    def _build_feature_vector(
        self,
        analysis: MatchAnalysis,
        home_stats: Optional[TeamStats],
        away_stats: Optional[TeamStats],
        home_elo: Optional[EloRating],
        away_elo: Optional[EloRating],
        referee: Optional[Referee] = None,
        home_rest_days: Optional[int] = None,
        away_rest_days: Optional[int] = None,
        home_team_id: Optional[int] = None,
        away_team_id: Optional[int] = None,
        home_prev_fixture: Optional[TeamFixture] = None,
        away_prev_fixture: Optional[TeamFixture] = None,
        h2h_features: Optional[tuple[float, float, float]] = None,
        home_home_ppg: float = 0.5,
        away_away_ppg: float = 0.5,
        home_recency: Optional[tuple[float, float]] = None,
        away_recency: Optional[tuple[float, float]] = None,
    ) -> Optional[list]:
        """Build feature vector for a match.

        Extended features include injuries, manager status, referee bias, H2H, venue form, and recency.
        """

        def safe_float(val, default=0.5):
            if val is None:
                return default
            if isinstance(val, Decimal):
                return float(val)
            return float(val)

        try:
            # Model predictions (8 features)
            elo_home = safe_float(analysis.elo_home_prob, 0.4)
            elo_draw = safe_float(analysis.elo_draw_prob, 0.25)
            elo_away = safe_float(analysis.elo_away_prob, 0.35)

            poisson_home = safe_float(analysis.poisson_home_prob, 0.4)
            poisson_draw = safe_float(analysis.poisson_draw_prob, 0.25)
            poisson_away = safe_float(analysis.poisson_away_prob, 0.35)

            poisson_over = safe_float(analysis.poisson_over_2_5_prob, 0.5)
            poisson_btts = safe_float(analysis.poisson_btts_prob, 0.5)

            # ELO difference (1 feature)
            home_elo_val = safe_float(home_elo.rating, 1500) if home_elo else 1500
            away_elo_val = safe_float(away_elo.rating, 1500) if away_elo else 1500
            elo_diff = (home_elo_val - away_elo_val) / 400  # Normalize to ~[-1, 1]

            # Form (2 features)
            home_form = safe_float(home_stats.form_points, 7.5) / 15 if home_stats else 0.5
            away_form = safe_float(away_stats.form_points, 7.5) / 15 if away_stats else 0.5

            # Goals (4 features)
            home_goals = safe_float(home_stats.avg_goals_scored, 1.4) / 3 if home_stats else 0.47
            away_goals = safe_float(away_stats.avg_goals_scored, 1.4) / 3 if away_stats else 0.47
            home_conceded = safe_float(home_stats.avg_goals_conceded, 1.4) / 3 if home_stats else 0.47
            away_conceded = safe_float(away_stats.avg_goals_conceded, 1.4) / 3 if away_stats else 0.47

            # Injuries (4 features) - normalized to 0-1 scale
            # injury_count: 0-10 scale, key_players_out: 0-5 scale
            home_injury_count = min(home_stats.injury_count, 10) / 10 if home_stats else 0.0
            away_injury_count = min(away_stats.injury_count, 10) / 10 if away_stats else 0.0
            home_key_out = min(home_stats.key_players_out, 5) / 5 if home_stats else 0.0
            away_key_out = min(away_stats.key_players_out, 5) / 5 if away_stats else 0.0

            # Manager (2 features) - binary for new manager effect
            home_new_manager = 1.0 if (home_stats and home_stats.is_new_manager) else 0.0
            away_new_manager = 1.0 if (away_stats and away_stats.is_new_manager) else 0.0

            # Referee (1 feature) - home win bias
            # Average EPL home win rate is ~46%, so normalize around that
            if referee and referee.home_win_pct is not None:
                referee_home_bias = (float(referee.home_win_pct) - 46) / 20  # Normalize to ~[-1, 1]
            else:
                referee_home_bias = 0.0  # Neutral if unknown

            # H2H features
            if h2h_features:
                h2h_home_dominance, h2h_home_goals, h2h_away_goals = h2h_features
            else:
                h2h_home_dominance, h2h_home_goals, h2h_away_goals = 0.0, 0.0, 0.0

            # Venue advantage
            venue_advantage = home_home_ppg - away_away_ppg  # Range roughly [-1, 1]

            # Recency-weighted form and momentum
            if home_recency:
                home_recent_form, home_momentum = home_recency
            else:
                home_recent_form, home_momentum = 0.0, 0.0

            if away_recency:
                away_recent_form, away_momentum = away_recency
            else:
                away_recent_form, away_momentum = 0.0, 0.0

            # Market odds - historical betting odds converted to implied probabilities
            # These represent "wisdom of crowds" from betting markets
            market_home_prob = 0.4  # Default if no odds available
            market_draw_prob = 0.27
            market_away_prob = 0.33
            if analysis.features:
                hist_odds = analysis.features.get("historical_odds", {})
                if hist_odds:
                    market_home_prob = hist_odds.get("implied_home_prob", 0.4)
                    market_draw_prob = hist_odds.get("implied_draw_prob", 0.27)
                    market_away_prob = hist_odds.get("implied_away_prob", 0.33)
                # Also check for market consensus from live odds
                elif "market_home_prob" in analysis.features:
                    market_home_prob = analysis.features.get("market_home_prob", 0.4)
                    market_draw_prob = analysis.features.get("market_draw_prob", 0.27)
                    market_away_prob = analysis.features.get("market_away_prob", 0.33)

            return [
                # Model predictions
                elo_home, elo_draw, elo_away,
                poisson_home, poisson_draw, poisson_away,
                poisson_over, poisson_btts,
                # Team strength
                elo_diff,
                # Form
                home_form, away_form,
                # Goals
                home_goals, away_goals,
                home_conceded, away_conceded,
                # Injuries
                home_injury_count, away_injury_count,
                home_key_out, away_key_out,
                # Manager
                home_new_manager, away_new_manager,
                # Referee
                referee_home_bias,
                # H2H
                h2h_home_dominance, h2h_home_goals, h2h_away_goals,
                # Venue form
                home_home_ppg, away_away_ppg, venue_advantage,
                # Recency-weighted form
                home_recent_form, away_recent_form,
                home_momentum, away_momentum,
                # Market odds (wisdom of crowds)
                market_home_prob, market_draw_prob, market_away_prob,
            ]

        except Exception as e:
            logger.warning(f"Failed to build feature vector: {e}")
            return None

    def _calculate_rest_days(
        self,
        session,
        team_id: int,
        match_date: datetime,
    ) -> Optional[int]:
        """Calculate days since team's last match.

        Uses TeamFixture table which includes all competitions (PL, CL, etc.)
        for accurate rest calculation.

        Args:
            session: Database session
            team_id: Team ID
            match_date: Date of current match

        Returns:
            Number of days since last match, or None if no previous match
        """
        # Find most recent fixture before this one (any competition)
        stmt = (
            select(TeamFixture)
            .where(TeamFixture.team_id == team_id)
            .where(TeamFixture.kickoff_time < match_date)
            .order_by(TeamFixture.kickoff_time.desc())
            .limit(1)
        )
        prev_fixture = session.execute(stmt).scalar_one_or_none()

        if not prev_fixture:
            return None

        delta = match_date - prev_fixture.kickoff_time
        return delta.days

    def _calculate_h2h_features(
        self,
        session,
        home_team_id: int,
        away_team_id: int,
        before_date: datetime,
        max_matches: int = 10,
    ) -> tuple[float, float, float]:
        """Calculate head-to-head features from historical matches.

        Args:
            session: Database session
            home_team_id: Home team ID
            away_team_id: Away team ID
            before_date: Only consider matches before this date
            max_matches: Maximum H2H matches to consider

        Returns:
            Tuple of (h2h_home_dominance, h2h_home_goals_avg, h2h_away_goals_avg)
        """
        # Get H2H matches (either team at home)
        stmt = (
            select(Match)
            .where(Match.status == MatchStatus.FINISHED)
            .where(Match.kickoff_time < before_date)
            .where(
                ((Match.home_team_id == home_team_id) & (Match.away_team_id == away_team_id)) |
                ((Match.home_team_id == away_team_id) & (Match.away_team_id == home_team_id))
            )
            .order_by(Match.kickoff_time.desc())
            .limit(max_matches)
        )
        h2h_matches = list(session.execute(stmt).scalars().all())

        if len(h2h_matches) < 2:  # Need minimum sample
            return 0.0, 0.0, 0.0

        # Calculate stats from perspective of current home team
        home_wins = 0
        away_wins = 0
        draws = 0
        home_goals = 0
        away_goals = 0

        for m in h2h_matches:
            if m.home_team_id == home_team_id:
                # Current home team was home in this H2H
                home_goals += m.home_score or 0
                away_goals += m.away_score or 0
                if m.home_score > m.away_score:
                    home_wins += 1
                elif m.home_score < m.away_score:
                    away_wins += 1
                else:
                    draws += 1
            else:
                # Current home team was away in this H2H
                home_goals += m.away_score or 0
                away_goals += m.home_score or 0
                if m.away_score > m.home_score:
                    home_wins += 1
                elif m.away_score < m.home_score:
                    away_wins += 1
                else:
                    draws += 1

        n = len(h2h_matches)
        # Dominance: (wins - losses) / total, range [-1, 1]
        h2h_home_dominance = (home_wins - away_wins) / n
        # Average goals
        h2h_home_goals_avg = home_goals / n / 3  # Normalize to ~[0, 1]
        h2h_away_goals_avg = away_goals / n / 3

        return h2h_home_dominance, h2h_home_goals_avg, h2h_away_goals_avg

    def _calculate_venue_form(
        self,
        session,
        team_id: int,
        is_home: bool,
        before_date: datetime,
        season: str,
    ) -> float:
        """Calculate team's points per game at home or away.

        Args:
            session: Database session
            team_id: Team ID
            is_home: True for home form, False for away form
            before_date: Only consider matches before this date
            season: Current season

        Returns:
            Points per game normalized to [0, 1]
        """
        if is_home:
            stmt = (
                select(Match)
                .where(Match.status == MatchStatus.FINISHED)
                .where(Match.home_team_id == team_id)
                .where(Match.season == season)
                .where(Match.kickoff_time < before_date)
            )
        else:
            stmt = (
                select(Match)
                .where(Match.status == MatchStatus.FINISHED)
                .where(Match.away_team_id == team_id)
                .where(Match.season == season)
                .where(Match.kickoff_time < before_date)
            )

        matches = list(session.execute(stmt).scalars().all())

        if len(matches) < 2:
            return 0.5  # Neutral default

        points = 0
        for m in matches:
            if is_home:
                if m.home_score > m.away_score:
                    points += 3
                elif m.home_score == m.away_score:
                    points += 1
            else:
                if m.away_score > m.home_score:
                    points += 3
                elif m.home_score == m.away_score:
                    points += 1

        ppg = points / len(matches)
        return ppg / 3  # Normalize to [0, 1]

    def _calculate_recency_form(
        self,
        session,
        team_id: int,
        before_date: datetime,
        n_matches: int = 10,
        decay: float = 0.8,
    ) -> tuple[float, float]:
        """Calculate recency-weighted form and momentum.

        Args:
            session: Database session
            team_id: Team ID
            before_date: Only consider matches before this date
            n_matches: Number of recent matches to consider
            decay: Exponential decay factor (0.8 means each older game worth 80% of previous)

        Returns:
            Tuple of (recent_form, momentum) both normalized to roughly [-1, 1]
        """
        # Get recent matches for this team
        stmt = (
            select(Match)
            .where(Match.status == MatchStatus.FINISHED)
            .where(Match.kickoff_time < before_date)
            .where(
                (Match.home_team_id == team_id) | (Match.away_team_id == team_id)
            )
            .order_by(Match.kickoff_time.desc())
            .limit(n_matches)
        )
        matches = list(session.execute(stmt).scalars().all())

        if len(matches) < 3:
            return 0.0, 0.0

        # Calculate points for each match
        points_list = []
        for m in matches:
            if m.home_team_id == team_id:
                pts = 3 if m.home_score > m.away_score else (1 if m.home_score == m.away_score else 0)
            else:
                pts = 3 if m.away_score > m.home_score else (1 if m.home_score == m.away_score else 0)
            points_list.append(pts)

        # Recency-weighted form (last 5 games)
        recent_weighted = 0.0
        recent_weight_sum = 0.0
        for i, pts in enumerate(points_list[:5]):
            w = decay ** i
            recent_weighted += pts * w
            recent_weight_sum += w

        recent_form = (recent_weighted / recent_weight_sum / 3) if recent_weight_sum > 0 else 0.5

        # Momentum: compare recent 3 games vs previous 3 games
        if len(points_list) >= 6:
            recent_3_avg = sum(points_list[:3]) / 3
            older_3_avg = sum(points_list[3:6]) / 3
            momentum = (recent_3_avg - older_3_avg) / 3  # Normalize to roughly [-1, 1]
        else:
            momentum = 0.0

        # Shift recent_form to center around 0 for better neural net training
        recent_form_centered = (recent_form - 0.5) * 2  # Now in [-1, 1]

        return recent_form_centered, momentum

    def train(
        self,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        validation_split: float = 0.2,
    ) -> dict:
        """Train the neural network using time-series cross-validation.

        Data is split chronologically: earlier matches for training,
        recent matches for validation. This prevents future data leakage.

        Returns:
            Training metrics
        """
        # Prepare data (already sorted by kickoff_time in prepare_training_data)
        X, y = self.prepare_training_data()

        if len(X) < 50:
            raise ValueError(f"Not enough training data: {len(X)} samples")

        # TIME-SERIES SPLIT: Use last N% of matches for validation
        # This simulates real-world prediction where we train on past, predict future
        n_val = int(len(X) * validation_split)
        train_end = len(X) - n_val

        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:], y[train_end:]

        logger.info(f"Time-series split: {len(X_train)} train, {len(X_val)} validation")

        # Create data loaders
        train_dataset = TensorDataset(
            torch.tensor(X_train),
            torch.tensor(y_train)
        )
        # drop_last=True to avoid BatchNorm issues with batch size of 1
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        X_val_t = torch.tensor(X_val).to(self.device)
        y_val_t = torch.tensor(y_val).to(self.device)

        # Build model with deeper architecture
        hidden_sizes = [128, 64, 32]
        self.build_model(hidden_sizes=hidden_sizes, dropout=0.4)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

        # Training loop
        best_val_acc = 0
        best_epoch = 0
        history = {"train_loss": [], "val_loss": [], "val_acc": []}

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_t)
                val_loss = criterion(val_outputs, y_val_t).item()
                val_preds = val_outputs.argmax(dim=1)
                val_acc = (val_preds == y_val_t).float().mean().item()

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            scheduler.step(val_loss)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                # Save best model
                self.save_model({
                    "epoch": epoch,
                    "val_acc": val_acc,
                    "train_samples": len(X_train),
                    "val_samples": len(X_val),
                }, hidden_sizes=hidden_sizes)

            if epoch % 20 == 0:
                logger.info(
                    f"Epoch {epoch}: train_loss={train_loss:.4f}, "
                    f"val_loss={val_loss:.4f}, val_acc={val_acc:.3f}"
                )

        # Load best model
        self.load_model()

        logger.info(f"Training complete. Best val_acc={best_val_acc:.3f} at epoch {best_epoch}")

        return {
            "best_val_acc": best_val_acc,
            "best_epoch": best_epoch,
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "final_train_loss": history["train_loss"][-1],
            "final_val_loss": history["val_loss"][-1],
        }

    def predict(
        self,
        analysis: MatchAnalysis,
        home_stats: Optional[TeamStats] = None,
        away_stats: Optional[TeamStats] = None,
        home_elo: Optional[EloRating] = None,
        away_elo: Optional[EloRating] = None,
        referee: Optional[Referee] = None,
        home_rest_days: Optional[int] = None,
        away_rest_days: Optional[int] = None,
        home_team_id: Optional[int] = None,
        away_team_id: Optional[int] = None,
        home_prev_fixture: Optional[TeamFixture] = None,
        away_prev_fixture: Optional[TeamFixture] = None,
    ) -> tuple[float, float, float]:
        """Predict match probabilities.

        Args:
            analysis: Match analysis with model predictions
            home_stats: Home team statistics (includes injuries, manager info)
            away_stats: Away team statistics
            home_elo: Home team ELO rating
            away_elo: Away team ELO rating
            referee: Referee assigned to match (for bias calculation)
            home_rest_days: Days since home team's last match
            away_rest_days: Days since away team's last match
            home_team_id: Home team ID (for congestion calculation)
            away_team_id: Away team ID (for congestion calculation)
            home_prev_fixture: Home team's previous fixture (for travel fatigue)
            away_prev_fixture: Away team's previous fixture (for travel fatigue)

        Returns:
            Tuple of (home_prob, draw_prob, away_prob)
        """
        if self.model is None:
            if not self.load_model():
                # Fall back to simple average if no model
                logger.warning("No model available, using simple average")
                return self._fallback_prediction(analysis)

        feature = self._build_feature_vector(
            analysis, home_stats, away_stats, home_elo, away_elo, referee,
            home_rest_days, away_rest_days,
            home_team_id, away_team_id,
            home_prev_fixture, away_prev_fixture
        )

        if feature is None:
            return self._fallback_prediction(analysis)

        self.model.eval()
        with torch.no_grad():
            X = torch.tensor([feature], dtype=torch.float32).to(self.device)
            probs = self.model(X)[0].cpu().numpy()

        return float(probs[0]), float(probs[1]), float(probs[2])

    def _fallback_prediction(self, analysis: MatchAnalysis) -> tuple[float, float, float]:
        """Fallback to weighted average when model unavailable."""
        def safe_float(val, default=0.33):
            return float(val) if val else default

        # Simple weighted average (original method)
        elo_w, poisson_w = 0.45, 0.55

        home = (
            safe_float(analysis.elo_home_prob) * elo_w +
            safe_float(analysis.poisson_home_prob) * poisson_w
        )
        draw = (
            safe_float(analysis.elo_draw_prob) * elo_w +
            safe_float(analysis.poisson_draw_prob) * poisson_w
        )
        away = (
            safe_float(analysis.elo_away_prob) * elo_w +
            safe_float(analysis.poisson_away_prob) * poisson_w
        )

        # Normalize
        total = home + draw + away
        return home/total, draw/total, away/total


def train_neural_stacker():
    """Train the neural stacker model."""
    stacker = NeuralStacker()
    metrics = stacker.train(epochs=100, batch_size=32)
    return metrics


def train_xgboost_stacker():
    """Train with XGBoost instead of neural network."""
    import xgboost as xgb
    from sklearn.metrics import accuracy_score

    stacker = NeuralStacker()
    X, y = stacker.prepare_training_data()

    # Time-series split
    n_val = int(len(X) * 0.2)
    train_end = len(X) - n_val
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:], y[train_end:]

    logger.info(f"XGBoost training: {len(X_train)} train, {len(X_val)} val samples")

    # Train XGBoost
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softprob',
        num_class=3,
        eval_metric='mlogloss',
        early_stopping_rounds=20,
        random_state=42,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=True
    )

    # Evaluate
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)

    logger.info(f"XGBoost validation accuracy: {accuracy:.3f}")

    # Feature importance
    importance = model.feature_importances_
    feature_importance = list(zip(stacker.FEATURE_NAMES, importance))
    feature_importance.sort(key=lambda x: x[1], reverse=True)

    print("\nTop 10 most important features:")
    for name, imp in feature_importance[:10]:
        print(f"  {name}: {imp:.4f}")

    return {
        "val_acc": accuracy,
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "feature_importance": feature_importance,
    }


if __name__ == "__main__":
    import logging
    import sys

    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

    if len(sys.argv) > 1 and sys.argv[1] == "--xgboost":
        print("Training XGBoost Stacker...")
        metrics = train_xgboost_stacker()
        print(f"\nXGBoost training complete:")
        print(f"  Validation accuracy: {metrics['val_acc']:.1%}")
    else:
        print("Training Neural Stacker...")
        metrics = train_neural_stacker()
        print(f"\nTraining complete:")
        print(f"  Best validation accuracy: {metrics['best_val_acc']:.1%}")
        print(f"  Training samples: {metrics['train_samples']}")
        print(f"  Validation samples: {metrics['val_samples']}")
