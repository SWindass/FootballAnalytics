"""Team-specific congestion impact model.

Calculates how each team performs under fixture congestion, taking into
account squad depth and European travel fatigue.

EXPERIMENTAL FINDINGS (Jan 2026):
---------------------------------
This model was tested as a feature in the neural stacker but did NOT improve
prediction accuracy:

    - With team-specific congestion features: 50.6% accuracy
    - With simplified rest_advantage feature: 50.8% accuracy
    - Without rest/congestion features: 52.8% accuracy

RESEARCH VALIDATION:
Our findings are consistent with published academic research:

    1. Meta-analysis (PMC7846542) found "no significant effect of fixture
       congestion on total distance covered" (p=0.134) - players maintain
       performance through pacing strategies.

    2. Research shows the relationship is NON-LINEAR - "linear statistical
       techniques may have a limited capacity to model such relationships"
       (PMC10850290). 72-hour recovery threshold has complex effects.

    3. Squad STABILITY may matter more than rest - Bekris et al found more
       player changes had NEGATIVE effects on league performance. ESPN analysis
       showed "teams that prioritized stability had successful seasons."

    4. 2025 SAGE study found counterintuitive effect: under congestion,
       offensive strength decreases BUT defensive strength improves at home.

    5. The main risk of congestion is INJURY, not match performance decline.

Team coefficient analysis revealed patterns consistent with squad depth theory:

    Teams that HANDLE congestion well (positive coefficient):
        - Man City: +0.214  (deep squad, rotation options)
        - Liverpool: +0.109  (but notably succeeded with minimal rotation)
        - Spurs: +0.085

    Teams that STRUGGLE with congestion (negative coefficient):
        - Southampton: -0.309
        - Brighton: -0.205
        - Wolves: -0.171

FUTURE IMPROVEMENTS:
    - Try non-linear modeling (polynomial features, neural network sub-layer)
    - Model injury probability separately, then feed into match prediction
    - Consider squad rotation stability as a feature instead of rest days

This module is retained for potential future use in dashboards or alternative
analysis, but is NOT currently used in match predictions.
"""

from collections import defaultdict
from datetime import datetime, timedelta
from typing import Optional
from decimal import Decimal

import structlog
from sqlalchemy import select

from app.db.database import SyncSessionLocal
from app.db.models import Match, MatchStatus, Team, TeamFixture

logger = structlog.get_logger()


# European travel distance categories (from UK)
# 1 = Short (1-2 hour flight)
# 2 = Medium (2-3 hour flight)
# 3 = Long (3-4 hour flight)
# 4 = Very Long (4+ hour flight)
TRAVEL_CATEGORIES = {
    # Short haul
    "Paris": 1, "PSG": 1, "Monaco": 1, "Lyon": 1, "Marseille": 1, "Lille": 1,
    "Ajax": 1, "PSV": 1, "Feyenoord": 1,
    "Club Brugge": 1, "Anderlecht": 1, "Gent": 1,
    "Celtic": 1, "Rangers": 1,
    # Medium haul
    "Real Madrid": 2, "Barcelona": 2, "Atletico": 2, "Sevilla": 2, "Valencia": 2,
    "Villarreal": 2, "Real Sociedad": 2, "Athletic Bilbao": 2,
    "Bayern": 2, "Dortmund": 2, "Leipzig": 2, "Leverkusen": 2, "Frankfurt": 2,
    "Wolfsburg": 2, "Gladbach": 2, "Stuttgart": 2,
    "Juventus": 2, "AC Milan": 2, "Inter": 2, "Napoli": 2, "Roma": 2, "Lazio": 2,
    "Atalanta": 2, "Fiorentina": 2,
    "Benfica": 2, "Porto": 2, "Sporting": 2, "Braga": 2,
    "Salzburg": 2, "Sturm Graz": 2,
    "Copenhagen": 2, "Brondby": 2, "Midtjylland": 2,
    "Malmo": 2, "Gothenburg": 2,
    "Young Boys": 2, "Basel": 2,
    # Long haul
    "Galatasaray": 3, "Fenerbahce": 3, "Besiktas": 3, "Trabzonspor": 3,
    "Olympiacos": 3, "AEK Athens": 3, "PAOK": 3,
    "Shakhtar": 3, "Dynamo Kyiv": 3,
    "Zagreb": 3, "Dinamo Zagreb": 3,
    "Red Star": 3, "Partizan": 3,
    "Slavia Prague": 3, "Sparta Prague": 3, "Viktoria Plzen": 3,
    "Ferencvaros": 3, "Legia Warsaw": 3,
    # Very long haul
    "Zenit": 4, "CSKA Moscow": 4, "Spartak Moscow": 4, "Lokomotiv": 4,
    "Qarabag": 4, "Astana": 4, "Kairat": 4,
}


def get_travel_category(opponent_name: str) -> int:
    """Get travel fatigue category for an opponent.

    Returns:
        0 = Domestic/Home
        1 = Short European (1-2h flight)
        2 = Medium European (2-3h flight)
        3 = Long European (3-4h flight)
        4 = Very long European (4h+ flight)
    """
    if not opponent_name:
        return 0
    for key, cat in TRAVEL_CATEGORIES.items():
        if key.lower() in opponent_name.lower():
            return cat
    # Default to medium for unknown European teams
    return 2


def calculate_travel_fatigue_modifier(
    prev_fixture: TeamFixture,
    is_european: bool = False,
) -> float:
    """Calculate travel fatigue modifier in equivalent days.

    Args:
        prev_fixture: The previous fixture
        is_european: Whether it was a European competition

    Returns:
        Additional fatigue in equivalent rest days (negative = more tired)
    """
    if not prev_fixture:
        return 0.0

    # Home games have minimal travel
    if prev_fixture.is_home:
        return 0.0

    # Domestic away - small modifier
    if not is_european:
        return -0.3

    # European away - varies by distance
    travel_cat = get_travel_category(prev_fixture.opponent_name)

    modifiers = {
        1: -0.5,   # Short haul
        2: -1.0,   # Medium haul
        3: -1.5,   # Long haul (Turkey, Greece, Eastern Europe)
        4: -2.5,   # Very long haul (Azerbaijan, Russia)
    }

    return modifiers.get(travel_cat, -1.0)


class CongestionModel:
    """Calculates team-specific congestion coefficients."""

    def __init__(self):
        self.coefficients: dict[int, float] = {}
        self.team_names: dict[int, str] = {}

    def calculate_coefficients(self, min_samples: int = 10) -> dict[int, float]:
        """Calculate congestion coefficients for all teams.

        The coefficient represents how well a team handles fixture congestion:
        - Positive = performs better with short rest (deep squad)
        - Negative = performs worse with short rest (thin squad)
        - Near zero = no significant impact

        Args:
            min_samples: Minimum short-rest games required for coefficient

        Returns:
            Dict mapping team_id to congestion coefficient
        """
        with SyncSessionLocal() as session:
            self.team_names = {
                t.id: t.short_name
                for t in session.execute(select(Team)).scalars().all()
            }

            matches = session.execute(
                select(Match)
                .where(Match.status == MatchStatus.FINISHED)
                .order_by(Match.kickoff_time)
            ).scalars().all()

            # Track performance by rest category
            team_performance = defaultdict(lambda: {"short": [], "normal": []})

            for match in matches:
                for is_home, team_id in [(True, match.home_team_id), (False, match.away_team_id)]:
                    # Find previous fixture
                    prev_fixture = session.execute(
                        select(TeamFixture)
                        .where(TeamFixture.team_id == team_id)
                        .where(TeamFixture.kickoff_time < match.kickoff_time)
                        .order_by(TeamFixture.kickoff_time.desc())
                        .limit(1)
                    ).scalar_one_or_none()

                    if not prev_fixture:
                        continue

                    # Calculate effective rest (base + travel modifier)
                    base_rest = (match.kickoff_time - prev_fixture.kickoff_time).days

                    is_euro = prev_fixture.competition in (
                        "CL", "EL", "ECL",
                        "Champions League", "Europa League", "Conference League"
                    )
                    travel_mod = calculate_travel_fatigue_modifier(prev_fixture, is_euro)
                    effective_rest = base_rest + travel_mod

                    # Calculate points
                    if is_home:
                        points = 3 if match.home_score > match.away_score else (
                            1 if match.home_score == match.away_score else 0
                        )
                    else:
                        points = 3 if match.away_score > match.home_score else (
                            1 if match.home_score == match.away_score else 0
                        )

                    # Categorize by effective rest
                    if effective_rest <= 4:
                        team_performance[team_id]["short"].append(points)
                    else:
                        team_performance[team_id]["normal"].append(points)

            # Calculate coefficients
            self.coefficients = {}

            for team_id, perf in team_performance.items():
                if len(perf["short"]) >= min_samples and len(perf["normal"]) >= min_samples:
                    short_ppg = sum(perf["short"]) / len(perf["short"])
                    normal_ppg = sum(perf["normal"]) / len(perf["normal"])
                    self.coefficients[team_id] = short_ppg - normal_ppg

            logger.info(f"Calculated congestion coefficients for {len(self.coefficients)} teams")
            return self.coefficients

    def get_coefficient(self, team_id: int) -> float:
        """Get congestion coefficient for a team.

        Returns 0.0 (neutral) if team not found.
        """
        return self.coefficients.get(team_id, 0.0)

    def calculate_congestion_impact(
        self,
        team_id: int,
        rest_days: int,
        prev_fixture: Optional[TeamFixture] = None,
    ) -> float:
        """Calculate the congestion impact for a team's upcoming match.

        Args:
            team_id: Team ID
            rest_days: Days since last match
            prev_fixture: Previous fixture (for travel calculation)

        Returns:
            Congestion impact score (-1 to +1 range, normalized)
            Positive = congestion helps this team
            Negative = congestion hurts this team
        """
        # Calculate effective rest
        if prev_fixture:
            is_euro = prev_fixture.competition in (
                "CL", "EL", "ECL",
                "Champions League", "Europa League", "Conference League"
            )
            travel_mod = calculate_travel_fatigue_modifier(prev_fixture, is_euro)
        else:
            travel_mod = 0.0

        effective_rest = rest_days + travel_mod

        # Get team's congestion coefficient
        coeff = self.get_coefficient(team_id)

        # Calculate impact based on rest level
        # Short rest (< 4 days) amplifies the coefficient
        # Normal rest (4-6 days) has minimal impact
        # Long rest (> 6 days) might indicate rustiness

        if effective_rest <= 3:
            # Very short rest - coefficient fully applies
            rest_factor = 1.0
        elif effective_rest <= 5:
            # Short-medium rest - coefficient partially applies
            rest_factor = 0.5
        elif effective_rest <= 7:
            # Normal rest - minimal impact
            rest_factor = 0.0
        else:
            # Long rest - slight negative (rustiness)
            rest_factor = -0.2

        # Scale the impact to roughly -0.5 to +0.5 range
        impact = coeff * rest_factor / 2

        return impact

    def print_coefficients(self):
        """Print all coefficients sorted by value."""
        sorted_coeffs = sorted(
            self.coefficients.items(),
            key=lambda x: x[1],
            reverse=True
        )

        print("\n" + "=" * 60)
        print("TEAM CONGESTION COEFFICIENTS")
        print("=" * 60)
        print(f"{'Team':<25} {'Coefficient':<12} {'Assessment'}")
        print("-" * 60)

        for team_id, coeff in sorted_coeffs:
            name = self.team_names.get(team_id, f"Team {team_id}")
            if coeff > 0.2:
                assessment = "✓ Handles congestion well"
            elif coeff < -0.2:
                assessment = "✗ Struggles with congestion"
            else:
                assessment = "~ Neutral"
            print(f"{name:<25} {coeff:+.3f}       {assessment}")


# Singleton instance
_congestion_model: Optional[CongestionModel] = None


def get_congestion_model() -> CongestionModel:
    """Get or create the congestion model singleton."""
    global _congestion_model
    if _congestion_model is None:
        _congestion_model = CongestionModel()
        _congestion_model.calculate_coefficients()
    return _congestion_model


if __name__ == "__main__":
    import logging
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

    print("Calculating team congestion coefficients...")
    model = CongestionModel()
    model.calculate_coefficients()
    model.print_coefficients()
