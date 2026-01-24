"""Generate AI narratives for match analyses.

Standalone job to generate/regenerate narratives for matches.
Can be run for a specific match, matchweek, or all pending.
"""

import asyncio
from datetime import datetime
from typing import Optional

import structlog
from sqlalchemy import select, and_

from app.core.config import get_settings
from app.db.database import SyncSessionLocal
from app.db.models import Match, MatchAnalysis, MatchStatus, Team, TeamStats
from batch.ai.narrative_generator import NarrativeGenerator

logger = structlog.get_logger()
settings = get_settings()


def generate_narrative_for_match(
    match_id: int,
    force: bool = False,
) -> Optional[str]:
    """Generate narrative for a single match.

    Args:
        match_id: Match ID to generate narrative for
        force: Regenerate even if narrative exists

    Returns:
        Generated narrative or None
    """
    with SyncSessionLocal() as session:
        # Load match and analysis
        match = session.get(Match, match_id)
        if not match:
            logger.error(f"Match {match_id} not found")
            return None

        analysis = session.execute(
            select(MatchAnalysis).where(MatchAnalysis.match_id == match_id)
        ).scalar_one_or_none()

        if not analysis:
            logger.error(f"No analysis found for match {match_id}")
            return None

        if analysis.narrative and not force:
            logger.info(f"Narrative already exists for match {match_id}")
            return analysis.narrative

        # Load teams
        teams = {t.id: t for t in session.execute(select(Team)).scalars().all()}
        home_team = teams[match.home_team_id]
        away_team = teams[match.away_team_id]

        # Load team stats
        matchweek = match.matchweek - 1 if match.matchweek > 1 else 1
        stmt = (
            select(TeamStats)
            .where(TeamStats.season == match.season)
            .where(TeamStats.matchweek == matchweek)
            .where(TeamStats.team_id.in_([match.home_team_id, match.away_team_id]))
        )
        stats = {s.team_id: s for s in session.execute(stmt).scalars().all()}

        # Build match data
        match_data = {
            "home_team": home_team.short_name,
            "away_team": away_team.short_name,
            "kickoff_time": match.kickoff_time,
            "venue": home_team.venue or "TBC",
        }

        # Build stats dicts
        home_stats = _build_stats_dict(stats.get(match.home_team_id), is_home=True)
        away_stats = _build_stats_dict(stats.get(match.away_team_id), is_home=False)

        # Build predictions
        predictions = {
            "home_win": float(analysis.consensus_home_prob or 0.33),
            "draw": float(analysis.consensus_draw_prob or 0.34),
            "away_win": float(analysis.consensus_away_prob or 0.33),
            "predicted_score": f"{float(analysis.predicted_home_goals or 1.3):.1f}-{float(analysis.predicted_away_goals or 1.1):.1f}",
        }

        # Get H2H
        h2h = _get_head_to_head(session, match.home_team_id, match.away_team_id, teams)

        # Generate narrative
        generator = NarrativeGenerator()
        narrative = asyncio.run(
            generator.generate_match_preview(
                match_data=match_data,
                home_stats=home_stats,
                away_stats=away_stats,
                predictions=predictions,
                h2h_history=h2h,
            )
        )

        # Save to database
        analysis.narrative = narrative
        analysis.narrative_generated_at = datetime.utcnow()
        session.commit()

        logger.info(
            f"Generated narrative for {home_team.short_name} vs {away_team.short_name}",
            match_id=match_id,
        )

        return narrative


def generate_narratives_for_matchweek(
    matchweek: int,
    season: str = None,
    force: bool = False,
) -> dict:
    """Generate narratives for all matches in a matchweek.

    Args:
        matchweek: Matchweek number
        season: Season (defaults to current)
        force: Regenerate even if narratives exist

    Returns:
        Summary of generated narratives
    """
    season = season or settings.current_season

    with SyncSessionLocal() as session:
        # Get matches for matchweek
        stmt = (
            select(Match)
            .where(Match.season == season)
            .where(Match.matchweek == matchweek)
        )
        matches = list(session.execute(stmt).scalars().all())

    generated = 0
    skipped = 0
    failed = 0

    for match in matches:
        try:
            result = generate_narrative_for_match(match.id, force=force)
            if result:
                generated += 1
            else:
                skipped += 1
        except Exception as e:
            logger.error(f"Failed to generate narrative for match {match.id}", error=str(e))
            failed += 1

    return {
        "matchweek": matchweek,
        "total": len(matches),
        "generated": generated,
        "skipped": skipped,
        "failed": failed,
    }


def generate_pending_narratives(limit: int = 10) -> dict:
    """Generate narratives for analyses missing them.

    Args:
        limit: Maximum narratives to generate

    Returns:
        Summary of generated narratives
    """
    with SyncSessionLocal() as session:
        # Find analyses without narratives for upcoming matches
        stmt = (
            select(MatchAnalysis)
            .join(Match, MatchAnalysis.match_id == Match.id)
            .where(MatchAnalysis.narrative.is_(None))
            .where(Match.status == MatchStatus.SCHEDULED)
            .order_by(Match.kickoff_time)
            .limit(limit)
        )
        analyses = list(session.execute(stmt).scalars().all())

    generated = 0
    failed = 0

    for analysis in analyses:
        try:
            result = generate_narrative_for_match(analysis.match_id)
            if result:
                generated += 1
        except Exception as e:
            logger.error(f"Failed to generate narrative", match_id=analysis.match_id, error=str(e))
            failed += 1

    return {
        "pending": len(analyses),
        "generated": generated,
        "failed": failed,
    }


def _build_stats_dict(stats: Optional[TeamStats], is_home: bool) -> dict:
    """Build stats dictionary for narrative prompt."""
    if not stats:
        return {"form": "N/A", "position": "N/A"}

    result = {
        "form": stats.form or "N/A",
        "position": "N/A",  # Would need league table query
        "goals_scored": stats.goals_scored,
        "goals_conceded": stats.goals_conceded,
        "injuries": [],
    }

    if stats.avg_xg_for:
        result["avg_xg_for"] = float(stats.avg_xg_for)

    if is_home:
        result["home_wins"] = stats.home_wins
        result["home_draws"] = stats.home_draws
        result["home_losses"] = stats.home_losses
    else:
        result["away_wins"] = stats.away_wins
        result["away_draws"] = stats.away_draws
        result["away_losses"] = stats.away_losses

    return result


def _get_head_to_head(session, home_id: int, away_id: int, teams: dict) -> list[dict]:
    """Get head-to-head history."""
    stmt = (
        select(Match)
        .where(
            ((Match.home_team_id == home_id) & (Match.away_team_id == away_id)) |
            ((Match.home_team_id == away_id) & (Match.away_team_id == home_id))
        )
        .where(Match.status == MatchStatus.FINISHED)
        .order_by(Match.kickoff_time.desc())
        .limit(5)
    )
    matches = list(session.execute(stmt).scalars().all())

    return [
        {
            "date": m.kickoff_time.strftime("%d %b %Y"),
            "home_team": teams[m.home_team_id].short_name,
            "away_team": teams[m.away_team_id].short_name,
            "home_score": m.home_score,
            "away_score": m.away_score,
        }
        for m in matches
    ]


if __name__ == "__main__":
    import argparse
    import logging

    parser = argparse.ArgumentParser(description="Generate AI match narratives")
    parser.add_argument("--match", type=int, help="Generate for specific match ID")
    parser.add_argument("--matchweek", type=int, help="Generate for matchweek")
    parser.add_argument("--pending", action="store_true", help="Generate for pending analyses")
    parser.add_argument("--force", action="store_true", help="Regenerate existing narratives")
    parser.add_argument("--limit", type=int, default=10, help="Max narratives to generate")

    args = parser.parse_args()

    # Reduce noise
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    if args.match:
        narrative = generate_narrative_for_match(args.match, force=args.force)
        if narrative:
            print(f"\n{'='*60}")
            print(narrative)
            print(f"{'='*60}")
    elif args.matchweek:
        result = generate_narratives_for_matchweek(args.matchweek, force=args.force)
        print(f"\nMatchweek {result['matchweek']}: {result['generated']} generated, {result['skipped']} skipped, {result['failed']} failed")
    elif args.pending:
        result = generate_pending_narratives(limit=args.limit)
        print(f"\nPending: {result['generated']} generated, {result['failed']} failed")
    else:
        parser.print_help()
