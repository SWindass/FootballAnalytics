"""Add FPL player data tables.

Revision ID: 20260131_220000
Revises: 20260126_220000
Create Date: 2026-01-31

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "f1a3b5c71d4e"
down_revision = "d8a2f5c91b2e"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add fpl_id to teams table
    op.add_column("teams", sa.Column("fpl_id", sa.Integer(), nullable=True))
    op.create_unique_constraint("uq_teams_fpl_id", "teams", ["fpl_id"])

    # Create players table
    op.create_table(
        "players",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("fpl_id", sa.Integer(), nullable=False, unique=True),
        sa.Column("team_id", sa.Integer(), sa.ForeignKey("teams.id"), nullable=True),
        # Basic info
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("web_name", sa.String(50), nullable=False),
        sa.Column("position", sa.String(3), nullable=False),
        # Value and ownership
        sa.Column("price", sa.Numeric(4, 1), nullable=False),
        sa.Column("selected_by_percent", sa.Numeric(5, 2), default=0),
        # Season totals
        sa.Column("total_points", sa.Integer(), default=0),
        sa.Column("points_per_game", sa.Numeric(4, 2), default=0),
        sa.Column("minutes", sa.Integer(), default=0),
        sa.Column("starts", sa.Integer(), default=0),
        # Goals and assists
        sa.Column("goals_scored", sa.Integer(), default=0),
        sa.Column("assists", sa.Integer(), default=0),
        sa.Column("clean_sheets", sa.Integer(), default=0),
        sa.Column("goals_conceded", sa.Integer(), default=0),
        # Form and ICT Index
        sa.Column("form", sa.Numeric(4, 1), default=0),
        sa.Column("influence", sa.Numeric(6, 1), default=0),
        sa.Column("creativity", sa.Numeric(6, 1), default=0),
        sa.Column("threat", sa.Numeric(6, 1), default=0),
        sa.Column("ict_index", sa.Numeric(6, 1), default=0),
        # Expected stats
        sa.Column("expected_goals", sa.Numeric(5, 2), default=0),
        sa.Column("expected_assists", sa.Numeric(5, 2), default=0),
        sa.Column("expected_goal_involvements", sa.Numeric(5, 2), default=0),
        sa.Column("expected_goals_conceded", sa.Numeric(5, 2), default=0),
        # Availability
        sa.Column("status", sa.String(1), default="a"),
        sa.Column("chance_of_playing", sa.Integer(), nullable=True),
        sa.Column("news", sa.Text(), nullable=True),
        # Timestamps
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now(), onupdate=sa.func.now()),
    )
    op.create_index("ix_players_team", "players", ["team_id"])
    op.create_index("ix_players_position", "players", ["position"])

    # Create player match performances table
    op.create_table(
        "player_match_performances",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("player_id", sa.Integer(), sa.ForeignKey("players.id"), nullable=False),
        sa.Column("match_id", sa.Integer(), sa.ForeignKey("matches.id"), nullable=True),
        # Match context
        sa.Column("season", sa.String(10), nullable=False),
        sa.Column("gameweek", sa.Integer(), nullable=False),
        sa.Column("opponent_team_id", sa.Integer(), sa.ForeignKey("teams.id"), nullable=True),
        sa.Column("was_home", sa.Boolean(), nullable=False),
        # Playing time
        sa.Column("minutes", sa.Integer(), default=0),
        # Points
        sa.Column("total_points", sa.Integer(), default=0),
        sa.Column("bonus", sa.Integer(), default=0),
        sa.Column("bps", sa.Integer(), default=0),
        # Stats
        sa.Column("goals_scored", sa.Integer(), default=0),
        sa.Column("assists", sa.Integer(), default=0),
        sa.Column("clean_sheets", sa.Integer(), default=0),
        sa.Column("goals_conceded", sa.Integer(), default=0),
        # ICT for this match
        sa.Column("influence", sa.Numeric(5, 1), default=0),
        sa.Column("creativity", sa.Numeric(5, 1), default=0),
        sa.Column("threat", sa.Numeric(5, 1), default=0),
        sa.Column("ict_index", sa.Numeric(5, 1), default=0),
        # Expected stats
        sa.Column("expected_goals", sa.Numeric(4, 2), default=0),
        sa.Column("expected_assists", sa.Numeric(4, 2), default=0),
        sa.Column("expected_goal_involvements", sa.Numeric(4, 2), default=0),
        # Value
        sa.Column("value", sa.Numeric(4, 1), default=0),
        sa.Column("selected", sa.Integer(), default=0),
        # Timestamp
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
    )
    op.create_index("ix_player_perf_player_season", "player_match_performances", ["player_id", "season"])
    op.create_index("ix_player_perf_gameweek", "player_match_performances", ["season", "gameweek"])
    op.create_unique_constraint("uq_player_gameweek", "player_match_performances", ["player_id", "season", "gameweek"])


def downgrade() -> None:
    op.drop_table("player_match_performances")
    op.drop_table("players")
    op.drop_constraint("uq_teams_fpl_id", "teams", type_="unique")
    op.drop_column("teams", "fpl_id")
