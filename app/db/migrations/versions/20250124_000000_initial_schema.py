"""Initial schema

Revision ID: 001
Revises:
Create Date: 2025-01-24

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Teams table
    op.create_table(
        'teams',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('external_id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(100), nullable=False),
        sa.Column('short_name', sa.String(50), nullable=False),
        sa.Column('tla', sa.String(3), nullable=False),
        sa.Column('crest_url', sa.String(255), nullable=True),
        sa.Column('venue', sa.String(100), nullable=True),
        sa.Column('founded', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('external_id')
    )

    # Matches table
    op.create_table(
        'matches',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('external_id', sa.Integer(), nullable=False),
        sa.Column('season', sa.String(10), nullable=False),
        sa.Column('matchweek', sa.Integer(), nullable=False),
        sa.Column('kickoff_time', sa.DateTime(), nullable=False),
        sa.Column('status', sa.String(20), nullable=True),
        sa.Column('home_team_id', sa.Integer(), nullable=False),
        sa.Column('away_team_id', sa.Integer(), nullable=False),
        sa.Column('home_score', sa.Integer(), nullable=True),
        sa.Column('away_score', sa.Integer(), nullable=True),
        sa.Column('home_ht_score', sa.Integer(), nullable=True),
        sa.Column('away_ht_score', sa.Integer(), nullable=True),
        sa.Column('home_xg', sa.Numeric(5, 2), nullable=True),
        sa.Column('away_xg', sa.Numeric(5, 2), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['home_team_id'], ['teams.id']),
        sa.ForeignKeyConstraint(['away_team_id'], ['teams.id']),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('external_id'),
        sa.CheckConstraint('home_team_id != away_team_id', name='ck_different_teams')
    )
    op.create_index('ix_matches_season_matchweek', 'matches', ['season', 'matchweek'])
    op.create_index('ix_matches_kickoff_time', 'matches', ['kickoff_time'])

    # ELO ratings table
    op.create_table(
        'elo_ratings',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('team_id', sa.Integer(), nullable=False),
        sa.Column('season', sa.String(10), nullable=False),
        sa.Column('matchweek', sa.Integer(), nullable=False),
        sa.Column('rating', sa.Numeric(7, 2), nullable=False),
        sa.Column('rating_change', sa.Numeric(6, 2), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['team_id'], ['teams.id']),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('team_id', 'season', 'matchweek', name='uq_team_season_matchweek')
    )
    op.create_index('ix_elo_ratings_team_season', 'elo_ratings', ['team_id', 'season'])

    # Team stats table
    op.create_table(
        'team_stats',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('team_id', sa.Integer(), nullable=False),
        sa.Column('season', sa.String(10), nullable=False),
        sa.Column('matchweek', sa.Integer(), nullable=False),
        sa.Column('form', sa.String(5), nullable=True),
        sa.Column('form_points', sa.Integer(), nullable=True),
        sa.Column('goals_scored', sa.Integer(), nullable=True),
        sa.Column('goals_conceded', sa.Integer(), nullable=True),
        sa.Column('avg_goals_scored', sa.Numeric(4, 2), nullable=True),
        sa.Column('avg_goals_conceded', sa.Numeric(4, 2), nullable=True),
        sa.Column('xg_for', sa.Numeric(6, 2), nullable=True),
        sa.Column('xg_against', sa.Numeric(6, 2), nullable=True),
        sa.Column('avg_xg_for', sa.Numeric(4, 2), nullable=True),
        sa.Column('avg_xg_against', sa.Numeric(4, 2), nullable=True),
        sa.Column('home_wins', sa.Integer(), nullable=True),
        sa.Column('home_draws', sa.Integer(), nullable=True),
        sa.Column('home_losses', sa.Integer(), nullable=True),
        sa.Column('away_wins', sa.Integer(), nullable=True),
        sa.Column('away_draws', sa.Integer(), nullable=True),
        sa.Column('away_losses', sa.Integer(), nullable=True),
        sa.Column('clean_sheets', sa.Integer(), nullable=True),
        sa.Column('failed_to_score', sa.Integer(), nullable=True),
        sa.Column('injuries', postgresql.JSONB(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['team_id'], ['teams.id']),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('team_id', 'season', 'matchweek', name='uq_team_stats_season_matchweek')
    )
    op.create_index('ix_team_stats_team_season', 'team_stats', ['team_id', 'season'])

    # Match analyses table
    op.create_table(
        'match_analyses',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('match_id', sa.Integer(), nullable=False),
        sa.Column('elo_home_prob', sa.Numeric(5, 4), nullable=True),
        sa.Column('elo_draw_prob', sa.Numeric(5, 4), nullable=True),
        sa.Column('elo_away_prob', sa.Numeric(5, 4), nullable=True),
        sa.Column('poisson_home_prob', sa.Numeric(5, 4), nullable=True),
        sa.Column('poisson_draw_prob', sa.Numeric(5, 4), nullable=True),
        sa.Column('poisson_away_prob', sa.Numeric(5, 4), nullable=True),
        sa.Column('poisson_over_2_5_prob', sa.Numeric(5, 4), nullable=True),
        sa.Column('poisson_btts_prob', sa.Numeric(5, 4), nullable=True),
        sa.Column('xgboost_home_prob', sa.Numeric(5, 4), nullable=True),
        sa.Column('xgboost_draw_prob', sa.Numeric(5, 4), nullable=True),
        sa.Column('xgboost_away_prob', sa.Numeric(5, 4), nullable=True),
        sa.Column('consensus_home_prob', sa.Numeric(5, 4), nullable=True),
        sa.Column('consensus_draw_prob', sa.Numeric(5, 4), nullable=True),
        sa.Column('consensus_away_prob', sa.Numeric(5, 4), nullable=True),
        sa.Column('predicted_home_goals', sa.Numeric(4, 2), nullable=True),
        sa.Column('predicted_away_goals', sa.Numeric(4, 2), nullable=True),
        sa.Column('narrative', sa.Text(), nullable=True),
        sa.Column('narrative_generated_at', sa.DateTime(), nullable=True),
        sa.Column('features', postgresql.JSONB(), nullable=True),
        sa.Column('confidence', sa.Numeric(4, 3), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['match_id'], ['matches.id']),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('match_id')
    )

    # Odds history table
    op.create_table(
        'odds_history',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('match_id', sa.Integer(), nullable=False),
        sa.Column('bookmaker', sa.String(50), nullable=False),
        sa.Column('market', sa.String(50), nullable=False),
        sa.Column('recorded_at', sa.DateTime(), nullable=True),
        sa.Column('home_odds', sa.Numeric(6, 2), nullable=True),
        sa.Column('draw_odds', sa.Numeric(6, 2), nullable=True),
        sa.Column('away_odds', sa.Numeric(6, 2), nullable=True),
        sa.Column('over_2_5_odds', sa.Numeric(6, 2), nullable=True),
        sa.Column('under_2_5_odds', sa.Numeric(6, 2), nullable=True),
        sa.Column('btts_yes_odds', sa.Numeric(6, 2), nullable=True),
        sa.Column('btts_no_odds', sa.Numeric(6, 2), nullable=True),
        sa.ForeignKeyConstraint(['match_id'], ['matches.id']),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_odds_history_match_recorded', 'odds_history', ['match_id', 'recorded_at'])
    op.create_index('ix_odds_history_bookmaker', 'odds_history', ['bookmaker'])

    # Value bets table
    op.create_table(
        'value_bets',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('match_id', sa.Integer(), nullable=False),
        sa.Column('outcome', sa.String(20), nullable=False),
        sa.Column('bookmaker', sa.String(50), nullable=False),
        sa.Column('model_probability', sa.Numeric(5, 4), nullable=False),
        sa.Column('implied_probability', sa.Numeric(5, 4), nullable=False),
        sa.Column('edge', sa.Numeric(5, 4), nullable=False),
        sa.Column('odds', sa.Numeric(6, 2), nullable=False),
        sa.Column('kelly_stake', sa.Numeric(5, 4), nullable=False),
        sa.Column('recommended_stake', sa.Numeric(5, 4), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=True),
        sa.Column('result', sa.String(10), nullable=True),
        sa.Column('profit_loss', sa.Numeric(8, 2), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['match_id'], ['matches.id']),
        sa.PrimaryKeyConstraint('id'),
        sa.CheckConstraint('edge > 0', name='ck_positive_edge')
    )
    op.create_index('ix_value_bets_match_active', 'value_bets', ['match_id', 'is_active'])
    op.create_index('ix_value_bets_created', 'value_bets', ['created_at'])


def downgrade() -> None:
    op.drop_table('value_bets')
    op.drop_table('odds_history')
    op.drop_table('match_analyses')
    op.drop_table('team_stats')
    op.drop_table('elo_ratings')
    op.drop_table('matches')
    op.drop_table('teams')
