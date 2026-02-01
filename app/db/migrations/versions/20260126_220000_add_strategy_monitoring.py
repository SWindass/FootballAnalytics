"""Add strategy monitoring tables

Revision ID: d8a2f5c91b2e
Revises: e4f3a7b82c1d
Create Date: 2026-01-26 22:00:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'd8a2f5c91b2e'
down_revision: str | None = 'a5e3f8c91d2b'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Create betting_strategies table
    op.create_table('betting_strategies',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(length=100), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('outcome_type', sa.String(length=20), nullable=False),
        sa.Column('parameters', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('status', sa.String(length=20), nullable=False, server_default='active'),
        sa.Column('status_reason', sa.Text(), nullable=True),
        sa.Column('total_bets', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('total_wins', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('total_profit', sa.Numeric(precision=10, scale=2), nullable=False, server_default='0'),
        sa.Column('historical_roi', sa.Numeric(precision=6, scale=4), nullable=True),
        sa.Column('rolling_50_roi', sa.Numeric(precision=6, scale=4), nullable=True),
        sa.Column('consecutive_losing_streak', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('last_optimized_at', sa.DateTime(), nullable=True),
        sa.Column('last_backtest_at', sa.DateTime(), nullable=True),
        sa.Column('optimization_version', sa.Integer(), nullable=False, server_default='1'),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('name')
    )
    op.create_index('ix_betting_strategies_status', 'betting_strategies', ['status'], unique=False)

    # Create strategy_monitoring_snapshots table
    op.create_table('strategy_monitoring_snapshots',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('strategy_id', sa.Integer(), nullable=False),
        sa.Column('snapshot_date', sa.DateTime(), nullable=False),
        sa.Column('snapshot_type', sa.String(length=20), nullable=False),
        sa.Column('rolling_30_bets', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('rolling_30_wins', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('rolling_30_roi', sa.Numeric(precision=6, scale=4), nullable=True),
        sa.Column('rolling_30_profit', sa.Numeric(precision=10, scale=2), nullable=True),
        sa.Column('rolling_50_bets', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('rolling_50_wins', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('rolling_50_roi', sa.Numeric(precision=6, scale=4), nullable=True),
        sa.Column('rolling_50_profit', sa.Numeric(precision=10, scale=2), nullable=True),
        sa.Column('cumulative_bets', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('cumulative_roi', sa.Numeric(precision=6, scale=4), nullable=True),
        sa.Column('z_score', sa.Numeric(precision=6, scale=4), nullable=True),
        sa.Column('cusum_statistic', sa.Numeric(precision=8, scale=4), nullable=True),
        sa.Column('is_drift_detected', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('alert_triggered', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('alert_type', sa.String(length=50), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['strategy_id'], ['betting_strategies.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_strategy_snapshots_strategy_date', 'strategy_monitoring_snapshots',
                    ['strategy_id', 'snapshot_date'], unique=False)
    op.create_index('ix_strategy_snapshots_type', 'strategy_monitoring_snapshots',
                    ['snapshot_type'], unique=False)

    # Create strategy_optimization_runs table
    op.create_table('strategy_optimization_runs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('strategy_id', sa.Integer(), nullable=False),
        sa.Column('run_date', sa.DateTime(), nullable=False),
        sa.Column('run_type', sa.String(length=20), nullable=False),
        sa.Column('data_start', sa.DateTime(), nullable=False),
        sa.Column('data_end', sa.DateTime(), nullable=False),
        sa.Column('n_matches_used', sa.Integer(), nullable=False),
        sa.Column('parameters_before', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('parameters_after', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('n_trials', sa.Integer(), nullable=False),
        sa.Column('best_roi_found', sa.Numeric(precision=6, scale=4), nullable=True),
        sa.Column('backtest_roi_before', sa.Numeric(precision=6, scale=4), nullable=True),
        sa.Column('backtest_roi_after', sa.Numeric(precision=6, scale=4), nullable=True),
        sa.Column('was_applied', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('applied_at', sa.DateTime(), nullable=True),
        sa.Column('not_applied_reason', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['strategy_id'], ['betting_strategies.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_optimization_runs_strategy_date', 'strategy_optimization_runs',
                    ['strategy_id', 'run_date'], unique=False)

    # Add strategy_id to value_bets
    op.add_column('value_bets', sa.Column('strategy_id', sa.Integer(), nullable=True))
    op.create_foreign_key('fk_value_bets_strategy', 'value_bets', 'betting_strategies',
                          ['strategy_id'], ['id'])
    op.create_index('ix_value_bets_strategy_id', 'value_bets', ['strategy_id'], unique=False)


def downgrade() -> None:
    # Remove strategy_id from value_bets
    op.drop_index('ix_value_bets_strategy_id', table_name='value_bets')
    op.drop_constraint('fk_value_bets_strategy', 'value_bets', type_='foreignkey')
    op.drop_column('value_bets', 'strategy_id')

    # Drop tables in reverse order
    op.drop_index('ix_optimization_runs_strategy_date', table_name='strategy_optimization_runs')
    op.drop_table('strategy_optimization_runs')

    op.drop_index('ix_strategy_snapshots_type', table_name='strategy_monitoring_snapshots')
    op.drop_index('ix_strategy_snapshots_strategy_date', table_name='strategy_monitoring_snapshots')
    op.drop_table('strategy_monitoring_snapshots')

    op.drop_index('ix_betting_strategies_status', table_name='betting_strategies')
    op.drop_table('betting_strategies')
