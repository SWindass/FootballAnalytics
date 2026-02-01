"""Add Dixon-Coles and Pi Rating prediction columns

Revision ID: 43ef894dec5c
Revises: f1a3b5c71d4e
Create Date: 2026-02-01 15:37:58.718628

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '43ef894dec5c'
down_revision: Union[str, None] = 'f1a3b5c71d4e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add Dixon-Coles model columns
    op.add_column('match_analyses', sa.Column('dixon_coles_home_prob', sa.Numeric(5, 4), nullable=True))
    op.add_column('match_analyses', sa.Column('dixon_coles_draw_prob', sa.Numeric(5, 4), nullable=True))
    op.add_column('match_analyses', sa.Column('dixon_coles_away_prob', sa.Numeric(5, 4), nullable=True))

    # Add Pi Rating model columns
    op.add_column('match_analyses', sa.Column('pi_rating_home_prob', sa.Numeric(5, 4), nullable=True))
    op.add_column('match_analyses', sa.Column('pi_rating_draw_prob', sa.Numeric(5, 4), nullable=True))
    op.add_column('match_analyses', sa.Column('pi_rating_away_prob', sa.Numeric(5, 4), nullable=True))


def downgrade() -> None:
    op.drop_column('match_analyses', 'pi_rating_away_prob')
    op.drop_column('match_analyses', 'pi_rating_draw_prob')
    op.drop_column('match_analyses', 'pi_rating_home_prob')
    op.drop_column('match_analyses', 'dixon_coles_away_prob')
    op.drop_column('match_analyses', 'dixon_coles_draw_prob')
    op.drop_column('match_analyses', 'dixon_coles_home_prob')
