"""Allow negative edge in value_bets table.

Revision ID: a5e3f8c91d2b
Revises: 77fd65dee607
Create Date: 2026-01-26 21:15:00

This migration removes the ck_positive_edge constraint to allow
home win value bets that have negative edge (market sees more value
than model, but form 12+ makes them profitable).

Proven strategies:
- Away wins: 5-12% positive edge, exclude home form 4-6 -> +32.9% ROI
- Home wins: form 12+ with NEGATIVE edge -> +27.7% ROI
"""
from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'a5e3f8c91d2b'
down_revision: str | None = '77fd65dee607'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Drop the positive edge constraint to allow negative edge home win bets
    op.drop_constraint('ck_positive_edge', 'value_bets', type_='check')


def downgrade() -> None:
    # Restore the positive edge constraint
    op.create_check_constraint(
        'ck_positive_edge',
        'value_bets',
        'edge > 0'
    )
