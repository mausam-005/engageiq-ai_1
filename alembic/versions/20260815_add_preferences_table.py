"""add preferences table

Revision ID: 20260815_add_preferences_table
Revises: e55261113c57
Create Date: 2026-08-15

"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "20260815_add_preferences_table"
down_revision: Union[str, Sequence[str], None] = "e55261113c57"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "preferences",
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("notification_enabled", sa.Boolean(), nullable=False, server_default=sa.text('true')),
        sa.Column("overlay_enabled", sa.Boolean(), nullable=False, server_default=sa.text('true')),
        sa.Column("audio_enabled", sa.Boolean(), nullable=False, server_default=sa.text('true')),
        sa.Column("quiet_hours_start", sa.String(length=5), nullable=True),
        sa.Column("quiet_hours_end", sa.String(length=5), nullable=True),
        sa.Column("sensitivity", sa.String(length=10), nullable=False, server_default='normal'),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("user_id"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("preferences")
