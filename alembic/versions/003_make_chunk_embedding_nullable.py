"""allow chunks.embedding to be null for staged ingestion

Revision ID: 003
Revises: 002
Create Date: 2026-02-22 15:00:00
"""

from __future__ import annotations

from typing import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "003"
down_revision: str | None = "002"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE chunks ALTER COLUMN embedding DROP NOT NULL")


def downgrade() -> None:
    op.execute(
        """
        UPDATE chunks
        SET embedding = array_fill(0::real, ARRAY[768])::vector
        WHERE embedding IS NULL
        """
    )
    op.execute("ALTER TABLE chunks ALTER COLUMN embedding SET NOT NULL")
