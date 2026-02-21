"""add retrieval indexes for chunks

Revision ID: 002
Revises: 001
Create Date: 2026-02-18 16:40:00
"""

from __future__ import annotations

from typing import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "002"
down_revision: str | None = "001"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_chunks_embedding_hnsw_cosine
        ON chunks
        USING hnsw (embedding vector_cosine_ops)
        """
    )
    op.create_index("ix_chunks_document_id", "chunks", ["document_id"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_chunks_document_id", table_name="chunks")
    op.execute("DROP INDEX IF EXISTS ix_chunks_embedding_hnsw_cosine")
