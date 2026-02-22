"""Chunk retrieval service using pgvector cosine distance."""

from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from rag_api.models.schema import Chunk

MAX_RETRIEVAL_TOP_K = 10


def _to_vector_literal(vector: Sequence[float]) -> str:
    return "[" + ",".join(format(value, ".15g") for value in vector) + "]"


async def retrieve_chunks(
    session: AsyncSession,
    query_embedding: Sequence[float],
    top_k: int,
) -> list[Chunk]:
    if top_k <= 0:
        return []

    limit = min(top_k, MAX_RETRIEVAL_TOP_K)
    statement = select(Chunk).from_statement(
        text(
            """
            SELECT c.*
            FROM chunks AS c
            WHERE c.embedding IS NOT NULL
            ORDER BY c.embedding <=> CAST(:query_embedding AS vector) ASC, c.id ASC
            LIMIT :top_k
            """
        )
    )
    result = await session.execute(
        statement,
        {
            "query_embedding": _to_vector_literal(query_embedding),
            "top_k": limit,
        },
    )
    return list(result.scalars())
