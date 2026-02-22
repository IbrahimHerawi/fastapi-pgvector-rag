from __future__ import annotations

from hashlib import sha256
from typing import Any

import pytest

pytest.importorskip("sqlalchemy")

from sqlalchemy import text

from rag_api.models.schema import Chunk, Document
from rag_api.services.retrieval import retrieve_chunks

EMBED_DIM = 768


def _embedding_with_xy(x: float, y: float) -> list[float]:
    vector = [0.0] * EMBED_DIM
    vector[0] = x
    vector[1] = y
    return vector


def _to_vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(format(value, ".15g") for value in vector) + "]"


async def _insert_document(db_session: Any, *, title: str) -> Document:
    content = f"{title} content"
    document = Document(
        title=title,
        source=f"https://example.com/{title.replace(' ', '-').lower()}",
        content=content,
        content_sha256=sha256(content.encode("utf-8")).hexdigest(),
    )
    db_session.add(document)
    await db_session.flush()
    return document


@pytest.mark.asyncio
async def test_retrieve_chunks_orders_by_cosine_distance_and_excludes_null_embeddings(
    db_session: Any,
) -> None:
    document = await _insert_document(db_session, title="Retrieval ordering")
    chunks = [
        Chunk(
            document_id=document.id,
            chunk_index=0,
            start_char=0,
            end_char=10,
            text="best match",
            embedding=None,
        ),
        Chunk(
            document_id=document.id,
            chunk_index=1,
            start_char=10,
            end_char=20,
            text="second match",
            embedding=None,
        ),
        Chunk(
            document_id=document.id,
            chunk_index=2,
            start_char=20,
            end_char=30,
            text="third match",
            embedding=None,
        ),
        Chunk(
            document_id=document.id,
            chunk_index=3,
            start_char=30,
            end_char=40,
            text="null embedding",
            embedding=None,
        ),
    ]
    db_session.add_all(chunks)
    await db_session.flush()

    await db_session.execute(
        text(
            "UPDATE chunks "
            "SET embedding = CAST(:embedding AS vector) "
            "WHERE id = :chunk_id"
        ),
        [
            {
                "chunk_id": chunks[0].id,
                "embedding": _to_vector_literal(_embedding_with_xy(1.0, 0.0)),
            },
            {
                "chunk_id": chunks[1].id,
                "embedding": _to_vector_literal(_embedding_with_xy(1.0, 0.25)),
            },
            {
                "chunk_id": chunks[2].id,
                "embedding": _to_vector_literal(_embedding_with_xy(0.0, 1.0)),
            },
        ],
    )

    retrieved = await retrieve_chunks(
        db_session,
        _embedding_with_xy(1.0, 0.0),
        top_k=10,
    )

    assert [chunk.id for chunk in retrieved] == [chunks[0].id, chunks[1].id, chunks[2].id]
    assert chunks[3].id not in {chunk.id for chunk in retrieved}


@pytest.mark.asyncio
async def test_retrieve_chunks_clamps_top_k_to_ten(
    db_session: Any,
) -> None:
    document = await _insert_document(db_session, title="Retrieval top-k clamp")
    chunks = [
        Chunk(
            document_id=document.id,
            chunk_index=index,
            start_char=index,
            end_char=index + 1,
            text=f"chunk {index}",
            embedding=None,
        )
        for index in range(12)
    ]
    db_session.add_all(chunks)
    await db_session.flush()

    await db_session.execute(
        text(
            "UPDATE chunks "
            "SET embedding = CAST(:embedding AS vector) "
            "WHERE id = :chunk_id"
        ),
        [
            {
                "chunk_id": chunk.id,
                "embedding": _to_vector_literal(
                    _embedding_with_xy(1.0, index / 10.0),
                ),
            }
            for index, chunk in enumerate(chunks)
        ],
    )

    retrieved = await retrieve_chunks(
        db_session,
        _embedding_with_xy(1.0, 0.0),
        top_k=50,
    )

    assert len(retrieved) == 10
    assert [chunk.chunk_index for chunk in retrieved] == list(range(10))
