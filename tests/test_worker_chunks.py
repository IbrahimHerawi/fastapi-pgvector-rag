from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
from typing import Any

import pytest

pytest.importorskip("sqlalchemy")

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from rag_api.models.schema import Chunk, Document, IngestionJob
from rag_api.services.chunking import chunk
from rag_api.worker.run import process_job


async def _insert_job_for_content(
    db_session: Any,
    *,
    title: str,
    source: str,
    content: str,
    created_at: datetime,
) -> IngestionJob:
    document = Document(
        title=title,
        source=source,
        content=content,
        content_sha256=sha256(content.encode("utf-8")).hexdigest(),
        created_at=created_at,
    )
    document.ingestion_job = IngestionJob(
        status="processing",
        created_at=created_at,
        updated_at=created_at,
    )
    db_session.add(document)
    await db_session.flush()
    return document.ingestion_job


def _session_factory_from_db_session(db_session: Any) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(
        bind=db_session.bind,
        class_=AsyncSession,
        expire_on_commit=False,
    )


@pytest.mark.asyncio
async def test_process_job_persists_expected_chunk_count(
    db_session: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    max_chars = 45
    overlap_chars = 8
    monkeypatch.setenv("CHUNK_MAX_CHARS", str(max_chars))
    monkeypatch.setenv("CHUNK_OVERLAP_CHARS", str(overlap_chars))

    content = (
        "Alpha line one.\n"
        "Alpha line two with extra text.\n\n"
        "Beta section starts and keeps going to force multiple chunks.\n"
        "Gamma section ends here."
    )
    job = await _insert_job_for_content(
        db_session,
        title="Chunk count doc",
        source="https://example.com/chunks/count",
        content=content,
        created_at=datetime(2026, 1, 8, tzinfo=UTC),
    )
    await db_session.commit()

    expected_chunks = chunk(
        text=content,
        max_chars=max_chars,
        overlap_chars=overlap_chars,
    )
    assert len(expected_chunks) > 1

    await process_job(
        job.id,
        session_factory=_session_factory_from_db_session(db_session),
    )

    result = await db_session.execute(
        select(Chunk.id).where(Chunk.document_id == job.document_id)
    )
    assert len(result.all()) == len(expected_chunks)


@pytest.mark.asyncio
async def test_process_job_persists_exact_chunk_offsets_and_text(
    db_session: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    max_chars = 60
    overlap_chars = 12
    monkeypatch.setenv("CHUNK_MAX_CHARS", str(max_chars))
    monkeypatch.setenv("CHUNK_OVERLAP_CHARS", str(overlap_chars))

    content = (
        "First paragraph begins with setup text.\n"
        "It includes a second sentence for better boundary choices.\n\n"
        "Second paragraph follows and should produce overlaps.\n"
        "Final sentence closes the sample."
    )
    job = await _insert_job_for_content(
        db_session,
        title="Offsets doc",
        source="https://example.com/chunks/offsets",
        content=content,
        created_at=datetime(2026, 1, 9, tzinfo=UTC),
    )
    await db_session.commit()

    expected_chunks = chunk(
        text=content,
        max_chars=max_chars,
        overlap_chars=overlap_chars,
    )

    await process_job(
        job.id,
        session_factory=_session_factory_from_db_session(db_session),
    )

    result = await db_session.execute(
        select(Chunk)
        .where(Chunk.document_id == job.document_id)
        .order_by(Chunk.chunk_index.asc())
    )
    rows = result.scalars().all()

    assert len(rows) == len(expected_chunks)
    for row, expected in zip(rows, expected_chunks):
        assert row.chunk_index == expected["chunk_index"]
        assert row.start_char == expected["start_char"]
        assert row.end_char == expected["end_char"]
        assert row.text == expected["text"]
        assert row.text == content[row.start_char : row.end_char]
        assert row.embedding is None
