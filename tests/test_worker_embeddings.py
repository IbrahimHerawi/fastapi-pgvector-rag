from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
from typing import Any

import pytest

pytest.importorskip("sqlalchemy")

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from rag_api.models.schema import IngestionJob
from rag_api.worker.run import process_job


async def _insert_job_for_content(
    db_session: Any,
    *,
    title: str,
    source: str,
    content: str,
    created_at: datetime,
) -> IngestionJob:
    from rag_api.models.schema import Document

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
async def test_process_job_embeds_all_chunks_and_marks_done(
    db_session: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CHUNK_MAX_CHARS", "48")
    monkeypatch.setenv("CHUNK_OVERLAP_CHARS", "8")

    next_value = 1

    async def _fake_embed_texts(_self: object, texts: list[str]) -> list[list[float]]:
        nonlocal next_value
        vectors: list[list[float]] = []
        for _text in texts:
            vectors.append([float(next_value)] * 768)
            next_value += 1
        return vectors

    monkeypatch.setattr("rag_api.worker.run.OllamaClient.embed_texts", _fake_embed_texts)

    content = (
        "Embedding test paragraph one with enough words to force chunking.\n"
        "Paragraph two continues to ensure we store multiple vectors.\n"
        "Paragraph three closes the sample text."
    )
    job = await _insert_job_for_content(
        db_session,
        title="Embedding success",
        source="https://example.com/worker/embed/success",
        content=content,
        created_at=datetime(2026, 1, 10, tzinfo=UTC),
    )
    await db_session.commit()

    await process_job(
        job.id,
        session_factory=_session_factory_from_db_session(db_session),
    )

    await db_session.refresh(job)
    assert job.status == "done"
    assert job.error is None

    result = await db_session.execute(
        text(
            """
            SELECT chunk_index, embedding::text AS embedding_text
            FROM chunks
            WHERE document_id = :document_id
            ORDER BY chunk_index ASC
            """
        ),
        {"document_id": str(job.document_id)},
    )
    rows = result.all()

    assert rows
    for expected_value, row in enumerate(rows, start=1):
        embedding_text = row.embedding_text
        assert embedding_text is not None
        values = embedding_text.strip("[]").split(",")
        assert len(values) == 768
        assert float(values[0]) == float(expected_value)


@pytest.mark.asyncio
async def test_process_job_embed_failure_marks_job_failed_with_error(
    db_session: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CHUNK_MAX_CHARS", "52")
    monkeypatch.setenv("CHUNK_OVERLAP_CHARS", "10")

    async def _failing_embed_texts(_self: object, _texts: list[str]) -> list[list[float]]:
        raise RuntimeError("embedding service unavailable")

    monkeypatch.setattr("rag_api.worker.run.OllamaClient.embed_texts", _failing_embed_texts)

    job = await _insert_job_for_content(
        db_session,
        title="Embedding failure",
        source="https://example.com/worker/embed/failure",
        content="This document should fail during embedding and mark the job failed.",
        created_at=datetime(2026, 1, 11, tzinfo=UTC),
    )
    await db_session.commit()

    with pytest.raises(RuntimeError, match="embedding service unavailable"):
        await process_job(
            job.id,
            session_factory=_session_factory_from_db_session(db_session),
        )

    await db_session.refresh(job)
    assert job.status == "failed"
    assert job.error is not None
    assert "embedding service unavailable" in job.error
