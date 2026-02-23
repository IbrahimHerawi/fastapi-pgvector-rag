from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
from typing import Any

import pytest
from sqlalchemy import select

pytest.importorskip("sqlalchemy")

from rag_api.models.schema import Document, IngestionJob
from rag_api.worker.run import claim_pending_job


async def _insert_pending_job(
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
        status="pending",
        created_at=created_at,
        updated_at=created_at,
    )

    db_session.add(document)
    await db_session.flush()
    return document.ingestion_job


@pytest.mark.asyncio
async def test_claim_pending_job_claims_distinct_rows_and_sets_processing(db_session: Any) -> None:
    first_job = await _insert_pending_job(
        db_session,
        title="Doc One",
        source="https://example.com/doc-one",
        content="First pending document.",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    second_job = await _insert_pending_job(
        db_session,
        title="Doc Two",
        source="https://example.com/doc-two",
        content="Second pending document.",
        created_at=datetime(2026, 1, 2, tzinfo=UTC),
    )
    await db_session.commit()

    first_claim = await claim_pending_job(db_session)
    second_claim = await claim_pending_job(db_session)

    assert first_claim is not None
    assert second_claim is not None
    assert first_claim != second_claim
    assert {first_claim, second_claim} == {first_job.id, second_job.id}

    status_result = await db_session.execute(
        select(IngestionJob.id, IngestionJob.status).where(
            IngestionJob.id.in_([first_job.id, second_job.id])
        )
    )
    statuses = {job_id: status for job_id, status in status_result.all()}

    assert statuses[first_claim] == "processing"
    assert statuses[second_claim] == "processing"

    pending_result = await db_session.execute(
        select(IngestionJob.id).where(
            IngestionJob.id.in_([first_job.id, second_job.id]),
            IngestionJob.status == "pending",
        )
    )
    assert pending_result.first() is None

    third_claim = await claim_pending_job(db_session)
    assert third_claim is None or third_claim not in {first_job.id, second_job.id}
