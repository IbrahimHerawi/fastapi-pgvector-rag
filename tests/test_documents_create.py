from __future__ import annotations

from hashlib import sha256
from typing import Any
from uuid import UUID

import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient

pytest.importorskip("sqlalchemy")

from sqlalchemy import select

from rag_api.core.db import get_session
from rag_api.models.schema import Document, IngestionJob
from rag_api.main import app


@pytest.mark.asyncio
async def test_create_document_persists_document_and_pending_job(
    db_session: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _override_get_session() -> Any:
        yield db_session

    app.dependency_overrides[get_session] = _override_get_session
    try:
        monkeypatch.setenv("API_KEY", "")
        payload = {
            "title": "Service runbook",
            "source": "https://example.com/runbook",
            "content": "This is an operations runbook for production incidents.",
        }

        async with LifespanManager(app):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as api_client:
                response = await api_client.post("/api/v1/documents", json=payload)
    finally:
        app.dependency_overrides.clear()

    assert response.status_code in {200, 201}
    body = response.json()
    assert body["status"] == "pending"

    document_id = UUID(body["document_id"])
    job_id = UUID(body["job_id"])

    document = await db_session.get(Document, document_id)
    assert document is not None
    assert document.content_sha256 == sha256(payload["content"].encode("utf-8")).hexdigest()

    ingestion_job = await db_session.get(IngestionJob, job_id)
    assert ingestion_job is not None
    assert ingestion_job.document_id == document_id
    assert ingestion_job.status == "pending"

    result = await db_session.execute(
        select(IngestionJob).where(IngestionJob.document_id == document_id)
    )
    assert len(result.scalars().all()) == 1


@pytest.mark.asyncio
async def test_create_document_rejects_content_exceeding_max_size(
    db_session: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _override_get_session() -> Any:
        yield db_session

    app.dependency_overrides[get_session] = _override_get_session
    try:
        monkeypatch.setenv("API_KEY", "")
        monkeypatch.setenv("MAX_DOC_CHARS", "5")

        async with LifespanManager(app):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as api_client:
                response = await api_client.post(
                    "/api/v1/documents",
                    json={
                        "title": "Too large document",
                        "source": "https://example.com/source",
                        "content": "123456",
                    },
                )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 400
    body = response.json()
    assert body["code"] == "bad_request"
    assert "maximum size" in body["message"]
    assert body["request_id"]
