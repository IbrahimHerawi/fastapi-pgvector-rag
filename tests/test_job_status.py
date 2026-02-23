from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime
from hashlib import sha256
from typing import Any, AsyncIterator
from uuid import uuid4

import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient

pytest.importorskip("sqlalchemy")

from rag_api.core.db import get_session
from rag_api.main import app
from rag_api.models.schema import Document, IngestionJob


@asynccontextmanager
async def _api_client_with_db_session(db_session: Any) -> AsyncIterator[AsyncClient]:
    async def _override_get_session() -> Any:
        yield db_session

    app.dependency_overrides[get_session] = _override_get_session
    try:
        async with LifespanManager(app):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as api_client:
                yield api_client
    finally:
        app.dependency_overrides.clear()


async def _insert_document_with_job(
    db_session: Any,
    *,
    title: str,
    source: str,
    content: str,
    ingestion_status: str,
    created_at: datetime,
    error: str | None = None,
) -> Document:
    document = Document(
        title=title,
        source=source,
        content=content,
        content_sha256=sha256(content.encode("utf-8")).hexdigest(),
        created_at=created_at,
    )
    document.ingestion_job = IngestionJob(
        status=ingestion_status,
        error=error,
        created_at=created_at,
        updated_at=created_at,
    )

    db_session.add(document)
    await db_session.flush()
    return document


@pytest.mark.asyncio
async def test_get_document_status_returns_pending_for_pending_job(
    db_session: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("API_KEY", "")

    document = await _insert_document_with_job(
        db_session,
        title="On-call guide",
        source="https://example.com/oncall",
        content="Escalation process and paging policy.",
        ingestion_status="pending",
        created_at=datetime(2026, 1, 4, tzinfo=UTC),
    )

    async with _api_client_with_db_session(db_session) as api_client:
        response = await api_client.get(f"/api/v1/documents/{document.id}/status")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "pending"
    assert "error" not in body


@pytest.mark.asyncio
async def test_get_document_status_returns_404_for_missing_document(
    db_session: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("API_KEY", "")

    async with _api_client_with_db_session(db_session) as api_client:
        response = await api_client.get(f"/api/v1/documents/{uuid4()}/status")

    assert response.status_code == 404
