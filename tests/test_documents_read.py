from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from typing import Any, AsyncIterator
from uuid import UUID, uuid4

import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient

pytest.importorskip("sqlalchemy")

from rag_api.core.db import get_session
from rag_api.models.schema import Document, IngestionJob
from rag_api.main import app


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


async def _insert_document(
    db_session: Any,
    *,
    title: str,
    source: str,
    content: str,
    created_at: datetime,
    ingestion_status: str = "pending",
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
        created_at=created_at,
        updated_at=created_at,
    )

    db_session.add(document)
    await db_session.flush()
    return document


@pytest.mark.asyncio
async def test_documents_list_pagination_returns_expected_count_and_order(
    db_session: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("API_KEY", "")

    base_time = datetime(2026, 1, 1, tzinfo=UTC)
    oldest = await _insert_document(
        db_session,
        title="Oldest doc",
        source="https://example.com/oldest",
        content="oldest content",
        created_at=base_time,
        ingestion_status="completed",
    )
    middle = await _insert_document(
        db_session,
        title="Middle doc",
        source="https://example.com/middle",
        content="middle content",
        created_at=base_time + timedelta(minutes=1),
        ingestion_status="pending",
    )
    await _insert_document(
        db_session,
        title="Newest doc",
        source="https://example.com/newest",
        content="newest content",
        created_at=base_time + timedelta(minutes=2),
        ingestion_status="failed",
    )

    async with _api_client_with_db_session(db_session) as api_client:
        response = await api_client.get("/api/v1/documents", params={"limit": 2, "offset": 1})

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 2
    assert [UUID(item["id"]) for item in body] == [middle.id, oldest.id]
    assert [item["ingestion_status"] for item in body] == ["pending", "completed"]
    assert all("content" not in item for item in body)


@pytest.mark.asyncio
async def test_get_document_by_id_returns_metadata(
    db_session: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("API_KEY", "")

    created_at = datetime(2026, 1, 2, tzinfo=UTC)
    document = await _insert_document(
        db_session,
        title="Runbook",
        source="https://example.com/runbook",
        content="Incident response checklist.",
        created_at=created_at,
        ingestion_status="pending",
    )

    async with _api_client_with_db_session(db_session) as api_client:
        response = await api_client.get(f"/api/v1/documents/{document.id}")

    assert response.status_code == 200
    body = response.json()
    assert UUID(body["id"]) == document.id
    assert body["title"] == "Runbook"
    assert body["source"] == "https://example.com/runbook"
    assert body["content_sha256"] == sha256("Incident response checklist.".encode("utf-8")).hexdigest()
    assert body["ingestion_status"] == "pending"
    assert "created_at" in body
    assert "content" not in body


@pytest.mark.asyncio
async def test_get_document_by_id_returns_404_when_missing(
    db_session: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("API_KEY", "")

    async with _api_client_with_db_session(db_session) as api_client:
        response = await api_client.get(f"/api/v1/documents/{uuid4()}")

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_get_document_include_content_controls_content_field(
    db_session: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("API_KEY", "")

    content = "Private body text."
    document = await _insert_document(
        db_session,
        title="Design notes",
        source="https://example.com/design",
        content=content,
        created_at=datetime(2026, 1, 3, tzinfo=UTC),
        ingestion_status="completed",
    )

    async with _api_client_with_db_session(db_session) as api_client:
        without_content = await api_client.get(f"/api/v1/documents/{document.id}")
        with_content = await api_client.get(
            f"/api/v1/documents/{document.id}",
            params={"include_content": True},
        )

    assert without_content.status_code == 200
    assert "content" not in without_content.json()

    assert with_content.status_code == 200
    body = with_content.json()
    assert body["content"] == content
    assert body["ingestion_status"] == "completed"
