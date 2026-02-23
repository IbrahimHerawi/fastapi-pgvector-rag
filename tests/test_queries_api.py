from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Any, AsyncIterator
from uuid import UUID, uuid4

import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient

pytest.importorskip("sqlalchemy")

from rag_api.core.db import get_session
from rag_api.main import app
from rag_api.models.schema import QueryLog


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


async def _insert_query_log(
    db_session: Any,
    *,
    question: str,
    answer: str,
    created_at: datetime,
    latency_ms: int = 120,
) -> QueryLog:
    query_log = QueryLog(
        question=question,
        answer=answer,
        retrieved_chunk_ids=[str(uuid4()), str(uuid4())],
        models={"embed": "embed-model-test", "chat": "chat-model-test"},
        latency_ms=latency_ms,
        created_at=created_at,
    )
    db_session.add(query_log)
    await db_session.flush()
    return query_log


@pytest.mark.asyncio
async def test_queries_list_pagination_returns_expected_count_and_order(
    db_session: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("API_KEY", "")

    base_time = datetime(2026, 1, 4, tzinfo=UTC)
    oldest = await _insert_query_log(
        db_session,
        question="oldest question",
        answer="oldest answer",
        created_at=base_time,
    )
    middle = await _insert_query_log(
        db_session,
        question="middle question",
        answer="middle answer",
        created_at=base_time + timedelta(minutes=1),
    )
    await _insert_query_log(
        db_session,
        question="newest question",
        answer="newest answer",
        created_at=base_time + timedelta(minutes=2),
    )

    async with _api_client_with_db_session(db_session) as api_client:
        response = await api_client.get("/api/v1/queries", params={"limit": 2, "offset": 1})

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 2
    assert [UUID(item["id"]) for item in body] == [middle.id, oldest.id]
    assert [item["question"] for item in body] == ["middle question", "oldest question"]


@pytest.mark.asyncio
async def test_get_query_by_id_returns_query_log_payload(
    db_session: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("API_KEY", "")

    query_log = await _insert_query_log(
        db_session,
        question="How do we rotate credentials?",
        answer="Use the secrets manager rotation endpoint.",
        created_at=datetime(2026, 1, 5, tzinfo=UTC),
        latency_ms=248,
    )

    async with _api_client_with_db_session(db_session) as api_client:
        response = await api_client.get(f"/api/v1/queries/{query_log.id}")

    assert response.status_code == 200
    body = response.json()
    assert UUID(body["id"]) == query_log.id
    assert body["question"] == "How do we rotate credentials?"
    assert body["answer"] == "Use the secrets manager rotation endpoint."
    assert [UUID(chunk_id) for chunk_id in body["retrieved_chunk_ids"]] == [
        UUID(chunk_id) for chunk_id in query_log.retrieved_chunk_ids
    ]
    assert body["models"] == {"embed": "embed-model-test", "chat": "chat-model-test"}
    assert body["latency_ms"] == 248
    assert "created_at" in body


@pytest.mark.asyncio
async def test_get_query_by_id_returns_404_when_missing(
    db_session: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("API_KEY", "")

    async with _api_client_with_db_session(db_session) as api_client:
        response = await api_client.get(f"/api/v1/queries/{uuid4()}")

    assert response.status_code == 404
