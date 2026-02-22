from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, AsyncIterator
from uuid import uuid4

import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient

pytest.importorskip("sqlalchemy")

from rag_api.core.db import get_session
from rag_api.main import app
from rag_api.services.ollama_client import OllamaClientError


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


@pytest.mark.asyncio
async def test_missing_document_returns_404_error_shape(
    db_session: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("API_KEY", "")
    request_id = "req-missing-document"

    async with _api_client_with_db_session(db_session) as api_client:
        response = await api_client.get(
            f"/api/v1/documents/{uuid4()}",
            headers={"X-Request-ID": request_id},
        )

    assert response.status_code == 404
    assert response.json() == {
        "code": "not_found",
        "message": "Document not found.",
        "request_id": request_id,
    }


@pytest.mark.asyncio
async def test_ask_ollama_failure_returns_503_error_shape(
    db_session: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("API_KEY", "")
    request_id = "req-ollama-failure"

    async def _failing_embed_texts(_self: object, _texts: list[str]) -> list[list[float]]:
        raise OllamaClientError("dial tcp timeout")

    monkeypatch.setattr("rag_api.api.v1.routers.OllamaClient.embed_texts", _failing_embed_texts)

    async with _api_client_with_db_session(db_session) as api_client:
        response = await api_client.post(
            "/api/v1/ask",
            json={"question": "What does the runbook say?"},
            headers={"X-Request-ID": request_id},
        )

    assert response.status_code == 503
    assert response.json() == {
        "code": "external_service_unavailable",
        "message": "Ollama service unavailable.",
        "request_id": request_id,
    }
