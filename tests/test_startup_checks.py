from __future__ import annotations

import importlib

import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient

from rag_api.main import app as base_app


@pytest.mark.asyncio
async def test_health_still_ok_with_database_available() -> None:
    async with LifespanManager(base_app):
        transport = ASGITransport(app=base_app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as api_client:
            response = await api_client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "app_env": "test"}


@pytest.mark.asyncio
async def test_db_unreachable_returns_clean_503_on_db_endpoints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("API_KEY", "")
    monkeypatch.setenv("TEST_DATABASE_URL", "")
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql+psycopg://postgres:postgres@127.0.0.1:1/rag",
    )

    import rag_api.core.db as db
    import rag_api.main as main_module

    db = importlib.reload(db)
    main_module = importlib.reload(main_module)
    app = main_module.app

    try:
        async with LifespanManager(app):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as api_client:
                document_response = await api_client.post(
                    "/api/v1/documents",
                    json={
                        "title": "Runbook",
                        "source": "https://example.com/runbook",
                        "content": "Recovery instructions.",
                    },
                )
                ask_response = await api_client.post(
                    "/api/v1/ask",
                    json={"question": "How do we recover service?"},
                )
    finally:
        app.dependency_overrides.clear()
        if db.engine is not None:
            await db.engine.dispose()

    for response in (document_response, ask_response):
        assert response.status_code == 503
        body = response.json()
        assert body["code"] == "database_unavailable"
        assert body["message"] == "Database unavailable."
        assert body["request_id"]
