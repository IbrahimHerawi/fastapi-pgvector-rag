import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient

from rag_api.main import app


@pytest.mark.asyncio
async def test_request_succeeds_when_api_key_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_KEY", "")

    async with LifespanManager(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.get("/api/v1/health")

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_missing_header_returns_401_when_api_key_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_KEY", "unit-test-key")

    async with LifespanManager(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.get("/api/v1/health")

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_correct_header_returns_200_when_api_key_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_KEY", "unit-test-key")

    async with LifespanManager(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.get(
                "/api/v1/health",
                headers={"X-API-Key": "unit-test-key"},
            )

    assert response.status_code == 200
