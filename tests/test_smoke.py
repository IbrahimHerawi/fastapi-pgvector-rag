import importlib

import pytest


def test_can_import_rag_api_modules() -> None:
    modules = (
        "rag_api",
        "rag_api.main",
        "rag_api.core",
        "rag_api.core.config",
        "rag_api.core.db",
        "rag_api.core.logging",
        "rag_api.core.errors",
        "rag_api.models",
        "rag_api.schemas",
        "rag_api.services",
        "rag_api.api",
        "rag_api.api.routers",
    )
    for module_name in modules:
        module = importlib.import_module(module_name)
        assert module is not None


def test_can_import_app_module() -> None:
    for module_name in ("app.main", "main"):
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
        assert module is not None
        return
    pytest.skip("No app module found yet.")


@pytest.mark.asyncio
async def test_health_endpoint_smoke(api_client) -> None:
    response = await api_client.get("/health")
    if response.status_code == 404:
        pytest.skip("/health endpoint is not implemented yet.")
    assert response.status_code == 200
