import asyncio
import importlib
import os
from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest
import pytest_asyncio

from rag_api.core.config import get_settings


def _resolve_test_database_url() -> str | None:
    database_url = os.getenv("TEST_DATABASE_URL")
    if database_url:
        return database_url
    return get_settings().TEST_DATABASE_URL


@pytest.fixture(autouse=True)
def settings_override(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("REQUEST_TIMEOUT_S", "2")
    yield


@pytest.fixture(scope="session")
def event_loop() -> Iterator[asyncio.AbstractEventLoop]:
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def event_loop_policy() -> asyncio.AbstractEventLoopPolicy:
    if hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
        return asyncio.WindowsSelectorEventLoopPolicy()
    return asyncio.get_event_loop_policy()


@pytest_asyncio.fixture
async def db_engine() -> AsyncIterator[Any]:
    database_url = _resolve_test_database_url()
    if not database_url:
        pytest.skip("TEST_DATABASE_URL is not set.")

    try:
        from sqlalchemy.ext.asyncio import create_async_engine
    except ModuleNotFoundError:
        pytest.skip("SQLAlchemy is not installed.")

    engine = create_async_engine(database_url)
    try:
        yield engine
    finally:
        await engine.dispose()


@pytest_asyncio.fixture
async def db_session(db_engine: Any) -> AsyncIterator[Any]:
    try:
        from sqlalchemy import event
        from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
    except ModuleNotFoundError:
        pytest.skip("SQLAlchemy is not installed.")

    async with db_engine.connect() as connection:
        outer_txn = await connection.begin()
        await connection.begin_nested()

        session_factory = async_sessionmaker(
            bind=connection,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        session = session_factory()

        @event.listens_for(session.sync_session, "after_transaction_end")
        def _restart_savepoint(sync_session: Any, transaction: Any) -> None:
            parent = getattr(transaction, "parent", None)
            if transaction.nested and (parent is None or not parent.nested):
                sync_session.begin_nested()

        try:
            yield session
        finally:
            await session.close()
            if outer_txn.is_active:
                await outer_txn.rollback()


def _load_fastapi_app() -> Any:
    for module_name in ("app.main", "main"):
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue

        app = getattr(module, "app", None)
        if app is not None:
            return app

        create_app = getattr(module, "create_app", None)
        if callable(create_app):
            return create_app()

    return None


@pytest.fixture
def app() -> Iterator[Any]:
    try:
        from fastapi import FastAPI
    except ModuleNotFoundError:
        pytest.skip("FastAPI is not installed.")

    fastapi_app = _load_fastapi_app()
    if fastapi_app is None:
        pytest.skip("No FastAPI app found in app.main or main.")
    if not isinstance(fastapi_app, FastAPI):
        pytest.skip("Imported app is not a FastAPI instance.")

    try:
        yield fastapi_app
    finally:
        fastapi_app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def api_client(app: Any) -> AsyncIterator[Any]:
    from asgi_lifespan import LifespanManager
    from httpx import ASGITransport, AsyncClient

    async with LifespanManager(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            yield client
