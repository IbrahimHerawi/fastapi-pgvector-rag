import importlib
import os

import pytest

from rag_api.core.config import get_settings


def _require_db_prereqs() -> None:
    pytest.importorskip("sqlalchemy")


@pytest.fixture
def test_database_url() -> str:
    database_url = os.getenv("TEST_DATABASE_URL") or get_settings().TEST_DATABASE_URL
    if not database_url:
        pytest.skip("TEST_DATABASE_URL is not set.")
    return database_url


@pytest.mark.asyncio
async def test_engine_uses_test_database_url(
    monkeypatch: pytest.MonkeyPatch, test_database_url: str
) -> None:
    _require_db_prereqs()

    monkeypatch.setenv("TEST_DATABASE_URL", test_database_url)
    import rag_api.core.db as db

    db = importlib.reload(db)

    try:
        assert db.engine is not None
        assert db.engine.url.render_as_string(hide_password=False) == test_database_url
    finally:
        if db.engine is not None:
            await db.engine.dispose()


@pytest.mark.asyncio
async def test_get_session_executes_select_one_and_closes(
    monkeypatch: pytest.MonkeyPatch, test_database_url: str
) -> None:
    _require_db_prereqs()

    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import AsyncSession

    monkeypatch.setenv("TEST_DATABASE_URL", test_database_url)
    import rag_api.core.db as db

    db = importlib.reload(db)

    close_called = False
    original_close = AsyncSession.close

    async def _tracked_close(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal close_called
        close_called = True
        await original_close(self, *args, **kwargs)

    monkeypatch.setattr(AsyncSession, "close", _tracked_close)

    try:
        async for session in db.get_session():
            result = await session.execute(text("SELECT 1"))
            assert result.scalar_one() == 1

        assert close_called
    finally:
        if db.engine is not None:
            await db.engine.dispose()
