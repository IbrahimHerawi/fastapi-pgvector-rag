"""Database engine and session dependency wiring."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import TYPE_CHECKING, Any, TypeVar

from rag_api.core.config import get_settings
from rag_api.core.errors import DatabaseUnavailable

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

try:
    from sqlalchemy import text
    from sqlalchemy.exc import DBAPIError, InterfaceError, OperationalError
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
except ModuleNotFoundError:
    AsyncSession = Any  # type: ignore[assignment]
    DBAPIError = None  # type: ignore[assignment]
    InterfaceError = None  # type: ignore[assignment]
    OperationalError = None  # type: ignore[assignment]
    async_sessionmaker = None  # type: ignore[assignment]
    create_async_engine = None  # type: ignore[assignment]
    text = None  # type: ignore[assignment]

T = TypeVar("T")


def get_database_url() -> str:
    settings = get_settings()
    if settings.APP_ENV == "test" and settings.TEST_DATABASE_URL:
        return settings.TEST_DATABASE_URL
    return settings.DATABASE_URL


def is_database_unavailable_error(exc: BaseException) -> bool:
    if OperationalError is not None and isinstance(exc, OperationalError):
        return True
    if InterfaceError is not None and isinstance(exc, InterfaceError):
        return True
    if DBAPIError is not None and isinstance(exc, DBAPIError):
        return bool(getattr(exc, "connection_invalidated", False))
    return False


async def run_with_db_retries(
    operation: Callable[[], Awaitable[T]],
    *,
    retries: int,
    initial_backoff_s: float,
    max_backoff_s: float,
) -> T:
    attempts = max(1, retries + 1)
    delay_seconds = max(0.0, initial_backoff_s)
    max_delay_seconds = max(delay_seconds, max_backoff_s)

    for attempt in range(1, attempts + 1):
        try:
            return await operation()
        except Exception as exc:  # pragma: no cover - exercised through callers
            if not is_database_unavailable_error(exc):
                raise
            if attempt == attempts:
                raise
            if delay_seconds > 0:
                await asyncio.sleep(delay_seconds)
                delay_seconds = min(max_delay_seconds, delay_seconds * 2)

    msg = "Database retry loop exhausted without returning or raising."
    raise RuntimeError(msg)


def _require_sqlalchemy() -> None:
    if create_async_engine is None or async_sessionmaker is None or text is None:
        raise ModuleNotFoundError(
            "SQLAlchemy is required for database features. Install sqlalchemy and a Postgres driver."
        )


def _create_engine() -> "AsyncEngine | None":
    if create_async_engine is None:
        return None

    settings = get_settings()
    return create_async_engine(
        get_database_url(),
        pool_pre_ping=True,
        connect_args={"connect_timeout": max(1, settings.DB_CONNECT_TIMEOUT_S)},
    )


engine = _create_engine()

if async_sessionmaker is not None and engine is not None:
    SessionLocal: "async_sessionmaker[AsyncSession] | None" = async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
else:
    SessionLocal = None


async def check_database_connection(
    *,
    session_factory: "async_sessionmaker[AsyncSession] | None" = None,
) -> None:
    _require_sqlalchemy()
    active_session_factory = session_factory or SessionLocal
    if active_session_factory is None:
        raise RuntimeError("Database session factory is not initialized.")

    async with active_session_factory() as session:
        await session.execute(text("SELECT 1"))


async def get_session() -> AsyncIterator["AsyncSession"]:
    _require_sqlalchemy()
    if SessionLocal is None:
        raise RuntimeError("Database session factory is not initialized.")

    settings = get_settings()

    async def _open_checked_session() -> "AsyncSession":
        session = SessionLocal()
        try:
            await session.execute(text("SELECT 1"))
            return session
        except Exception:
            await session.close()
            raise

    try:
        session = await run_with_db_retries(
            _open_checked_session,
            retries=settings.DB_RETRY_ATTEMPTS,
            initial_backoff_s=settings.DB_RETRY_BACKOFF_S,
            max_backoff_s=settings.DB_RETRY_MAX_BACKOFF_S,
        )
    except Exception as exc:
        if is_database_unavailable_error(exc):
            raise DatabaseUnavailable("Database unavailable.") from exc
        raise

    try:
        yield session
    finally:
        await session.close()
