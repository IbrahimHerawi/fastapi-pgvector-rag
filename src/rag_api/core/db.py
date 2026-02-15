"""Database engine and session dependency wiring."""

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from rag_api.core.config import get_settings

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

try:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
except ModuleNotFoundError:
    AsyncSession = Any  # type: ignore[assignment]
    async_sessionmaker = None  # type: ignore[assignment]
    create_async_engine = None  # type: ignore[assignment]


def get_database_url() -> str:
    settings = get_settings()
    if settings.APP_ENV == "test" and settings.TEST_DATABASE_URL:
        return settings.TEST_DATABASE_URL
    return settings.DATABASE_URL


def _require_sqlalchemy() -> None:
    if create_async_engine is None or async_sessionmaker is None:
        raise ModuleNotFoundError(
            "SQLAlchemy is required for database features. Install sqlalchemy and a Postgres driver."
        )


def _create_engine() -> "AsyncEngine | None":
    if create_async_engine is None:
        return None
    return create_async_engine(get_database_url(), pool_pre_ping=True)


engine = _create_engine()

if async_sessionmaker is not None and engine is not None:
    SessionLocal: "async_sessionmaker[AsyncSession] | None" = async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
else:
    SessionLocal = None


async def get_session() -> AsyncIterator["AsyncSession"]:
    _require_sqlalchemy()
    if SessionLocal is None:
        raise RuntimeError("Database session factory is not initialized.")

    async with SessionLocal() as session:
        yield session
