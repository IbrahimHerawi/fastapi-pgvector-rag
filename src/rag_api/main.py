"""FastAPI application bootstrap."""

import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

import httpx
from fastapi import FastAPI

from rag_api.api.v1.routers import router
from rag_api.core.config import get_settings
from rag_api.core.db import (
    check_database_connection,
    is_database_unavailable_error,
    run_with_db_retries,
)
from rag_api.core.errors import register_exception_handlers
from rag_api.core.logging import configure_logging

logger = logging.getLogger(__name__)
settings = get_settings()
configure_logging(level=settings.LOG_LEVEL)


async def _run_startup_checks() -> None:
    if settings.APP_ENV == "test" or not settings.STARTUP_CHECKS_ENABLED:
        return

    try:
        await run_with_db_retries(
            check_database_connection,
            retries=settings.DB_RETRY_ATTEMPTS,
            initial_backoff_s=settings.DB_RETRY_BACKOFF_S,
            max_backoff_s=settings.DB_RETRY_MAX_BACKOFF_S,
        )
        logger.info("Startup database check succeeded.")
    except Exception as exc:
        if is_database_unavailable_error(exc):
            logger.warning("Startup database check failed: %s", exc)
        else:
            logger.exception("Startup database check failed unexpectedly.")

    ollama_base_url = settings.OLLAMA_BASE_URL.rstrip("/")
    try:
        async with httpx.AsyncClient(base_url=ollama_base_url, timeout=1.0) as client:
            response = await client.get("/api/tags")
            response.raise_for_status()
        logger.info("Startup Ollama reachability check succeeded.")
    except Exception as exc:
        logger.warning("Startup Ollama reachability check failed: %s", exc)


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    await _run_startup_checks()
    yield


app = FastAPI(title="RAG API", lifespan=_lifespan)
register_exception_handlers(app)
app.include_router(router, prefix=settings.API_V1_PREFIX)
