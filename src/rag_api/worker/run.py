"""Background worker loop and ingestion job claiming helpers."""

from __future__ import annotations

import asyncio
import signal
from types import FrameType
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import select

from rag_api.core.db import SessionLocal
from rag_api.models.schema import IngestionJob

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

PENDING_STATUS = "pending"
PROCESSING_STATUS = "processing"
DONE_STATUS = "done"
FAILED_STATUS = "failed"
DEFAULT_IDLE_BACKOFF_SECONDS = 1.0

_running = True


def _shutdown_handler(signum: int, _frame: FrameType | None) -> None:
    global _running
    _running = False
    print(f"Worker received signal {signum}, shutting down.")


def _require_session_factory() -> "async_sessionmaker[AsyncSession]":
    if SessionLocal is None:
        raise RuntimeError("Database session factory is not initialized.")
    return SessionLocal


async def claim_pending_job(session: "AsyncSession") -> UUID | None:
    """Claim one pending ingestion job and move it to processing."""

    query = (
        select(IngestionJob)
        .where(IngestionJob.status == PENDING_STATUS)
        .order_by(IngestionJob.created_at.asc(), IngestionJob.id.asc())
        .limit(1)
        .with_for_update(skip_locked=True)
    )
    result = await session.execute(query)
    job = result.scalar_one_or_none()
    if job is None:
        await session.rollback()
        return None

    job.status = PROCESSING_STATUS
    job.error = None
    await session.commit()
    return job.id


async def _set_job_status(
    session: "AsyncSession",
    job_id: UUID,
    *,
    status: str,
    error: str | None = None,
) -> None:
    job = await session.get(IngestionJob, job_id)
    if job is None:
        await session.rollback()
        return

    job.status = status
    job.error = error
    await session.commit()


async def process_job(job_id: UUID) -> None:
    """Placeholder for actual ingestion processing logic."""

    _ = job_id


async def run_forever(
    *,
    session_factory: "async_sessionmaker[AsyncSession] | None" = None,
    idle_backoff_seconds: float = DEFAULT_IDLE_BACKOFF_SECONDS,
) -> None:
    """Continuously claim, process, and finalize ingestion jobs."""

    active_session_factory = session_factory or _require_session_factory()

    while _running:
        async with active_session_factory() as session:
            job_id = await claim_pending_job(session)

        if job_id is None:
            await asyncio.sleep(idle_backoff_seconds)
            continue

        try:
            await process_job(job_id)
        except Exception as exc:
            async with active_session_factory() as session:
                await _set_job_status(session, job_id, status=FAILED_STATUS, error=str(exc))
            continue

        async with active_session_factory() as session:
            await _set_job_status(session, job_id, status=DONE_STATUS)


def main() -> None:
    signal.signal(signal.SIGINT, _shutdown_handler)
    signal.signal(signal.SIGTERM, _shutdown_handler)

    print("Worker started. Waiting for jobs...")
    asyncio.run(run_forever())
    print("Worker stopped.")
