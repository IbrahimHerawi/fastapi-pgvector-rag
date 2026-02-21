"""Versioned API routers."""

from hashlib import sha256

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from rag_api.api.deps import require_api_key
from rag_api.core.config import get_settings
from rag_api.core.db import get_session
from rag_api.models.schema import Document, IngestionJob
from rag_api.schemas import DocumentCreateRequest

router = APIRouter(dependencies=[Depends(require_api_key)])


@router.get("/health")
async def health() -> dict[str, str]:
    settings = get_settings()
    return {"status": "ok", "app_env": settings.APP_ENV}


@router.post("/documents", status_code=status.HTTP_201_CREATED)
async def create_document(
    payload: DocumentCreateRequest,
    session: AsyncSession = Depends(get_session),
) -> dict[str, str]:
    settings = get_settings()
    if len(payload.content) > settings.MAX_DOC_CHARS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"content exceeds maximum size ({settings.MAX_DOC_CHARS} characters).",
        )

    content_hash = sha256(payload.content.encode("utf-8")).hexdigest()

    document = Document(
        title=payload.title,
        source=payload.source,
        content=payload.content,
        content_sha256=content_hash,
    )
    session.add(document)
    await session.flush()

    ingestion_job = IngestionJob(document_id=document.id, status="pending")
    session.add(ingestion_job)
    await session.flush()
    await session.commit()

    return {
        "document_id": str(document.id),
        "job_id": str(ingestion_job.id),
        "status": ingestion_job.status,
    }
