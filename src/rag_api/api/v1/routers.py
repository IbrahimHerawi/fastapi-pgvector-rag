"""Versioned API routers."""

from hashlib import sha256
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_api.api.deps import require_api_key
from rag_api.core.config import get_settings
from rag_api.core.db import get_session
from rag_api.models.schema import Document, IngestionJob
from rag_api.schemas import DocumentCreateRequest, DocumentMetadataResponse

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


def _serialize_document_metadata(
    document: Document,
    ingestion_status: str | None,
    *,
    include_content: bool,
) -> DocumentMetadataResponse:
    return DocumentMetadataResponse(
        id=document.id,
        title=document.title,
        source=document.source,
        content_sha256=document.content_sha256,
        created_at=document.created_at,
        ingestion_status=ingestion_status,
        content=document.content if include_content else None,
    )


@router.get(
    "/documents",
    response_model=list[DocumentMetadataResponse],
    response_model_exclude_none=True,
)
async def list_documents(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    session: AsyncSession = Depends(get_session),
) -> list[DocumentMetadataResponse]:
    query = (
        select(Document, IngestionJob.status)
        .outerjoin(IngestionJob, IngestionJob.document_id == Document.id)
        .order_by(Document.created_at.desc(), Document.id.desc())
        .limit(limit)
        .offset(offset)
    )
    result = await session.execute(query)
    rows = result.all()
    return [
        _serialize_document_metadata(document, ingestion_status, include_content=False)
        for document, ingestion_status in rows
    ]


@router.get(
    "/documents/{document_id}",
    response_model=DocumentMetadataResponse,
    response_model_exclude_none=True,
)
async def get_document(
    document_id: UUID,
    include_content: bool = Query(default=False),
    session: AsyncSession = Depends(get_session),
) -> DocumentMetadataResponse:
    query = (
        select(Document, IngestionJob.status)
        .outerjoin(IngestionJob, IngestionJob.document_id == Document.id)
        .where(Document.id == document_id)
    )
    result = await session.execute(query)
    row = result.first()
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )

    document, ingestion_status = row
    return _serialize_document_metadata(
        document,
        ingestion_status,
        include_content=include_content,
    )
