"""Versioned API routers."""

from hashlib import sha256
from time import perf_counter
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_api.api.deps import require_api_key
from rag_api.core.config import get_settings
from rag_api.core.db import get_session
from rag_api.models.schema import Chunk, Document, IngestionJob, QueryLog
from rag_api.schemas import (
    AskRequest,
    AskResponse,
    DocumentCreateRequest,
    DocumentJobStatusResponse,
    DocumentMetadataResponse,
    Source,
)
from rag_api.services.generation import generate_answer
from rag_api.services.ollama_client import OllamaClient
from rag_api.services.retrieval import MAX_RETRIEVAL_TOP_K, retrieve_chunks

router = APIRouter(dependencies=[Depends(require_api_key)])
SOURCE_SNIPPET_CHARS = 240


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


@router.get(
    "/documents/{document_id}/status",
    response_model=DocumentJobStatusResponse,
    response_model_exclude_none=True,
)
async def get_document_status(
    document_id: UUID,
    session: AsyncSession = Depends(get_session),
) -> DocumentJobStatusResponse:
    ingestion_job_query = select(IngestionJob.status, IngestionJob.error).where(
        IngestionJob.document_id == document_id
    )
    ingestion_job_result = await session.execute(ingestion_job_query)
    ingestion_job = ingestion_job_result.first()

    if ingestion_job is None:
        document = await session.get(Document, document_id)
        if document is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found.",
            )
        return DocumentJobStatusResponse(status="pending")

    ingestion_status, error = ingestion_job
    return DocumentJobStatusResponse(
        status=ingestion_status,
        error=error if ingestion_status == "failed" else None,
    )


def _resolve_top_k(requested_top_k: int | None) -> int:
    configured_top_k = get_settings().RETRIEVE_TOP_K if requested_top_k is None else requested_top_k
    return min(max(configured_top_k, 1), MAX_RETRIEVAL_TOP_K)


def _build_source_snippet(text: str) -> str:
    snippet = text[:SOURCE_SNIPPET_CHARS].strip()
    if snippet:
        return snippet
    return text[:SOURCE_SNIPPET_CHARS]


async def _load_documents_for_chunks(
    session: AsyncSession,
    chunks: list[Chunk],
) -> dict[UUID, tuple[str, str]]:
    if not chunks:
        return {}

    document_ids = {chunk.document_id for chunk in chunks}
    document_rows = await session.execute(
        select(Document.id, Document.title, Document.source).where(Document.id.in_(document_ids))
    )
    return {doc_id: (title, source) for doc_id, title, source in document_rows}


@router.post(
    "/ask",
    response_model=AskResponse,
    response_model_exclude_none=True,
)
async def ask_question(
    payload: dict[str, Any],
    session: AsyncSession = Depends(get_session),
) -> AskResponse:
    try:
        request = AskRequest.model_validate(payload)
    except ValidationError as exc:
        first_error = exc.errors()[0] if exc.errors() else {}
        message = first_error.get("msg", "Invalid ask payload.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid ask payload: {message}",
        ) from exc

    question = request.question
    top_k = _resolve_top_k(request.top_k)
    request_started_at = perf_counter()
    settings = get_settings()

    async with OllamaClient() as ollama_client:
        query_vectors = await ollama_client.embed_texts([question])

    if not query_vectors:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Embedding service returned no vectors for question.",
        )

    retrieved_chunks = await retrieve_chunks(
        session=session,
        query_embedding=query_vectors[0],
        top_k=top_k,
    )
    doc_map = await _load_documents_for_chunks(session, retrieved_chunks)

    retrieved_sources: list[dict[str, Any]] = []
    for chunk in retrieved_chunks:
        title, source = doc_map.get(chunk.document_id, ("Untitled", "unknown"))
        retrieved_sources.append(
            {
                "chunk_id": chunk.id,
                "document_id": chunk.document_id,
                "title": title,
                "source": source,
                "chunk_index": chunk.chunk_index,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                "text": chunk.text,
            }
        )

    answer = await generate_answer(question, retrieved_sources)

    query_log = QueryLog(
        question=question,
        answer=answer,
        retrieved_chunk_ids=[str(chunk.id) for chunk in retrieved_chunks],
        models={
            "embed": settings.OLLAMA_EMBED_MODEL,
            "chat": settings.OLLAMA_CHAT_MODEL,
        },
        latency_ms=int((perf_counter() - request_started_at) * 1000),
    )
    session.add(query_log)
    await session.flush()
    await session.commit()

    return AskResponse(
        answer=answer,
        sources=[
            Source(
                chunk_id=source["chunk_id"],
                document_id=source["document_id"],
                title=source["title"],
                source=source["source"],
                chunk_index=source["chunk_index"],
                start_char=source["start_char"],
                end_char=source["end_char"],
                snippet=_build_source_snippet(str(source["text"])),
            )
            for source in retrieved_sources
        ],
        query_log_id=query_log.id,
    )
