from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from rag_api.schemas import (
    AskRequest,
    AskResponse,
    DocumentCreateRequest,
    DocumentCreateResponse,
    DocumentResponse,
    QueryLogResponse,
)


def test_document_create_request_valid() -> None:
    schema = DocumentCreateRequest(
        title="Architecture notes",
        source="https://example.com/docs/architecture",
        content="Core system design and decisions.",
    )

    assert schema.title == "Architecture notes"
    assert schema.source == "https://example.com/docs/architecture"


def test_document_create_request_invalid_raises_validation_error() -> None:
    with pytest.raises(ValidationError):
        DocumentCreateRequest(
            title="",
            source="https://example.com/docs/architecture",
            content="Core system design and decisions.",
        )


def test_document_create_response_valid() -> None:
    document_id = uuid4()
    ingestion_job_id = uuid4()

    schema = DocumentCreateResponse(
        document_id=document_id,
        ingestion_job_id=ingestion_job_id,
        status="queued",
    )

    assert schema.id == document_id
    assert schema.ingestion_job_id == ingestion_job_id
    assert schema.status == "queued"


def test_document_create_response_invalid_raises_validation_error() -> None:
    with pytest.raises(ValidationError):
        DocumentCreateResponse(id="not-a-uuid", status="queued")


def test_document_response_valid() -> None:
    document_id = uuid4()
    created_at = datetime.now(UTC)

    schema = DocumentResponse(
        id=document_id,
        title="Runbook",
        source="https://example.com/runbook",
        content="Incident response checklist.",
        content_sha256="a" * 64,
        created_at=created_at,
        status="completed",
    )

    assert schema.id == document_id
    assert schema.ingestion_status == "completed"
    assert schema.created_at == created_at


def test_document_response_invalid_raises_validation_error() -> None:
    with pytest.raises(ValidationError):
        DocumentResponse(
            id=uuid4(),
            title="Runbook",
            source="https://example.com/runbook",
            content="Incident response checklist.",
            content_sha256="invalid-hash",
            created_at=datetime.now(UTC),
        )


def test_ask_request_valid() -> None:
    schema = AskRequest(question="How do we rotate credentials?", top_k=3)

    assert schema.question == "How do we rotate credentials?"
    assert schema.top_k == 3


def test_ask_request_invalid_raises_validation_error() -> None:
    with pytest.raises(ValidationError):
        AskRequest(question="", top_k=3)


def test_ask_response_valid_with_sources() -> None:
    source_chunk_id = uuid4()
    source_document_id = uuid4()
    query_log_id = uuid4()

    schema = AskResponse(
        answer="Rotate credentials via the secrets manager workflow.",
        sources=[
            {
                "chunk_id": source_chunk_id,
                "document_id": source_document_id,
                "document_title": "Security Runbook",
                "source": "https://example.com/security-runbook",
                "chunk_index": 0,
                "text": "Use the secrets manager rotation endpoint.",
                "score": 0.88,
            }
        ],
        query_log_id=query_log_id,
    )

    assert schema.answer.startswith("Rotate credentials")
    assert len(schema.sources) == 1
    assert schema.sources[0].chunk_id == source_chunk_id
    assert schema.sources[0].snippet == "Use the secrets manager rotation endpoint."
    assert schema.query_log_id == query_log_id


def test_ask_response_invalid_raises_validation_error() -> None:
    with pytest.raises(ValidationError):
        AskResponse(
            answer="answer",
            sources=[
                {
                    "chunk_id": uuid4(),
                    "document_id": uuid4(),
                    "title": "Security Runbook",
                    "source": "https://example.com/security-runbook",
                    "chunk_index": 0,
                    "snippet": "Use the secrets manager rotation endpoint.",
                    "score": 2.0,
                }
            ],
        )


def test_query_log_response_valid() -> None:
    first_chunk_id = uuid4()
    second_chunk_id = uuid4()
    created_at = datetime.now(UTC)

    schema = QueryLogResponse(
        id=uuid4(),
        question="How do we rotate credentials?",
        answer="Use the secrets manager rotation endpoint.",
        retrieved_chunk_ids=[first_chunk_id, str(second_chunk_id)],
        models={"chat": "llama3.1:8b", "embed": "nomic-embed-text"},
        latency_ms=248,
        created_at=created_at,
    )

    assert schema.retrieved_chunk_ids == [first_chunk_id, second_chunk_id]
    assert schema.models["chat"] == "llama3.1:8b"
    assert schema.created_at == created_at


def test_query_log_response_invalid_raises_validation_error() -> None:
    with pytest.raises(ValidationError):
        QueryLogResponse(
            id=uuid4(),
            question="How do we rotate credentials?",
            answer="Use the secrets manager rotation endpoint.",
            retrieved_chunk_ids=[uuid4()],
            models={"chat": "llama3.1:8b"},
            latency_ms=-1,
            created_at=datetime.now(UTC),
        )

