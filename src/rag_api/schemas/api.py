"""Pydantic schemas for API request/response contracts."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated
from uuid import UUID

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

TitleStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=512)]
SourceStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1024)]
ContentStr = Annotated[str, StringConstraints(min_length=1, max_length=200_000)]
QuestionStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=8_000)]
AnswerStr = Annotated[str, StringConstraints(min_length=1)]
StatusStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=64)]
Sha256Str = Annotated[str, StringConstraints(pattern=r"^[A-Fa-f0-9]{64}$")]
NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class APIModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DocumentCreateRequest(APIModel):
    title: TitleStr
    source: SourceStr
    content: ContentStr


class DocumentCreateResponse(APIModel):
    id: UUID = Field(validation_alias=AliasChoices("id", "document_id"))
    ingestion_job_id: UUID | None = None
    status: StatusStr = "queued"
    created_at: datetime | None = None


class DocumentMetadataResponse(APIModel):
    id: UUID = Field(validation_alias=AliasChoices("id", "document_id"))
    title: TitleStr
    source: SourceStr
    content_sha256: Sha256Str
    created_at: datetime
    ingestion_status: StatusStr | None = Field(
        default=None,
        validation_alias=AliasChoices("ingestion_status", "status"),
    )
    content: ContentStr | None = None


class DocumentResponse(DocumentMetadataResponse):
    content: ContentStr


class DocumentJobStatusResponse(APIModel):
    status: StatusStr
    error: NonEmptyStr | None = None


class Source(APIModel):
    chunk_id: UUID
    document_id: UUID = Field(
        validation_alias=AliasChoices("document_id", "doc_id"),
        serialization_alias="doc_id",
    )
    title: TitleStr = Field(validation_alias=AliasChoices("title", "document_title"))
    source: SourceStr
    chunk_index: int = Field(ge=0)
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)
    snippet: NonEmptyStr = Field(validation_alias=AliasChoices("snippet", "text"))
    score: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_offsets(self) -> Source:
        if self.end_char < self.start_char:
            raise ValueError("end_char must be greater than or equal to start_char")
        return self


class AskRequest(APIModel):
    question: QuestionStr
    top_k: int | None = Field(default=None, ge=1)


class AskResponse(APIModel):
    answer: AnswerStr
    sources: list[Source] = Field(default_factory=list)
    query_log_id: UUID | None = None


class QueryLogResponse(APIModel):
    id: UUID = Field(validation_alias=AliasChoices("id", "query_id"))
    question: QuestionStr
    answer: AnswerStr
    retrieved_chunk_ids: list[UUID] = Field(default_factory=list)
    models: dict[str, str]
    latency_ms: int = Field(ge=0)
    created_at: datetime

    @field_validator("models")
    @classmethod
    def _validate_models(cls, value: dict[str, str]) -> dict[str, str]:
        if not value:
            raise ValueError("models must not be empty")
        for key, model_name in value.items():
            if not key.strip():
                raise ValueError("models keys must not be blank")
            if not model_name.strip():
                raise ValueError("models values must not be blank")
        return value


AskSource = Source
