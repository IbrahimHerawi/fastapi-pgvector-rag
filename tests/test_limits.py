from __future__ import annotations

from contextlib import asynccontextmanager
from hashlib import sha256
from typing import Any, AsyncIterator

import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient

pytest.importorskip("sqlalchemy")

from sqlalchemy import text

from rag_api.core.db import get_session
from rag_api.main import app
from rag_api.models.schema import Chunk, Document

EMBED_DIM = 768


def _embedding_with_xy(x: float, y: float) -> list[float]:
    vector = [0.0] * EMBED_DIM
    vector[0] = x
    vector[1] = y
    return vector


def _to_vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(format(value, ".15g") for value in vector) + "]"


@asynccontextmanager
async def _api_client_with_db_session(db_session: Any) -> AsyncIterator[AsyncClient]:
    async def _override_get_session() -> Any:
        yield db_session

    app.dependency_overrides[get_session] = _override_get_session
    try:
        async with LifespanManager(app):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as api_client:
                yield api_client
    finally:
        app.dependency_overrides.clear()


async def _seed_document_with_chunks(
    db_session: Any,
    *,
    chunk_count: int,
) -> list[Chunk]:
    content = "".join(f"chunk-{index} " for index in range(chunk_count))
    document = Document(
        title="Limits runbook",
        source="https://example.com/limits",
        content=content,
        content_sha256=sha256(content.encode("utf-8")).hexdigest(),
    )
    db_session.add(document)
    await db_session.flush()

    chunks = [
        Chunk(
            document_id=document.id,
            chunk_index=index,
            start_char=index,
            end_char=index + 1,
            text=f"chunk {index}",
            embedding=None,
        )
        for index in range(chunk_count)
    ]
    db_session.add_all(chunks)
    await db_session.flush()
    return chunks


async def _set_embeddings(db_session: Any, chunks: list[Chunk]) -> None:
    await db_session.execute(
        text(
            "UPDATE chunks "
            "SET embedding = CAST(:embedding AS vector) "
            "WHERE id = :chunk_id"
        ),
        [
            {
                "chunk_id": chunk.id,
                "embedding": _to_vector_literal(_embedding_with_xy(1.0, index / 10.0)),
            }
            for index, chunk in enumerate(chunks)
        ],
    )


@pytest.mark.asyncio
async def test_oversized_doc_rejected(
    db_session: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("API_KEY", "")
    monkeypatch.setenv("MAX_DOC_CHARS", "5")

    async with _api_client_with_db_session(db_session) as api_client:
        response = await api_client.post(
            "/api/v1/documents",
            json={
                "title": "Too large document",
                "source": "https://example.com/source",
                "content": "123456",
            },
        )

    assert response.status_code == 400
    body = response.json()
    assert body["code"] == "bad_request"
    assert "maximum size" in body["message"]


@pytest.mark.asyncio
async def test_oversized_question_rejected(
    db_session: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("API_KEY", "")

    async with _api_client_with_db_session(db_session) as api_client:
        response = await api_client.post(
            "/api/v1/ask",
            json={"question": "x" * 5001, "top_k": 3},
        )

    assert response.status_code == 400
    body = response.json()
    assert body["code"] == "bad_request"
    assert "5000" in body["message"]


@pytest.mark.asyncio
async def test_top_k_above_ten_is_clamped(
    db_session: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("API_KEY", "")
    chunks = await _seed_document_with_chunks(db_session, chunk_count=12)
    await _set_embeddings(db_session, chunks)

    async def _fake_embed_texts(_self: object, _texts: list[str]) -> list[list[float]]:
        return [_embedding_with_xy(1.0, 0.0)]

    async def _fake_chat(_self: object, _messages: list[dict[str, Any]]) -> str:
        return "answer"

    monkeypatch.setattr("rag_api.api.v1.routers.OllamaClient.embed_texts", _fake_embed_texts)
    monkeypatch.setattr("rag_api.services.generation.OllamaClient.chat", _fake_chat)

    async with _api_client_with_db_session(db_session) as api_client:
        response = await api_client.post(
            "/api/v1/ask",
            json={"question": "Need summary", "top_k": 50},
        )

    assert response.status_code == 200
    body = response.json()
    assert len(body["sources"]) <= 10
