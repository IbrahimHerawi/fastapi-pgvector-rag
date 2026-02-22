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
from rag_api.models.schema import Chunk, Document
from rag_api.main import app

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
) -> tuple[Document, list[Chunk]]:
    content = "".join(f"chunk-{index} " for index in range(chunk_count))
    document = Document(
        title="Operations guide",
        source="https://example.com/ops-guide",
        content=content,
        content_sha256=sha256(content.encode("utf-8")).hexdigest(),
    )
    db_session.add(document)
    await db_session.flush()

    chunks: list[Chunk] = []
    cursor = 0
    for index in range(chunk_count):
        chunk_text = f"Chunk {index} " + (f"detail-{index} " * 40)
        chunk = Chunk(
            document_id=document.id,
            chunk_index=index,
            start_char=cursor,
            end_char=cursor + len(chunk_text),
            text=chunk_text,
            embedding=None,
        )
        chunks.append(chunk)
        cursor += len(chunk_text)

    db_session.add_all(chunks)
    await db_session.flush()
    return document, chunks


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
async def test_ask_returns_answer_and_sources_with_offsets_and_snippets(
    db_session: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("API_KEY", "")
    document, chunks = await _seed_document_with_chunks(db_session, chunk_count=3)
    await _set_embeddings(db_session, chunks)

    async def _fake_embed_texts(_self: object, texts: list[str]) -> list[list[float]]:
        assert texts == ["How do I recover the service?"]
        return [_embedding_with_xy(1.0, 0.0)]

    async def _fake_chat(_self: object, _messages: list[dict[str, Any]]) -> str:
        return "Use the runbook recovery checklist."

    monkeypatch.setattr("rag_api.api.v1.routers.OllamaClient.embed_texts", _fake_embed_texts)
    monkeypatch.setattr("rag_api.services.generation.OllamaClient.chat", _fake_chat)

    async with _api_client_with_db_session(db_session) as api_client:
        response = await api_client.post(
            "/api/v1/ask",
            json={"question": "How do I recover the service?", "top_k": 2},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Use the runbook recovery checklist."
    assert len(body["sources"]) == 2

    expected_chunks = chunks[:2]
    for index, source in enumerate(body["sources"]):
        expected_chunk = expected_chunks[index]
        assert source["chunk_id"] == str(expected_chunk.id)
        assert source["doc_id"] == str(document.id)
        assert source["start_char"] == expected_chunk.start_char
        assert source["end_char"] == expected_chunk.end_char
        assert source["snippet"] == expected_chunk.text[:240]
        assert len(source["snippet"]) <= 240


@pytest.mark.asyncio
async def test_ask_clamps_top_k_to_ten(
    db_session: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("API_KEY", "")
    _, chunks = await _seed_document_with_chunks(db_session, chunk_count=12)
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
    assert len(body["sources"]) == 10
    assert [item["chunk_id"] for item in body["sources"]] == [str(chunk.id) for chunk in chunks[:10]]


@pytest.mark.asyncio
async def test_ask_empty_question_returns_400(
    db_session: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("API_KEY", "")

    async with _api_client_with_db_session(db_session) as api_client:
        response = await api_client.post(
            "/api/v1/ask",
            json={"question": "   ", "top_k": 3},
        )

    assert response.status_code == 400
    assert "Invalid ask payload" in response.json()["detail"]
