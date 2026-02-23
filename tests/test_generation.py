from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest

from rag_api.services.generation import generate_answer


pytestmark = pytest.mark.asyncio


async def test_generate_answer_returns_chat_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_answer = "Deterministic model response."

    class _FakeOllamaClient:
        async def __aenter__(self) -> _FakeOllamaClient:
            return self

        async def __aexit__(
            self,
            _exc_type: type[BaseException] | None,
            _exc: BaseException | None,
            _tb: Any,
        ) -> None:
            return None

        async def chat(self, _messages: list[dict[str, str]]) -> str:
            return expected_answer

    monkeypatch.setattr("rag_api.services.generation.OllamaClient", _FakeOllamaClient)

    sources = [
        {
            "chunk_id": uuid4(),
            "document_id": uuid4(),
            "title": "Runbook",
            "snippet": "Restart the service and verify health checks.",
        }
    ]

    answer = await generate_answer("How do I restart the service?", sources)

    assert answer == expected_answer


async def test_generate_answer_passes_sources_into_chat_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_messages: list[dict[str, str]] | None = None

    class _FakeOllamaClient:
        async def __aenter__(self) -> _FakeOllamaClient:
            return self

        async def __aexit__(
            self,
            _exc_type: type[BaseException] | None,
            _exc: BaseException | None,
            _tb: Any,
        ) -> None:
            return None

        async def chat(self, messages: list[dict[str, str]]) -> str:
            nonlocal captured_messages
            captured_messages = messages
            return "ok"

    monkeypatch.setattr("rag_api.services.generation.OllamaClient", _FakeOllamaClient)

    document_id = uuid4()
    sources = [
        SimpleNamespace(
            chunk_id=uuid4(),
            document_id=document_id,
            title="Operations guide",
            snippet="Step one: cordon the node.",
        ),
        {
            "chunk_id": uuid4(),
            "document_id": document_id,
            "title": "Operations guide",
            "snippet": "Step two: recycle the service.",
        },
    ]

    await generate_answer("What are the recovery steps?", sources)

    assert captured_messages is not None
    assert len(captured_messages) == 2
    assert captured_messages[0]["role"] == "system"
    assert captured_messages[1]["role"] == "user"

    user_content = captured_messages[1]["content"]
    assert "Sources:" in user_content
    assert "[S1]" in user_content
    assert "[S2]" in user_content
    assert "title=Operations guide" in user_content
    assert "Step one: cordon the node." in user_content
    assert "Step two: recycle the service." in user_content
