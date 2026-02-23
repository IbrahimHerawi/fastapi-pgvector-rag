from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

from rag_api.services.prompting import MAX_CHUNK_TEXT_CHARS, build_messages


def _user_message_content(messages: list[dict[str, str]]) -> str:
    assert messages[-1]["role"] == "user"
    return messages[-1]["content"]


def test_build_messages_formats_sources_with_incrementing_labels() -> None:
    doc_one_id = uuid4()
    doc_two_id = uuid4()
    chunk_one = SimpleNamespace(id=uuid4(), document_id=doc_one_id, text="alpha details")
    chunk_two = {
        "id": uuid4(),
        "document_id": doc_two_id,
        "text": "beta details",
    }

    messages = build_messages(
        question="Where is the runbook?",
        chunks=[chunk_one, chunk_two],
        doc_titles={
            doc_one_id: "Runbook A",
            str(doc_two_id): "Runbook B",
        },
    )

    content = _user_message_content(messages)

    assert "[S1]" in content
    assert "[S2]" in content
    assert content.index("[S1]") < content.index("[S2]")
    assert f"chunk_id={chunk_one.id}" in content
    assert "title=Runbook A" in content
    assert "title=Runbook B" in content


def test_build_messages_truncates_chunk_text_to_safe_max() -> None:
    doc_id = uuid4()
    long_text = "x" * (MAX_CHUNK_TEXT_CHARS + 50)
    chunk = SimpleNamespace(id=uuid4(), document_id=doc_id, text=long_text)

    messages = build_messages(
        question="Summarize source.",
        chunks=[chunk],
        doc_titles={doc_id: "Large source"},
    )
    content = _user_message_content(messages)

    assert ("x" * MAX_CHUNK_TEXT_CHARS) in content
    assert ("x" * (MAX_CHUNK_TEXT_CHARS + 1)) not in content
    assert "...[truncated]" in content


def test_build_messages_includes_question() -> None:
    question = "How do we rotate credentials?"
    chunk = SimpleNamespace(id=uuid4(), document_id=uuid4(), text="Rotate every 90 days.")

    messages = build_messages(
        question=question,
        chunks=[chunk],
        doc_titles={},
    )
    content = _user_message_content(messages)

    assert question in content
