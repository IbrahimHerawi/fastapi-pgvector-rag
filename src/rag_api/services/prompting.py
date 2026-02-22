"""Prompt construction helpers for grounded RAG chat requests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

MAX_CHUNK_TEXT_CHARS = 1200


def _read_chunk_field(chunk: Any, *field_names: str) -> Any:
    if isinstance(chunk, Mapping):
        for field_name in field_names:
            if field_name in chunk:
                return chunk[field_name]

    for field_name in field_names:
        if hasattr(chunk, field_name):
            return getattr(chunk, field_name)

    msg = f"chunk is missing required fields: {', '.join(field_names)}"
    raise KeyError(msg)


def _resolve_title(document_id: Any, doc_titles: Mapping[Any, str]) -> str:
    if document_id in doc_titles:
        return doc_titles[document_id]

    document_id_str = str(document_id)
    if document_id_str in doc_titles:
        return doc_titles[document_id_str]

    return "Untitled"


def _truncate_chunk_text(text: str) -> str:
    if len(text) <= MAX_CHUNK_TEXT_CHARS:
        return text
    return f"{text[:MAX_CHUNK_TEXT_CHARS]}...[truncated]"


def build_messages(
    question: str,
    chunks: Sequence[Any],
    doc_titles: Mapping[Any, str],
) -> list[dict[str, str]]:
    source_entries: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        chunk_id = _read_chunk_field(chunk, "id", "chunk_id")
        document_id = _read_chunk_field(chunk, "document_id")
        chunk_text = str(_read_chunk_field(chunk, "text", "snippet"))
        title = _resolve_title(document_id, doc_titles)

        source_entries.append(
            f"[S{index}] chunk_id={chunk_id} | title={title}\n"
            f"{_truncate_chunk_text(chunk_text)}"
        )

    sources_block = "\n\n".join(source_entries) if source_entries else "(no sources retrieved)"

    return [
        {
            "role": "system",
            "content": (
                "Answer only from the provided sources. "
                "If the answer is not present, say the sources do not contain it."
            ),
        },
        {
            "role": "user",
            "content": (
                "Question:\n"
                f"{question}\n\n"
                "Sources:\n"
                f"{sources_block}\n\n"
                "Use source labels like [S1], [S2] when citing."
            ),
        },
    ]
