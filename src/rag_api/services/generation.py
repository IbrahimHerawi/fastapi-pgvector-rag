"""Answer generation service backed by Ollama chat."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from rag_api.services.ollama_client import OllamaClient
from rag_api.services.prompting import build_messages


def _read_field(source: Any, *field_names: str) -> Any:
    if isinstance(source, Mapping):
        for field_name in field_names:
            if field_name in source:
                return source[field_name]

    for field_name in field_names:
        if hasattr(source, field_name):
            return getattr(source, field_name)

    msg = f"source is missing required fields: {', '.join(field_names)}"
    raise KeyError(msg)


def _read_optional_field(source: Any, *field_names: str) -> Any | None:
    if isinstance(source, Mapping):
        for field_name in field_names:
            if field_name in source:
                return source[field_name]

    for field_name in field_names:
        if hasattr(source, field_name):
            return getattr(source, field_name)

    return None


def _build_doc_titles(sources: Sequence[Any]) -> dict[Any, str]:
    doc_titles: dict[Any, str] = {}

    for source in sources:
        document_id = _read_field(source, "document_id")
        raw_title = _read_optional_field(source, "title", "document_title")
        if raw_title is None:
            continue

        title = str(raw_title).strip()
        if not title:
            continue

        if document_id in doc_titles or str(document_id) in doc_titles:
            continue

        doc_titles[document_id] = title

    return doc_titles


async def generate_answer(question: str, sources: Sequence[Any]) -> str:
    messages = build_messages(
        question=question,
        chunks=sources,
        doc_titles=_build_doc_titles(sources),
    )

    async with OllamaClient() as ollama_client:
        return await ollama_client.chat(messages)
