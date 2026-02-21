"""Deterministic text chunking with overlap and source offsets."""

from __future__ import annotations

import re
from typing import TypedDict

_SENTENCE_BOUNDARY_PATTERN = re.compile(r"[.!?](?:[\"')\]]+)?(?:\s+|$)")


class Chunk(TypedDict):
    chunk_index: int
    start_char: int
    end_char: int
    text: str


def _find_split_after_double_newline(text: str, start: int, limit: int) -> int | None:
    split_at = text.rfind("\n\n", start, limit)
    if split_at == -1:
        return None
    candidate = split_at + 2
    if candidate <= start:
        return None
    return candidate


def _find_split_after_newline(text: str, start: int, limit: int) -> int | None:
    split_at = text.rfind("\n", start, limit)
    if split_at == -1:
        return None
    candidate = split_at + 1
    if candidate <= start:
        return None
    return candidate


def _find_sentenceish_split(text: str, start: int, limit: int) -> int | None:
    window = text[start:limit]
    last_match_end: int | None = None
    for match in _SENTENCE_BOUNDARY_PATTERN.finditer(window):
        candidate = start + match.end()
        if candidate > start:
            last_match_end = candidate
    return last_match_end


def _find_whitespace_split(text: str, start: int, limit: int) -> int | None:
    for idx in range(limit - 1, start - 1, -1):
        if text[idx].isspace():
            candidate = idx + 1
            if candidate > start:
                return candidate
    return None


def _select_end(text: str, start: int, max_chars: int) -> int:
    limit = min(start + max_chars, len(text))
    if limit == len(text):
        return limit

    for finder in (
        _find_split_after_double_newline,
        _find_split_after_newline,
        _find_sentenceish_split,
        _find_whitespace_split,
    ):
        candidate = finder(text, start, limit)
        if candidate is not None:
            return candidate

    return limit


def chunk(text: str, max_chars: int, overlap_chars: int) -> list[Chunk]:
    """Split ``text`` into chunks with deterministic offsets and overlap."""
    if max_chars <= 0:
        msg = "max_chars must be greater than 0"
        raise ValueError(msg)
    if overlap_chars < 0:
        msg = "overlap_chars must be >= 0"
        raise ValueError(msg)
    if overlap_chars >= max_chars:
        msg = "overlap_chars must be less than max_chars"
        raise ValueError(msg)
    if not text:
        return []

    chunks: list[Chunk] = []
    text_len = len(text)
    start = 0
    chunk_index = 0

    while start < text_len:
        end = _select_end(text=text, start=start, max_chars=max_chars)

        # Keep overlap exact whenever possible by avoiding too-small intermediate chunks.
        if end < text_len and (end - start) <= overlap_chars:
            end = min(start + max_chars, text_len)

        if end <= start:
            end = min(start + max_chars, text_len)

        chunks.append(
            {
                "chunk_index": chunk_index,
                "start_char": start,
                "end_char": end,
                "text": text[start:end],
            }
        )

        if end >= text_len:
            break

        next_start = end - overlap_chars
        if next_start <= start:
            next_start = end

        start = next_start
        chunk_index += 1

    return chunks
