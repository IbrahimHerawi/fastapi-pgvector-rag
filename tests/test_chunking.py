import pytest

from rag_api.services.chunking import chunk


def _assert_chunk_integrity(text: str, chunks: list[dict[str, object]], max_chars: int) -> None:
    assert chunks
    assert chunks[0]["start_char"] == 0

    for idx, item in enumerate(chunks):
        start = item["start_char"]
        end = item["end_char"]

        assert item["chunk_index"] == idx
        assert isinstance(start, int)
        assert isinstance(end, int)
        assert 0 <= start < end <= len(text)
        assert end - start <= max_chars
        assert item["text"] == text[start:end]


def _assert_exact_overlap_when_possible(
    text: str,
    chunks: list[dict[str, object]],
    overlap_chars: int,
) -> None:
    for prev, current in zip(chunks, chunks[1:]):
        prev_start = prev["start_char"]
        prev_end = prev["end_char"]
        prev_len = prev_end - prev_start

        if prev_end < len(text) and prev_len > overlap_chars:
            assert current["start_char"] == prev_end - overlap_chars


def test_chunk_short_text_returns_one_chunk_with_exact_offsets() -> None:
    text = "Short text."
    chunks = chunk(text=text, max_chars=100, overlap_chars=20)

    assert chunks == [
        {
            "chunk_index": 0,
            "start_char": 0,
            "end_char": len(text),
            "text": text,
        }
    ]


def test_chunk_long_text_respects_max_len_offsets_and_overlap() -> None:
    text = (
        "Paragraph one has enough words to require splitting. It keeps going for a while.\n\n"
        "Paragraph two also has several sentences. Another sentence appears here.\n"
        "Final line adds extra words so the chunker must create additional chunks."
    )
    max_chars = 80
    overlap_chars = 15

    chunks = chunk(text=text, max_chars=max_chars, overlap_chars=overlap_chars)

    assert len(chunks) > 1
    _assert_chunk_integrity(text=text, chunks=chunks, max_chars=max_chars)
    _assert_exact_overlap_when_possible(text=text, chunks=chunks, overlap_chars=overlap_chars)


def test_chunk_prefers_double_newline_over_other_splits() -> None:
    text = (
        "Alpha sentence one.\n\n"
        "Beta line has extra words and a newline here\n"
        "Gamma keeps going so we do not hit end yet."
    )
    max_chars = 90

    chunks = chunk(text=text, max_chars=max_chars, overlap_chars=5)

    expected_end = text.index("\n\n") + 2
    assert chunks[0]["end_char"] == expected_end
    assert chunks[0]["text"] == text[:expected_end]


def test_chunk_prefers_newline_over_sentenceish_when_no_double_newline() -> None:
    text = (
        "First sentence ends here. Middle text stays on this line\n"
        "Tail text remains so split is not final."
    )
    max_chars = 70

    chunks = chunk(text=text, max_chars=max_chars, overlap_chars=5)

    expected_end = text.index("\n") + 1
    assert chunks[0]["end_char"] == expected_end


def test_chunk_prefers_sentenceish_over_whitespace() -> None:
    text = (
        "Sentence one ends here. Sentence two has many words and keeps going beyond "
        "the max so a split is needed."
    )
    max_chars = 80

    chunks = chunk(text=text, max_chars=max_chars, overlap_chars=10)

    expected_end = text.index(". ") + 2
    assert chunks[0]["end_char"] == expected_end


def test_chunk_uses_whitespace_fallback_when_no_other_boundaries_exist() -> None:
    text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi"
    max_chars = 31

    chunks = chunk(text=text, max_chars=max_chars, overlap_chars=5)

    expected_end = max(idx + 1 for idx, char in enumerate(text[:max_chars]) if char.isspace())
    assert chunks[0]["end_char"] == expected_end


def test_chunk_is_deterministic() -> None:
    text = (
        "Deterministic chunking should produce the same output each time.\n"
        "This content is long enough to create multiple chunks over multiple calls."
    )

    first = chunk(text=text, max_chars=50, overlap_chars=12)
    second = chunk(text=text, max_chars=50, overlap_chars=12)

    assert first == second


def test_chunk_zero_overlap_starts_next_chunk_at_previous_end() -> None:
    text = " ".join(f"token{i:02d}" for i in range(30))
    chunks = chunk(text=text, max_chars=35, overlap_chars=0)

    for prev, current in zip(chunks, chunks[1:]):
        assert current["start_char"] == prev["end_char"]


def test_chunk_empty_text_returns_empty_list() -> None:
    assert chunk(text="", max_chars=20, overlap_chars=5) == []


@pytest.mark.parametrize(
    ("max_chars", "overlap_chars"),
    [
        (0, 0),
        (-1, 0),
        (10, -1),
        (10, 10),
        (10, 11),
    ],
)
def test_chunk_rejects_invalid_arguments(max_chars: int, overlap_chars: int) -> None:
    with pytest.raises(ValueError):
        chunk(text="content", max_chars=max_chars, overlap_chars=overlap_chars)
