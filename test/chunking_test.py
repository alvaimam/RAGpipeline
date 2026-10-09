import pytest
from ingest import chunk_text

def test_empty_text_returns_no_chunks():
    assert chunk_text("") == []


def test_text_shorter_than_chunk_size_returns_single_chunk():
    text = "one two three four five"
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_text_exactly_chunk_size_returns_single_chunk():
    words = [f"word{i}" for i in range(500)]
    text = " ".join(words)
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    assert len(chunks) == 1


def test_no_words_are_lost():
    words = [f"word{i}" for i in range(1200)]
    text = " ".join(words)
    chunks = chunk_text(text, chunk_size=500, overlap=50)

    all_words_in_chunks = set()
    for chunk in chunks:
        all_words_in_chunks.update(chunk.split())

    assert set(words) <= all_words_in_chunks  


def test_consecutive_chunks_actually_overlap():
    words = [f"word{i}" for i in range(1200)]
    text = " ".join(words)
    chunks = chunk_text(text, chunk_size=500, overlap=50)

    for i in range(len(chunks) - 1):
        end_of_current = chunks[i].split()[-50:]
        start_of_next = chunks[i + 1].split()[:50]
        assert end_of_current == start_of_next


def test_zero_overlap_produces_no_duplicate_words():
    words = [f"word{i}" for i in range(1000)]
    text = " ".join(words)
    chunks = chunk_text(text, chunk_size=500, overlap=0)

    flat = [w for chunk in chunks for w in chunk.split()]
    assert flat == words  


def test_overlap_greater_equal_chunk_size_raises_instead_of_hanging():
    with pytest.raises(ValueError):
        chunk_text("overlap", chunk_size=10, overlap=10)