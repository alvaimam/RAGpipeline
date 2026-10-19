import pytest
import ingest
from ingest import extract_text, extract_txt, load_document


def test_txt_extractor_reads_content(tmp_path):
    f = tmp_path / "note.txt"
    f.write_text("hello world", encoding="utf-8")
    assert extract_txt(str(f)) == "hello world"



def test_unsupported_extension_raises(tmp_path):
    f = tmp_path / "photo.jpg"
    f.write_bytes(b"not text")
    with pytest.raises(ValueError, match="Unsupported file type"):
        extract_text(str(f))


def test_file_without_extension_raises(tmp_path):
    f = tmp_path / "README"
    f.write_text("hi")
    with pytest.raises(ValueError):
        extract_text(str(f))


def test_pdf_is_routed_to_pdf_extractor(monkeypatch):
    monkeypatch.setitem(ingest.EXTRACTORS, ".pdf", lambda path: "fake pdf text")
    assert extract_text("report.pdf") == "fake pdf text"


def test_extension_match_is_case_insensitive(monkeypatch):
    monkeypatch.setitem(ingest.EXTRACTORS, ".pdf", lambda path: "fake pdf text")
    assert extract_text("REPORT.PDF") == "fake pdf text"


def test_load_document_returns_text_for_valid_file(tmp_path):
    f = tmp_path / "note.txt"
    f.write_text("some content")
    assert load_document(str(f)) == "some content"


def test_load_document_skips_unsupported(tmp_path):
    f = tmp_path / "photo.jpg"
    f.write_bytes(b"x")
    assert load_document(str(f)) is None


def test_load_document_skips_empty_text(tmp_path):
    f = tmp_path / "empty.txt"
    f.write_text("   \n  ")
    assert load_document(str(f)) is None