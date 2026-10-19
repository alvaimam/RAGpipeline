# test_ingest_integration.py
import chromadb
from ingest import ingest_directory

def fake_embed(text):
    return [float(len(text)), 1.0, 0.0]   # fixed-size dummy vector, no Ollama needed

def test_ingest_stores_good_files_and_skips_bad(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("hello world")
    (docs / "photo.jpg").write_bytes(b"x")

    client = chromadb.PersistentClient(path=str(tmp_path / "db"))
    collection = client.get_or_create_collection("test_docs")

    count = ingest_directory(str(docs), collection, fake_embed)

    assert count == 1
    assert collection.count() == 1
    assert collection.get()["metadatas"][0]["source"] == "a.txt"

def test_reingest_does_not_duplicate(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("hello world")

    client = chromadb.PersistentClient(path=str(tmp_path / "db"))
    collection = client.get_or_create_collection("test_docs")

    ingest_directory(str(docs), collection, fake_embed)
    ingest_directory(str(docs), collection, fake_embed)

    assert collection.count() == 1   # upsert replaced, not duplicated