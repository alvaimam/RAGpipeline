# ingest.py
import os
import logging
from config import resolve_config
from db import get_collection
from defaults import DEFAULTS
from embedding_client import get_embedding_client

logger = logging.getLogger(__name__)
DOCS_DIR = "docs"

_docling_converter = None


def extract_txt(path):
    with open(path, "r", errors="ignore") as f:
        return f.read()


def extract_with_docling(path):
    global _docling_converter
    if _docling_converter is None:
        from docling.document_converter import DocumentConverter
        _docling_converter = DocumentConverter()
    return _docling_converter.convert(path).document.export_to_markdown()


EXTRACTORS = {
    ".txt": extract_txt,
    ".pdf": extract_with_docling,
}


def extract_text(path):
    ext = os.path.splitext(path)[1].lower()
    extractor = EXTRACTORS.get(ext)
    if extractor is None:
        raise ValueError(f"Unsupported file type: {ext} ({path})")
    return extractor(path)


def load_document(path):
    try:
        content = extract_text(path)
    except ValueError as e:
        logger.warning("Skipping %s: %s", path, e)
        return None
    if not content.strip():
        logger.warning("Skipping %s: no text extracted", path)
        return None
    return content


def chunk_text(text, chunk_size=500, overlap=50):
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        chunks.append(" ".join(words[start:start + chunk_size]))
        start += chunk_size - overlap
    return chunks


def ingest_directory(docs_dir, collection, embed_fn):
    ids, embeddings, texts, metadatas = [], [], [], []
    for fname in os.listdir(docs_dir):
        path = os.path.join(docs_dir, fname)
        if not os.path.isfile(path):
            continue

        content = load_document(path)
        if content is None:
            continue
        for i, chunk in enumerate(chunk_text(content)):
            ids.append(f"{fname}-{i}")
            embeddings.append(embed_fn(chunk))
            texts.append(chunk)
            metadatas.append({"source": fname})

    if ids:
        collection.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    return len(ids)


def main():
    config = resolve_config(DEFAULTS)
    embedder = get_embedding_client(config.embedding.provider, model=config.embedding.model)
    collection = get_collection(embedder)
    count = ingest_directory(DOCS_DIR, collection, embedder.embed)
    print(f"Ingested {count} chunks from {DOCS_DIR}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()