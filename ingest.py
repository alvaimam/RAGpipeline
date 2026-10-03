import os
import chromadb

from docling.document_converter import DocumentConverter

_docling_converter = DocumentConverter()
DOCS_DIR = "docs"

def extract_txt(path):
    with open(path, "r", errors="ignore") as f:
        return f.read()


def extract_with_docling(path):
    result = _docling_converter.convert(path)
    return result.document.export_to_markdown()


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


def main():
    client = chromadb.PersistentClient(path="chroma_db")
    collection = client.get_or_create_collection("my_docs")

    ids, embeddings, texts, metadatas = [], [], [], []
    for fname in os.listdir(DOCS_DIR):
        path = os.path.join(DOCS_DIR, fname)
        if not os.path.isfile(path):
            continue

        content = extract_text(path)
        for i, chunk in enumerate(chunk_text(content)):
            ids.append(f"{fname}-{i}")
            embeddings.append(embed(chunk))
            texts.append(chunk)
            metadatas.append({"source": fname})

    collection.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    print(f"Ingested {len(ids)} chunks from {DOCS_DIR}")