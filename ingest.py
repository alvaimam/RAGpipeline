import os
import chromadb

from docling.document_converter import DocumentConverter

_docling_converter = DocumentConverter()
DOCS_DIR = "docs"

def extract_txt(path):
    with open(path, "r", errors="ignore") as f:
        return f.read()
    
def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping word-based chunks."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks    

def embed(text):
    resp = requests.post("http://localhost:11434/api/embeddings", json={
        "model": "nomic-embed-text",
        "prompt": text
    })
    return resp.json()["embedding"]

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