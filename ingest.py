import os
import requests
import chromadb
import pdfplumber

DOCS_DIR = "docs"
CHUNK_SIZE = 500

def chunk_text(text, size=CHUNK_SIZE):
    return [text[i:i+size] for i in range(0, len(text), size)]

def embed(text):
    resp = requests.post("http://localhost:11434/api/embeddings", json={
        "model": "nomic-embed-text",
        "prompt": text
    })
    return resp.json()["embedding"]

def extract_txt(path):
    with open(path, "r", errors="ignore") as f:
        return f.read()

def extract_pdf(path):
    pages = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
    return "\n".join(pages)

EXTRACTORS = {
    ".rtf": extract_txt,
    ".pdf": extract_pdf,
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

if __name__ == "__main__":
    main()