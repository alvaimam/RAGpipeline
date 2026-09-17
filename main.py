from config import resolve_config
from pipeline import RAGPipeline

import chromadb
import requests


tenant_defaults = {
    "query_transform": "auto",
    "retrieval": {"n_results_per_query": 5},
    "rerank": {"enabled": True, "top_k": 4},
}

def embed(text):
    resp = requests.post("http://localhost:11434/api/embeddings", json={
        "model": "nomic-embed-text",
        "prompt": text
    })
    return resp.json()["embedding"]

client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection("my_docs")

config = resolve_config(tenant_defaults)
pipeline = RAGPipeline(collection=collection, embed_fn=embed, config=config)

if __name__ == "__main__":
    while True:
        q = input("\nAsk something (or 'quit'): ")
        if q.lower() == "quit":
            break
        print("\n" + pipeline.ask(q))