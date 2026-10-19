# main.py  (query / RAG)
from config import resolve_config
from db import get_collection
from defaults import DEFAULTS
from embedding_client import get_embedding_client
from pipeline import RAGPipeline

config = resolve_config(DEFAULTS)
embedder = get_embedding_client(config.embedding.provider, model=config.embedding.model)
collection = get_collection(embedder)

pipeline = RAGPipeline(collection=collection, embed_fn=embedder.embed, config=config)

if __name__ == "__main__":
    while True:
        q = input("\nAsk something (or 'quit'): ")
        if q.lower() == "quit":
            break
        print("\n" + pipeline.ask(q))