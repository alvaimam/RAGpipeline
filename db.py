# db.py
import chromadb


def get_collection(embedder, name="my_docs", path="chroma_db"):
    """
    Open the collection, recording which embedding model built it.
    Refuses to open it with a different embedder, because vectors from
    different models can't be compared (and may have different sizes).
    """
    client = chromadb.PersistentClient(path=path)
    current = {
        "embedding_provider": embedder.provider,
        "embedding_model": embedder.model,
    }

    try:
        collection = client.get_collection(name)
    except Exception:  # collection doesn't exist yet
        return client.create_collection(name, metadata=current)

    stored = collection.metadata or {}
    stored_id = (stored.get("embedding_provider"), stored.get("embedding_model"))
    if stored_id != (None, None) and stored_id != (current["embedding_provider"], current["embedding_model"]):
        raise ValueError(
            f"Collection '{name}' was built with embedding {stored_id[0]}/{stored_id[1]}, "
            f"but you are using {current['embedding_provider']}/{current['embedding_model']}. "
            f"Re-ingest into a new collection or switch back."
        )
    return collection