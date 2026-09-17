class Retriever:
    def __init__(self, collection, embed_fn):
        self.collection = collection
        self.embed_fn = embed_fn

    def retrieve(self, queries: list[str], n_results: int = 5) -> list[dict]:
        """Runs dense retrieval per query string, merges + dedupes by chunk text."""
        seen = {}
        for q in queries:
            embedding = self.embed_fn(q)
            results = self.collection.query(query_embeddings=[embedding], n_results=n_results)
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            distances = results.get("distances", [[None] * len(docs)])[0]
            for doc, meta, dist in zip(docs, metas, distances):
                key = doc  # dedupe on chunk text; use a stable id if you have one
                if key not in seen or (dist is not None and dist < seen[key]["distance"]):
                    seen[key] = {"text": doc, "source": meta.get("source"), "distance": dist}
        # sort by distance ascending (lower = more similar, for most metrics)
        ranked = sorted(seen.values(), key=lambda x: (x["distance"] is None, x["distance"]))
        return ranked