class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, chunks: list[dict], top_k: int = 4) -> list[dict]:
        pairs = [(query, c["text"]) for c in chunks]
        scores = self.model.predict(pairs)
        for c, s in zip(chunks, scores):
            c["rerank_score"] = float(s)
        ranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
        return ranked[:top_k]