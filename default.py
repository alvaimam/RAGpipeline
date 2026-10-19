DEFAULTS = {
    "embedding": {"provider": "ollama", "model": "nomic-embed-text"},
    "query_transform": "auto",
    "retrieval": {"n_results_per_query": 5},
    "rerank": {"enabled": True, "top_k": 4},
}