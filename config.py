from dataclasses import dataclass, field
from typing import Literal, Optional

import requests

def llm_model(text):
    resp = requests.post("http://localhost:11434/api/embeddings", json={
        "model": "nomic-embed-text",
        "prompt": text
    })
    return resp.json()["embedding"]

@dataclass
class RerankConfig:
    enabled: bool = False
    model: str = "BAAI/bge-reranker-base"
    top_k: int = 4

@dataclass
class RetrievalConfig:
    mode: Literal["dense", "hybrid"] = "dense"
    n_results_per_query: int = 5
    dense_weight: float = 0.7  # only used in hybrid mode

@dataclass
class RAGConfig:
    query_transform: Literal["none", "decompose", "hyde", "decompose_then_hyde", "auto"] = "none"
    max_sub_queries: int = 3
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    rerank: RerankConfig = field(default_factory=RerankConfig)
    fallback_on_transform_failure: str = "original_query"

def resolve_config(tenant_defaults: dict, collection_override: Optional[dict] = None,
                    call_override: Optional[dict] = None) -> RAGConfig:
    merged = {**tenant_defaults, **(collection_override or {}), **(call_override or {})}
    retrieval = RetrievalConfig(**merged.get("retrieval", {}))
    rerank = RerankConfig(**merged.get("rerank", {}))
    return RAGConfig(
        query_transform=merged.get("query_transform", "none"),
        max_sub_queries=merged.get("max_sub_queries", 3),
        retrieval=retrieval,
        rerank=rerank,
        fallback_on_transform_failure=merged.get("fallback_on_transform_failure", "original_query"),
    )