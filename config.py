from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class EmbeddingConfig:
    provider: str = "ollama"
    model: str = "nomic-embed-text"


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
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    rerank: RerankConfig = field(default_factory=RerankConfig)
    fallback_on_transform_failure: str = "original_query"


def resolve_config(tenant_defaults: dict, collection_override: Optional[dict] = None,
                    call_override: Optional[dict] = None) -> RAGConfig:
    merged = {**tenant_defaults, **(collection_override or {}), **(call_override or {})}
    embedding = EmbeddingConfig(**merged.get("embedding", {}))
    retrieval = RetrievalConfig(**merged.get("retrieval", {}))
    rerank = RerankConfig(**merged.get("rerank", {}))
    return RAGConfig(
        query_transform=merged.get("query_transform", "none"),
        max_sub_queries=merged.get("max_sub_queries", 3),
        embedding=embedding,
        retrieval=retrieval,
        rerank=rerank,
        fallback_on_transform_failure=merged.get("fallback_on_transform_failure", "original_query"),
    )