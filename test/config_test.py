import pytest
from config import RAGConfig, resolve_config


def test_empty_defaults_give_dataclass_defaults():
    cfg = resolve_config({})
    assert isinstance(cfg, RAGConfig)
    assert cfg.query_transform == "none"
    assert cfg.max_sub_queries == 3
    assert cfg.retrieval.n_results_per_query == 5
    assert cfg.rerank.enabled is False
    assert cfg.rerank.top_k == 4
    assert cfg.embedding.provider == "ollama"
    assert cfg.embedding.model == "nomic-embed-text"


def test_tenant_defaults_are_applied():
    cfg = resolve_config({
        "query_transform": "auto",
        "retrieval": {"n_results_per_query": 8},
        "rerank": {"enabled": True, "top_k": 2},
        "embedding": {"provider": "openai", "model": "text-embedding-3-small"},
    })
    assert cfg.query_transform == "auto"
    assert cfg.retrieval.n_results_per_query == 8
    assert cfg.rerank.enabled is True
    assert cfg.rerank.top_k == 2
    assert cfg.embedding.provider == "openai"
    assert cfg.embedding.model == "text-embedding-3-small"


def test_precedence_call_over_collection_over_tenant():
    assert resolve_config({"max_sub_queries": 1}).max_sub_queries == 1
    assert resolve_config({"max_sub_queries": 1}, {"max_sub_queries": 2}).max_sub_queries == 2
    assert resolve_config(
        {"max_sub_queries": 1}, {"max_sub_queries": 2}, {"max_sub_queries": 3}
    ).max_sub_queries == 3


def test_resolve_does_not_mutate_its_inputs():
    tenant = {"retrieval": {"n_results_per_query": 8}}
    collection = {"retrieval": {"n_results_per_query": 2}}
    resolve_config(tenant, collection)
    assert tenant == {"retrieval": {"n_results_per_query": 8}}
    assert collection == {"retrieval": {"n_results_per_query": 2}}


def test_nested_override_replaces_whole_section_not_just_one_key():
    # Documents CURRENT behavior: the merge is shallow ({**a, **b}), so a
    # collection override of "retrieval" throws away the tenant's other
    # retrieval keys. If you switch to a deep merge, flip these assertions:
    # mode should become "hybrid".
    cfg = resolve_config(
        {"retrieval": {"mode": "hybrid", "n_results_per_query": 8}},
        {"retrieval": {"n_results_per_query": 2}},
    )
    assert cfg.retrieval.n_results_per_query == 2
    assert cfg.retrieval.mode == "dense"  # tenant's "hybrid" was lost


def test_typo_inside_a_section_raises():
    with pytest.raises(TypeError):
        resolve_config({"retrieval": {"n_result_per_query": 5}})  # typo


def test_unknown_top_level_key_is_silently_ignored():
    # Documents CURRENT behavior: a typo like "rerankk" is dropped without error.
    cfg = resolve_config({"rerankk": {"enabled": True}})
    assert cfg.rerank.enabled is False