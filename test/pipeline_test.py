import pytest

import pipeline as pipeline_module
from config import resolve_config
from pipeline import RAGPipeline

CHUNKS = [
    {"source": "a.txt", "text": "Paris is the capital of France."},
    {"source": "b.txt", "text": "Berlin is the capital of Germany."},
]


# --- fakes: no Ollama, no Chroma, no reranker model needed ---

class FakeLLM:
    def __init__(self, answer="the answer"):
        self.answer = answer
        self.prompts = []

    def generate(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return self.answer


class BrokenLLM:
    def generate(self, prompt, **kwargs):
        raise ConnectionError("ollama down")


class FakeRetriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.calls = []

    def retrieve(self, queries, n_results):
        self.calls.append({"queries": queries, "n_results": n_results})
        return list(self.chunks)


class FakeReranker:
    """Reverses the chunks, then cuts to top_k, so tests can see it ran."""

    def __init__(self):
        self.calls = []

    def rerank(self, query, chunks, top_k):
        self.calls.append({"query": query, "top_k": top_k})
        return list(reversed(chunks))[:top_k]


class PassthroughTransformer:
    def transform(self, query, max_sub_queries):
        return [query]


class FakeTransformer:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = []

    def transform(self, query, max_sub_queries):
        self.calls.append({"query": query, "max_sub_queries": max_sub_queries})
        if self.error:
            raise self.error
        return self.result


def make_pipeline(monkeypatch, config_dict=None, chunks=CHUNKS, transformer=None, llm=None):
    llm = llm or FakeLLM()
    retriever = FakeRetriever(chunks)
    transformer = transformer or PassthroughTransformer()
    monkeypatch.setattr(pipeline_module, "get_llm_client", lambda provider, **kw: llm)
    monkeypatch.setattr(pipeline_module, "Retriever", lambda collection, embed_fn: retriever)
    monkeypatch.setattr(pipeline_module, "get_transformer", lambda mode, llm_: transformer)
    p = RAGPipeline(collection=None, embed_fn=None, config=resolve_config(config_dict or {}))
    return p, llm, retriever


# --- ask: answer and prompt ---

def test_answer_is_the_llm_output(monkeypatch):
    p, _, _ = make_pipeline(monkeypatch, llm=FakeLLM("Paris"))
    assert p.ask("capital of France?") == "Paris"


def test_prompt_contains_question_chunks_and_sources(monkeypatch):
    p, llm, _ = make_pipeline(monkeypatch)
    p.ask("capital of France?")
    prompt = llm.prompts[0]
    assert "capital of France?" in prompt
    assert "Paris is the capital of France." in prompt
    assert "[a.txt]" in prompt
    assert "[b.txt]" in prompt


def test_no_chunks_still_asks_llm_with_say_so_instruction(monkeypatch):
    # "I don't know" is produced by the LLM following the prompt, not by code.
    # This test only proves the instruction is in the prompt.
    p, llm, _ = make_pipeline(monkeypatch, chunks=[])
    p.ask("anything")
    assert len(llm.prompts) == 1
    assert "say so" in llm.prompts[0]


def test_llm_error_propagates(monkeypatch):
    p, _, _ = make_pipeline(monkeypatch, llm=BrokenLLM())
    with pytest.raises(ConnectionError):
        p.ask("q")


# --- ask: query transform and retrieval ---

def test_retrieval_uses_n_results_from_config(monkeypatch):
    p, _, retriever = make_pipeline(monkeypatch, {"retrieval": {"n_results_per_query": 7}})
    p.ask("q")
    assert retriever.calls[0]["n_results"] == 7


def test_retrieval_receives_the_transformed_queries(monkeypatch):
    transformer = FakeTransformer(result=["sub one", "sub two"])
    p, _, retriever = make_pipeline(monkeypatch, transformer=transformer)
    p.ask("original")
    assert retriever.calls[0]["queries"] == ["sub one", "sub two"]


def test_transformer_gets_max_sub_queries_from_config(monkeypatch):
    transformer = FakeTransformer(result=["x"])
    p, _, _ = make_pipeline(monkeypatch, {"max_sub_queries": 2}, transformer=transformer)
    p.ask("original")
    assert transformer.calls[0] == {"query": "original", "max_sub_queries": 2}


def test_transform_failure_falls_back_to_original_query(monkeypatch):
    transformer = FakeTransformer(error=RuntimeError("llm down"))
    p, _, retriever = make_pipeline(monkeypatch, transformer=transformer)
    p.ask("original")
    assert retriever.calls[0]["queries"] == ["original"]


# --- ask: rerank ---

def test_rerank_disabled_does_not_call_reranker(monkeypatch):
    p, _, _ = make_pipeline(monkeypatch, {"rerank": {"enabled": False}})
    reranker = FakeReranker()
    p._reranker = reranker
    p.ask("q")
    assert reranker.calls == []


def test_rerank_enabled_uses_original_query_and_top_k(monkeypatch):
    p, llm, _ = make_pipeline(
        monkeypatch,
        {"rerank": {"enabled": True, "top_k": 1}},
        transformer=FakeTransformer(result=["rewritten"]),
    )
    reranker = FakeReranker()
    p._reranker = reranker
    p.ask("original")

    assert reranker.calls[0]["query"] == "original"  # not the rewritten one
    assert reranker.calls[0]["top_k"] == 1
    # only what the reranker returned reaches the prompt (reversed, cut to 1 -> Berlin)
    assert "Berlin is the capital of Germany." in llm.prompts[0]
    assert "Paris is the capital of France." not in llm.prompts[0]


def test_rerank_enabled_but_no_chunks_skips_reranker(monkeypatch):
    p, _, _ = make_pipeline(monkeypatch, {"rerank": {"enabled": True}}, chunks=[])
    reranker = FakeReranker()
    p._reranker = reranker
    p.ask("q")
    assert reranker.calls == []


# --- reranker lazy loading ---

def test_reranker_is_none_when_disabled(monkeypatch):
    p, _, _ = make_pipeline(monkeypatch, {"rerank": {"enabled": False}})
    assert p.reranker is None


def test_reranker_is_created_once_and_only_when_first_needed(monkeypatch):
    created = []
    monkeypatch.setattr(
        pipeline_module, "Reranker", lambda model: created.append(model) or FakeReranker()
    )
    p, _, _ = make_pipeline(monkeypatch, {"rerank": {"enabled": True, "model": "my/model"}})

    assert created == []  # nothing loaded yet
    p.reranker
    p.reranker
    assert created == ["my/model"]  # loaded once, with the configured model