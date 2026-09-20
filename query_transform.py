import json
from llm_client import LLMClient


class QueryTransformer:
    """Base class — swap this out per strategy."""

    def __init__(self, llm_client: LLMClient | None = None):
        self.llm = llm_client

    def transform(self, query: str, max_sub_queries: int = 3) -> list[str]:
        raise NotImplementedError


class NoneTransformer(QueryTransformer):
    def transform(self, query: str, max_sub_queries: int = 3) -> list[str]:
        return [query]


class DecomposeTransformer(QueryTransformer):
    def transform(self, query: str, max_sub_queries: int = 3) -> list[str]:
        prompt = f"""Split the following user question into distinct, atomic sub-questions.
If it's already a single question, return just that one question.
Return ONLY a JSON array of strings, nothing else. Max {max_sub_queries} sub-questions.
Question: {query}"""
        try:
            raw = self.llm.generate(prompt)
            raw = raw.replace("```json", "").replace("```", "").strip()
            sub_queries = json.loads(raw)
            if not isinstance(sub_queries, list) or not sub_queries:
                raise ValueError("bad format")
            return sub_queries[:max_sub_queries]
        except Exception:
            return [query]  # fallback to original on parse failure


class HyDETransformer(QueryTransformer):
    def transform(self, query: str, max_sub_queries: int = 3) -> list[str]:
        prompt = f"""Write a short hypothetical passage (3-4 sentences) that would answer this question,
as if it came from an official document. Do not mention that it's hypothetical.
Question: {query}"""
        try:
            hypothetical_doc = self.llm.generate(prompt)
            return [hypothetical_doc]
        except Exception:
            return [query]


class DecomposeThenHyDETransformer(QueryTransformer):
    def __init__(self, llm_client: LLMClient | None = None):
        super().__init__(llm_client)
        self.decomposer = DecomposeTransformer(llm_client)
        self.hyde = HyDETransformer(llm_client)

    def transform(self, query: str, max_sub_queries: int = 3) -> list[str]:
        sub_queries = self.decomposer.transform(query, max_sub_queries)
        hypothetical_docs = []
        for sq in sub_queries:
            hypothetical_docs.extend(self.hyde.transform(sq))
        return hypothetical_docs


def looks_multi_intent(query: str) -> bool:
    markers = [" and ", "?", ";", " also "]
    return sum(query.lower().count(m) for m in markers) >= 2 or query.count("?") >= 2


def is_short_and_vague(query: str) -> bool:
    return len(query.split()) <= 4


class AutoTransformer(QueryTransformer):
    def transform(self, query: str, max_sub_queries: int = 3) -> list[str]:
        if looks_multi_intent(query):
            return DecomposeTransformer(self.llm).transform(query, max_sub_queries)
        elif is_short_and_vague(query):
            return HyDETransformer(self.llm).transform(query, max_sub_queries)
        else:
            return [query]


TRANSFORMERS = {
    "none": NoneTransformer,
    "decompose": DecomposeTransformer,
    "hyde": HyDETransformer,
    "decompose_then_hyde": DecomposeThenHyDETransformer,
    "auto": AutoTransformer,
}


def get_transformer(name: str, llm_client: LLMClient | None = None) -> QueryTransformer:
    return TRANSFORMERS.get(name, NoneTransformer)(llm_client)
