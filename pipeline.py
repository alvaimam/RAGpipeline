from config import RAGConfig
from llm_client import get_llm_client
from query_transform import get_transformer
from retrieval import Retriever
from rerank import Reranker


class RAGPipeline:
    def __init__(self,collection,embed_fn,config: RAGConfig,provider: str = "ollama",model: str = "llama3.2",
        **llm_kwargs,
    ):
        self.collection = collection
        self.embed_fn = embed_fn
        self.config = config
        self.llm = get_llm_client(provider, model=model, **llm_kwargs)
        self.retriever = Retriever(collection, embed_fn)
        self._reranker = None  # lazy-load, it's a heavier import

    @property
    def reranker(self):
        if self._reranker is None and self.config.rerank.enabled:
            self._reranker = Reranker(self.config.rerank.model)
        return self._reranker

    def ask(self, query: str) -> str:
        transformer = get_transformer(self.config.query_transform, self.llm)
        try:
            transformed_queries = transformer.transform(query, self.config.max_sub_queries)
        except Exception:
            transformed_queries = [query]

        chunks = self.retriever.retrieve(
            transformed_queries,
            n_results=self.config.retrieval.n_results_per_query,
        )
        if self.config.rerank.enabled and chunks:
            chunks = self.reranker.rerank(query, chunks, top_k=self.config.rerank.top_k)

        context = "\n\n".join(f"[{c['source']}]\n{c['text']}" for c in chunks)
        prompt = f"""Answer the question using only the context below. If the context doesn't contain the answer, say so.
                Context:{context}
                Question: {query}"""
        return self.llm.generate(prompt)
