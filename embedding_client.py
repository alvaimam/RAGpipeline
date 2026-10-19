"""
Provider-agnostic embedding client.

Mirrors llm_client.py: every backend implements `embed(text) -> list[float]`,
so ingest.py and main.py never need to know which provider is behind it.
Pick one by name via `get_embedding_client`.

Note: the embedding model used at ingestion MUST be the same one used at query
time. Switching provider/model means re-ingesting into a new collection.
"""

from abc import ABC, abstractmethod
import os
import requests


class EmbeddingClient(ABC):
    """Base class: every backend implements embed()."""

    provider: str = ""
    model: str = ""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        raise NotImplementedError


class OllamaEmbedder(EmbeddingClient):
    provider = "ollama"

    def __init__(
        self,
        model: str = "nomic-embed-text",
        url: str = "http://localhost:11434/api/embeddings",
    ):
        self.model = model
        self.url = url

    def embed(self, text: str) -> list[float]:
        resp = requests.post(
            self.url,
            json={"model": self.model, "prompt": text},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["embedding"]


class OpenAIEmbedder(EmbeddingClient):
    provider = "openai"

    def __init__(self, model: str = "text-embedding-3-small", api_key: str | None = None):
        self.model = model
        self.api_key = api_key or os.environ["OPENAI_API_KEY"]
        self.url = "https://api.openai.com/v1/embeddings"

    def embed(self, text: str) -> list[float]:
        resp = requests.post(
            self.url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "input": text},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]


EMBEDDING_CLIENTS = {
    "ollama": OllamaEmbedder,
    "openai": OpenAIEmbedder,
}


def get_embedding_client(provider: str, **kwargs) -> EmbeddingClient:
    """
    Factory: get_embedding_client("ollama", model="nomic-embed-text")
             get_embedding_client("openai", model="text-embedding-3-small")
    """
    provider = provider.lower()
    if provider not in EMBEDDING_CLIENTS:
        raise ValueError(
            f"Unknown embedding provider '{provider}'. Choose from {list(EMBEDDING_CLIENTS)}"
        )
    return EMBEDDING_CLIENTS[provider](**kwargs)