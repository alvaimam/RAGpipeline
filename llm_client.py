"""
Provider-agnostic LLM client.

Every backend implements the same `generate(prompt) -> str` interface, so the
rest of the codebase (query_transform.py, pipeline.py) never needs to know
which provider is behind it. Pick a provider by name via `get_llm_client`.
"""

from abc import ABC, abstractmethod
import os
import requests


class LLMClient(ABC):
    """Base class — every backend implements generate()."""

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError


class OllamaClient(LLMClient):
    def __init__(self, model: str = "llama3.2", url: str = "http://localhost:11434/api/generate"):
        self.model = model
        self.url = url

    def generate(self, prompt: str, **kwargs) -> str:
        resp = requests.post(
            self.url,
            json={"model": self.model, "prompt": prompt, "stream": False, **kwargs},
        )
        resp.raise_for_status()
        return resp.json()["response"].strip()


class OpenAIClient(LLMClient):
    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None):
        self.model = model
        self.api_key = api_key or os.environ["OPENAI_API_KEY"]
        self.url = "https://api.openai.com/v1/chat/completions"

    def generate(self, prompt: str, **kwargs) -> str:
        resp = requests.post(
            self.url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                **kwargs,
            },
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()


class AnthropicClient(LLMClient):
    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        api_key: str | None = None,
        max_tokens: int = 1024,
    ):
        self.model = model
        self.api_key = api_key or os.environ["ANTHROPIC_API_KEY"]
        self.max_tokens = max_tokens
        self.url = "https://api.anthropic.com/v1/messages"

    def generate(self, prompt: str, **kwargs) -> str:
        resp = requests.post(
            self.url,
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": self.model,
                "max_tokens": kwargs.pop("max_tokens", self.max_tokens),
                "messages": [{"role": "user", "content": prompt}],
                **kwargs,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return "".join(
            block["text"] for block in data["content"] if block["type"] == "text"
        ).strip()


LLM_CLIENTS = {
    "ollama": OllamaClient,
    "openai": OpenAIClient,
    "anthropic": AnthropicClient,
    "claude": AnthropicClient,  # alias
}


def get_llm_client(provider: str, **kwargs) -> LLMClient:
    """
    Factory: get_llm_client("openai", model="gpt-4o-mini")
             get_llm_client("anthropic", model="claude-sonnet-4-6")
             get_llm_client("ollama", model="llama3.2")
    """
    provider = provider.lower()
    if provider not in LLM_CLIENTS:
        raise ValueError(f"Unknown provider '{provider}'. Choose from {list(LLM_CLIENTS)}")
    return LLM_CLIENTS[provider](**kwargs)
