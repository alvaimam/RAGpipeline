import pytest
import embedding_client as ec
from embedding_client import get_embedding_client
from db import get_collection


class FakeResp:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self._data


class DummyEmbedder:
    """Stands in for a real embedder; get_collection only reads provider/model."""

    def __init__(self, provider, model):
        self.provider = provider
        self.model = model


def test_unknown_provider_raises():
    with pytest.raises(ValueError, match="Unknown embedding provider"):
        get_embedding_client("nope")


def test_provider_name_is_case_insensitive():
    assert get_embedding_client("OLLAMA").provider == "ollama"


def test_ollama_embedder_sends_chosen_model(monkeypatch):
    sent = {}

    def fake_post(url, json=None, **kwargs):
        sent.update(json)
        return FakeResp({"embedding": [0.1, 0.2]})

    monkeypatch.setattr(ec.requests, "post", fake_post)
    embedder = get_embedding_client("ollama", model="mxbai-embed-large")

    assert embedder.embed("hi") == [0.1, 0.2]
    assert sent["model"] == "mxbai-embed-large"
    assert sent["prompt"] == "hi"


def test_openai_embedder_parses_response(monkeypatch):
    sent = {}

    def fake_post(url, headers=None, json=None, **kwargs):
        sent.update(json)
        return FakeResp({"data": [{"embedding": [1.0, 2.0, 3.0]}]})

    monkeypatch.setattr(ec.requests, "post", fake_post)
    embedder = get_embedding_client("openai", model="text-embedding-3-small", api_key="test-key")

    assert embedder.embed("hi") == [1.0, 2.0, 3.0]
    assert sent["model"] == "text-embedding-3-small"
    assert sent["input"] == "hi"


def test_collection_rejects_different_embedding_model(tmp_path):
    path = str(tmp_path / "db")
    get_collection(DummyEmbedder("ollama", "nomic-embed-text"), path=path)

    with pytest.raises(ValueError, match="built with embedding"):
        get_collection(DummyEmbedder("openai", "text-embedding-3-small"), path=path)


def test_collection_reopens_with_same_embedding_model(tmp_path):
    path = str(tmp_path / "db")
    get_collection(DummyEmbedder("ollama", "nomic-embed-text"), path=path)
    get_collection(DummyEmbedder("ollama", "nomic-embed-text"), path=path)  # no error