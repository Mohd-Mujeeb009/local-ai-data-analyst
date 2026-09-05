"""Tests for model resolution.

Network calls are stubbed - these cover the selection and failure logic, which
is where model retirements actually bite.
"""

import pytest

from backend import llm_client
from backend.llm_client import LLMError, resolve_model


@pytest.fixture(autouse=True)
def clear_cache():
    """Resolution is memoised per key; isolate each test from the last."""
    llm_client._model_cache.clear()
    yield
    llm_client._model_cache.clear()


def stub_available(monkeypatch, models):
    monkeypatch.setattr(llm_client, "available_models", lambda _key: set(models))


def test_prefers_first_reachable_candidate(monkeypatch):
    stub_available(monkeypatch, ["openai/gpt-oss-120b", "openai/gpt-oss-20b"])
    assert resolve_model("key", "text") == "openai/gpt-oss-120b"


def test_falls_through_to_later_candidate(monkeypatch):
    """A retired first choice degrades to the next, rather than breaking."""
    stub_available(monkeypatch, ["openai/gpt-oss-20b"])
    assert resolve_model("key", "text") == "openai/gpt-oss-20b"


def test_resolves_vision_separately(monkeypatch):
    stub_available(monkeypatch, ["qwen/qwen3.6-27b", "openai/gpt-oss-120b"])
    assert resolve_model("key", "vision") == "qwen/qwen3.6-27b"


def test_raises_when_no_candidate_is_available(monkeypatch):
    """
    Previously this returned the first candidate and let the completion call
    fail with an opaque provider 400.
    """
    stub_available(monkeypatch, ["some-unrelated-model"])
    with pytest.raises(LLMError) as exc:
        resolve_model("key", "text")

    message = str(exc.value)
    assert "None of the configured text models" in message
    assert "openai/gpt-oss-120b" in message   # names what it tried
    assert "GROQ_TEXT_MODEL" in message       # names the override


def test_raises_when_listing_fails(monkeypatch):
    stub_available(monkeypatch, [])
    with pytest.raises(LLMError, match="Could not list models"):
        resolve_model("key", "text")


def test_env_override_takes_precedence(monkeypatch):
    monkeypatch.setattr(
        llm_client, "TEXT_MODEL_CANDIDATES",
        ["my-pinned-model", "openai/gpt-oss-120b"],
    )
    stub_available(monkeypatch, ["my-pinned-model", "openai/gpt-oss-120b"])
    assert resolve_model("key", "text") == "my-pinned-model"


def test_result_is_cached(monkeypatch):
    calls = []

    def counting(_key):
        calls.append(1)
        return {"openai/gpt-oss-120b"}

    monkeypatch.setattr(llm_client, "available_models", counting)
    resolve_model("key", "text")
    resolve_model("key", "text")
    assert len(calls) == 1
