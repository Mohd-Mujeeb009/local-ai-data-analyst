"""
Groq API client.

Wraps model resolution, streaming and error translation. The API key is always
passed in explicitly - it is never read from or written to the process
environment, so concurrent users of a deployed app cannot see each other's key.
"""

from groq import Groq

from backend.config import (
    ANSWER_TEMPERATURE,
    MAX_TOKENS,
    TEXT_MODEL_CANDIDATES,
    VISION_MODEL_CANDIDATES,
    candidates,
)

# Resolved model IDs are cached per key so we probe the models endpoint once
# per session rather than on every message.
_model_cache = {}


class LLMError(RuntimeError):
    """A failure talking to the provider, with a message fit for the UI."""


def _client(api_key):
    if not api_key:
        raise LLMError("No API key provided. Enter your Groq key in the sidebar.")
    try:
        return Groq(api_key=api_key)
    except Exception as exc:
        raise LLMError(f"Could not initialise the Groq client: {exc}") from exc


def available_models(api_key):
    """
    List model IDs the key can reach.

    Returns:
        set[str]: Available model IDs, or an empty set if the call fails.
    """
    try:
        response = _client(api_key).models.list()
        return {m.id for m in response.data}
    except Exception:
        return set()


def check_connection(api_key):
    """Report whether the key authenticates against Groq."""
    if not api_key:
        return False
    try:
        _client(api_key).models.list()
        return True
    except Exception:
        return False


def resolve_model(api_key, kind="text"):
    """
    Pick the best available model of a given kind.

    Falls through the configured candidate list rather than trusting a single
    hardcoded ID, so a provider retirement degrades instead of breaking.

    Args:
        api_key: Groq API key.
        kind: Either "text" or "vision".

    Returns:
        str: A model ID.

    Raises:
        LLMError: If none of the candidates are reachable.
    """
    cache_key = (api_key[-8:] if api_key else "", kind)
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    wanted = candidates(
        TEXT_MODEL_CANDIDATES if kind == "text" else VISION_MODEL_CANDIDATES
    )
    reachable = available_models(api_key)

    if not reachable:
        raise LLMError(
            "Could not list models on this account. Check the API key and network, "
            "then try again."
        )

    chosen = next((m for m in wanted if m in reachable), None)
    if not chosen:
        # Previously this fell back to wanted[0] and let the completion call
        # fail with a bare provider 400. Naming the candidates turns a confusing
        # error into an actionable one, which matters because model retirements
        # are the most common way this app breaks.
        raise LLMError(
            f"None of the configured {kind} models are available on this account.\n"
            f"Tried: {', '.join(wanted)}.\n"
            f"Set GROQ_{kind.upper()}_MODEL to a model ID from "
            "https://console.groq.com/docs/models"
        )

    _model_cache[cache_key] = chosen
    return chosen


def _translate(exc):
    """Convert a provider exception into a message worth showing a user."""
    text = str(exc).lower()
    if "authentication" in text or "api key" in text or "401" in text:
        return LLMError("Invalid API key. Check the key in the sidebar.")
    if "rate" in text and "limit" in text or "429" in text:
        return LLMError("Rate limit reached on Groq's free tier. Wait a moment and retry.")
    if "does not exist" in text or "decommission" in text or "404" in text:
        return LLMError(
            f"The requested model is unavailable: {exc}. "
            "Set GROQ_TEXT_MODEL or GROQ_VISION_MODEL to a current model ID."
        )
    if "context" in text and ("length" in text or "window" in text):
        return LLMError("The conversation grew past the model's context window. Clear the chat to continue.")
    return LLMError(f"Groq API error: {exc}")


def complete(api_key, messages, model=None, temperature=ANSWER_TEMPERATURE,
             json_mode=False, return_usage=False):
    """
    Run a non-streaming completion.

    Args:
        api_key: Groq API key.
        messages: Message dicts in OpenAI format.
        model: Model ID. Resolved automatically when omitted.
        temperature: Sampling temperature.
        json_mode: Constrain the response to a JSON object.
        return_usage: Also return the provider's token counts. Used by the
            evaluation harness to report real cost per query; the app itself
            ignores it.

    Returns:
        str: The response text, or (text, usage_dict) when return_usage is set.
        The usage dict carries "prompt_tokens", "completion_tokens" and "model".

    Raises:
        LLMError: On any provider failure.
    """
    client = _client(api_key)
    resolved = model or resolve_model(api_key, "text")
    kwargs = {
        "model": resolved,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": MAX_TOKENS,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    try:
        response = client.chat.completions.create(**kwargs)
        text = response.choices[0].message.content or ""
    except Exception as exc:
        raise _translate(exc) from exc

    if not return_usage:
        return text

    usage = getattr(response, "usage", None)
    return text, {
        "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
        "model": resolved,
    }


def stream(api_key, messages, model=None, temperature=ANSWER_TEMPERATURE):
    """
    Run a streaming completion, yielding text chunks as they arrive.

    Groq's throughput makes streaming the difference between a spinner and a
    response that appears to type itself, so it is the default path for anything
    the user reads.

    Yields:
        str: Successive content deltas.

    Raises:
        LLMError: On any provider failure.
    """
    client = _client(api_key)
    try:
        response = client.chat.completions.create(
            model=model or resolve_model(api_key, "text"),
            messages=messages,
            temperature=temperature,
            max_tokens=MAX_TOKENS,
            stream=True,
        )
        for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
    except Exception as exc:
        raise _translate(exc) from exc


def stream_vision(api_key, question, image_base64, mime_type, history=None):
    """
    Stream a vision completion for a single image.

    The image rides only on the current turn; prior turns are replayed as text
    so a sticky upload cannot silently re-bill every later message.

    Yields:
        str: Successive content deltas.
    """
    from backend.prompts import VISION_PROMPT

    messages = [{"role": "system", "content": VISION_PROMPT}]
    for msg in history or []:
        messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": question},
            {"type": "image_url",
             "image_url": {"url": f"data:{mime_type};base64,{image_base64}"}},
        ],
    })

    yield from stream(api_key, messages, model=resolve_model(api_key, "vision"))
