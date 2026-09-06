"""
Local embeddings via sentence-transformers.

Runs `bge-small-en-v1.5` on the machine rather than through an API: document
text never leaves the process, and there is no per-token cost on re-indexing.

Embedding is the slow part of ingesting a PDF, and users re-upload the same
document constantly. The cache is keyed by a hash of the model name plus the
chunk text, so an unchanged chunk is never embedded twice - even across
restarts, and even if the surrounding document changed.
"""

import hashlib
import json
from pathlib import Path

MODEL_NAME = "BAAI/bge-small-en-v1.5"

# bge asks for this prefix on queries but not on documents. Omitting it costs a
# few points of retrieval quality, and it is the single easiest thing to get
# wrong with this model family.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_model = None


class EmbeddingUnavailable(RuntimeError):
    """Raised when sentence-transformers or its model cannot be loaded."""


def cache_dir():
    """Directory holding cached vectors."""
    return Path.home() / ".cache" / "ai-data-analyst" / "embeddings"


def _load_model():
    """
    Load the sentence-transformers model once per process.

    Raises:
        EmbeddingUnavailable: If the optional dependency is not installed.
    """
    global _model
    if _model is not None:
        return _model

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise EmbeddingUnavailable(
            "PDF retrieval needs the optional RAG dependencies. "
            "Install them with: pip install -r requirements-rag.txt"
        ) from exc

    _model = SentenceTransformer(MODEL_NAME)
    return _model


def _key(text):
    """Content hash identifying one cached vector."""
    digest = hashlib.sha256(f"{MODEL_NAME}\x00{text}".encode()).hexdigest()
    return digest[:40]


def _cache_path(key):
    # Shard by prefix so a directory listing stays usable with many documents.
    return cache_dir() / key[:2] / f"{key}.json"


def _read_cache(key):
    path = _cache_path(key)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        # A corrupt entry is a cache miss, never an error.
        return None


def _write_cache(key, vector):
    path = _cache_path(key)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(vector), encoding="utf-8")
    except OSError:
        pass  # an unwritable cache must not fail the query


def embed_documents(texts, use_cache=True):
    """
    Embed chunk texts, reusing cached vectors where possible.

    Args:
        texts: Chunk texts, already carrying their heading prefix.
        use_cache: Set False to bypass the cache entirely.

    Returns:
        list[list[float]]: One normalised vector per input, in input order.

    Raises:
        EmbeddingUnavailable: If the model cannot be loaded.
    """
    if not texts:
        return []

    vectors = [None] * len(texts)
    pending = []

    if use_cache:
        for index, text in enumerate(texts):
            cached = _read_cache(_key(text))
            if cached is None:
                pending.append(index)
            else:
                vectors[index] = cached
    else:
        pending = list(range(len(texts)))

    if pending:
        model = _load_model()
        fresh = model.encode(
            [texts[i] for i in pending],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        for index, vector in zip(pending, fresh):
            as_list = [float(v) for v in vector]
            vectors[index] = as_list
            if use_cache:
                _write_cache(_key(texts[index]), as_list)

    return vectors


def embed_query(question):
    """
    Embed a search query, with the prefix bge expects.

    Returns:
        list[float]: A normalised vector.
    """
    model = _load_model()
    vector = model.encode(
        QUERY_PREFIX + question,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return [float(v) for v in vector]


def available():
    """Report whether embeddings can be produced in this environment."""
    try:
        _load_model()
        return True
    except (EmbeddingUnavailable, Exception):  # noqa: B014 - model load can fail many ways
        return False
