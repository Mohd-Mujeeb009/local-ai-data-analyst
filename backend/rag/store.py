"""
Persistent vector store backed by ChromaDB.

One collection per document, keyed by a content hash of the file. Re-uploading
the same PDF therefore hits an existing collection and skips embedding entirely,
which is the common case - people upload the same report repeatedly.

Chunk bodies are stored alongside the vectors so retrieval returns quotable text
and a heading path for citation, without needing the original file again.
"""

import hashlib
import shutil
from pathlib import Path

_client = None


class StoreUnavailable(RuntimeError):
    """Raised when ChromaDB is not installed or cannot open its directory."""


def store_dir():
    """Directory holding the persistent index."""
    return Path.home() / ".cache" / "ai-data-analyst" / "chroma"


def document_id(data):
    """
    Content hash identifying a document.

    Args:
        data: The raw file bytes, or the extracted text.

    Returns:
        str: A short stable identifier, usable as a collection name.
    """
    if isinstance(data, str):
        data = data.encode("utf-8")
    return "doc_" + hashlib.sha256(data).hexdigest()[:24]


def _get_client():
    """
    Open the persistent Chroma client once per process.

    Raises:
        StoreUnavailable: If the optional dependency is missing.
    """
    global _client
    if _client is not None:
        return _client

    try:
        import chromadb
        from chromadb.config import Settings
    except ImportError as exc:
        raise StoreUnavailable(
            "PDF retrieval needs the optional RAG dependencies. "
            "Install them with: pip install -r requirements-rag.txt"
        ) from exc

    store_dir().mkdir(parents=True, exist_ok=True)
    _client = chromadb.PersistentClient(
        path=str(store_dir()),
        settings=Settings(anonymized_telemetry=False, allow_reset=True),
    )
    return _client


def has_document(doc_id):
    """Report whether this document is already indexed."""
    try:
        client = _get_client()
        return doc_id in {c.name for c in client.list_collections()}
    except StoreUnavailable:
        return False


def index_document(doc_id, chunks, vectors):
    """
    Store a document's chunks and vectors.

    Args:
        doc_id: The document's content hash.
        chunks: Chunk dicts from `chunker.chunk_document`.
        vectors: One vector per chunk, in the same order.

    Returns:
        int: The number of chunks stored.
    """
    client = _get_client()

    # Replace rather than append: a re-index of the same id must not duplicate.
    if has_document(doc_id):
        client.delete_collection(doc_id)

    collection = client.create_collection(
        name=doc_id, metadata={"hnsw:space": "cosine"}
    )
    collection.add(
        ids=[c["id"] for c in chunks],
        embeddings=vectors,
        documents=[c["body"] for c in chunks],
        metadatas=[
            {"heading_path": c["heading_path"], "order": c["order"], "doc_id": doc_id}
            for c in chunks
        ],
    )
    return len(chunks)


def all_chunks(doc_id):
    """
    Return every stored chunk for a document, in document order.

    Needed by the lexical half of hybrid retrieval, which scores the full
    corpus rather than querying an index.

    Returns:
        list[dict]: Each with "id", "body", "heading_path", "order".
    """
    collection = _get_client().get_collection(doc_id)
    stored = collection.get(include=["documents", "metadatas"])

    chunks = [
        {
            "id": cid,
            "body": body,
            "heading_path": (meta or {}).get("heading_path", ""),
            "order": (meta or {}).get("order", 0),
        }
        # Chroma returns these three lists in lockstep.
        for cid, body, meta in zip(
            stored["ids"], stored["documents"], stored["metadatas"], strict=True
        )
    ]
    return sorted(chunks, key=lambda c: c["order"])


def search(doc_id, query_vector, top_k):
    """
    Dense nearest-neighbour search within one document.

    Returns:
        list[dict]: Each with "id", "body", "heading_path", "score" (higher is
            better; Chroma returns cosine distance, which is inverted here).
    """
    collection = _get_client().get_collection(doc_id)
    found = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    results = []
    for cid, body, meta, distance in zip(
        found["ids"][0], found["documents"][0],
        found["metadatas"][0], found["distances"][0], strict=True,
    ):
        results.append({
            "id": cid,
            "body": body,
            "heading_path": (meta or {}).get("heading_path", ""),
            "score": 1.0 - float(distance),
        })
    return results


def reset():
    """Delete the entire index. Used by tests and by a manual cache clear."""
    global _client
    _client = None
    if store_dir().exists():
        shutil.rmtree(store_dir(), ignore_errors=True)
