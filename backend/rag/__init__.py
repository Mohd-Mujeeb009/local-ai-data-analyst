"""
Retrieval-augmented answering for PDF documents.

    ingest(text, raw_bytes)      -> doc_id, indexing the document if new
    retrieve(doc_id, question)   -> the chunks that should answer it

Deliberately scoped to documents. Spreadsheets go through the analysis pipeline
instead - see `available()` and the README section on why.
"""

from backend.rag.chunker import chunk_document
from backend.rag.embedder import EmbeddingUnavailable
from backend.rag.retriever import RetrievalUnavailable, format_citations, retrieve
from backend.rag.store import StoreUnavailable, document_id

__all__ = [
    "EmbeddingUnavailable",
    "RagUnavailable",
    "RetrievalUnavailable",
    "StoreUnavailable",
    "available",
    "chunk_document",
    "document_id",
    "format_citations",
    "ingest",
    "retrieve",
]

# One exception type for callers that only need to know "retrieval is off".
RagUnavailable = (EmbeddingUnavailable, StoreUnavailable, RetrievalUnavailable)


def available():
    """
    Report whether the optional retrieval stack is installed.

    The app degrades to truncated-context PDF answering when this is False, so
    the base install stays light and CI does not need to download model weights.
    """
    try:
        import chromadb  # noqa: F401
        import rank_bm25  # noqa: F401
        import sentence_transformers  # noqa: F401
    except ImportError:
        return False
    return True


def ingest(text, key_bytes=None, progress=None):
    """
    Chunk, embed and index a document, skipping work already done.

    Args:
        text: The extracted document text.
        key_bytes: Raw file bytes, used for the content hash. Falls back to the
            text, which is equivalent for identifying a re-upload.
        progress: Optional callable taking a status string, for UI feedback.

    Returns:
        tuple[str, int]: The document id and its chunk count.

    Raises:
        EmbeddingUnavailable, StoreUnavailable: If the stack is not installed.
    """
    from backend.rag import embedder, store

    doc_id = document_id(key_bytes if key_bytes is not None else text)

    if store.has_document(doc_id):
        # Re-upload of a document already indexed: nothing to recompute.
        if progress:
            progress("Using existing index")
        return doc_id, len(store.all_chunks(doc_id))

    if progress:
        progress("Splitting document")
    chunks = chunk_document(text, doc_id)

    if progress:
        progress(f"Embedding {len(chunks)} sections")
    vectors = embedder.embed_documents([c["text"] for c in chunks])

    if progress:
        progress("Indexing")
    store.index_document(doc_id, chunks, vectors)

    return doc_id, len(chunks)
