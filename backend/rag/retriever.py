"""
Hybrid retrieval: BM25 + dense, fused with RRF, then cross-encoder reranked.

Each stage exists because the one before it has a specific blind spot.

**BM25** catches exact tokens that embeddings blur - product codes, defined
terms, "Section 4.2", a surname. Dense retrieval reliably misses these because
a rare literal token carries little semantic signal.

**Dense** catches paraphrase, which BM25 cannot do at all: a question about
"headcount reductions" will not lexically match "we reduced staffing by 8%".

**Reciprocal rank fusion** combines them without needing their scores to be
comparable. BM25 returns unbounded term-frequency sums and cosine similarity
returns [-1, 1]; normalising those onto a shared scale requires tuning that
does not transfer between corpora. RRF only reads rank positions, so it needs
no tuning and cannot be dominated by whichever scorer happens to be louder.

**Cross-encoder reranking** is available but off by default. Bi-encoders embed
the query and the chunk separately and never compare them directly; a
cross-encoder reads both together and can tell that a chunk mentioning the right
entity answers the wrong question. That is the theory, and it is why the stage
exists. The measurement did not bear it out on the benchmark corpus - 82% vs 84%
Recall@1 for roughly 200x the latency - so the default follows the evidence
rather than the theory. See config.RERANK_BY_DEFAULT for the numbers and the
caveat about corpus size.
"""

import re

from backend.config import RERANK_BY_DEFAULT, RETRIEVAL_TOP_K

RERANKER_NAME = "BAAI/bge-reranker-base"

FUSION_K = 60          # RRF damping; 60 is the value from the original paper
CANDIDATES = 50        # how many the cheap stages hand to the reranker
DEFAULT_TOP_K = RETRIEVAL_TOP_K

_reranker = None
TOKEN = re.compile(r"[a-z0-9]+")


class RetrievalUnavailable(RuntimeError):
    """Raised when an optional retrieval dependency is missing."""


def tokenize(text):
    """Lowercase alphanumeric tokens, for lexical scoring."""
    return TOKEN.findall(text.lower())


def bm25_search(chunks, question, top_k):
    """
    Lexical search over a document's chunks.

    Args:
        chunks: Chunk dicts with "id" and "body".
        question: The user's question.
        top_k: How many to return.

    Returns:
        list[dict]: Chunks with a "score", best first.

    Raises:
        RetrievalUnavailable: If rank_bm25 is not installed.
    """
    try:
        from rank_bm25 import BM25Okapi
    except ImportError as exc:
        raise RetrievalUnavailable(
            "PDF retrieval needs the optional RAG dependencies. "
            "Install them with: pip install -r requirements-rag.txt"
        ) from exc

    if not chunks:
        return []

    # Score against the heading-prefixed form, so a section title is matchable
    # even when the body never repeats it.
    corpus = [
        tokenize(f"{c.get('heading_path', '')} {c['body']}") for c in chunks
    ]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(tokenize(question))

    # BM25 returns exactly one score per document.
    ranked = sorted(
        zip(chunks, scores, strict=True), key=lambda pair: pair[1], reverse=True
    )
    return [{**chunk, "score": float(score)} for chunk, score in ranked[:top_k]]


def reciprocal_rank_fusion(rankings, k=FUSION_K):
    """
    Fuse several ranked lists by rank position alone.

    Args:
        rankings: An iterable of ranked lists of chunk dicts.
        k: Damping constant. Larger values flatten the contribution of top ranks.

    Returns:
        list[dict]: Fused list, best first, each carrying "fusion_score".
    """
    scores, by_id = {}, {}

    for ranking in rankings:
        for position, chunk in enumerate(ranking):
            cid = chunk["id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + position + 1)
            by_id.setdefault(cid, chunk)

    ordered = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
    return [{**by_id[cid], "fusion_score": score} for cid, score in ordered]


def _load_reranker():
    """Load the cross-encoder once per process."""
    global _reranker
    if _reranker is not None:
        return _reranker

    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise RetrievalUnavailable(
            "PDF retrieval needs the optional RAG dependencies. "
            "Install them with: pip install -r requirements-rag.txt"
        ) from exc

    _reranker = CrossEncoder(RERANKER_NAME)
    return _reranker


def rerank(question, candidates, top_k):
    """
    Reorder candidates with a cross-encoder and keep the best.

    Args:
        question: The user's question.
        candidates: Chunk dicts from fusion.
        top_k: How many to keep.

    Returns:
        list[dict]: Reranked chunks with "rerank_score", best first.
    """
    if not candidates:
        return []

    model = _load_reranker()
    pairs = [
        (question, f"{c.get('heading_path', '')}: {c['body']}".strip(": "))
        for c in candidates
    ]
    scores = model.predict(pairs, show_progress_bar=False)

    scored = [
        {**chunk, "rerank_score": float(score)}
        for chunk, score in zip(candidates, scores, strict=True)
    ]
    scored.sort(key=lambda c: c["rerank_score"], reverse=True)
    return scored[:top_k]


def retrieve(doc_id, question, top_k=DEFAULT_TOP_K, candidates=CANDIDATES,
             use_reranker=None):
    """
    Run the full hybrid pipeline for one question.

    Args:
        doc_id: The indexed document's id.
        question: The user's question.
        top_k: How many chunks to return.
        candidates: How many the cheap stages hand to the reranker.
        use_reranker: Whether to run the cross-encoder. Defaults to
            config.RERANK_BY_DEFAULT, which is off - see the measurement
            recorded there. The evaluation harness sets it explicitly to
            isolate each stage's contribution.

    Returns:
        list[dict]: The chosen chunks, best first, each with "id",
            "heading_path", "body" and its stage scores.
    """
    from backend.rag import embedder, store

    if use_reranker is None:
        use_reranker = RERANK_BY_DEFAULT

    chunks = store.all_chunks(doc_id)
    if not chunks:
        return []

    lexical = bm25_search(chunks, question, candidates)
    dense = store.search(doc_id, embedder.embed_query(question), candidates)

    fused = reciprocal_rank_fusion([lexical, dense])[:candidates]

    if not use_reranker:
        return fused[:top_k]

    try:
        return rerank(question, fused, top_k)
    except Exception:
        # The reranker improves ordering; it is not required to produce one.
        # Its weights are a large separate download, so an offline machine or a
        # stalled fetch would otherwise fail the whole query rather than return
        # the fused results - which are already good. Degrade, do not break.
        return [{**chunk, "reranked": False} for chunk in fused[:top_k]]


def format_citations(chunks):
    """
    Render retrieved chunks as context for the model.

    Each is labelled with its id so the model can cite it and the UI can match
    the citation back to a snippet.

    Returns:
        str: The context block.
    """
    parts = []
    for index, chunk in enumerate(chunks, 1):
        heading = chunk.get("heading_path") or "(untitled section)"
        parts.append(f"[{index}] {heading}\n{chunk['body']}")
    return "\n\n".join(parts)
