"""
Measure PDF retrieval: truncation vs dense-only vs hybrid vs hybrid+rerank.

    python evals/run_pdf_eval.py
    python evals/run_pdf_eval.py --top-k 5 --write-readme

No API key and no LLM. Each question targets a section known by construction, so
Recall@k is measured directly - which isolates retrieval quality from whatever
the language model does with the passages afterwards.

The four configurations answer four separate questions:

  truncation   - what the app did before: feed the first 12k characters. The
                 measure is whether the target section falls inside that prefix.
  dense        - embeddings alone
  hybrid       - BM25 + dense fused with RRF
  hybrid+rerank- the full pipeline, cross-encoder over the fused candidates
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import MAX_PDF_CHARS  # noqa: E402
from backend.rag import chunker, embedder, retriever, store  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "evals" / "documents"
QUESTIONS = ROOT / "evals" / "pdf_questions.yaml"

README_START = "<!-- RETRIEVAL_RESULTS_START -->"
README_END = "<!-- RETRIEVAL_RESULTS_END -->"

CONFIGS = ["truncation", "dense", "hybrid", "hybrid+rerank"]


def load_questions():
    """Load the frozen retrieval question set."""
    with open(QUESTIONS, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_documents():
    """Load the corpus, keyed by document name."""
    return {p.stem: p.read_text(encoding="utf-8") for p in sorted(DOCS.glob("*.txt"))}


def index_all(documents):
    """
    Index every document, returning name -> doc_id.

    Embeddings are cached by content hash, so only the first run pays for this.
    """
    ids = {}
    for name, text in documents.items():
        doc_id = store.document_id(text)
        if not store.has_document(doc_id):
            chunks = chunker.chunk_document(text, doc_id)
            vectors = embedder.embed_documents([c["text"] for c in chunks])
            store.index_document(doc_id, chunks, vectors)
        ids[name] = doc_id
    return ids


def hit(passages, target_heading):
    """Report whether any retrieved passage came from the target section."""
    return any(
        target_heading.lower() in (p.get("heading_path") or "").lower()
        for p in passages
    )


def truncation_hit(text, question):
    """
    Whether truncation could reach the answer at all.

    The old behaviour fed the first MAX_PDF_CHARS characters, so a section
    beginning past that point was simply absent from the model's context. This
    is generous to truncation: it counts a hit whenever the target section
    starts inside the prefix, without asking whether the model then found it.
    """
    return question["target_offset"] < MAX_PDF_CHARS and len(text) >= 0


def retrieve_for(config, doc_id, question, top_k):
    """Run one configuration for one question."""
    if config == "dense":
        return store.search(doc_id, embedder.embed_query(question), top_k)
    if config == "hybrid":
        return retriever.retrieve(doc_id, question, top_k=top_k, use_reranker=False)
    return retriever.retrieve(doc_id, question, top_k=top_k, use_reranker=True)


def evaluate(top_k, verbose=True):
    """
    Run every configuration over every question.

    Returns:
        dict: config -> {"recall", "recall_beyond", "mean_latency", "n"}.
    """
    questions = load_questions()
    documents = load_documents()

    if verbose:
        print(f"indexing {len(documents)} documents...", flush=True)
    doc_ids = index_all(documents)

    results = {}
    for config in CONFIGS:
        hits, beyond_hits, beyond_total, latencies = [], [], 0, []

        if verbose:
            print(f"\n=== {config} ===", flush=True)

        for question in questions:
            text = documents[question["doc"]]
            started = time.time()

            if config == "truncation":
                found = truncation_hit(text, question)
            else:
                passages = retrieve_for(
                    config, doc_ids[question["doc"]], question["question"], top_k
                )
                found = hit(passages, question["target_heading"])

            latencies.append(time.time() - started)
            hits.append(found)

            if question["beyond_truncation"]:
                beyond_total += 1
                beyond_hits.append(found)

            if verbose and not found:
                print(f"  MISS {question['id']:34} {question['question'][:44]}",
                      flush=True)

        results[config] = {
            "recall": sum(hits) / len(hits),
            "recall_beyond": (sum(beyond_hits) / beyond_total) if beyond_total else None,
            "mean_latency": statistics.mean(latencies),
            "n": len(hits),
        }

        if verbose:
            r = results[config]
            print(f"  Recall@{top_k}: {r['recall']:.0%}  "
                  f"(beyond 12k: {r['recall_beyond']:.0%})  "
                  f"{r['mean_latency']*1000:.0f}ms/query", flush=True)

    return results


def render_markdown(results, top_k):
    """Render the comparison table."""
    labels = {
        "truncation": f"Truncation at {MAX_PDF_CHARS//1000}k (old)",
        "dense": "Dense only",
        "hybrid": "Hybrid (BM25 + dense + RRF) — current",
        "hybrid+rerank": "Hybrid + rerank (opt-in)",
    }

    lines = [
        f"| Configuration | Recall@{top_k} | Recall@{top_k} beyond 12k | Latency |",
        "|---|---|---|---|",
    ]
    for config in CONFIGS:
        r = results[config]
        beyond = "n/a" if r["recall_beyond"] is None else f"{r['recall_beyond']:.0%}"
        lines.append(
            f"| {labels[config]} | {r['recall']:.0%} | {beyond} | "
            f"{r['mean_latency']*1000:.0f}ms |"
        )
    return "\n".join(lines)


def write_readme(table, top_k, n):
    """Replace the marked retrieval block in the README."""
    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")

    if README_START not in text or README_END not in text:
        print("  markers not found in README", file=sys.stderr)
        return False

    block = (
        f"{README_START}\n{table}\n\n"
        f"<sub>{n} questions over 5 synthetic reports of 16-21k characters each, "
        f"in [`evals/pdf_questions.yaml`](evals/pdf_questions.yaml). Each question "
        f"targets a known section, so recall is measured directly with no model in "
        f"the loop. Measured {time.strftime('%Y-%m-%d')}. Reproduce with "
        f"`python evals/run_pdf_eval.py --write-readme`.</sub>\n{README_END}"
    )

    start = text.index(README_START)
    end = text.index(README_END) + len(README_END)
    readme.write_text(text[:start] + block + text[end:], encoding="utf-8")
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--write-readme", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    results = evaluate(args.top_k, verbose=not args.quiet)
    table = render_markdown(results, args.top_k)
    print("\n" + table + "\n")

    if args.write_readme and write_readme(table, args.top_k, results["dense"]["n"]):
        print("updated README.md")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
