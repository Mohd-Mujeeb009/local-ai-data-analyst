"""
Tests for PDF retrieval.

Split by dependency: chunking and fusion are pure Python and always run, while
anything needing model weights is skipped when the optional stack is absent, so
the default CI install stays light.
"""

import pytest

from backend import rag
from backend.rag.chunker import _heading_level, chunk_document
from backend.rag.retriever import (
    bm25_search,
    format_citations,
    reciprocal_rank_fusion,
    tokenize,
)

needs_stack = pytest.mark.skipif(
    not rag.available(),
    reason="optional retrieval stack not installed (pip install -r requirements-rag.txt)",
)

REPORT = """QUARTERLY REPORT Q4 2025

1. Executive Summary

Group revenue reached 412 million dollars, up 6% year over year.

2. Segment Performance

2.1 Regional Results

EMEA
Revenue fell 12% year over year to 98 million dollars, driven by weaker
enterprise renewals. Headcount was reduced by 8%.

Americas
Revenue grew 21% to 214 million dollars, the strongest result in five years.

3. Risk Factors

3.1 Supply Chain

We depend on a limited number of contract manufacturers.
"""


class TestHeadingDetection:
    @pytest.mark.parametrize("line,level", [
        ("1. Financial Review", 1),
        ("2.1 Segment Performance", 2),
        ("3.2.1 Currency Effects", 3),
        ("# Markdown Heading", 1),
        ("### Deeper", 3),
    ])
    def test_numbering_states_its_own_level(self, line, level):
        assert _heading_level(line, 0)[0] == level

    @pytest.mark.parametrize("line", [
        "The company performed well.",
        "we reduced staffing by eight percent",
        "This line is far too long to be a heading and just keeps going on and on past any plausible limit",
        "",
        "Costs increased,",
    ])
    def test_prose_is_not_a_heading(self, line):
        assert _heading_level(line, 0) is None

    def test_unnumbered_heading_nests_below_its_section(self):
        """An ALL CAPS label inside a numbered section is a sub-heading."""
        assert _heading_level("EMEA", 2)[0] == 3

    def test_single_capitalised_word_is_a_heading(self):
        assert _heading_level("Americas", 1) == (2, "Americas")

    def test_short_lowercase_word_is_not(self):
        assert _heading_level("and", 1) is None


class TestChunking:
    def test_builds_nested_heading_paths(self):
        paths = {c["heading_path"] for c in chunk_document(REPORT, "d")}
        assert "Segment Performance > Regional Results > EMEA" in paths

    def test_siblings_do_not_nest_inside_each_other(self):
        """
        EMEA and Americas are peers. Nesting one inside the other would produce
        a citation that misdescribes the document.
        """
        paths = {c["heading_path"] for c in chunk_document(REPORT, "d")}
        assert "Segment Performance > Regional Results > Americas" in paths
        assert not any("EMEA > Americas" in p for p in paths)

    def test_sections_are_not_merged(self):
        """The two segments' figures must not land in one chunk."""
        chunks = chunk_document(REPORT, "d")
        shared = [c for c in chunks if "12%" in c["body"] and "21%" in c["body"]]
        assert not shared

    def test_embedded_text_carries_the_heading_path(self):
        """
        The whole point of the prefix: "fell 12%" is unretrievable alone.
        """
        chunks = chunk_document(REPORT, "d")
        emea = next(c for c in chunks if "12%" in c["body"])
        assert emea["text"].startswith("Segment Performance > Regional Results > EMEA:")
        assert "EMEA" not in emea["body"]  # body stays quotable

    def test_ids_are_unique_and_ordered(self):
        chunks = chunk_document(REPORT, "d")
        assert len({c["id"] for c in chunks}) == len(chunks)
        assert [c["order"] for c in chunks] == list(range(len(chunks)))

    def test_unstructured_text_still_chunks(self):
        chunks = chunk_document("A wall of text with no headings. " * 200, "p")
        assert chunks
        assert all(c["heading_path"] == "" for c in chunks)

    def test_long_sections_split_on_sentences(self):
        body = " ".join(f"Sentence {i} concerns revenue." for i in range(300))
        chunks = chunk_document("1. Section\n\n" + body, "long", max_chars=600)
        assert len(chunks) > 1
        assert all(len(c["body"]) <= 800 for c in chunks)
        assert all(c["body"].rstrip().endswith(".") for c in chunks)

    def test_empty_document_yields_nothing_usable(self):
        assert chunk_document("   \n\n  ", "e") == [] or all(
            not c["body"].strip() for c in chunk_document("   \n\n  ", "e")
        )


class TestLexicalSearch:
    def test_tokenize_drops_punctuation(self):
        assert tokenize("Revenue fell 12%, sharply!") == [
            "revenue", "fell", "12", "sharply"
        ]

    def test_finds_exact_rare_token(self):
        """
        BM25 exists in the pipeline precisely for literals that embeddings blur.
        """
        chunks = chunk_document(REPORT, "d")
        top = bm25_search(chunks, "EMEA renewals", 1)
        assert "EMEA" in top[0]["heading_path"]

    def test_returns_empty_for_no_chunks(self):
        assert bm25_search([], "anything", 5) == []


class TestFusion:
    def test_rewards_agreement_between_rankings(self):
        a = [{"id": "x"}, {"id": "y"}, {"id": "z"}]
        b = [{"id": "z"}, {"id": "x"}, {"id": "y"}]
        fused = reciprocal_rank_fusion([a, b])
        assert fused[0]["id"] == "x"  # ranked 1st and 2nd, best combined

    def test_includes_items_from_only_one_ranking(self):
        a = [{"id": "only_lexical"}]
        b = [{"id": "only_dense"}]
        ids = {c["id"] for c in reciprocal_rank_fusion([a, b])}
        assert ids == {"only_lexical", "only_dense"}

    def test_is_scale_free(self):
        """
        RRF reads rank position only. A scorer with huge magnitudes must not
        dominate one with small magnitudes - that is why it is used here.
        """
        loud = [{"id": "a", "score": 1e6}, {"id": "b", "score": 9e5}]
        quiet = [{"id": "b", "score": 0.02}, {"id": "a", "score": 0.01}]
        fused = reciprocal_rank_fusion([loud, quiet])
        assert abs(fused[0]["fusion_score"] - fused[1]["fusion_score"]) < 1e-9

    def test_empty_input(self):
        assert reciprocal_rank_fusion([]) == []


class TestCitations:
    def test_numbers_passages_for_the_model(self):
        text = format_citations([
            {"heading_path": "Risk Factors > Supply Chain", "body": "We depend on..."},
            {"heading_path": "", "body": "Orphan text"},
        ])
        assert "[1] Risk Factors > Supply Chain" in text
        assert "[2] (untitled section)" in text


class TestAvailability:
    def test_available_matches_importability(self):
        from importlib.util import find_spec

        expected = all(
            find_spec(m) for m in ("chromadb", "rank_bm25", "sentence_transformers")
        )
        assert rag.available() is bool(expected)


@pytest.fixture(scope="module")
def indexed():
    """Index the sample report once for the whole module."""
    from backend.rag import store

    doc_id, count = rag.ingest(REPORT)
    yield doc_id, count
    store.reset()


@needs_stack
class TestEndToEnd:
    """Exercises the real embedder, store and reranker."""

    def test_ingest_indexes_every_chunk(self, indexed):
        doc_id, count = indexed
        assert count == len(chunk_document(REPORT, doc_id))

    def test_reingest_is_idempotent(self, indexed):
        doc_id, count = indexed
        again, count_again = rag.ingest(REPORT)
        assert (again, count_again) == (doc_id, count)

    def test_retrieval_finds_the_right_section(self, indexed):
        doc_id, _ = indexed
        top = rag.retrieve(doc_id, "How did EMEA perform?", top_k=3)
        assert top
        assert "EMEA" in top[0]["heading_path"]

    def test_retrieval_respects_top_k(self, indexed):
        doc_id, _ = indexed
        assert len(rag.retrieve(doc_id, "revenue", top_k=2)) == 2

    def test_dense_retrieval_handles_paraphrase(self, indexed):
        """No lexical overlap with "Headcount was reduced by 8%"."""
        doc_id, _ = indexed
        top = rag.retrieve(doc_id, "did they cut staff?", top_k=3)
        assert any("EMEA" in c["heading_path"] for c in top)

    def test_survives_an_unavailable_reranker(self, indexed, monkeypatch):
        """
        The cross-encoder is a large separate download. If it cannot load -
        offline machine, stalled fetch - the query must still return the fused
        results rather than failing outright.
        """
        from backend.rag import retriever

        def refuse(*_args, **_kwargs):
            raise OSError("weights unavailable")

        monkeypatch.setattr(retriever, "rerank", refuse)

        doc_id, _ = indexed
        results = rag.retrieve(
            doc_id, "How did EMEA perform?", top_k=3, use_reranker=True
        )
        assert results
        assert all(c["reranked"] is False for c in results)
