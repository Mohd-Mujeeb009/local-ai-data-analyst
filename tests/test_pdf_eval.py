"""
Tests for the retrieval evaluation harness.

The corpus and question set have to stay internally consistent: if a question
points at a section the generator no longer emits, recall silently measures the
wrong thing.
"""

import yaml

from backend.config import MAX_PDF_CHARS
from evals.run_pdf_eval import (
    QUESTIONS,
    hit,
    load_documents,
    load_questions,
    render_markdown,
    truncation_hit,
)


class TestQuestionSet:
    def test_has_fifty_questions(self):
        assert len(load_questions()) == 50

    def test_ids_are_unique(self):
        ids = [q["id"] for q in load_questions()]
        assert len(ids) == len(set(ids))

    def test_covers_five_documents(self):
        assert len({q["doc"] for q in load_questions()}) == 5

    def test_every_question_names_a_real_document(self):
        documents = load_documents()
        for question in load_questions():
            assert question["doc"] in documents

    def test_target_heading_appears_in_its_document(self):
        """A question pointing at a heading that no longer exists would make
        recall measure nothing."""
        documents = load_documents()
        for question in load_questions():
            assert question["target_heading"] in documents[question["doc"]]

    def test_offsets_point_at_real_positions(self):
        documents = load_documents()
        for question in load_questions():
            assert 0 <= question["target_offset"] < len(documents[question["doc"]])

    def test_some_questions_lie_beyond_truncation(self):
        """
        The benchmark only demonstrates anything if part of it is unreachable
        by the old approach.
        """
        beyond = [q for q in load_questions() if q["beyond_truncation"]]
        assert len(beyond) >= 10

    def test_beyond_flag_matches_the_offset(self):
        for question in load_questions():
            expected = question["target_offset"] > MAX_PDF_CHARS
            assert question["beyond_truncation"] is expected

    def test_yaml_parses_standalone(self):
        with open(QUESTIONS, encoding="utf-8") as handle:
            assert isinstance(yaml.safe_load(handle), list)


class TestCorpus:
    def test_documents_exceed_the_truncation_limit(self):
        """Each document must be long enough for truncation to lose something."""
        for name, text in load_documents().items():
            assert len(text) > MAX_PDF_CHARS, name


class TestScoring:
    def test_hit_matches_target_section(self):
        passages = [{"heading_path": "Risk Factors > Supply Chain Concentration"}]
        assert hit(passages, "Supply Chain Concentration")

    def test_hit_is_case_insensitive(self):
        assert hit([{"heading_path": "RISK FACTORS > FUEL"}], "Fuel")

    def test_miss_when_no_passage_matches(self):
        assert not hit([{"heading_path": "Outlook"}], "Supply Chain")

    def test_miss_on_empty_results(self):
        assert not hit([], "Anything")

    def test_handles_passage_without_heading(self):
        assert not hit([{"heading_path": ""}], "Outlook")

    def test_truncation_reaches_early_sections(self):
        assert truncation_hit("x" * 20000, {"target_offset": 500})

    def test_truncation_cannot_reach_late_sections(self):
        """The core claim: past the cut, the text was never in context."""
        assert not truncation_hit("x" * 20000, {"target_offset": MAX_PDF_CHARS + 1})


class TestReporting:
    def test_renders_every_configuration(self):
        results = {
            c: {"recall": 0.8, "recall_beyond": 0.7, "mean_latency": 0.05, "n": 50}
            for c in ["truncation", "dense", "hybrid", "hybrid+rerank"]
        }
        table = render_markdown(results, 1)
        assert "Recall@1" in table
        for label in ["Truncation", "Dense only", "Hybrid", "rerank"]:
            assert label in table

    def test_reports_missing_beyond_recall_as_na(self):
        results = {
            c: {"recall": 1.0, "recall_beyond": None, "mean_latency": 0.0, "n": 50}
            for c in ["truncation", "dense", "hybrid", "hybrid+rerank"]
        }
        assert "n/a" in render_markdown(results, 5)


class TestRerankerDefault:
    def test_reranking_is_off_by_default(self):
        """
        The benchmark showed 82% vs 84% Recall@1 for ~200x the latency, so the
        default follows the measurement. Flipping this back on should be a
        deliberate act with a fresh measurement behind it.
        """
        from backend.config import RERANK_BY_DEFAULT

        assert RERANK_BY_DEFAULT is False

    def test_retrieve_defers_to_the_config_flag(self):
        import inspect

        from backend.rag import retriever

        default = inspect.signature(retriever.retrieve).parameters["use_reranker"].default
        assert default is None  # resolved from config at call time
