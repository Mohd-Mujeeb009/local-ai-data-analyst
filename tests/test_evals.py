"""
Tests for the evaluation harness.

Grading is the part of the harness that has to be trustworthy: a lenient
normaliser would flatter whichever system it favours, and a strict one would
mark correct answers wrong for shape. These cover both directions.
"""

import pandas as pd
import pytest
import yaml

from evals.baseline import parse_answer, summarize_dataframe
from evals.run_eval import QUESTIONS, cost_of, grade, load_questions, normalise, summarise


class TestQuestionSet:
    def test_loads_fifty_questions(self):
        assert len(load_questions()) == 50

    def test_has_five_unanswerable(self):
        questions = load_questions()
        assert sum(1 for q in questions if q["kind"] == "unanswerable") == 5

    def test_covers_every_required_category(self):
        categories = {q["category"] for q in load_questions()}
        assert categories >= {
            "simple_aggregate", "group_by", "filter_aggregate",
            "top_n", "date_range", "multi_column", "unanswerable",
        }

    def test_ids_are_unique(self):
        ids = [q["id"] for q in load_questions()]
        assert len(ids) == len(set(ids))

    def test_every_question_is_well_formed(self):
        for question in load_questions():
            assert question["question"].strip()
            assert question["kind"] in {"numeric", "string", "list", "unanswerable"}
            assert question["answer"] is not None

    def test_ground_truth_matches_the_dataset(self):
        """
        The frozen answers must still describe the shipped file. If the dataset
        is ever regenerated without rebuilding the YAML, this fails loudly
        instead of silently grading against stale numbers.
        """
        df = pd.read_csv("examples/sales_2025.csv")
        answers = {q["id"]: q["answer"] for q in load_questions()}

        assert answers["q01"] == pytest.approx(df.revenue.sum(), rel=1e-6)
        assert answers["q02"] == len(df)
        assert answers["q05"] == df.units.sum()
        assert answers["q09"] == df.groupby("region").revenue.sum().idxmax()
        assert answers["q28"] == df.groupby("product").revenue.sum().idxmax()

    def test_yaml_is_parseable_without_the_loader(self):
        with open(QUESTIONS, encoding="utf-8") as handle:
            assert isinstance(yaml.safe_load(handle), list)


class TestNormalise:
    """Both systems must get credit for a right answer in any reasonable shape."""

    @pytest.mark.parametrize("value", [
        4156518.25,
        pd.Series([4156518.25]),
        pd.DataFrame({"revenue": [4156518.25]}),
    ])
    def test_numeric_from_any_shape(self, value):
        assert normalise(value, "numeric") == pytest.approx(4156518.25)

    def test_numeric_rejects_multi_row(self):
        assert normalise(pd.Series([1.0, 2.0]), "numeric") is None

    def test_numeric_rejects_text(self):
        assert normalise("North", "numeric") is None

    def test_label_from_idxmax_string(self):
        assert normalise("North", "string") == "North"

    def test_label_from_nlargest_series(self):
        """nlargest(1) puts the answer in the index, not the values."""
        series = pd.Series([1279888.51], index=["North"])
        assert normalise(series, "string") == "North"

    def test_label_from_frame_text_column(self):
        frame = pd.DataFrame({"region": ["North"], "revenue": [1.0]})
        assert normalise(frame, "string") == "North"

    def test_list_from_series_index(self):
        series = pd.Series([3, 2, 1], index=["a", "b", "c"])
        assert normalise(series, "list") == ["a", "b", "c"]

    def test_list_from_frame_column(self):
        frame = pd.DataFrame({"product": ["a", "b"], "revenue": [2, 1]})
        assert normalise(frame, "list") == ["a", "b"]

    def test_list_from_plain_list(self):
        assert normalise(["a", "b"], "list") == ["a", "b"]

    def test_unanswerable_detected_in_string(self):
        assert normalise("UNANSWERABLE", "unanswerable") == "UNANSWERABLE"

    def test_unanswerable_not_claimed_when_a_number_was_returned(self):
        """Answering a question the data cannot support is the failure mode."""
        assert normalise(42.0, "unanswerable") != "UNANSWERABLE"

    def test_none_stays_none(self):
        assert normalise(None, "numeric") is None

    def test_empty_frame_is_none(self):
        assert normalise(pd.DataFrame(), "numeric") is None


class TestGrade:
    def test_numeric_within_tolerance(self):
        assert grade(100.0, 100.5, "numeric", 0.01)

    def test_numeric_outside_tolerance(self):
        assert not grade(100.0, 110.0, "numeric", 0.01)

    def test_numeric_absolute_floor_for_small_values(self):
        """Relative tolerance alone would be unusably tight near zero."""
        assert grade(0.0, 0.005, "numeric", 0.01)

    def test_string_case_insensitive(self):
        assert grade("North", "north  ", "string")

    def test_string_mismatch(self):
        assert not grade("North", "South", "string")

    def test_list_order_matters(self):
        assert grade(["a", "b"], ["a", "b"], "list")
        assert not grade(["a", "b"], ["b", "a"], "list")

    def test_list_length_matters(self):
        assert not grade(["a", "b", "c"], ["a", "b"], "list")

    def test_unanswerable_correct(self):
        assert grade("UNANSWERABLE", "UNANSWERABLE", "unanswerable")

    def test_unanswerable_wrong_when_answered(self):
        assert not grade("UNANSWERABLE", "42.0", "unanswerable")

    def test_none_is_always_wrong(self):
        assert not grade(1.0, None, "numeric")


class TestCosting:
    def test_prices_known_model(self):
        usage = [{"model": "openai/gpt-oss-120b",
                  "prompt_tokens": 1_000_000, "completion_tokens": 0}]
        assert cost_of(usage) == pytest.approx(0.15)

    def test_sums_across_calls(self):
        usage = [
            {"model": "openai/gpt-oss-120b", "prompt_tokens": 1_000_000,
             "completion_tokens": 0},
            {"model": "openai/gpt-oss-120b", "prompt_tokens": 0,
             "completion_tokens": 1_000_000},
        ]
        assert cost_of(usage) == pytest.approx(0.90)

    def test_unknown_model_reports_unknown_not_zero(self):
        """A missing price must not silently read as a free query."""
        assert cost_of([{"model": "mystery", "prompt_tokens": 100,
                         "completion_tokens": 100}]) is None


class TestSummarise:
    def make_rows(self):
        return [
            {"kind": "numeric", "category": "simple_aggregate", "correct": True,
             "latency": 1.0, "cost": 0.001, "error": None},
            {"kind": "numeric", "category": "simple_aggregate", "correct": False,
             "latency": 2.0, "cost": 0.002, "error": None},
            {"kind": "unanswerable", "category": "unanswerable", "correct": True,
             "latency": 3.0, "cost": 0.003, "error": None},
        ]

    def test_accuracy_excludes_unanswerable(self):
        assert summarise(self.make_rows())["accuracy"] == pytest.approx(0.5)

    def test_overall_accuracy_includes_everything(self):
        assert summarise(self.make_rows())["overall_accuracy"] == pytest.approx(2 / 3)

    def test_unanswerable_rate_tracked_separately(self):
        assert summarise(self.make_rows())["unanswerable_detected"] == 1.0

    def test_means(self):
        summary = summarise(self.make_rows())
        assert summary["mean_latency"] == pytest.approx(2.0)
        assert summary["mean_cost"] == pytest.approx(0.002)

    def test_handles_all_unknown_costs(self):
        rows = [{"kind": "numeric", "category": "c", "correct": True,
                 "latency": 1.0, "cost": None, "error": None}]
        assert summarise(rows)["mean_cost"] is None


class TestBaseline:
    def test_summary_contains_only_a_sample(self):
        """
        The baseline's whole limitation is that it sees five rows. If this ever
        started including the full frame the comparison would be meaningless.
        """
        df = pd.read_csv("examples/sales_2025.csv")
        summary = summarize_dataframe(df)
        assert len(summary) < 6000
        assert summary.count("\n") < len(df)

    @pytest.mark.parametrize("raw,expected", [
        ('{"answer": 42}', 42),
        ('```json\n{"answer": "North"}\n```', "North"),
        ('{"answer": ["a", "b"]}', ["a", "b"]),
        ('{"answer": "UNANSWERABLE"}', "UNANSWERABLE"),
    ])
    def test_parses_answers(self, raw, expected):
        assert parse_answer(raw) == expected

    @pytest.mark.parametrize("raw", ["not json", "[1,2]", '{"result": 1}'])
    def test_rejects_unusable_responses(self, raw):
        assert parse_answer(raw) is None
