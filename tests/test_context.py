"""Tests for context builders."""

import pandas as pd

from backend.config import MAX_PDF_CHARS, MAX_SCHEMA_COLUMNS
from backend.context import describe_schema, summarize_pdf


class TestDescribeSchema:
    def test_reports_shape_and_columns(self):
        df = pd.DataFrame({"product": ["A", "B"], "revenue": [1, 2]})
        text = describe_schema(df)
        assert "2 rows x 2 columns" in text
        assert "product" in text
        assert "revenue" in text

    def test_reports_numeric_range(self):
        text = describe_schema(pd.DataFrame({"n": [5, 10, 15]}))
        assert "range 5 to 15" in text

    def test_lists_low_cardinality_values(self):
        text = describe_schema(pd.DataFrame({"region": ["n", "s", "n"]}))
        assert "2 distinct" in text
        assert "values:" in text

    def test_samples_high_cardinality_values(self):
        df = pd.DataFrame({"id": [f"id-{i}" for i in range(50)]})
        text = describe_schema(df)
        assert "50 distinct" in text
        assert "e.g." in text

    def test_counts_nulls(self):
        text = describe_schema(pd.DataFrame({"a": [1, None, 3]}))
        assert "1 null" in text

    def test_truncates_wide_frames(self):
        wide = pd.DataFrame({f"c{i}": [1] for i in range(MAX_SCHEMA_COLUMNS + 20)})
        text = describe_schema(wide)
        assert "more columns omitted" in text

    def test_survives_mixed_type_column(self):
        df = pd.DataFrame({"mixed": [1, "two", None, {"a": 1}]})
        assert "mixed" in describe_schema(df)


class TestSummarizePdf:
    def test_marks_short_text_complete(self):
        text = summarize_pdf("a short document")
        assert "complete" in text
        assert "a short document" in text

    def test_declares_truncation(self):
        text = summarize_pdf("x" * (MAX_PDF_CHARS + 5000))
        assert "TRUNCATED" in text
        assert "not provided" in text
