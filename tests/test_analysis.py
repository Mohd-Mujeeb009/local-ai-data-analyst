"""Tests for plan parsing and result formatting."""

import pandas as pd
import pytest

from backend.analysis import AnalysisError, format_result, parse_plan
from backend.config import MAX_RESULT_CHARS


class TestParsePlan:
    def test_parses_valid_plan(self):
        plan = parse_plan(
            '{"code": "result = df.head()", '
            '"chart": {"type": "bar", "x": "a", "y": ["b"], "title": "T"}, '
            '"explanation": "takes the head"}'
        )
        assert plan["code"] == "result = df.head()"
        assert plan["chart"]["type"] == "bar"
        assert plan["explanation"] == "takes the head"

    def test_strips_markdown_fences(self):
        raw = '```json\n{"code": "result = 1", "chart": null}\n```'
        assert parse_plan(raw)["code"] == "result = 1"

    def test_strips_fences_from_code_field(self):
        raw = '{"code": "```python\\nresult = 1\\n```", "chart": null}'
        assert parse_plan(raw)["code"] == "result = 1"

    def test_chart_none_becomes_null(self):
        plan = parse_plan('{"code": "result = 1", "chart": {"type": "none"}}')
        assert plan["chart"] is None

    def test_invalid_chart_type_dropped(self):
        plan = parse_plan('{"code": "result = 1", "chart": {"type": "pie"}}')
        assert plan["chart"] is None

    def test_rejects_non_json(self):
        with pytest.raises(AnalysisError, match="Could not read"):
            parse_plan("here is your analysis")

    def test_rejects_missing_code(self):
        with pytest.raises(AnalysisError, match="no code"):
            parse_plan('{"chart": null}')

    def test_rejects_non_object(self):
        with pytest.raises(AnalysisError, match="not a JSON object"):
            parse_plan("[1, 2, 3]")


class TestFormatResult:
    def test_formats_scalar(self):
        assert format_result(42) == "42"

    def test_formats_frame(self):
        text = format_result(pd.DataFrame({"a": [1, 2]}))
        assert "a" in text and "1" in text

    def test_formats_series(self):
        assert "north" in format_result(pd.Series({"north": 1, "south": 2}))

    def test_caps_output_size(self):
        text = format_result(pd.DataFrame({"a": range(100_000)}))
        assert len(text) <= MAX_RESULT_CHARS + 40

    def test_declares_row_elision(self):
        """pandas elides long frames silently; the explainer must be told."""
        text = format_result(pd.DataFrame({"a": range(500)}))
        assert "showing 50 of 500 rows" in text

    def test_no_elision_notice_for_short_frames(self):
        assert "showing" not in format_result(pd.DataFrame({"a": [1, 2, 3]}))
