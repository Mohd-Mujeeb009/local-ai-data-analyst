"""Tests for chart spec validation.

Only `validate_spec` is covered here - it is the pure decision layer. The
`render` functions are thin Streamlit calls with no logic worth mocking.
"""

import pandas as pd

from frontend.charts import validate_spec


def frame():
    return pd.DataFrame({
        "product": ["A", "B", "C"],
        "revenue": [100, 200, 300],
        "units": [1, 2, 3],
    })


def test_accepts_valid_spec():
    spec = validate_spec(
        {"type": "bar", "x": "product", "y": ["revenue"], "title": "Rev"}, frame()
    )
    assert spec["type"] == "bar"
    assert spec["x"] == "product"
    assert spec["y"] == ["revenue"]


def test_drops_columns_absent_from_result():
    """A plan can name a column the generated code never produced."""
    spec = validate_spec({"type": "bar", "x": "product", "y": ["nope"]}, frame())
    assert spec["y"] == ["revenue", "units"]  # falls back to numeric columns


def test_clears_x_absent_from_result():
    spec = validate_spec({"type": "bar", "x": "ghost", "y": ["revenue"]}, frame())
    assert spec["x"] is None


def test_rejects_single_row():
    """One row is a number, not a chart."""
    assert validate_spec({"type": "bar", "y": ["revenue"]}, frame().head(1)) is None


def test_rejects_scalar_result():
    assert validate_spec({"type": "bar", "y": ["a"]}, 42) is None


def test_rejects_missing_spec():
    assert validate_spec(None, frame()) is None


def test_rejects_frame_without_numeric_columns():
    df = pd.DataFrame({"a": ["x", "y", "z"], "b": ["p", "q", "r"]})
    assert validate_spec({"type": "bar", "y": []}, df) is None


def test_accepts_series():
    series = pd.Series([1, 2, 3], index=["a", "b", "c"], name="revenue")
    spec = validate_spec({"type": "line", "y": ["revenue"]}, series)
    assert spec["type"] == "line"
    assert spec["y"] == ["revenue"]


def test_recognises_index_name_as_x():
    df = frame().set_index("product")
    spec = validate_spec({"type": "bar", "x": "product", "y": ["revenue"]}, df)
    assert spec["x"] == "product"
