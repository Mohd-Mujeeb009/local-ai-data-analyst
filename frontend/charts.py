"""
Chart rendering driven by the analysis plan.

The model names the chart type and the columns as part of its plan, so what gets
drawn reflects the question that was asked. Every spec is re-validated against
the actual result frame here - a plan can name a column the code did not produce.
"""

import streamlit as st


def validate_spec(spec, result):
    """
    Check a chart spec against the frame it will be drawn from.

    Args:
        spec: The plan's chart dict, or None.
        result: The executed result.

    Returns:
        dict or None: A normalised spec with an "index" key naming the column to
            plot against, or None when the result is not chartable.
    """
    import pandas as pd

    if not spec or not isinstance(result, (pd.DataFrame, pd.Series)):
        return None

    frame = result.to_frame() if isinstance(result, pd.Series) else result
    if len(frame) < 2:
        return None  # a single row is a number, not a chart

    columns = set(frame.columns)
    if frame.index.name:
        columns.add(frame.index.name)

    y_cols = [c for c in (spec.get("y") or []) if c in frame.columns]
    if not y_cols:
        y_cols = list(frame.select_dtypes("number").columns)
    if not y_cols:
        return None

    x = spec.get("x")
    if x not in columns:
        x = None  # fall back to the frame's own index

    return {
        "type": spec.get("type", "bar"),
        "x": x,
        "y": y_cols,
        "title": spec.get("title") or "",
    }


def render(spec, result, key):
    """
    Draw a chart from a validated spec.

    Args:
        spec: Raw chart spec from the analysis plan.
        result: The executed result.
        key: A stable widget key, so Streamlit reruns do not collide.
    """
    import pandas as pd

    valid = validate_spec(spec, result)
    if not valid:
        return

    frame = result.to_frame() if isinstance(result, pd.Series) else result.copy()

    if valid["x"] and valid["x"] in frame.columns:
        frame = frame.set_index(valid["x"])

    plot_data = frame[valid["y"]]

    if valid["title"]:
        st.caption(valid["title"])

    try:
        if valid["type"] == "line":
            st.line_chart(plot_data)
        elif valid["type"] == "area":
            st.area_chart(plot_data)
        elif valid["type"] == "scatter":
            flat = frame.reset_index()
            st.scatter_chart(flat, x=flat.columns[0], y=valid["y"][0])
        else:
            st.bar_chart(plot_data)
    except Exception as exc:
        # A chart is a nice-to-have; never let one take down the answer above it.
        st.caption(f"Could not render chart: {exc}")


def render_result(result, key):
    """
    Show the raw computed result beneath an answer.

    Args:
        result: The executed result.
        key: A stable widget key.
    """
    import pandas as pd

    if isinstance(result, pd.DataFrame):
        st.dataframe(result, use_container_width=True)
    elif isinstance(result, pd.Series):
        st.dataframe(result.to_frame(), use_container_width=True)
    else:
        st.code(str(result), language="text")
