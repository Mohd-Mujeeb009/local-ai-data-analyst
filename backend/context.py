"""
Context builders - turn uploaded files into compact model-readable context.

The DataFrame path deliberately sends the *schema*, not the data: the analysis
pipeline computes real answers from the real frame, so shipping rows would only
burn context and invite the model to guess from a sample.
"""

from backend.config import MAX_PDF_CHARS, MAX_PREVIEW_ROWS, MAX_SCHEMA_COLUMNS


def describe_schema(df):
    """
    Build a schema description for the analysis planner.

    Args:
        df: A pandas DataFrame.

    Returns:
        str: Column names, dtypes, cardinality and a few sample values per
            column - enough to write correct pandas, not enough to fake results.
    """
    lines = [f"Shape: {len(df):,} rows x {len(df.columns)} columns", "", "Columns:"]

    columns = list(df.columns)
    shown = columns[:MAX_SCHEMA_COLUMNS]

    for col in shown:
        series = df[col]
        dtype = str(series.dtype)
        detail = f"  - {col!r} ({dtype})"

        try:
            nulls = int(series.isna().sum())
            if nulls:
                detail += f", {nulls} null"

            if series.dtype.kind in "ifc":
                detail += f", range {series.min()} to {series.max()}"
            else:
                uniques = series.dropna().unique()
                detail += f", {len(uniques)} distinct"
                if 0 < len(uniques) <= 8:
                    detail += f", values: {list(uniques)}"
                elif len(uniques) > 8:
                    detail += f", e.g. {list(uniques[:4])}"
        except (TypeError, ValueError):
            # Mixed-type or exotic columns: the name and dtype still help.
            pass

        lines.append(detail)

    if len(columns) > MAX_SCHEMA_COLUMNS:
        lines.append(f"  [... {len(columns) - MAX_SCHEMA_COLUMNS} more columns omitted]")

    lines += ["", f"First {MAX_PREVIEW_ROWS} rows (shape reference only):",
              df.head(MAX_PREVIEW_ROWS).to_string()]

    return "\n".join(lines)


def summarize_pdf(text):
    """
    Build document context from extracted PDF text.

    Args:
        text: Full extracted text.

    Returns:
        str: The text, truncated to the configured budget with the cut declared
            so the model can say its answer is partial.
    """
    total = len(text)
    if total <= MAX_PDF_CHARS:
        return f"Document ({total:,} characters, complete):\n\n{text}"

    return (
        f"Document ({total:,} characters, TRUNCATED to the first {MAX_PDF_CHARS:,}). "
        f"Content beyond this point was not provided.\n\n{text[:MAX_PDF_CHARS]}"
    )
