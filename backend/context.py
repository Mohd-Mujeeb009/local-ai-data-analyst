"""
Context builders — convert uploaded data into LLM-friendly context strings.
"""

import pandas as pd


def summarize_dataframe(df):
    """
    Generate a concise summary of a DataFrame for the LLM.

    Args:
        df: A pandas DataFrame.

    Returns:
        str: A formatted summary including row count, columns, dtypes, and a sample.
    """
    dtypes_info = "\n".join(f"  - {col}: {dtype}" for col, dtype in df.dtypes.items())

    return f"""
Dataset loaded successfully.
- Rows: {len(df)}
- Columns: {list(df.columns)}

Column types:
{dtypes_info}

Basic statistics:
{df.describe(include='all').to_string()}

Sample data (first 5 rows):
{df.head(5).to_csv(index=False)}
"""


def summarize_pdf_text(text):
    """
    Generate a context string from extracted PDF text.

    Args:
        text: Extracted text content from a PDF.

    Returns:
        str: A formatted summary with the text content (truncated if very long).
    """
    # Truncate very long PDFs to keep within LLM context limits
    max_chars = 8000
    truncated = text[:max_chars]
    suffix = f"\n\n[... truncated, showing {max_chars} of {len(text)} characters]" if len(text) > max_chars else ""

    return f"""
PDF document loaded successfully.
- Total characters: {len(text)}

Content:
{truncated}{suffix}
"""
