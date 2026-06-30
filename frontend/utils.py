"""
Utility functions for detecting visualization intent in LLM responses.
"""


def wants_chart(text):
    """Check if the LLM response suggests a chart/graph would be helpful."""
    keywords = ["chart", "graph", "plot", "visual", "compare", "bar chart",
                "line chart", "pie chart", "histogram", "scatter"]
    return any(k in text.lower() for k in keywords)


def wants_table(text):
    """Check if the LLM response suggests showing a data table."""
    keywords = ["table", "tabular", "spreadsheet", "rows and columns"]
    return any(k in text.lower() for k in keywords)
