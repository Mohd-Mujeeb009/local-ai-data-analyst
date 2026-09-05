"""
Root-level entry point.

Lets the app start with `streamlit run app.py` from the project root, which is
what the README documents and what Streamlit Community Cloud expects by default.
"""

import runpy
from pathlib import Path

runpy.run_path(
    str(Path(__file__).parent / "frontend" / "app.py"),
    run_name="__main__",
)
