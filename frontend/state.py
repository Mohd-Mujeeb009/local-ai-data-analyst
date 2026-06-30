"""
State management for the Streamlit app.
"""

import streamlit as st


def init_state():
    """Initialize all session state variables with defaults."""
    defaults = {
        "messages": [],        # Chat history
        "df": None,            # Loaded DataFrame (CSV/Excel)
        "pdf_text": None,      # Extracted PDF text
        "image_base64": None,  # Base64-encoded uploaded image
        "data_context": "",    # Context string sent to LLM
        "file_type": None,     # Type of uploaded file: 'data', 'pdf', 'image'
        "file_name": None,     # Name of the currently loaded file
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
