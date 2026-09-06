"""Session state for the Streamlit app."""

import streamlit as st

DEFAULTS = {
    "messages": [],        # chat history: role, content, and any analysis payload
    "api_key": "",         # held in session only, never in os.environ
    "key_status": None,    # None (unchecked), True (valid), False (invalid)
    "checked_key": "",     # the key that key_status refers to
    "df": None,            # loaded DataFrame
    "pdf_text": None,      # extracted PDF text
    "image_base64": None,  # base64 image data
    "image_mime": None,    # MIME type of the uploaded image
    "image_bytes": None,   # raw bytes, for display
    "file_name": None,     # currently loaded file
    "file_type": None,     # "data" | "pdf" | "image"
    "show_code": True,     # reveal generated pandas alongside answers
    "doc_id": None,        # retrieval index id for the loaded PDF, if indexed
    "doc_chunks": 0,       # how many passages that index holds
}


def init_state():
    """Populate any session keys that are not set yet."""
    for key, value in DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clear_file():
    """Drop the loaded file and everything derived from it."""
    for key in ("df", "pdf_text", "image_base64", "image_mime",
                "image_bytes", "file_name", "file_type", "doc_id", "doc_chunks"):
        st.session_state[key] = DEFAULTS[key]


def clear_chat():
    """Drop the conversation, keeping the loaded file in place."""
    st.session_state.messages = []


def text_history(limit):
    """
    Return recent messages as plain role/content dicts for the provider.

    Analysis payloads are stripped - the API only accepts role and content.
    """
    return [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[-limit:]
    ]
