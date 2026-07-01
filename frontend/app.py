"""
Minimal Streamlit frontend for Local AI Data Analyst.
This file provides a clean, simple UI while preserving core features:
- Enter Groq API key
- Upload CSV/Excel/PDF/Image
- Chat interface that queries `backend.llama_client.chat_with_llama`
"""

import sys
import os
import streamlit as st

# Allow importing from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.llama_client import chat_with_llama, check_api_connection
from backend.file_handler import load_file, encode_image_to_base64, get_file_type
from backend.context import summarize_dataframe, summarize_pdf_text
from frontend.state import init_state
from frontend.utils import wants_chart, wants_table
from frontend.visualizer import render_table, render_chart, render_image

st.set_page_config(page_title="Local AI Data Analyst", page_icon="📊", layout="wide")

init_state()

# --- Sidebar ---
with st.sidebar:
    st.header("Settings")

    api_key_input = st.text_input("Groq API Key", type="password", value=st.session_state.get("groq_api_key", ""))
    if api_key_input:
        st.session_state.groq_api_key = api_key_input
        os.environ["GROQ_API_KEY"] = api_key_input
        if check_api_connection(api_key_input):
            st.success("API connected")
        else:
            st.error("Invalid API key")

    st.markdown("---")

    st.subheader("Upload")
    file = st.file_uploader("CSV, Excel, PDF, or Image", type=["csv", "xlsx", "xls", "pdf", "png", "jpg", "jpeg"])

    if file and file.name != st.session_state.get("file_name"):
        st.session_state.file_name = file.name
        ftype = get_file_type(file)
        st.session_state.file_type = ftype

        try:
            if ftype == "data":
                st.session_state.df = load_file(file)
                st.session_state.pdf_text = None
                st.session_state.image_base64 = None
                st.session_state.data_context = summarize_dataframe(st.session_state.df)
                st.success(f"Loaded {file.name} — {len(st.session_state.df)} rows")
            elif ftype == "pdf":
                st.session_state.pdf_text = load_file(file)
                st.session_state.df = None
                st.session_state.image_base64 = None
                st.session_state.data_context = summarize_pdf_text(st.session_state.pdf_text)
                st.success(f"Loaded {file.name}")
            elif ftype == "image":
                st.session_state.image_base64 = encode_image_to_base64(file)
                st.session_state.df = None
                st.session_state.pdf_text = None
                st.session_state.data_context = "Image uploaded for analysis"
                st.success(f"Loaded {file.name}")
            else:
                st.error("Unsupported file type")
        except Exception as e:
            st.error(f"Failed to load file: {e}")

    st.markdown("---")
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.experimental_rerun()

# --- Main ---
st.title("Local AI Data Analyst")
st.write("Upload a dataset or image in the sidebar, enter your Groq API key, then ask questions below.")

# Show simple file info
if st.session_state.get("file_name"):
    st.info(f"Loaded: {st.session_state.file_name}")

# Display uploaded image if available
if st.session_state.get("image_base64") and st.session_state.get("file_type") == "image":
    # streamlit image expects a file-like or bytes; use render_image helper when possible
    st.image(st.session_state.image_base64, use_column_width=True)

# Chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_prompt = st.chat_input("Ask about your data, PDF, or image...")

if user_prompt:
    if not st.session_state.get("groq_api_key"):
        st.warning("Enter your Groq API key in the sidebar")
    else:
        has_context = any([st.session_state.get("df"), st.session_state.get("pdf_text"), st.session_state.get("image_base64")])
        if not has_context:
            st.warning("Please upload a file first")
        else:
            st.session_state.messages.append({"role": "user", "content": user_prompt})
            with st.chat_message("user"):
                st.markdown(user_prompt)

            recent = st.session_state.messages[-8:]
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    reply = chat_with_llama(
                        messages=recent,
                        data_context=st.session_state.get("data_context", ""),
                        image_base64=st.session_state.get("image_base64"),
                        api_key=st.session_state.get("groq_api_key"),
                    )
                st.markdown(reply)

            st.session_state.messages.append({"role": "assistant", "content": reply})

            # Simple auto-visualization
            if st.session_state.get("df") is not None:
                if wants_chart(reply):
                    render_chart(st.session_state.df)
                if wants_table(reply):
                    render_table(st.session_state.df)
