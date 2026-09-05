"""
AI Data Analyst - Streamlit entry point.

Run with:  streamlit run app.py
"""

import os
import sys

import streamlit as st

# Support `streamlit run frontend/app.py` as well as the root-level app.py shim.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.analysis import AnalysisError, run_analysis, stream_explanation  # noqa: E402
from backend.config import HISTORY_TURNS, MAX_UPLOAD_MB  # noqa: E402
from backend.context import summarize_pdf  # noqa: E402
from backend.file_handler import (  # noqa: E402
    FileError,
    get_file_type,
    load_dataframe,
    load_image,
    load_pdf,
)
from backend.llm_client import LLMError, check_connection, stream, stream_vision  # noqa: E402
from backend.prompts import CHAT_PROMPT, DOCUMENT_PROMPT  # noqa: E402
from frontend import charts  # noqa: E402
from frontend.state import clear_chat, clear_file, init_state, text_history  # noqa: E402

st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_state()


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
def render_sidebar():
    """Draw the sidebar: API key, upload, and controls."""
    with st.sidebar:
        st.subheader("Groq API key")

        # Fall back to an env var or Streamlit secret so a deployed instance can
        # be pre-configured, but never write the key back into os.environ.
        default_key = os.environ.get("GROQ_API_KEY", "")
        if not default_key:
            try:
                default_key = st.secrets.get("GROQ_API_KEY", "")
            except Exception:
                default_key = ""

        key = st.text_input(
            "API key",
            type="password",
            value=st.session_state.api_key or default_key,
            label_visibility="collapsed",
            placeholder="gsk_...",
            help="Free key from console.groq.com. Held in this session only.",
        )
        st.session_state.api_key = key

        # Validate once per distinct key, not on every rerun - this is a network
        # call and firing it per keystroke makes the whole app feel slow.
        if key and key != st.session_state.checked_key:
            with st.spinner("Checking key..."):
                st.session_state.key_status = check_connection(key)
                st.session_state.checked_key = key

        if key and st.session_state.key_status is True:
            st.success("Connected")
        elif key and st.session_state.key_status is False:
            st.error("Key rejected")
        else:
            st.info("Enter a key to begin")

        st.divider()
        st.subheader("Data")

        upload = st.file_uploader(
            "Upload a file",
            type=["csv", "xlsx", "xls", "pdf", "png", "jpg", "jpeg", "gif", "webp"],
            help=f"CSV, Excel, PDF or image. Up to {MAX_UPLOAD_MB} MB.",
        )
        if upload is not None and upload.name != st.session_state.file_name:
            handle_upload(upload)

        if st.session_state.file_name:
            st.caption(f"Loaded: **{st.session_state.file_name}**")
            if st.session_state.file_type == "data":
                df = st.session_state.df
                left, right = st.columns(2)
                left.metric("Rows", f"{len(df):,}")
                right.metric("Columns", len(df.columns))

        st.divider()

        st.session_state.show_code = st.toggle(
            "Show generated code",
            value=st.session_state.show_code,
            help="Reveal the pandas that produced each answer.",
        )

        left, right = st.columns(2)
        if left.button("Clear chat", use_container_width=True):
            clear_chat()
            st.rerun()
        if right.button("Reset all", use_container_width=True):
            clear_chat()
            clear_file()
            st.rerun()


def handle_upload(upload):
    """Load an upload into session state, replacing whatever was there."""
    file_type = get_file_type(upload.name)
    clear_file()

    try:
        if file_type == "data":
            st.session_state.df = load_dataframe(upload)
        elif file_type == "pdf":
            st.session_state.pdf_text = load_pdf(upload)
        elif file_type == "image":
            st.session_state.image_bytes = upload.getvalue()
            upload.seek(0)
            data, mime = load_image(upload)
            st.session_state.image_base64 = data
            st.session_state.image_mime = mime
        else:
            st.error(f"Unsupported file type: {upload.name}")
            return
    except FileError as exc:
        st.error(str(exc))
        return

    st.session_state.file_name = upload.name
    st.session_state.file_type = file_type
    clear_chat()


# --------------------------------------------------------------------------
# Message rendering
# --------------------------------------------------------------------------
def render_analysis(analysis, key):
    """Draw the code, chart and result attached to an assistant turn."""
    if st.session_state.show_code and analysis.get("code"):
        with st.expander("How this was calculated"):
            if analysis.get("explanation"):
                st.caption(analysis["explanation"])
            st.code(analysis["code"], language="python")

    result = analysis.get("result")
    if result is not None:
        charts.render(analysis.get("chart"), result, key=key)
        with st.expander("Show data"):
            charts.render_result(result, key=key)


def render_message(message, index):
    """
    Draw one chat message, including any analysis attached to it.

    Charts and results are re-rendered from stored state on every run, so they
    persist across reruns instead of vanishing the moment a widget is touched.
    """
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("analysis"):
            render_analysis(message["analysis"], index)


def stream_into_chat(chunks):
    """
    Render a stream into the chat and return the assembled text.

    `st.write_stream` handles the incremental repaint; wrapping it lets a
    provider error mid-stream surface as a message instead of a traceback.
    """
    try:
        return st.write_stream(chunks)
    except (LLMError, AnalysisError) as exc:
        st.error(str(exc))
        return None


# --------------------------------------------------------------------------
# Turn handlers
# --------------------------------------------------------------------------
def answer_dataframe(question):
    """Plan, execute and explain an analysis over the loaded DataFrame."""
    history = text_history(HISTORY_TURNS)[:-1]

    with st.status("Analysing...", expanded=False) as status:
        try:
            analysis = run_analysis(
                st.session_state.api_key, question, st.session_state.df, history
            )
        except (AnalysisError, LLMError) as exc:
            status.update(label="Analysis failed", state="error")
            st.error(str(exc))
            return None, None
        status.update(label="Done", state="complete")

    text = stream_into_chat(
        stream_explanation(
            st.session_state.api_key, question, analysis["code"], analysis["output"]
        )
    )
    if text is None:
        return None, None

    return text, {
        "code": analysis["code"],
        "chart": analysis["chart"],
        "explanation": analysis["explanation"],
        "result": analysis["result"],
    }


def answer_document(question):
    """Answer a question grounded in the extracted PDF text."""
    messages = [
        {"role": "system", "content": DOCUMENT_PROMPT},
        {"role": "system", "content": summarize_pdf(st.session_state.pdf_text)},
        *text_history(HISTORY_TURNS),
    ]
    return stream_into_chat(stream(st.session_state.api_key, messages)), None


def answer_image(question):
    """Answer a question about the uploaded image."""
    history = text_history(HISTORY_TURNS)[:-1]
    chunks = stream_vision(
        st.session_state.api_key,
        question,
        st.session_state.image_base64,
        st.session_state.image_mime,
        history,
    )
    return stream_into_chat(chunks), None


def answer_general(question):
    """Answer without any file loaded."""
    messages = [
        {"role": "system", "content": CHAT_PROMPT},
        *text_history(HISTORY_TURNS),
    ]
    return stream_into_chat(stream(st.session_state.api_key, messages)), None


HANDLERS = {
    "data": answer_dataframe,
    "pdf": answer_document,
    "image": answer_image,
}


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
render_sidebar()

st.title("AI Data Analyst")
st.caption(
    "Ask questions about your data in plain English. Answers are computed from "
    "your actual file, not guessed from a sample."
)

if st.session_state.file_type == "image" and st.session_state.image_bytes:
    st.image(st.session_state.image_bytes, width=480)

if not st.session_state.messages and st.session_state.file_type == "data":
    st.info(
        "Try: *Which rows have the highest values?*, *Summarise this dataset*, "
        "or *Show the trend over time*."
    )

for i, message in enumerate(st.session_state.messages):
    render_message(message, i)

question = st.chat_input("Ask about your data...")

if question:
    if not st.session_state.api_key:
        st.warning("Add your Groq API key in the sidebar first.")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            handler = HANDLERS.get(st.session_state.file_type, answer_general)
            reply, analysis = handler(question)

            if reply:
                message = {"role": "assistant", "content": reply}
                if analysis:
                    message["analysis"] = analysis
                    render_analysis(analysis, len(st.session_state.messages))
                st.session_state.messages.append(message)
            else:
                # Drop the orphaned user turn so a retry starts clean.
                st.session_state.messages.pop()
