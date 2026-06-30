"""
📊 Local AI Data Analyst — Main Streamlit App
A ChatGPT-style chatbot that analyzes data and images using Groq API (LLaMA 3.3 / Vision).
Free API key from: https://console.groq.com
"""

import sys
import os
import streamlit as st

# Add project root to path so backend/frontend imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.llama_client import chat_with_llama, check_api_connection
from backend.file_handler import load_file, encode_image_to_base64, get_file_type
from backend.context import summarize_dataframe, summarize_pdf_text
from frontend.state import init_state
from frontend.utils import wants_chart, wants_table
from frontend.visualizer import render_table, render_chart, render_image

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Local AI Data Analyst",
    page_icon="📊",
    layout="wide",
)

# ---------------- CUSTOM STYLING ----------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    /* Global font */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #888;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 8px;
    }
    .status-connected {
        background: #d4edda;
        color: #155724;
    }
    .status-disconnected {
        background: #f8d7da;
        color: #721c24;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    [data-testid="stSidebar"] * {
        color: #e0e0e0 !important;
    }
    [data-testid="stSidebar"] .stTextInput label,
    [data-testid="stSidebar"] .stFileUploader label {
        color: #aaa !important;
        font-size: 0.85rem;
    }

    /* Chat message styling */
    [data-testid="stChatMessage"] {
        border-radius: 12px;
        margin-bottom: 8px;
    }

    /* Model info card */
    .model-info {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d1b69 100%);
        border-radius: 12px;
        padding: 12px 16px;
        margin: 8px 0;
        font-size: 0.8rem;
    }
    .model-info strong {
        color: #a78bfa !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- INITIALIZE STATE ----------------
init_state()

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    # API Key input
    st.markdown("### 🔑 Groq API Key")
    api_key_input = st.text_input(
        "Enter your free API key",
        type="password",
        value=st.session_state.get("groq_api_key", ""),
        help="Get a free key at https://console.groq.com",
        placeholder="gsk_...",
    )

    # Save API key to session state and environment
    if api_key_input:
        st.session_state.groq_api_key = api_key_input
        os.environ["GROQ_API_KEY"] = api_key_input

        # Check connection
        api_ok = check_api_connection(api_key_input)
        if api_ok:
            st.markdown('<span class="status-badge status-connected">🟢 API Connected</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-badge status-disconnected">🔴 Invalid Key</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge status-disconnected">🔑 Key Required</span>', unsafe_allow_html=True)
        st.caption("👉 [Get free API key](https://console.groq.com)")
        api_ok = False

    st.markdown("---")

    # Model info
    st.markdown("""
    <div class="model-info">
        <strong>Text:</strong> LLaMA 3.3 70B<br/>
        <strong>Vision:</strong> LLaMA 3.2 90B Vision<br/>
        <strong>Provider:</strong> Groq (Free Tier)
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # File uploader in sidebar
    st.markdown("### 📁 Upload a File")
    file = st.file_uploader(
        "CSV, Excel, PDF, or Image",
        type=["csv", "xlsx", "xls", "pdf", "png", "jpg", "jpeg"],
        help="Upload a dataset to analyze or an image to describe.",
    )

    # Process uploaded file
    if file and file.name != st.session_state.file_name:
        file_type = get_file_type(file)
        st.session_state.file_type = file_type
        st.session_state.file_name = file.name

        try:
            if file_type == "data":
                st.session_state.df = load_file(file)
                st.session_state.pdf_text = None
                st.session_state.image_base64 = None
                st.session_state.data_context = summarize_dataframe(st.session_state.df)
                st.success(f"✅ **{file.name}** — {len(st.session_state.df)} rows, {len(st.session_state.df.columns)} cols")

            elif file_type == "pdf":
                st.session_state.pdf_text = load_file(file)
                st.session_state.df = None
                st.session_state.image_base64 = None
                st.session_state.data_context = summarize_pdf_text(st.session_state.pdf_text)
                st.success(f"✅ **{file.name}** — {len(st.session_state.pdf_text):,} chars")

            elif file_type == "image":
                st.session_state.image_base64 = encode_image_to_base64(file)
                st.session_state.df = None
                st.session_state.pdf_text = None
                st.session_state.data_context = "An image has been uploaded for analysis."
                st.success(f"✅ **{file.name}** — ready for vision analysis")

            else:
                st.error(f"❌ Unsupported file type: {file.name}")

        except ValueError as e:
            st.error(f"❌ {str(e)}")

    # Show loaded file info
    if st.session_state.file_name:
        st.markdown("---")
        st.markdown(f"**Loaded:** `{st.session_state.file_name}`")

        if st.session_state.df is not None:
            st.markdown(f"📊 {len(st.session_state.df)} rows × {len(st.session_state.df.columns)} cols")
        elif st.session_state.pdf_text:
            st.markdown(f"📄 {len(st.session_state.pdf_text):,} characters")
        elif st.session_state.image_base64:
            st.markdown("🖼️ Image ready for analysis")

    # Clear chat button
    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------------- MAIN AREA ----------------
st.markdown('<h1 class="main-header">📊 Local AI Data Analyst</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Analyze your data, PDFs & images with AI — powered by Groq ⚡</p>', unsafe_allow_html=True)

# Show hint if no API key
if not api_key_input:
    st.info("👈 **Enter your Groq API key in the sidebar to get started.** Get a free key at [console.groq.com](https://console.groq.com)")

# Show uploaded image in main area
if st.session_state.image_base64 and st.session_state.file_type == "image":
    if file and get_file_type(file) == "image":
        file.seek(0)
        render_image(file)

# ---------------- CHAT HISTORY ----------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------- USER INPUT ----------------
user_prompt = st.chat_input("Ask about your data, PDF, or image...")

if user_prompt:
    # Check if API key is set
    if not api_key_input:
        st.warning("🔑 Please enter your Groq API key in the sidebar first.")
    else:
        # Check if any file is loaded
        has_context = (
            st.session_state.df is not None
            or st.session_state.pdf_text is not None
            or st.session_state.image_base64 is not None
        )

        if not has_context:
            st.warning("⬆️ Please upload a file first using the sidebar.")
        else:
            # Add user message to history
            st.session_state.messages.append({"role": "user", "content": user_prompt})

            with st.chat_message("user"):
                st.markdown(user_prompt)

            # Build recent messages for context window
            recent_messages = st.session_state.messages[-8:]

            # Call Groq API
            with st.chat_message("assistant"):
                with st.spinner("⚡ Thinking..."):
                    assistant_reply = chat_with_llama(
                        messages=recent_messages,
                        data_context=st.session_state.data_context,
                        image_base64=st.session_state.image_base64,
                        api_key=api_key_input,
                    )

                st.markdown(assistant_reply)

            # Save assistant reply
            st.session_state.messages.append(
                {"role": "assistant", "content": assistant_reply}
            )

            # ---------------- AUTO VISUALIZATION ----------------
            if st.session_state.df is not None:
                if wants_chart(assistant_reply):
                    render_chart(st.session_state.df)

                if wants_table(assistant_reply):
                    render_table(st.session_state.df)
