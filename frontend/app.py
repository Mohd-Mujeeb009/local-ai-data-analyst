import streamlit as st
import pandas as pd
import requests
from io import StringIO

OLLAMA_URL = "http://localhost:11434/api/chat"

st.set_page_config(layout="wide")
st.title("📊 ChatGPT-Style Data Chatbot (LLaMA 3.2)")

# ---------------- STATE ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "df" not in st.session_state:
    st.session_state.df = None

if "data_context" not in st.session_state:
    st.session_state.data_context = ""

# ---------------- FILE UPLOAD ----------------
file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

if file:
    if file.name.endswith(".csv"):
        st.session_state.df = pd.read_csv(file)
    else:
        st.session_state.df = pd.read_excel(file)

    st.session_state.data_context = f"""
Dataset loaded successfully.
Columns: {list(st.session_state.df.columns)}
Rows: {len(st.session_state.df)}

Sample:
{st.session_state.df.head(5).to_csv(index=False)}
"""

    st.success("Dataset uploaded and ready for analysis.")

# ---------------- CHAT HISTORY ----------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ---------------- USER INPUT ----------------
user_prompt = st.chat_input("Ask about your data...")

if user_prompt and st.session_state.df is not None:
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    with st.chat_message("user"):
        st.write(user_prompt)

    # ---------------- CONTEXT ----------------
    recent_messages = st.session_state.messages[-6:]

    system_prompt = """
You are a senior data analyst AI.
Speak naturally like ChatGPT.
Explain insights clearly and concisely.
Remember the dataset and prior messages.
If a visualization helps, say it explicitly.
Do NOT mention technical details.
"""

    messages = [{"role": "system", "content": system_prompt}]
    messages.append({"role": "system", "content": st.session_state.data_context})
    messages.extend(recent_messages)

    # ---------------- LLaMA CALL ----------------
    res = requests.post(
    OLLAMA_URL,
    json={
        "model": "llama3.2",
        "messages": messages,
        "stream": False   # 🔴 REQUIRED
    }
)

    try:
        assistant_reply = res.json()["message"]["content"]
    except Exception:
        assistant_reply = res.text

    # ---------------- ASSISTANT MESSAGE ----------------
    with st.chat_message("assistant"):
        st.write(assistant_reply)

    st.session_state.messages.append(
        {"role": "assistant", "content": assistant_reply}
    )

    # ---------------- AUTO VISUAL DETECTION ----------------
    text = assistant_reply.lower()

    if any(k in text for k in ["chart", "graph", "visual", "plot", "compare"]):
        st.markdown("### 📈 Visualization")

        numeric_cols = st.session_state.df.select_dtypes("number").columns

        if len(numeric_cols) >= 1:
            chart_df = st.session_state.df[numeric_cols]
            st.bar_chart(chart_df)

    if "table" in text:
        st.markdown("### 📋 Data Table")
        st.dataframe(st.session_state.df)
