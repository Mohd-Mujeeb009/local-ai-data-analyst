import streamlit as st

def init_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "df" not in st.session_state:
        st.session_state.df = None

    if "data_context" not in st.session_state:
        st.session_state.data_context = ""
