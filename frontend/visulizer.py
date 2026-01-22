import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def render_table(df):
    st.markdown("### 📋 Table")
    st.dataframe(df)

def render_chart(df):
    numeric_cols = df.select_dtypes("number").columns

    if len(numeric_cols) == 0:
        st.warning("No numeric columns to visualize.")
        return

    st.markdown("### 📊 Chart")
    st.bar_chart(df[numeric_cols])
