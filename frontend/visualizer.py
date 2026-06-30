"""
Visualization helpers — render charts, tables, and images in Streamlit.
"""

import streamlit as st
import pandas as pd


def render_table(df):
    """Display a styled data table."""
    st.markdown("### 📋 Data Table")
    st.dataframe(df, use_container_width=True)


def render_chart(df):
    """Display a bar chart of numeric columns."""
    numeric_cols = df.select_dtypes("number").columns

    if len(numeric_cols) == 0:
        st.warning("No numeric columns available to visualize.")
        return

    st.markdown("### 📊 Auto-Generated Chart")

    # If there are too many numeric columns, let user pick
    if len(numeric_cols) > 5:
        selected = st.multiselect(
            "Select columns to chart:",
            options=list(numeric_cols),
            default=list(numeric_cols[:3]),
        )
        if selected:
            st.bar_chart(df[selected])
    else:
        st.bar_chart(df[numeric_cols])


def render_image(image_file):
    """Display an uploaded image."""
    st.markdown("### 🖼️ Uploaded Image")
    st.image(image_file, use_container_width=True)
