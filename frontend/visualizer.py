"""
Visualization helpers — render charts, tables, and images in Streamlit.
Enhanced with premium styling and better chart capabilities.
"""

import streamlit as st
import pandas as pd


def render_table(df):
    """Display a styled data table with metadata."""
    st.markdown("---")
    st.markdown("### 📋 Data Table")

    # Show quick stats
    num_cols = len(df.select_dtypes("number").columns)
    cat_cols = len(df.select_dtypes(exclude="number").columns)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", f"{len(df):,}")
    with col2:
        st.metric("Numeric Cols", num_cols)
    with col3:
        st.metric("Text Cols", cat_cols)

    st.dataframe(
        df,
        use_container_width=True,
        height=min(400, (len(df) + 1) * 35 + 38),
    )


def render_chart(df):
    """Display an auto-generated chart with column selection."""
    numeric_cols = df.select_dtypes("number").columns

    if len(numeric_cols) == 0:
        st.warning("No numeric columns available to visualize.")
        return

    st.markdown("---")
    st.markdown("### 📊 Auto-Generated Chart")

    # Chart type selector
    chart_type = st.radio(
        "Chart type:",
        ["Bar", "Line", "Area"],
        horizontal=True,
        key="chart_type_selector",
    )

    # If there are too many numeric columns, let user pick
    if len(numeric_cols) > 5:
        selected = st.multiselect(
            "Select columns to chart:",
            options=list(numeric_cols),
            default=list(numeric_cols[:3]),
            key="chart_cols_selector",
        )
        if not selected:
            st.info("👆 Select at least one column to visualize.")
            return
        plot_data = df[selected]
    else:
        plot_data = df[numeric_cols]

    # Render the selected chart type
    if chart_type == "Bar":
        st.bar_chart(plot_data)
    elif chart_type == "Line":
        st.line_chart(plot_data)
    elif chart_type == "Area":
        st.area_chart(plot_data)


def render_image(image_file):
    """Display an uploaded image with a styled container."""
    st.markdown("---")
    st.markdown("### 🖼️ Uploaded Image")
    st.image(image_file, use_container_width=True)
