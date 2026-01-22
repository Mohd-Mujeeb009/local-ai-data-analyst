def summarize_dataframe(df):
    return f"""
Dataset summary:
- Rows: {len(df)}
- Columns: {list(df.columns)}

Sample data:
{df.head(5).to_csv(index=False)}
"""
