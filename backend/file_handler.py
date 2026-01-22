import pandas as pd
from pypdf import PdfReader

def load_file(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    elif file.name.endswith(".xlsx"):
        return pd.read_excel(file)
    elif file.name.endswith(".pdf"):
        reader = PdfReader(file)
        return "\n".join(p.extract_text() for p in reader.pages)
    else:
        return None
