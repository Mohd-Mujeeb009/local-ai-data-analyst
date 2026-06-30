"""
File handler — loads uploaded files into usable formats.
Supports CSV, Excel, PDF, and images.
"""

import base64
import pandas as pd
from pypdf import PdfReader


def load_file(file):
    """
    Load an uploaded file and return the appropriate data structure.

    Args:
        file: A Streamlit UploadedFile object.

    Returns:
        - pd.DataFrame for CSV/Excel files
        - str for PDF files (extracted text)
        - None for unsupported files

    Raises:
        ValueError: If the file cannot be parsed.
    """
    name = file.name.lower()

    try:
        if name.endswith(".csv"):
            return pd.read_csv(file)
        elif name.endswith(".xlsx") or name.endswith(".xls"):
            return pd.read_excel(file)
        elif name.endswith(".pdf"):
            reader = PdfReader(file)
            text = "\n".join(
                page.extract_text() or "" for page in reader.pages
            )
            if not text.strip():
                raise ValueError("PDF appears to be empty or contains only images.")
            return text
        else:
            return None
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Failed to read file '{file.name}': {str(e)}")


def encode_image_to_base64(file):
    """
    Read an uploaded image file and return its base64-encoded string.

    Args:
        file: A Streamlit UploadedFile object (image).

    Returns:
        str: Base64-encoded image data.
    """
    file_bytes = file.read()
    return base64.b64encode(file_bytes).decode("utf-8")


def get_file_type(file):
    """
    Determine the type category of an uploaded file.

    Args:
        file: A Streamlit UploadedFile object.

    Returns:
        str: One of 'data' (CSV/Excel), 'pdf', 'image', or 'unknown'.
    """
    name = file.name.lower()
    if name.endswith((".csv", ".xlsx", ".xls")):
        return "data"
    elif name.endswith(".pdf"):
        return "pdf"
    elif name.endswith((".png", ".jpg", ".jpeg", ".gif", ".webp")):
        return "image"
    else:
        return "unknown"
