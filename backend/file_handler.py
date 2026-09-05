"""
File loading for uploaded CSV, Excel, PDF and image files.

Every loader guards size before parsing - an unbounded `read_csv` on a large
upload will exhaust memory long before it raises anything useful.
"""

import base64

from backend.config import MAX_UPLOAD_MB

DATA_EXTENSIONS = (".csv", ".xlsx", ".xls")
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".webp")
PDF_EXTENSIONS = (".pdf",)

# Extension -> MIME, so a JPEG is not announced to the vision model as a PNG.
MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


class FileError(ValueError):
    """An upload that could not be read, with a message fit for the UI."""


def get_file_type(filename):
    """
    Categorise an upload by extension.

    Args:
        filename: The uploaded file's name.

    Returns:
        str: One of "data", "pdf", "image", "unknown".
    """
    name = filename.lower()
    if name.endswith(DATA_EXTENSIONS):
        return "data"
    if name.endswith(PDF_EXTENSIONS):
        return "pdf"
    if name.endswith(IMAGE_EXTENSIONS):
        return "image"
    return "unknown"


def get_mime_type(filename):
    """Return the image MIME type for a filename, defaulting to PNG."""
    name = filename.lower()
    for ext, mime in MIME_TYPES.items():
        if name.endswith(ext):
            return mime
    return "image/png"


def check_size(file):
    """
    Reject uploads above the configured limit.

    Raises:
        FileError: If the file exceeds MAX_UPLOAD_MB.
    """
    size = getattr(file, "size", None)
    if size is None:
        return
    megabytes = size / (1024 * 1024)
    if megabytes > MAX_UPLOAD_MB:
        raise FileError(
            f"File is {megabytes:.1f} MB, above the {MAX_UPLOAD_MB} MB limit. "
            "Sample or split it before uploading."
        )


def load_dataframe(file):
    """
    Load a CSV or Excel upload into a DataFrame.

    Returns:
        pd.DataFrame

    Raises:
        FileError: If the file is too large, empty, or unparseable.
    """
    import pandas as pd

    check_size(file)
    name = file.name.lower()

    try:
        df = pd.read_csv(file) if name.endswith(".csv") else pd.read_excel(file)
    except Exception as exc:
        raise FileError(f"Could not read {file.name}: {exc}") from exc

    if df.empty:
        raise FileError(f"{file.name} contains no rows.")

    return drop_index_column(df)


def drop_index_column(df):
    """
    Drop a leading unnamed column that is only a re-exported row index.

    Exported CSVs routinely carry an "Unnamed: 0" column holding 0..n-1. It is
    not data, and leaving it in pollutes the schema the planner reasons over.
    Removed only when it matches the row numbers exactly, so real data is safe.

    Args:
        df: The freshly loaded DataFrame.

    Returns:
        pd.DataFrame: The frame, with the stray index column removed if present.
    """
    import pandas as pd

    if df.empty:
        return df

    first = df.columns[0]
    if not (isinstance(first, str) and first.startswith("Unnamed:")):
        return df

    column = df[first]
    if column.dtype.kind in "iu" and column.reset_index(drop=True).equals(
        pd.Series(range(len(df)))
    ):
        return df.drop(columns=[first])

    return df


def load_pdf(file):
    """
    Extract text from a PDF upload.

    Returns:
        str: The extracted text.

    Raises:
        FileError: If the file is too large or has no extractable text.
    """
    from pypdf import PdfReader

    check_size(file)

    try:
        reader = PdfReader(file)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as exc:
        raise FileError(f"Could not read {file.name}: {exc}") from exc

    if not text.strip():
        raise FileError(
            f"No text found in {file.name}. Scanned PDFs need OCR before upload."
        )

    return text


def load_image(file):
    """
    Read an image upload as base64.

    Returns:
        tuple[str, str]: (base64 data, MIME type).

    Raises:
        FileError: If the file is too large or empty.
    """
    check_size(file)

    data = file.read()
    if not data:
        raise FileError(f"{file.name} is empty.")

    return base64.b64encode(data).decode("utf-8"), get_mime_type(file.name)
