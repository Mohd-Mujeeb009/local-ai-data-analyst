"""Tests for upload handling."""

import io

import pytest

from backend.file_handler import (
    FileError,
    check_size,
    get_file_type,
    get_mime_type,
    load_dataframe,
)


class FakeUpload(io.BytesIO):
    """Stands in for a Streamlit UploadedFile."""

    def __init__(self, data, name, size=None):
        super().__init__(data)
        self.name = name
        self.size = size if size is not None else len(data)


@pytest.mark.parametrize("name,expected", [
    ("sales.csv", "data"),
    ("Report.XLSX", "data"),
    ("book.xls", "data"),
    ("paper.pdf", "pdf"),
    ("chart.PNG", "image"),
    ("photo.jpeg", "image"),
    ("anim.gif", "image"),
    ("notes.txt", "unknown"),
    ("archive.zip", "unknown"),
])
def test_get_file_type(name, expected):
    assert get_file_type(name) == expected


@pytest.mark.parametrize("name,expected", [
    ("a.png", "image/png"),
    ("a.jpg", "image/jpeg"),
    ("a.JPEG", "image/jpeg"),
    ("a.webp", "image/webp"),
    ("a.unknown", "image/png"),
])
def test_get_mime_type(name, expected):
    """A JPEG must not be announced to the vision model as a PNG."""
    assert get_mime_type(name) == expected


def test_check_size_rejects_large_file():
    huge = FakeUpload(b"x", "big.csv", size=200 * 1024 * 1024)
    with pytest.raises(FileError, match="above the"):
        check_size(huge)


def test_check_size_allows_normal_file():
    check_size(FakeUpload(b"col\n1\n", "small.csv"))


def test_load_dataframe_reads_csv():
    upload = FakeUpload(b"product,revenue\nA,100\nB,200\n", "s.csv")
    df = load_dataframe(upload)
    assert len(df) == 2
    assert list(df.columns) == ["product", "revenue"]


def test_load_dataframe_rejects_empty():
    with pytest.raises(FileError, match="no rows"):
        load_dataframe(FakeUpload(b"product,revenue\n", "s.csv"))


def test_load_dataframe_drops_exported_index_column():
    """Exported CSVs carry an Unnamed: 0 index column that is not data."""
    upload = FakeUpload(b",product,revenue\n0,A,100\n1,B,200\n", "s.csv")
    df = load_dataframe(upload)
    assert list(df.columns) == ["product", "revenue"]


def test_load_dataframe_keeps_unnamed_column_holding_real_data():
    """An unnamed column that is not a row index must survive."""
    upload = FakeUpload(b",product\n7,A\n99,B\n", "s.csv")
    df = load_dataframe(upload)
    assert len(df.columns) == 2


def test_load_dataframe_reports_unparseable():
    with pytest.raises(FileError):
        load_dataframe(FakeUpload(b"\x00\x01\x02binary", "x.xlsx"))
