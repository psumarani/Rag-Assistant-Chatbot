"""Unit tests for app.document_loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.document_loader import (
    DocumentLoadError,
    EmptyDocumentError,
    load_document,
    load_documents,
)
from app.utils import UnsupportedFileTypeError


def test_load_document_reads_txt_file(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.txt"
    file_path.write_text("This is a test document about cats.")

    documents = load_document(file_path)

    assert len(documents) == 1
    assert "cats" in documents[0].page_content
    assert documents[0].metadata["source_file"] == "sample.txt"


def test_load_document_raises_for_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_document(tmp_path / "missing.txt")


def test_load_document_raises_for_unsupported_extension(tmp_path: Path) -> None:
    file_path = tmp_path / "data.csv"
    file_path.write_text("a,b,c")
    with pytest.raises(UnsupportedFileTypeError):
        load_document(file_path)


def test_load_document_raises_for_empty_file(tmp_path: Path) -> None:
    file_path = tmp_path / "empty.txt"
    file_path.write_text("   \n  ")
    with pytest.raises(EmptyDocumentError):
        load_document(file_path)


def test_load_documents_skips_failures_and_continues(tmp_path: Path) -> None:
    good_file = tmp_path / "good.txt"
    good_file.write_text("Valid content here.")
    bad_file = tmp_path / "bad.csv"
    bad_file.write_text("x,y,z")

    documents = load_documents([good_file, bad_file])

    assert len(documents) == 1
    assert documents[0].metadata["source_file"] == "good.txt"
