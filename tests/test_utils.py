"""Unit tests for app.utils."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.utils import (
    FileTooLargeError,
    UnsupportedFileTypeError,
    generate_chunk_id,
    validate_file_extension,
    validate_file_size,
)


def test_generate_chunk_id_is_unique_and_short() -> None:
    first_id = generate_chunk_id()
    second_id = generate_chunk_id()
    assert first_id != second_id
    assert len(first_id) == 12


def test_validate_file_extension_accepts_supported_type(tmp_path: Path) -> None:
    file_path = tmp_path / "notes.txt"
    file_path.write_text("hello")
    assert validate_file_extension(file_path) == ".txt"


def test_validate_file_extension_rejects_unsupported_type(tmp_path: Path) -> None:
    file_path = tmp_path / "archive.zip"
    file_path.write_text("hello")
    with pytest.raises(UnsupportedFileTypeError):
        validate_file_extension(file_path)


def test_validate_file_size_raises_for_missing_file(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.txt"
    with pytest.raises(FileNotFoundError):
        validate_file_size(missing_path)


def test_validate_file_size_raises_when_too_large(tmp_path: Path, monkeypatch) -> None:
    file_path = tmp_path / "big.txt"
    file_path.write_text("x")

    import app.utils as utils_module

    monkeypatch.setattr(utils_module, "MAX_UPLOAD_SIZE_BYTES", 0)
    with pytest.raises(FileTooLargeError):
        validate_file_size(file_path)
