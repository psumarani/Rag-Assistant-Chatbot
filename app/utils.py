"""Shared, side-effect-light utility functions used across the app."""

from __future__ import annotations

import functools
import time
import uuid
from pathlib import Path
from typing import Any, Callable, TypeVar

from config.constants import MAX_UPLOAD_SIZE_BYTES, MAX_UPLOAD_SIZE_MB, SUPPORTED_EXTENSIONS
from app.logger import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class UnsupportedFileTypeError(ValueError):
    """Raised when a file extension is not in SUPPORTED_EXTENSIONS."""


class FileTooLargeError(ValueError):
    """Raised when an uploaded file exceeds MAX_UPLOAD_SIZE_BYTES."""


def generate_chunk_id() -> str:
    """Generate a short, unique identifier for a document chunk.

    Returns:
        A UUID4 hex string truncated to 12 characters — short enough to
        display in the UI, long enough to be effectively unique.
    """
    return uuid.uuid4().hex[:12]


def validate_file_extension(file_path: Path) -> str:
    """Validate that a file's extension is supported.

    Args:
        file_path: Path to the file being validated.

    Returns:
        The lowercase file extension (including the leading dot).

    Raises:
        UnsupportedFileTypeError: If the extension is not supported.
    """
    extension = file_path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise UnsupportedFileTypeError(
            f"Unsupported file type '{extension}' for '{file_path.name}'. "
            f"Supported types are: {supported}."
        )
    return extension


def validate_file_size(file_path: Path) -> None:
    """Validate that a file does not exceed the maximum allowed upload size.

    Args:
        file_path: Path to the file being validated.

    Raises:
        FileTooLargeError: If the file exceeds MAX_UPLOAD_SIZE_BYTES.
        FileNotFoundError: If the file does not exist.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: '{file_path}'")

    size_bytes = file_path.stat().st_size
    if size_bytes > MAX_UPLOAD_SIZE_BYTES:
        size_mb = size_bytes / (1024 * 1024)
        raise FileTooLargeError(
            f"File '{file_path.name}' is {size_mb:.2f} MB, which exceeds "
            f"the maximum allowed size of {MAX_UPLOAD_SIZE_MB} MB."
        )


def timed(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator that logs the execution time of the wrapped function.

    Args:
        func: The function to time.

    Returns:
        A wrapped function that logs its own execution duration at
        DEBUG level and re-raises any exception unmodified.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            elapsed = time.perf_counter() - start
            logger.debug("'%s' completed in %.3f seconds", func.__name__, elapsed)

    return wrapper


def ensure_directory(directory: Path) -> Path:
    """Create a directory (including parents) if it does not already exist.

    Args:
        directory: The directory path to ensure exists.

    Returns:
        The same directory path, for convenient chaining.
    """
    directory.mkdir(parents=True, exist_ok=True)
    return directory
