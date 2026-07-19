"""Environment-driven application settings.

Loads configuration from a local `.env` file using `python-dotenv` and
exposes it as a single, typed, immutable `Settings` dataclass instance
(`settings`) that the rest of the application imports.

No other module should call `os.getenv` directly — this is the single
entry point for environment configuration (Single Responsibility +
Don't Repeat Yourself).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from config.constants import (
    DATA_DIR,
    DOCUMENTS_DIR,
    LOGS_DIR,
    VECTOR_STORE_DIR,
)

# Load variables from a .env file in the project root, if present.
# Does nothing (silently) if the file does not exist yet.
load_dotenv()


class MissingAPIKeyError(RuntimeError):
    """Raised when GOOGLE_API_KEY is not set in the environment."""


def _get_str(key: str, default: str) -> str:
    """Return a string environment variable or its default."""
    value = os.getenv(key)
    return value if value not in (None, "") else default


def _get_int(key: str, default: int) -> int:
    """Return an int environment variable or its default, with validation."""
    raw_value = os.getenv(key)
    if raw_value in (None, ""):
        return default
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"Environment variable '{key}' must be an integer, got: {raw_value!r}"
        ) from exc


def _get_float(key: str, default: float) -> float:
    """Return a float environment variable or its default, with validation."""
    raw_value = os.getenv(key)
    if raw_value in (None, ""):
        return default
    try:
        return float(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"Environment variable '{key}' must be a float, got: {raw_value!r}"
        ) from exc


@dataclass(frozen=True)
class Settings:
    """Immutable application settings loaded from environment variables.

    Attributes:
        google_api_key: API key for Google Gemini. May be empty at import
            time; validated lazily via `require_api_key()` so the app can
            still start (e.g. to show a friendly setup message in the UI)
            without a key configured.
        gemini_model: Gemini model identifier to use for generation.
        chunk_size: Number of characters per text chunk.
        chunk_overlap: Number of overlapping characters between chunks.
        top_k: Number of chunks to retrieve per query.
        temperature: Sampling temperature for Gemini generation.
        max_output_tokens: Maximum tokens Gemini may generate per response.
        data_dir: Root directory for all data artifacts.
        documents_dir: Directory where uploaded source documents are stored.
        vector_store_dir: Directory where the FAISS index is persisted.
        logs_dir: Directory where log files are written.
    """

    google_api_key: str
    gemini_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    temperature: float
    max_output_tokens: int
    data_dir: Path
    documents_dir: Path
    vector_store_dir: Path
    logs_dir: Path

    def require_api_key(self) -> str:
        """Return the Gemini API key, raising if it has not been configured.

        Returns:
            The configured Google Gemini API key.

        Raises:
            MissingAPIKeyError: If GOOGLE_API_KEY is unset, empty, or still
                the placeholder value from `.env.example`.
        """
        placeholder_values = {"", "YOUR_GEMINI_API_KEY_HERE"}
        if self.google_api_key in placeholder_values:
            raise MissingAPIKeyError(
                "GOOGLE_API_KEY is missing or still set to its placeholder "
                "value. Create a '.env' file (see '.env.example') and set "
                "GOOGLE_API_KEY to a valid Google AI Studio API key."
            )
        return self.google_api_key


def load_settings() -> Settings:
    """Build a `Settings` instance from the current environment.

    Returns:
        A populated, validated `Settings` dataclass instance.

    Raises:
        ValueError: If a numeric environment variable cannot be parsed.
    """
    chunk_size = _get_int("CHUNK_SIZE", 1000)
    chunk_overlap = _get_int("CHUNK_OVERLAP", 200)

    if chunk_size <= 0:
        raise ValueError("CHUNK_SIZE must be a positive integer.")
    if chunk_overlap < 0:
        raise ValueError("CHUNK_OVERLAP must be zero or a positive integer.")
    if chunk_overlap >= chunk_size:
        raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")

    top_k = _get_int("TOP_K", 4)
    if top_k <= 0:
        raise ValueError("TOP_K must be a positive integer.")

    temperature = _get_float("TEMPERATURE", 0.2)
    if not 0.0 <= temperature <= 2.0:
        raise ValueError("TEMPERATURE must be between 0.0 and 2.0.")

    max_output_tokens = _get_int("MAX_OUTPUT_TOKENS", 2048)
    if max_output_tokens <= 0:
        raise ValueError("MAX_OUTPUT_TOKENS must be a positive integer.")

    return Settings(
        google_api_key=_get_str("GOOGLE_API_KEY", ""),
        gemini_model=_get_str("GEMINI_MODEL", "gemini-2.5-flash"),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        data_dir=DATA_DIR,
        documents_dir=DOCUMENTS_DIR,
        vector_store_dir=VECTOR_STORE_DIR,
        logs_dir=LOGS_DIR,
    )


# Singleton settings instance imported by the rest of the application.
settings: Settings = load_settings()
