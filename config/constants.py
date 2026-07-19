"""Static constants for the Professional RAG Assistant.

This module holds values that are fixed at the code level and are NOT
expected to change between environments (unlike `config/settings.py`,
which loads environment-dependent values from `.env`).

Keeping these separate from `settings.py` follows the Single
Responsibility Principle: constants.py answers "what are the fixed
rules of this system?" while settings.py answers "what does this
particular deployment look like?".
"""

from pathlib import Path
from typing import Final

# --------------------------------------------------------------------------- #
# Project paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
DOCUMENTS_DIR: Final[Path] = DATA_DIR / "documents"
VECTOR_STORE_DIR: Final[Path] = DATA_DIR / "vector_store"
LOGS_DIR: Final[Path] = PROJECT_ROOT / "logs"

# --------------------------------------------------------------------------- #
# Vector store artifact names
# --------------------------------------------------------------------------- #
FAISS_INDEX_NAME: Final[str] = "faiss_index"
FAISS_METADATA_FILE: Final[str] = "faiss_index_metadata.json"

# --------------------------------------------------------------------------- #
# Supported document types
# --------------------------------------------------------------------------- #
SUPPORTED_EXTENSIONS: Final[frozenset[str]] = frozenset(
    {".pdf", ".docx", ".txt", ".md", ".markdown"}
)

# --------------------------------------------------------------------------- #
# Embedding model
# --------------------------------------------------------------------------- #
DEFAULT_EMBEDDING_MODEL: Final[str] = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION: Final[int] = 384

# --------------------------------------------------------------------------- #
# Security limits
# --------------------------------------------------------------------------- #
MAX_UPLOAD_SIZE_MB: Final[int] = 25
MAX_UPLOAD_SIZE_BYTES: Final[int] = MAX_UPLOAD_SIZE_MB * 1024 * 1024

# --------------------------------------------------------------------------- #
# Retrieval defaults (fallbacks only — real values come from settings.py)
# --------------------------------------------------------------------------- #
MIN_SIMILARITY_THRESHOLD: Final[float] = 0.0
MAX_SIMILARITY_THRESHOLD: Final[float] = 1.0

# --------------------------------------------------------------------------- #
# Fallback answer when no relevant context is found
# --------------------------------------------------------------------------- #
NO_ANSWER_FOUND_MESSAGE: Final[str] = (
    "I could not find enough information in the uploaded documents."
)

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
LOG_FILE_NAME: Final[str] = "rag_assistant.log"
LOG_FORMAT: Final[str] = (
    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)
LOG_DATE_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"
