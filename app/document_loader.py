"""Document loading module.

Sole responsibility: turn files on disk (PDF, DOCX, TXT, Markdown) into
LangChain `Document` objects. Does not chunk, embed, or index — see
`chunking.py`, `embeddings.py`, and `vector_store.py` for those steps.
"""

from __future__ import annotations

from pathlib import Path

from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_core.documents import Document

from app.logger import get_logger
from app.utils import timed, validate_file_extension, validate_file_size

logger = get_logger(__name__)


class DocumentLoadError(RuntimeError):
    """Raised when a document exists and has a supported type, but fails to load."""


class EmptyDocumentError(ValueError):
    """Raised when a document loads successfully but contains no usable text."""


_LOADER_REGISTRY = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt": TextLoader,
    ".md": UnstructuredMarkdownLoader,
    ".markdown": UnstructuredMarkdownLoader,
}


def _build_loader(file_path: Path, extension: str):
    """Instantiate the correct LangChain loader for a given extension."""
    loader_cls = _LOADER_REGISTRY[extension]
    if loader_cls is TextLoader:
        return loader_cls(str(file_path), encoding="utf-8")
    return loader_cls(str(file_path))


@timed
def load_document(file_path: Path) -> list[Document]:
    """Load a single document from disk into LangChain `Document` objects.

    Args:
        file_path: Path to the source file.

    Returns:
        A list of LangChain `Document` objects (a PDF may yield one
        Document per page; other formats typically yield one Document).

    Raises:
        FileNotFoundError: If `file_path` does not exist.
        UnsupportedFileTypeError: If the file extension is not supported.
        FileTooLargeError: If the file exceeds the configured size limit.
        EmptyDocumentError: If the file loads but contains no text.
        DocumentLoadError: If the underlying loader raises (e.g. a
            corrupted PDF).
    """
    logger.info("Loading document: %s", file_path)

    validate_file_size(file_path)
    extension = validate_file_extension(file_path)

    try:
        loader = _build_loader(file_path, extension)
        documents = loader.load()
    except (FileNotFoundError, ValueError):
        raise
    except Exception as exc:  # noqa: BLE001 — third-party loaders raise varied types
        raise DocumentLoadError(
            f"Failed to load '{file_path.name}': the file may be corrupted "
            f"or in an unexpected format. Original error: {exc}"
        ) from exc

    non_empty_documents = [doc for doc in documents if doc.page_content.strip()]
    if not non_empty_documents:
        raise EmptyDocumentError(
            f"'{file_path.name}' loaded successfully but contains no "
            f"extractable text."
        )

    for index, document in enumerate(non_empty_documents):
        document.metadata["source_file"] = file_path.name
        document.metadata["file_path"] = str(file_path)
        document.metadata.setdefault("page", index)

    logger.info(
        "Loaded %d document section(s) from '%s'",
        len(non_empty_documents),
        file_path.name,
    )
    return non_empty_documents


@timed
def load_documents(file_paths: list[Path]) -> list[Document]:
    """Load multiple documents, skipping (and logging) any that fail.

    Args:
        file_paths: Paths to the source files to load.

    Returns:
        A combined list of LangChain `Document` objects from every file
        that loaded successfully. Files that fail are logged as warnings
        and excluded rather than aborting the whole batch.
    """
    all_documents: list[Document] = []
    for file_path in file_paths:
        try:
            all_documents.extend(load_document(file_path))
        except (
            FileNotFoundError,
            EmptyDocumentError,
            DocumentLoadError,
        ) as exc:
            logger.warning("Skipping '%s': %s", file_path.name, exc)
        except Exception as exc:  # noqa: BLE001 — unsupported extension, size, etc.
            logger.warning("Skipping '%s': %s", file_path.name, exc)

    logger.info(
        "Loaded %d total document section(s) from %d file(s)",
        len(all_documents),
        len(file_paths),
    )
    return all_documents
