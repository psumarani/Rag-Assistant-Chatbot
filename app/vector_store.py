"""FAISS vector store management.

Sole responsibility: create, load, save, incrementally update, and
delete the FAISS index that backs semantic search. Callers interact
only with `Document` objects and never touch FAISS internals directly.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.embeddings import get_embedding_model
from app.logger import get_logger
from app.utils import ensure_directory, timed
from config.constants import FAISS_INDEX_NAME

logger = get_logger(__name__)


class VectorStoreNotFoundError(FileNotFoundError):
    """Raised when attempting to load a FAISS index that doesn't exist on disk."""


class EmptyVectorStoreError(ValueError):
    """Raised when attempting to build an index from zero chunks."""


def vector_store_exists(persist_dir: Path) -> bool:
    """Check whether a persisted FAISS index exists at the given directory.

    Args:
        persist_dir: Directory that would contain the FAISS index files.

    Returns:
        True if both the FAISS index and its docstore file are present.
    """
    index_file = persist_dir / f"{FAISS_INDEX_NAME}.faiss"
    store_file = persist_dir / f"{FAISS_INDEX_NAME}.pkl"
    return index_file.exists() and store_file.exists()


@timed
def create_vector_store(chunks: list[Document], persist_dir: Path) -> FAISS:
    """Build a new FAISS index from document chunks and persist it to disk.

    Args:
        chunks: Chunked `Document` objects to embed and index.
        persist_dir: Directory to save the FAISS index into.

    Returns:
        The newly created, in-memory FAISS vector store.

    Raises:
        EmptyVectorStoreError: If `chunks` is empty.
    """
    if not chunks:
        raise EmptyVectorStoreError("Cannot create a vector store from zero chunks.")

    logger.info("Creating FAISS index from %d chunk(s)", len(chunks))
    embedding_model = get_embedding_model()
    vector_store = FAISS.from_documents(documents=chunks, embedding=embedding_model)

    save_vector_store(vector_store, persist_dir)
    logger.info("FAISS index created and saved to '%s'", persist_dir)
    return vector_store


def save_vector_store(vector_store: FAISS, persist_dir: Path) -> None:
    """Persist a FAISS vector store to disk.

    Args:
        vector_store: The in-memory FAISS store to save.
        persist_dir: Directory to save the index into (created if needed).
    """
    ensure_directory(persist_dir)
    vector_store.save_local(folder_path=str(persist_dir), index_name=FAISS_INDEX_NAME)
    logger.info("Saved FAISS index to '%s'", persist_dir)


@timed
def load_vector_store(persist_dir: Path) -> FAISS:
    """Load a previously persisted FAISS vector store from disk.

    Args:
        persist_dir: Directory containing the saved FAISS index.

    Returns:
        The loaded FAISS vector store.

    Raises:
        VectorStoreNotFoundError: If no index exists at `persist_dir`.
    """
    if not vector_store_exists(persist_dir):
        raise VectorStoreNotFoundError(
            f"No FAISS index found at '{persist_dir}'. Build a knowledge "
            f"base first by uploading documents."
        )

    logger.info("Loading FAISS index from '%s'", persist_dir)
    embedding_model = get_embedding_model()
    vector_store = FAISS.load_local(
        folder_path=str(persist_dir),
        embeddings=embedding_model,
        index_name=FAISS_INDEX_NAME,
        # Safe here: we only ever load an index this same application wrote.
        allow_dangerous_deserialization=True,
    )
    logger.info("Loaded FAISS index from '%s'", persist_dir)
    return vector_store


@timed
def update_vector_store(
    vector_store: FAISS, new_chunks: list[Document], persist_dir: Path
) -> FAISS:
    """Incrementally add new chunks to an existing FAISS index and persist it.

    Args:
        vector_store: The existing in-memory FAISS store to extend.
        new_chunks: Newly chunked `Document` objects to add.
        persist_dir: Directory to re-save the updated index into.

    Returns:
        The same FAISS vector store instance, now including the new chunks.

    Raises:
        EmptyVectorStoreError: If `new_chunks` is empty.
    """
    if not new_chunks:
        raise EmptyVectorStoreError("Cannot update a vector store with zero new chunks.")

    logger.info("Adding %d new chunk(s) to existing FAISS index", len(new_chunks))
    vector_store.add_documents(new_chunks)
    save_vector_store(vector_store, persist_dir)
    return vector_store


def delete_vector_store(persist_dir: Path) -> None:
    """Delete a persisted FAISS index directory entirely.

    Args:
        persist_dir: Directory containing the saved FAISS index.
    """
    if persist_dir.exists():
        shutil.rmtree(persist_dir)
        logger.info("Deleted FAISS index at '%s'", persist_dir)
    else:
        logger.warning("Attempted to delete nonexistent vector store at '%s'", persist_dir)


@timed
def load_or_create_vector_store(chunks: list[Document], persist_dir: Path) -> FAISS:
    """Load an existing index if present, otherwise create one from `chunks`.

    If an index already exists, `chunks` are added to it incrementally
    instead of rebuilding from scratch (incremental indexing).

    Args:
        chunks: Chunked `Document` objects for the current upload batch.
        persist_dir: Directory where the FAISS index is persisted.

    Returns:
        The resulting FAISS vector store, containing both any
        previously indexed chunks and the newly added ones.
    """
    if vector_store_exists(persist_dir):
        vector_store = load_vector_store(persist_dir)
        if chunks:
            vector_store = update_vector_store(vector_store, chunks, persist_dir)
        return vector_store
    return create_vector_store(chunks, persist_dir)
