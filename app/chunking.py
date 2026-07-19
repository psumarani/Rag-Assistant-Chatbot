"""Text splitting module.

Sole responsibility: split loaded `Document` objects into smaller,
overlapping chunks suitable for embedding, while preserving and
enriching metadata (chunk id, chunk index, source file).
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.logger import get_logger
from app.utils import generate_chunk_id, timed
from config.settings import settings

logger = get_logger(__name__)


def _build_splitter(
    chunk_size: int | None = None, chunk_overlap: int | None = None
) -> RecursiveCharacterTextSplitter:
    """Construct a configured `RecursiveCharacterTextSplitter`.

    Args:
        chunk_size: Override for the configured chunk size. Defaults to
            `settings.chunk_size`.
        chunk_overlap: Override for the configured chunk overlap.
            Defaults to `settings.chunk_overlap`.

    Returns:
        A ready-to-use text splitter.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=chunk_overlap or settings.chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


@timed
def split_documents(
    documents: list[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """Split documents into overlapping chunks with enriched metadata.

    Args:
        documents: LangChain `Document` objects to split.
        chunk_size: Optional override for chunk size (characters).
        chunk_overlap: Optional override for chunk overlap (characters).

    Returns:
        A list of chunked `Document` objects. Each chunk's metadata
        includes `chunk_id`, `chunk_index`, and inherits `source_file`
        and `page` from its parent document.
    """
    if not documents:
        logger.warning("split_documents called with an empty document list")
        return []

    splitter = _build_splitter(chunk_size, chunk_overlap)
    chunks = splitter.split_documents(documents)

    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = generate_chunk_id()
        chunk.metadata["chunk_index"] = index
        chunk.metadata["chunk_char_count"] = len(chunk.page_content)

    logger.info(
        "Split %d document section(s) into %d chunk(s) "
        "(chunk_size=%d, chunk_overlap=%d)",
        len(documents),
        len(chunks),
        chunk_size or settings.chunk_size,
        chunk_overlap or settings.chunk_overlap,
    )
    return chunks
