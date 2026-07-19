"""Retrieval module.

Sole responsibility: run similarity search against a FAISS vector store
and return a structured, UI-friendly result (chunk text, similarity
score, source file, and metadata) rather than raw LangChain objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_community.vectorstores import FAISS

from app.logger import get_logger
from app.utils import timed
from config.constants import MAX_SIMILARITY_THRESHOLD, MIN_SIMILARITY_THRESHOLD
from config.settings import settings

logger = get_logger(__name__)


@dataclass(frozen=True)
class RetrievedChunk:
    """A single retrieved chunk, ready for display or prompt assembly.

    Attributes:
        content: The chunk's text content.
        similarity_score: Similarity in [0.0, 1.0]; higher is more relevant.
        source_file: The originating document's file name.
        metadata: Full chunk metadata (chunk_id, chunk_index, page, etc.).
    """

    content: str
    similarity_score: float
    source_file: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _distance_to_similarity(distance: float) -> float:
    """Convert a FAISS L2 distance into a bounded [0.0, 1.0] similarity score.

    FAISS (with normalized embeddings) returns squared L2 distance, where
    0 means identical. This maps it onto an intuitive similarity scale
    without requiring callers to reason about distance metrics.

    Args:
        distance: Raw distance score returned by FAISS.

    Returns:
        A similarity score where 1.0 is a perfect match and values
        approach 0.0 as distance grows.
    """
    similarity = 1.0 / (1.0 + max(distance, 0.0))
    return max(MIN_SIMILARITY_THRESHOLD, min(MAX_SIMILARITY_THRESHOLD, similarity))


@timed
def retrieve_relevant_chunks(
    vector_store: FAISS,
    query: str,
    top_k: int | None = None,
    similarity_threshold: float = 0.0,
    metadata_filter: dict[str, Any] | None = None,
) -> list[RetrievedChunk]:
    """Retrieve the most relevant chunks for a query from the vector store.

    Args:
        vector_store: The FAISS store to search.
        query: The user's natural-language question.
        top_k: Number of chunks to retrieve. Defaults to `settings.top_k`.
        similarity_threshold: Minimum similarity score (0.0-1.0) a chunk
            must meet to be included. Chunks below this are discarded.
        metadata_filter: Optional exact-match metadata filter, e.g.
            `{"source_file": "handbook.pdf"}`, applied by FAISS at
            search time.

    Returns:
        A list of `RetrievedChunk` objects, ordered from most to least
        relevant. Empty if the query is blank or nothing meets the
        similarity threshold.
    """
    if not query or not query.strip():
        logger.warning("retrieve_relevant_chunks called with an empty query")
        return []

    effective_top_k = top_k or settings.top_k
    logger.info(
        "Retrieving top %d chunk(s) for query: %r (threshold=%.2f, filter=%s)",
        effective_top_k,
        query,
        similarity_threshold,
        metadata_filter,
    )

    raw_results = vector_store.similarity_search_with_score(
        query=query,
        k=effective_top_k,
        filter=metadata_filter,
    )

    retrieved_chunks: list[RetrievedChunk] = []
    for document, distance in raw_results:
        similarity = _distance_to_similarity(distance)
        if similarity < similarity_threshold:
            continue
        retrieved_chunks.append(
            RetrievedChunk(
                content=document.page_content,
                similarity_score=round(similarity, 4),
                source_file=document.metadata.get("source_file", "unknown"),
                metadata=document.metadata,
            )
        )

    logger.info(
        "Retrieved %d chunk(s) above similarity threshold %.2f",
        len(retrieved_chunks),
        similarity_threshold,
    )
    return retrieved_chunks
