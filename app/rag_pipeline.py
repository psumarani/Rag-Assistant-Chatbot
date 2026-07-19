"""RAG pipeline orchestration module.

Sole responsibility: wire together document loading, chunking, vector
storage, retrieval, and generation into the two operations the UI
needs — building the knowledge base and answering a question. This is
the only module that should import from every other `app.*` module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from app.chunking import split_documents
from app.document_loader import load_documents
from app.llm import generate_answer
from app.logger import get_logger
from app.prompts import build_rag_prompt
from app.retriever import RetrievedChunk, retrieve_relevant_chunks
from app.utils import timed
from app.vector_store import (
    VectorStoreNotFoundError,
    delete_vector_store,
    load_or_create_vector_store,
    load_vector_store,
    vector_store_exists,
)
from config.constants import NO_ANSWER_FOUND_MESSAGE
from config.settings import settings

logger = get_logger(__name__)


@dataclass(frozen=True)
class KnowledgeBaseStats:
    """Summary statistics after building or updating the knowledge base.

    Attributes:
        files_processed: Number of files that were successfully loaded.
        document_sections: Number of loaded Document sections (e.g. PDF pages).
        chunks_created: Number of chunks created from those sections.
    """

    files_processed: int
    document_sections: int
    chunks_created: int


@dataclass(frozen=True)
class QueryResult:
    """The full result of answering a question against the knowledge base.

    Attributes:
        answer: The generated answer text.
        sources: The chunks retrieved and used to ground the answer.
        found_answer: False if the model reported it could not find the
            answer in the retrieved context.
    """

    answer: str
    sources: list[RetrievedChunk] = field(default_factory=list)
    found_answer: bool = True


class RAGPipeline:
    """High-level façade over the full RAG workflow.

    Holds no long-lived state beyond the vector store persistence
    directory, so it is safe to instantiate fresh per Streamlit session.
    """

    def __init__(self, persist_dir: Path | None = None) -> None:
        self._persist_dir = persist_dir or settings.vector_store_dir

    def has_knowledge_base(self) -> bool:
        """Return True if a persisted vector store already exists."""
        return vector_store_exists(self._persist_dir)

    @timed
    def build_knowledge_base(self, file_paths: list[Path]) -> KnowledgeBaseStats:
        """Load, chunk, embed, and index a batch of uploaded documents.

        New documents are added incrementally if a knowledge base
        already exists, rather than rebuilding it from scratch.

        Args:
            file_paths: Paths to the uploaded source files.

        Returns:
            Summary statistics about what was processed.

        Raises:
            EmptyVectorStoreError: If every file failed to load, leaving
                zero chunks to index.
        """
        documents = load_documents(file_paths)
        chunks = split_documents(documents)

        load_or_create_vector_store(chunks, self._persist_dir)

        stats = KnowledgeBaseStats(
            files_processed=len(file_paths),
            document_sections=len(documents),
            chunks_created=len(chunks),
        )
        logger.info("Knowledge base build complete: %s", stats)
        return stats

    @timed
    def answer_question(
        self,
        question: str,
        top_k: int | None = None,
        similarity_threshold: float = 0.0,
    ) -> QueryResult:
        """Answer a question using only the indexed documents.

        Args:
            question: The user's natural-language question.
            top_k: Number of chunks to retrieve. Defaults to `settings.top_k`.
            similarity_threshold: Minimum similarity score for a chunk
                to be considered relevant.

        Returns:
            A `QueryResult` with the generated answer and its sources.

        Raises:
            VectorStoreNotFoundError: If no knowledge base has been built yet.
            GeminiGenerationError: If the Gemini API call fails.
            MissingAPIKeyError: If GOOGLE_API_KEY is not configured.
        """
        if not self.has_knowledge_base():
            raise VectorStoreNotFoundError(
                "No knowledge base found. Upload documents and click "
                "'Build Knowledge Base' before asking questions."
            )

        vector_store = load_vector_store(self._persist_dir)
        chunks = retrieve_relevant_chunks(
            vector_store=vector_store,
            query=question,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
        )

        if not chunks:
            logger.info("No chunks met the similarity threshold for the query")
            return QueryResult(answer=NO_ANSWER_FOUND_MESSAGE, sources=[], found_answer=False)

        messages = build_rag_prompt(question, chunks)
        answer = generate_answer(messages)
        found_answer = NO_ANSWER_FOUND_MESSAGE not in answer

        return QueryResult(answer=answer, sources=chunks, found_answer=found_answer)

    def reset_knowledge_base(self) -> None:
        """Delete the persisted vector store, clearing all indexed documents."""
        delete_vector_store(self._persist_dir)
        logger.info("Knowledge base reset")
