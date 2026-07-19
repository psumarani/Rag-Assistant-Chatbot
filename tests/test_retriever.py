"""Unit tests for app.retriever."""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

from app.retriever import retrieve_relevant_chunks
from app.vector_store import create_vector_store


def test_retrieve_relevant_chunks_returns_structured_results(tmp_path: Path) -> None:
    chunks = [
        Document(
            page_content="Python is a popular programming language.",
            metadata={"source_file": "lang.txt"},
        ),
        Document(
            page_content="Bananas are a good source of potassium.",
            metadata={"source_file": "fruit.txt"},
        ),
    ]
    store = create_vector_store(chunks, tmp_path / "store")

    results = retrieve_relevant_chunks(store, "Tell me about Python programming", top_k=1)

    assert len(results) == 1
    assert results[0].source_file == "lang.txt"
    assert 0.0 <= results[0].similarity_score <= 1.0


def test_retrieve_relevant_chunks_returns_empty_for_blank_query(tmp_path: Path) -> None:
    chunks = [Document(page_content="Some content.", metadata={"source_file": "a.txt"})]
    store = create_vector_store(chunks, tmp_path / "store")

    results = retrieve_relevant_chunks(store, "   ")

    assert results == []


def test_retrieve_relevant_chunks_respects_similarity_threshold(tmp_path: Path) -> None:
    chunks = [Document(page_content="Completely unrelated topic.", metadata={"source_file": "a.txt"})]
    store = create_vector_store(chunks, tmp_path / "store")

    results = retrieve_relevant_chunks(
        store, "quantum physics equations", top_k=1, similarity_threshold=0.99
    )

    assert results == []
