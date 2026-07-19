"""Unit tests for app.chunking."""

from __future__ import annotations

from langchain_core.documents import Document

from app.chunking import split_documents


def test_split_documents_produces_chunks_with_metadata() -> None:
    long_text = "sentence. " * 500
    documents = [Document(page_content=long_text, metadata={"source_file": "a.txt"})]

    chunks = split_documents(documents, chunk_size=200, chunk_overlap=20)

    assert len(chunks) > 1
    for index, chunk in enumerate(chunks):
        assert chunk.metadata["chunk_index"] == index
        assert "chunk_id" in chunk.metadata
        assert chunk.metadata["source_file"] == "a.txt"
        assert len(chunk.page_content) <= 200 + 20


def test_split_documents_handles_empty_list() -> None:
    assert split_documents([]) == []
