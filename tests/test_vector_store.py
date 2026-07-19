"""Unit tests for app.vector_store."""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.documents import Document

from app.vector_store import (
    EmptyVectorStoreError,
    VectorStoreNotFoundError,
    create_vector_store,
    load_vector_store,
    update_vector_store,
    vector_store_exists,
)


@pytest.fixture
def sample_chunks() -> list[Document]:
    return [
        Document(page_content="The cat sat on the mat.", metadata={"source_file": "a.txt"}),
        Document(page_content="Dogs are loyal animals.", metadata={"source_file": "a.txt"}),
    ]


def test_create_vector_store_raises_for_empty_chunks(tmp_path: Path) -> None:
    with pytest.raises(EmptyVectorStoreError):
        create_vector_store([], tmp_path / "store")


def test_create_and_load_vector_store_roundtrip(tmp_path: Path, sample_chunks) -> None:
    persist_dir = tmp_path / "store"

    create_vector_store(sample_chunks, persist_dir)
    assert vector_store_exists(persist_dir)

    loaded_store = load_vector_store(persist_dir)
    results = loaded_store.similarity_search("cat", k=1)
    assert len(results) == 1


def test_load_vector_store_raises_when_missing(tmp_path: Path) -> None:
    with pytest.raises(VectorStoreNotFoundError):
        load_vector_store(tmp_path / "does_not_exist")


def test_update_vector_store_adds_new_chunks(tmp_path: Path, sample_chunks) -> None:
    persist_dir = tmp_path / "store"
    store = create_vector_store(sample_chunks, persist_dir)

    new_chunk = [Document(page_content="Birds can fly.", metadata={"source_file": "b.txt"})]
    updated_store = update_vector_store(store, new_chunk, persist_dir)

    results = updated_store.similarity_search("fly", k=1)
    assert "Birds" in results[0].page_content
