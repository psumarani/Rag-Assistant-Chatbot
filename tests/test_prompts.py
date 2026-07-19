"""Unit tests for app.prompts."""

from __future__ import annotations

from app.prompts import build_rag_prompt, format_context
from app.retriever import RetrievedChunk
from config.constants import NO_ANSWER_FOUND_MESSAGE


def test_format_context_returns_marker_for_empty_chunks() -> None:
    result = format_context([])
    assert "No relevant context" in result


def test_format_context_includes_source_and_score() -> None:
    chunks = [
        RetrievedChunk(content="Some text.", similarity_score=0.87, source_file="doc.txt")
    ]
    result = format_context(chunks)
    assert "doc.txt" in result
    assert "0.87" in result
    assert "Some text." in result


def test_build_rag_prompt_embeds_question_and_context() -> None:
    chunks = [
        RetrievedChunk(content="Relevant fact.", similarity_score=0.9, source_file="doc.txt")
    ]
    messages = build_rag_prompt("What is the fact?", chunks)

    assert any("What is the fact?" in message["content"] for message in messages)
    assert any(NO_ANSWER_FOUND_MESSAGE in message["content"] for message in messages)
