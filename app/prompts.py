"""Prompt templates module.

Sole responsibility: define the prompts sent to Gemini, and enforce the
project's core rule at the prompt level — never answer from outside
the retrieved context.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from app.retriever import RetrievedChunk
from config.constants import NO_ANSWER_FOUND_MESSAGE

SYSTEM_PROMPT = f"""You are a precise, careful document assistant.

STRICT RULES — follow these without exception:
1. Answer ONLY using the information present in the "Context" section below.
2. Never use outside knowledge, assumptions, or information not present \
in the context, even if you are confident it is correct.
3. If the context does not contain enough information to answer the \
question, respond with exactly: "{NO_ANSWER_FOUND_MESSAGE}"
4. When you do answer, mention which source document(s) the information \
came from.
5. Be concise and directly answer the question — do not pad your \
response with unnecessary commentary."""

_HUMAN_TEMPLATE = """Context:
{context}

Question:
{question}

Answer the question using only the context above, and mention the \
source document(s) you used."""

RAG_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", _HUMAN_TEMPLATE),
    ]
)


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Assemble retrieved chunks into a single labeled context string.

    Args:
        chunks: Retrieved chunks, ordered from most to least relevant.

    Returns:
        A formatted string with each chunk labeled by its source file
        and similarity score, ready to insert into `RAG_PROMPT_TEMPLATE`.
        Returns an empty-context marker if `chunks` is empty.
    """
    if not chunks:
        return "No relevant context was found in the uploaded documents."

    formatted_sections = [
        f"[Source: {chunk.source_file} | Similarity: {chunk.similarity_score:.2f}]\n"
        f"{chunk.content}"
        for chunk in chunks
    ]
    return "\n\n---\n\n".join(formatted_sections)


def build_rag_prompt(question: str, chunks: list[RetrievedChunk]) -> list[dict[str, str]]:
    """Build the final message list to send to the Gemini chat model.

    Args:
        question: The user's natural-language question.
        chunks: Retrieved chunks to use as grounding context.

    Returns:
        A list of role/content message dicts formatted from
        `RAG_PROMPT_TEMPLATE`, ready to pass to the LLM.
    """
    context = format_context(chunks)
    messages = RAG_PROMPT_TEMPLATE.format_messages(context=context, question=question)
    return [{"role": message.type, "content": message.content} for message in messages]
