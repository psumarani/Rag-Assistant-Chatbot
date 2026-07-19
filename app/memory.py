"""Session conversation memory module.

Sole responsibility: hold and manage the in-memory conversation history
for the current Streamlit session. This is intentionally simple (a
plain list) since persistence across app restarts is not required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from app.retriever import RetrievedChunk


@dataclass(frozen=True)
class ConversationTurn:
    """A single question/answer exchange in the conversation history.

    Attributes:
        question: The user's question.
        answer: The generated answer.
        sources: The chunks that were retrieved and used to ground the answer.
        timestamp: UTC time the turn was created.
    """

    question: str
    answer: str
    sources: list[RetrievedChunk] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ConversationMemory:
    """Manages the ordered list of conversation turns for a session."""

    def __init__(self) -> None:
        self._turns: list[ConversationTurn] = []

    def add_turn(
        self, question: str, answer: str, sources: list[RetrievedChunk] | None = None
    ) -> None:
        """Append a new question/answer turn to the history.

        Args:
            question: The user's question.
            answer: The generated answer.
            sources: Chunks used to ground the answer, if any.
        """
        self._turns.append(
            ConversationTurn(question=question, answer=answer, sources=sources or [])
        )

    @property
    def turns(self) -> list[ConversationTurn]:
        """Return all conversation turns in chronological order."""
        return list(self._turns)

    def clear(self) -> None:
        """Remove all conversation turns, resetting the chat history."""
        self._turns.clear()

    def is_empty(self) -> bool:
        """Return True if no turns have been recorded yet."""
        return len(self._turns) == 0

    def __len__(self) -> int:
        return len(self._turns)
