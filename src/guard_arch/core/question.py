"""Pending user questions: pause-and-resume interaction for ask_user_question.

When the agent calls ask_user_question mid-run, the tool suspends the run on an
asyncio.Future until the user's answer arrives (via `answer()` from an API layer
or an interactive handler from the CLI), then the run resumes with the answer.

Supports structured questions: multiple questions, each with selectable options,
single/multi select mode, and custom answer support.
"""

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from typing import Any

# 等待用户回答的超时：超时后 agent 自行决定下一步（不等死）
QUESTION_TIMEOUT_SECONDS = 300.0


@dataclass
class QuestionItem:
    """A single structured question with options."""

    question: str
    options: list[str] = field(default_factory=list)
    question_type: str = "single"  # "single" | "multi"


@dataclass
class PendingQuestion:
    """A pending question set waiting for user answer."""

    question_id: str
    questions: list[QuestionItem]
    future: asyncio.Future


class QuestionManager:
    """Tracks at most one pending question per session."""

    def __init__(self) -> None:
        self._pending: dict[str, PendingQuestion] = {}

    def create(
        self,
        session_id: str,
        questions: list[dict[str, Any]] | None = None,
    ) -> tuple[str, asyncio.Future]:
        """Register a new pending question for the session; returns (question_id, future).

        questions: list of {question: str, options: list[str], question_type: str}
        """
        question_id = f"q-{uuid.uuid4().hex[:8]}"
        future: asyncio.Future = asyncio.get_event_loop().create_future()

        items: list[QuestionItem] = []
        if questions:
            for q in questions:
                items.append(
                    QuestionItem(
                        question=q.get("question", ""),
                        options=q.get("options", []),
                        question_type=q.get("question_type", "single"),
                    )
                )

        self._pending[session_id] = PendingQuestion(
            question_id=question_id,
            questions=items,
            future=future,
        )
        return question_id, future

    def has_pending(self, session_id: str) -> bool:
        entry = self._pending.get(session_id)
        return entry is not None and not entry.future.done()

    def answer(self, session_id: str, answer: str) -> bool:
        """Resolve the session's pending question with the user's answer.

        Returns True if a pending question was resolved, False otherwise.
        """
        entry = self._pending.pop(session_id, None)
        if entry is None:
            return False
        if entry.future.done():
            return False
        entry.future.set_result(answer)
        return True

    def cancel(self, session_id: str) -> None:
        """Drop the session's pending question without resolving (e.g. on timeout)."""
        self._pending.pop(session_id, None)
