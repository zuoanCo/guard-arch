"""Pending user questions: pause-and-resume interaction for ask_user_question.

When the agent calls ask_user_question mid-run, the tool suspends the run on an
asyncio.Future until the user's answer arrives (via `answer()` from an API layer
or an interactive handler from the CLI), then the run resumes with the answer.
"""

import asyncio
import uuid

# 等待用户回答的超时：超时后工具返回错误文本，agent 自行决定下一步（不等死）
QUESTION_TIMEOUT_SECONDS = 300.0


class QuestionManager:
    """Tracks at most one pending question per session."""

    def __init__(self) -> None:
        # session_id -> (question_id, Future[str])
        self._pending: dict[str, tuple[str, asyncio.Future]] = {}

    def create(self, session_id: str) -> tuple[str, asyncio.Future]:
        """Register a new pending question for the session; returns (question_id, future)."""
        question_id = f"q-{uuid.uuid4().hex[:8]}"
        future: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending[session_id] = (question_id, future)
        return question_id, future

    def has_pending(self, session_id: str) -> bool:
        entry = self._pending.get(session_id)
        return entry is not None and not entry[1].done()

    def answer(self, session_id: str, answer: str) -> bool:
        """Resolve the session's pending question with the user's answer.

        Returns True if a pending question was resolved, False otherwise.
        """
        entry = self._pending.pop(session_id, None)
        if entry is None:
            return False
        _question_id, future = entry
        if future.done():
            return False
        future.set_result(answer)
        return True

    def cancel(self, session_id: str) -> None:
        """Drop the session's pending question without resolving (e.g. on timeout)."""
        self._pending.pop(session_id, None)
