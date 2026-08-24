"""Lightweight per-session todo list: lets the agent plan and track multi-step work.

The agent replaces its task list wholesale via the `todo_write` tool
(list of {content, status}), and reads it back with `todo_read`.
Lists are ephemeral working state (in-memory per session), not durable memory.
"""

from dataclasses import dataclass

TODO_STATUSES = ("pending", "in_progress", "completed")

_STATUS_MARKS = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}


@dataclass
class TodoItem:
    content: str
    status: str = "pending"  # one of TODO_STATUSES


class TodoManager:
    """Holds one task list per session id; replaced wholesale on each todo_write."""

    def __init__(self) -> None:
        self._lists: dict[str, list[TodoItem]] = {}

    def write(self, session_id: str, todos: list[dict]) -> list[TodoItem]:
        """Replace the session's todo list with `todos` (each {content, status?})."""
        items: list[TodoItem] = []
        for raw in todos:
            content = str(raw.get("content", "")).strip()
            if not content:
                raise ValueError("each todo needs a non-empty 'content'")
            status = str(raw.get("status", "pending"))
            if status not in TODO_STATUSES:
                raise ValueError(
                    f"unknown status {status!r}; expected one of {TODO_STATUSES}"
                )
            items.append(TodoItem(content=content, status=status))
        self._lists[session_id] = items
        return items

    def read(self, session_id: str) -> list[TodoItem]:
        """Return the session's current todo list (empty list if never written)."""
        return list(self._lists.get(session_id, []))

    def render(self, session_id: str) -> str:
        """Render the list as checkbox lines, e.g. '[x] step one' — for the todo_read tool."""
        items = self._lists.get(session_id, [])
        if not items:
            return "no todos yet"
        return "\n".join(f"{_STATUS_MARKS[i.status]} {i.content}" for i in items)
