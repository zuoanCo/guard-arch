"""Four-layer memory: conversation / user / project / agent, SQLite-backed."""

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MEMORY_LAYERS = ("conversation", "user", "project", "agent")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversation (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS session_state (
    session_id TEXT PRIMARY KEY,
    messages_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS kv (
    layer TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (layer, key)
);
"""


class MemoryManager:
    """Persists memory to `<workspace>/.guard_arch/memory.db`."""

    def __init__(self, workspace_root: str | Path):
        self.dir = Path(workspace_root) / ".guard_arch"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.dir / "memory.db"
        self._conn = sqlite3.connect(self.db_path)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # -- conversation layer -------------------------------------------------

    def add_message(self, session_id: str, role: str, content: str) -> None:
        self._conn.execute(
            "INSERT INTO conversation (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (session_id, role, content, datetime.now(UTC).isoformat()),
        )
        self._conn.commit()

    def recent_messages(self, session_id: str, limit: int = 20) -> list[dict[str, str]]:
        rows = self._conn.execute(
            "SELECT role, content FROM conversation WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
        return [{"role": role, "content": content} for role, content in reversed(rows)]

    def save_conversation_state(self, session_id: str, messages: Any) -> None:
        """Persist serialized pydantic-ai ModelMessages for a session."""
        payload = json.dumps(messages, ensure_ascii=False, default=str)
        self._conn.execute(
            "INSERT INTO session_state (session_id, messages_json, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(session_id) DO UPDATE SET messages_json = excluded.messages_json, "
            "updated_at = excluded.updated_at",
            (session_id, payload, datetime.now(UTC).isoformat()),
        )
        self._conn.commit()

    def load_conversation_state(self, session_id: str) -> Any | None:
        row = self._conn.execute(
            "SELECT messages_json FROM session_state WHERE session_id = ?", (session_id,)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def clear_conversation(self, session_id: str) -> None:
        self._conn.execute("DELETE FROM conversation WHERE session_id = ?", (session_id,))
        self._conn.execute("DELETE FROM session_state WHERE session_id = ?", (session_id,))
        self._conn.commit()

    # -- kv layers (user / project / agent) ----------------------------------

    def remember(self, layer: str, key: str, value: str) -> None:
        if layer not in MEMORY_LAYERS:
            raise ValueError(f"unknown memory layer {layer!r}; expected one of {MEMORY_LAYERS}")
        self._conn.execute(
            "INSERT INTO kv (layer, key, value, updated_at) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(layer, key) DO UPDATE SET value = excluded.value, "
            "updated_at = excluded.updated_at",
            (layer, key, value, datetime.now(UTC).isoformat()),
        )
        self._conn.commit()

    def recall(self, layer: str, key: str | None = None) -> dict[str, str]:
        if key is not None:
            row = self._conn.execute(
                "SELECT value FROM kv WHERE layer = ? AND key = ?", (layer, key)
            ).fetchone()
            return {key: row[0]} if row else {}
        rows = self._conn.execute(
            "SELECT key, value FROM kv WHERE layer = ? ORDER BY key", (layer,)
        ).fetchall()
        return dict(rows)

    def search(
        self, query: str, layer: str | None = None, limit: int = 10
    ) -> dict[str, dict[str, str]]:
        """Keyword search over the kv memory layers (user/project/agent).

        Matches entries whose key OR value contains `query` (SQL LIKE, case-insensitive
        for ASCII). Returns {layer: {key: value}} for matching entries, most recently
        updated first, capped at `limit` per layer. Empty dict when nothing matches.
        """
        layers = (layer,) if layer else ("user", "project", "agent")
        like = f"%{query}%"
        results: dict[str, dict[str, str]] = {}
        for name in layers:
            rows = self._conn.execute(
                "SELECT key, value FROM kv WHERE layer = ? "
                "AND (key LIKE ? OR value LIKE ?) "
                "ORDER BY updated_at DESC LIMIT ?",
                (name, like, like, limit),
            ).fetchall()
            if rows:
                results[name] = dict(rows)
        return results

    def context_snippet(self, max_items_per_layer: int = 10) -> str:
        """Render user/project/agent memory for injection into the system prompt."""
        sections: list[str] = []
        for layer in ("user", "project", "agent"):
            items = self.recall(layer)
            if not items:
                continue
            lines = [f"[{layer} memory]"]
            for key, value in list(items.items())[:max_items_per_layer]:
                lines.append(f"- {key}: {value}")
            sections.append("\n".join(lines))
        return "\n\n".join(sections)
