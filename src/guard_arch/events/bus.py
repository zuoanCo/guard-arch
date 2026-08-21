"""Event bus: decouples the runtime from the UI layer."""

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass
class Event:
    type: str
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


Subscriber = Callable[[Event], Any]


class EventBus:
    """Sync + async subscribers; '*' subscribes to every event."""

    def __init__(self) -> None:
        self._subscribers: dict[str, list[Subscriber]] = {}

    def subscribe(self, event_type: str, callback: Subscriber) -> None:
        self._subscribers.setdefault(event_type, []).append(callback)

    def unsubscribe(self, event_type: str, callback: Subscriber) -> None:
        if callback in self._subscribers.get(event_type, []):
            self._subscribers[event_type].remove(callback)

    async def emit(self, event: Event) -> None:
        callbacks = self._subscribers.get(event.type, []) + self._subscribers.get("*", [])
        for callback in callbacks:
            result = callback(event)
            if inspect.isawaitable(result):
                await result
