"""Run bookkeeping: one Run per agent task execution."""

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class RunStatus(StrEnum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass
class RunEventRecord:
    type: str
    data: dict[str, Any]
    timestamp: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )


@dataclass
class Run:
    id: str
    agent_id: str
    session_id: str
    status: RunStatus = RunStatus.RUNNING
    started_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    finished_at: str | None = None
    output: str | None = None
    error: str | None = None
    events: list[RunEventRecord] = field(default_factory=list)


class RunManager:
    def __init__(self) -> None:
        self.runs: dict[str, Run] = {}

    def start(self, agent_id: str, session_id: str) -> Run:
        run = Run(id=f"run-{uuid.uuid4().hex[:8]}", agent_id=agent_id, session_id=session_id)
        self.runs[run.id] = run
        return run

    def record_event(self, run: Run, event_type: str, data: dict[str, Any]) -> None:
        run.events.append(RunEventRecord(event_type, data))

    def finish(self, run: Run, status: RunStatus, output: str | None = None, error: str | None = None) -> Run:
        run.status = status
        run.output = output
        run.error = error
        run.finished_at = datetime.now(UTC).isoformat()
        return run
