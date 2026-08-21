"""FastAPI service layer over AgentRuntime.

Exposes the agent runtime to any HTTP frontend (web / mobile). Chat is
streamed as Server-Sent Events translated 1:1 from EventBus events.

Threading model: everything runs on the uvicorn event loop. The runtime is
async, so each /chat request spawns the agent run as an asyncio.Task on the
same loop and streams events through an asyncio.Queue. Multiple concurrent
SSE sessions share the per-workspace runtime; subscribers filter events by
session_id so sessions never see each other's events.
"""

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from guard_arch import PROJECT_ROOT
from guard_arch.core.agent import AgentRegistry
from guard_arch.core.skill import SkillRegistry
from guard_arch.events.bus import Event
from guard_arch.runtime import AgentRuntime

logger = logging.getLogger(__name__)

TERMINAL_EVENTS = ("agent_finished", "error")


class CreateSessionRequest(BaseModel):
    workspace: str | None = None


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    agent: str = "assistant"
    workspace: str | None = None
    model: str | None = None
    auto_approve: bool = False


def format_sse(event_type: str, data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=False, default=str)
    return f"event: {event_type}\ndata: {payload}\n\n"


class APIServer:
    """Holds cross-request state: session -> workspace map and runtime cache."""

    def __init__(self) -> None:
        self.agent_registry = AgentRegistry([PROJECT_ROOT / "agents"])
        self.skill_registry = SkillRegistry([PROJECT_ROOT / "skills"])
        self.sessions: dict[str, str] = {}  # session_id -> workspace root
        self.runtimes: dict[tuple[str, bool], AgentRuntime] = {}

    def get_runtime(self, workspace: str | None, auto_approve: bool) -> AgentRuntime:
        root = str(Path(workspace or ".").resolve())
        key = (root, auto_approve)
        if key not in self.runtimes:
            # API mode has no interactive approval callback: ASK resolves to
            # deny (the permission_required event is still emitted so clients
            # can surface it). auto_approve is a development convenience;
            # production should wire a real confirmation callback instead.
            # DENY rules (e.g. rm -rf) are never bypassed by auto_approve.
            self.runtimes[key] = AgentRuntime(root, auto_approve=auto_approve)
        return self.runtimes[key]


def create_app() -> FastAPI:
    app = FastAPI(title="Guard Arch API")
    # Development: wide-open CORS. Restrict origins for production.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    server = APIServer()

    @app.get("/health")
    async def health() -> dict:
        return {"ok": True}

    @app.get("/api/v1/agents")
    async def list_agents() -> list[dict]:
        return [
            {
                "id": a.id,
                "name": a.name,
                "model": a.model,
                "skills": a.skills,
                "tools": a.tools,
            }
            for a in server.agent_registry.all()
        ]

    @app.get("/api/v1/skills")
    async def list_skills() -> list[dict]:
        return [
            {"name": s.name, "description": s.description, "tools": s.tools}
            for s in server.skill_registry.all()
        ]

    @app.post("/api/v1/sessions")
    async def create_session(req: CreateSessionRequest) -> dict:
        runtime = server.get_runtime(req.workspace, auto_approve=False)
        session_id = f"s-{uuid.uuid4().hex[:8]}"
        server.sessions[session_id] = str(runtime.workspace.root)
        return {"session_id": session_id, "workspace": str(runtime.workspace.root)}

    @app.get("/api/v1/sessions/{session_id}/messages")
    async def get_messages(session_id: str) -> dict:
        workspace = server.sessions.get(session_id)
        if workspace is None:
            raise HTTPException(status_code=404, detail=f"unknown session: {session_id}")
        runtime = server.get_runtime(workspace, auto_approve=False)
        return {
            "session_id": session_id,
            "messages": runtime.memory.recent_messages(session_id, limit=1000),
        }

    @app.post("/api/v1/chat")
    async def chat(req: ChatRequest) -> StreamingResponse:
        session_id = req.session_id or f"s-{uuid.uuid4().hex[:8]}"
        workspace = req.workspace or server.sessions.get(session_id)
        try:
            runtime = server.get_runtime(workspace, req.auto_approve)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        server.sessions[session_id] = str(runtime.workspace.root)

        queue: asyncio.Queue[Event] = asyncio.Queue()

        def on_event(event: Event) -> None:
            # Same event loop as the request; filter out concurrent sessions.
            if event.data.get("session_id") == session_id:
                queue.put_nowait(event)

        runtime.bus.subscribe("*", on_event)

        async def stream() -> AsyncIterator[str]:
            run_task = asyncio.create_task(
                runtime.run(
                    req.message,
                    agent_id=req.agent,
                    session_id=session_id,
                    model_role=req.model,
                )
            )
            try:
                while True:
                    event = await queue.get()
                    yield format_sse(event.type, event.data)
                    if event.type in TERMINAL_EVENTS:
                        break
                await run_task  # surface unexpected failures to logs
            except asyncio.CancelledError:
                run_task.cancel()
                raise
            finally:
                runtime.bus.unsubscribe("*", on_event)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run("guard_arch.api.app:app", host="127.0.0.1", port=8100)


if __name__ == "__main__":
    main()
