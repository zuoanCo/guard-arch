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


class AnswerRequest(BaseModel):
    """对挂起中的 ask_user_question 的回答（注入后原 run 继续执行）。"""

    session_id: str
    answer: str
    workspace: str | None = None


class MemoryWriteRequest(BaseModel):
    """写入一条长期记忆（layer: user/project/agent）。"""

    layer: str
    key: str
    value: str
    workspace: str | None = None


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

    @app.get("/api/v1/tools")
    async def list_tools(workspace: str | None = None) -> list[dict]:
        """工具清单（名称/描述/来源），来自默认工作区的 ToolRegistry。"""
        runtime = server.get_runtime(workspace, auto_approve=False)
        return [
            {"name": t.name, "description": t.description, "source": t.source}
            for t in runtime.tool_registry.all()
        ]

    @app.get("/api/v1/rules")
    async def list_rules(workspace: str | None = None) -> dict:
        """核心行为规则清单（harness 宪法：代码持有默认值，rules.yaml 可控）。"""
        runtime = server.get_runtime(workspace, auto_approve=False)
        return {
            "rules": [
                {"id": r.id, "text": r.text, "enabled": r.enabled}
                for r in runtime.rules_registry.all()
            ]
        }

    @app.get("/api/v1/models")
    async def list_models(workspace: str | None = None) -> dict:
        """可用模型角色列表（ModelRouter 配置的角色名）。"""
        runtime = server.get_runtime(workspace, auto_approve=False)
        return {"roles": runtime.model_router.role_names()}

    @app.post("/api/v1/sessions")
    async def create_session(req: CreateSessionRequest) -> dict:
        runtime = server.get_runtime(req.workspace, auto_approve=False)
        session_id = f"s-{uuid.uuid4().hex[:8]}"
        server.sessions[session_id] = str(runtime.workspace.root)
        return {"session_id": session_id, "workspace": str(runtime.workspace.root)}

    @app.get("/api/v1/sessions")
    async def list_sessions(like: str | None = None, workspace: str | None = None) -> dict:
        """会话列表（最近活跃排序，含条数/预览；like 可按前缀过滤归属）。"""
        runtime = server.get_runtime(workspace, auto_approve=False)
        return {"sessions": runtime.memory.list_sessions(like=like)}

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

    @app.delete("/api/v1/sessions/{session_id}")
    async def delete_session(session_id: str) -> dict:
        """删除会话（清空对话历史与模型状态）。"""
        workspace = server.sessions.get(session_id)
        if workspace is None:
            raise HTTPException(status_code=404, detail=f"unknown session: {session_id}")
        runtime = server.get_runtime(workspace, auto_approve=False)
        runtime.memory.clear_conversation(session_id)
        server.sessions.pop(session_id, None)
        return {"deleted": session_id}

    @app.get("/api/v1/sessions/{session_id}/todos")
    async def get_todos(session_id: str) -> dict:
        """会话的任务清单（todo_write 写入的列表）。"""
        workspace = server.sessions.get(session_id)
        if workspace is None:
            raise HTTPException(status_code=404, detail=f"unknown session: {session_id}")
        runtime = server.get_runtime(workspace, auto_approve=False)
        todos = runtime.todo_manager.read(session_id)
        return {
            "session_id": session_id,
            "todos": [{"content": t.content, "status": t.status} for t in todos],
        }

    @app.get("/api/v1/memory")
    async def read_memory(
        layer: str, key: str | None = None, workspace: str | None = None
    ) -> dict:
        """读取长期记忆（layer=user/project/agent；key 缺省返回该层全部）。"""
        runtime = server.get_runtime(workspace, auto_approve=False)
        return {"layer": layer, "items": runtime.memory.recall(layer, key)}

    @app.post("/api/v1/memory")
    async def write_memory(req: MemoryWriteRequest) -> dict:
        """写入一条长期记忆。"""
        runtime = server.get_runtime(req.workspace, auto_approve=False)
        try:
            runtime.memory.remember(req.layer, req.key, req.value)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"remembered": {"layer": req.layer, "key": req.key}}

    @app.delete("/api/v1/memory/{layer}/{key}")
    async def delete_memory(layer: str, key: str, workspace: str | None = None) -> dict:
        """删除一条长期记忆。"""
        runtime = server.get_runtime(workspace, auto_approve=False)
        removed = runtime.memory.forget(layer, key)
        return {"deleted": removed}

    @app.get("/api/v1/runs")
    async def list_runs(workspace: str | None = None) -> dict:
        """最近的 run 记录（状态/agent/session/事件数）。"""
        runtime = server.get_runtime(workspace, auto_approve=False)
        runs = list(runtime.run_manager.runs.values())[-20:]
        return {
            "runs": [
                {
                    "id": r.id,
                    "agent_id": r.agent_id,
                    "session_id": r.session_id,
                    "status": str(r.status),
                    "started_at": r.started_at,
                    "finished_at": r.finished_at,
                    "event_count": len(r.events),
                }
                for r in runs
            ]
        }

    @app.get("/api/v1/runs/{run_id}")
    async def get_run(run_id: str, workspace: str | None = None) -> dict:
        """run 详情（含全部事件记录）。"""
        runtime = server.get_runtime(workspace, auto_approve=False)
        run = runtime.run_manager.runs.get(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"unknown run: {run_id}")
        return {
            "id": run.id,
            "agent_id": run.agent_id,
            "session_id": run.session_id,
            "status": str(run.status),
            "started_at": run.started_at,
            "finished_at": run.finished_at,
            "output": run.output,
            "error": run.error,
            "events": [{"type": e.type, "data": e.data, "timestamp": e.timestamp} for e in run.events],
        }

    @app.post("/api/v1/chat/answer")
    async def answer_question(req: AnswerRequest) -> dict:
        """提交对挂起提问的回答：注入后原 run 继续执行（resumed=false 表示无挂起问题）。"""
        runtime = server.get_runtime(req.workspace, auto_approve=False)
        resumed = runtime.answer_question(req.session_id, req.answer)
        return {"resumed": resumed}

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
