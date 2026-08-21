"""AgentRuntime: composes registries, memory, permissions, sandbox and the
event bus, and drives the pydantic-ai agent loop."""

import functools
import inspect
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessagesTypeAdapter,
    PartDeltaEvent,
    TextPartDelta,
)
from pydantic_ai.models import Model

from guard_arch import PROJECT_ROOT
from guard_arch.core.agent import AgentDefinition, AgentRegistry
from guard_arch.core.context import ContextEngine
from guard_arch.core.memory import MemoryManager
from guard_arch.core.model import ModelRouter
from guard_arch.core.run import Run, RunManager, RunStatus
from guard_arch.core.skill import SkillManifest, SkillRegistry
from guard_arch.core.tool import Tool, ToolRegistry
from guard_arch.core.workspace import Workspace
from guard_arch.events.bus import Event, EventBus
from guard_arch.mcp.client import MCPLoader
from guard_arch.permissions.engine import PermissionDecision, PermissionEngine
from guard_arch.tools.filesystem import make_filesystem_tools
from guard_arch.tools.terminal import make_terminal_tools

logger = logging.getLogger(__name__)


@dataclass
class RunResult:
    output: str
    run: Run
    ok: bool = True
    error: str | None = None


class AgentRuntime:
    def __init__(
        self,
        workspace: str | Path,
        *,
        agents_dirs: list[str | Path] | None = None,
        skills_dirs: list[str | Path] | None = None,
        models_config: str | Path | None = None,
        mcp_config: str | Path | None = None,
        auto_approve: bool = False,
        event_bus: EventBus | None = None,
        approval_handler=None,
        model_override: Model | None = None,
    ):
        self.workspace = Workspace(workspace)
        self.bus = event_bus or EventBus()
        self.permission_engine = PermissionEngine(
            auto_approve=auto_approve, approval_callback=approval_handler
        )
        self.memory = MemoryManager(self.workspace.root)
        self.context_engine = ContextEngine()
        self.run_manager = RunManager()
        self.model_router = ModelRouter.from_file(
            models_config or PROJECT_ROOT / "config" / "models.yaml"
        )
        self.agent_registry = AgentRegistry(agents_dirs or [PROJECT_ROOT / "agents"])
        self.skill_registry = SkillRegistry(skills_dirs or [PROJECT_ROOT / "skills"])
        self.model_override = model_override

        self.tool_registry = ToolRegistry()
        for tool in (
            make_filesystem_tools(self.workspace)
            + make_terminal_tools(self.workspace)
            + [self._make_remember_tool()]
        ):
            self.tool_registry.register(tool)

        mcp_path = mcp_config or PROJECT_ROOT / "config" / "mcp.json"
        self.mcp_toolsets = MCPLoader(mcp_path, self.permission_engine).load_toolsets()

    # -- tools ---------------------------------------------------------------

    def _make_remember_tool(self) -> Tool:
        memory = self.memory

        def remember(layer: str, key: str, value: str) -> str:
            """Store a durable fact. layer: 'user', 'project' or 'agent'."""
            try:
                memory.remember(layer, key, value)
            except ValueError as exc:
                return f"Error: {exc}"
            return f"remembered [{layer}] {key}"

        return Tool("remember", "Store a durable fact in user/project/agent memory", remember)

    def _resolve_tools(self, agent_def: AgentDefinition, skills: list[SkillManifest]) -> list[Tool]:
        names: list[str] = list(agent_def.tools)
        for skill in skills:
            for name in skill.tools:
                if name not in names:
                    names.append(name)
        tools: list[Tool] = []
        for name in names:
            if self.tool_registry.has(name):
                tools.append(self.tool_registry.get(name))
            else:
                logger.warning(
                    "agent %r references unknown tool %r; skipped", agent_def.id, name
                )
        return tools

    # -- agent loop ----------------------------------------------------------

    async def run(
        self,
        message: str,
        *,
        agent_id: str = "assistant",
        session_id: str = "default",
        model_role: str | None = None,
    ) -> RunResult:
        agent_def = self.agent_registry.get(agent_id)
        skills = [self.skill_registry.get(name) for name in agent_def.skills]
        tools = self._resolve_tools(agent_def, skills)
        run = self.run_manager.start(agent_def.id, session_id)
        emit = self._emitter(run)

        await emit("agent_started", {"agent": agent_def.id, "session": session_id})
        try:
            model = self.model_override or self.model_router.select(model_role or agent_def.model)
            system_prompt = self.context_engine.build_system_prompt(
                agent_def, skills, self.memory, self.workspace
            )

            agent: Agent[None, str] = Agent(
                model,
                system_prompt=system_prompt,
                toolsets=self.mcp_toolsets or None,
            )
            for tool in tools:
                agent.tool_plain(name=tool.name, description=tool.description)(
                    self._dispatch(tool, emit)
                )

            history = self._load_history(session_id)
            result = await agent.run(
                message,
                message_history=history,
                event_stream_handler=self._stream_handler(run, emit),
            )
            output = str(result.output)
            self.memory.save_conversation_state(
                session_id,
                ModelMessagesTypeAdapter.dump_python(result.all_messages()),
            )
            self.memory.add_message(session_id, "user", message)
            self.memory.add_message(session_id, "assistant", output)
            self.run_manager.finish(run, RunStatus.SUCCEEDED, output=output)
            await emit("agent_finished", {"output": output})
            return RunResult(output=output, run=run)
        except Exception as exc:
            self.run_manager.finish(run, RunStatus.FAILED, error=str(exc))
            await emit("error", {"error": f"{type(exc).__name__}: {exc}"})
            return RunResult(output="", run=run, ok=False, error=str(exc))

    # -- internals -------------------------------------------------------------

    def _load_history(self, session_id: str) -> list | None:
        data = self.memory.load_conversation_state(session_id)
        if not data:
            return None
        try:
            return ModelMessagesTypeAdapter.validate_python(data)
        except Exception:  # noqa: BLE001 - corrupt history should not break a session
            logger.warning("discarding corrupt conversation state for session %r", session_id)
            return None

    def _emitter(self, run: Run):
        async def emit(event_type: str, data: dict[str, Any]) -> None:
            self.run_manager.record_event(run, event_type, data)
            await self.bus.emit(
                Event(event_type, {"run_id": run.id, "session_id": run.session_id, **data})
            )

        return emit

    def _stream_handler(self, run: Run, emit):
        async def handler(ctx, events) -> None:
            async for event in events:
                if isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
                    await emit("message_delta", {"delta": event.delta.content_delta})

        return handler

    def _dispatch(self, tool: Tool, emit):
        engine = self.permission_engine

        async def dispatch(*args, **kwargs):
            call_id = uuid.uuid4().hex[:8]
            await emit("tool_call", {"tool": tool.name, "args": kwargs, "call_id": call_id})
            decision = engine.decide(tool.name, kwargs)
            if decision is PermissionDecision.ASK:
                await emit(
                    "permission_required",
                    {"tool": tool.name, "args": kwargs, "call_id": call_id},
                )
            if not await engine.authorize(tool.name, kwargs):
                output = f"Error: permission denied ({decision}) for tool {tool.name}"
                await emit(
                    "tool_result",
                    {"tool": tool.name, "call_id": call_id, "ok": False, "output": output},
                )
                return output
            try:
                result = tool.handler(*args, **kwargs)
                if inspect.isawaitable(result):
                    result = await result
                output = str(result)
                ok = not output.startswith("Error:")
            except Exception as exc:  # noqa: BLE001 - report to the model, don't crash
                output, ok = f"Error: {type(exc).__name__}: {exc}", False
            await emit(
                "tool_result",
                {"tool": tool.name, "call_id": call_id, "ok": ok, "output": output[:2000]},
            )
            return output

        # pydantic-ai derives the input schema from the handler signature.
        return functools.wraps(tool.handler)(dispatch)
