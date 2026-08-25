"""AgentRuntime: composes registries, memory, permissions, sandbox and the
event bus, and drives the pydantic-ai agent loop."""

import asyncio
import functools
import inspect
import json
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
    ThinkingPartDelta,
)
from pydantic_ai.models import Model

from guard_arch import PROJECT_ROOT
from guard_arch.core.agent import AgentDefinition, AgentRegistry
from guard_arch.core.compact import DEFAULT_COMPACTION_THRESHOLD_TOKENS, HistoryCompactor
from guard_arch.core.context import ContextEngine
from guard_arch.core.intake import analyze_request
from guard_arch.core.memory import MemoryManager
from guard_arch.core.model import ModelRouter
from guard_arch.core.plan import TodoManager
from guard_arch.core.question import QUESTION_TIMEOUT_SECONDS, QuestionManager
from guard_arch.core.run import Run, RunManager, RunStatus
from guard_arch.core.skill import SkillManifest, SkillRegistry
from guard_arch.core.think import think as run_thinking
from guard_arch.core.tool import Tool, ToolRegistry
from guard_arch.core.workspace import Workspace
from guard_arch.events.bus import Event, EventBus
from guard_arch.mcp.client import MCPLoader
from guard_arch.permissions.engine import PermissionDecision, PermissionEngine
from guard_arch.tools.filesystem import make_filesystem_tools
from guard_arch.tools.terminal import make_terminal_tools
from guard_arch.tools.web import make_web_tools

logger = logging.getLogger(__name__)

# 基础能力集（base capabilities）：每次 run 自动并入 agent 的工具集，无需 YAML 声明。
# 只读/低危能力铺满：工作区读取与搜索、联网搜索与抓取、长期记忆读写。
# 写文件/改文件/执行命令不在基础集内——需在 agent YAML 显式声明且过权限门控。
BASE_TOOL_NAMES = (
    "read_file",
    "list_directory",
    "search_text",
    "web_search",
    "web_fetch",
    "remember",
    "recall_memory",
)

# intake 门禁快速通道：消息短于该长度（打招呼/简短应答等）跳过门禁分析，
# 直接进入主执行——简单消息不需要为"是否澄清"多付一次模型调用的延迟
INTAKE_SKIP_MAX_CHARS = 12


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
        compaction_threshold_tokens: int = DEFAULT_COMPACTION_THRESHOLD_TOKENS,
        question_handler=None,
        max_concurrent_runs: int = 4,
    ):
        self.workspace = Workspace(workspace)
        self.bus = event_bus or EventBus()
        self.permission_engine = PermissionEngine(
            auto_approve=auto_approve, approval_callback=approval_handler
        )
        self.memory = MemoryManager(self.workspace.root)
        self.context_engine = ContextEngine()
        self.run_manager = RunManager()
        # 会话级任务清单（agent 用 todo_write/todo_read 工具自己规划、追踪多步任务）
        self.todo_manager = TodoManager()
        # 用户问答管理：ask_user_question 挂起 run 等待回答，回答后原 run 继续执行
        self.question_manager = QuestionManager()
        # CLI 交互式提问回调（(question) -> answer）；None 时走 Future 等待（API 模式）
        self.question_handler = question_handler
        # 长会话历史压缩器：历史超过 token 阈值时把旧消息摘要化，保住上下文窗口
        self.compactor = HistoryCompactor(threshold_tokens=compaction_threshold_tokens)
        # 资源限制：并发 run 上限（超出排队等待，防并发打爆模型额度/系统资源）
        self._run_semaphore = asyncio.Semaphore(max_concurrent_runs)
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
            + make_web_tools()
            + [self._make_remember_tool(), self._make_recall_tool()]
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

    def _make_recall_tool(self) -> Tool:
        memory = self.memory

        def recall_memory(query: str, layer: str = "") -> str:
            """Search durable memory by keyword. query: keyword to match against memory
            keys/values. layer: optional, one of 'user'/'project'/'agent'; leave empty
            to search all layers. Returns matching entries grouped by layer."""
            matches = memory.search(query, layer or None)
            if not matches:
                return f"no memory entries match {query!r}"
            lines: list[str] = []
            for name, items in matches.items():
                lines.append(f"[{name} memory]")
                for key, value in items.items():
                    lines.append(f"- {key}: {value}")
            return "\n".join(lines)

        return Tool(
            "recall_memory",
            "Search durable memory (user/project/agent layers) by keyword; use to "
            "recall facts beyond what was auto-injected into the system prompt",
            recall_memory,
        )

    def _resolve_tools(self, agent_def: AgentDefinition, skills: list[SkillManifest]) -> list[Tool]:
        # 基础能力集自动并入（去重）：每个 agent 默认拥有只读/搜索/web/记忆等原能力
        names: list[str] = list(BASE_TOOL_NAMES)
        for name in agent_def.tools:
            if name not in names:
                names.append(name)
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

    def _make_todo_tools(self, session_id: str, emit) -> list[Tool]:
        """Per-run todo tools bound to the session: agent plans/tracks multi-step work."""
        manager = self.todo_manager

        async def todo_write(todos_json: str) -> str:
            """Replace the session task list. todos_json: JSON array of
            {"content": str, "status": "pending"|"in_progress"|"completed"}."""
            try:
                raw = json.loads(todos_json)
                if not isinstance(raw, list):
                    raise ValueError("todos_json must be a JSON array")
                items = manager.write(session_id, raw)
            except (json.JSONDecodeError, ValueError) as exc:
                return f"Error: {exc}"
            await emit(
                "todo_updated",
                {"todos": [{"content": i.content, "status": i.status} for i in items]},
            )
            return f"todo list updated ({len(items)} items)"

        def todo_read() -> str:
            """Read the session's current task list (checkbox lines)."""
            return manager.render(session_id)

        return [
            Tool(
                "todo_write",
                "Create or replace the session's task list to plan and track "
                "multi-step work (pass a JSON array of {content, status})",
                todo_write,
            ),
            Tool("todo_read", "Read the session's current task list", todo_read),
        ]

    def _make_meta_tools(self, run: Run, emit) -> list[Tool]:
        """Per-run meta capabilities: every agent gets these without YAML declaration.

        - ask_user_question: mid-run channel back to the user when the agent needs
          a decision or missing info (emits user_question event; the model should
          pose the question in its reply and wait for the user's next message)
        - list_capabilities: runtime inventory of the agent's "limbs" — all tools,
          skills and MCP toolsets — so the model can check what's usable for the
          problem at hand before planning the execution chain
        """

        async def ask_user_question(question: str) -> str:
            """Ask the user a question when you need their decision or missing
            information to proceed. The run SUSPENDS until the user answers
            (interactive handler in CLI, async answer injection in API mode),
            then resumes with the answer — a real interactive step, not turn end."""
            session_id = run.session_id
            # CLI 模式：交互式处理器直接向用户提问并同步拿到回答
            if self.question_handler is not None:
                answer = self.question_handler(question)
                if inspect.isawaitable(answer):
                    answer = await answer
                return f"用户回答：{answer}"
            # API 模式：发 user_question 事件后挂起等待 answer_question() 注入回答
            question_id, future = self.question_manager.create(session_id)
            await emit("user_question", {"question": question, "question_id": question_id})
            try:
                answer = await asyncio.wait_for(future, timeout=QUESTION_TIMEOUT_SECONDS)
            except TimeoutError:
                self.question_manager.cancel(session_id)
                return "Error: 用户长时间未回答；请自行决定下一步或在回复中说明需要该信息"
            await emit("user_answered", {"question_id": question_id, "answer": answer})
            return f"用户回答：{answer}"

        def list_capabilities() -> str:
            """List all capabilities available in this runtime: tools you can call,
            skills (working methodologies) loaded, and MCP toolsets attached.
            Use this to check what's available for the problem before planning."""
            lines = ["## 工具（可直接调用解决子问题）"]
            for t in self.tool_registry.all():
                lines.append(f"- {t.name}: {t.description}")
            skills = self.skill_registry.all()
            if skills:
                lines.append("## 技能（方法论/工作方式，按需遵循其指导）")
                for s in skills:
                    lines.append(f"- {s.name}: {s.description}")
            if self.mcp_toolsets:
                lines.append(f"## MCP 工具集（外部服务能力，已挂载 {len(self.mcp_toolsets)} 个）")
            return "\n".join(lines)

        return [
            Tool(
                "ask_user_question",
                "Ask the user a question when you need their decision or missing "
                "information to proceed with the task",
                ask_user_question,
            ),
            Tool(
                "list_capabilities",
                "List all tools, skills and MCP toolsets available in this runtime; "
                "use to check what's usable for the problem before planning execution",
                list_capabilities,
            ),
        ]

    def _make_dispatch_tool(self, parent_run: Run, emit) -> Tool:
        """Per-run sub-agent tool: delegate a self-contained task to a child agent."""

        async def dispatch_agent(agent_id: str, task: str) -> str:
            """Dispatch a self-contained task to a sub-agent by agent id.
            The sub-agent runs in an isolated context (no parent history) and
            only its final text output comes back. Use for focused subtasks."""
            try:
                agent_def = self.agent_registry.get(agent_id)
            except KeyError as exc:
                return f"Error: {exc}"
            await emit("subagent_started", {"agent": agent_id, "task": task})
            result = await self._run_subagent(agent_def, task, parent_run=parent_run)
            await emit(
                "subagent_finished",
                {"agent": agent_id, "ok": result.ok, "error": result.error},
            )
            if not result.ok:
                return f"Error: subagent {agent_id!r} failed: {result.error}"
            return result.output

        return Tool(
            "dispatch_agent",
            "Delegate a self-contained subtask to a sub-agent (by agent id) and get "
            "its final answer; the sub-agent has its own fresh context",
            dispatch_agent,
        )

    async def _run_subagent(self, agent_def: AgentDefinition, task: str, *, parent_run: Run) -> RunResult:
        """Run a child agent one-shot in an isolated context; returns its final output.

        The child gets a fresh run (no conversation history), its own events
        (tagged with parent_run_id), and the agent's configured tools — but NOT
        dispatch_agent itself, so subagents cannot recurse (depth-1 cap).
        """
        skills = [self.skill_registry.get(name) for name in agent_def.skills]
        tools = self._resolve_tools(agent_def, skills)
        child_run = self.run_manager.start(agent_def.id, f"{parent_run.session_id}:sub")
        child_emit = self._emitter(child_run)

        await child_emit(
            "agent_started", {"agent": agent_def.id, "parent_run_id": parent_run.id}
        )
        try:
            model = self.model_override or self.model_router.select(agent_def.model)
            system_prompt = self.context_engine.build_system_prompt(
                agent_def, skills, self.memory, self.workspace, tools=tools
            )
            agent: Agent[None, str] = Agent(
                model,
                system_prompt=system_prompt,
                toolsets=self.mcp_toolsets or None,
            )
            for tool in tools:
                agent.tool_plain(name=tool.name, description=tool.description)(
                    self._dispatch(tool, child_emit)
                )
            result = await agent.run(
                task,
                event_stream_handler=self._stream_handler(child_run, child_emit),
            )
            output = str(result.output)
            self.run_manager.finish(child_run, RunStatus.SUCCEEDED, output=output)
            await child_emit(
                "agent_finished", {"output": output, "parent_run_id": parent_run.id}
            )
            return RunResult(output=output, run=child_run)
        except Exception as exc:
            self.run_manager.finish(child_run, RunStatus.FAILED, error=str(exc))
            await child_emit(
                "error",
                {"error": f"{type(exc).__name__}: {exc}", "parent_run_id": parent_run.id},
            )
            return RunResult(output="", run=child_run, ok=False, error=str(exc))

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
        # 并发资源限制：超出 max_concurrent_runs 的 run 在此排队
        async with self._run_semaphore:
            return await self._run_inner(
                message,
                agent_def=agent_def,
                skills=skills,
                tools=tools,
                run=run,
                emit=emit,
                session_id=session_id,
                model_role=model_role,
            )

    async def _run_inner(
        self,
        message: str,
        *,
        agent_def: AgentDefinition,
        skills: list[SkillManifest],
        tools: list[Tool],
        run: Run,
        emit,
        session_id: str,
        model_role: str | None,
    ) -> RunResult:
        try:
            model = self.model_override or self.model_router.select(model_role or agent_def.model)

            # 框架思考阶段（agent YAML 里 thinking: true 才启用）：
            # 先让模型针对需求+能力面做一次分析（harness thinking，区别于模型内部思维链），
            # 分析作为 thinking 事件下发，并注入主执行输入指导后续 tool_call/text
            thinking_note = ""
            if agent_def.thinking:
                capabilities = "\n".join(f"- {t.name}: {t.description}" for t in tools)
                context = ""
                history = self._load_history(session_id)
                if history:
                    from guard_arch.core.compact import render_messages

                    context = render_messages(history[-6:])
                analysis = await run_thinking(
                    message, model, capabilities=capabilities, context=context
                )
                await emit("thinking", {"delta": analysis})
                thinking_note = analysis

            # 需求分析门禁（agent YAML 里 intake: true 才启用）：
            # 先做一次结构化分析调用，结果硬性分支执行路径——
            # 不清晰 → 短路返回澄清问题（不启动主执行链路）；清晰 → 正常执行
            # 快速通道：过短的消息（打招呼/简短应答）不存在歧义，跳过门禁省一次模型调用
            if agent_def.intake and len(message.strip()) > INTAKE_SKIP_MAX_CHARS:
                # 门禁判断需知情：传入 agent 能力清单 + 最近对话上下文，
                # 避免把"能用工具解决的需求"误判为需要澄清
                history = self._load_history(session_id)
                context = ""
                if history:
                    from guard_arch.core.compact import render_messages

                    context = render_messages(history[-6:])
                capabilities = "\n".join(
                    f"- {t.name}: {t.description}"
                    for t in [
                        *tools,
                        *self._make_todo_tools(session_id, emit),
                        self._make_dispatch_tool(run, emit),
                        *self._make_meta_tools(run, emit),
                    ]
                )
                analysis = await analyze_request(
                    message, model, capabilities=capabilities, context=context
                )
                await emit(
                    "intake_analyzed",
                    {
                        "clarity": analysis.clarity,
                        "summary": analysis.summary,
                        "questions": analysis.questions,
                        "plan": analysis.plan,
                    },
                )
                if analysis.clarity == "needs_clarification" and analysis.questions:
                    output = "\n".join(analysis.questions)
                    # 短路路径的回复不经过模型流式输出，补发 message_delta 让流式消费者收到文本
                    await emit("message_delta", {"delta": output})
                    # 短路轮次也必须写入模型历史（session_state），否则被拦截的对话
                    # 在后续轮次的上下文中等于没发生过，agent 会"失忆"
                    self._append_to_history(session_id, message, output)
                    self.run_manager.finish(run, RunStatus.SUCCEEDED, output=output)
                    await emit("agent_finished", {"output": output})
                    return RunResult(output=output, run=run)
            # 该 agent 本次 run 实际可用的全部工具：① agent/skills 声明的注册表工具
            # ② 会话级 todo 工具 ③ 子代理派发工具（dispatch_agent）
            # ④ 元能力工具（ask_user_question / list_capabilities，每 run 自动挂载）
            run_tools = [
                *tools,
                *self._make_todo_tools(session_id, emit),
                self._make_dispatch_tool(run, emit),
                *self._make_meta_tools(run, emit),
            ]
            # system prompt 中注入能力面（工具清单），让模型知道自己"能做什么"
            system_prompt = self.context_engine.build_system_prompt(
                agent_def, skills, self.memory, self.workspace, tools=run_tools
            )

            agent: Agent[None, str] = Agent(
                model,
                system_prompt=system_prompt,
                toolsets=self.mcp_toolsets or None,
            )
            for tool in run_tools:
                agent.tool_plain(name=tool.name, description=tool.description)(
                    self._dispatch(tool, emit)
                )

            history = self._load_history(session_id)
            history = await self._maybe_compact_history(session_id, history, model, emit)
            # 有框架思考结论时注入主执行输入：思考阶段得出的方向指导执行链路
            run_input = (
                f"<harness_analysis>\n{thinking_note}\n</harness_analysis>\n\n{message}"
                if thinking_note
                else message
            )
            result = await agent.run(
                run_input,
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

    def _append_to_history(self, session_id: str, user_message: str, assistant_output: str) -> None:
        """Append one user/assistant exchange to the session's model-message history.

        Used by paths that bypass agent.run (e.g. intake-gate short-circuit) so the
        exchange still appears in future turns' conversation context.
        """
        from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

        history = self._load_history(session_id) or []
        history.append(ModelRequest(parts=[UserPromptPart(content=user_message)]))
        history.append(ModelResponse(parts=[TextPart(content=assistant_output)]))
        self.memory.save_conversation_state(
            session_id, ModelMessagesTypeAdapter.dump_python(history)
        )

    async def _maybe_compact_history(self, session_id, history, model, emit) -> list | None:
        """Compact an over-threshold history before the run (no-op when history is short)."""
        if not history or not self.compactor.needs_compaction(history):
            return history
        compacted = await self.compactor.compact(history, model)
        await emit(
            "history_compacted",
            {"session": session_id, "before": len(history), "after": len(compacted)},
        )
        return compacted

    def _emitter(self, run: Run):
        async def emit(event_type: str, data: dict[str, Any]) -> None:
            self.run_manager.record_event(run, event_type, data)
            await self.bus.emit(
                Event(event_type, {"run_id": run.id, "session_id": run.session_id, **data})
            )

        return emit

    def answer_question(self, session_id: str, answer: str) -> bool:
        """Inject the user's answer into a run suspended on ask_user_question.

        Returns True if a pending question for the session was resolved.
        """
        return self.question_manager.answer(session_id, answer)

    def has_pending_question(self, session_id: str) -> bool:
        return self.question_manager.has_pending(session_id)

    def _stream_handler(self, run: Run, emit):
        async def handler(ctx, events) -> None:
            async for event in events:
                if isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
                    await emit("message_delta", {"delta": event.delta.content_delta})
                elif isinstance(event, PartDeltaEvent) and isinstance(event.delta, ThinkingPartDelta):
                    # 模型的分析思考过程（thinking 执行类型）作为独立事件下发
                    await emit("thinking", {"delta": event.delta.content_delta})

        return handler

    # -- 工具派发链：tool_call 事件 → 权限门控 → （超时+静默重试）执行 → 验证 → tool_result --

    @staticmethod
    def _is_transient_error(output: str) -> bool:
        """判断工具失败是否瞬时故障（可静默重试）：网络抖动/限流/超时/5xx。"""
        text = output.lower()
        transient_markers = (
            "timeout",
            "timed out",
            "connecterror",
            "connection",
            "429",
            "rate limit",
            "502",
            "503",
            "504",
            "temporarily unavailable",
        )
        return any(marker in text for marker in transient_markers)

    async def _execute_with_retry(self, tool: Tool, emit, call_id: str, *args, **kwargs) -> tuple[str, bool]:
        """执行工具：超时限制 + 瞬时故障静默重试（确认失败才暴露给模型）。"""
        attempts = 1 + max(0, tool.retry_attempts)
        for attempt in range(1, attempts + 1):
            try:
                result = tool.handler(*args, **kwargs)
                if inspect.isawaitable(result):
                    result = await asyncio.wait_for(result, timeout=tool.timeout_seconds)
                output = str(result)
                ok = not output.startswith("Error:")
            except TimeoutError:
                output, ok = f"Error: tool {tool.name} timed out after {tool.timeout_seconds}s", False
            except Exception as exc:  # noqa: BLE001 - report to the model, don't crash
                output, ok = f"Error: {type(exc).__name__}: {exc}", False

            # 成功 / 非瞬时故障 / 重试已用尽：返回结果
            if ok or not self._is_transient_error(output) or attempt == attempts:
                return output, ok
            # 瞬时故障：静默重试（发 tool_retry 事件供观测，不作为失败暴露）
            await emit(
                "tool_retry",
                {"tool": tool.name, "call_id": call_id, "attempt": attempt, "next_in_seconds": attempt},
            )
            await asyncio.sleep(min(attempt, 5))  # 线性退避，上限 5s
        return output, ok  # pragma: no cover

    async def _verify_result(self, tool: Tool, emit, call_id: str, args: dict, output: str) -> str:
        """验证阶段：运行工具的验证器检查执行结果（而非模型自述），发 tool_verified 事件。"""
        if tool.verifier is None:
            return output
        try:
            note = tool.verifier(args, output)
            if inspect.isawaitable(note):
                note = await note
        except Exception as exc:  # noqa: BLE001 - 验证器异常不阻断，如实记录
            note = f"验证器执行失败: {exc}"
        if note:
            await emit(
                "tool_verified",
                {"tool": tool.name, "call_id": call_id, "ok": False, "note": note},
            )
            return f"{output}\n[验证失败] {note}"
        await emit("tool_verified", {"tool": tool.name, "call_id": call_id, "ok": True, "note": ""})
        return output

    def _dispatch(self, tool: Tool, emit):
        engine = self.permission_engine

        async def dispatch(*args, **kwargs):
            call_id = uuid.uuid4().hex[:8]
            risk = engine.risk_of(tool.name, kwargs)
            await emit(
                "tool_call",
                {"tool": tool.name, "args": kwargs, "call_id": call_id, "risk": str(risk)},
            )
            decision = engine.decide(tool.name, kwargs)
            if decision is PermissionDecision.ASK:
                await emit(
                    "permission_required",
                    {"tool": tool.name, "args": kwargs, "call_id": call_id, "risk": str(risk)},
                )
            if not await engine.authorize(tool.name, kwargs):
                output = f"Error: permission denied ({decision}) for tool {tool.name}"
                await emit(
                    "tool_result",
                    {"tool": tool.name, "call_id": call_id, "ok": False, "output": output},
                )
                return output
            output, ok = await self._execute_with_retry(tool, emit, call_id, *args, **kwargs)
            if ok:
                output = await self._verify_result(tool, emit, call_id, kwargs, output)
            await emit(
                "tool_result",
                {"tool": tool.name, "call_id": call_id, "ok": ok, "output": output[:2000]},
            )
            return output

        # pydantic-ai derives the input schema from the handler signature.
        return functools.wraps(tool.handler)(dispatch)
