"""Tests for production-grade harness: risk levels, timeout, retry, verification,
workspace instruction files."""

import asyncio

import pytest

from guard_arch.core.tool import Tool
from guard_arch.permissions.engine import PermissionDecision, PermissionEngine, RiskLevel
from guard_arch.runtime import AgentRuntime

MODELS_YAML = """
models:
  test-plain:
    provider: test
    output_text: "ok"
"""


@pytest.fixture
def workspace(tmp_path):
    (tmp_path / "models.yaml").write_text(MODELS_YAML, encoding="utf-8")
    agents = tmp_path / "agents"
    agents.mkdir()
    (agents / "assistant.yaml").write_text(
        "id: assistant\nname: Assistant\nmodel: test-plain\n", encoding="utf-8"
    )
    return tmp_path


def make_runtime(workspace, **kwargs):
    return AgentRuntime(
        workspace,
        agents_dirs=[workspace / "agents"],
        skills_dirs=[workspace / "no-skills"],
        models_config=workspace / "models.yaml",
        mcp_config=workspace / "no-mcp.json",
        **kwargs,
    )


# ---------- 约束：风险评级 ----------

def test_high_risk_never_auto_approved():
    """HIGH 高危规则（rm -rf 等）即使 auto_approve=True 也一律 DENY。"""
    engine = PermissionEngine(auto_approve=True)
    decision = engine.decide("run_command", {"command": "rm -rf /"})
    assert decision is PermissionDecision.DENY
    assert engine.risk_of("run_command", {"command": "rm -rf /"}) is RiskLevel.HIGH


def test_risk_levels_classification():
    """只读工具 LOW、写/联网 MID、未知工具按 MID fail-safe。"""
    engine = PermissionEngine()
    assert engine.risk_of("read_file", {"path": "a"}) is RiskLevel.LOW
    assert engine.risk_of("write_file", {"path": "a"}) is RiskLevel.MID
    assert engine.risk_of("unknown_tool", {}) is RiskLevel.MID  # fail-safe 默认


# ---------- 约束：工具执行超时 ----------

async def test_tool_timeout_kills_hung_tool(workspace):
    """超过 timeout_seconds 的工具被按超时失败处理，不会拖死整个 run。"""
    runtime = make_runtime(workspace, auto_approve=True)
    events = []
    runtime.bus.subscribe("tool_result", lambda e: events.append(e))

    async def slow_tool() -> str:
        await asyncio.sleep(10)
        return "done"

    tool = Tool("slow", "slow tool", slow_tool, timeout_seconds=0.05)
    emit = runtime._emitter(runtime.run_manager.start("assistant", "t-timeout"))
    dispatch = runtime._dispatch(tool, emit)
    output = await dispatch()

    assert "timed out" in output
    assert events and events[0].data["ok"] is False


# ---------- 纠正：瞬时故障静默重试 ----------

async def test_transient_error_retried_silently_then_succeeds(workspace):
    """瞬时故障（timeout/限流）静默重试：发 tool_retry 事件（观测用），
    重试成功后按成功返回，不向模型暴露中间失败态。"""
    runtime = make_runtime(workspace, auto_approve=True)
    events = []
    runtime.bus.subscribe("*", lambda e: events.append(e))

    calls = []

    async def flaky_tool() -> str:
        calls.append(1)
        if len(calls) < 3:
            return "Error: connection timeout"
        return "finally ok"

    tool = Tool("flaky", "flaky tool", flaky_tool, retry_attempts=3)
    emit = runtime._emitter(runtime.run_manager.start("assistant", "t-retry"))
    dispatch = runtime._dispatch(tool, emit)
    output = await dispatch()

    assert output == "finally ok"
    assert len(calls) == 3  # 失败 2 次 + 第 3 次成功
    retry_events = [e for e in events if e.type == "tool_retry"]
    assert len(retry_events) == 2  # 两次静默重试都有观测事件
    results = [e for e in events if e.type == "tool_result"]
    assert results[0].data["ok"] is True


async def test_non_transient_error_not_retried(workspace):
    """非瞬时故障（如权限/参数错误）不重试，直接返回错误。"""
    runtime = make_runtime(workspace, auto_approve=True)

    calls = []

    def bad_args_tool() -> str:
        calls.append(1)
        return "Error: old_string not found in x.txt"

    tool = Tool("badargs", "bad args tool", bad_args_tool, retry_attempts=3)
    emit = runtime._emitter(runtime.run_manager.start("assistant", "t-noretry"))
    dispatch = runtime._dispatch(tool, emit)
    output = await dispatch()

    assert output.startswith("Error:")
    assert len(calls) == 1  # 非瞬时故障不重试


# ---------- 验证：写后回读确认 ----------

async def test_write_file_verified_by_reread(workspace):
    """write_file 成功后验证器回读确认：发 tool_verified(ok) 事件。"""
    runtime = make_runtime(workspace)
    events = []
    runtime.bus.subscribe("tool_verified", lambda e: events.append(e))

    write_tool = runtime.tool_registry.get("write_file")
    emit = runtime._emitter(runtime.run_manager.start("assistant", "t-verify"))
    dispatch = runtime._dispatch(write_tool, emit)
    output = await dispatch(path="note.txt", content="验证内容")

    assert "wrote" in output
    assert events and events[0].data["ok"] is True
    # 文件确实写入了（验证的是结果不是自述）
    assert (workspace / "note.txt").read_text(encoding="utf-8") == "验证内容"


# ---------- 上下文：工作区指令文件注入 ----------

def test_workspace_instruction_files_injected(workspace):
    """工作区根目录的 GUARD.md/AGENTS.md/CLAUDE.md 自动进入 system prompt。"""
    (workspace / "GUARD.md").write_text("本项目使用 ruff 做代码检查。", encoding="utf-8")
    runtime = make_runtime(workspace)
    agent_def = runtime.agent_registry.get("assistant")

    prompt = runtime.context_engine.build_system_prompt(
        agent_def, [], runtime.memory, runtime.workspace
    )

    assert "GUARD.md" in prompt
    assert "ruff 做代码检查" in prompt
