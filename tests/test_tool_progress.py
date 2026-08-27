"""Tests for the tool-call lifecycle: tool_call → tool_progress (×N) → tool_result."""

import pytest

from guard_arch.core.tool import Tool, report_progress
from guard_arch.runtime import AgentRuntime

MODELS_YAML = """
models:
  test-caller:
    provider: test
    script:
      - tool: slow_tool
        args: {path: a.txt}
      - text: "完成"
"""


@pytest.fixture
def workspace(tmp_path):
    (tmp_path / "models.yaml").write_text(MODELS_YAML, encoding="utf-8")
    agents = tmp_path / "agents"
    agents.mkdir()
    (agents / "assistant.yaml").write_text(
        "id: assistant\nname: A\nmodel: test-caller\ntools: [slow_tool]\nthinking: false\n",
        encoding="utf-8",
    )
    return tmp_path


def make_runtime(workspace):
    return AgentRuntime(
        workspace,
        agents_dirs=[workspace / "agents"],
        skills_dirs=[workspace / "no-skills"],
        models_config=workspace / "models.yaml",
        mcp_config=workspace / "no-mcp.json",
        # 自定义工具未配置权限规则，默认 ASK 会被拒绝；测试聚焦生命周期事件，直接放行
        auto_approve=True,
    )


async def test_tool_progress_lifecycle(workspace):
    """上报进度的工具：tool_call（开始）→ tool_progress（进行中，携带部分数据）→ tool_result（结束）。"""
    runtime = make_runtime(workspace)

    async def slow_tool(path: str) -> str:
        await report_progress("已启动", "step-1")
        await report_progress("进行中", "step-2")
        return f"content of {path}"

    runtime.tool_registry.register(Tool("slow_tool", "test tool", slow_tool))

    events = []
    runtime.bus.subscribe("*", lambda e: events.append(e))

    result = await runtime.run("跑一下", session_id="tp1")

    assert result.ok
    lifecycle = [
        e for e in events if e.type in ("tool_call", "tool_progress", "tool_result")
    ]
    assert [e.type for e in lifecycle] == [
        "tool_call",
        "tool_progress",
        "tool_progress",
        "tool_result",
    ]
    progresses = [e for e in lifecycle if e.type == "tool_progress"]
    assert progresses[0].data["note"] == "已启动"
    assert progresses[0].data["data"] == "step-1"
    assert progresses[1].data["note"] == "进行中"
    # 进度事件与调用事件同属一次调用（call_id 关联）
    assert progresses[0].data["call_id"] == lifecycle[0].data["call_id"]
    assert lifecycle[-1].data["output"] == "content of a.txt"


async def test_tool_without_progress_has_clean_lifecycle(workspace):
    """不上报进度的工具保持简洁：tool_call → tool_result，无中间事件。"""

    def fast_tool(path: str) -> str:
        return f"content of {path}"

    runtime = make_runtime(workspace)
    runtime.tool_registry.register(Tool("slow_tool", "test tool", fast_tool))

    events = []
    runtime.bus.subscribe("*", lambda e: events.append(e))

    result = await runtime.run("跑一下", session_id="tp2")

    assert result.ok
    lifecycle = [
        e.type for e in events if e.type in ("tool_call", "tool_progress", "tool_result")
    ]
    assert lifecycle == ["tool_call", "tool_result"]
