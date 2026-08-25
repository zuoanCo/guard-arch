"""Tests for the harness thinking phase (framework-driven analysis before execution)."""

import pytest

from guard_arch.runtime import AgentRuntime

MODELS_YAML = """
models:
  test-plain:
    provider: test
    output_text: "执行完成"
"""


@pytest.fixture
def workspace(tmp_path):
    (tmp_path / "models.yaml").write_text(MODELS_YAML, encoding="utf-8")
    agents = tmp_path / "agents"
    agents.mkdir()
    (agents / "thinker.yaml").write_text(
        "id: thinker\nname: Thinker\nmodel: test-plain\nthinking: true\n", encoding="utf-8"
    )
    (agents / "plain.yaml").write_text(
        "id: plain\nname: Plain\nmodel: test-plain\n", encoding="utf-8"
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


async def test_thinking_phase_emits_event_before_execution(workspace, monkeypatch):
    """thinking: true 的 agent：run 先跑框架思考阶段（thinking 事件），再进入主执行。"""
    thought = []

    async def fake_think(message, model, **kwargs):
        thought.append(message)
        return "用户想要健身计划，需要先用 ask_user_question 确认目标"

    monkeypatch.setattr("guard_arch.runtime.run_thinking", fake_think)

    runtime = make_runtime(workspace)
    events = []
    runtime.bus.subscribe("*", lambda e: events.append(e))

    result = await runtime.run("帮我做个健身计划", agent_id="thinker", session_id="th1")

    assert result.ok
    assert thought, "thinking phase must run before the main execution"
    # thinking 事件在主执行结果之前下发，内容为框架分析
    thinking_events = [e for e in events if e.type == "thinking"]
    assert thinking_events and "ask_user_question" in thinking_events[0].data["delta"]
    assert result.output == "执行完成"  # 主执行正常完成


async def test_agent_without_thinking_skips_phase(workspace, monkeypatch):
    """未开启 thinking 的 agent 不支付思考阶段的额外模型调用。"""
    called = []

    async def fake_think(message, model, **kwargs):
        called.append(message)
        return "analysis"

    monkeypatch.setattr("guard_arch.runtime.run_thinking", fake_think)

    runtime = make_runtime(workspace)
    events = []
    runtime.bus.subscribe("thinking", lambda e: events.append(e))

    result = await runtime.run("直接跑", agent_id="plain", session_id="th2")

    assert result.ok
    assert not called
    assert not events  # 无 thinking 事件
