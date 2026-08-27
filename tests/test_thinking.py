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
    # 默认开启：不声明 thinking 字段的 agent 也会跑思考阶段
    (agents / "default-on.yaml").write_text(
        "id: default-on\nname: DefaultOn\nmodel: test-plain\n", encoding="utf-8"
    )
    (agents / "plain.yaml").write_text(
        "id: plain\nname: Plain\nmodel: test-plain\nthinking: false\n", encoding="utf-8"
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
    """thinking: false 的 agent 不支付思考阶段的额外模型调用。"""
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


async def test_thinking_default_on(workspace, monkeypatch):
    """不声明 thinking 字段的 agent 默认开启思考阶段。"""
    called = []

    async def fake_think(message, model, **kwargs):
        called.append(message)
        return "默认开启的分析"

    monkeypatch.setattr("guard_arch.runtime.run_thinking", fake_think)

    runtime = make_runtime(workspace)
    events = []
    runtime.bus.subscribe("thinking", lambda e: events.append(e))

    result = await runtime.run("默认开启验证", agent_id="default-on", session_id="th3")

    assert result.ok
    assert called, "thinking 默认开启：未声明字段也应跑思考阶段"
    assert events and events[0].data["delta"] == "默认开启的分析"


async def test_thinking_streams_deltas(workspace, monkeypatch):
    """思考内容流式下发：每个增量一个 thinking 事件，按序到达。"""
    chunks = ["用户想", "要健身计划", "，先确认目标"]

    async def fake_think(message, model, *, on_delta=None, **kwargs):
        assert on_delta is not None, "runtime 应注入流式回调"
        for chunk in chunks:
            await on_delta(chunk)
        return "".join(chunks)

    monkeypatch.setattr("guard_arch.runtime.run_thinking", fake_think)

    runtime = make_runtime(workspace)
    events = []
    runtime.bus.subscribe("thinking", lambda e: events.append(e))

    result = await runtime.run("帮我做个健身计划", agent_id="thinker", session_id="th4")

    assert result.ok
    assert [e.data["delta"] for e in events] == chunks


async def test_thinking_failure_degrades_gracefully(workspace, monkeypatch):
    """思考阶段抛错不阻断主执行：无 thinking 事件，run 正常完成。"""

    async def boom(message, model, **kwargs):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr("guard_arch.runtime.run_thinking", boom)

    runtime = make_runtime(workspace)
    events = []
    runtime.bus.subscribe("thinking", lambda e: events.append(e))

    result = await runtime.run("降级验证", agent_id="thinker", session_id="th5")

    assert result.ok
    assert result.output == "执行完成"
    assert not events
