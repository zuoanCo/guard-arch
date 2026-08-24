"""Tests for the intake gate: structured pre-run analysis branches the execution path."""

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
    # intake: true 的 agent——每次 run 先过需求分析门禁
    (agents / "gated.yaml").write_text(
        "id: gated\nname: Gated\nmodel: test-plain\nintake: true\n", encoding="utf-8"
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


async def test_intake_short_circuits_when_clarification_needed(workspace, monkeypatch):
    """Intake gate returns needs_clarification + questions: the run short-circuits —
    the clarifying questions become this turn's reply, main execution never starts."""
    from guard_arch.core.intake import IntakeAnalysis

    async def fake_analyze(message, model, **kwargs):
        return IntakeAnalysis(
            clarity="needs_clarification",
            summary="用户想做方案但没说类型",
            questions=["你想做哪类方案？", "方案的目标是什么？"],
            plan=[],
        )

    monkeypatch.setattr("guard_arch.runtime.analyze_request", fake_analyze)

    runtime = make_runtime(workspace)
    events = []
    runtime.bus.subscribe("*", lambda e: events.append(e))

    result = await runtime.run("帮我做个方案", agent_id="gated", session_id="i1")

    assert result.ok
    # 本轮回复就是澄清问题（而非主执行链路的输出）
    assert "你想做哪类方案？" in result.output
    assert "方案的目标是什么？" in result.output
    # intake_analyzed 事件携带了结构化分析结果（clarity/questions）
    intake_events = [e for e in events if e.type == "intake_analyzed"]
    assert intake_events and intake_events[0].data["clarity"] == "needs_clarification"
    assert intake_events[0].data["questions"]
    # 短路回复也会经 message_delta 事件下发（流式消费者能收到澄清问题文本）
    deltas = [e for e in events if e.type == "message_delta"]
    assert deltas and "你想做哪类方案？" in deltas[0].data["delta"]


async def test_intake_proceeds_when_clear(workspace, monkeypatch):
    """Intake gate returns clear + plan: run proceeds to the main execution chain
    and the model's actual reply is returned."""
    from guard_arch.core.intake import IntakeAnalysis

    async def fake_analyze(message, model, **kwargs):
        return IntakeAnalysis(
            clarity="clear",
            summary="写一首春天的诗",
            questions=[],
            plan=["构思意象", "成诗"],
        )

    monkeypatch.setattr("guard_arch.runtime.analyze_request", fake_analyze)

    runtime = make_runtime(workspace)
    events = []
    runtime.bus.subscribe("intake_analyzed", lambda e: events.append(e))

    result = await runtime.run("写一首春天的诗", agent_id="gated", session_id="i2")

    assert result.ok
    assert result.output == "执行完成"  # 主执行链路正常跑完（test 模型的固定回复）
    assert events and events[0].data["clarity"] == "clear"
    assert events[0].data["plan"] == ["构思意象", "成诗"]


async def test_agent_without_intake_skips_gate(workspace, monkeypatch):
    """Agents without `intake: true` never pay the extra analysis call."""
    called = []

    async def fake_analyze(message, model, **kwargs):
        called.append(message)
        raise AssertionError("intake gate must not run for agents without intake: true")

    monkeypatch.setattr("guard_arch.runtime.analyze_request", fake_analyze)

    agents = workspace / "agents"
    (agents / "plain.yaml").write_text(
        "id: plain\nname: Plain\nmodel: test-plain\n", encoding="utf-8"
    )
    runtime = make_runtime(workspace)
    result = await runtime.run("直接跑", agent_id="plain", session_id="i3")

    assert result.ok
    assert result.output == "执行完成"
    assert not called  # 未开启 intake 的 agent 不触发门禁调用
