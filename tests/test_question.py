"""Tests for real ask_user_question interaction: run suspends, answer resumes it."""

import asyncio

import pytest

from guard_arch.runtime import AgentRuntime

MODELS_YAML = """
models:
  test-ask:
    provider: test
    script:
      - tool: ask_user_question
        args: {question: "你希望方案侧重成本还是速度？"}
      - text: "根据你的回答，方案已定。"
"""


@pytest.fixture
def workspace(tmp_path):
    (tmp_path / "models.yaml").write_text(MODELS_YAML, encoding="utf-8")
    agents = tmp_path / "agents"
    agents.mkdir()
    (agents / "assistant.yaml").write_text(
        "id: assistant\nname: Assistant\nmodel: test-ask\n", encoding="utf-8"
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


async def test_run_suspends_on_question_and_resumes_with_answer(workspace):
    """API 模式：run 在 ask_user_question 处挂起（user_question 事件下发），
    runtime.answer_question 注入回答后原 run 继续执行并完成。"""
    runtime = make_runtime(workspace)
    events = []
    runtime.bus.subscribe("*", lambda e: events.append(e))

    run_task = asyncio.create_task(runtime.run("帮我规划一下", model_role="test-ask", session_id="q1"))

    # 等 run 挂起到问题上（user_question 事件出现）
    for _ in range(100):
        if runtime.has_pending_question("q1"):
            break
        await asyncio.sleep(0.01)
    assert runtime.has_pending_question("q1")
    assert not run_task.done()  # run 挂起等待中，未结束

    # 注入用户回答 → run 恢复执行
    assert runtime.answer_question("q1", "侧重成本") is True
    result = await asyncio.wait_for(run_task, timeout=10)

    assert result.ok
    types = [e.type for e in events]
    assert "user_question" in types
    assert "user_answered" in types
    # 工具结果里带着用户回答回给了模型，agent 继续完成了任务
    results = [e for e in events if e.type == "tool_result" and e.data["tool"] == "ask_user_question"]
    assert results and "侧重成本" in results[0].data["output"]
    assert types[-1] == "agent_finished"


async def test_answer_without_pending_question_returns_false(workspace):
    runtime = make_runtime(workspace)
    assert runtime.answer_question("nobody", "answer") is False
    assert runtime.has_pending_question("nobody") is False


async def test_question_handler_interactive_mode(workspace):
    """CLI 模式：question_handler 直接同步拿到回答，不走 Future 挂起。"""
    runtime = make_runtime(workspace, question_handler=lambda q: f"回答-{q[:4]}")
    events = []
    runtime.bus.subscribe("tool_result", lambda e: events.append(e))

    result = await runtime.run("帮我规划一下", model_role="test-ask", session_id="q2")

    assert result.ok
    results = [e for e in events if e.data["tool"] == "ask_user_question"]
    assert results and "回答-" in results[0].data["output"]
