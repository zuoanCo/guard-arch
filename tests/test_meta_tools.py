"""Tests for per-run meta tools: ask_user_question and list_capabilities."""

import pytest

from guard_arch.runtime import AgentRuntime

MODELS_YAML = """
models:
  test-ask:
    provider: test
    script:
      - tool: ask_user_question
        args: {question: "你希望方案侧重成本还是速度？"}
      - text: "已向你提问，请回答后我继续。"
  test-list:
    provider: test
    script:
      - tool: list_capabilities
        args: {}
      - text: "能力清单已获取"
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


async def test_ask_user_question_via_interactive_handler(workspace):
    """ask_user_question（交互模式）：question_handler 直接拿到回答回给模型。
    挂起等待/API 回答注入的完整流程见 tests/test_question.py。"""
    runtime = make_runtime(workspace, question_handler=lambda q: "侧重成本")
    events = []
    runtime.bus.subscribe("*", lambda e: events.append(e))

    result = await runtime.run("帮我规划一下", model_role="test-ask", session_id="m1")

    assert result.ok
    results = [e for e in events if e.type == "tool_result" and e.data["tool"] == "ask_user_question"]
    assert results and results[0].data["ok"] is True
    assert "侧重成本" in results[0].data["output"]


async def test_list_capabilities_returns_runtime_inventory(workspace):
    """list_capabilities: returns the inventory of registered tools (and skills/MCP
    sections when present) so the model can plan around its actual capability set."""
    runtime = make_runtime(workspace)
    events = []
    runtime.bus.subscribe("tool_result", lambda e: events.append(e))

    result = await runtime.run("看看你都有什么能力", model_role="test-list", session_id="m2")

    assert result.ok
    list_results = [e for e in events if e.data["tool"] == "list_capabilities"]
    assert list_results and list_results[0].data["ok"] is True
    output = list_results[0].data["output"]
    # 清单包含全局注册的工具（文件/终端/web/记忆等），供模型规划执行链
    assert "web_fetch" in output
    assert "remember" in output
