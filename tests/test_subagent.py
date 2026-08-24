"""Tests for the sub-agent dispatch tool and the session todo tools."""

import pytest

from guard_arch.runtime import AgentRuntime

MODELS_YAML = """
models:
  test-parent:
    provider: test
    script:
      - tool: dispatch_agent
        args: {agent_id: "worker", task: "总结一下这个任务"}
      - text: "子代理已完成。"
  test-child:
    provider: test
    output_text: "子代理的最终输出"
  test-todo:
    provider: test
    script:
      - tool: todo_write
        args:
          todos_json: '[{"content": "第一步", "status": "completed"}, {"content": "第二步", "status": "in_progress"}]'
      - text: "计划已更新"
  test-dispatch-unknown:
    provider: test
    script:
      - tool: dispatch_agent
        args: {agent_id: "ghost", task: "x"}
      - text: "done"
"""


@pytest.fixture
def workspace(tmp_path):
    (tmp_path / "models.yaml").write_text(MODELS_YAML, encoding="utf-8")
    agents = tmp_path / "agents"
    agents.mkdir()
    (agents / "main.yaml").write_text(
        "id: main\nname: Main\nmodel: test-parent\n", encoding="utf-8"
    )
    (agents / "worker.yaml").write_text(
        "id: worker\nname: Worker\nmodel: test-child\n", encoding="utf-8"
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


async def test_dispatch_agent_runs_subagent_and_returns_output(workspace):
    """Parent dispatches a task to the 'worker' sub-agent; child's final output
    comes back as the dispatch_agent tool result, and subagent events fire."""
    runtime = make_runtime(workspace)
    events = []
    runtime.bus.subscribe("*", lambda e: events.append(e))

    result = await runtime.run(
        "派个活给子代理", agent_id="main", model_role="test-parent", session_id="s1"
    )

    assert result.ok
    types = [e.type for e in events]
    # 子代理生命周期事件：父 run 上发出 subagent_started / subagent_finished
    assert "subagent_started" in types
    assert "subagent_finished" in types
    # dispatch_agent 的工具结果（tool_result 事件）里是子代理的最终输出文本
    dispatch_results = [
        e for e in events if e.type == "tool_result" and e.data["tool"] == "dispatch_agent"
    ]
    assert dispatch_results and dispatch_results[0].data["ok"] is True
    assert "子代理的最终输出" in dispatch_results[0].data["output"]


async def test_dispatch_agent_unknown_agent_returns_error_not_crash(workspace):
    """Dispatching to an unknown agent id returns an Error string to the model
    (tool failure is feedback, not a crash)."""
    runtime = make_runtime(workspace)

    result = await runtime.run(
        "派给不存在的代理", agent_id="main", model_role="test-dispatch-unknown", session_id="s2"
    )

    assert result.ok  # run 本身正常结束，错误以文本形式回给模型
    dispatch_results = [
        e for e in result.run.events if e.type == "tool_result" and e.data["tool"] == "dispatch_agent"
    ]
    assert dispatch_results and dispatch_results[0].data["ok"] is False
    assert "unknown agent" in dispatch_results[0].data["output"]


async def test_todo_write_updates_list_and_emits_event(workspace):
    """todo_write replaces the session task list and emits todo_updated."""
    runtime = make_runtime(workspace)
    events = []
    runtime.bus.subscribe("todo_updated", lambda e: events.append(e))

    result = await runtime.run(
        "做个两步计划", agent_id="main", model_role="test-todo", session_id="s3"
    )

    assert result.ok
    assert events and len(events[0].data["todos"]) == 2
    assert events[0].data["todos"][0] == {"content": "第一步", "status": "completed"}
    rendered = runtime.todo_manager.render("s3")
    assert "第一步" in rendered and "第二步" in rendered
    assert "[x]" in rendered and "[>]" in rendered


async def test_todo_read_empty_by_default(workspace):
    """todo_read on a fresh session returns a friendly empty marker."""
    runtime = make_runtime(workspace)
    assert runtime.todo_manager.render("never-written") == "no todos yet"
