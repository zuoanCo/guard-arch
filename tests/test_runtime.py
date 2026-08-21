import json

import pytest

from guard_arch import PROJECT_ROOT
from guard_arch.events.bus import EventBus
from guard_arch.runtime import AgentRuntime

MODELS_YAML = """
models:
  test-plain:
    provider: test
    output_text: "ok"
  test:
    provider: test
    script:
      - tool: read_file
        args: {path: hello.txt}
      - text: "文件内容已读取。"
  test-dangerous:
    provider: test
    script:
      - tool: run_command
        args: {command: "rm -rf /"}
      - text: "done"
  test-shell:
    provider: test
    script:
      - tool: run_command
        args: {command: "echo hello-from-tool"}
      - text: "done"
"""


@pytest.fixture
def workspace(tmp_path):
    (tmp_path / "hello.txt").write_text("你好 guard_arch", encoding="utf-8")
    (tmp_path / "models.yaml").write_text(MODELS_YAML, encoding="utf-8")
    return tmp_path


def make_runtime(workspace, **kwargs):
    return AgentRuntime(
        workspace,
        models_config=workspace / "models.yaml",
        mcp_config=workspace / "no-mcp.json",
        **kwargs,
    )


async def test_end_to_end_tool_call_and_events(workspace):
    runtime = make_runtime(workspace)
    events = []
    runtime.bus.subscribe("*", lambda e: events.append(e.type))

    result = await runtime.run("读一下 hello.txt", model_role="test", session_id="t1")

    assert result.ok
    assert result.output == "文件内容已读取。"
    types = list(events)
    assert types[0] == "agent_started"
    assert "tool_call" in types and "tool_result" in types
    assert types[-1] == "agent_finished"
    assert types.index("tool_call") < types.index("tool_result") < types.index("agent_finished")

    tool_calls = [r for r in result.run.events if r.type == "tool_call"]
    assert tool_calls[0].data["tool"] == "read_file"
    assert tool_calls[0].data["args"] == {"path": "hello.txt"}
    tool_results = [r for r in result.run.events if r.type == "tool_result"]
    assert tool_results[0].data["ok"] is True
    assert "你好 guard_arch" in tool_results[0].data["output"]


async def test_conversation_persisted_across_runs(workspace):
    runtime = make_runtime(workspace)
    await runtime.run("第一条", model_role="test", session_id="t2")
    state = runtime.memory.load_conversation_state("t2")
    assert state is not None
    history = runtime._load_history("t2")
    assert history is not None and len(history) >= 2


async def test_dangerous_command_denied_by_default(workspace):
    runtime = make_runtime(workspace)
    events = []
    runtime.bus.subscribe("tool_result", lambda e: events.append(e))
    result = await runtime.run("清理一下", model_role="test-dangerous", session_id="t3")
    assert result.ok
    denied = [e for e in events if e.data["tool"] == "run_command"]
    assert denied and denied[0].data["ok"] is False
    assert "permission denied" in denied[0].data["output"]


async def test_auto_approve_runs_benign_command(workspace):
    runtime = make_runtime(workspace, auto_approve=True)
    events = []
    runtime.bus.subscribe("tool_result", lambda e: events.append(e))
    result = await runtime.run("跑个命令", model_role="test-shell", session_id="t4")
    assert result.ok
    results = [e for e in events if e.data["tool"] == "run_command"]
    assert results and results[0].data["ok"] is True
    assert "hello-from-tool" in results[0].data["output"]


async def test_auto_approve_never_bypasses_deny_rules(workspace):
    runtime = make_runtime(workspace, auto_approve=True)
    events = []
    runtime.bus.subscribe("tool_result", lambda e: events.append(e))
    result = await runtime.run("清理一下", model_role="test-dangerous", session_id="t5")
    assert result.ok
    denied = [e for e in events if e.data["tool"] == "run_command"]
    assert denied and denied[0].data["ok"] is False
    assert "permission denied" in denied[0].data["output"]


async def test_event_bus_async_and_sync_subscribers():
    bus = EventBus()
    got = []

    async def async_sub(event):
        got.append(("async", event.type))

    bus.subscribe("agent_started", async_sub)
    bus.subscribe("*", lambda e: got.append(("sync", e.type)))
    from guard_arch.events.bus import Event

    await bus.emit(Event("agent_started"))
    assert got == [("async", "agent_started"), ("sync", "agent_started")]


async def test_agent_config_driven(workspace):
    """Agents come from YAML, not classes: a custom agents dir works."""
    agents = workspace / "myagents"
    agents.mkdir()
    (agents / "mini.yaml").write_text(
        "id: mini\nname: Mini\nmodel: test\nskills: []\ntools: [list_directory]\n"
        "instructions: 只列目录。",
        encoding="utf-8",
    )
    runtime = AgentRuntime(
        workspace,
        agents_dirs=[agents],
        skills_dirs=[workspace / "no-skills"],
        models_config=workspace / "models.yaml",
        mcp_config=workspace / "no-mcp.json",
    )
    result = await runtime.run("hi", agent_id="mini", model_role="test-plain")
    assert result.ok
    # mini only has list_directory; read_file must not be registered on its agent
    assert runtime.agent_registry.get("mini").tools == ["list_directory"]


def test_runtime_uses_default_project_config():
    assert (PROJECT_ROOT / "config" / "models.yaml").exists()
    assert (PROJECT_ROOT / "agents" / "assistant.yaml").exists()


async def test_mcp_config_missing_degrades_gracefully(workspace):
    runtime = make_runtime(workspace)
    assert runtime.mcp_toolsets == []


async def test_mcp_bad_config_degrades_gracefully(workspace):
    bad = workspace / "mcp.json"
    bad.write_text(json.dumps({"unexpected": True}), encoding="utf-8")
    runtime = AgentRuntime(
        workspace,
        models_config=workspace / "models.yaml",
        mcp_config=bad,
    )
    assert runtime.mcp_toolsets == []
