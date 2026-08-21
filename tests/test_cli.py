from guard_arch.cli import CLIContext, handle_slash_command, main
from guard_arch.runtime import AgentRuntime


def make_ctx(tmp_path) -> CLIContext:
    runtime = AgentRuntime(tmp_path, mcp_config=tmp_path / "no-mcp.json")
    return CLIContext(runtime=runtime)


def test_slash_help(tmp_path):
    ctx = make_ctx(tmp_path)
    output, should_exit = handle_slash_command("/help", ctx)
    assert "/model" in output and "/exit" in output
    assert should_exit is False


def test_slash_model_switch(tmp_path):
    ctx = make_ctx(tmp_path)
    output, _ = handle_slash_command("/model", ctx)
    assert "default" in output
    output, _ = handle_slash_command("/model test", ctx)
    assert "test" in output
    assert ctx.model_role == "test"
    output, _ = handle_slash_command("/model nope", ctx)
    assert "未知模型角色" in output
    assert ctx.model_role == "test"


def test_slash_lists(tmp_path):
    ctx = make_ctx(tmp_path)
    skills_out, _ = handle_slash_command("/skills", ctx)
    assert "coding" in skills_out and "research" in skills_out
    agents_out, _ = handle_slash_command("/agents", ctx)
    assert "assistant" in agents_out


def test_slash_clear_and_exit(tmp_path):
    ctx = make_ctx(tmp_path)
    ctx.runtime.memory.add_message(ctx.session_id, "user", "hi")
    output, _ = handle_slash_command("/clear", ctx)
    assert "清空" in output
    assert ctx.runtime.memory.recent_messages(ctx.session_id) == []
    _, should_exit = handle_slash_command("/exit", ctx)
    assert should_exit is True


def test_slash_unknown(tmp_path):
    ctx = make_ctx(tmp_path)
    output, should_exit = handle_slash_command("/bogus", ctx)
    assert "未知命令" in output
    assert should_exit is False


def test_cli_single_message_test_model(tmp_path, capsys):
    code = main(["--workspace", str(tmp_path), "--model", "test", "--message", "你好"])
    assert code == 0
    out = capsys.readouterr().out
    assert "Guard Arch" in out  # scripted output_text from config/models.yaml


def test_cli_single_message_tool_flow(tmp_path, capsys):
    (tmp_path / "README.md").write_text("# demo project", encoding="utf-8")
    code = main(
        ["--workspace", str(tmp_path), "--model", "test-demo", "--message", "读一下 README"]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "read_file" in out  # tool call status line rendered
    assert "README.md" in out
