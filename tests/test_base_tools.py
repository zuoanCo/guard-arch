"""Tests for the base capability set: every agent gets foundational tools by default."""

import pytest

from guard_arch.runtime import BASE_TOOL_NAMES, AgentRuntime

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
    # 一个 tools 列表为空、无 skills 的极简 agent——验证基础能力集仍自动并入
    (agents / "bare.yaml").write_text(
        "id: bare\nname: Bare\nmodel: test-plain\ntools: []\n", encoding="utf-8"
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


def test_base_tools_auto_included_for_bare_agent(workspace):
    """An agent with empty tools/skills still gets the base capability set resolved."""
    runtime = make_runtime(workspace)
    agent_def = runtime.agent_registry.get("bare")

    resolved = runtime._resolve_tools(agent_def, skills=[])
    resolved_names = {t.name for t in resolved}

    # 基础能力集（只读文件/搜索/web/记忆）全部自动并入，无需 YAML 声明
    for name in BASE_TOOL_NAMES:
        assert name in resolved_names, f"base tool {name!r} missing from resolved tools"


def test_agent_declared_tools_merge_with_base(workspace):
    """Agent-declared tools merge with (not replace) the base set, deduplicated."""
    agents = workspace / "agents"
    (agents / "writer.yaml").write_text(
        "id: writer\nname: Writer\nmodel: test-plain\ntools: [write_file, web_fetch]\n",
        encoding="utf-8",
    )
    runtime = make_runtime(workspace)
    agent_def = runtime.agent_registry.get("writer")

    resolved_names = [t.name for t in runtime._resolve_tools(agent_def, skills=[])]

    # 声明的 write_file 并入；web_fetch 与基础集去重（只出现一次）
    assert "write_file" in resolved_names
    assert resolved_names.count("web_fetch") == 1
