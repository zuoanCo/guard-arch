"""Tests for the harness core rules: code-held defaults + rules.yaml overrides."""

import pytest

from guard_arch.core.rules import DEFAULT_RULES, RulesRegistry
from guard_arch.runtime import AgentRuntime

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
    (agents / "assistant.yaml").write_text(
        "id: assistant\nname: Assistant\nmodel: test-plain\n", encoding="utf-8"
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


def test_default_rules_rendered_into_system_prompt(workspace):
    """核心规则（代码持有）默认注入每个 agent 的 system prompt。"""
    runtime = make_runtime(workspace)
    agent_def = runtime.agent_registry.get("assistant")

    prompt = runtime.context_engine.build_system_prompt(
        agent_def, [], runtime.memory, runtime.workspace
    )

    assert "核心行为规则" in prompt
    # 默认规则的关键内容在 prompt 里（代码持有，不是 agent YAML 里的）
    assert "recall_memory" in prompt
    assert "web_search" in prompt


def test_rules_yaml_override_disable_replace_add(workspace, tmp_path):
    """rules.yaml 可控：禁用某条、替换某条文本、追加新规则。"""
    rules_yaml = tmp_path / "rules.yaml"
    rules_yaml.write_text(
        "rules:\n"
        "  concise: {enabled: false}\n"
        "  recall_first: {text: '自定义的记忆优先规则'}\n"
        "  my_rule: {text: '团队自定义新规则'}\n",
        encoding="utf-8",
    )
    registry = RulesRegistry(rules_yaml)

    by_id = {r.id: r for r in registry.all()}
    assert by_id["concise"].enabled is False
    assert by_id["recall_first"].text == "自定义的记忆优先规则"
    assert by_id["my_rule"].text == "团队自定义新规则"

    rendered = registry.render()
    assert "简洁高效" not in rendered  # 被禁用的规则不出现在渲染结果里
    assert "自定义的记忆优先规则" in rendered
    assert "团队自定义新规则" in rendered


def test_rules_yaml_missing_uses_code_defaults(tmp_path):
    """rules.yaml 不存在时纯代码默认规则生效。"""
    registry = RulesRegistry(tmp_path / "no-rules.yaml")
    assert len(registry.all()) == len(DEFAULT_RULES)
    assert registry.render()  # 渲染非空


async def test_rules_endpoint_and_cli_view(workspace):
    """可控口：CLI /rules 命令能查看当前生效规则。"""
    from guard_arch.cli import CLIContext, handle_slash_command

    runtime = make_runtime(workspace)
    ctx = CLIContext(runtime=runtime)
    output, _ = handle_slash_command("/rules", ctx)

    assert "recall_first" in output
    assert "realtime_web" in output
    assert "✓" in output  # 启用标记
