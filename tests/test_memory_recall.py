"""Tests for on-demand memory retrieval (recall_memory tool + MemoryManager.search)."""

import pytest

from guard_arch.runtime import AgentRuntime

MODELS_YAML = """
models:
  test-recall:
    provider: test
    script:
      - tool: recall_memory
        args: {query: "偏好", layer: ""}
      - text: "已回忆起相关记忆"
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
        "id: assistant\nname: Assistant\nmodel: test-plain\ntools: [recall_memory]\n",
        encoding="utf-8",
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


# ---------- MemoryManager.search 单元层 ----------

def test_search_matches_key_and_value_across_layers(workspace):
    """search() 命中 key 或 value 含关键词的记忆条目（跨层，含层名分组）。"""
    runtime = make_runtime(workspace)
    runtime.memory.remember("user", "偏好-主题", "深色模式")
    runtime.memory.remember("project", "技术栈", "FastAPI + React Native")

    by_key = runtime.memory.search("偏好")  # 命中 user 层的 key
    assert by_key == {"user": {"偏好-主题": "深色模式"}}

    by_value = runtime.memory.search("FastAPI")  # 命中 project 层的 value
    assert by_value == {"project": {"技术栈": "FastAPI + React Native"}}


def test_search_filters_by_layer(workspace):
    """search(layer=...) 只在指定层内检索，其他层同关键词条目不返回。"""
    runtime = make_runtime(workspace)
    runtime.memory.remember("user", "语言偏好", "中文")
    runtime.memory.remember("project", "语言", "Python")

    only_project = runtime.memory.search("语言", layer="project")
    assert only_project == {"project": {"语言": "Python"}}

    assert runtime.memory.search("不存在的词") == {}


# ---------- recall_memory 工具（agent 主动按需召回记忆） ----------

async def test_recall_memory_tool_returns_matching_entries(workspace):
    """agent 调用 recall_memory 工具检索记忆：命中条目按层分组渲染返回给模型。"""
    runtime = make_runtime(workspace)
    runtime.memory.remember("user", "偏好-主题", "深色模式")
    events = []
    runtime.bus.subscribe("tool_result", lambda e: events.append(e))

    result = await runtime.run("查一下我的偏好", model_role="test-recall", session_id="r1")

    assert result.ok
    recall_results = [e for e in events if e.data["tool"] == "recall_memory"]
    assert recall_results and recall_results[0].data["ok"] is True
    assert "深色模式" in recall_results[0].data["output"]
    assert "[user memory]" in recall_results[0].data["output"]


async def test_recall_memory_tool_no_match_returns_friendly_text(workspace):
    """检索无命中时工具返回友好的未命中提示（而非报错），模型可据此继续。"""
    runtime = make_runtime(workspace)
    events = []
    runtime.bus.subscribe("tool_result", lambda e: events.append(e))

    result = await runtime.run("随便查查", model_role="test-recall", session_id="r2")

    assert result.ok
    recall_results = [e for e in events if e.data["tool"] == "recall_memory"]
    assert recall_results and "no memory entries match" in recall_results[0].data["output"]
