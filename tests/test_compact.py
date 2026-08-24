"""Tests for conversation history compaction (long sessions get summarized)."""

import pytest
from pydantic_ai.messages import (
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.models.test import TestModel

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


def _fake_history(pairs: int, padding: int = 200) -> list:
    """Build a fake conversation history of `pairs` user/assistant rounds,
    each message padded with `padding` chars to control its token size."""
    history = []
    for i in range(pairs):
        history.append(ModelRequest(parts=[UserPromptPart(content=f"问题{i} " + "x" * padding)]))
        history.append(ModelResponse(parts=[TextPart(content=f"回答{i} " + "y" * padding)]))
    return history


async def test_long_history_is_compacted_before_run(workspace):
    """History over the token threshold gets compacted: old messages are replaced
    by a model-written summary, recent tail is kept, and a history_compacted event fires."""
    # 压缩阈值调低到 100 token，让几十条 padded 消息的历史必然超阈值触发压缩
    runtime = make_runtime(workspace, compaction_threshold_tokens=100)
    # 预置一段 30 轮（60 条消息）的超长会话历史
    runtime.memory.save_conversation_state(
        "long-session", ModelMessagesTypeAdapter.dump_python(_fake_history(30))
    )
    events = []
    runtime.bus.subscribe("history_compacted", lambda e: events.append(e))

    result = await runtime.run("继续聊", session_id="long-session", model_role="test-plain")

    assert result.ok
    # 触发了压缩事件（事件里带压缩前后的历史条数）
    assert events, "expected history_compacted event for over-threshold history"
    assert events[0].data["before"] == 60
    assert events[0].data["after"] < 60
    # 压缩后存回的会话状态也应比原始 60 条短（摘要 + 最近若干条 + 本轮新消息）
    state = runtime._load_history("long-session")
    assert state is not None and len(state) < 60


async def test_short_history_is_not_compacted(workspace):
    """History under the threshold passes through untouched: no compaction, no event."""
    runtime = make_runtime(workspace, compaction_threshold_tokens=100)
    runtime.memory.save_conversation_state(
        "short-session", ModelMessagesTypeAdapter.dump_python(_fake_history(2, padding=5))
    )
    events = []
    runtime.bus.subscribe("history_compacted", lambda e: events.append(e))

    result = await runtime.run("继续聊", session_id="short-session", model_role="test-plain")

    assert result.ok
    assert not events, "short history must not trigger compaction"


async def test_compaction_summary_prepended_and_recent_tail_kept(workspace):
    """Unit-level: compact() replaces old messages with one summary message and
    keeps the last keep_recent messages verbatim."""
    from guard_arch.core.compact import HistoryCompactor

    compactor = HistoryCompactor(threshold_tokens=1, keep_recent=4)
    history = _fake_history(10, padding=50)  # 20 条消息
    model = TestModel(custom_output_text="早前对话摘要")

    compacted = await compactor.compact(history, model)

    # 压缩结果 = 1 条摘要消息 + 末尾 4 条原文（keep_recent）
    assert len(compacted) == 1 + 4
    assert compacted[0].parts[0].content.startswith("[Summary of the earlier conversation")
    assert "早前对话摘要" in compacted[0].parts[0].content
    # 末尾 4 条与原始历史末尾 4 条一致（最近对话细节不丢）
    assert compacted[1:] == history[-4:]
