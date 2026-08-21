import pytest

from guard_arch.core.memory import MemoryManager


@pytest.fixture
def memory(tmp_path):
    m = MemoryManager(tmp_path)
    yield m
    m.close()


def test_db_created_under_workspace(tmp_path, memory):
    assert memory.db_path == tmp_path / ".guard_arch" / "memory.db"
    assert memory.db_path.exists()


def test_kv_layers_roundtrip(memory):
    memory.remember("user", "editor", "vscode")
    memory.remember("project", "test_command", "pnpm test")
    memory.remember("agent", "lesson", "always read before write")
    assert memory.recall("user") == {"editor": "vscode"}
    assert memory.recall("project", "test_command") == {"test_command": "pnpm test"}
    assert memory.recall("agent")["lesson"] == "always read before write"


def test_remember_overwrites(memory):
    memory.remember("project", "k", "v1")
    memory.remember("project", "k", "v2")
    assert memory.recall("project", "k") == {"k": "v2"}


def test_unknown_layer_rejected(memory):
    with pytest.raises(ValueError, match="unknown memory layer"):
        memory.remember("bogus", "k", "v")


def test_context_snippet_includes_all_layers(memory):
    memory.remember("user", "lang", "中文")
    memory.remember("project", "framework", "fastapi")
    snippet = memory.context_snippet()
    assert "[user memory]" in snippet
    assert "lang: 中文" in snippet
    assert "[project memory]" in snippet
    assert "framework: fastapi" in snippet
    assert "[agent memory]" not in snippet  # empty layers omitted


def test_conversation_messages(memory):
    memory.add_message("s1", "user", "你好")
    memory.add_message("s1", "assistant", "你好！")
    memory.add_message("s2", "user", "其他会话")
    messages = memory.recent_messages("s1")
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert messages[0]["content"] == "你好"
    assert len(memory.recent_messages("s1", limit=1)) == 1
    assert len(memory.recent_messages("s2")) == 1


def test_conversation_state_roundtrip_and_clear(memory):
    memory.save_conversation_state("s1", [{"kind": "request"}])
    assert memory.load_conversation_state("s1") == [{"kind": "request"}]
    memory.clear_conversation("s1")
    assert memory.load_conversation_state("s1") is None
    assert memory.recent_messages("s1") == []
