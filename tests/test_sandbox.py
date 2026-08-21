import pytest

from guard_arch.core.workspace import SandboxError, Workspace
from guard_arch.tools.filesystem import make_filesystem_tools


@pytest.fixture
def workspace(tmp_path):
    (tmp_path / "hello.txt").write_text("你好 guard_arch", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    return Workspace(tmp_path)


def _tool(workspace, name):
    return {t.name: t for t in make_filesystem_tools(workspace)}[name]


def test_resolve_inside_workspace(workspace):
    assert workspace.resolve("hello.txt") == workspace.root / "hello.txt"
    assert workspace.resolve(workspace.root / "sub") == workspace.root / "sub"


def test_resolve_rejects_escape(workspace):
    with pytest.raises(SandboxError):
        workspace.resolve("../outside.txt")
    with pytest.raises(SandboxError):
        workspace.resolve("sub/../../outside.txt")
    with pytest.raises(SandboxError):
        workspace.resolve("C:/Windows/System32/drivers")


def test_read_write_roundtrip(workspace):
    read_file = _tool(workspace, "read_file").handler
    write_file = _tool(workspace, "write_file").handler
    assert read_file("hello.txt") == "你好 guard_arch"
    assert write_file("new/notes.md", "# notes").startswith("wrote")
    assert read_file("new/notes.md") == "# notes"


def test_tools_reject_escape(workspace):
    read_file = _tool(workspace, "read_file").handler
    write_file = _tool(workspace, "write_file").handler
    assert read_file("../secret.txt").startswith("Error:")
    assert write_file("C:/Windows/evil.txt", "x").startswith("Error:")


def test_edit_file(workspace):
    edit_file = _tool(workspace, "edit_file").handler
    assert edit_file("hello.txt", "guard_arch", "Guard Arch").startswith("edited")
    assert _tool(workspace, "read_file").handler("hello.txt") == "你好 Guard Arch"
    assert edit_file("hello.txt", "not-there", "x").startswith("Error:")


def test_list_and_search(workspace):
    list_directory = _tool(workspace, "list_directory").handler
    search_text = _tool(workspace, "search_text").handler
    listing = list_directory(".")
    assert "hello.txt" in listing and "sub/" in listing
    assert "你好" in search_text("guard_arch", ".")
    assert search_text("no-such-needle", ".") == "(no matches)"
