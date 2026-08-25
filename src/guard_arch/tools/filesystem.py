"""Filesystem tools. Every path goes through the sandbox before use."""

import fnmatch

from guard_arch.core.tool import Tool
from guard_arch.core.workspace import SandboxError, Workspace

MAX_READ_CHARS = 50_000


def _error(exc: Exception) -> str:
    return f"Error: {exc}"


def make_filesystem_tools(workspace: Workspace) -> list[Tool]:
    def read_file(path: str) -> str:
        """Read a UTF-8 text file inside the workspace."""
        try:
            resolved = workspace.resolve(path)
            text = resolved.read_text(encoding="utf-8")
        except (SandboxError, OSError, UnicodeDecodeError) as exc:
            return _error(exc)
        if len(text) > MAX_READ_CHARS:
            return text[:MAX_READ_CHARS] + f"\n... [truncated at {MAX_READ_CHARS} chars]"
        return text

    def write_file(path: str, content: str) -> str:
        """Write (create or overwrite) a UTF-8 text file inside the workspace."""
        try:
            resolved = workspace.resolve(path)
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(content, encoding="utf-8")
        except (SandboxError, OSError) as exc:
            return _error(exc)
        return f"wrote {len(content)} chars to {path}"

    def edit_file(path: str, old_string: str, new_string: str) -> str:
        """Replace the first occurrence of old_string with new_string in a file."""
        try:
            resolved = workspace.resolve(path)
            text = resolved.read_text(encoding="utf-8")
        except (SandboxError, OSError, UnicodeDecodeError) as exc:
            return _error(exc)
        if old_string not in text:
            return f"Error: old_string not found in {path}"
        count = text.count(old_string)
        resolved.write_text(text.replace(old_string, new_string, 1), encoding="utf-8")
        note = f" ({count - 1} more occurrence(s) left unchanged)" if count > 1 else ""
        return f"edited {path}{note}"

    def list_directory(path: str = ".") -> str:
        """List entries of a directory inside the workspace (dirs end with /)."""
        try:
            resolved = workspace.resolve(path)
            if not resolved.is_dir():
                return f"Error: not a directory: {path}"
            entries = sorted(resolved.iterdir(), key=lambda p: (p.is_file(), p.name))
        except (SandboxError, OSError) as exc:
            return _error(exc)
        lines = [entry.name + ("/" if entry.is_dir() else "") for entry in entries]
        return "\n".join(lines) if lines else "(empty directory)"

    def search_text(pattern: str, path: str = ".", glob: str = "*") -> str:
        """Search file contents for a substring under a workspace directory."""
        try:
            root = workspace.resolve(path)
        except SandboxError as exc:
            return _error(exc)
        if not root.is_dir():
            return f"Error: not a directory: {path}"
        matches: list[str] = []
        for file in sorted(root.rglob("*")):
            if not file.is_file() or not fnmatch.fnmatch(file.name, glob):
                continue
            if ".guard_arch" in file.parts or ".venv" in file.parts:
                continue
            try:
                for lineno, line in enumerate(file.read_text(encoding="utf-8").splitlines(), 1):
                    if pattern in line:
                        rel = file.relative_to(workspace.root)
                        matches.append(f"{rel}:{lineno}: {line.strip()[:200]}")
            except (OSError, UnicodeDecodeError):
                continue
            if len(matches) >= 100:
                return "\n".join(matches) + "\n... [truncated at 100 matches]"
        return "\n".join(matches) if matches else "(no matches)"

    # 验证器：写操作完成后回读文件确认结果（验证执行结果，而非模型自述"写好了"）
    def verify_write(args: dict, output: str) -> str | None:
        if output.startswith("Error:"):
            return None  # 写入本身失败，无需验证
        path = args.get("path", "")
        try:
            resolved = workspace.resolve(path)
            if not resolved.exists():
                return f"文件 {path} 写入后不存在"
            # 写入内容为空但文件非空（或反之）视为不一致
            if args.get("content") is not None and resolved.read_text(encoding="utf-8") != args["content"]:
                return f"文件 {path} 回读内容与写入不一致"
        except (SandboxError, OSError, UnicodeDecodeError) as exc:
            return f"回读验证失败: {exc}"
        return None

    def verify_edit(args: dict, output: str) -> str | None:
        if output.startswith("Error:"):
            return None
        path = args.get("path", "")
        try:
            resolved = workspace.resolve(path)
            text = resolved.read_text(encoding="utf-8")
            if args.get("new_string") and args["new_string"] not in text:
                return f"文件 {path} 中未找到替换后的内容"
        except (SandboxError, OSError, UnicodeDecodeError) as exc:
            return f"回读验证失败: {exc}"
        return None

    return [
        Tool("read_file", "Read a UTF-8 text file inside the workspace", read_file),
        Tool(
            "write_file",
            "Write a UTF-8 text file inside the workspace",
            write_file,
            verifier=verify_write,
        ),
        Tool(
            "edit_file",
            "Replace a string occurrence in a workspace file",
            edit_file,
            verifier=verify_edit,
        ),
        Tool("list_directory", "List a workspace directory", list_directory),
        Tool("search_text", "Substring-search workspace files", search_text),
    ]
