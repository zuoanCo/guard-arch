"""Terminal tool. Commands run in the workspace directory; the permission
engine gates execution before this handler is invoked."""

import subprocess

from guard_arch.core.tool import Tool
from guard_arch.core.workspace import Workspace

MAX_OUTPUT_CHARS = 30_000


def make_terminal_tools(workspace: Workspace) -> list[Tool]:
    def run_command(command: str, timeout: int = 60) -> str:
        """Run a shell command in the workspace directory and return its output."""
        timeout = max(1, min(int(timeout), 300))
        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=workspace.root,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return f"Error: command timed out after {timeout}s"
        except OSError as exc:
            return f"Error: {exc}"
        output = (proc.stdout or "") + (proc.stderr or "")
        if len(output) > MAX_OUTPUT_CHARS:
            output = output[:MAX_OUTPUT_CHARS] + f"\n... [truncated at {MAX_OUTPUT_CHARS} chars]"
        return f"exit_code={proc.returncode}\n{output.strip()}"

    return [
        Tool(
            "run_command",
            "Run a shell command in the workspace (dangerous commands are blocked, "
            "others may require user confirmation)",
            run_command,
        )
    ]
