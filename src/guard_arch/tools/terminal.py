"""Terminal tool. Commands run in the workspace directory; the permission
engine gates execution before this handler is invoked."""

import asyncio

from guard_arch.core.tool import Tool, report_progress
from guard_arch.core.workspace import Workspace

MAX_OUTPUT_CHARS = 30_000


def make_terminal_tools(workspace: Workspace) -> list[Tool]:
    async def run_command(command: str, timeout: int = 60) -> str:
        """Run a shell command in the workspace directory and return its output."""
        timeout = max(1, min(int(timeout), 300))
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                cwd=workspace.root,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
        except OSError as exc:
            return f"Error: {exc}"

        await report_progress(f"命令已启动：{command}")
        lines: list[str] = []

        async def read_output() -> None:
            assert proc.stdout is not None
            async for raw in proc.stdout:
                line = raw.decode("utf-8", errors="replace").rstrip()
                lines.append(line)
                # 每行输出都上报进度（携带该行内容供 UI 实时展示）
                await report_progress(f"运行中（已输出 {len(lines)} 行）", line)

        try:
            await asyncio.wait_for(read_output(), timeout=timeout)
        except TimeoutError:
            proc.kill()
            await proc.wait()
            return f"Error: command timed out after {timeout}s"
        await proc.wait()
        output = "\n".join(lines)
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
