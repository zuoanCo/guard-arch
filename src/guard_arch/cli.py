"""Interactive CLI for Guard Arch."""

import argparse
import asyncio
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Confirm

from guard_arch.core.model import ModelConfigError
from guard_arch.events.bus import Event
from guard_arch.runtime import AgentRuntime

console = Console()


@dataclass
class CLIContext:
    runtime: AgentRuntime
    agent_id: str = "assistant"
    session_id: str = "default"
    model_role: str | None = None
    streamed_text: str = field(default="", repr=False)


HELP_TEXT = """\
/help            显示帮助
/model [role]    查看或切换模型角色（default/reasoning/cheap/coding/test/...）
/skills          列出已加载的 skills
/agents          列出已加载的 agents
/clear           清空当前会话的对话历史
/exit            退出
"""


def summarize_args(args: dict) -> str:
    parts = []
    for key, value in args.items():
        text = str(value)
        if len(text) > 60:
            text = text[:57] + "..."
        parts.append(f"{key}={text}")
    return " ".join(parts)


def handle_slash_command(text: str, ctx: CLIContext) -> tuple[str, bool]:
    """Handle a slash command. Returns (output_text, should_exit)."""
    command, _, arg = text.partition(" ")
    command = command.lower()
    arg = arg.strip()

    if command == "/help":
        return HELP_TEXT, False
    if command == "/model":
        if not arg:
            current = ctx.model_role or ctx.runtime.agent_registry.get(ctx.agent_id).model
            return f"当前模型角色: {current}（可用: {', '.join(ctx.runtime.model_router.role_names())}）", False
        if arg not in ctx.runtime.model_router.role_names():
            return f"未知模型角色: {arg}（可用: {', '.join(ctx.runtime.model_router.role_names())}）", False
        ctx.model_role = arg
        return f"已切换模型角色: {arg}", False
    if command == "/skills":
        lines = [
            f"- {skill.name}: {skill.description}" for skill in ctx.runtime.skill_registry.all()
        ]
        return "\n".join(lines) or "(no skills)", False
    if command == "/agents":
        lines = [
            f"- {a.id}: {a.name} (model={a.model}, skills={a.skills})"
            for a in ctx.runtime.agent_registry.all()
        ]
        return "\n".join(lines) or "(no agents)", False
    if command == "/clear":
        ctx.runtime.memory.clear_conversation(ctx.session_id)
        return "对话历史已清空。", False
    if command in ("/exit", "/quit"):
        return "再见！", True
    return f"未知命令: {command}（输入 /help 查看帮助）", False


def make_approval_handler(interactive: bool):
    def approve(tool_name: str, args: dict, reason: str) -> bool:
        console.print(
            f"[yellow]⚠ 权限请求[/yellow] {tool_name} {summarize_args(args)} [dim]({reason})[/dim]"
        )
        if not interactive:
            console.print("[red]非交互模式，已拒绝。可用 --auto-approve 自动允许。[/red]")
            return False
        try:
            return Confirm.ask("允许执行？", default=False)
        except (EOFError, KeyboardInterrupt):
            console.print("[red]已拒绝。[/red]")
            return False

    return approve


def wire_display(ctx: CLIContext) -> None:
    def on_message_delta(event: Event) -> None:
        delta = event.data.get("delta", "")
        ctx.streamed_text += delta
        console.print(delta, end="", highlight=False, markup=False)

    def on_tool_call(event: Event) -> None:
        console.print(
            f"\n[green]✓[/green] [cyan]{event.data['tool']}[/cyan] "
            f"[dim]{summarize_args(event.data.get('args', {}))}[/dim]"
        )

    def on_tool_result(event: Event) -> None:
        status = "[green]ok[/green]" if event.data.get("ok") else "[red]failed[/red]"
        console.print(f"  [dim]└─ {status}[/dim]")

    def on_error(event: Event) -> None:
        console.print(f"\n[red]错误: {event.data.get('error')}[/red]")

    ctx.runtime.bus.subscribe("message_delta", on_message_delta)
    ctx.runtime.bus.subscribe("tool_call", on_tool_call)
    ctx.runtime.bus.subscribe("tool_result", on_tool_result)
    ctx.runtime.bus.subscribe("error", on_error)


async def run_turn(ctx: CLIContext, message: str) -> bool:
    ctx.streamed_text = ""
    result = await ctx.runtime.run(
        message,
        agent_id=ctx.agent_id,
        session_id=ctx.session_id,
        model_role=ctx.model_role,
    )
    if not result.ok:
        return False
    output = result.output
    remainder = output[len(ctx.streamed_text):] if output.startswith(ctx.streamed_text) else output
    if remainder:
        if ctx.streamed_text:
            console.print(remainder, end="", highlight=False, markup=False)
        else:
            console.print(Markdown(output))
    console.print()
    return True


async def interactive_loop(ctx: CLIContext) -> None:
    console.print(
        f"[bold]Guard Arch[/bold] [dim]v{__import__('guard_arch').__version__}[/dim] — "
        f"agent={ctx.agent_id}, session={ctx.session_id}\n"
        f"[dim]输入 /help 查看命令，/exit 退出。[/dim]"
    )
    while True:
        try:
            text = console.input("\n[bold blue]你 ❯[/bold blue] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n再见！")
            return
        if not text:
            continue
        if text.startswith("/"):
            output, should_exit = handle_slash_command(text, ctx)
            console.print(output)
            if should_exit:
                return
            continue
        await run_turn(ctx, text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="guard", description="Guard Arch CLI")
    parser.add_argument("--workspace", default=".", help="工作区根目录（默认当前目录）")
    parser.add_argument("--agent", default="assistant", help="agent id（默认 assistant）")
    parser.add_argument("--model", default=None, help="模型角色覆盖（如 test/default/coding）")
    parser.add_argument("--session", default="default", help="会话 id（对话历史按会话持久化）")
    parser.add_argument("--message", default=None, help="非交互模式：跑一轮后退出")
    parser.add_argument("--auto-approve", action="store_true", help="自动允许所有权限请求")
    return parser


def main(argv: list[str] | None = None) -> int:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stdin.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass
    args = build_parser().parse_args(argv)
    interactive = args.message is None
    try:
        runtime = AgentRuntime(
            Path(args.workspace),
            auto_approve=args.auto_approve,
            approval_handler=make_approval_handler(interactive),
        )
    except ModelConfigError as exc:
        console.print(f"[red]配置错误: {exc}[/red]")
        return 2
    ctx = CLIContext(
        runtime=runtime,
        agent_id=args.agent,
        session_id=args.session or f"s-{uuid.uuid4().hex[:8]}",
        model_role=args.model,
    )
    wire_display(ctx)
    try:
        if args.message is not None:
            ok = asyncio.run(run_turn(ctx, args.message))
            return 0 if ok else 1
        asyncio.run(interactive_loop(ctx))
        return 0
    finally:
        runtime.memory.close()


if __name__ == "__main__":
    raise SystemExit(main())
