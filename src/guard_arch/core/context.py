"""Context engine: assembles the system prompt with a token budget."""

import platform
import sys
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from guard_arch.core.agent import AgentDefinition
    from guard_arch.core.memory import MemoryManager
    from guard_arch.core.rules import RulesRegistry
    from guard_arch.core.skill import SkillManifest
    from guard_arch.core.tool import Tool
    from guard_arch.core.workspace import Workspace

DEFAULT_TOKEN_BUDGET = 16_000


def _approx_tokens(text: str) -> int:
    # Rough heuristic: ~4 chars per token for mixed CJK/English prompts.
    return max(1, len(text) // 4)


# 工作区指令文件：项目级约定（类似 CLAUDE.md），存在即注入 system prompt
WORKSPACE_INSTRUCTION_FILES = ("GUARD.md", "AGENTS.md", "CLAUDE.md")


class ContextEngine:
    def __init__(self, token_budget: int = DEFAULT_TOKEN_BUDGET, rules: "RulesRegistry | None" = None):
        self.token_budget = token_budget
        # 核心行为规则（harness 宪法）：缺省用代码默认规则
        self.rules = rules

    def workspace_instructions(self, workspace: "Workspace") -> str:
        """读取工作区根目录下的指令文件（GUARD.md/AGENTS.md/CLAUDE.md），合并为一段。"""
        parts: list[str] = []
        for name in WORKSPACE_INSTRUCTION_FILES:
            path = workspace.root / name
            if path.is_file():
                try:
                    content = path.read_text(encoding="utf-8").strip()
                except (OSError, UnicodeDecodeError):
                    continue
                if content:
                    parts.append(f"[{name}]\n{content}")
        return "\n\n".join(parts)

    def environment_info(self, workspace: "Workspace") -> str:
        return (
            f"- OS: {platform.system()} {platform.release()}\n"
            f"- Shell: pwsh (Windows)\n"
            f"- Python: {sys.version.split()[0]}\n"
            f"- Workspace: {workspace.root}\n"
            f"- Date: {datetime.now().astimezone().isoformat(timespec='seconds')}"
        )

    def build_system_prompt(
        self,
        agent: "AgentDefinition",
        skills: list["SkillManifest"],
        memory: "MemoryManager",
        workspace: "Workspace",
        tools: "list[Tool] | None" = None,
        memory_scope: str = "",
    ) -> str:
        # 核心行为规则来自 RulesRegistry（代码持有 + rules.yaml 可控），
        # 与用户可编辑的 agent 人设（instructions）分层，不会被随手改掉
        rules_text = self.rules.render() if self.rules is not None else ""
        base = f"You are {agent.name}, an AI agent running inside Guard Arch.\n"
        if rules_text:
            base += f"核心行为规则（必须遵守）：\n{rules_text}"
        sections: list[tuple[str, str]] = [("base", base)]
        if agent.instructions:
            sections.append(("instructions", agent.instructions.strip()))
        # 能力面显性化：把该 agent 实际被授予的工具（名称+用途）注入 prompt，
        # 模型据此知道自己"能做什么"，在需要时自主调用，而不是被裸问裸答
        tool_lines = [f"- {t.name}: {t.description}" for t in (tools or [])]
        if tool_lines:
            sections.append(
                (
                    "tools",
                    "## 你可用的工具（需要时主动调用，不要假装没有能力）\n" + "\n".join(tool_lines),
                )
            )
        for skill in skills:
            body = skill.instructions.strip()
            if body:
                sections.append((f"skill:{skill.name}", f"## Skill: {skill.name}\n{body}"))
        # 项目级指令（工作区指令文件）：团队/项目约定，优先级高于通用 base
        workspace_text = self.workspace_instructions(workspace)
        if workspace_text:
            sections.append(("workspace", f"## Project Instructions\n{workspace_text}"))
        memory_text = memory.context_snippet(scope=memory_scope)
        if memory_text:
            sections.append(("memory", f"## Memory\n{memory_text}"))
        sections.append(
            ("environment", f"## Environment\n{self.environment_info(workspace)}")
        )

        budget = self.token_budget
        out: list[str] = []
        for name, text in sections:
            cost = _approx_tokens(text)
            if name == "base" or cost <= budget:
                out.append(text)
                budget -= cost
            # over-budget optional sections are dropped (base always kept)
        return "\n\n".join(out)
