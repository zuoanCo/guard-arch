"""Context engine: assembles the system prompt with a token budget."""

import platform
import sys
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from guard_arch.core.agent import AgentDefinition
    from guard_arch.core.memory import MemoryManager
    from guard_arch.core.skill import SkillManifest
    from guard_arch.core.tool import Tool
    from guard_arch.core.workspace import Workspace

DEFAULT_TOKEN_BUDGET = 16_000


def _approx_tokens(text: str) -> int:
    # Rough heuristic: ~4 chars per token for mixed CJK/English prompts.
    return max(1, len(text) // 4)


class ContextEngine:
    def __init__(self, token_budget: int = DEFAULT_TOKEN_BUDGET):
        self.token_budget = token_budget

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
    ) -> str:
        base = (
            f"You are {agent.name}, an AI agent running inside Guard Arch.\n"
            "工作方式：先分析用户的真实需求——需求完整、目标明确时直接高效执行；"
            "需求不完整、缺少关键信息或存在多种合理解读（分支）时，"
            "先用一两个精准的问题与用户确认关键点，确认清楚后再行动，不要靠猜；"
            "需要外部信息、执行动作或回忆事实时，主动使用下方列出的工具；"
            "避免不必要的步骤和啰嗦，用尽量少的动作高效解决问题。"
        )
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
        memory_text = memory.context_snippet()
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
