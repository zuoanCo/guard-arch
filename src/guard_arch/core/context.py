"""Context engine: assembles the system prompt with a token budget."""

import platform
import sys
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from guard_arch.core.agent import AgentDefinition
    from guard_arch.core.memory import MemoryManager
    from guard_arch.core.skill import SkillManifest
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
    ) -> str:
        base = (
            f"You are {agent.name}, an AI agent running inside Guard Arch.\n"
            "You can use the provided tools to inspect and modify files inside the workspace, "
            "run commands, and remember useful facts. All file paths are confined to the "
            "workspace. Think step by step, prefer reading before writing, and verify changes."
        )
        sections: list[tuple[str, str]] = [("base", base)]
        if agent.instructions:
            sections.append(("instructions", agent.instructions.strip()))
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
