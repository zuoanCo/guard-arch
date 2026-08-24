"""Configuration-driven agent definitions loaded from agents/*.yaml."""

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class AgentDefinition(BaseModel):
    id: str
    name: str
    model: str = "default"  # model role name, resolved by the ModelRouter
    skills: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    instructions: str = ""
    # 需求分析门禁：开启后每次 run 先做一次结构化分析（core/intake.py），
    # 需求不清晰时短路返回澄清问题，不进入主执行链路
    intake: bool = False


class AgentRegistry:
    def __init__(self, agents_dirs: list[str | Path]):
        self._agents: dict[str, AgentDefinition] = {}
        for d in agents_dirs:
            self.load_dir(d)

    def load_dir(self, directory: str | Path) -> None:
        directory = Path(directory)
        if not directory.is_dir():
            return
        for path in sorted(directory.glob("*.yaml")) + sorted(directory.glob("*.yml")):
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            definition = AgentDefinition(**data)
            self._agents[definition.id] = definition

    def register(self, definition: AgentDefinition) -> AgentDefinition:
        self._agents[definition.id] = definition
        return definition

    def get(self, agent_id: str) -> AgentDefinition:
        try:
            return self._agents[agent_id]
        except KeyError:
            raise KeyError(
                f"unknown agent: {agent_id!r} (registered: {sorted(self._agents)})"
            ) from None

    def all(self) -> list[AgentDefinition]:
        return list(self._agents.values())
