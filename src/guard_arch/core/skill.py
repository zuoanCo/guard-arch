"""Skills: SKILL.md files with YAML frontmatter (name/description/tools) + Markdown instructions."""

import re
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?(.*)\Z", re.DOTALL)


class SkillManifest(BaseModel):
    name: str
    description: str = ""
    tools: list[str] = Field(default_factory=list)
    instructions: str = ""
    path: str = ""

    @classmethod
    def from_markdown(cls, text: str, *, path: str = "") -> "SkillManifest":
        match = _FRONTMATTER_RE.match(text)
        if not match:
            raise ValueError(f"SKILL.md has no YAML frontmatter: {path}")
        meta = yaml.safe_load(match.group(1)) or {}
        instructions = match.group(2).strip()
        return cls(**meta, instructions=instructions, path=path)


class SkillRegistry:
    def __init__(self, skills_dirs: list[str | Path]):
        self._skills: dict[str, SkillManifest] = {}
        for d in skills_dirs:
            self.load_dir(d)

    def load_dir(self, directory: str | Path) -> None:
        directory = Path(directory)
        if not directory.is_dir():
            return
        for path in sorted(directory.rglob("SKILL.md")):
            manifest = SkillManifest.from_markdown(
                path.read_text(encoding="utf-8"), path=str(path)
            )
            self._skills[manifest.name] = manifest

    def get(self, name: str) -> SkillManifest:
        try:
            return self._skills[name]
        except KeyError:
            raise KeyError(
                f"unknown skill: {name!r} (registered: {sorted(self._skills)})"
            ) from None

    def all(self) -> list[SkillManifest]:
        return list(self._skills.values())
