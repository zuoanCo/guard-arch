"""Unified tool abstraction.

Native, MCP and plugin tools are all represented as a `Tool` with a name,
a description and a callable handler whose signature defines the input schema.
"""

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Tool:
    name: str
    description: str
    handler: Callable[..., Any]
    input_schema: dict[str, Any] = field(default_factory=dict)
    source: str = "native"  # native | mcp | plugin

    def schema(self) -> dict[str, Any]:
        if self.input_schema:
            return self.input_schema
        sig = inspect.signature(self.handler)
        return {
            "type": "object",
            "properties": {
                name: {"type": "string"}
                for name, param in sig.parameters.items()
                if param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)
            },
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> Tool:
        self._tools[tool.name] = tool
        return tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError:
            raise KeyError(f"unknown tool: {name!r} (registered: {sorted(self._tools)})") from None

    def has(self, name: str) -> bool:
        return name in self._tools

    def all(self) -> list[Tool]:
        return list(self._tools.values())

    def names(self) -> list[str]:
        return sorted(self._tools)
