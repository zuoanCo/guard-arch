"""MCP integration.

Reads `config/mcp.json` (Claude Desktop `mcpServers` shape) and exposes the
servers as pydantic-ai toolsets attached to the agent. Missing config or
broken servers degrade to a warning, never a crash. MCP tool calls pass
through the PermissionEngine via `process_tool_call`.
"""

import json
import logging
from pathlib import Path
from typing import Any

from pydantic_ai.mcp import MCPToolset, StdioTransport
from pydantic_ai.toolsets.prefixed import PrefixedToolset

from guard_arch.permissions.engine import PermissionEngine

logger = logging.getLogger(__name__)


class MCPLoader:
    def __init__(self, config_path: str | Path, permission_engine: PermissionEngine):
        self.config_path = Path(config_path)
        self.permission_engine = permission_engine

    def load_toolsets(self) -> list[Any]:
        if not self.config_path.exists():
            logger.info("no MCP config at %s; skipping MCP servers", self.config_path)
            return []
        try:
            data = json.loads(self.config_path.read_text(encoding="utf-8"))
            servers = data.get("mcpServers")
            if not isinstance(servers, dict):
                raise ValueError("expected `mcpServers` object in MCP config")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            logger.warning("failed to load MCP config %s: %s", self.config_path, exc)
            return []

        toolsets: list[Any] = []
        for name, server in servers.items():
            try:
                toolsets.append(self._build_toolset(name, server))
            except Exception as exc:  # noqa: BLE001 - degrade, never crash
                logger.warning("skipping MCP server %r: %s", name, exc)
        return toolsets

    def _build_toolset(self, name: str, server: dict[str, Any]) -> Any:
        if "command" in server:
            transport = StdioTransport(
                command=server["command"],
                args=list(server.get("args") or []),
                env=server.get("env"),
                cwd=str(server["cwd"]) if server.get("cwd") is not None else None,
            )
            toolset = MCPToolset(transport, id=name, process_tool_call=self._guard)
        elif "url" in server:
            toolset = MCPToolset(
                server["url"], id=name, headers=server.get("headers"),
                process_tool_call=self._guard,
            )
        else:
            raise ValueError(f"MCP server {name!r} must have either `command` or `url`")
        return PrefixedToolset(toolset, name)

    async def _guard(self, ctx, call_tool, tool_name, args):
        allowed = await self.permission_engine.authorize(f"mcp:{tool_name}", args)
        if not allowed:
            return f"Error: permission denied for MCP tool {tool_name}"
        try:
            return await call_tool(tool_name, args)
        except Exception as exc:  # noqa: BLE001 - report to the model, don't crash the run
            logger.warning("MCP tool %s failed: %s", tool_name, exc)
            return f"Error: MCP tool {tool_name} failed: {exc}"
