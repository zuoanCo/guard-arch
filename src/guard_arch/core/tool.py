"""Unified tool abstraction.

Native, MCP and plugin tools are all represented as a `Tool` with a name,
a description and a callable handler whose signature defines the input schema.

Production-grade fields:
- timeout_seconds: resource limit per call (kills hung tools)
- retry_attempts: silent retries on transient failures before surfacing
- verifier: post-execution check — the harness verifies the RESULT, not the
  model's claim (e.g. re-read a file after write to confirm it landed)
"""

import inspect
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

# 验证器签名：(调用参数, 工具输出) -> 验证结论文本（None 表示无需说明）
Verifier = Callable[[dict[str, Any], str], str | None | Awaitable[str | None]]

# 进度上报回调签名：(状态说明, 部分数据) -> None
ProgressReporter = Callable[[str, str | None], Awaitable[None]]

# 当前工具调用的进度上报通道（由 Runtime 派发层注入）。
# 用 ContextVar 而非 handler 参数传递：pydantic-ai 会从 handler 签名派生工具的
# input schema，任何额外参数都会泄露给模型；ContextVar 对 schema 完全不可见。
_progress_reporter: ContextVar[ProgressReporter | None] = ContextVar(
    "tool_progress_reporter", default=None
)


async def report_progress(note: str, data: str | None = None) -> None:
    """工具内部上报执行进度（开始 → 进行中 → 结束生命周期的"进行中"环节）。

    note: 人可读的状态说明（如"已建立连接""已接收 2.0 KB"）。
    data: 可选的部分数据（如已抓取的内容片段），供 UI 展示中间结果。
    无注入通道时（单测直接调 handler 等场景）静默为空操作。
    """
    reporter = _progress_reporter.get()
    if reporter is not None:
        await reporter(note, data)


@dataclass
class Tool:
    name: str
    description: str
    handler: Callable[..., Any]
    input_schema: dict[str, Any] = field(default_factory=dict)
    source: str = "native"  # native | mcp | plugin
    # 资源限制：单次调用超时（秒），超时按失败处理（防工具卡死拖住整个 run）
    timeout_seconds: float = 60.0
    # 纠正：瞬时故障（网络抖动/限流等）静默重试次数，0=不重试直接报错
    retry_attempts: int = 0
    # 验证：写操作等关键工具的完成后验证器（验证结果而非模型自述）
    verifier: Verifier | None = None

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
