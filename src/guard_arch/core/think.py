"""Harness-driven thinking phase: the framework makes the model analyze first.

Before the main agent run, the runtime performs one explicit thinking step:
the model analyzes the user's request against the agent's capabilities and
produces a short analysis (what's wanted, what's needed, how to proceed).
The analysis is streamed as typed `thinking` delta events (visible to clients)
and injected into the main run's input so it guides the execution chain.

This is harness thinking — a framework-driven phase — distinct from the
model's internal reasoning tokens (which may be disabled entirely).
"""

from collections.abc import Awaitable, Callable

from pydantic_ai import Agent
from pydantic_ai.messages import PartDeltaEvent, TextPartDelta
from pydantic_ai.models import Model

_THINK_PROMPT = (
    "You are the thinking phase of an AI agent harness. Analyze the user's request "
    "and produce a SHORT working analysis (2-4 sentences, same language as the user): "
    "what the user actually wants, whether external capabilities are needed "
    "(memory recall, web search/fetch, tools, asking the user), and the immediate "
    "next step. Be concrete and terse — this analysis guides the execution phase."
)


async def think(
    message: str,
    model: Model,
    *,
    capabilities: str = "",
    context: str = "",
    on_delta: Callable[[str], Awaitable[None]] | None = None,
) -> str:
    """Run the harness thinking phase for one user message; return the analysis text.

    on_delta: 传入时思考内容流式下发（每个增量回调一次）；模型不支持流式时
    由 pydantic-ai 用完整响应合成增量事件，拿不到增量则兜底一次性回调全文。
    """
    parts = [f"User's request:\n{message}"]
    if context:
        parts.append(f"Recent conversation context:\n{context}")
    if capabilities:
        parts.append(f"Available capabilities:\n{capabilities}")
    thinker: Agent[None, str] = Agent(model, system_prompt=_THINK_PROMPT)

    if on_delta is None:
        result = await thinker.run("\n\n".join(parts))
        return str(result.output).strip()

    streamed = False

    async def stream_handler(_ctx, events) -> None:
        nonlocal streamed
        async for event in events:
            if isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
                delta = event.delta.content_delta
                if delta:
                    streamed = True
                    await on_delta(delta)

    result = await thinker.run("\n\n".join(parts), event_stream_handler=stream_handler)
    text = str(result.output).strip()
    if not streamed and text:
        await on_delta(text)
    return text
