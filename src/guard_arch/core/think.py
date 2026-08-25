"""Harness-driven thinking phase: the framework makes the model analyze first.

Before the main agent run, the runtime performs one explicit thinking step:
the model analyzes the user's request against the agent's capabilities and
produces a short analysis (what's wanted, what's needed, how to proceed).
The analysis is emitted as typed `thinking` events (visible to clients) and
injected into the main run's input so it guides the execution chain.

This is harness thinking — a framework-driven phase — distinct from the
model's internal reasoning tokens (which may be disabled entirely).
"""

from pydantic_ai import Agent
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
) -> str:
    """Run the harness thinking phase for one user message; return the analysis text."""
    parts = [f"User's request:\n{message}"]
    if context:
        parts.append(f"Recent conversation context:\n{context}")
    if capabilities:
        parts.append(f"Available capabilities:\n{capabilities}")
    thinker: Agent[None, str] = Agent(model, system_prompt=_THINK_PROMPT)
    result = await thinker.run("\n\n".join(parts))
    return str(result.output).strip()
