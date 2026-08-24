"""Intake gate: a structured pre-run analysis that decides the execution path.

Before the main agent run, the runtime makes one structured-output model call
(output is a typed IntakeAnalysis, not free text) and branches on the result:
- clarity == "clear"              -> proceed with the main run (optionally with plan)
- clarity == "needs_clarification"-> short-circuit: return the clarifying questions
  as this turn's reply WITHOUT starting the main execution chain.

This makes "clarify before execute" an architectural control flow enforced by
the runtime, not a prompt the model may ignore. Cost: one extra model call per
user turn; enable per agent via `intake: true` in the agent YAML.
"""

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models import Model


class IntakeAnalysis(BaseModel):
    """Structured verdict of the intake gate for one user message."""

    clarity: str = Field(
        description=(
            "'clear' if the request is complete and unambiguous enough to act on "
            "directly; 'needs_clarification' if key information is missing or the "
            "request has multiple plausible interpretations (branches)"
        )
    )
    summary: str = Field(description="one-line restatement of what the user actually wants")
    questions: list[str] = Field(
        default_factory=list,
        description=(
            "1-2 precise clarifying questions to ask the user when clarity is "
            "'needs_clarification'; empty list when clear"
        ),
    )
    plan: list[str] = Field(
        default_factory=list,
        description="short ordered execution steps for the request when clear; empty otherwise",
    )


_INTAKE_PROMPT = (
    "You are the intake gate of an AI agent. You will receive the user's latest "
    "message, optionally with recent conversation context and the agent's capability "
    "list. Decide whether the agent can act directly (clarity='clear') or must first "
    "ask the user 1-2 precise clarifying questions (clarity='needs_clarification').\n"
    "Rules:\n"
    "- If the need can be satisfied by the agent's capabilities (e.g. recalling memory, "
    "searching the web, fetching a URL, managing a todo list), choose 'clear' and let "
    "the agent act — never block a request the agent can handle with its tools.\n"
    "- If recent context already answers the ambiguity, choose 'clear'.\n"
    "- Only choose 'needs_clarification' when acting without the user's answer would "
    "risk doing the wrong thing; never ask about trivial preferences. Be terse."
)


async def analyze_request(
    message: str,
    model: Model,
    *,
    capabilities: str = "",
    context: str = "",
) -> IntakeAnalysis:
    """Run the structured intake analysis for one user message.

    Uses structured output (output_type=IntakeAnalysis), so the result is a
    validated typed object the runtime can branch on deterministically.
    `capabilities` (tool names/descriptions) and `context` (recent conversation)
    are provided so the gate doesn't block requests the agent could handle itself.
    """
    parts = [f"User's latest message:\n{message}"]
    if context:
        parts.append(f"Recent conversation context:\n{context}")
    if capabilities:
        parts.append(f"Agent's available capabilities:\n{capabilities}")
    gate: Agent[None, IntakeAnalysis] = Agent(
        model, system_prompt=_INTAKE_PROMPT, output_type=IntakeAnalysis
    )
    result = await gate.run("\n\n".join(parts))
    return result.output
