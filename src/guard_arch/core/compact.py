"""Conversation compaction: shrink long histories by summarizing old messages.

When a session's message history grows past a token threshold, the older part
of the history (everything except the most recent `keep_recent` messages) is
condensed by the model itself into a short summary; the compacted history
becomes [summary message] + [recent messages untouched]. This keeps long
sessions inside the context window while preserving recent conversational
fidelity.
"""

from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import Model

# 历史估算 token 超过该值才触发压缩（与 context.py 相同的粗略启发：~4 字符/token）
DEFAULT_COMPACTION_THRESHOLD_TOKENS = 8_000
# 压缩时保留末尾最近 N 条消息原文不动（保最近几轮的对话细节）
DEFAULT_KEEP_RECENT_MESSAGES = 6

_SUMMARY_PROMPT = (
    "Condense the following earlier conversation into a compact summary for an AI agent "
    "resuming this session. Preserve: user goals and requirements, decisions made, "
    "important facts/files mentioned, and any pending tasks. Be terse (a few bullet lines)."
)


def _approx_tokens(text: str) -> int:
    """Rough heuristic: ~4 chars per token for mixed CJK/English text."""
    return max(1, len(text) // 4)


def render_messages(messages: list[ModelMessage]) -> str:
    """Render message history as plain transcript lines ('User: …' / 'Assistant: …')."""
    lines: list[str] = []
    for message in messages:
        for part in message.parts:
            if isinstance(part, UserPromptPart):
                lines.append(f"User: {part.content}")
            elif isinstance(part, TextPart):
                lines.append(f"Assistant: {part.content}")
            elif isinstance(part, ToolCallPart):
                lines.append(f"Assistant called tool {part.tool_name}({part.args})")
            elif isinstance(part, ToolReturnPart):
                content = str(part.content)
                lines.append(f"Tool {part.tool_name} returned: {content[:500]}")
    return "\n".join(lines)


class HistoryCompactor:
    """Decides when a history is too long, and compacts it via a one-shot LLM summary."""

    def __init__(
        self,
        threshold_tokens: int = DEFAULT_COMPACTION_THRESHOLD_TOKENS,
        keep_recent: int = DEFAULT_KEEP_RECENT_MESSAGES,
    ):
        self.threshold_tokens = threshold_tokens
        self.keep_recent = keep_recent

    def needs_compaction(self, history: list[ModelMessage]) -> bool:
        """True when the rendered transcript exceeds the token threshold."""
        return _approx_tokens(render_messages(history)) > self.threshold_tokens

    async def compact(
        self, history: list[ModelMessage], model: Model
    ) -> list[ModelMessage]:
        """Summarize all but the last `keep_recent` messages; return compacted history.

        The returned history is [one synthetic user message containing the summary]
        + [the untouched recent tail], so the model resumes with a short brief of
        the past plus the fresh recent exchange.
        """
        old = history[: -self.keep_recent] if len(history) > self.keep_recent else []
        recent = history[-self.keep_recent :] if len(history) > self.keep_recent else history
        if not old:
            return history

        summarizer: Agent[None, str] = Agent(model, system_prompt=_SUMMARY_PROMPT)
        result = await summarizer.run(
            "Summarize this earlier conversation history:\n\n" + render_messages(old)
        )
        summary_message = ModelRequest(
            parts=[
                UserPromptPart(
                    content=(
                        "[Summary of the earlier conversation in this session, "
                        "auto-compacted to save context]\n" + str(result.output)
                    )
                )
            ]
        )
        return [summary_message, *recent]
