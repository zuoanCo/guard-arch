"""Permission engine: allow / ask / deny rules for tool calls."""

import fnmatch
import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class PermissionDecision(StrEnum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


# Commands that are never allowed to run.
DANGEROUS_COMMAND_PATTERNS = [
    r"\brm\s+[^\n|;&]*-[a-zA-Z]*r[a-zA-Z]*f",  # rm -rf / rm -fr
    r"\brm\s+[^\n|;&]*-[a-zA-Z]*f[a-zA-Z]*r",
    r"\brmdir\s+/s",
    r"\brd\s+/s",
    r"\bdel\s+[^\n|;&]*(/f|/s)",
    r"\bformat\s+[a-zA-Z]:",
    r"\bshutdown\b",
    r"\breboot\b",
    r"\bmkfs\b",
    r"\bdd\s+if=",
    r":\(\)\s*\{\s*:\|:",  # fork bomb :(){ :|:& };:
    r"\bRemove-Item\b[^\n|;&]*-Recurse[^\n|;&]*-Force",
    r"\bgit\s+push\b[^\n|;&]*--force",
]


@dataclass
class PermissionRule:
    decision: PermissionDecision
    tool: str = "*"  # fnmatch glob against the tool name
    pattern: str | None = None  # regex against the JSON-serialized arguments

    def matches(self, tool_name: str, args: dict[str, Any]) -> bool:
        if not fnmatch.fnmatchcase(tool_name, self.tool):
            return False
        if self.pattern is None:
            return True
        try:
            payload = json.dumps(args, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            payload = str(args)
        return re.search(self.pattern, payload, re.IGNORECASE | re.DOTALL) is not None


def default_rules() -> list[PermissionRule]:
    rules: list[PermissionRule] = [
        # Destructive shell commands: never allowed.
        PermissionRule(PermissionDecision.DENY, "run_command", pattern)
        for pattern in DANGEROUS_COMMAND_PATTERNS
    ]
    rules += [
        # Read-only filesystem tools are always safe inside the sandbox.
        PermissionRule(PermissionDecision.ALLOW, "read_file"),
        PermissionRule(PermissionDecision.ALLOW, "list_directory"),
        PermissionRule(PermissionDecision.ALLOW, "search_text"),
        # Writes are allowed inside the sandbox.
        PermissionRule(PermissionDecision.ALLOW, "write_file"),
        PermissionRule(PermissionDecision.ALLOW, "edit_file"),
        PermissionRule(PermissionDecision.ALLOW, "remember"),
        # Any other shell command needs confirmation.
        PermissionRule(PermissionDecision.ASK, "run_command"),
    ]
    return rules


#: async or sync callable: (tool_name, args, reason) -> bool
ApprovalCallback = Callable[[str, dict[str, Any], str], bool | Awaitable[bool]]


class PermissionEngine:
    """First matching rule wins; unknown tools default to ASK."""

    def __init__(
        self,
        rules: list[PermissionRule] | None = None,
        *,
        auto_approve: bool = False,
        approval_callback: ApprovalCallback | None = None,
    ):
        self.rules = rules if rules is not None else default_rules()
        self.auto_approve = auto_approve
        self.approval_callback = approval_callback

    def decide(self, tool_name: str, args: dict[str, Any]) -> PermissionDecision:
        decision: PermissionDecision | None = None
        for rule in self.rules:
            if rule.matches(tool_name, args):
                decision = rule.decision
                break
        if decision is None:
            decision = PermissionDecision.ASK
        # deny rules are absolute: auto_approve never overrides them
        if decision is PermissionDecision.DENY:
            return PermissionDecision.DENY
        if self.auto_approve:
            return PermissionDecision.ALLOW
        return decision

    async def authorize(self, tool_name: str, args: dict[str, Any]) -> bool:
        decision = self.decide(tool_name, args)
        if decision is PermissionDecision.ALLOW:
            return True
        if decision is PermissionDecision.DENY:
            return False
        if self.approval_callback is None:
            return False
        result = self.approval_callback(tool_name, args, "action requires confirmation")
        if isinstance(result, Awaitable):
            result = await result
        return bool(result)
