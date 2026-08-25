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


class RiskLevel(StrEnum):
    """工具调用的风险评级：决定默认处置方式与 auto_approve 的覆盖范围。

    LOW  只读/可逆操作（读文件、搜索、查记忆）：默认 ALLOW，auto_approve 覆盖
    MID  有副作用但可逆/有边界（写文件、联网、跑普通命令）：默认 ASK，auto_approve 覆盖
    HIGH 不可逆/高危（删除、git push --force、格式化）：默认 DENY，auto_approve 不可覆盖
    """

    LOW = "low"
    MID = "mid"
    HIGH = "high"


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
    risk: RiskLevel = RiskLevel.MID  # 风险评级（默认 MID）

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
        # HIGH 高危：不可逆破坏性命令，永远拒绝（auto_approve 不可覆盖）
        PermissionRule(PermissionDecision.DENY, "run_command", pattern, risk=RiskLevel.HIGH)
        for pattern in DANGEROUS_COMMAND_PATTERNS
    ]
    rules += [
        # LOW 低风险：只读/可逆操作，默认放行
        PermissionRule(PermissionDecision.ALLOW, "read_file", risk=RiskLevel.LOW),
        PermissionRule(PermissionDecision.ALLOW, "list_directory", risk=RiskLevel.LOW),
        PermissionRule(PermissionDecision.ALLOW, "search_text", risk=RiskLevel.LOW),
        PermissionRule(PermissionDecision.ALLOW, "recall_memory", risk=RiskLevel.LOW),
        PermissionRule(PermissionDecision.ALLOW, "list_capabilities", risk=RiskLevel.LOW),
        PermissionRule(PermissionDecision.ALLOW, "todo_read", risk=RiskLevel.LOW),
        # MID 中风险：有副作用但有边界（沙箱内写入/联网/自我管理），默认放行、CLI 可改 ASK
        PermissionRule(PermissionDecision.ALLOW, "web_search", risk=RiskLevel.MID),
        PermissionRule(PermissionDecision.ALLOW, "web_fetch", risk=RiskLevel.MID),
        PermissionRule(PermissionDecision.ALLOW, "write_file", risk=RiskLevel.MID),
        PermissionRule(PermissionDecision.ALLOW, "edit_file", risk=RiskLevel.MID),
        PermissionRule(PermissionDecision.ALLOW, "remember", risk=RiskLevel.MID),
        PermissionRule(PermissionDecision.ALLOW, "todo_write", risk=RiskLevel.MID),
        PermissionRule(PermissionDecision.ALLOW, "dispatch_agent", risk=RiskLevel.MID),
        PermissionRule(PermissionDecision.ALLOW, "ask_user_question", risk=RiskLevel.MID),
        # MID 中风险：普通 shell 命令需人工确认
        PermissionRule(PermissionDecision.ASK, "run_command", risk=RiskLevel.MID),
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
        risk = RiskLevel.MID  # 未命中规则的工具按中风险对待（fail-safe 默认）
        for rule in self.rules:
            if rule.matches(tool_name, args):
                decision = rule.decision
                risk = rule.risk
                break
        if decision is None:
            decision = PermissionDecision.ASK
        # HIGH 高危 / DENY 规则是绝对的：auto_approve 永不覆盖
        if decision is PermissionDecision.DENY or risk is RiskLevel.HIGH:
            return PermissionDecision.DENY
        if self.auto_approve:
            return PermissionDecision.ALLOW
        return decision

    def risk_of(self, tool_name: str, args: dict[str, Any]) -> RiskLevel:
        """返回某次工具调用的风险评级（未命中规则按 MID 对待）。"""
        for rule in self.rules:
            if rule.matches(tool_name, args):
                return rule.risk
        return RiskLevel.MID

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
