"""Core rules: the harness-owned behavioral constitution applied to every agent.

Foundational operating rules live in CODE — not in user-editable agent YAML —
so they can't be casually changed or lost. They are still controllable:

- `config/rules.yaml` can disable a rule, replace its text, or add new rules
- CLI `/rules` lists the effective rules
- API `GET /api/v1/rules` exposes them

ContextEngine renders the enabled rules into every agent's system prompt.
"""

from pathlib import Path

import yaml
from pydantic import BaseModel


class CoreRule(BaseModel):
    """One foundational behavior rule. `id` is the stable handle for overrides."""

    id: str
    text: str
    enabled: bool = True


# 默认核心规则（harness 宪法）：代码持有，对所有 agent 生效
DEFAULT_RULES: list[CoreRule] = [
    CoreRule(
        id="analyze_first",
        text="先分析后行动：需求明确就直接高效执行；需求不完整、缺关键信息或有多种解读时，"
        "先用 ask_user_question 向用户确认关键点（一次问清，不连环追问），确认后再行动。",
    ),
    CoreRule(
        id="concise",
        text="简洁高效：不绕弯、不啰嗦，用尽量少的动作解决问题。",
    ),
    CoreRule(
        id="use_tools",
        text="主动使用工具：需要外部信息、执行动作或回忆事实时调用对应工具，"
        "不假装没有能力、不凭空编造。",
    ),
    CoreRule(
        id="recall_first",
        text="凡涉及用户自身的问题（名字、偏好、之前说过的事）：先查对话历史，"
        "历史没有就用 recall_memory 检索，绝不在没查过的情况下反问用户或说不知道。",
    ),
    CoreRule(
        id="realtime_web",
        text="凡需要实时/外部信息（天气、新闻、汇率、资料检索等）：必须先调用 web_search "
        "或 web_fetch 获取真实数据再回答，禁止直接说“无法获取实时信息”。",
    ),
    CoreRule(
        id="honest_failure",
        text="工具尝试失败（网络错误、无结果等）时，如实说明并给出可行的替代建议，不假装知道。",
    ),
    CoreRule(
        id="remember_facts",
        text="用户透露了值得长期记住的事实（偏好、姓名、习惯、项目信息）时，"
        "主动用 remember 存入 user memory，下次对话自然用上。",
    ),
    CoreRule(
        id="plan_multistep",
        text="多步任务（有多个环节的事）先用 todo_write 列简短计划，执行中及时更新状态。",
    ),
]


class RulesRegistry:
    """Effective rule set: code-held defaults + optional config/rules.yaml overrides.

    rules.yaml format:
        rules:
          concise: {enabled: false}            # 禁用某条规则
          realtime_web: {text: "替换文本"}      # 替换某条规则的文本
          my_new_rule: {text: "新增规则"}       # 追加新规则
    """

    def __init__(self, overrides_path: str | Path | None = None):
        self._rules: dict[str, CoreRule] = {r.id: r for r in DEFAULT_RULES}
        if overrides_path:
            self._apply_overrides(Path(overrides_path))

    def _apply_overrides(self, path: Path) -> None:
        if not path.exists():
            return
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for rule_id, override in (data.get("rules") or {}).items():
            if rule_id in self._rules:
                # 已有规则：按字段覆盖（禁用 / 换文本）
                self._rules[rule_id] = self._rules[rule_id].model_copy(update=override)
            else:
                # 新 id：作为新规则追加
                self._rules[rule_id] = CoreRule(id=rule_id, **override)

    def all(self) -> list[CoreRule]:
        return list(self._rules.values())

    def render(self) -> str:
        """把启用中的规则渲染为编号列表（注入 system prompt 用）。"""
        enabled = [r for r in self._rules.values() if r.enabled]
        return "\n".join(f"{i}. {rule.text}" for i, rule in enumerate(enabled, 1))
