from __future__ import annotations

from typing import Sequence

from ..perception.state import PageState
from .core import AgentAction, AgentGoal, AgentTransition


class SettingsPlanner:
    """A deterministic first planner for safe navigation inside Settings."""

    def __init__(self, max_scrolls: int = 3) -> None:
        self.max_scrolls = max_scrolls

    def plan(
        self,
        goal: AgentGoal,
        state: PageState,
        history: Sequence[AgentTransition],
    ) -> AgentAction:
        if goal.is_satisfied(state):
            return AgentAction("finish", "目标页面已经通过 Activity 验证。")
        if state.current_app != goal.allowed_package:
            return AgentAction("abort", "当前页面已经离开允许的系统设置应用。")
        candidates = goal.clickable_candidates(state)
        if candidates:
            target = candidates[0]
            return AgentAction(
                "tap",
                f"当前页面存在允许点击的“{target}”。",
                target,
            )
        scrolls = sum(item.action.kind == "scroll_down" for item in history)
        if scrolls < self.max_scrolls:
            return AgentAction("scroll_down", "目标当前不可见，继续浏览设置列表。")
        return AgentAction("abort", "在安全浏览次数内没有找到目标设置项。")
