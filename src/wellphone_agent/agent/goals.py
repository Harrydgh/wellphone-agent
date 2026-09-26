from __future__ import annotations

import re

from .core import AgentError, AgentGoal


SAFE_SETTINGS_GOALS = {
    "WLAN": AgentGoal(
        description="打开系统 WLAN 设置页面",
        allowed_package="com.android.settings",
        target_label="WLAN",
        success_activity_contains=("WifiSettings",),
    ),
}


def parse_safe_goal(task: str) -> AgentGoal:
    """Resolve natural language only to locally approved, verifiable goals."""

    normalized = re.sub(r"[\s_-]+", "", task).lower()
    if any(marker in normalized for marker in ("wlan", "wifi", "无线网络")):
        return SAFE_SETTINGS_GOALS["WLAN"]
    raise AgentError(
        "当前自然语言 Agent 只开放“打开 WLAN 设置”这一项安全任务。"
    )
