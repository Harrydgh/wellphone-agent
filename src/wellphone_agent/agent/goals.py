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
    "BLUETOOTH": AgentGoal(
        description="打开系统蓝牙设置页面",
        allowed_package="com.android.settings",
        target_label="蓝牙",
        success_activity_contains=("Bluetooth", "SubSettings"),
        success_text_contains=("蓝牙",),
    ),
    "MOBILE_NETWORK": AgentGoal(
        description="打开系统移动网络设置页面",
        allowed_package="com.android.settings",
        target_label="移动网络",
        success_activity_contains=("MobileNetwork", "SimSettings", "SubSettings"),
        success_text_contains=("移动网络",),
        additional_allowed_packages=("com.android.phone",),
    ),
    "MY_DEVICE": AgentGoal(
        description="打开系统我的设备页面",
        allowed_package="com.android.settings",
        target_label="我的设备",
        success_activity_contains=("MyDevice", "DeviceInfo", "AboutDevice", "SubSettings"),
        success_text_contains=("我的设备",),
    ),
    "MORE_CONNECTIONS": AgentGoal(
        description="打开系统更多连接页面",
        allowed_package="com.android.settings",
        target_label="更多连接",
        success_activity_contains=(
            "ConnectionAndSharing",
            "WirelessSettings",
            "SubSettings",
        ),
        success_text_contains=("更多连接", "连接与共享"),
    ),
}


GOAL_ALIASES = (
    (("wlan", "wifi", "无线网络"), "WLAN"),
    (("蓝牙", "bluetooth"), "BLUETOOTH"),
    (("移动网络", "蜂窝网络", "mobiledata"), "MOBILE_NETWORK"),
    (("我的设备", "设备信息", "本机信息"), "MY_DEVICE"),
    (("更多连接", "连接与共享"), "MORE_CONNECTIONS"),
)


def parse_safe_goal(task: str) -> AgentGoal:
    """Resolve natural language only to locally approved, verifiable goals."""

    normalized = re.sub(r"[\s_-]+", "", task).lower()
    for aliases, goal_name in GOAL_ALIASES:
        if any(marker in normalized for marker in aliases):
            return SAFE_SETTINGS_GOALS[goal_name]
    raise AgentError(
        "当前自然语言 Agent 只开放 WLAN、蓝牙、移动网络、我的设备和更多连接。"
    )
