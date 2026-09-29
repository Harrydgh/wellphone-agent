from __future__ import annotations

import re
from dataclasses import replace

from .fusion import PerceivedElement


class PrivacyRedactor:
    """Remove common personal and order identifiers before model payloads."""

    PHONE = re.compile(r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)")
    EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
    LONG_ID = re.compile(r"(?<!\d)\d{8,}(?!\d)")
    ACCOUNT_ID = re.compile(r"\b[A-Za-z][A-Za-z0-9._-]*\d{6,}\b")
    WIFI_BAND_NAME = re.compile(r"^.+(?:[-_ ]?(?:2\.4|5|6)G)[>›]?$", re.IGNORECASE)
    ADDRESS_MARKERS = ("收货地址", "配送地址", "详细地址", "送至")
    RECIPIENT_MARKERS = ("收货人", "联系人")

    def redact(self, text: str) -> str:
        value = self.ACCOUNT_ID.sub("[账号标识已隐藏]", text)
        value = self.PHONE.sub("[手机号已隐藏]", value)
        value = self.EMAIL.sub("[邮箱已隐藏]", value)
        value = self.LONG_ID.sub("[编号已隐藏]", value)
        if self.WIFI_BAND_NAME.fullmatch(value.strip()):
            return "[网络名称已隐藏]"
        for marker in self.ADDRESS_MARKERS:
            if marker in value:
                return f"{marker}：[地址已隐藏]"
        for marker in self.RECIPIENT_MARKERS:
            if marker in value:
                return f"{marker}：[姓名已隐藏]"
        return value

    def redact_elements(
        self, elements: tuple[PerceivedElement, ...]
    ) -> tuple[PerceivedElement, ...]:
        return tuple(replace(item, text=self.redact(item.text)) for item in elements)

    @staticmethod
    def is_redacted(text: str) -> bool:
        return "已隐藏]" in text
