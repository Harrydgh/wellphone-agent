from __future__ import annotations

import re
from typing import Protocol
from urllib.parse import urlparse

from .adb import AdbClient, AdbError


class UnsafeActionError(ValueError):
    """An action could affect the primary display or contains unsafe input."""


class VirtualDisplayInput(Protocol):
    def tap(self, x: int, y: int) -> None: ...

    def swipe(
        self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 400
    ) -> None: ...

    def keyevent(self, keycode: int) -> None: ...

    def type_text(self, text: str) -> None: ...

    def paste_text(self, text: str) -> None: ...


ANDROID_KEYCODES = {
    "KEYCODE_HOME": 3,
    "KEYCODE_BACK": 4,
    "KEYCODE_ENTER": 66,
    "KEYCODE_DEL": 67,
    "KEYCODE_TAB": 61,
    "KEYCODE_ESCAPE": 111,
    "KEYCODE_MOVE_END": 123,
}


class ActionController:
    def __init__(
        self,
        adb: AdbClient,
        serial: str,
        display_id: int,
        input_backend: VirtualDisplayInput | None = None,
    ) -> None:
        if display_id <= 0:
            raise UnsafeActionError(
                f"Refusing to control primary/invalid display: {display_id}"
            )
        self.adb = adb
        self.serial = serial
        self.display_id = display_id
        self.input_backend = input_backend

    def _require_input(self) -> VirtualDisplayInput:
        if self.input_backend is None:
            raise UnsafeActionError(
                "Virtual-display input requires an active scrcpy control session."
            )
        return self.input_backend

    @staticmethod
    def _keycode_value(keycode: str) -> int:
        if keycode.isdigit():
            return int(keycode)
        if not re.fullmatch(r"KEYCODE_[A-Z0-9_]+", keycode):
            raise UnsafeActionError(f"Invalid keycode: {keycode}")
        if keycode in ANDROID_KEYCODES:
            return ANDROID_KEYCODES[keycode]
        if len(keycode) == len("KEYCODE_A") and keycode.startswith("KEYCODE_"):
            letter = keycode[-1]
            if "A" <= letter <= "Z":
                return 29 + ord(letter) - ord("A")
        raise UnsafeActionError(
            f"Keycode is not in the supported safe subset: {keycode}"
        )

    def tap(self, x: int, y: int) -> None:
        if x < 0 or y < 0:
            raise UnsafeActionError("Tap coordinates must be non-negative.")
        self._require_input().tap(x, y)

    def swipe(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        duration_ms: int = 400,
    ) -> None:
        if min(x1, y1, x2, y2) < 0 or duration_ms <= 0:
            raise UnsafeActionError("Swipe coordinates and duration are invalid.")
        self._require_input().swipe(x1, y1, x2, y2, duration_ms)

    def keyevent(self, keycode: str) -> None:
        self._require_input().keyevent(self._keycode_value(keycode))

    def keycombination(self, *keycodes: str) -> None:
        raise UnsafeActionError(
            "Key combinations are not enabled until modifier-state handling is added."
        )

    def type_ascii(self, text: str) -> None:
        if not text or not re.fullmatch(r"[A-Za-z0-9 ._\-]+", text):
            raise UnsafeActionError(
                "First milestone text input supports only ASCII letters, numbers, "
                "spaces, dot, underscore and hyphen."
            )
        self._require_input().type_text(text)

    def type_text(self, text: str) -> None:
        """Type a user-supplied search phrase through the bound virtual display."""
        normalized = text.strip()
        if (
            not normalized
            or len(normalized.encode("utf-8")) > 120
            or any(ord(character) < 32 for character in normalized)
        ):
            raise UnsafeActionError(
                "Search text must be 1 to 120 UTF-8 bytes without control characters."
            )
        self._require_input().type_text(normalized)

    def replace_text(self, text: str, *, max_existing_characters: int = 80) -> None:
        """Replace focused field content without using shell or clipboard input."""
        if max_existing_characters <= 0 or max_existing_characters > 120:
            raise UnsafeActionError("Text replacement delete limit is invalid.")
        normalized = text.strip()
        if (
            not normalized
            or len(normalized.encode("utf-8")) > 120
            or any(ord(character) < 32 for character in normalized)
        ):
            raise UnsafeActionError(
                "Search text must be 1 to 120 UTF-8 bytes without control characters."
            )
        backend = self._require_input()
        backend.keyevent(ANDROID_KEYCODES["KEYCODE_MOVE_END"])
        for _ in range(max_existing_characters):
            backend.keyevent(ANDROID_KEYCODES["KEYCODE_DEL"])
        backend.paste_text(normalized)

    def open_url(self, url: str, package: str) -> None:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise UnsafeActionError("Only absolute http/https URLs are allowed.")
        if not re.fullmatch(r"[A-Za-z0-9_.]+", package):
            raise UnsafeActionError(f"Invalid Android package: {package}")
        result = self.adb.run(
            "shell",
            "am",
            "start",
            "--display",
            str(self.display_id),
            "-a",
            "android.intent.action.VIEW",
            "-d",
            url,
            package,
            serial=self.serial,
            check=False,
        )
        combined = f"{result.stdout}\n{result.stderr}"
        if result.returncode != 0 or "Error:" in combined:
            raise AdbError(f"Could not open URL on display {self.display_id}: {combined}")

    def app_is_on_display(self, package: str) -> bool:
        output = self.adb.shell(self.serial, "dumpsys", "activity", "activities")
        marker = f"Display: mDisplayId={self.display_id}"
        start = output.find(marker)
        if start < 0:
            return False
        next_display = output.find("Display: mDisplayId=", start + len(marker))
        section = output[start:] if next_display < 0 else output[start:next_display]
        return package in section
