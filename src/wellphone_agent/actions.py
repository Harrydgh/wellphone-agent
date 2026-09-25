from __future__ import annotations

import re
from urllib.parse import urlparse

from .adb import AdbClient, AdbError


class UnsafeActionError(ValueError):
    """An action could affect the primary display or contains unsafe input."""


class ActionController:
    def __init__(self, adb: AdbClient, serial: str, display_id: int) -> None:
        if display_id <= 0:
            raise UnsafeActionError(
                f"Refusing to control primary/invalid display: {display_id}"
            )
        self.adb = adb
        self.serial = serial
        self.display_id = display_id

    def _input(self, *args: str) -> None:
        self.adb.shell(
            self.serial,
            "input",
            "-d",
            str(self.display_id),
            *args,
        )

    def tap(self, x: int, y: int) -> None:
        if x < 0 or y < 0:
            raise UnsafeActionError("Tap coordinates must be non-negative.")
        self._input("tap", str(x), str(y))

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
        self._input(
            "swipe",
            str(x1),
            str(y1),
            str(x2),
            str(y2),
            str(duration_ms),
        )

    def keyevent(self, keycode: str) -> None:
        if not re.fullmatch(r"KEYCODE_[A-Z0-9_]+|\d+", keycode):
            raise UnsafeActionError(f"Invalid keycode: {keycode}")
        self._input("keyevent", keycode)

    def type_ascii(self, text: str) -> None:
        if not text or not re.fullmatch(r"[A-Za-z0-9 ._\-]+", text):
            raise UnsafeActionError(
                "First milestone text input supports only ASCII letters, numbers, "
                "spaces, dot, underscore and hyphen."
            )
        self._input("text", text.replace(" ", "%s"))

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

