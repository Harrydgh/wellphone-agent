from __future__ import annotations

import queue
import re
import subprocess
import threading
import time
from collections import deque
from pathlib import Path


class VirtualDisplayError(RuntimeError):
    """Virtual display creation or lifecycle operation failed."""


class VirtualDisplaySession:
    """Own one scrcpy virtual display process and its logical display id."""

    DISPLAY_PATTERN = re.compile(r"New display: .*\(id=(\d+)\)")

    def __init__(
        self,
        scrcpy: Path,
        serial: str,
        *,
        size: str = "1080x1920",
        dpi: int = 420,
        start_app: str = "com.android.browser",
        title: str = "Wellphone Agent Virtual Display",
    ) -> None:
        self.scrcpy = scrcpy
        self.serial = serial
        self.size = size
        self.dpi = dpi
        self.start_app = start_app
        self.title = title
        self.process: subprocess.Popen[str] | None = None
        self.display_id: int | None = None
        self._lines: deque[str] = deque(maxlen=100)
        self._events: queue.Queue[str] = queue.Queue()
        self._reader: threading.Thread | None = None

    @property
    def command(self) -> list[str]:
        return [
            str(self.scrcpy),
            f"--serial={self.serial}",
            f"--new-display={self.size}/{self.dpi}",
            f"--start-app={self.start_app}",
            "--display-ime-policy=local",
            "--keep-active",
            f"--window-title={self.title}",
        ]

    def _read_output(self) -> None:
        assert self.process is not None
        assert self.process.stdout is not None
        for raw_line in self.process.stdout:
            line = raw_line.rstrip()
            self._lines.append(line)
            self._events.put(line)

    def start(self, timeout: float = 20.0) -> int:
        if self.process is not None:
            raise VirtualDisplayError("Virtual display session has already been started.")

        creation_flags = 0
        if hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):
            creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP

        self.process = subprocess.Popen(
            self.command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=creation_flags,
        )
        self._reader = threading.Thread(target=self._read_output, daemon=True)
        self._reader.start()

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise VirtualDisplayError(self._failure_message("scrcpy exited early"))
            try:
                line = self._events.get(timeout=0.25)
            except queue.Empty:
                continue
            match = self.DISPLAY_PATTERN.search(line)
            if match:
                self.display_id = int(match.group(1))
                return self.display_id

        self.stop()
        raise VirtualDisplayError(self._failure_message("timed out creating display"))

    def _failure_message(self, reason: str) -> str:
        output = "\n".join(self._lines)
        return f"{reason}. Recent scrcpy output:\n{output}".rstrip()

    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def stop(self, timeout: float = 5.0) -> None:
        process = self.process
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=timeout)

    def recent_output(self) -> tuple[str, ...]:
        return tuple(self._lines)

    def __enter__(self) -> "VirtualDisplaySession":
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.stop()

