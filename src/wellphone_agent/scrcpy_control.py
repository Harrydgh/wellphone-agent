from __future__ import annotations

import queue
import random
import re
import socket
import struct
import subprocess
import threading
import time
from collections import deque
from pathlib import Path

from .adb import AdbClient, AdbError


class ScrcpyControlError(RuntimeError):
    """A scrcpy server/control-channel operation failed."""


class ScrcpyControlSession:
    """Own a scrcpy virtual display and its matching binary control channel."""

    DISPLAY_PATTERN = re.compile(r"New display: .*\(id=(\d+)\)")
    VERSION_PATTERN = re.compile(r"^scrcpy\s+([^\s]+)", re.MULTILINE)
    REMOTE_SERVER = "/data/local/tmp/wellphone-scrcpy-server.jar"

    ACTION_DOWN = 0
    ACTION_UP = 1
    ACTION_MOVE = 2
    ACTION_HOVER_MOVE = 7
    POINTER_ID_MOUSE = 0xFFFFFFFFFFFFFFFF

    def __init__(
        self,
        adb: AdbClient,
        scrcpy: Path,
        serial: str,
        *,
        size: str = "1080x1920",
        dpi: int = 420,
    ) -> None:
        match = re.fullmatch(r"(\d+)x(\d+)", size)
        if not match:
            raise ScrcpyControlError(f"Invalid virtual display size: {size}")
        self.adb = adb
        self.scrcpy = scrcpy
        self.serial = serial
        self.size = size
        self.width = int(match.group(1))
        self.height = int(match.group(2))
        self.dpi = dpi
        self.display_id: int | None = None
        self.process: subprocess.Popen[str] | None = None
        self._video_socket: socket.socket | None = None
        self._control_socket: socket.socket | None = None
        self._port: int | None = None
        self._scid: int | None = None
        self._lines: deque[str] = deque(maxlen=100)
        self._events: queue.Queue[str] = queue.Queue()
        self._threads: list[threading.Thread] = []
        self._send_lock = threading.Lock()

    @property
    def server_path(self) -> Path:
        path = self.scrcpy.with_name("scrcpy-server")
        if not path.is_file():
            raise ScrcpyControlError(f"Matching scrcpy-server was not found: {path}")
        return path

    def _scrcpy_version(self) -> str:
        result = subprocess.run(
            [str(self.scrcpy), "--version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            check=False,
        )
        match = self.VERSION_PATTERN.search(f"{result.stdout}\n{result.stderr}")
        if result.returncode != 0 or not match:
            raise ScrcpyControlError("Could not determine the installed scrcpy version.")
        return match.group(1)

    @staticmethod
    def _free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            return int(listener.getsockname()[1])

    def _server_command(self, version: str, scid: int) -> list[str]:
        return [
            str(self.adb.executable),
            "-s",
            self.serial,
            "shell",
            f"CLASSPATH={self.REMOTE_SERVER}",
            "app_process",
            "/",
            "com.genymobile.scrcpy.Server",
            version,
            f"scid={scid:08x}",
            "log_level=info",
            "video=true",
            "audio=false",
            "control=true",
            "video_codec=h264",
            "tunnel_forward=true",
            "send_device_meta=false",
            "send_dummy_byte=true",
            f"new_display={self.size}/{self.dpi}",
            "display_ime_policy=local",
            "keep_active=true",
        ]

    def require_interactive_unlocked_device(self) -> None:
        """Fail closed when MIUI would silently discard virtual-display input."""

        power = self.adb.shell(self.serial, "dumpsys", "power")
        policy = self.adb.shell(self.serial, "dumpsys", "window", "policy")
        interactive = bool(
            re.search(r"mWakefulness=(Awake|Interactive)", power, re.IGNORECASE)
        )
        # MIUI may leave showingAndNotOccluded=true after unlock. Its
        # KeyguardStateMonitor mIsShowing value reflects the current state.
        keyguard = re.search(r"mIsShowing=(true|false)", policy, re.IGNORECASE)
        locked = (
            keyguard.group(1).lower() == "true"
            if keyguard
            else bool(re.search(r"showingAndNotOccluded=true", policy))
        )
        if not interactive or locked:
            raise ScrcpyControlError(
                "手机主屏当前处于息屏或锁屏状态。请先点亮并解锁手机，再运行控制测试；"
                "MIUI 会丢弃锁屏期间发往虚拟屏的点击和按键。"
            )

    def _read_output(self) -> None:
        assert self.process is not None
        assert self.process.stdout is not None
        for raw_line in self.process.stdout:
            line = raw_line.rstrip()
            self._lines.append(line)
            self._events.put(line)

    @staticmethod
    def _drain(connection: socket.socket) -> None:
        try:
            while connection.recv(65536):
                pass
        except OSError:
            pass

    @staticmethod
    def _connect(port: int, deadline: float) -> socket.socket:
        last_error: OSError | None = None
        while time.monotonic() < deadline:
            connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            connection.settimeout(2)
            try:
                connection.connect(("127.0.0.1", port))
                connection.settimeout(None)
                return connection
            except OSError as exc:
                last_error = exc
                connection.close()
                time.sleep(0.1)
        raise ScrcpyControlError(f"Could not connect to scrcpy server: {last_error}")

    @classmethod
    def _connect_first(cls, port: int, deadline: float) -> socket.socket:
        """Connect after adb forward can actually reach the device listener."""

        last_handshake = b""
        while time.monotonic() < deadline:
            connection = cls._connect(port, deadline)
            connection.settimeout(min(1, max(0.1, deadline - time.monotonic())))
            try:
                last_handshake = connection.recv(1)
            except (OSError, TimeoutError):
                last_handshake = b""
            if last_handshake == b"\x00":
                connection.settimeout(None)
                return connection
            connection.close()
            time.sleep(0.1)
        raise ScrcpyControlError(
            f"Could not complete scrcpy forward handshake: {last_handshake!r}."
        )

    def start(self, timeout: float = 25.0) -> int:
        if self.process is not None:
            raise ScrcpyControlError("The scrcpy control session is already started.")

        self.require_interactive_unlocked_device()
        version = self._scrcpy_version()
        server = self.server_path
        self.adb.run("push", str(server), self.REMOTE_SERVER, serial=self.serial)
        self._port = self._free_port()
        self._scid = random.SystemRandom().randrange(1, 0x7FFFFFFF)
        self.adb.run(
            "forward",
            f"tcp:{self._port}",
            f"localabstract:scrcpy_{self._scid:08x}",
            serial=self.serial,
        )

        creation_flags = 0
        if hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):
            creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP
        self.process = subprocess.Popen(
            self._server_command(version, self._scid),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=creation_flags,
        )
        output_thread = threading.Thread(target=self._read_output, daemon=True)
        output_thread.start()
        self._threads.append(output_thread)

        try:
            deadline = time.monotonic() + timeout
            self._video_socket = self._connect_first(self._port, deadline)
            # With adb forward, a local TCP connect may succeed before the device
            # server is listening. The dummy byte confirms the first socket was
            # really accepted before we open the control socket.
            self._control_socket = self._connect(self._port, deadline)
            for connection in (self._video_socket, self._control_socket):
                thread = threading.Thread(
                    target=self._drain,
                    args=(connection,),
                    daemon=True,
                )
                thread.start()
                self._threads.append(thread)

            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise ScrcpyControlError(self._failure_message("server exited early"))
                try:
                    line = self._events.get(timeout=0.25)
                except queue.Empty:
                    continue
                match = self.DISPLAY_PATTERN.search(line)
                if match:
                    self.display_id = int(match.group(1))
                    return self.display_id
            raise ScrcpyControlError(self._failure_message("timed out creating display"))
        except Exception:
            self.stop()
            raise

    def _failure_message(self, reason: str) -> str:
        return f"{reason}. Recent server output:\n" + "\n".join(self._lines)

    def _send(self, payload: bytes) -> None:
        connection = self._control_socket
        if connection is None or self.display_id is None:
            raise ScrcpyControlError("The scrcpy control session is not ready.")
        try:
            with self._send_lock:
                connection.sendall(payload)
        except OSError as exc:
            raise ScrcpyControlError(f"Could not send scrcpy control message: {exc}") from exc

    def _touch_message(
        self,
        action: int,
        x: int,
        y: int,
        *,
        pressure: int,
        pointer_id: int | None = None,
        action_button: int = 0,
        buttons: int = 0,
    ) -> bytes:
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise ScrcpyControlError(
                f"Touch point ({x}, {y}) is outside {self.width}x{self.height}."
            )
        return struct.pack(
            ">BBQiiHHHII",
            2,
            action,
            self.POINTER_ID_MOUSE if pointer_id is None else pointer_id,
            x,
            y,
            self.width,
            self.height,
            pressure,
            action_button,
            buttons,
        )

    def tap(self, x: int, y: int) -> None:
        self._send(self._touch_message(self.ACTION_HOVER_MOVE, x, y, pressure=0))
        self._send(
            self._touch_message(
                self.ACTION_DOWN,
                x,
                y,
                pressure=0xFFFF,
                action_button=1,
                buttons=1,
            )
        )
        time.sleep(0.05)
        self._send(
            self._touch_message(
                self.ACTION_UP,
                x,
                y,
                pressure=0,
                action_button=1,
                buttons=0,
            )
        )

    def swipe(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        duration_ms: int = 400,
    ) -> None:
        if duration_ms <= 0:
            raise ScrcpyControlError("Swipe duration must be positive.")
        steps = max(2, min(60, round(duration_ms / 16)))
        self._send(self._touch_message(self.ACTION_HOVER_MOVE, x1, y1, pressure=0))
        self._send(
            self._touch_message(
                self.ACTION_DOWN,
                x1,
                y1,
                pressure=0xFFFF,
                action_button=1,
                buttons=1,
            )
        )
        interval = duration_ms / steps / 1000
        for step in range(1, steps):
            time.sleep(interval)
            x = round(x1 + (x2 - x1) * step / steps)
            y = round(y1 + (y2 - y1) * step / steps)
            self._send(
                self._touch_message(
                    self.ACTION_MOVE,
                    x,
                    y,
                    pressure=0xFFFF,
                    action_button=1,
                    buttons=1,
                )
            )
        time.sleep(interval)
        self._send(
            self._touch_message(
                self.ACTION_UP,
                x2,
                y2,
                pressure=0,
                action_button=1,
                buttons=0,
            )
        )

    def keyevent(self, keycode: int) -> None:
        if keycode < 0:
            raise ScrcpyControlError(f"Invalid Android keycode: {keycode}")
        if keycode == 4:
            for action in (self.ACTION_DOWN, self.ACTION_UP):
                self._send(bytes([4, action]))
            return
        for action in (self.ACTION_DOWN, self.ACTION_UP):
            self._send(struct.pack(">BBIII", 0, action, keycode, 0, 0))

    def type_text(self, text: str) -> None:
        encoded = text.encode("utf-8")
        if not encoded or len(encoded) > 300:
            raise ScrcpyControlError("Text must contain 1 to 300 UTF-8 bytes.")
        self._send(bytes([1]) + struct.pack(">I", len(encoded)) + encoded)

    def start_app(self, package: str) -> None:
        encoded = package.encode("utf-8")
        if not encoded or len(encoded) > 255:
            raise ScrcpyControlError("App package must contain 1 to 255 UTF-8 bytes.")
        self._send(bytes([16, len(encoded)]) + encoded)

    def stop(self, timeout: float = 5.0) -> None:
        for connection in (self._control_socket, self._video_socket):
            if connection is not None:
                try:
                    connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                connection.close()
        self._control_socket = None
        self._video_socket = None

        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=timeout)

        if self._port is not None:
            self.adb.run(
                "forward",
                "--remove",
                f"tcp:{self._port}",
                serial=self.serial,
                check=False,
            )
        self.adb.shell(self.serial, "rm", "-f", self.REMOTE_SERVER, check=False)

    def recent_output(self) -> tuple[str, ...]:
        return tuple(self._lines)

    def __enter__(self) -> "ScrcpyControlSession":
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.stop()
