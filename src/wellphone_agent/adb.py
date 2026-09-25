from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


class AdbError(RuntimeError):
    """ADB command failed or returned unusable output."""


@dataclass(frozen=True)
class Device:
    serial: str
    state: str
    model: str | None = None
    product: str | None = None
    transport_id: str | None = None


@dataclass(frozen=True)
class DeviceInfo:
    serial: str
    model: str
    android_version: str
    sdk: str
    miui_version: str | None
    browser_packages: tuple[str, ...]


class AdbClient:
    def __init__(self, executable: Path, timeout: float = 15.0) -> None:
        self.executable = executable
        self.timeout = timeout

    def run(
        self,
        *args: str,
        serial: str | None = None,
        check: bool = True,
        timeout: float | None = None,
    ) -> subprocess.CompletedProcess[str]:
        command = [str(self.executable)]
        if serial:
            command.extend(["-s", serial])
        command.extend(args)
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout or self.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise AdbError(f"ADB command timed out: {' '.join(command)}") from exc

        if check and result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise AdbError(f"ADB command failed ({result.returncode}): {detail}")
        return result

    def start_server(self) -> None:
        self.run("start-server")

    def devices(self) -> list[Device]:
        output = self.run("devices", "-l").stdout
        devices: list[Device] = []
        for line in output.splitlines()[1:]:
            line = line.strip()
            if not line:
                continue
            columns = line.split()
            if len(columns) < 2:
                continue
            attributes = dict(
                item.split(":", 1) for item in columns[2:] if ":" in item
            )
            devices.append(
                Device(
                    serial=columns[0],
                    state=columns[1],
                    model=attributes.get("model"),
                    product=attributes.get("product"),
                    transport_id=attributes.get("transport_id"),
                )
            )
        return devices

    def discover_wireless_endpoints(self) -> list[str]:
        output = self.run("mdns", "services", check=False).stdout
        endpoints: list[str] = []
        for line in output.splitlines():
            if "_adb-tls-connect._tcp" not in line:
                continue
            match = re.search(r"((?:\d{1,3}\.){3}\d{1,3}:\d+)$", line.strip())
            if match:
                endpoints.append(match.group(1))
        return list(dict.fromkeys(endpoints))

    def connect(self, endpoint: str) -> bool:
        result = self.run("connect", endpoint, check=False)
        output = f"{result.stdout}\n{result.stderr}".lower()
        return result.returncode == 0 and (
            "connected to" in output or "already connected" in output
        )

    def choose_device(self, preferred_serial: str | None = None) -> Device:
        online = [device for device in self.devices() if device.state == "device"]
        if preferred_serial:
            for device in online:
                if device.serial == preferred_serial:
                    return device
            raise AdbError(f"Preferred device is not online: {preferred_serial}")

        direct = [device for device in online if not device.serial.endswith("._tcp")]
        candidates = direct or online
        if not candidates:
            raise AdbError("No authorized Android device is online.")
        return candidates[0]

    def shell(self, serial: str, *args: str, check: bool = True) -> str:
        return self.run("shell", *args, serial=serial, check=check).stdout.strip()

    def device_info(self, serial: str) -> DeviceInfo:
        def prop(name: str) -> str:
            return self.shell(serial, "getprop", name)

        packages = self.shell(serial, "pm", "list", "packages")
        browser_markers = ("browser", "chrome", "firefox", "edge", "quark", "ucbrowser")
        browsers = []
        for line in packages.splitlines():
            package = line[len("package:") :] if line.startswith("package:") else line
            package = package.strip()
            if any(marker in package.lower() for marker in browser_markers):
                browsers.append(package)

        return DeviceInfo(
            serial=serial,
            model=prop("ro.product.model") or "unknown",
            android_version=prop("ro.build.version.release") or "unknown",
            sdk=prop("ro.build.version.sdk") or "unknown",
            miui_version=prop("ro.miui.ui.version.name") or None,
            browser_packages=tuple(sorted(browsers)),
        )


def ensure_connected(client: AdbClient, preferred_serial: str | None = None) -> Device:
    client.start_server()
    try:
        return client.choose_device(preferred_serial)
    except AdbError:
        if preferred_serial and re.fullmatch(r"[\d.]+:\d+", preferred_serial):
            client.connect(preferred_serial)
        else:
            for endpoint in client.discover_wireless_endpoints():
                client.connect(endpoint)
        return client.choose_device(preferred_serial)
