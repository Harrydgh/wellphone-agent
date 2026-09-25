from __future__ import annotations

import time
from pathlib import Path

from .actions import ActionController
from .adb import AdbClient, ensure_connected
from .display import VirtualDisplaySession
from .runlog import RunLogger
from .tools import find_adb, find_scrcpy


def run_browser_demo(
    *,
    project_root: Path,
    preferred_serial: str | None,
    package: str,
    url: str,
    hold_seconds: float,
) -> Path:
    adb = AdbClient(find_adb())
    device = ensure_connected(adb, preferred_serial)
    log = RunLogger(project_root / "logs")
    log.write("run_started", serial=device.serial, package=package, url=url)

    display = VirtualDisplaySession(
        find_scrcpy(),
        device.serial,
        start_app=package,
    )
    try:
        display_id = display.start()
        log.write("display_ready", display_id=display_id)
        controller = ActionController(adb, device.serial, display_id)

        time.sleep(2)
        controller.open_url(url, package)
        log.write("url_opened", display_id=display_id, url=url)
        time.sleep(4)

        controller.swipe(540, 1500, 540, 650, 500)
        log.write(
            "action",
            action="swipe",
            display_id=display_id,
            start=[540, 1500],
            end=[540, 650],
        )
        time.sleep(1)

        verified = controller.app_is_on_display(package)
        log.write("verification", app_on_virtual_display=verified)
        if not verified:
            raise RuntimeError(
                f"Browser {package} was not verified on display {display_id}."
            )

        if hold_seconds > 0:
            time.sleep(hold_seconds)
        log.write("run_completed", success=True)
        return log.path
    except Exception as exc:
        log.write("run_failed", error=type(exc).__name__, detail=str(exc))
        raise
    finally:
        display.stop()
        log.write("display_stopped")

