from __future__ import annotations

import os
import shutil
from pathlib import Path


class ToolNotFoundError(FileNotFoundError):
    """Raised when a required Android tool cannot be located."""


def _winget_candidates(executable: str) -> list[Path]:
    local_app_data = os.environ.get("LOCALAPPDATA")
    if not local_app_data:
        return []

    packages = Path(local_app_data) / "Microsoft" / "WinGet" / "Packages"
    if not packages.exists():
        return []

    pattern = f"Genymobile.scrcpy_*/*/{executable}"
    return sorted(packages.glob(pattern), reverse=True)


def find_tool(name: str, env_name: str) -> Path:
    override = os.environ.get(env_name)
    if override:
        path = Path(override).expanduser().resolve()
        if path.is_file():
            return path
        raise ToolNotFoundError(f"{env_name} points to a missing file: {path}")

    on_path = shutil.which(name)
    if on_path:
        return Path(on_path).resolve()

    candidates = _winget_candidates(name)
    if candidates:
        return candidates[0].resolve()

    raise ToolNotFoundError(
        f"Could not find {name}. Install Genymobile scrcpy or set {env_name}."
    )


def find_adb() -> Path:
    return find_tool("adb.exe" if os.name == "nt" else "adb", "WELLPHONE_ADB")


def find_scrcpy() -> Path:
    return find_tool(
        "scrcpy.exe" if os.name == "nt" else "scrcpy", "WELLPHONE_SCRCPY"
    )

