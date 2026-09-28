from __future__ import annotations

import hashlib
import subprocess
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import av
import numpy as np
from av.error import FFmpegError
from PIL import Image


class FrameCaptureError(RuntimeError):
    """A virtual-display frame could not be recorded or decoded."""


@dataclass(frozen=True)
class FrameSnapshot:
    path: Path
    captured_at: str
    width: int
    height: int
    sha256: str
    perceptual_hash: str
    display_id: int

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["path"] = str(self.path)
        return data


def image_perceptual_hash(image: Image.Image) -> str:
    """Return a 64-bit difference hash for fast visual comparisons."""

    grayscale = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = np.asarray(grayscale, dtype=np.int16)
    differences = pixels[:, 1:] > pixels[:, :-1]
    value = 0
    for bit in differences.flatten():
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def perceptual_distance(first: str, second: str) -> float:
    """Return normalized Hamming distance between two 64-bit hashes."""

    return (int(first, 16) ^ int(second, 16)).bit_count() / 64.0


class VirtualDisplayCapture:
    """Capture a clean frame from an existing scrcpy virtual display."""

    def __init__(
        self,
        scrcpy: Path,
        serial: str,
        display_id: int,
        output_directory: Path,
        *,
        clip_seconds: int = 1,
    ) -> None:
        if display_id <= 0:
            raise FrameCaptureError(
                f"Refusing to capture primary/invalid display: {display_id}"
            )
        if clip_seconds < 1:
            raise FrameCaptureError("clip_seconds must be at least 1 second")
        self.scrcpy = scrcpy
        self.serial = serial
        self.display_id = display_id
        self.output_directory = output_directory
        self.clip_seconds = clip_seconds

    def _record_command(self, clip_path: Path) -> list[str]:
        return [
            str(self.scrcpy),
            f"--serial={self.serial}",
            f"--display-id={self.display_id}",
            "--no-playback",
            "--no-control",
            "--no-window",
            "--no-audio",
            f"--record={clip_path}",
            f"--time-limit={self.clip_seconds}",
        ]

    def capture(self, label: str = "frame") -> FrameSnapshot:
        self.output_directory.mkdir(parents=True, exist_ok=True)
        captured = datetime.now(timezone.utc)
        timestamp = captured.strftime("%Y%m%d-%H%M%S-%f")
        safe_label = "".join(
            character if character.isalnum() or character in "-_" else "-"
            for character in label
        ).strip("-") or "frame"
        image_path = self.output_directory / f"{safe_label}-{timestamp}.png"
        clip_path = self.output_directory / f".capture-{uuid.uuid4().hex}.mp4"

        try:
            result = subprocess.run(
                self._record_command(clip_path),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.clip_seconds + 20,
                check=False,
            )
            if result.returncode != 0 or not clip_path.is_file():
                detail = "\n".join(
                    part.strip()
                    for part in (result.stdout, result.stderr)
                    if part and part.strip()
                )
                raise FrameCaptureError(f"scrcpy recording failed: {detail}")

            last_frame = None
            with av.open(str(clip_path)) as container:
                for frame in container.decode(video=0):
                    last_frame = frame
            if last_frame is None:
                raise FrameCaptureError("The virtual-display recording had no video frames.")

            image = last_frame.to_image().convert("RGB")
            image.save(image_path, format="PNG", optimize=True)
            image.save(self.output_directory / "latest.png", format="PNG", optimize=True)
            image_bytes = image_path.read_bytes()
            return FrameSnapshot(
                path=image_path.resolve(),
                captured_at=captured.isoformat(),
                width=image.width,
                height=image.height,
                sha256=hashlib.sha256(image_bytes).hexdigest(),
                perceptual_hash=image_perceptual_hash(image),
                display_id=self.display_id,
            )
        except subprocess.TimeoutExpired as exc:
            raise FrameCaptureError("Timed out recording the virtual display.") from exc
        except (FFmpegError, OSError) as exc:
            raise FrameCaptureError(f"Could not decode the recorded frame: {exc}") from exc
        finally:
            clip_path.unlink(missing_ok=True)
