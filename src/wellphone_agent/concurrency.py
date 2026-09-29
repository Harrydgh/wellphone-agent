from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from .agent.core import AgentError
from .runlog import RunLogger


class DisplayInspector(Protocol):
    def current_app_activity(self, display_id: int) -> tuple[str | None, str | None]: ...


@dataclass(frozen=True)
class MainDisplaySample:
    captured_at: str
    package: str | None
    activity: str | None
    error: str | None = None


@dataclass(frozen=True)
class MainDisplaySnapshot:
    samples: tuple[MainDisplaySample, ...]
    apps_seen: tuple[str, ...]
    app_transitions: int
    protected_app_seen: int
    sampling_errors: int
    violation_reason: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "samples": [asdict(sample) for sample in self.samples],
            "apps_seen": list(self.apps_seen),
            "app_transitions": self.app_transitions,
            "protected_app_seen": self.protected_app_seen,
            "sampling_errors": self.sampling_errors,
            "violation_reason": self.violation_reason,
        }


class MainDisplayMonitor:
    """Observe display 0 and fail closed if the task App migrates onto it."""

    def __init__(
        self,
        inspector: DisplayInspector,
        protected_package: str,
        *,
        logger: RunLogger | None = None,
        interval_seconds: float = 1.0,
        max_consecutive_errors: int = 3,
    ) -> None:
        if interval_seconds <= 0 or max_consecutive_errors <= 0:
            raise ValueError("Main display monitor limits are invalid.")
        self.inspector = inspector
        self.protected_package = protected_package
        self.logger = logger
        self.interval_seconds = interval_seconds
        self.max_consecutive_errors = max_consecutive_errors
        self._samples: list[MainDisplaySample] = []
        self._violation_reason: str | None = None
        self._consecutive_errors = 0
        self._state_lock = threading.Lock()
        self._sample_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def _log(self, event: str, **data: object) -> None:
        if self.logger is not None:
            self.logger.write(event, monitor="main_display", **data)

    def sample_once(self) -> MainDisplaySample:
        with self._sample_lock:
            try:
                package, activity = self.inspector.current_app_activity(0)
                sample = MainDisplaySample(
                    datetime.now(timezone.utc).isoformat(), package, activity
                )
            except Exception as exc:
                package = None
                sample = MainDisplaySample(
                    datetime.now(timezone.utc).isoformat(),
                    None,
                    None,
                    type(exc).__name__,
                )

            with self._state_lock:
                self._samples.append(sample)
                if sample.error:
                    self._consecutive_errors += 1
                    if self._consecutive_errors >= self.max_consecutive_errors:
                        self._violation_reason = "主屏监测连续失败，为避免误操作已安全停止。"
                else:
                    self._consecutive_errors = 0
                if package == self.protected_package:
                    self._violation_reason = (
                        "检测到任务 App 出现在手机主屏，"
                        "可能发生同 App 任务迁移，已安全停止。"
                    )
            self._log(
                "main_display_sample",
                package=sample.package,
                activity=sample.activity,
                error=sample.error,
            )
            return sample

    def start(self) -> None:
        if self._thread is not None:
            return
        self.sample_once()
        self._thread = threading.Thread(
            target=self._run,
            name="wellphone-main-display-monitor",
            daemon=True,
        )
        self._thread.start()
        self._log("main_display_monitor_started")

    def _run(self) -> None:
        while not self._stop_event.wait(self.interval_seconds):
            self.sample_once()

    def assert_safe(self) -> None:
        self.sample_once()
        with self._state_lock:
            reason = self._violation_reason
        if reason:
            self._log("main_display_violation", reason=reason)
            raise AgentError(reason)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_seconds * 2))
            self._thread = None
        self._log("main_display_monitor_stopped")

    def snapshot(self) -> MainDisplaySnapshot:
        with self._state_lock:
            samples = tuple(self._samples)
            reason = self._violation_reason
        apps: list[str] = []
        transitions = 0
        previous: str | None = None
        protected_seen = 0
        for sample in samples:
            if sample.package == self.protected_package:
                protected_seen += 1
            if sample.package and sample.package not in apps:
                apps.append(sample.package)
            if sample.package and previous and sample.package != previous:
                transitions += 1
            if sample.package:
                previous = sample.package
        return MainDisplaySnapshot(
            samples=samples,
            apps_seen=tuple(apps),
            app_transitions=transitions,
            protected_app_seen=protected_seen,
            sampling_errors=sum(1 for sample in samples if sample.error),
            violation_reason=reason,
        )


def write_acceptance_report(directory: Path, payload: dict[str, object]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    path = directory / f"concurrency-acceptance-{stamp}.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path
