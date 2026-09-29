from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


class RunLogger:
    def __init__(
        self,
        directory: Path,
        *,
        listener: Callable[[dict[str, object]], None] | None = None,
    ) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.path = directory / f"run-{stamp}.jsonl"
        self._lock = threading.Lock()
        self.listener = listener

    def write(self, event: str, **data: Any) -> None:
        record = {
            "time": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **data,
        }
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        if self.listener is not None:
            try:
                self.listener(record)
            except Exception:
                # Console rendering must never weaken or interrupt automation.
                pass
