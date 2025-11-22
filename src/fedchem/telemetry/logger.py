from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import json
from pathlib import Path


@dataclass
class TelemetryRecord:
    data: Dict[str, Any]


class TelemetryLogger:
    """Telemetry logger with optional JSONL persistence."""

    def __init__(self, jsonl_path: Optional[str] = None) -> None:
        self.records: List[TelemetryRecord] = []
        self._path = Path(jsonl_path) if jsonl_path else None
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)

    def log_round(self, **kwargs: Any) -> None:
        rec = TelemetryRecord(dict(kwargs))
        self.records.append(rec)
        if self._path is not None:
            with self._path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec.data) + "\n")
