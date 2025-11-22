from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any, Dict, List


def append_pds_diagnostics_csv(
    path: str | Path,
    *,
    mean_block_rmse: float,
    block_ranges: List[tuple[int, int]] | List[List[int]],
    cond_numbers: List[float],
    block_rmse: List[float],
    extra: Dict[str, Any] | None = None,
) -> None:
    """Append PDS diagnostics to a CSV for later analysis.

    Columns include timestamp, core diagnostics, and selected extra context
    (e.g., dataset, transfer size, window/overlap/ridge).
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "timestamp",
        "mean_block_rmse",
        "block_ranges",
        "cond_numbers",
        "block_rmse",
    ]
    # Include known extra keys in a stable order if present
    extra_keys: list[str] = []
    extras: Dict[str, Any] = extra or {}
    for k in ["dataset", "n_transfer", "window", "overlap", "ridge", "shift_bias", "shift_scale"]:
        if k in extras:
            extra_keys.append(k)
    header = header + extra_keys

    new_file = not p.exists()
    with p.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(header)
        row = [
            int(time.time()),
            float(mean_block_rmse),
            str(block_ranges),
            str([float(c) for c in cond_numbers]),
            str([float(r) for r in block_rmse]),
        ]
        for k in extra_keys:
            row.append(extras.get(k))
        writer.writerow(row)
