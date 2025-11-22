from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

from fedchem.utils.manifest_utils import compute_combo_id


@dataclass(frozen=True)
class ResultsTree:
    """Canonical layout for experiment outputs."""

    base: Path
    raw_runs: Path
    aggregated: Path
    tables: Path
    figures: Path


def ensure_results_tree(base_dir: str | Path = "results") -> ResultsTree:
    """Create (if needed) and return the default results directory tree.

    Parameters
    ----------
    base_dir:
        Root directory for results. Defaults to ``results`` in the project root.
    """

    base = Path(base_dir)
    raw = base / "raw_runs"
    aggregated = base / "aggregated"
    tables = base / "tables"
    figures = base / "figures"

    for path in (base, raw, aggregated, tables, figures):
        path.mkdir(parents=True, exist_ok=True)

    return ResultsTree(base=base, raw_runs=raw, aggregated=aggregated, tables=tables, figures=figures)


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_metrics(summary: Mapping[str, Any]) -> Dict[str, Optional[float]]:
    """Return a flat metrics dictionary from a manifest summary block."""

    final_block = summary.get("final") if isinstance(summary.get("final"), Mapping) else {}
    stats_block = summary.get("statistics") if isinstance(summary.get("statistics"), Mapping) else {}

    rmsep = _safe_float(final_block.get("rmsep")) or _safe_float(stats_block.get("rmsep_final"))
    r2 = _safe_float(final_block.get("r2")) or _safe_float(stats_block.get("r2_final"))
    coverage = _safe_float(final_block.get("coverage")) or _safe_float(stats_block.get("coverage_final"))
    avg_set = _safe_float(stats_block.get("avg_interval_width"))

    bytes_sent = stats_block.get("total_bytes_sent")
    bytes_recv = stats_block.get("total_bytes_recv")
    total_bytes = stats_block.get("total_bytes")
    if total_bytes is None and isinstance(bytes_sent, (int, float)) and isinstance(bytes_recv, (int, float)):
        total_bytes = float(bytes_sent) + float(bytes_recv)

    n_rounds = stats_block.get("n_rounds")
    round_time = None
    duration = _safe_float(final_block.get("duration_sec"))
    if duration is not None and isinstance(n_rounds, (int, float)) and n_rounds:
        round_time = duration / float(n_rounds)

    return {
        "RMSEP": rmsep,
        "R2": r2,
        "Coverage": coverage,
        "Avg_Pred_Set_Size": avg_set,
        "Bytes_Sent": _safe_float(bytes_sent),
        "Bytes_Received": _safe_float(bytes_recv),
        "Total_Bytes": _safe_float(total_bytes),
        "Rounds": _safe_float(n_rounds),
        "Round_Time": round_time,
    }


def manifest_to_raw_records(
    manifest: Mapping[str, Any],
    design_point: Optional[Mapping[str, Any]] = None,
    *,
    run_label: str,
    script_name: str,
    manifest_path: Optional[Path] = None,
    extra_metadata: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Convert a manifest dictionary to one or more raw-run records.

    Parameters
    ----------
    manifest:
        Parsed manifest JSON blob from an objective generator.
    design_point:
        Optional dictionary describing the experimental design factors used for the run.
    run_label:
        Logical label for the run (e.g., ``objective_1``).
    script_name:
        Name of the script that produced the manifest.
    manifest_path:
        Path to the manifest on disk used for traceability metadata.
    extra_metadata:
        Optional dictionary of extra metadata merged into each record.
    """

    if manifest is None:
        return []
    summary = manifest.get("log_summary") or manifest.get("summary")
    if not isinstance(summary, Mapping):
        return []
    runtime = manifest.get("runtime") if isinstance(manifest.get("runtime"), Mapping) else {}
    config = manifest.get("config") if isinstance(manifest.get("config"), Mapping) else {}
    combo_id = manifest.get("combo_id") or compute_combo_id(config)
    factors = dict(design_point or {})

    timestamp = datetime.now(tz=timezone.utc).isoformat()
    records: List[Dict[str, Any]] = []
    for method, stats in summary.items():
        if not isinstance(stats, Mapping):
            continue
        metrics = _extract_metrics(stats)
        record: Dict[str, Any] = {
            "timestamp": timestamp,
            "run_label": run_label,
            "script": script_name,
            "combo_id": combo_id,
            "method": method,
            "design_factors": factors,
            "seed": config.get("seed"),
            "site_code": factors.get("Site"),
            "instrument_code": factors.get("Site"),
            "runtime": {
                "wall_time_total_sec": runtime.get("wall_time_total_sec"),
                "total_bytes": runtime.get("total_bytes"),
                "total_bytes_mb": runtime.get("total_bytes_mb"),
            },
            "metrics": metrics,
        }
        if manifest_path is not None:
            record["manifest_path"] = str(manifest_path)
        if extra_metadata:
            record.setdefault("extra_metadata", {}).update(dict(extra_metadata))
        records.append(record)
    return records


def write_raw_records(records: Iterable[Mapping[str, Any]], tree: ResultsTree, *, combo_id: str, run_label: str) -> Path:
    """Append raw run records to ``results/raw_runs`` and return the JSONL path."""

    output_dir = tree.raw_runs / run_label
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / f"{combo_id}.jsonl"
    with dest.open("a", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")
    return dest
