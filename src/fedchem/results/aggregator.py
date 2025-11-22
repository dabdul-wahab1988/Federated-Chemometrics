from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional

import pandas as pd

from .manager import ResultsTree, ensure_results_tree

_METRIC_FIELDS = {
    "rmsep": "metrics.RMSEP",
    "r2": "metrics.R2",
    "coverage": "metrics.Coverage",
    "avg_pred_set_size": "metrics.Avg_Pred_Set_Size",
    "bytes": "metrics.Total_Bytes",
    "round_time": "metrics.Round_Time",
}

_AUX_FIELDS = {
    "dp_target_eps": "extra_metadata.dp_target_eps",
    "dp_delta": "extra_metadata.dp_delta",
    "clip_norm": "extra_metadata.clip_norm",
}


def _jsonl_paths(tree: ResultsTree) -> Iterable[Path]:
    for jsonl in sorted(tree.raw_runs.rglob("*.jsonl")):
        if jsonl.is_file():
            yield jsonl


def load_raw_records(tree_or_root: str | Path | ResultsTree) -> pd.DataFrame:
    """Load flattened raw-run records from the ``results/raw_runs`` tree."""

    tree = tree_or_root if isinstance(tree_or_root, ResultsTree) else ensure_results_tree(tree_or_root)
    rows: list[dict[str, Any]] = []
    for path in _jsonl_paths(tree):
        try:
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            continue
    if not rows:
        return pd.DataFrame()
    return pd.json_normalize(rows)


def _rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    rename_map = {v: k for k, v in _METRIC_FIELDS.items() if v in df.columns}
    rename_map.update({v: k for k, v in _AUX_FIELDS.items() if v in df.columns})
    rename_map["design_factors.Site"] = "site_factor"
    rename_map["design_factors.Spectral_Drift"] = "spectral_drift"
    rename_map["design_factors.Drift_Type"] = "drift_type"
    rename_map["design_factors.DP_Target_Eps"] = "design_dp_target_eps"
    rename_map["design_factors.Clip_Norm"] = "design_clip_norm"
    rename_map["design_factors.Participation_Schedule"] = "participation_schedule"
    rename_map["design_factors.Compression_Schedule"] = "compression_schedule"
    rename_map["design_factors.Rounds"] = "design_rounds"
    rename_map["design_factors.Local_Epochs"] = "design_local_epochs"
    rename_map["design_factors.Local_Batch_Size"] = "design_local_batch_size"
    rename_map["design_factors.Learning_Rate"] = "design_learning_rate"
    rename_map["design_factors.Conformal_Targets"] = "design_conformal_targets"
    rename_map["design_factors.Conformal_Calibration"] = "design_conformal_calibration"
    rename_map["design_factors.Conformal_Method"] = "design_conformal_method"
    rename_map["design_factors.Holdout_Strategy"] = "design_holdout_strategy"
    rename_map["design_factors.CT_Federated_Variant"] = "design_ct_variant"
    rename_map["design_factors.Baseline"] = "design_baseline"
    rename_map["design_factors.Federated_Method"] = "design_federated_method"
    rename_map["design_factors.Drift_Type"] = "drift_type"
    return df.rename(columns=rename_map)


def build_methods_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with the columns expected by downstream table builders."""

    if df.empty:
        return df
    df = _rename_columns(df)
    base_cols = [
        "run_label",
        "combo_id",
        "method",
        "site_code",
        "instrument_code",
        "site_factor",
        "spectral_drift",
        "drift_type",
        "design_dp_target_eps",
        "design_clip_norm",
        "dp_target_eps",
        "dp_delta",
        "clip_norm",
    ]
    metric_cols = list(_METRIC_FIELDS.keys())
    keep_cols = [col for col in base_cols + metric_cols if col in df.columns]
    summary = df[keep_cols].copy()
    summary.rename(columns={"site_code": "site_id"}, inplace=True)
    return summary


def export_methods_summary(summary_df: pd.DataFrame, tree: ResultsTree) -> dict[str, Path]:
    """Write per-run ``methods_summary.csv`` files and return their directories."""

    outputs: dict[str, Path] = {}
    if summary_df.empty:
        return outputs
    for run_label, group in summary_df.groupby("run_label"):
        out_dir = tree.aggregated / (run_label or "unknown_run")
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "methods_summary.csv"
        group.to_csv(path, index=False)
        outputs[str(run_label)] = out_dir
    return outputs


def _to_site_entry(row: Mapping[str, Any]) -> Dict[str, Any]:
    method = str(row.get("method") or "")
    rmsep = row.get("rmsep")
    r2 = row.get("r2")
    coverage = row.get("coverage")
    entry: Dict[str, Any] = {
        "site_id": row.get("site_id") or row.get("instrument_code") or row.get("site_factor"),
        "method": method,
        "rmsep": rmsep,
        "r2": r2,
        "coverage": coverage,
        "interval_width_pds": row.get("avg_pred_set_size"),
        "communication_bytes": row.get("bytes"),
        "dp_reported_eps": row.get("dp_target_eps") or row.get("design_dp_target_eps"),
        "spectral_drift": row.get("spectral_drift"),
        "drift_type": row.get("drift_type"),
    }
    if "fed" in method.lower():
        entry["rmsep_fedavg"] = rmsep
        entry["r2_fedavg"] = r2
        entry["coverage_fedavg"] = coverage
    else:
        entry["rmsep_pds"] = rmsep
        entry["r2_pds"] = r2
        entry["coverage_pds"] = coverage
    return {k: v for k, v in entry.items() if v is not None}


def export_federated_results(summary_df: pd.DataFrame, tree: ResultsTree) -> None:
    """Write ``federated_results.json`` files mirroring the layout expected by plotting scripts."""

    if summary_df.empty:
        return
    for run_label, group in summary_df.groupby("run_label"):
        out_dir = tree.aggregated / (run_label or "unknown_run")
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_label": run_label,
            "combo_ids": sorted({c for c in group.get("combo_id", []) if isinstance(c, str)}),
            "sites": [_to_site_entry(row) for row in group.to_dict(orient="records")],
        }
        (out_dir / "federated_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def summarize_metrics(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean/std grouped by design factors, method, and run label."""

    if summary_df.empty:
        return pd.DataFrame()
    metric_cols = [col for col in ("rmsep", "r2", "coverage", "avg_pred_set_size", "bytes", "round_time") if col in summary_df.columns]
    group_cols = ["run_label", "method", "site_id"]
    group_cols += [col for col in ("spectral_drift", "drift_type", "dp_target_eps", "design_dp_target_eps") if col in summary_df.columns]

    def _agg(series: pd.Series) -> Dict[str, float]:
        clean = series.dropna().astype(float)
        if clean.empty:
            return {"mean": float("nan"), "std": float("nan")}
        return {"mean": float(clean.mean()), "std": float(clean.std(ddof=0))}

    rows: list[dict[str, Any]] = []
    for keys, group in summary_df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        for metric in metric_cols:
            stats = _agg(group[metric])
            row[f"{metric}_mean"] = stats["mean"]
            row[f"{metric}_std"] = stats["std"]
        rows.append(row)
    return pd.DataFrame(rows)
