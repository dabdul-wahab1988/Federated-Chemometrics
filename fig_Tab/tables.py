from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import functools
import numpy as np
import pandas as pd

from .data_access import load_publication_database, load_aggregated_frames

BASELINE_METHODS = ["Centralized_PLS", "Site_Specific", "PDS", "SBC"]
FED_METHODS = ["FedAvg", "FedPLS", "FedProx", "Global_calibrate_after_fed"]
METHOD_DISPLAY = {
    "Centralized_PLS": "Pooled PLS",
    "Site_Specific": "Site-specific PLS",
    "PDS": "PDS",
    "SBC": "SBC",
    "FedAvg": "FedAvg",
    "FedPLS": "FedPLS",
    "FedProx": "FedProx",
    "Global_calibrate_after_fed": "Global Cal.",
}
DP_COLUMNS = ["design_factors.DP_Target_Eps", "dp_target_eps", "dp_epsilon"]
TRANSFER_COLUMNS = ["design_factors.Transfer_Samples", "transfer_k"]
RMSEP_COLUMNS = ["metrics.RMSEP", "rmsep"]
R2_COLUMNS = ["metrics.R2", "r2"]
BYTES_COLUMNS = ["runtime.total_bytes_mb", "bytes_mb"]
ROUND_TIME_COLUMNS = ["runtime.wall_time_total_sec", "round_time_sec", "metrics.Round_Time"]
INSTRUMENT_COLUMNS = ["instrument_code", "site_id", "site_code"]


def _ensure_outdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _first_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _format_mean_std(mean: float, std: float, decimals: int = 3) -> str:
    if pd.isna(mean) and pd.isna(std):
        return ""
    if pd.isna(std) or std == 0:
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def _write_table(df: pd.DataFrame, output_path: Path, description: str) -> None:
    _ensure_outdir(output_path.parent)
    df.to_csv(output_path, index=False)
    print(f"[OK] Saved table: {output_path.name} ({description})")


@functools.lru_cache(maxsize=4)
def _load_publication_frames(results_root_str: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (performance_metrics, master_database) frames."""
    results_root = Path(results_root_str)
    pub_path = results_root / "publication_database"
    if pub_path.exists():
        master_df, perf_df, _, _ = load_publication_database(pub_path)
        if master_df is None:
            master_df = pd.DataFrame()
        if perf_df is None:
            perf_df = pd.DataFrame()
        if not master_df.empty:
            master_df = _map_publication_columns(master_df)
        return perf_df, master_df

    methods_df, _, raw_df = load_aggregated_frames(results_root)
    if raw_df is None:
        raw_df = pd.DataFrame()
    if methods_df is None:
        methods_df = pd.DataFrame()
    return methods_df, raw_df


def _map_publication_columns(master_df: pd.DataFrame) -> pd.DataFrame:
    df = master_df.copy()
    column_mapping = {
        "transfer_k": "design_factors.Transfer_Samples",
        "dp_epsilon": "design_factors.DP_Target_Eps",
        "rmsep": "metrics.RMSEP",
        "r2": "metrics.R2",
        "bytes_mb": "runtime.total_bytes_mb",
        "bytes_sent": "metrics.Bytes_Sent",
        "bytes_received": "metrics.Bytes_Received",
        "total_bytes": "metrics.Total_Bytes",
        "round_time_sec": "runtime.wall_time_total_sec",
    }
    rename_dict = {k: v for k, v in column_mapping.items() if k in df.columns}
    df = df.rename(columns=rename_dict)

    if "run_label" not in df.columns:
        df["run_label"] = "objective_1"
    else:
        df["run_label"] = df["run_label"].fillna("objective_1")

    eps_col = _first_column(df, DP_COLUMNS)
    if eps_col:
        df[eps_col] = df[eps_col].astype(str).str.lower().replace({"infinity": "inf"})

    transfer_col = _first_column(df, TRANSFER_COLUMNS)
    if transfer_col:
        # Attempt numeric conversion; if it fails, keep original values (preserve the previous "ignore" behavior)
        try:
            df[transfer_col] = pd.to_numeric(df[transfer_col], errors="raise")
        except (ValueError, TypeError):
            # Leave the column as-is when conversion is not possible
            pass
        df[transfer_col] = converted.where(~converted.isna(), df[transfer_col])

    return df


def _load_master_frame(results_root: Path) -> pd.DataFrame:
    perf_df, master_df = _load_publication_frames(str(results_root))
    if not master_df.empty:
        return master_df
    return perf_df if perf_df is not None else pd.DataFrame()


def _select_instrument_column(df: pd.DataFrame) -> Optional[str]:
    return _first_column(df, INSTRUMENT_COLUMNS)


def table_baseline_performance(results_root: Path, output_path: Path) -> None:
    df = _load_master_frame(results_root)
    if df.empty:
        print("[SKIP] No data available for baseline table.")
        return

    eps_col = _first_column(df, DP_COLUMNS)
    rmsep_col = _first_column(df, RMSEP_COLUMNS)
    r2_col = _first_column(df, R2_COLUMNS)
    instr_col = _select_instrument_column(df) or "instrument_code"

    subset = df[df["method"].isin(BASELINE_METHODS)].copy()
    if eps_col and eps_col in subset.columns:
        subset = subset[subset[eps_col].astype(str) == "inf"]

    if subset.empty or rmsep_col is None:
        print("[SKIP] Baseline methods missing in dataset.")
        return

    pivots: List[pd.DataFrame] = []
    rmsep_pivot = subset.pivot_table(index=instr_col, columns="method", values=rmsep_col, aggfunc="mean")
    rmsep_pivot = rmsep_pivot.reindex(columns=BASELINE_METHODS)
    rmsep_pivot = rmsep_pivot.rename(columns=lambda c: f"RMSEP_{METHOD_DISPLAY.get(c, c)}")
    pivots.append(rmsep_pivot)

    if r2_col:
        r2_pivot = subset.pivot_table(index=instr_col, columns="method", values=r2_col, aggfunc="mean")
        r2_pivot = r2_pivot.reindex(columns=BASELINE_METHODS)
        r2_pivot = r2_pivot.rename(columns=lambda c: f"R2_{METHOD_DISPLAY.get(c, c)}")
        pivots.append(r2_pivot)

    table_df = pd.concat(pivots, axis=1).reset_index()
    table_df = table_df.rename(columns={instr_col: "instrument"})
    _write_table(table_df, output_path, "Baseline performance comparison")


def table_privacy_utility_summary(results_root: Path, output_path: Path) -> None:
    df = _load_master_frame(results_root)
    if df.empty:
        print("[SKIP] No data for privacy table.")
        return

    rmsep_col = _first_column(df, RMSEP_COLUMNS)
    r2_col = _first_column(df, R2_COLUMNS)
    bytes_col = _first_column(df, BYTES_COLUMNS)
    round_col = _first_column(df, ROUND_TIME_COLUMNS)
    eps_col = _first_column(df, DP_COLUMNS)
    transfer_col = _first_column(df, TRANSFER_COLUMNS)

    if rmsep_col is None or eps_col is None or transfer_col is None:
        print("[SKIP] Missing core columns for privacy table.")
        return

    subset = df[df["method"].isin(FED_METHODS)].copy()
    if subset.empty:
        print("[SKIP] No federated records for privacy table.")
        return

    agg_map = {
        rmsep_col: ["mean", "std"],
    }
    if r2_col:
        agg_map[r2_col] = ["mean"]
    if bytes_col:
        agg_map[bytes_col] = ["mean"]
    if round_col:
        agg_map[round_col] = ["mean"]

    grouped = subset.groupby([transfer_col, eps_col]).agg(agg_map)
    grouped.columns = ["_".join(filter(None, col)).strip("_") for col in grouped.columns.to_flat_index()]
    grouped = grouped.reset_index().rename(columns={transfer_col: "transfer_k", eps_col: "dp_epsilon"})

    grouped["RMSEP (mean±std)"] = grouped.apply(
        lambda row: _format_mean_std(row.get(f"{rmsep_col}_mean", np.nan), row.get(f"{rmsep_col}_std", np.nan)),
        axis=1,
    )

    rename_cols: Dict[str, str] = {
        f"{rmsep_col}_mean": "RMSEP_mean",
        f"{rmsep_col}_std": "RMSEP_std",
    }
    if r2_col:
        rename_cols[f"{r2_col}_mean"] = "R2_mean"
    if bytes_col:
        rename_cols[f"{bytes_col}_mean"] = "Communication_MB"
    if round_col:
        rename_cols[f"{round_col}_mean"] = "Round_Time_sec"

    grouped = grouped.rename(columns=rename_cols)
    ordered_cols = ["transfer_k", "dp_epsilon", "RMSEP (mean±std)"] + [c for c in rename_cols.values() if c in grouped.columns]
    grouped = grouped[ordered_cols].sort_values(["transfer_k", "dp_epsilon"], key=lambda s: s.map(str))
    _write_table(grouped, output_path, "Privacy vs utility summary")


def table_conformal_coverage(results_root: Path, output_path: Path) -> None:
    pub_path = results_root / "publication_database"
    if not pub_path.exists():
        print("[SKIP] publication_database missing for conformal table.")
        return
    _, _, conf_df, _ = load_publication_database(pub_path)
    if conf_df is None or conf_df.empty:
        print("[SKIP] No conformal metrics available.")
        return

    rename_map = {
        "Site": "site_id",
        "transfer_k": "transfer_k",
        "instrument_code": "instrument_code",
        "Global_Coverage": "Global_Coverage",
        "Global_MeanWidth": "Global_MeanWidth",
        "Mondrian_Coverage": "Mondrian_Coverage",
        "Mondrian_MeanWidth": "Mondrian_MeanWidth",
        "Alpha": "alpha",
        "Nominal": "nominal",
    }
    conf_df = conf_df.rename(columns=rename_map)
    cols = [c for c in ["transfer_k", "instrument_code", "alpha", "nominal", "Global_Coverage", "Global_MeanWidth", "Mondrian_Coverage", "Mondrian_MeanWidth"] if c in conf_df.columns]
    table_df = conf_df[cols].sort_values(["transfer_k", "instrument_code"])
    _write_table(table_df, output_path, "Conformal coverage summary")


def generate_all_tables(
    results_root: Path,
    config_path: Path,
    tables_dir: Path,
    data_dir: Path,
    n_wavelengths: Optional[int] = None,
) -> None:
    del config_path, data_dir, n_wavelengths  # Unused but kept for API compatibility
    _ensure_outdir(tables_dir)

    tasks = [
        ("Table_3_Baseline_Performance_Comparison.csv", table_baseline_performance),
        ("Table_4_Privacy_Utility_Summary.csv", table_privacy_utility_summary),
        ("Table_5_Conformal_Coverage_Summary.csv", table_conformal_coverage),
    ]

    for filename, func in tasks:
        try:
            func(results_root, tables_dir / filename)
        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"[ERROR] Failed to build {filename}: {exc}")
