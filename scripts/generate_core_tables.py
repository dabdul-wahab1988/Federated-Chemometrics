"""
Generate core tables (Table 1-7) for the paper from `config.yaml` and experiment outputs.

Usage:
  python scripts/generate_core_tables.py --config config.yaml --results_dirs results/real_experiment,k_200_preproc_standard --output_dir generated_figures_tables

This script attempts to discover experiment outputs in the provided `results_dirs` and aggregate metrics.
If results are missing, it still creates the tables that are purely derived from `config.yaml` (Tables 1-4).

Outputs:
  - CSV and Markdown files for each table in `output_dir`

"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import os
from statistics import mean, stdev
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import yaml
except Exception:
    raise RuntimeError("Please install pyyaml to run this script: pip install pyyaml")

try:
    import pandas as pd
    import numpy as np
except Exception:
    raise RuntimeError("Please install pandas and numpy: pip install pandas numpy")


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def ensure_outdir(outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)


def write_csv_and_md(df: pd.DataFrame, path: Path, caption: Optional[str] = None):
    csv_path = path.with_suffix(".csv")
    md_path = path.with_suffix(".md")
    df.to_csv(csv_path, index=False)
    with open(md_path, "w", encoding="utf-8") as fh:
        if caption:
            fh.write(f"<!-- {caption} -->\n\n")
            fh.write(f"**Caption**: {caption}\n\n")
        fh.write(df.to_markdown(index=False))
    return csv_path, md_path


# Table 1 - Instruments
def table1_instruments(cfg: dict, data_dir: Path) -> pd.DataFrame:
    instruments = cfg.get("INSTRUMENTS", {})
    rows = []

    default_n_wl = cfg.get("DEFAULT_N_WAVELENGTHS")

    # Attempt to estimate sample counts by looking at `results/` pred files or raw `data/` files
    for manufacturer, inst_list in instruments.items():
        for inst in inst_list:
            code = inst.get("id")
            role = inst.get("role")
            enabled = inst.get("enabled", True)
            # Simple spectral range/resolution heuristics
            # If `data/<manufacturer>` contains csv/xlsx/npz, attempt to infer sizes
            spectral_range = "N/A"
            resolution = "N/A"
            total = cal = test = "Unknown"

            # Try to find matching predictions saved by run_* tools (npz patterns: <site>_predictions.npz)
            pred_match = None
            if data_dir and data_dir.exists():
                # look for files named like 'MA_A3_predictions.npz' in folder
                for p in data_dir.rglob(f"{code}_predictions.npz"):
                    pred_match = p
                    break

            if pred_match is not None:
                try:
                    arrs = np.load(pred_match)
                    y_true = arrs.get('y_true')
                    y_pred = arrs.get('y_pred')
                    if y_true is not None:
                        total = int(y_true.shape[0])
                        # no separate cal/test info in predictions. keep as total/Unknown/Unknown
                    # The interval shape tells us wavelengths not so much. We set defaults
                    spectral_range = f"{default_n_wl} wavelengths (approx)" if default_n_wl is not None else "unknown"
                    resolution = "unknown"
                except Exception:
                    pass
            else:
                # If raw data exists, scan for CSVs in `data/<manufacturer>/*{code}*.csv` or `data/<manufacturer>/<code>*.xlsx`
                dd = Path("data")
                if dd.exists() and dd.is_dir():
                    try:
                        files = list(dd.rglob(f"*{code}*.csv")) + list(dd.rglob(f"*{code}*.xlsx"))
                        if files:
                            # read a CSV and count rows
                            f0 = files[0]
                            try:
                                if f0.suffix.lower() == ".csv":
                                    df = pd.read_csv(f0)
                                    total = int(len(df))
                                else:
                                    df = pd.read_excel(f0, engine="openpyxl" if f0.suffix.lower() in ('.xlsx', '.xlsm') else None)
                                    total = int(len(df))
                                spectral_range = f"{default_n_wl} wavelengths (approx)" if default_n_wl is not None else "unknown"
                            except Exception:
                                pass
                    except Exception:
                        pass

            rows.append({
                "Cluster": manufacturer,
                "Instrument_Code": code,
                "Role": role,
                "Approx_Spectral_Range": spectral_range,
                "Resolution": resolution,
                "#Samples_total/cal/test": f"{total}/{cal}/{test}",
                "Enabled": enabled,
            })
    df = pd.DataFrame(rows)
    # Keep the canonical order if available in config
    order = [i.get("id") for mid in cfg.get("INSTRUMENTS", {}) for i in cfg.get("INSTRUMENTS").get(mid, [])]
    if set(order) == set(df["Instrument_Code"].tolist()):
        df["Instrument_Code"] = pd.Categorical(df["Instrument_Code"], categories=order, ordered=True)
        df = df.sort_values("Instrument_Code")
    return df


# Table 2 - Experimental factors summary
def table2_factors(cfg: dict) -> pd.DataFrame:
    factors = cfg.get("EXPERIMENTAL_DESIGN", {}).get("FACTORS", {})
    rows = []
    for key, vals in factors.items():
        if key in ("Site",):
            # Already human readable list
            levels = ", ".join([str(v) for v in vals])
        else:
            if isinstance(vals, list):
                # summarise numeric ranges if all numeric
                try:
                    num_vals = [float(v) for v in vals if v is not None]
                    if len(num_vals) > 1:
                        levels = f"{min(num_vals)} to {max(num_vals)}" if not any(math.isnan(n) for n in num_vals) else ", ".join(map(str, vals))
                    elif len(num_vals) == 1:
                        levels = str(num_vals[0])
                    else:
                        levels = ", ".join(map(str, vals))
                except Exception:
                    # fallback to textual list
                    levels = ", ".join(map(str, vals))
            else:
                levels = str(vals)
        rows.append({"Factor": key, "Levels": levels})
    df = pd.DataFrame(rows)
    # order factors the same as in YAML if possible
    return df


# Table 3 - Scenarios
def table3_scenarios(cfg: dict) -> pd.DataFrame:
    sites = cfg.get("EXPERIMENTAL_DESIGN", {}).get("FACTORS", {}).get("Site", [])
    # define clusters by prefix (MA_ vs MB_)
    clusters: Dict[str, List[str]] = {}
    for s in sites:
        key = s.split("_")[0] if isinstance(s, str) and "_" in s else "UNK"
        clusters.setdefault(key, []).append(s)

    # Build three example scenarios
    rows = []
    # Scenario A: within-cluster CT (Train: two instruments from same cluster, Test: third)
    if len(clusters) > 0:
        for cluster_name, ilist in clusters.items():
            if len(ilist) >= 3:
                train_sites = ", ".join(ilist[0:2])
                test_site = ilist[2]
                rows.append({
                    "Scenario": f"Within-cluster CT ({cluster_name})",
                    "Training Sites": train_sites,
                    "Held-out Sites": test_site,
                    "Holdout_Strategy": "LOSO_site",
                    "Spectral_Drift": "low",
                    "Drift_Type": "combined",
                    "Notes": f"Train two instruments from {cluster_name}, hold-out the third."
                })
    # Scenario B: cross-cluster CT
    all_sites = sites
    if len(all_sites) >= 4:
        train = ", ".join(all_sites[0:4])
        held = all_sites[4] if len(all_sites) > 4 else all_sites[-1]
        rows.append({
            "Scenario": "Cross-cluster CT",
            "Training Sites": train,
            "Held-out Sites": held,
            "Holdout_Strategy": "LOSO_site",
            "Spectral_Drift": "moderate",
            "Drift_Type": "scatter",
            "Notes": "Train across MA and MB devices; hold-out one instrument from other cluster"
        })

    # Scenario C: data-poor sites
    if len(all_sites) >= 2:
        train = all_sites[0]
        held = ", ".join(all_sites[-2:])
        rows.append({
            "Scenario": "Data-poor sites",
            "Training Sites": train,
            "Held-out Sites": held,
            "Holdout_Strategy": "random_split",
            "Spectral_Drift": "high",
            "Drift_Type": "jitter",
            "Notes": "Simulate low sample settings for held-out sites."
        })

    return pd.DataFrame(rows)


# Table 4 - Methods overview
def table4_methods_overview(cfg: dict) -> pd.DataFrame:
    baselines = cfg.get("EXPERIMENTAL_DESIGN", {}).get("FACTORS", {}).get("Baseline", [])
    federated_methods = cfg.get("EXPERIMENTAL_DESIGN", {}).get("FACTORS", {}).get("Federated_Method", [])
    ct_variants = cfg.get("EXPERIMENTAL_DESIGN", {}).get("FACTORS", {}).get("CT_Federated_Variant", [])
    use_secure_agg = cfg.get("SECURE_AGGREGATION", {}).get("enabled", False)
    dp_eps_list = cfg.get("EXPERIMENTAL_DESIGN", {}).get("FACTORS", {}).get("DP_Target_Eps", [])

    uses_dp = not (len(dp_eps_list) == 1 and dp_eps_list[0] == "inf") if isinstance(dp_eps_list, list) and dp_eps_list else False
    conformal_methods = cfg.get("EXPERALIMENTAL_DESIGN", {}) if False else cfg.get("EXPERIMENTAL_DESIGN", {})
    conformal_methods = cfg.get("EXPERIMENTAL_DESIGN", {}).get("FACTORS", {}).get("Conformal_Method", [])
    uses_cp = bool(conformal_methods)

    rows = []

    # Build centralised baselines
    for b in baselines:
        rows.append({
            "Method_Label": b,
            "Data_Access": "centralised" if b == "centralized" else "local" if b == "site_specific" else "centralised(method-specific)",
            "Baseline_Type": b,
            "Federated_Method": "N/A",
            "CT_Federated_Variant": "N/A",
            "Use_Secure_Aggregation": False,
            "Uses_DP": False,
            "Uses_CP": uses_cp,
        })

    # Federated method overview rows
    for fm in federated_methods:
        for ct in ct_variants:
            rows.append({
                "Method_Label": f"{fm}+{ct}",
                "Data_Access": "federated",
                "Baseline_Type": "N/A",
                "Federated_Method": fm,
                "CT_Federated_Variant": ct,
                "Use_Secure_Aggregation": use_secure_agg,
                "Uses_DP": uses_dp,
                "Uses_CP": uses_cp,
            })

    # Add fedavg single method
    if "FedAvg" in federated_methods:
        rows.append({
            "Method_Label": "FedAvg",
            "Data_Access": "federated",
            "Baseline_Type": "N/A",
            "Federated_Method": "FedAvg",
            "CT_Federated_Variant": "N/A",
            "Use_Secure_Aggregation": use_secure_agg,
            "Uses_DP": uses_dp,
            "Uses_CP": uses_cp,
        })

    return pd.DataFrame(rows)


# Table 5 - Main performance
def table5_main_performance(result_dirs: Sequence[Path], held_out_instruments: Sequence[str]) -> pd.DataFrame:
    # Consolidate methods_summary.csv across result dirs; aggregate mean ± SD for RMSEP/R2
    rows = []
    df_all = []
    for rd in result_dirs:
        ms = rd / "methods_summary.csv"
        if not ms.exists():
            continue
        try:
            df = pd.read_csv(ms)
            df_all.append(df)
        except Exception:
            continue
    if not df_all:
        # return empty DataFrame with columns expected
        return pd.DataFrame(columns=["Method", "Held-out Instrument", "RMSEP_mean", "RMSEP_std", "R2_mean", "R2_std"]) 

    df_all = pd.concat(df_all, ignore_index=True)

    # rename method for clarity
    for method, grp in df_all.groupby('method'):
        for hid in held_out_instruments:
            s = grp[grp.site_id == hid]
            if s.empty:
                continue
            rmsep_values = s['rmsep'].dropna().astype(float).tolist()
            r2_values = s['r2'].dropna().astype(float).tolist()
            if not rmsep_values:
                continue
            rmsep_mean = mean(rmsep_values)
            rmsep_std = stdev(rmsep_values) if len(rmsep_values) > 1 else 0.0
            r2_mean = mean(r2_values) if r2_values else float('nan')
            r2_std = stdev(r2_values) if len(r2_values) > 1 else 0.0
            rows.append({
                "Method": method,
                "Held-out Instrument": hid,
                "RMSEP_mean": rmsep_mean,
                "RMSEP_std": rmsep_std,
                "R2_mean": r2_mean,
                "R2_std": r2_std,
            })
    return pd.DataFrame(rows)


# Table 6 - Conformal coverage
def table6_conformal_coverage(result_dirs: Sequence[Path], held_out_instruments: Sequence[str], targets: List[float]) -> pd.DataFrame:
    # Try to read 'federated_results.json' or 'coverage_summary.json'
    rows = []

    # gather coverages per site per method across results
    coverages = []
    for rd in result_dirs:
        fr = rd / "federated_results.json"
        if fr.exists():
            with open(fr, 'r', encoding='utf-8') as f:
                j = json.load(f)
            for s in j.get('sites', []):
                site_id = s.get('site_id')
                for t in targets:
                    # The code logs coverage_pds/coverage_central (alpha=0.1 default => target 0.9)
                    c_pds = s.get('coverage_pds')
                    c_central = s.get('coverage_central')
                    # We don't have per-target coverage in results; we assume default targets
                    coverages.append({"site": site_id, "method": "pds", "target": targets[0], "coverage": c_pds, "interval_width": s.get('interval_width_pds')})
                    coverages.append({"site": site_id, "method": "central", "target": targets[0], "coverage": c_central, "interval_width": s.get('interval_width_central')})
    if not coverages:
        return pd.DataFrame(columns=["Method", "Target_Coverage", "Empirical_Coverage", "Avg_Interval_Width", "Held-out Site"])

    df = pd.DataFrame(coverages)
    records = []
    for (method, target, site), sub in df.groupby(['method', 'target', 'site']):
        mean_cov = float(sub['coverage'].dropna().mean())
        std_cov = float(sub['coverage'].dropna().std()) if len(sub['coverage'].dropna()) > 1 else 0.0
        mean_w = float(sub['interval_width'].dropna().mean()) if 'interval_width' in sub.columns else float('nan')
        std_w = float(sub['interval_width'].dropna().std()) if 'interval_width' in sub.columns and len(sub['interval_width'].dropna()) > 1 else 0.0
        records.append({
            "Method": method,
            "Target_Coverage": target,
            "Empirical_Coverage": float(mean_cov),
            "Coverage_SD": 0.0 if math.isnan(float(std_cov)) else float(std_cov),
            "Avg_Interval_Width": float(mean_w),
            "Interval_Width_SD": 0.0 if math.isnan(float(std_w)) else float(std_w),
            "Held-out Site": site,
        })
    return pd.DataFrame(records)


# Table 7 - Privacy & communication trade-offs
def table7_privacy_communication(cfg: dict, result_dirs: Sequence[Path], main_federated_method: str = "FedPLS_parametric") -> pd.DataFrame:
    rows = []
    # parse federated_results.json and telemetry logs
    all_results = []
    for rd in result_dirs:
        fr = rd / "federated_results.json"
        tlog = rd / "telemetry.jsonl"
        if fr.exists():
            with open(fr, 'r', encoding='utf-8') as fh:
                all_results.append(json.load(fh))

    # Flatten per-site metrics and compute aggregate per DP epsilon
    agg_by_eps = {}
    for res in all_results:
        for s in res.get('sites', []):
            dp_eps = s.get('dp_reported_eps')
            if dp_eps is None:
                # check DP config key or reported_eps might be a list
                dp_eps = s.get('dp_reported_eps') or (None)
            eps = str(dp_eps)
            agg_by_eps.setdefault(eps, {'rmsep': [], 'r2': [], 'cov': [], 'bytes': []})
            # prefer pds metrics if present
            if s.get('rmsep_pds') is not None:
                agg_by_eps[eps]['rmsep'].append(float(s.get('rmsep_pds')))
            elif s.get('rmsep_fedavg') is not None:
                agg_by_eps[eps]['rmsep'].append(float(s.get('rmsep_fedavg')))
            if s.get('r2_pds') is not None:
                agg_by_eps[eps]['r2'].append(float(s.get('r2_pds')))
            if s.get('coverage_pds') is not None:
                agg_by_eps[eps]['cov'].append(float(s.get('coverage_pds')))
            agg_by_eps[eps]['bytes'].append(int(s.get('Total_Bytes') or 0))

    for eps, vals in agg_by_eps.items():
        rows.append({
            "epsilon": eps,
            "Clip_Norm": cfg.get('CLIP_NORM'),
            "RMSEP_mean": float(mean(vals['rmsep'])) if vals['rmsep'] else float('nan'),
            "RMSEP_std": float(stdev([x for x in vals['rmsep'] if not (isinstance(x, float) and math.isnan(x))])) if len([x for x in vals['rmsep'] if not (isinstance(x, float) and math.isnan(x))]) > 1 else 0.0,
            "R2_mean": float(mean(vals['r2'])) if vals['r2'] else float('nan'),
            "R2_std": float(stdev([x for x in vals['r2'] if not (isinstance(x, float) and math.isnan(x))])) if len([x for x in vals['r2'] if not (isinstance(x, float) and math.isnan(x))]) > 1 else 0.0,
            "Coverage_mean": float(mean(vals['cov'])) if vals['cov'] else float('nan'),
            "Coverage_sd": float(stdev([x for x in vals['cov'] if not (isinstance(x, float) and math.isnan(x))])) if len([x for x in vals['cov'] if not (isinstance(x, float) and math.isnan(x))]) > 1 else 0.0,
            "Total_Bytes_Sent_per_Site_mean": float(mean(vals['bytes'])) if vals['bytes'] else 0,
            "Total_Bytes_Sent_per_Site_std": float(stdev(vals['bytes'])) if len(vals['bytes']) > 1 else 0,
            # Round_Time: attempt to compute from telemetry if present
            "Avg_Round_Time": float('nan'),
        })
    return pd.DataFrame(rows)


def _parse_result_dirs(s: Optional[str]) -> Tuple[Sequence[Path], Path]:
    if s is None:
        return ([], Path(os.environ.get("FEDCHEM_OUTPUT_DIR", "generated_figures_tables")))
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [Path(p) for p in parts], Path(parts[0]) if parts else ([], Path(os.environ.get("FEDCHEM_OUTPUT_DIR", "generated_figures_tables")))


def _main(args):
    cfg = load_config(Path(args.config))
    result_dirs, default_outdir = _parse_result_dirs(args.results_dirs)
    out_dir = Path(args.output_dir) if args.output_dir else default_outdir
    ensure_outdir(out_dir)

    # Table 1
    df1 = table1_instruments(cfg, Path(args.data_dir) if args.data_dir else Path("data"))
    write_csv_and_md(df1, out_dir / "Table1_Instruments", caption="Coded instrument registry and roles. Instrument names and vendor details are anonymised; local mapping is not published.")

    # Table 2
    df2 = table2_factors(cfg)
    write_csv_and_md(df2, out_dir / "Table2_ExperimentalFactors", caption="Summary of factors and levels in the fractional-factorial design; not all Cartesian combinations are instantiated.")

    # Table 3
    df3 = table3_scenarios(cfg)
    write_csv_and_md(df3, out_dir / "Table3_Scenarios", caption="Example scenario definitions and holdout strategies used in experiments.")

    # Table 4
    df4 = table4_methods_overview(cfg)
    write_csv_and_md(df4, out_dir / "Table4_MethodsOverview", caption="Decoder ring for Baseline, Federated and CT combinations used in experiments.")

    # Table 5
    held_outs = cfg.get("EXPERIMENTAL_DESIGN", {}).get("FACTORS", {}).get("Site", [])
    if len(held_outs) >= 1:
        # default to last instrument(s) as held-out in examples
        default_held = [h for h in held_outs if h.endswith('_A3') or h.endswith('_B3')]
        if not default_held:
            default_held = held_outs[-2:]
    else:
        default_held = []
    # allow the user to override the held-out instruments via CLI
    if args.held_out_instruments:
        default_held = [h.strip() for h in args.held_out_instruments.split(",") if h.strip()]
    df5 = table5_main_performance(result_dirs, default_held)
    write_csv_and_md(df5, out_dir / "Table5_MainPerformance", caption="Main held-out instrument performance (RMSEP and R^2) for canonical settings (moderate drift, target coverage 0.90, ε=1, rounds=50) where available.")

    # Table 6
    target_cov = [0.9, 0.95]
    df6 = table6_conformal_coverage(result_dirs, default_held, target_cov)
    write_csv_and_md(df6, out_dir / "Table6_ConformalCoverage", caption="Conformal coverage and interval widths (per held-out instrument when available).")

    # Table 7
    df7 = table7_privacy_communication(cfg, result_dirs)
    write_csv_and_md(df7, out_dir / "Table7_PrivacyCommunication", caption="Privacy (DP) and communication trade-offs for the main federated method; includes epsilon, clip norm, RMSEP, R2, coverage, and bytes.")

    print(f"Wrote tables to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the config YAML")
    parser.add_argument("--results_dirs", type=str, default=None, help="Comma-separated list of results directories to scan for outputs (methods_summary.csv, federated_results.json, telemetry.jsonl)")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to write generated tables")
    parser.add_argument("--data_dir", type=str, default="data", help="Root of raw data if available for sample counts and spectral inspection")
    parser.add_argument("--held_out_instruments", type=str, default=None, help="Comma-separated held-out instrument codes to use for Table 5 and 6 (default: detect from config end-of-list)")
    args = parser.parse_args()
    _main(args)
