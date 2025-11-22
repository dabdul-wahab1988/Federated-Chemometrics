"""
Generate a reproducible site paired transfer dataset and run PDS experiments A and B.

Outputs:
- data/site_design/: per-site X.npy, y.npy, meta.json, **dataset.csv**, **dataset.xlsx**
- data/site_design/site_design_combined.(csv|xlsx) aggregating every synthetic sample
- generated_figures_tables/figure_site.png and meta (via generate_objective_site)
- generated_figures_tables/site_pds_diagnostics.csv (per-site diagnostics for A and B)
- generated_figures_tables/site_manifest.json

Run: python tools/generate_site_design.py

This script uses the project's SpectralSimulator and existing objective scripts to
produce datasets compatible with the repo's benchmarking functions and now also
exports synthetic samples in spreadsheet-friendly formats for direct exploration.
"""

from __future__ import annotations
import json
import os
from pathlib import Path
import os
import numpy as np
import pandas as pd
import warnings
from typing import Dict, Any, Optional, List

# Ensure repo path
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "site_design"
# Allow env override for output dir (e.g., set by run_all_objectives to place everything under results tree)
default_out = ROOT / "generated_figures_tables"
OUT_DIR = Path(os.environ.get("FEDCHEM_OUTPUT_DIR", str(default_out)))
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Project imports
from fedchem.simulator.core import SpectralSimulator
from fedchem.preprocess.pipeline import PreprocessPipeline
from fedchem.utils.config import load_and_seed_config
from fedchem.ct.pds_transfer import PDSTransfer
from fedchem.models.pls import PLSModel
from fedchem.metrics.metrics import rmsep, r2

# reuse helper from generate_objective_site for plotting if desired
import matplotlib.pyplot as plt
import math


def _feature_columns(wavelengths: List[float]) -> List[str]:
    """Create human-readable column headers for spectral wavelengths."""
    return [f"wl_{int(round(w))}" for w in wavelengths]


def _build_dataframe(site: str, X: np.ndarray, y: np.ndarray, sample_roles: List[str], wavelengths: List[float]) -> pd.DataFrame:
    if len(sample_roles) != X.shape[0]:
        raise ValueError("sample_roles length must match number of samples")
    cols = _feature_columns(wavelengths)
    df = pd.DataFrame(X, columns=cols)
    df.insert(0, "sample_index", np.arange(X.shape[0], dtype=int))
    df.insert(0, "sample_role", sample_roles)
    df.insert(0, "site", site)
    df["target"] = y
    return df


def _write_tabular(df: pd.DataFrame, csv_path: Path, xlsx_path: Path) -> None:
    df.to_csv(csv_path, index=False)
    try:
        df.to_excel(xlsx_path, index=False)
    except Exception as exc:  # pragma: no cover - optional Excel writer dependency
        warnings.warn(f"Could not write Excel file {xlsx_path}: {exc}")


def build_paired_dataset(n_sites: int = 3, n_transfer: int = 200, n_extra_per_site: int = 20, seed: int = 42):
    """Create a paired dataset: reference bank + per-site distorted copies.

    - Generate a reference bank of n_transfer + n_extra_per_site samples
    - For each site, produce a paired copy of the first n_transfer samples via apply_distortions
    - For each site, append n_extra_per_site additional site-unique samples to serve as local test samples
    Returns a data dict mapping site-> {X,y} and wavelengths, meta
    """
    rng = np.random.default_rng(int(seed))
    wavelengths = list(np.linspace(400, 2500, 256))
    sim = SpectralSimulator(seed=int(seed))

    # Create a reference bank with total_ref = n_transfer + n_extra_per_site
    total_ref = n_transfer + n_extra_per_site
    ref_sites = sim.generate_sites(n_sites=1, n_samples_per_site=total_ref, wavelengths=wavelengths, modalities=["nir"]) 
    ref = ref_sites["site_0"]
    X_ref = np.asarray(ref["X"], dtype=float)
    y_ref = np.asarray(ref["y"], dtype=float)

    data: Dict[str, Dict[str, Any]] = {}
    per_site_tabular: Dict[str, Dict[str, str]] = {}
    combined_frames: List[pd.DataFrame] = []

    # per-site distortion configs (varied deterministically)
    base_configs = [
        {"scatter": 1.02, "baseline": 0.02, "noise_std": 0.01},
        {"scatter": 0.98, "baseline": -0.01, "noise_std": 0.015},
        {"scatter": 1.05, "baseline": 0.03, "noise_std": 0.02},
        {"scatter": 0.95, "baseline": -0.02, "noise_std": 0.025},
        {"scatter": 1.00, "baseline": 0.0, "noise_std": 0.012},
    ]

    for s in range(n_sites):
        cfg = base_configs[s % len(base_configs)]
        # paired portion
        X_pair = sim.apply_distortions(X_ref[:n_transfer], cfg)
        y_pair = y_ref[:n_transfer].copy()
        # extra site-unique samples: generate fresh site-local spectra
        site_extra = sim.generate_sites(n_sites=1, n_samples_per_site=n_extra_per_site, wavelengths=wavelengths, modalities=["nir"])['site_0']
        X_extra = np.asarray(site_extra['X'], dtype=float)
        # compute y_extra consistently using same linear weight as simulator: w = linspace(0.1,1.0,d)
        w = np.linspace(0.1, 1.0, X_extra.shape[1])
        y_extra = X_extra @ w + rng.normal(0.0, 0.1, size=(X_extra.shape[0],))
        # assemble site data: paired rows first, then extras
        X_site = np.vstack([X_pair, X_extra]).astype(np.float32)
        y_site = np.hstack([y_pair, y_extra]).astype(np.float32)
        data[f"site_{s}"] = {"X": X_site, "y": y_site}

    # Save reference bank (useful for traceability)
    ref_dir = DATA_DIR / "reference"
    ref_dir.mkdir(parents=True, exist_ok=True)
    np.save(ref_dir / "X_ref.npy", X_ref.astype(np.float32))
    np.save(ref_dir / "y_ref.npy", y_ref.astype(np.float32))
    (ref_dir / "meta.json").write_text(json.dumps({"wavelengths": wavelengths, "total_ref": int(total_ref)}, indent=2))
    ref_roles = ["reference"] * X_ref.shape[0]
    ref_df = _build_dataframe("reference", X_ref, y_ref, ref_roles, wavelengths)
    ref_csv = ref_dir / "dataset.csv"
    ref_xlsx = ref_dir / "dataset.xlsx"
    _write_tabular(ref_df, ref_csv, ref_xlsx)
    reference_tabular = {"csv": str(ref_csv), "xlsx": str(ref_xlsx)}

    # Save per-site data
    for k, v in data.items():
        site_dir = DATA_DIR / k
        site_dir.mkdir(exist_ok=True, parents=True)
        np.save(site_dir / "X.npy", np.asarray(v['X'], dtype=np.float32))
        np.save(site_dir / "y.npy", np.asarray(v['y'], dtype=np.float32))
        meta = {"n_samples": int(v['X'].shape[0]), "paired_transfer_n": int(n_transfer)}
        (site_dir / "meta.json").write_text(json.dumps(meta, indent=2))

        roles = ["paired"] * min(n_transfer, v['X'].shape[0])
        if len(roles) < v['X'].shape[0]:
            roles.extend(["local"] * (v['X'].shape[0] - len(roles)))
        df = _build_dataframe(k, v['X'], v['y'], roles, wavelengths)
        csv_path = site_dir / "dataset.csv"
        xlsx_path = site_dir / "dataset.xlsx"
        _write_tabular(df, csv_path, xlsx_path)
        per_site_tabular[k] = {"csv": str(csv_path), "xlsx": str(xlsx_path)}
        combined_frames.append(df)

    combined_tabular = None
    if combined_frames:
        combined_df = pd.concat(combined_frames, ignore_index=True)
        combined_csv = DATA_DIR / "site_design_combined.csv"
        combined_xlsx = DATA_DIR / "site_design_combined.xlsx"
        _write_tabular(combined_df, combined_csv, combined_xlsx)
        combined_tabular = {"csv": str(combined_csv), "xlsx": str(combined_xlsx)}

    # manifest
    if combined_tabular is None:
        combined_tabular = {}

    manifest = {
        "n_sites": n_sites,
        "n_transfer": n_transfer,
        "n_extra_per_site": n_extra_per_site,
        "seed": int(seed),
        "wavelengths": len(wavelengths),
        "tabular_exports": {
            "reference": reference_tabular,
            "per_site": per_site_tabular,
            "combined": combined_tabular,
        },
    }
    (DATA_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))

    return data, wavelengths, {"seed": seed, "n_sites": n_sites}


def compute_pds_experiment(data: Dict[str, Dict[str, Any]], transfer_n: int = 200, preprocess: Optional[PreprocessPipeline] = None):
    """Run a PDS experiment across sites and return diagnostics.

    If preprocess is provided, it will be fit on the pooled X_pool and used to transform
    X_pool, X_tr, X_te before training PDSTransfer and the central PLS model.
    """
    sites = sorted(list(data.keys()))
    # build X_pool as in generate_objective_5
    pool_X_list = [data[s]["X"][:80] for s in sites if data[s]["X"].shape[0] >= 100]
    if len(pool_X_list) > 0:
        X_pool = np.vstack(pool_X_list)
        y_pool = np.hstack([data[s]["y"][:80] for s in sites if data[s]["y"].shape[0] >= 100])
    else:
        pool_X_list = [data[s]["X"][: max(1,int(0.6*data[s]["X"].shape[0]))] for s in sites]
        X_pool = np.vstack(pool_X_list)
        y_pool = np.hstack([data[s]["y"][: max(1,int(0.6*data[s]["y"].shape[0]))] for s in sites])

    # Optionally fit preprocess pipeline on X_pool
    pipeline = None
    if preprocess is not None:
        pipeline = preprocess.fit(X_pool)
        X_pool_proc = pipeline.transform(X_pool)
    else:
        X_pool_proc = X_pool

    # Fit central model on pooled (possibly preprocessed) data
    central = PLSModel().fit(X_pool_proc, y_pool)

    methods = ['Centralized', 'Site-specific', 'PDS', 'SBC']
    method_rmse = {m: [] for m in methods}
    per_site_preds = {}
    diags_rows = []

    for s in sites:
        X = data[s]["X"].astype(float)
        y = data[s]["y"].astype(float)
        n = X.shape[0]
        n_tr = max(1, int(0.8 * n))
        X_tr, y_tr, X_te, y_te = X[:n_tr], y[:n_tr], X[n_tr:], y[n_tr:]

        # local site model
        model_site = PLSModel().fit(X_tr, y_tr)
        yhat_site = model_site.predict(X_te)
        method_rmse['Site-specific'].append(float(rmsep(y_te, yhat_site)))
        # centralized predictions (central trained on pooled X_pool_proc) -> need to transform X_te similarly
        if pipeline is not None:
            X_te_proc = pipeline.transform(X_te)
        else:
            X_te_proc = X_te
        yhat_c = central.predict(X_te_proc)
        method_rmse['Centralized'].append(float(rmsep(y_te, yhat_c)))

        # PDS: prepare k and data for transfer fit
        k = min(transfer_n, n_tr, X_pool.shape[0])
        X_pool_k = X_pool_proc[:k]
        # target must be site training rows, optionally preprocess them using same pipeline
        if pipeline is not None:
            X_tr_k = pipeline.transform(X_tr[:k])
        else:
            X_tr_k = X_tr[:k]

        # Fit PDSTransfer on (X_pool_k, X_tr_k)
        try:
            pds = PDSTransfer(window=32, overlap=16, ridge=1e-1).fit(X_pool_k, X_tr_k)
            diags = pds.diagnostics
            mode = pds.to_dict().get("mode")
            # Transform test set (must be preprocessed if pipeline is used)
            X_te_in = X_te_proc if pipeline is None else pipeline.transform(X_te)
            Xte_trans = pds.transform(X_te_in)
            # central predict on transformed features
            yhat_pds = central.predict(Xte_trans)
            method_rmse['PDS'].append(float(rmsep(y_te, yhat_pds)))
            # SBC
            yhat_tr = central.predict(pipeline.transform(X_tr) if pipeline is not None else X_tr)
            A = np.vstack([yhat_tr, np.ones_like(yhat_tr)]).T
            coeffs, *_ = np.linalg.lstsq(A, y_tr, rcond=None)
            a, b = coeffs
            yhat_adj = a * yhat_c + b
            method_rmse['SBC'].append(float(rmsep(y_te, yhat_adj)))

            # collect diagnostics
            diags_rows.append({
                "site": s,
                "mode": mode,
                "max_block_cond": float(np.max(diags.cond_numbers) if diags.cond_numbers else np.nan),
                "mean_block_rmse": float(diags.mean_rmse),
                "pds_rmsep": float(rmsep(y_te, yhat_pds)),
                "central_rmsep": float(rmsep(y_te, yhat_c)),
                "site_specific_rmsep": float(rmsep(y_te, yhat_site)),
                "r2_pds": float(r2(y_te, yhat_pds)),
            })
            per_site_preds[s] = {"y_te": y_te.tolist(), "yhat_pds": yhat_pds.tolist()}
        except Exception as e:
            diags_rows.append({"site": s, "error": str(e)})
            method_rmse['PDS'].append(float('nan'))
            method_rmse['SBC'].append(float('nan'))
            per_site_preds[s] = {"y_te": y_te.tolist(), "yhat_pds": [float('nan')] * len(y_te)}

    return {
        "methods": methods,
        "sites": sites,
        "method_rmse": method_rmse,
        "per_site_preds": per_site_preds,
        "diags": diags_rows,
    }


def run_and_save(seed: int = 42):
    # build dataset
    data, wavelengths, meta = build_paired_dataset(n_sites=3, n_transfer=200, n_extra_per_site=20, seed=seed)

    # Experiment A: k=200, default preprocessing (none)
    print("Running Experiment A: transfer_n=200, no preprocessing")
    resA = compute_pds_experiment(data, transfer_n=200, preprocess=None)
    (OUT_DIR / "site_pds_diagnostics_A.json").write_text(json.dumps(resA["diags"], indent=2))

    # Create a simple figure summarizing mean RMSEP per method for A
    methods = resA["methods"]
    meansA = [float(np.mean(resA["method_rmse"][m])) for m in methods]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(methods, meansA)
    ax.set_yscale('log')
    ax.set_ylabel('RMSEP (log scale)')
    ax.set_title('Experiment A: Mean RMSEP per method')
    plt.tight_layout()
    fig_pathA = OUT_DIR / "figure_site_A.png"
    fig.savefig(fig_pathA, dpi=200)
    plt.close(fig)
    (OUT_DIR / "figure_site_A_meta.json").write_text(json.dumps({"methods": methods, "mean_rmse": {m: float(np.mean(resA["method_rmse"][m])) for m in methods}}, indent=2))

    # Save dataset manifest and per-site files already written by build_paired_dataset

    # Experiment B: standardize before PDS (fit scaler on X_pool)
    print("Running Experiment B: transfer_n=200, standard scaling before PDS")
    cfg = load_and_seed_config()
    drift_cfg = cfg.get('DRIFT_AUGMENT') if isinstance(cfg.get('DRIFT_AUGMENT'), dict) else None
    def _merge_drift_env_overrides(dcfg):
        import os
        if not isinstance(dcfg, dict):
            dcfg = {}
        mapping = {
            'FEDCHEM_DRIFT_AUGMENT_JITTER_WAVELENGTH_PX': 'jitter_wavelength_px',
            'FEDCHEM_DRIFT_AUGMENT_MULTIPLICATIVE_SCATTER': 'multiplicative_scatter',
            'FEDCHEM_DRIFT_AUGMENT_BASELINE_OFFSET': 'baseline_offset',
            'FEDCHEM_DRIFT_AUGMENT_WHITE_NOISE_SIGMA': 'white_noise_sigma',
            'FEDCHEM_DRIFT_AUGMENT_AUGMENTATION_SEED': 'augmentation_seed',
            'FEDCHEM_DRIFT_AUGMENT_APPLY_AUGMENTATION_DURING_TRAINING': 'apply_augmentation_during_training',
            'FEDCHEM_DRIFT_AUGMENT_APPLY_TEST_SHIFTS': 'apply_test_shifts',
        }
        for env_key, cfg_key in mapping.items():
            if env_key in os.environ:
                val = os.environ.get(env_key)
                if val is None:
                    continue
                if cfg_key.startswith('apply_'):
                    dcfg[cfg_key] = val in {'1', 'true', 'True', 'yes', 'on'}
                    continue
                try:
                    if '.' in val:
                        dcfg[cfg_key] = float(val)
                    else:
                        dcfg[cfg_key] = int(val)
                except Exception:
                    try:
                        dcfg[cfg_key] = float(val)
                    except Exception:
                        dcfg[cfg_key] = val
        return dcfg

    drift_cfg = _merge_drift_env_overrides(drift_cfg)
    preprocess = PreprocessPipeline(scale="standard", drift_cfg=drift_cfg, augmentation_seed=seed)
    resB = compute_pds_experiment(data, transfer_n=200, preprocess=preprocess)
    (OUT_DIR / "site_pds_diagnostics_B.json").write_text(json.dumps(resB["diags"], indent=2))

    # Create a simple figure summarizing mean RMSEP per method for B
    methodsB = resB["methods"]
    meansB = [float(np.mean(resB["method_rmse"][m])) for m in methodsB]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(methodsB, meansB, color='C1')
    ax.set_yscale('log')
    ax.set_ylabel('RMSEP (log scale)')
    ax.set_title('Experiment B: Mean RMSEP per method (standardized)')
    plt.tight_layout()
    fig_pathB = OUT_DIR / "figure_site_B.png"
    fig.savefig(fig_pathB, dpi=200)
    plt.close(fig)
    (OUT_DIR / "figure_site_B_meta.json").write_text(json.dumps({"methods": methodsB, "mean_rmse": {m: float(np.mean(resB["method_rmse"][m])) for m in methodsB}}, indent=2))

    # Save a combined CSV for easy inspection
    import csv
    csv_path = OUT_DIR / "site_pds_diagnostics.csv"
    keys = ["site", "mode", "max_block_cond", "mean_block_rmse", "pds_rmsep", "central_rmsep", "site_specific_rmsep", "r2_pds"]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for row in resA["diags"]:
            out = {k: row.get(k, None) for k in keys}
            out = {"site": out["site"], **{f"A_{k}": out[k] for k in keys if k != "site"}}
            # find matching B
            brow = next((r for r in resB["diags"] if r.get("site") == row.get("site")), {})
            for k in keys:
                if k == "site":
                    continue
                out[f"B_{k}"] = brow.get(k, None)
            # normalize site value
            writer.writerow({
                "site": out["site"],
                "mode": f"A:{out.get('A_mode')}|B:{out.get('B_mode')}",
                "max_block_cond": f"A:{out.get('A_max_block_cond')}|B:{out.get('B_max_block_cond')}",
                "mean_block_rmse": f"A:{out.get('A_mean_block_rmse')}|B:{out.get('B_mean_block_rmse')}",
                "pds_rmsep": f"A:{out.get('A_pds_rmsep')}|B:{out.get('B_pds_rmsep')}",
                "central_rmsep": f"A:{out.get('A_central_rmsep')}|B:{out.get('B_central_rmsep')}",
                "site_specific_rmsep": f"A:{out.get('A_site_specific_rmsep')}|B:{out.get('B_site_specific_rmsep')}",
                "r2_pds": f"A:{out.get('A_r2_pds')}|B:{out.get('B_r2_pds')}",
            })

    # Save combined manifest summarizing experiments
    manifest = {
        "seed": int(seed),
        "experiment_A": {"transfer_n": 200, "preprocess": None},
        "experiment_B": {"transfer_n": 200, "preprocess": {"scale": "standard"}},
        "files": {"diagnostics_A": str((OUT_DIR / "site_pds_diagnostics_A.json")), "diagnostics_B": str((OUT_DIR / "site_pds_diagnostics_B.json")), "csv": str(csv_path)},
    }
    (OUT_DIR / "site_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("Done. Outputs written to:")
    print(" ", OUT_DIR)
    print(" ", DATA_DIR)


if __name__ == "__main__":
    run_and_save(seed=42)
