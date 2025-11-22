"""
Wrapper script to run canonical experiment combos for the paper's core figures.

Usage:
  python scripts/run_for_figures.py --data_root data --out_dir results/figures --seed 42 --workers 1

This script uses `tools/run_real_site_experiment.py` to execute experiments for canonical combos
that correspond to the core figures and the tables described earlier. It runs a small set of
representative combinations per figure and saves outputs into separate subfolders under `--out_dir`.

Notes:
- The script runs sequentially by default. For parallel runs, use `--workers` > 1 (spawns subprocesses).
- The script uses CLI options for `run_real_site_experiment.py` to avoid environment variable management.

"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
import os
from typing import Dict, List
import shlex

# Canonical combos mapping
CANONICAL_COMBOS = {
    "figure3_rmsep_methods": [
        # Methods: site_specific, classical_CT_PDS, FedAvg, FedPLS_parametric
        {"label": "site_specific", "dp_eps": None, "clip_feature": None, "method": "site_specific"},
        {"label": "classical_ct_pds", "dp_eps": None, "clip_feature": None, "method": "classical_ct_pds"},
        {"label": "fedavg", "dp_eps": None, "clip_feature": None, "method": "FedAvg"},
        {"label": "fedpls_parametric", "dp_eps": None, "clip_feature": 1.0, "method": "FedPLS_parametric"},
    ],
    "figure4_conformal_coverage": [
        # 90 and 95 targets: set conformal method 'split' and target via config (split is default), call twice
        {"label": "conformal_0.90", "dp_eps": None, "conformal": "split", "conformal_targets": 0.90},
        {"label": "conformal_0.95", "dp_eps": None, "conformal": "split", "conformal_targets": 0.95},
    ],
    "figure5_robustness_drift": [
        # Spectral_Drift: none, low, moderate, high
        {"label": "drift_none", "dp_eps": 1.0, "spectral_drift": "none"},
        {"label": "drift_low", "dp_eps": 1.0, "spectral_drift": "low"},
        {"label": "drift_moderate", "dp_eps": 1.0, "spectral_drift": "moderate"},
        {"label": "drift_high", "dp_eps": 1.0, "spectral_drift": "high"},
    ],
    "figure6_privacy_tradeoffs": [
        # DP_Target_Eps: inf, 10, 1, 0.1 and Clip_Norm: 1.0, 5.0, none
    ],
}

# Build combinations for Figure 6
for eps in ["inf", 10.0, 1.0, 0.1]:
    for clip in [1.0, 5.0, "none"]:
        CANONICAL_COMBOS["figure6_privacy_tradeoffs"].append(
            {"label": f"eps_{eps}_clip_{clip}", "dp_eps": eps, "clip_feature": clip}
        )

# For each run, use run_real_site_experiment.py so we get the expected output files
RUN_SCRIPT = Path("tools") / "run_real_site_experiment.py"


def _build_cmd(data_root: Path, out_dir: Path, combo: Dict, seed: int = 42):
    cmd = [sys.executable, str(RUN_SCRIPT), "--data_root", str(data_root), "--k", "200", "--preproc", "standard", "--seed", str(seed)]
    # dp
    if combo.get("dp_eps") is not None and str(combo.get('dp_eps')).lower() != 'inf':
        cmd += ["--dp_epsilon", str(combo.get('dp_eps'))]
    # clip
    if combo.get("clip_feature") is not None:
        if str(combo.get('clip_feature')).lower() == 'none':
            # no clip
            pass
        else:
            cmd += ["--clip_feature", str(combo.get('clip_feature'))]
            # set clip target to same
            cmd += ["--clip_target", str(combo.get('clip_feature'))]
    # conformal
    if combo.get("conformal"):
        cmd += ["--conformal", combo.get("conformal")]
    # spectral drift
    if combo.get("spectral_drift"):
        cmd += ["--output_dir", str(out_dir / str(combo.get("label")))]
        # The script `run_real_site_experiment` respects env vars for drift. Set them as args via env.
    # output dir override
    if not any(p.startswith("--output_dir") for p in cmd):
        cmd += ["--output_dir", str(out_dir / str(combo.get("label")))]
    return cmd


def run_combo(data_root: Path, out_dir: Path, combo: Dict, env: Dict[str, str] | None = None) -> int:
    cmd = _build_cmd(data_root, out_dir, combo)
    # Add environment variables for drift and other factors
    env_vars = (env or {}).copy() if env else {}
    # Spectral drift mapping to DRIFT_AUGMENT_* env vars
    if combo.get("spectral_drift"):
        level = str(combo.get("spectral_drift")).lower()
        # Default augmentation defaults (mirror config defaults if not provided)
        jitter_default = 0.01
        scatter_default = 0.05
        baseline_default = 0.005
        noise_default = 0.005
        mapping = {
            'none': 0.0,
            'low': 0.25,
            'moderate': 1.0,
            'high': 2.0,
        }
        factor = mapping.get(level, 0.0)
        env_vars["FEDCHEM_DRIFT_AUGMENT_JITTER_WAVELENGTH_PX"] = str(jitter_default * factor)
        env_vars["FEDCHEM_DRIFT_AUGMENT_MULTIPLICATIVE_SCATTER"] = str(scatter_default * factor)
        env_vars["FEDCHEM_DRIFT_AUGMENT_BASELINE_OFFSET"] = str(baseline_default * factor)
        env_vars["FEDCHEM_DRIFT_AUGMENT_WHITE_NOISE_SIGMA"] = str(noise_default * factor)
        env_vars["FEDCHEM_DRIFT_AUGMENT_AUGMENTATION_SEED"] = str(99)
        env_vars["FEDCHEM_DRIFT_AUGMENT_APPLY_AUGMENTATION_DURING_TRAINING"] = '1' if factor > 0 else '0'
        env_vars["FEDCHEM_DRIFT_AUGMENT_APPLY_TEST_SHIFTS"] = '1' if factor > 0 else '0'
    # Set conformal targets via env var if provided
    if combo.get("conformal_targets") is not None:
        env_vars["FEDCHEM_CONFORMAL_TARGETS"] = str(combo.get("conformal_targets"))
    # Set federated method if specified
    if combo.get("method"):
        env_vars["FEDCHEM_FEDPLS_METHOD"] = str(combo.get("method"))
    print(f"Running combo: {combo['label']}")
    print(" ", " ".join(shlex.quote(str(x)) for x in cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, env={**env_vars, **os.environ})
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
    return result.returncode


def main(argv=None):
    p = argparse.ArgumentParser(description="Run representative combos for figures and tables")
    p.add_argument("--data_root", type=Path, default=Path("data"))
    p.add_argument("--out_dir", type=Path, default=Path("results/figures"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=1, help="Number of parallel runs to start; 1 means sequential")
    args = p.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # This wrapper is intentionally simple: it runs each combo sequentially by default
    for key in ["figure3_rmsep_methods", "figure4_conformal_coverage", "figure5_robustness_drift", "figure6_privacy_tradeoffs"]:
        combos = CANONICAL_COMBOS.get(key, [])
        print(f"Starting runs for {key}, {len(combos)} combo(s)")
        for combo in combos:
            # create a subdir per figure + combo label
            figure_dir = args.out_dir / key
            figure_dir.mkdir(parents=True, exist_ok=True)
            combo_env = {}
            # For convenience, also set FEDCHEM_EXPERIMENT_LABEL in env for verbosity
            combo_env["FEDCHEM_EXPERIMENT_LABEL"] = f"{key}_{combo['label']}"
            ret = run_combo(args.data_root, figure_dir, combo, env=combo_env)
            if ret != 0:
                print(f"Warning: run failed for combo: {combo['label']}")
    print("All figure combos finished.")


if __name__ == "__main__":
    main()
