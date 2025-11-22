"""Prepare a `real_5site` layout from simple input files (IDRC-like or CSV).

This is a lightweight helper. It accepts a folder with per-site CSVs or a single
IDRC-style directory and writes the standardized layout used by the experiments.
"""
from pathlib import Path
import numpy as np
import argparse
import json


def prepare_from_numpy(input_dir: Path, output_dir: Path, site_names=None, k=200):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if site_names is None:
        site_names = sorted([p.name for p in Path(input_dir).iterdir() if p.is_dir()])
    for site in site_names:
        src = Path(input_dir) / site
        dst = output_dir / site
        dst.mkdir(parents=True, exist_ok=True)
        # expect files X.npy, y.npy optionally
        X = np.load(src / "X.npy")
        y = np.load(src / "y.npy")
        # paired is first k samples, local is remaining
        k_loc = min(k, X.shape[0])
        np.save(dst / "X_paired.npy", X[:k_loc])
        np.save(dst / "y_paired.npy", y[:k_loc])
        if X.shape[0] > k_loc:
            np.save(dst / "X_local.npy", X[k_loc:])
            np.save(dst / "y_local.npy", y[k_loc:])
        else:
            # duplicate small local set
            np.save(dst / "X_local.npy", X[:max(1, min(20, X.shape[0]))])
            np.save(dst / "y_local.npy", y[:max(1, min(20, X.shape[0]))])

    # manifest
    manifest = {"site_names": site_names, "k": k}
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


def _cli():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--k", type=int, default=200)
    args = p.parse_args()
    prepare_from_numpy(args.input_dir, args.output_dir, k=args.k)


if __name__ == "__main__":
    _cli()
