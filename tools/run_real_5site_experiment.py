"""Thin wrapper to maintain backward compatibility for test imports.

Exposes `run_federated_protocol` by delegating to `scripts.run_real_site_experiment.run_federated_protocol`.
"""
from __future__ import annotations

from pathlib import Path
import importlib.util
from pathlib import Path
_spec = importlib.util.spec_from_file_location(
    "run_real_site_experiment",
    Path(__file__).resolve().parents[1] / "tools" / "run_real_site_experiment.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
run_federated_protocol = getattr(_mod, "run_federated_protocol")

__all__ = ["run_federated_protocol"]

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=Path, required=True)
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--output_dir", type=Path, default=Path("results"))
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()
    run_federated_protocol(args.data_root, k=args.k, output_dir=args.output_dir, seed=args.seed)
