"""
Generate a reproducible 3-site paired transfer dataset and run PDS experiments A and B.

Outputs:
- data/3site_design/: per-site X.npy, y.npy, meta.json
- generated_figures_tables/figure_3.png and meta (via generate_objective_3)
- generated_figures_tables/3site_pds_diagnostics.csv (per-site diagnostics for A and B)
- generated_figures_tables/3site_manifest.json

Run: python tools/generate_3site_design.py

This script uses the project's SpectralSimulator and existing objective scripts to
produce datasets compatible with the repo's benchmarking functions.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "3site_design"