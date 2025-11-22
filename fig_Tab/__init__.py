"""
Utilities for generating manuscript-ready figures and tables.

This package provides small, focused entry points that operate on an
existing `results` tree (e.g. `final1/`) and the project `config.yaml`
to build:

  - Tables T1–T5 and S1–S5 as CSV files
  - Figures F1–F5 and S1–S4 as publication-grade PNG/PDF images

All heavy lifting (data loading, aggregation, conformal / DP metrics)
is delegated to the core `fedchem` package and the existing experiment
artifacts under `final1/`.
"""

from .style import PlotStyle

__all__ = ["PlotStyle"]

