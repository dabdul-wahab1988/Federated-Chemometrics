#!/usr/bin/env python3
"""Aggregate raw results, build tables, and render the manuscript figures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
import seaborn as sns

# If the project package (`fedchem`) isn't installed in the environment, allow running
# scripts directly from a checkout by ensuring `src/` is on `sys.path`.
try:
    # Try a lightweight import to let normal execution proceed when installed
    import fedchem  # noqa: E402
except Exception:
    import sys
    _proj_root = Path(__file__).resolve().parents[1]
    _src_dir = _proj_root / "src"
    if _src_dir.exists():
        # Prepend to ensure we take local checkout over any installed package
        sys.path.insert(0, str(_src_dir))

# Always add the project root to sys.path so scripts can import each other
_proj_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_proj_root))

from fedchem.results import ensure_results_tree
from fedchem.results.aggregator import (
    build_methods_summary,
    export_federated_results,
    export_methods_summary,
    load_raw_records,
    summarize_metrics,
)
from fedchem.utils.config import load_config, get_instruments_from_config

import scripts.generate_core_tables as core_tables


def _load_cfg(config_path: Path) -> dict:
    return load_config(config_path)


def _ensure_fig_dir(tree, name: str) -> str:
    path = tree.figures / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def _figure1_conceptual(cfg: dict, tree) -> None:
    """Simple conceptual diagram for Figure 1."""

    sites = get_instruments_from_config(cfg) or cfg.get("EXPERIMENTAL_DESIGN", {}).get("FACTORS", {}).get("Site", [])
    sites = list(sites)[:6] if sites else ["MA_A1", "MA_A2", "MA_A3", "MB_B1", "MB_B2", "MB_B3"]
    left_x = 0
    right_x = 6
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")
    ax.set_xlim(-1, 7)
    ax.set_ylim(-1, len(sites) + 1)

    for idx, site in enumerate(sites):
        y = len(sites) - idx
        ax.scatter(left_x, y, s=500, c="#1f77b4")
        ax.text(left_x, y, site, color="white", ha="center", va="center", fontsize=10, weight="bold")
        ax.annotate(
            "",
            xy=(right_x - 1, y),
            xytext=(left_x + 0.5, y),
            arrowprops=dict(arrowstyle="->", lw=1.5, color="#1f77b4"),
        )

    ax.scatter(right_x, len(sites) / 2, s=700, c="#ff7f0e")
    ax.text(right_x, len(sites) / 2, "Central Server", ha="center", va="center", color="white", fontsize=12, weight="bold")
    ax.text(1.5, len(sites) + 0.2, "Classical CT", fontsize=10, weight="bold")
    ax.text(3.5, len(sites) + 0.2, "Federated FL + DP + CP", fontsize=10, weight="bold")

    fig.tight_layout()
    fig.savefig(_ensure_fig_dir(tree, "figure_1_conceptual.png"), dpi=200)
    plt.close(fig)

def _figure2_pipeline(tree) -> None:
    """Flow-style pipeline diagram for Figure 2."""

    steps = [
        "Experimental design\n(fractional factorial)",
        "For each design point:\nrun federated rounds",
        "Apply DP + secure aggregation\n(if enabled)",
        "Conformal calibration\n(per-site / pooled)",
        "Evaluation per\nholdout strategy",
    ]
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis("off")
    ax.set_xlim(0, len(steps) * 2)
    ax.set_ylim(0, 2)
    for idx, text in enumerate(steps):
        x = idx * 2
        box = Rectangle((x, 0.5), 1.8, 1, facecolor="#e0e0e0", edgecolor="#555555")
        ax.add_patch(box)
        ax.text(x + 0.9, 1, text, ha="center", va="center", fontsize=10)
        if idx < len(steps) - 1:
            ax.annotate(
                "",
                xy=(x + 1.8, 1),
                xytext=(x + 2.1, 1),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="#555555"),
            )
    fig.tight_layout()
    fig.savefig(_ensure_fig_dir(tree, "figure_2_pipeline.png"), dpi=200)
    plt.close(fig)
    plt.close(fig)


def _plot_rmsep_by_method(summary_df: pd.DataFrame, tree) -> None:
    df = summary_df.dropna(subset=["rmsep"]).copy()
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=df, x="method", y="rmsep", hue="site_id", ax=ax)
    ax.set_ylabel("RMSEP")
    ax.set_xlabel("Method")
    ax.set_title("Figure 3 – RMSEP across held-out coded instruments")
    fig.tight_layout()
    fig.savefig(_ensure_fig_dir(tree, "figure_3_rmsep.png"), dpi=200)
    plt.close(fig)


def _plot_conformal_coverage(summary_df: pd.DataFrame, tree) -> None:
    df = summary_df.dropna(subset=["coverage"]).copy()
    if df.empty:
        return
    df["target"] = 0.90
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.scatterplot(data=df, x="target", y="coverage", hue="method", style="site_id", ax=ax)
    ax.plot([0, 1], [0, 1], ls="--", color="grey")
    ax.set_xlim(0.85, 0.95)
    ax.set_ylim(0.85, 0.98)
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("Figure 4 – Conformal coverage vs nominal")
    fig.tight_layout()
    fig.savefig(_ensure_fig_dir(tree, "figure_4_conformal.png"), dpi=200)
    plt.close(fig)


def _plot_spectral_drift(summary_df: pd.DataFrame, tree) -> None:
    if "spectral_drift" not in summary_df.columns:
        return
    df = summary_df.dropna(subset=["spectral_drift", "rmsep"]).copy()
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=df, x="spectral_drift", y="rmsep", hue="method", ax=ax, marker="o")
    ax.set_title("Figure 5 – RMSEP vs spectral drift level")
    ax.set_ylabel("RMSEP")
    ax.set_xlabel("Spectral drift level")
    fig.tight_layout()
    fig.savefig(_ensure_fig_dir(tree, "figure_5_drift.png"), dpi=200)
    plt.close(fig)


def _plot_privacy_tradeoff(summary_df: pd.DataFrame, tree) -> None:
    eps_col = "dp_target_eps" if "dp_target_eps" in summary_df.columns else "design_dp_target_eps"
    if eps_col not in summary_df.columns:
        return
    df = summary_df.dropna(subset=[eps_col, "rmsep"]).copy()
    if df.empty:
        return
    df["epsilon"] = df[eps_col].astype(str)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=df, x="epsilon", y="rmsep", hue="method", marker="o", ax=ax)
    ax.set_title("Figure 6 – Privacy / utility trade-offs")
    ax.set_ylabel("RMSEP")
    ax.set_xlabel("Target ε")
    fig.tight_layout()
    fig.savefig(_ensure_fig_dir(tree, "figure_6_privacy.png"), dpi=200)
    plt.close(fig)


def _build_tables(config_path: Path, result_dirs: Sequence[Path], tree, data_dir: Path) -> None:
    if not result_dirs:
        return
    ns = SimpleNamespace(
        config=str(config_path),
        results_dirs=",".join(str(p) for p in result_dirs),
        output_dir=str(tree.tables),
        data_dir=str(data_dir),
        held_out_instruments=None,
    )
    core_tables._main(ns)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Aggregate raw results into tables and figures.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="Path to config.yaml")
    parser.add_argument("--results-dir", type=Path, default=Path("results"), help="Root results directory (default: ./results)")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Data directory used for Table 1 instrumentation counts")
    parser.add_argument('--with-figures', action='store_true', help='Generate figures (call fig_1)')
    parser.add_argument('--only-figures', action='store_true', help='Skip tables and summary generation; only generate figures')
    parser.add_argument('--figure-output-dir', type=Path, default=None, help='Directory to write figures (default: <results>/figures)')
    parser.add_argument('--figure-format', type=str, default='png', choices=['png', 'pdf'], help='Figure output format')
    parser.add_argument('--figure-dpi', type=int, default=180, help='Figure output DPI')
    parser.add_argument('--figure-plot-types', nargs='+', default=['heatmap'], help='Types of figures to generate via fig_1')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    from importlib import import_module
    tree = ensure_results_tree(args.results_dir)
    cfg = _load_cfg(args.config)
    df_raw = load_raw_records(tree)
    if df_raw.empty:
        print("No raw run records available under results/raw_runs. Run experiments first.")
        return

    raw_csv = tree.aggregated / "raw_records.csv"
    df_raw.to_csv(raw_csv, index=False)

    summary_df = build_methods_summary(df_raw)
    summary_csv = tree.aggregated / "methods_summary_all.csv"
    summary_df.to_csv(summary_csv, index=False)

    run_dirs = export_methods_summary(summary_df, tree)
    export_federated_results(summary_df, tree)
    metrics_df = summarize_metrics(summary_df)
    metrics_df.to_csv(tree.aggregated / "metrics_summary.csv", index=False)

    _build_tables(args.config, list(run_dirs.values()), tree, args.data_dir)

    sns.set_theme(style="whitegrid")
    _figure1_conceptual(cfg, tree)
    _figure2_pipeline(tree)
    _plot_rmsep_by_method(summary_df, tree)
    _plot_conformal_coverage(summary_df, tree)
    _plot_spectral_drift(summary_df, tree)
    _plot_privacy_tradeoff(summary_df, tree)
    print(f"Artifacts written to {tree.base}")
    # Optionally generate figures via new fig_1 script / module
    if args.with_figures or args.only_figures:
        try:
            fig_mod = import_module('scripts.fig_1')
        except Exception:
            print('Unable to import scripts.fig_1 for figure generation; ensure scripts are on sys.path')
            return
        # Build figure args
        fig_kwargs = {
            'format': args.figure_format,
            'dpi': args.figure_dpi,
            'output_dir': str(args.figure_output_dir or (tree.base / 'figures')),
            'plot_types': args.figure_plot_types,
            'manifest_root': str(tree.base),
            'config': str(args.config),
        }
        # Call the fig_1 module as a runner by simulating invocation
        try:
            # prefer to call the module main if it has argument parsing
            if hasattr(fig_mod, 'main'):
                # build argv for fig_1 module
                fig_argv = [
                    '--manifest-root', str(tree.base),
                    '--output-dir', str(args.figure_output_dir or (tree.base / 'figures')),
                    '--format', args.figure_format,
                    '--dpi', str(args.figure_dpi),
                ]
                # append plot types
                if args.figure_plot_types:
                    fig_argv += ['--plot-types'] + args.figure_plot_types
                fig_mod.main(fig_argv)
            else:
                # fallback: run default behavior (will use defaults)
                pass
        except Exception as e:
            print('Failed to generate figures using fig_1:', e)


if __name__ == "__main__":
    main()
