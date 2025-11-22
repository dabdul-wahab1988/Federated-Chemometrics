#!/usr/bin/env python
"""
Grid versions of FedChem manuscript figures, with rows = transfer_k levels.

Assumes the presence of:
  - master_database_20251120_200830.json

Each figure:

  Figure 2 grid: 3 x 2 panels
    Rows       -> transfer_k in [20, 80, 200]
    Col 0 (A*) -> RMSEP by method & instrument with Centralized_PLS baselines
    Col 1 (B*) -> ΔRMSEP vs Centralized_PLS per instrument

  Figure 3 grid: 3 x 3 panels
    Rows       -> transfer_k in [20, 80, 200]
    Col 0      -> Federated methods (FedPLS, FedProx, Global_calibrate_after_fed), 
                  RMSEP averaged over instruments
    Col 1      -> FedPLS RMSEP per instrument (MA_A2, MB_B2)
    Col 2      -> Classical CT baselines (PDS, SBC), RMSEP averaged over instruments

  Figure 4 grid: 3 x 3 panels (privacy–utility)
    Rows       -> transfer_k in [20, 80, 200]
    Col 0      -> RMSEP vs epsilon for given method
    Col 1      -> Total bytes & wall time vs epsilon (dual-axis)
    Col 2      -> RMSEP vs total_bytes scatter, points annotated by epsilon

  Figure 5 grid: 3 x 3 panels (conformal)
    Rows       -> transfer_k in [20, 80, 200]
    Col 0      -> Global_Coverage vs epsilon, with nominal line and over-coverage shading
    Col 1      -> Global_MeanWidth vs epsilon
    Col 2      -> Global_MeanWidth vs RMSEP, annotated by epsilon

Usage:
    python improved_results_plots_grid.py \
        --db master_database_20251120_200830.json \
        --outdir figs_grid \
        --method FedPLS

"""

from pathlib import Path
import json
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def add_dp_helpers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add:
      - dp_label: 'inf' for non-DP (NaN), otherwise '0.1', '1', '10', ...
      - dp_numeric: np.inf for non-DP, numeric epsilon otherwise.
    """
    df = df.copy()

    def _label_eps(x):
        if pd.isna(x):
            return "inf"
        xf = float(x)
        return str(int(xf)) if xf.is_integer() else str(xf)

    df["dp_label"] = df["dp_epsilon"].apply(_label_eps)
    df["dp_numeric"] = df["dp_epsilon"].fillna(np.inf)
    return df


# ----------------------------------------------------------------------
# Figure 2 grid – Baseline vs federated, multi transfer_k
# ----------------------------------------------------------------------

def plot_fig2_baseline_vs_federated_grid(
    df: pd.DataFrame,
    transfer_ks=(20, 80, 200),
    save_path: Path | None = None,
) -> None:
    """
    Figure 2 grid:
      Rows = transfer_k in transfer_ks
      Col 0: RMSEP by method & instrument, with Centralized_PLS baselines
      Col 1: ΔRMSEP vs Centralized_PLS per instrument
    """
    df = add_dp_helpers(df)
    non_dp = df[df["dp_label"] == "inf"].copy()

    method_order = [
        "Site_Specific",
        "PDS",
        "SBC",
        "Centralized_PLS",
        "FedPLS",
        "FedProx",
        "Global_calibrate_after_fed",
    ]
    instrument_order = ["MA_A2", "MB_B2"]

    nrows = len(transfer_ks)
    ncols = 2
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(6 * ncols, 3 * nrows), sharex="col"
    )
    if nrows == 1:
        axes = np.array([axes])

    for i, k in enumerate(transfer_ks):
        sub = non_dp[
            (non_dp["transfer_k"] == k) & (non_dp["method"].isin(method_order))
        ].copy()
        if sub.empty:
            continue

        agg = (
            sub.groupby(["method", "instrument_code"])
            .agg(rmsep_mean=("rmsep", "mean"), rmsep_std=("rmsep", "std"))
            .reset_index()
        )

        agg["method"] = pd.Categorical(
            agg["method"], categories=method_order, ordered=True
        )
        agg["instrument_code"] = pd.Categorical(
            agg["instrument_code"], categories=instrument_order, ordered=True
        )
        agg = agg.sort_values(["method", "instrument_code"])

        x = np.arange(len(method_order))
        width = 0.35

        axA = axes[i, 0]
        axB = axes[i, 1]

        # Panel A – RMSEP
        vals = {inst: [] for inst in instrument_order}
        errs = {inst: [] for inst in instrument_order}
        for m in method_order:
            for inst in instrument_order:
                row = agg[
                    (agg["method"] == m)
                    & (agg["instrument_code"] == inst)
                ]
                if len(row) == 1:
                    mean = float(row["rmsep_mean"])
                    std = float(row["rmsep_std"])
                    vals[inst].append(mean)
                    errs[inst].append(0.0 if np.isnan(std) else std)
                else:
                    vals[inst].append(np.nan)
                    errs[inst].append(0.0)

        axA.bar(
            x - width / 2,
            vals["MA_A2"],
            width,
            yerr=errs["MA_A2"],
            label="MA_A2",
        )
        axA.bar(
            x + width / 2,
            vals["MB_B2"],
            width,
            yerr=errs["MB_B2"],
            label="MB_B2",
        )

        # Centralized_PLS baselines (one per instrument)
        for inst, color in zip(instrument_order, ["tab:blue", "tab:orange"]):
            row = agg[
                (agg["method"] == "Centralized_PLS")
                & (agg["instrument_code"] == inst)
            ]
            if len(row) == 1:
                baseline = float(row["rmsep_mean"])
                axA.axhline(
                    baseline,
                    linestyle="--",
                    linewidth=1,
                    color=color,
                    alpha=0.7,
                )

        if i == 0:
            axA.set_title("(A) RMSEP by method & instrument")
            handles, labels = axA.get_legend_handles_labels()
            uniq = dict(zip(labels, handles))
            axA.legend(uniq.values(), uniq.keys(), fontsize=8)
        axA.set_ylabel(f"RMSEP (k={k})")
        axA.grid(True, axis="y", linestyle="--", alpha=0.4)

        if i == nrows - 1:
            axA.set_xticks(x)
            axA.set_xticklabels(method_order, rotation=40, ha="right")
        else:
            axA.set_xticks(x)
            axA.set_xticklabels([])

        # Panel B – ΔRMSEP vs Centralized_PLS
        baseline_dict: dict[str, float] = {}
        for inst in instrument_order:
            row = agg[
                (agg["method"] == "Centralized_PLS")
                & (agg["instrument_code"] == inst)
            ]
            baseline_dict[inst] = float(row["rmsep_mean"]) if len(row) == 1 else np.nan

        delta_vals = {inst: [] for inst in instrument_order}
        for m in method_order:
            for inst in instrument_order:
                row = agg[
                    (agg["method"] == m)
                    & (agg["instrument_code"] == inst)
                ]
                if len(row) == 1:
                    delta_vals[inst].append(
                        float(row["rmsep_mean"]) - baseline_dict[inst]
                    )
                else:
                    delta_vals[inst].append(np.nan)

        axB.axhline(0.0, color="k", linestyle="--", linewidth=1)
        axB.plot(
            x - width / 2,
            delta_vals["MA_A2"],
            marker="o",
            linestyle="-",
            label="MA_A2",
        )
        axB.plot(
            x + width / 2,
            delta_vals["MB_B2"],
            marker="s",
            linestyle="-",
            label="MB_B2",
        )
        if i == 0:
            axB.set_title("(B) ΔRMSEP vs Centralized_PLS")
            axB.legend(fontsize=8)
        axB.set_ylabel(f"ΔRMSEP (k={k})")
        axB.grid(True, axis="y", linestyle="--", alpha=0.4)

        if i == nrows - 1:
            axB.set_xticks(x)
            axB.set_xticklabels(method_order, rotation=40, ha="right")
        else:
            axB.set_xticks(x)
            axB.set_xticklabels([])

    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# ----------------------------------------------------------------------
# Figure 3 grid – Transfer budget, multi transfer_k
# ----------------------------------------------------------------------

def plot_fig3_transfer_budget_grid(
    df: pd.DataFrame,
    transfer_ks=(20, 80, 200),
    save_path: Path | None = None,
) -> None:
    """
    Figure 3 grid (reframed):

      Rows = transfer_k in transfer_ks
      Col 0: Federated methods (FedPLS, FedProx, Global_calibrate_after_fed),
             RMSEP averaged over instruments
      Col 1: FedPLS RMSEP per instrument
      Col 2: Classical CT baselines (PDS, SBC), RMSEP averaged over instruments

    This complements the original "RMSEP vs transfer_k" view by showing,
    for each transfer budget level, the method-by-method pattern.
    """
    df = df.copy()
    non_dp = df[df["dp_epsilon"].isna()].copy()

    fed_methods = ["FedPLS", "FedProx", "Global_calibrate_after_fed"]
    ct_methods = ["PDS", "SBC"]

    nrows = len(transfer_ks)
    ncols = 3
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 3 * nrows), sharey=True
    )
    if nrows == 1:
        axes = np.array([axes])

    for i, k in enumerate(transfer_ks):
        sub = non_dp[non_dp["transfer_k"] == k].copy()
        if sub.empty:
            continue

        # Col 0 – federated methods (avg over instruments)
        ax = axes[i, 0]
        sub_fed = sub[sub["method"].isin(fed_methods)]
        if not sub_fed.empty:
            agg_fed = (
                sub_fed.groupby("method")
                .agg(rmsep_mean=("rmsep", "mean"), rmsep_std=("rmsep", "std"))
                .reset_index()
            )
            methods_here = [
                m
                for m in fed_methods
                if m in agg_fed["method"].values
            ]
            x = np.arange(len(methods_here))
            means = [
                float(
                    agg_fed.loc[agg_fed["method"] == m, "rmsep_mean"].iloc[0]
                )
                for m in methods_here
            ]
            stds = [
                float(
                    agg_fed.loc[agg_fed["method"] == m, "rmsep_std"].iloc[0]
                )
                for m in methods_here
            ]
            stds = [0.0 if np.isnan(s) else s for s in stds]
            bars = ax.bar(x, means, yerr=stds)
            # Add value labels on bars
            for bar, mean in zip(bars, means):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    f'{mean:.3f}',
                    ha='center',
                    va='bottom',
                    fontsize=12
                )
            if i == 0:
                ax.set_title("Federated methods\n(avg over instruments)")
            ax.set_ylabel(f"RMSEP (k={k})")
            if i == nrows - 1:
                ax.set_xticks(x)
                ax.set_xticklabels(methods_here, rotation=30, ha="right")
            else:
                ax.set_xticks(x)
                ax.set_xticklabels([])
            ax.grid(True, axis="y", linestyle="--", alpha=0.4)

        # Col 1 – FedPLS by instrument
        ax = axes[i, 1]
        sub_fedpls = sub[sub["method"] == "FedPLS"]
        if not sub_fedpls.empty:
            agg_fedpls = (
                sub_fedpls.groupby("instrument_code")
                .agg(
                    rmsep_mean=("rmsep", "mean"),
                    rmsep_std=("rmsep", "std"),
                )
                .reset_index()
            )
            insts = list(agg_fedpls["instrument_code"])
            x = np.arange(len(insts))
            means = [
                float(
                    agg_fedpls.loc[
                        agg_fedpls["instrument_code"] == inst, "rmsep_mean"
                    ].iloc[0]
                )
                for inst in insts
            ]
            stds = [
                float(
                    agg_fedpls.loc[
                        agg_fedpls["instrument_code"] == inst, "rmsep_std"
                    ].iloc[0]
                )
                for inst in insts
            ]
            stds = [0.0 if np.isnan(s) else s for s in stds]
            bars = ax.bar(x, means, yerr=stds)
            # Add value labels on bars
            for bar, mean in zip(bars, means):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    f'{mean:.3f}',
                    ha='center',
                    va='bottom',
                    fontsize=12
                )
            if i == 0:
                ax.set_title("FedPLS by instrument")
            if i == nrows - 1:
                ax.set_xticks(x)
                ax.set_xticklabels(insts)
            else:
                ax.set_xticks(x)
                ax.set_xticklabels([])
            ax.grid(True, axis="y", linestyle="--", alpha=0.4)

        # Col 2 – CT baselines (PDS, SBC)
        ax = axes[i, 2]
        sub_ct = sub[sub["method"].isin(ct_methods)]
        if not sub_ct.empty:
            agg_ct = (
                sub_ct.groupby("method")
                .agg(rmsep_mean=("rmsep", "mean"), rmsep_std=("rmsep", "std"))
                .reset_index()
            )
            methods_here = [
                m
                for m in ct_methods
                if m in agg_ct["method"].values
            ]
            x = np.arange(len(methods_here))
            means = [
                float(
                    agg_ct.loc[agg_ct["method"] == m, "rmsep_mean"].iloc[0]
                )
                for m in methods_here
            ]
            stds = [
                float(
                    agg_ct.loc[agg_ct["method"] == m, "rmsep_std"].iloc[0]
                )
                for m in methods_here
            ]
            stds = [0.0 if np.isnan(s) else s for s in stds]
            bars = ax.bar(x, means, yerr=stds)
            # Add value labels on bars
            for bar, mean in zip(bars, means):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    f'{mean:.3f}',
                    ha='center',
                    va='bottom',
                    fontsize=12
                )
            if i == 0:
                ax.set_title("Classical CT (PDS/SBC)\n(avg over instruments)")
            if i == nrows - 1:
                ax.set_xticks(x)
                ax.set_xticklabels(methods_here, rotation=30, ha="right")
            else:
                ax.set_xticks(x)
                ax.set_xticklabels([])
            ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# ----------------------------------------------------------------------
# Figure 4 grid – Privacy–utility, multi transfer_k
# ----------------------------------------------------------------------

def plot_fig4_privacy_utility_grid(
    df: pd.DataFrame,
    method: str = "FedPLS",
    transfer_ks=(20, 80, 200),
    save_path: Path | None = None,
) -> None:
    """
    Figure 4 grid:

      Rows = transfer_k in transfer_ks
      Col 0: RMSEP vs epsilon (per row, per k)
      Col 1: Total bytes & wall time vs epsilon (dual-axis, legend on top row)
      Col 2: RMSEP vs total_bytes, points annotated by epsilon label
    """
    df = add_dp_helpers(df)
    nrows = len(transfer_ks)
    ncols = 3
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 3 * nrows), sharex="col"
    )
    if nrows == 1:
        axes = np.array([axes])

    for i, k in enumerate(transfer_ks):
        sub = df[(df["method"] == method) & (df["transfer_k"] == k)].copy()
        if sub.empty:
            continue

        agg = (
            sub.groupby("dp_label")
            .agg(
                dp_numeric=("dp_numeric", "mean"),
                rmsep_mean=("rmsep", "mean"),
                rmsep_std=("rmsep", "std"),
                coverage_mean=("Global_Coverage", "mean"),
                coverage_std=("Global_Coverage", "std"),
                width_mean=("Global_MeanWidth", "mean"),
                width_std=("Global_MeanWidth", "std"),
                bytes_mean=("total_bytes", "mean"),
                bytes_std=("total_bytes", "std"),
                time_mean=("wall_time_sec", "mean"),
                time_std=("wall_time_sec", "std"),
            )
            .reset_index()
        )
        agg = agg.sort_values("dp_numeric")
        eps_labels = list(agg["dp_label"])
        x_pos = np.arange(len(eps_labels))

        # Col 0 – RMSEP vs epsilon
        axA = axes[i, 0]
        axA.errorbar(
            x_pos,
            agg["rmsep_mean"],
            yerr=agg["rmsep_std"].fillna(0.0),
            marker="o",
            linestyle="-",
        )
        if i == nrows - 1:
            axA.set_xticks(x_pos)
            axA.set_xticklabels(eps_labels)
            axA.set_xlabel("DP epsilon (ε)")
        else:
            axA.set_xticks(x_pos)
            axA.set_xticklabels([])
        axA.set_ylabel(f"RMSEP (k={k})")
        if i == 0:
            axA.set_title(f"(A) RMSEP vs ε ({method})")
        axA.grid(True, linestyle="--", alpha=0.4)

        diff = agg["rmsep_mean"].max() - agg["rmsep_mean"].min()
        if diff < 0.001:
            axA.text(
                0.05,
                0.05,
                "no visible DP effect at 0.1%",
                transform=axA.transAxes,
                fontsize=10,
                ha="left",
                va="bottom",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
            )

        # Col 1 – communication vs epsilon (dual axis)
        ax1 = axes[i, 1]
        ax2 = ax1.twinx()
        # Convert bytes to MB
        bytes_mb_mean = agg["bytes_mean"] / (1024 * 1024)
        bytes_mb_std = agg["bytes_std"].fillna(0.0) / (1024 * 1024)
        ax1.errorbar(
            x_pos,
            bytes_mb_mean,
            yerr=bytes_mb_std,
            marker="o",
            linestyle="-",
            label="Total bytes (MB)",
        )
        ax2.errorbar(
            x_pos,
            agg["time_mean"],
            yerr=agg["time_std"].fillna(0.0),
            marker="s",
            linestyle="--",
            label="Wall time (s)",
        )
        if i == nrows - 1:
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(eps_labels)
            ax1.set_xlabel("DP epsilon (ε)")
        else:
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels([])

        ax1.set_ylabel("Total bytes (MB)", color="tab:blue")
        ax2.set_ylabel("Wall time (s)", color="tab:orange")
        ax2.spines["right"].set_position(("axes", 1.08))
        ax1.grid(True, linestyle="--", alpha=0.4)
        if i == 0:
            ax1.set_title(f"(B) Communication vs ε ({method})")
            h1, l1 = ax1.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax1.legend(h1 + h2, l1 + l2, fontsize=7, loc="best")

        # Col 2 – RMSEP vs total_bytes scatter
        axC = axes[i, 2]
        
        # Define marker styles for epsilon values (same as Fig 5)
        eps_markers = {
            "inf": "o",
            "0.1": "s",
            "1": "^",
            "10": "D",
            "100": "v",
        }
        default_marker = "p"
        
        # Convert bytes to MB
        agg["bytes_mb"] = agg["bytes_mean"] / (1024 * 1024)
        
        # Plot with markers for epsilon (single color since all same method)
        eps_handles = {}
        for _, row in agg.iterrows():
            eps_lbl = row["dp_label"]
            marker = eps_markers.get(eps_lbl, default_marker)
            sc = axC.scatter(
                row["rmsep_mean"],
                row["bytes_mb"],
                s=60,
                marker=marker,
                color="tab:blue",
            )
            if eps_lbl not in eps_handles:
                eps_handles[eps_lbl] = axC.scatter(
                    [], [], s=60, marker=marker, color="tab:blue", label=f"ε={eps_lbl}"
                )
        
        if i == nrows - 1:
            axC.set_xlabel("RMSEP on test set")
        axC.set_ylabel("Total bytes (MB)" if i == 1 else f"Total bytes (MB, k={k})")
        
        if i == 0:
            axC.legend(
                handles=list(eps_handles.values()),
                labels=[f"ε={lbl}" for lbl in eps_handles.keys()],
                fontsize=7,
                loc="best",
                title="Epsilon"
            )
        if i == 0:
            axC.set_title(f"(C) Privacy–utility points ({method})")
        else:
            # Add legend to other rows too
            axC.legend(
                handles=list(eps_handles.values()),
                labels=[f"ε={lbl}" for lbl in eps_handles.keys()],
                fontsize=7,
                loc="best",
                title="Epsilon"
            )
        axC.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# ----------------------------------------------------------------------
# Figure 5 grid – Conformal coverage & width, multi transfer_k
# ----------------------------------------------------------------------

def plot_fig5_conformal_grid(
    df: pd.DataFrame,
    transfer_ks=(20, 80, 200),
    methods=None,
    save_path: Path | None = None,
) -> None:
    """
    Figure 5 grid:

      Rows = transfer_k in transfer_ks
      Col 0: Global_Coverage vs epsilon, with nominal=0.90 line + over-coverage shading
      Col 1: Mondrian_Coverage vs epsilon (group-wise coverage per instrument)
      Col 2: Global_MeanWidth vs epsilon
      Col 3: Global_MeanWidth vs RMSEP, with markers for epsilon
    """
    df = add_dp_helpers(df)
    if methods is None:
        methods = [
            "Centralized_PLS",
            "Site_Specific",
            "FedPLS",
            "FedProx",
            "Global_calibrate_after_fed",
        ]

    nrows = len(transfer_ks)
    ncols = 4
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 3 * nrows), sharex="col"
    )
    if nrows == 1:
        axes = np.array([axes])

    for i, k in enumerate(transfer_ks):
        sub = df[df["transfer_k"] == k].copy()
        sub = sub[sub["method"].isin(methods)]
        if sub.empty:
            continue

        nominal_vals = sub["Nominal"].dropna().unique()
        nominal = float(nominal_vals[0]) if len(nominal_vals) else None

        agg = (
            sub.groupby(["method", "dp_label"])
            .agg(
                dp_numeric=("dp_numeric", "mean"),
                coverage_mean=("Global_Coverage", "mean"),
                coverage_std=("Global_Coverage", "std"),
                mondrian_coverage_mean=("Mondrian_Coverage", "mean"),
                mondrian_coverage_std=("Mondrian_Coverage", "std"),
                width_mean=("Global_MeanWidth", "mean"),
                width_std=("Global_MeanWidth", "std"),
                rmsep_mean=("rmsep", "mean"),
                rmsep_std=("rmsep", "std"),
            )
            .reset_index()
        )
        agg = agg.sort_values(["method", "dp_numeric"])

        eps_numeric = (
            agg[["dp_numeric", "dp_label"]]
            .drop_duplicates()
            .sort_values("dp_numeric")
        )
        eps_labels = list(eps_numeric["dp_label"])
        xmap = {lbl: j for j, lbl in enumerate(eps_labels)}
        xs = list(range(len(eps_labels)))

        # Col 0 – coverage vs epsilon
        axA = axes[i, 0]
        for m in methods:
            subm = agg[agg["method"] == m]
            if subm.empty:
                continue
            x = [xmap[lbl] for lbl in subm["dp_label"]]
            axA.errorbar(
                x,
                subm["coverage_mean"],
                yerr=subm["coverage_std"].fillna(0.0),
                marker="o",
                linestyle="-",
                label=m,
            )
        if i == nrows - 1:
            axA.set_xticks(xs)
            axA.set_xticklabels(eps_labels)
            axA.set_xlabel("DP epsilon (ε)")
        else:
            axA.set_xticks(xs)
            axA.set_xticklabels([])
        axA.set_ylabel(f"Coverage (k={k})")
        if nominal is not None:
            axA.axhline(
                nominal,
                color="k",
                linestyle="--",
                linewidth=1,
                label=f"Nominal={nominal:.2f}",
            )
            axA.axhspan(nominal, 1.05, color="grey", alpha=0.1)
        axA.set_ylim(0.0, 1.05)
        if i == 0:
            axA.set_title("(A) Global coverage vs ε")
            axA.legend(fontsize=7)
        axA.grid(True, linestyle="--", alpha=0.4)

        # Col 1 – Mondrian coverage vs epsilon
        axB = axes[i, 1]
        for m in methods:
            subm = agg[agg["method"] == m]
            if subm.empty:
                continue
            x = [xmap[lbl] for lbl in subm["dp_label"]]
            axB.errorbar(
                x,
                subm["mondrian_coverage_mean"],
                yerr=subm["mondrian_coverage_std"].fillna(0.0),
                marker="s",
                linestyle="-",
                label=m,
            )
        if i == nrows - 1:
            axB.set_xticks(xs)
            axB.set_xticklabels(eps_labels)
            axB.set_xlabel("DP epsilon (ε)")
        else:
            axB.set_xticks(xs)
            axB.set_xticklabels([])
        axB.set_ylabel(f"Mondrian coverage (k={k})")
        if nominal is not None:
            axB.axhline(
                nominal,
                color="k",
                linestyle="--",
                linewidth=1,
                label=f"Nominal={nominal:.2f}",
            )
            axB.axhspan(nominal, 1.05, color="grey", alpha=0.1)
        axB.set_ylim(0.0, 1.05)
        if i == 0:
            axB.set_title("(B) Mondrian coverage vs ε\n(per-instrument)")
        axB.grid(True, linestyle="--", alpha=0.4)

        # Col 2 – interval width vs epsilon
        axC = axes[i, 2]
        for m in methods:
            subm = agg[agg["method"] == m]
            if subm.empty:
                continue
            x = [xmap[lbl] for lbl in subm["dp_label"]]
            axC.errorbar(
                x,
                subm["width_mean"],
                yerr=subm["width_std"].fillna(0.0),
                marker="o",
                linestyle="-",
                label=m,
            )
        if i == nrows - 1:
            axC.set_xticks(xs)
            axC.set_xticklabels(eps_labels)
            axC.set_xlabel("DP epsilon (ε)")
        else:
            axC.set_xticks(xs)
            axC.set_xticklabels([])
        axC.set_ylabel(f"Mean width (k={k})")
        if i == 0:
            axC.set_title("(C) Interval width vs ε")
        axC.grid(True, linestyle="--", alpha=0.4)

        # Col 3 – width vs RMSEP
        axD = axes[i, 3]
        
        # Define marker styles for epsilon values
        eps_markers = {
            "inf": "o",
            "0.1": "s",
            "1": "^",
            "10": "D",
            "100": "v",
        }
        # Get default marker for any epsilon not in the dict
        default_marker = "p"
        
        # Plot with different markers for eps and colors for methods
        method_handles = {}
        eps_handles = {}
        
        for m in methods:
            subm = agg[agg["method"] == m]
            if subm.empty:
                continue
            for _, row in subm.iterrows():
                eps_lbl = row["dp_label"]
                marker = eps_markers.get(eps_lbl, default_marker)
                sc = axD.scatter(
                    row["rmsep_mean"],
                    row["width_mean"],
                    s=60,
                    marker=marker,
                    label=m,
                )
                # Store handles for legend
                if m not in method_handles:
                    method_handles[m] = sc
                if eps_lbl not in eps_handles:
                    # Create a dummy scatter for epsilon legend
                    eps_handles[eps_lbl] = axD.scatter(
                        [], [], s=60, marker=marker, color="gray", label=f"ε={eps_lbl}"
                    )
        
        if i == nrows - 1:
            axD.set_xlabel("RMSEP on test set")
        axD.set_ylabel(f"Mean width (k={k})")
        axD.set_xscale("log")
        if i == 0:
            axD.set_title("(D) Accuracy vs interval width")
            # Create combined legend: methods + epsilon markers
            method_legend = axD.legend(
                handles=list(method_handles.values()),
                labels=list(method_handles.keys()),
                fontsize=7,
                loc="upper left",
                title="Method"
            )
            axD.add_artist(method_legend)
            axD.legend(
                handles=list(eps_handles.values()),
                labels=[f"ε={lbl}" for lbl in eps_handles.keys()],
                fontsize=7,
                loc="upper right",
                title="Epsilon"
            )
        axD.grid(True, linestyle="--", alpha=0.4)

    # Increase axis and tick label font size
    for ax in axes.flat:
        ax.tick_params(axis='both', labelsize=12)  # Increase tick label font size
        ax.xaxis.label.set_size(11)  # Increase x-axis label font size
        ax.yaxis.label.set_size(11)  # Increase y-axis label font size

    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate grid R&D plots (rows = transfer_k) from master_database JSON"
    )
    parser.add_argument(
        "--db",
        type=str,
        default="final2/publication_database/json/master_database_20251120_200830.json",
        help="Path to master_database JSON",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="figs_grid",
        help="Output directory for figures",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="FedPLS",
        help="Method to focus on for Figure 4 (default: FedPLS)",
    )
    args = parser.parse_args()

    db_path = Path(args.db)
    outdir = Path(args.outdir)

    with open(db_path, "r") as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    transfer_ks = (20, 80, 200)

    plot_fig2_baseline_vs_federated_grid(
        df,
        transfer_ks=transfer_ks,
        save_path=outdir / "fig2_baseline_vs_federated_grid.png",
    )

    plot_fig3_transfer_budget_grid(
        df,
        transfer_ks=transfer_ks,
        save_path=outdir / "fig3_transfer_budget_grid.png",
    )

    plot_fig4_privacy_utility_grid(
        df,
        method=args.method,
        transfer_ks=transfer_ks,
        save_path=outdir / "fig4_privacy_utility_grid.png",
    )

    plot_fig5_conformal_grid(
        df,
        transfer_ks=transfer_ks,
        save_path=outdir / "fig5_conformal_coverage_width_grid.png",
    )


if __name__ == "__main__":
    main()
