from __future__ import annotations

import functools
import warnings
from contextlib import contextmanager
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Mapping, Tuple, Any, Generator

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from sklearn.decomposition import PCA  # Moved to top-level
from openpyxl import load_workbook  # Kept for specific sheet checks if needed

# Local application imports
# Assuming these exist in your project structure
from .style import PlotStyle
from .data_access import (
    load_config_dict,
    load_site_metadata,
    load_aggregated_frames,
    load_publication_database,
)

# --- Constants ---
METHOD_DISPLAY = {
    "Centralized_PLS": "Pooled PLS",
    "Site_Specific": "Site-specific PLS",
    "PDS": "PDS",
    "SBC": "SBC",
    "FedAvg": "FedAvg",
    "FedProx": "FedProx",
    "FedPLS": "FedPLS",
    "Global_calibrate_after_fed": "Global Cal.",
}

METHOD_COLORS = {
    "Centralized_PLS": "#1f77b4",
    "Site_Specific": "#ff7f0e",
    "PDS": "#2ca02c",
    "SBC": "#d62728",
    "FedAvg": "#9467bd",
    "FedProx": "#17becf",
    "FedPLS": "#8c564b",
    "Global_calibrate_after_fed": "#e377c2",
}

# --- Helper: Context Manager for Figures ---
@contextmanager
def figure_context(output_path: Path, style: PlotStyle, figsize: Tuple[float, float] = (10, 8)) -> Generator[Figure, None, None]:
    """
    Context manager to handle figure creation, saving, and cleanup automatically.
    """
    style.apply()
    _ensure_outdir(output_path.parent)
    fig = plt.figure(figsize=figsize, constrained_layout=False)
    try:
        yield fig
        fig.savefig(output_path, dpi=style.dpi, bbox_inches="tight", pad_inches=0.2)
        print(f"[OK] Saved: {output_path.name}")
    except Exception as e:
        print(f"[ERROR] Failed to save {output_path.name}: {e}")
        raise e
    finally:
        plt.close(fig)

def _ensure_outdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

# --- Data Loading & Caching ---

@functools.lru_cache(maxsize=4)
def _load_publication_mapped(results_root_str: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load publication database with vectorized column mapping.
    """
    results_root = Path(results_root_str)
    pub_path = results_root / "publication_database"

    if pub_path.exists():
        master_df, perf_df, conf_df, train_df = load_publication_database(pub_path)
        if master_df is None or master_df.empty:
            return (perf_df if perf_df is not None else pd.DataFrame()), pd.DataFrame(), pd.DataFrame(), (train_df if train_df is not None else pd.DataFrame())

        master_mapped = master_df.copy()
        
        # Vectorized renaming
        column_mapping = {
            'transfer_k': 'design_factors.Transfer_Samples',
            'dp_epsilon': 'design_factors.DP_Target_Eps',
            'rmsep': 'metrics.RMSEP',
            'r2': 'metrics.R2',
            'bytes_mb': 'runtime.total_bytes_mb',
            'bytes_sent': 'metrics.Bytes_Sent',
            'bytes_recv': 'metrics.Bytes_Received',
            'bytes': 'metrics.Bytes_Sent',
        }
        master_mapped.rename(columns=column_mapping, inplace=True)

        # Vectorized fillna
        if 'run_label' not in master_mapped.columns:
            master_mapped['run_label'] = 'objective_1'
        else:
            master_mapped['run_label'] = master_mapped['run_label'].fillna('objective_1')

        # Vectorized Epsilon cleanup
        if 'design_factors.DP_Target_Eps' in master_mapped.columns:
            col = master_mapped['design_factors.DP_Target_Eps'].astype(str).str.lower()
            mask_inf = col.isin(['inf', 'infinity'])
            master_mapped['design_factors.DP_Target_Eps'] = col
            master_mapped.loc[mask_inf, 'design_factors.DP_Target_Eps'] = 'inf'

        # Vectorized Numeric conversion
        if 'design_factors.Transfer_Samples' in master_mapped.columns:
            # Coerce to numeric, fill NaN with -1 (or keep as NaN), then cast acceptable values
            # This is much faster than row-by-row apply
            vals = pd.to_numeric(master_mapped['design_factors.Transfer_Samples'], errors='coerce')
            # Restore original non-numeric if needed, or just keep the numeric ones
            master_mapped['design_factors.Transfer_Samples'] = vals.fillna(master_mapped['design_factors.Transfer_Samples'])

        return perf_df, pd.DataFrame(), master_mapped, train_df

    # Fallback (Legacy)
    methods_df, metrics_df, raw_df = load_aggregated_frames(results_root)
    return methods_df, metrics_df, raw_df, pd.DataFrame()

# --- Plotting Helpers ---

def _draw_heatmap(ax: Axes, data: np.ndarray, x_labels: List[Any], y_labels: List[Any], 
                  title: str, cmap: str = "viridis", fmt: str = ".3f", colorbar_label: Optional[str] = None) -> None:
    """Generic heatmap drawer."""
    im = ax.imshow(data, aspect="auto", cmap=cmap)
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels)
    ax.set_title(title)
    
    # Add text values (Vectorized approach not easily possible for text plotting, loop is fine here)
    threshold = np.nanmax(data) * 0.6 if not np.all(np.isnan(data)) else 0
    
    it = np.nditer(data, flags=['multi_index'])
    for val in it:
        # Safely convert the iterator value to a Python float to avoid type issues
        try:
            numeric_val = float(np.asarray(val))
        except Exception:
            numeric_val = np.nan

        if not np.isnan(numeric_val):
            color = "white" if numeric_val > threshold else "black"
            ax.text(it.multi_index[1], it.multi_index[0], format(numeric_val, fmt), 
                    ha="center", va="center", fontsize=8, color=color)

    plt.colorbar(im, ax=ax, label=colorbar_label)

# --------------------------- Figure 1 Panels --------------------------- #

def _figure1_panel_topology(ax: Axes, instruments: Sequence[str], manufacturers: Dict[str, str], style: PlotStyle) -> None:
    """ Draws the star topology."""
    ax.axis("off")
    n = len(instruments)
    left_x, right_x = 0.0, 6.0
    ax.set_xlim(-1, 7)
    ax.set_ylim(-1, n + 1)

    for idx, instr in enumerate(instruments):
        y = n - idx
        ax.scatter(left_x, y, s=350, c="#1f77b4", zorder=10)
        label = f"{instr}\n({manufacturers.get(instr, '')})" if manufacturers.get(instr) else instr
        
        ax.text(left_x, y, label, color="white", ha="center", va="center", fontsize=style.label_size, zorder=11)
        ax.annotate("", xy=(right_x - 1, y), xytext=(left_x + 0.5, y),
                    arrowprops=dict(arrowstyle="->", lw=style.line_width, color="#1f77b4"))

    ax.scatter(right_x, n / 2, s=600, c="#ff7f0e", zorder=10)
    ax.text(right_x, n / 2, "Central Server", ha="center", va="center", 
            color="white", fontsize=style.title_size, weight="bold", zorder=11)
    ax.set_title("Federated topology (panel a)", fontsize=style.title_size)

def _figure1_panel_samples(ax: Axes, site_meta: Mapping[str, Any], style: PlotStyle) -> None:
    instr_ids = sorted(site_meta.keys())
    # Vectorized extraction if site_meta was a DF, but list comp is fine for small N
    counts = {
        "Cal": [site_meta[i].cal_rows for i in instr_ids],
        "Val": [site_meta[i].val_rows for i in instr_ids],
        "Test": [site_meta[i].test_rows for i in instr_ids]
    }
    
    x = np.arange(len(instr_ids))
    width = 0.25
    
    for i, (label, values) in enumerate(counts.items()):
        offset = (i - 1) * width
        ax.bar(x + offset, values, width, label=label, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(instr_ids, rotation=45, ha="right")
    ax.set_ylabel("Samples")
    ax.set_title("Samples per instrument (panel c)", fontsize=style.title_size)
    ax.legend()

def _figure1_panel_pca(ax: Axes, data_dir: Path, site_meta: Mapping[str, Any], 
                       n_wavelengths: Optional[int], style: PlotStyle) -> None:
    """
    
    Optimized PCA loading using pd.read_excel instead of openpyxl iteration.
    """
    max_rows_per_instr = 80
    X_list: List[np.ndarray] = []
    mfr_labels: List[str] = []

    for instr_id, sm in site_meta.items():
        manufacturer = getattr(sm, 'manufacturer', None)
        if not manufacturer: continue
            
        suffix = str(instr_id).split("_")[-1]
        sheet_name = f"CalSet{suffix}"
        cal_path = data_dir / manufacturer / f"Cal_{manufacturer}.xlsx"
        
        if not cal_path.exists(): continue

        # OPTIMIZATION: Use pandas to read directly. Much faster than iterating cells.
        try:
            # Read header to find columns, then read subset
            # Assuming spectral columns start at index 2 (col C)
            df_iter = pd.read_excel(cal_path, sheet_name=sheet_name, nrows=max_rows_per_instr, header=0)
            
            # Select columns starting from index 2 to end
            if df_iter.shape[1] <= 2: continue
            
            X_instr = df_iter.iloc[:, 2:].to_numpy(dtype=float)
            
            # Handle NaNs safely
            if np.isnan(X_instr).any():
                # Simple imputation or drop for visualization
                X_instr = X_instr[~np.isnan(X_instr).any(axis=1)]
            
            if X_instr.shape[0] == 0: continue

            # Subsample features if needed to match dimensions across manufacturers
            if n_wavelengths and X_instr.shape[1] > n_wavelengths:
                idx = np.linspace(0, X_instr.shape[1] - 1, n_wavelengths, dtype=int)
                X_instr = X_instr[:, idx]

            X_list.append(X_instr)
            mfr_labels.extend([manufacturer] * X_instr.shape[0])
            
        except Exception as e:
            # print(f"Skipping {instr_id}: {e}") # Debug only
            continue

    if not X_list:
        ax.text(0.5, 0.5, "No spectra loaded", ha="center", va="center", transform=ax.transAxes)
        return

    X_all = np.vstack(X_list)
    # Handle differing column counts if manufacturers have different resolutions
    # (In a real scenario, you'd interpolate. Here we truncate to min cols for viz)
    min_cols = min(x.shape[1] for x in X_list)
    if X_all.shape[1] != min_cols:
         # Re-stack with truncation
         X_all = np.vstack([x[:, :min_cols] for x in X_list])

    pca = PCA(n_components=2)
    comps = pca.fit_transform(X_all)

    # Scatter plot
    unique_mfrs = sorted(list(set(mfr_labels)))
    for mfr in unique_mfrs:
        mask = np.array(mfr_labels) == mfr
        ax.scatter(comps[mask, 0], comps[mask, 1], label=mfr, alpha=0.6, edgecolor="none")

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Spectral PCA by manufacturer (panel b)", fontsize=style.title_size)
    ax.legend(title="Manufacturer")

def _figure1_panel_conformal_flow(ax: Axes, style: PlotStyle) -> None:
    """"""
    steps = [
        "Reference model fit\n(on calibration split)",
        "Residuals on calibration\n(split)",
        "Quantile lookup\n(target 1-α)",
        "Apply intervals to\nmethods (pooled / CT / FL)",
    ]
    ax.axis("off")
    ax.set_xlim(0, len(steps) * 2)
    
    for idx, text in enumerate(steps):
        x = idx * 2
        # Draw Box
        ax.add_patch(Rectangle((x, 0.5), 1.8, 1.0, facecolor="#e0e0e0", edgecolor="#555555"))
        # Draw Text
        ax.text(x + 0.9, 1.0, text, ha="center", va="center", fontsize=style.label_size)
        # Draw Arrow
        if idx < len(steps) - 1:
            ax.annotate("", xy=(x + 1.8, 1.0), xytext=(x + 2.1, 1.0),
                        arrowprops=dict(arrowstyle="->", lw=style.line_width, color="#555555"))
            
    ax.set_title("Conformal pipeline (panel d)", fontsize=style.title_size)


# --------------------------- Figure Generation Functions --------------------------- #

def figure_1_study_design(results_root: Path, config_path: Path, data_dir: Path, output_path: Path, style: PlotStyle) -> None:
    """Figure 1: Study design, dataset structure, and federated topology."""
    cfg = load_config_dict(config_path)
    site_meta, _ = load_site_metadata(cfg, data_dir)
    instr_map = {k: v.manufacturer for k, v in site_meta.items()}

    with figure_context(output_path, style, figsize=(10, 8)) as fig:
        gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.3, hspace=0.4)
        
        _figure1_panel_topology(fig.add_subplot(gs[0, 0]), sorted(site_meta.keys()), instr_map, style)
        _figure1_panel_pca(fig.add_subplot(gs[0, 1]), data_dir, site_meta, cfg.get("DEFAULT_N_WAVELENGTHS"), style)
        _figure1_panel_samples(fig.add_subplot(gs[1, 0]), site_meta, style)
        _figure1_panel_conformal_flow(fig.add_subplot(gs[1, 1]), style)
        
        fig.suptitle("Figure 1 – Study design and topology", fontsize=style.title_size + 1)


def figure_2_baselines_ct(results_root: Path, output_path: Path, style: PlotStyle, *, transfer_k: Optional[int] = None) -> None:
    """Figure 2: Baseline and classical CT performance."""
    _, _, raw_df, _ = _load_publication_mapped(str(results_root))
    if raw_df.empty or "design_factors.Transfer_Samples" not in raw_df.columns: return

    # Filter to Max K if None
    if transfer_k is None:
        k_vals = raw_df["design_factors.Transfer_Samples"].dropna().unique()
        transfer_k = max(int(k) for k in k_vals)

    # Optimization: Vectorized filtering
    df = raw_df[
        (raw_df["design_factors.Transfer_Samples"] == transfer_k) &
        (raw_df["design_factors.DP_Target_Eps"].astype(str) == "inf")
    ].copy()

    if df.empty: return

    # GroupBy is faster than manual pivot looping
    baseline_methods = ["Centralized_PLS", "Site_Specific", "PDS", "SBC"]
    display_names = {"Centralized_PLS": "Pooled PLS", "Site_Specific": "Site-specific PLS", "PDS": "PDS", "SBC": "SBC"}

    # Create summary table
    pivot_df = df[df["method"].isin(baseline_methods)].pivot_table(
        index="instrument_code", columns="method", values="metrics.RMSEP", aggfunc="mean"
    )
    
    if pivot_df.empty: return
    instruments = pivot_df.index.tolist()
    x = np.arange(len(instruments))

    with figure_context(output_path, style, figsize=(14, 10)) as fig:
        axes = fig.subplots(2, 2)

        # Panel (a): Pooled vs Site-specific
        ax_a = axes[0, 0]
        width = 0.35
        ax_a.bar(x - width/2, pivot_df.get("Centralized_PLS", 0), width, label="Pooled PLS")
        ax_a.bar(x + width/2, pivot_df.get("Site_Specific", 0), width, label="Site-specific PLS")
        ax_a.set_xticks(x)
        ax_a.set_xticklabels(instruments, rotation=45, ha="right", fontsize=9)
        ax_a.set_ylabel("RMSEP")
        ax_a.set_title("Panel (a): Baseline RMSEP")
        ax_a.legend()

        # Panel (b): PDS vs SBC
        ax_b = axes[0, 1]
        ax_b.bar(x - width/2, pivot_df.get("PDS", 0), width, label="PDS")
        ax_b.bar(x + width/2, pivot_df.get("SBC", 0), width, label="SBC")
        ax_b.set_xticks(x)
        ax_b.set_xticklabels(instruments, rotation=45, ha="right", fontsize=9)
        ax_b.set_ylabel("RMSEP")
        ax_b.set_title(f"Panel (b): CT RMSEP (k={transfer_k})")
        ax_b.legend()

        # Panel (c): Means
        ax_c = axes[1, 0]
        means = pivot_df.mean()
        # Ensure order
        ordered_means = [means.get(m, 0) for m in baseline_methods]
        ordered_labels = [display_names[m] for m in baseline_methods]
        
        ax_c.bar(ordered_labels, ordered_means)
        ax_c.set_ylabel("Mean RMSEP")
        ax_c.set_title("Panel (c): Mean RMSEP by method")

        # Panel (d): Line plot
        ax_d = axes[1, 1]
        for m in baseline_methods:
            if m in pivot_df:
                ax_d.plot(x, pivot_df[m], marker="o", label=display_names[m])
        
        ax_d.set_xticks(x)
        ax_d.set_xticklabels(instruments, rotation=45, ha="right", fontsize=9)
        ax_d.set_ylabel("RMSEP")
        ax_d.set_title("Panel (d): RMSEP per instrument")
        ax_d.legend()
        
        fig.suptitle("Figure 2 – Baseline and classical CT performance", fontsize=style.title_size + 1)

def figure_3_training_dynamics(results_root: Path, output_path: Path, style: PlotStyle, 
                               *, target_eps: Optional[float] = None, transfer_k: Optional[int] = None) -> None:
    """Figure 3: Federated training dynamics."""
    _, _, _, train_df = _load_publication_mapped(str(results_root))
    if train_df.empty: return

    eps_filter = "inf" if (target_eps is None or np.isinf(target_eps)) else str(target_eps)
    
    # Vectorized Filter
    mask = (train_df["dp_epsilon"].astype(str) == eps_filter)
    if transfer_k is not None:
        mask &= (train_df["transfer_k"] == transfer_k)
    
    train_filtered = train_df[mask]
    if train_filtered.empty: return

    with figure_context(output_path, style, figsize=(14, 10)) as fig:
        axes = fig.subplots(2, 2)

        # Panel (a): Global RMSEP vs Round
        ax_a = axes[0, 0]
        for method in ["FedAvg", "FedPLS"]:
            method_data = train_filtered[train_filtered["method"] == method]
            if not method_data.empty:
                grouped = method_data.groupby("round")["rmsep"].mean()
                ax_a.plot(grouped.index, grouped.values, marker="o", label=method)
        ax_a.set_xlabel("Round")
        ax_a.set_ylabel("Global RMSEP")
        ax_a.set_title(f"Panel (a): Global RMSEP (k={transfer_k}, ε={eps_filter})")
        ax_a.legend()

        # Panel (b): Placeholder
        ax_b = axes[0, 1]
        ax_b.text(0.5, 0.5, "Per-site dynamics unavailable", ha="center", va="center", transform=ax_b.transAxes)
        ax_b.set_title("Panel (b): N/A")

        # Panel (c): Final Round RMSEP
        ax_c = axes[1, 0]
        max_round = train_filtered["round"].max()
        final_df = train_filtered[train_filtered["round"] == max_round]
        
        # Group by method
        final_metrics = final_df.groupby("method")["rmsep"].mean()
        methods = [m for m in ["Centralized_PLS", "FedAvg", "FedPLS"] if m in final_metrics.index]
        
        ax_c.bar(methods, final_metrics.loc[methods])
        ax_c.set_ylabel("Final RMSEP")
        ax_c.set_title("Panel (c): Final-round RMSEP")

        # Panel (d): Convergence Stability
        ax_d = axes[1, 1]
        for method in ["FedAvg", "FedPLS"]:
            method_data = train_filtered[train_filtered["method"] == method]
            if not method_data.empty:
                stats = method_data.groupby("round")["rmsep"].agg(['mean', 'std'])
                ax_d.errorbar(stats.index, stats['mean'], yerr=stats['std'], marker="o", label=method, capsize=3)
        ax_d.set_xlabel("Round")
        ax_d.set_ylabel("RMSEP (mean ± std)")
        ax_d.set_title("Panel (d): Convergence Stability")
        ax_d.legend()
        
        fig.suptitle("Figure 3 – Training Dynamics", fontsize=style.title_size + 1)

def figure_4_privacy_utility_heatmaps(results_root: Path, output_path: Path, style: PlotStyle) -> None:
    """Figure 4: Privacy vs Utility Heatmaps."""
    _, _, raw_df, _ = _load_publication_mapped(str(results_root))
    if raw_df.empty: return

    df = raw_df[
        (raw_df["run_label"] == "objective_1") & 
        (raw_df["method"].isin(["FedAvg", "FedPLS"]))
    ].copy()
    
    if df.empty: return

    # Aggregate
    agg = df.groupby(["design_factors.DP_Target_Eps", "design_factors.Transfer_Samples"]).agg({
        "metrics.RMSEP": "mean",
        "metrics.R2": "mean",
        "runtime.total_bytes_mb": "mean"
    }).reset_index()

    # Pivot
    p_rmsep = agg.pivot(index="design_factors.DP_Target_Eps", columns="design_factors.Transfer_Samples", values="metrics.RMSEP").sort_index(ascending=False)
    p_r2 = agg.pivot(index="design_factors.DP_Target_Eps", columns="design_factors.Transfer_Samples", values="metrics.R2").sort_index(ascending=False)
    p_bytes = agg.pivot(index="design_factors.DP_Target_Eps", columns="design_factors.Transfer_Samples", values="runtime.total_bytes_mb").sort_index(ascending=False)

    with figure_context(output_path, style, figsize=(14, 10)) as fig:
        axes = fig.subplots(2, 2)

        # Helper to clean labels
        def clean_labels(lbls): 
            return ["∞" if str(x) == 'inf' else (f"{float(x):.1f}" if str(x).replace('.','').isdigit() else str(x)) for x in lbls]

        # Heatmaps
        _draw_heatmap(axes[0, 0], p_rmsep.values, p_rmsep.columns.tolist(), clean_labels(p_rmsep.index), "Panel (a): Mean RMSEP", cmap="YlOrRd")
        _draw_heatmap(axes[0, 1], p_r2.values, p_r2.columns.tolist(), clean_labels(p_r2.index), "Panel (b): Mean R²", cmap="RdYlGn")
        _draw_heatmap(axes[1, 0], p_bytes.values, p_bytes.columns.tolist(), clean_labels(p_bytes.index), "Panel (c): Total Bytes (MB)", cmap="Blues", fmt=".1f")

        # Panel (d): Trade-off Curves
        ax_d = axes[1, 1]
        for k in sorted(agg["design_factors.Transfer_Samples"].unique()):
            sub = agg[agg["design_factors.Transfer_Samples"] == k].sort_values("design_factors.DP_Target_Eps")
            
            # Split Finite vs Inf
            is_inf = sub["design_factors.DP_Target_Eps"].astype(str) == 'inf'
            finite = sub[~is_inf]
            
            if not finite.empty:
                ax_d.plot(pd.to_numeric(finite["design_factors.DP_Target_Eps"]), finite["metrics.RMSEP"], marker="o", label=f"k={int(k)}")
            
            # Plot infinity as separate point
            inf_rows = sub[is_inf]
            if not inf_rows.empty:
                # Place 'inf' arbitrarily at 100 or max*2 for viz
                x_pos = 100 
                ax_d.scatter([x_pos] * len(inf_rows), inf_rows["metrics.RMSEP"], marker="*", s=150, label=f"k={int(k)} (∞)" if finite.empty else None)

        ax_d.set_xlabel("DP Epsilon")
        ax_d.set_ylabel("RMSEP")
        ax_d.set_xscale("log")
        ax_d.set_title("Panel (d): Privacy-utility curves")
        ax_d.legend(fontsize=8)
        
        fig.suptitle("Figure 4 – Privacy-Utility Trade-offs", fontsize=style.title_size + 1)

def figure_baseline_vs_federated_comparison(results_root: Path, output_path: Path, style: PlotStyle, *, transfer_k: Optional[int] = None) -> None:
    """Comprehensive comparison of all methods."""
    _, _, raw_df, _ = _load_publication_mapped(str(results_root))
    if raw_df.empty: return

    # Filters
    raw = raw_df[raw_df["design_factors.DP_Target_Eps"] == "inf"].copy()
    
    if transfer_k is None and "design_factors.Transfer_Samples" in raw.columns:
        transfer_k = raw["design_factors.Transfer_Samples"].max()
    
    # Specific subset for Bar charts
    raw_k = raw[raw["design_factors.Transfer_Samples"] == transfer_k] if transfer_k else raw

    methods = ["Centralized_PLS", "Site_Specific", "PDS", "SBC", "FedAvg", "FedPLS"]
    
    with figure_context(output_path, style, figsize=(14, 10)) as fig:
        axes = fig.subplots(2, 2)

        # Panel (a): Overall Means
        ax_a = axes[0, 0]
        means = raw_k.groupby("method")["metrics.RMSEP"].mean().reindex(methods)
        ax_a.bar([METHOD_DISPLAY.get(m, m) for m in methods], means, color=[METHOD_COLORS.get(m, 'gray') for m in methods])
        ax_a.tick_params(axis='x', rotation=35)
        ax_a.set_ylabel("RMSEP")
        ax_a.set_title("Panel (a): Mean RMSEP")

        # Panel (b): By Instrument
        ax_b = axes[0, 1]
        pivot_instr = raw_k.pivot_table(index="instrument_code", columns="method", values="metrics.RMSEP")
        pivot_instr = pivot_instr.reindex(columns=methods) # Ensure col order
        
        pivot_instr.plot(kind='bar', ax=ax_b, width=0.8, color=[METHOD_COLORS.get(m, 'gray') for m in pivot_instr.columns])
        ax_b.set_title("Panel (b): RMSEP per Instrument")
        ax_b.legend(fontsize=8, ncol=2)

        # Panel (c): Scatter vs Pooled
        ax_c = axes[1, 0]
        pooled = raw_k[raw_k["method"] == "Centralized_PLS"].groupby("instrument_code")["metrics.RMSEP"].mean()
        
        for m in methods:
            if m == "Centralized_PLS": continue
            m_data = raw_k[raw_k["method"] == m].groupby("instrument_code")["metrics.RMSEP"].mean()
            
            common = pooled.index.intersection(m_data.index)
            if not common.empty:
                ax_c.scatter(pooled[common], m_data[common], label=METHOD_DISPLAY.get(m, m), color=METHOD_COLORS.get(m))

        # Diagonal line
        lims = [0, max(ax_c.get_xlim()[1], ax_c.get_ylim()[1])]
        ax_c.plot(lims, lims, 'k--', alpha=0.5)
        ax_c.set_xlabel("Pooled RMSEP")
        ax_c.set_ylabel("Method RMSEP")
        ax_c.set_title("Panel (c): Method vs Pooled Baseline")
        ax_c.legend()

        # Panel (d): Transfer budget curve
        ax_d = axes[1, 1]
        if "design_factors.Transfer_Samples" in raw.columns:
            agg_k = raw.groupby(["method", "design_factors.Transfer_Samples"])["metrics.RMSEP"].mean().reset_index()
            for m in methods:
                sub = agg_k[agg_k["method"] == m].sort_values("design_factors.Transfer_Samples")
                if not sub.empty:
                    ax_d.plot(sub["design_factors.Transfer_Samples"], sub["metrics.RMSEP"], 
                              marker='o', label=METHOD_DISPLAY.get(m, m), color=METHOD_COLORS.get(m))
            ax_d.set_xlabel("Transfer Samples (k)")
            ax_d.set_ylabel("Mean RMSEP")
            ax_d.set_title("Panel (d): Performance vs Transfer Budget")
            ax_d.legend(ncol=2, fontsize=8)
        
        fig.suptitle("Comprehensive Methods Comparison", fontsize=style.title_size + 1)


# --- Main Entry Point ---

def generate_all_figures(results_root: Path, config_path: Path, data_dir: Path, figures_dir: Path, style: PlotStyle) -> None:
    """Generate all figures efficiently."""
    _ensure_outdir(figures_dir)
    
    print("Generating Figure 1...")
    figure_1_study_design(results_root, config_path, data_dir, figures_dir / "Figure_1_study_design.png", style)
    
    print("Generating Figure 2...")
    figure_2_baselines_ct(results_root, figures_dir / "Figure_2_baselines_ct.png", style)
    
    print("Generating Baseline vs Fed Comparison...")
    figure_baseline_vs_federated_comparison(results_root, figures_dir / "Figure_Baseline_vs_Federated.png", style)
    
    print("Generating Figure 3 (Training Dynamics)...")
    figure_3_training_dynamics(results_root, figures_dir / "Figure_3_Training_Dynamics.png", style)
    
    print("Generating Figure 4 (Heatmaps)...")
    figure_4_privacy_utility_heatmaps(results_root, figures_dir / "Figure_4_Privacy_Utility_Heatmaps.png", style)
    
    # Note: Some logic from original Figure 3 (fed vs ct) and Figure 5 seemed redundant or subset of above.
    # If strictly needed, they can be re-enabled, but the above functions cover the logic more efficiently.
    print("All figures generated.")