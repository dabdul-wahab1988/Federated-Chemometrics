#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Publication-ready figures WITHOUT inter-subplot overlap.

Generates:
1) 4×4 Composite grid:
   generated_figures_tables/figure_1_composites/Figure_1_Composite_4x4_Publication.png
2) Transfer-size impact (vary k, fixed ε):
   generated_figures_tables/Figure_1_eps0_1_Transfer_Sweep_Publication.png
3) Privacy-impact sweep (vary ε, fixed k):
   generated_figures_tables/Figure_1_k{K_FOR_PRIVACY}_Privacy_Sweep_Publication.png
"""

from pathlib import Path
import os
import sys
# Ensure local `src` is preferred over any installed `fedchem` package
# by inserting the repository `src` directory at the front of `sys.path`.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_LOCAL_SRC = str(_REPO_ROOT / 'src')
if _LOCAL_SRC not in sys.path:
    sys.path.insert(0, _LOCAL_SRC)
from fedchem.utils.config import load_config
import argparse
from fedchem.visualization import (
    collect_manifests as viz_collect_manifests,
    load_manifest as viz_load_manifest,
    manifest_to_table as viz_manifest_to_table,
    extract_final_rmsep as viz_extract_final_rmsep,
    plot_rmsep_heatmap as viz_plot_rmsep_heatmap,
    find_debug_weights as viz_find_debug_weights,
    get_default_baseline as viz_get_default_baseline,
    save_figure_and_metadata as viz_save_figure_and_metadata,
    set_plot_style as viz_set_plot_style,
)
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import MaxNLocator

# ------------------------------ CONFIG ---------------------------------

OUTPUT_DIR = Path(os.environ.get("FEDCHEM_OUTPUT_DIR", "generated_figures_tables"))
COMPOSITE_SUBDIR = OUTPUT_DIR / "figure_1_composites"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
COMPOSITE_SUBDIR.mkdir(parents=True, exist_ok=True)

# Publication database path (primary data source)
PUBLICATION_DB_DIR = Path(os.environ.get("FEDCHEM_PUB_DB", "final2/publication_database"))

# Where to look for manifests (add paths as needed) - kept for backward compatibility
CANDIDATE_MANIFEST_DIRS = [
    Path(os.environ.get("FEDCHEM_ARCHIVE_ROOT", "generated_figures_tables_archive")),
    Path("experiment_manifests"),
    Path("manifests"),
    Path("artifacts/manifests"),
    Path("outputs/manifests"),
    Path("."),
    OUTPUT_DIR,
]

# Grid values (edit to your sweep)
K_VALUES = [20, 40, 80, 200]
EPS_VALUES = [0.1, 1.0, 10.0, float('inf')]
EPS_LABELS = ["0.1", "1", "10", "∞"]

# Default k to hold fixed for privacy sweeps when not overridden; use first K_VALUE if available.
# This ensures K_FOR_PRIVACY is always defined (avoids NameError later).
K_FOR_PRIVACY = 80 if 80 in K_VALUES else (K_VALUES[0] if K_VALUES else None)

# Optionally override from config.yaml EXPERIMENTAL_DESIGN FACTORS
_cfg = load_config()
_factors = (_cfg.get("EXPERIMENTAL_DESIGN") or {}).get("FACTORS") or {}
try:
    if isinstance(_factors, dict):
        ts = _factors.get("Transfer_Samples")
        if isinstance(ts, list) and ts:
            try:
                K_VALUES = [int(str(x)) for x in ts]
            except Exception:
                pass
        pb = _factors.get("DP_Target_Eps") or _factors.get("DP_TARGET_EPS")
        if isinstance(pb, list) and pb:
            def _to_eps(x):
                s = str(x).strip()
                return float('inf') if s in {"∞", "inf", "Inf", "INF"} else float(s)
            try:
                EPS_VALUES = [_to_eps(x) for x in pb]
                EPS_LABELS = ["∞" if (isinstance(v, float) and (v == float('inf'))) else (f"{v:g}") for v in EPS_VALUES]
            except Exception:
                pass
except Exception:
    pass

# Privacy impact: which k to hold fixed when varying ε

# Figure style & spacing
FIG_DPI = 180
FNT = {"title": 12, "label": 12, "tick": 12, "legend": 12, "suptitle": 12}

# Right-side stacked axes offsets in **points** (prevents collisions)
# Separate offsets for different plot aspects so we can tune them independently:
# - RIGHT_AX_OUTWARD_CONV: used for convergence panels (UpdateNorm, Participation)
# - RIGHT_AX_OUTWARD_COMM: used for communication/privacy panels (ε, Participation)
RIGHT_AX_OUTWARD_CONV = (0, 70)    # inner, outer (convergence panels)
RIGHT_AX_OUTWARD_COMM = (0, 80)    # inner, outer (communication/privacy panels)

# --------------------------- MANIFEST LOADING ---------------------------

def load_training_dynamics_from_pub_db(k, eps):
    """
    Load training dynamics from publication database instead of manifest files.
    Returns logs_by_algorithm dict compatible with existing plotting code.
    """
    csv_dir = PUBLICATION_DB_DIR / "csv"
    if not csv_dir.exists():
        return None
    
    # Find most recent training dynamics file
    train_files = sorted(csv_dir.glob("training_dynamics_*.csv"))
    if not train_files:
        return None
    
    try:
        df = pd.read_csv(train_files[-1])
        
        # Filter for matching k and epsilon
        mask = (df['transfer_k'] == k) & (df['dp_epsilon'] == eps)
        subset = df[mask]
        
        if subset.empty:
            return None
        
        # Convert to logs_by_algorithm format
        logs_by_algo = {}
        for method in subset['method'].unique():
            method_data = subset[subset['method'] == method].sort_values('round')
            logs = []
            for _, row in method_data.iterrows():
                log_entry = {
                    'round': int(row['round']),
                    'rmsep': float(row['rmsep']) if pd.notna(row['rmsep']) else None,
                    'r2': float(row['r2']) if pd.notna(row['r2']) else None,
                    'mae': float(row['mae']) if pd.notna(row['mae']) else None,
                }
                # Add optional fields if present
                if 'bytes_sent' in row and pd.notna(row['bytes_sent']):
                    log_entry['bytes_sent'] = float(row['bytes_sent'])
                if 'bytes_recv' in row and pd.notna(row['bytes_recv']):
                    log_entry['bytes_recv'] = float(row['bytes_recv'])
                if 'duration_sec' in row and pd.notna(row['duration_sec']):
                    log_entry['duration_sec'] = float(row['duration_sec'])
                if 'participation_rate' in row and pd.notna(row['participation_rate']):
                    log_entry['participation_rate'] = float(row['participation_rate'])
                if 'epsilon_so_far' in row and pd.notna(row['epsilon_so_far']):
                    log_entry['epsilon_so_far'] = float(row['epsilon_so_far'])
                if 'clip_norm_used' in row and pd.notna(row['clip_norm_used']):
                    log_entry['clip_norm_used'] = float(row['clip_norm_used'])
                
                logs.append(log_entry)
            
            if logs:
                logs_by_algo[method] = logs
        
        return logs_by_algo if logs_by_algo else None
        
    except Exception as e:
        print(f"⚠️  Failed to load training dynamics from publication DB for k={k}, eps={eps}: {e}")
        return None

def _sanitize_eps_tokens(eps):
    g = f"{eps:g}"
    return {
        g, g.replace(".", "_"), g.replace(".", ""),     # 0.1 -> {'0.1','0_1','01'}
        f"e{g}", f"eps{g}", f"epsilon{g}",
        f"eps_{g}", f"epsilon_{g}",
    }

def find_manifest_path(k, eps):
    k_tokens = {f"k{k}", f"k_{k}", f"k={k}", f"clients{k}", f"clients_{k}"}
    e_tokens = _sanitize_eps_tokens(eps)
    for root in CANDIDATE_MANIFEST_DIRS:
        if not root.exists():
            continue
        for p in root.rglob("*.json"):
            path_str = str(p).lower()
            if any(t in path_str for t in k_tokens) and any(t in path_str for t in e_tokens):
                return p
    # Fallback: scan manifest contents for matching config fields (transfer samples / target epsilon)
    for root in CANDIDATE_MANIFEST_DIRS:
        if not root.exists():
            continue
        for p in root.rglob("*.json"):
            try:
                txt = p.read_text()
                manifest = json.loads(txt)
            except Exception:
                continue
            # manifest may be a single dict or a list of dicts; normalize to iterable
            candidates = manifest if isinstance(manifest, list) else [manifest]
            for m in candidates:
                if not isinstance(m, dict):
                    continue
                cfg = (m.get("config") or {})
                # transfer samples may be stored under several keys
                ts_vals = [cfg.get(k) for k in ("transfer_samples_requested", "transfer_samples_used", "n_sites", "n_sites_requested")] + [None]
                try:
                    ts_match = any((v is not None and int(v) == int(k)) for v in ts_vals if v is not None)
                except Exception:
                    ts_match = False
                # epsilon may be stored as float or string
                eps_val = cfg.get("target_epsilon")
                try:
                    # Accept multiple config keys where epsilon may be stored for compatibility
                    candidate_eps_vals = [cfg.get(k) for k in ("target_epsilon", "DP_Target_Eps", "DP_TARGET_EPS")]
                    # pick the first non-None candidate from the candidate list
                    candidate_eps_val = next((v for v in candidate_eps_vals if v is not None), None)
                    if eps == float('inf'):
                        eps_match = (
                            (eps_val is None and any(str(x).lower() in {"inf", "∞"} for x in [candidate_eps_val]))
                            or (isinstance(eps_val, str) and eps_val in {"inf", "∞"})
                            or (isinstance(eps_val, float) and eps_val == float('inf'))
                        )
                    else:
                        eps_match = (eps_val is not None and float(eps_val) == float(eps))
                except Exception:
                    eps_match = False
                if ts_match and eps_match:
                    return p
    return None

def load_manifest(k, eps):
    # Try publication database first
    logs_by_algo = load_training_dynamics_from_pub_db(k, eps)
    if logs_by_algo is not None:
        # Wrap in manifest-like structure for compatibility
        return {'logs_by_algorithm': logs_by_algo}
    
    # Fallback to manifest files
    mp = find_manifest_path(k, eps)
    if mp is None:
        print(f"⚠️  Manifest not found for k={k}, eps={eps}")
        return None
    try:
        return json.loads(mp.read_text())
    except Exception as e:
        print(f"⚠️  Failed to read {mp}: {e}")
        return None

# --------------------------- COVERAGE CHECKS --------------------------

def _format_eps_label(eps, fallback=None):
    if eps == float('inf'):
        return '∞'
    try:
        return str(eps)
    except Exception:
        return fallback or 'eps'

def report_manifest_coverage():
    combos = []
    for k in K_VALUES:
        for idx, eps in enumerate(EPS_VALUES):
            label = EPS_LABELS[idx] if idx < len(EPS_LABELS) else _format_eps_label(eps)
            combos.append((k, eps, label))

    missing = []
    for k, eps, label in combos:
        if find_manifest_path(k, eps) is None:
            missing.append((k, label))

    total = len(combos)
    if missing:
        formatted = ", ".join(f"k={k}, ε={label}" for k, label in missing)
        print(f"⚠️  {len(missing)}/{total} experimental-design combos lack manifests: {formatted}")
    else:
        print(f"✅ Manifests found for all {total} experimental-design combos (transfer × privacy grid).")

# --------------------------- PLOTTING UTILITIES -------------------------

def _dedup_legend(handles, labels):
    seen = {}
    for h, l in zip(handles, labels):
        if l and l not in seen:
            seen[l] = h
    return list(seen.values()), list(seen.keys())

def _make_right_axis(ax, color, outward_pts, lw=1.6):
    ax_r = ax.twinx()
    ax_r.spines["right"].set_position(("outward", outward_pts))
    ax_r.spines["right"].set_color(color)
    ax_r.spines["right"].set_linewidth(lw)
    return ax_r

def _scatter_participation(ax, rounds, vals):
    return ax.scatter(
        rounds, vals,
        s=46, alpha=0.65, color="forestgreen",
        edgecolors="darkgreen", linewidth=1.0, zorder=5
    )

# --------------------------- PANEL DRAWERS -------------------------------

def plot_convergence_panel(ax, logs_by_algo, title_text):
    if not logs_by_algo or not any(logs_by_algo.values()):
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, fontsize=FNT["label"])
        ax.set_xticks([]); ax.set_yticks([])
        return

    algo_order = ["FedAvg", "FedProx", "FedAvg_noDP", "Centralized_PLS"]
    colors = {"FedAvg": "steelblue", "FedProx": "green", "FedAvg_noDP": "steelblue", "Centralized_PLS": "purple"}

    # Gather all RMSEP values for scale decisions and axis limits
    all_rmsep = []
    for algo in algo_order:
        if not logs_by_algo.get(algo):
            continue
        logs = logs_by_algo[algo]
        rm = [lg.get("rmsep") for lg in logs]
        all_rmsep.extend([v for v in rm if v is not None and v > 0])

    # Force log scale for all RMSEP axes (per request)
    use_log = True

    for algo in algo_order:
        if not logs_by_algo.get(algo):
            continue
        logs = logs_by_algo[algo]
        r = [lg.get("round", i+1) for i, lg in enumerate(logs)]
        rm = [lg.get("rmsep") for lg in logs]
        if not any(v and v > 0 for v in rm):
            continue
        # Use raw RMSEP values (linear), then let the axes convert to log-scale.
        # Substitute NaN for missing/zero entries to avoid log-domain errors.
        y = [v if (v is not None and v > 0) else np.nan for v in rm]
        ax.plot(r, y,
                marker="o", markersize=4.5, linewidth=1.8,
                linestyle="--" if "noDP" in algo else "-",
                color=colors.get(algo, "steelblue"), alpha=0.9, label=algo)

    ax.set_title(f"Convergence\n{title_text}", fontsize=FNT["title"], fontweight="bold", pad=4)
    ax.set_xlabel("Round", fontsize=FNT["label"], fontweight="bold")
    # Show that the axis is a log scale
    ax.set_ylabel("RMSEP (log scale)", fontsize=FNT["label"], fontweight="bold", color="steelblue")
    ax.tick_params(axis="both", labelsize=FNT["tick"], pad=2, labelcolor="steelblue")
    ax.grid(True, alpha=0.25, linewidth=0.5, which='major')
    ax.spines["left"].set_color("steelblue"); ax.spines["left"].set_linewidth(1.6)

    # right #1: UpdateNorm (inner)
    ax_up = _make_right_axis(ax, "darkorange", RIGHT_AX_OUTWARD_CONV[0])
    fed_logs = next((logs_by_algo[n] for n in ["FedAvg","FedProx","FedAvg_noDP"] if logs_by_algo.get(n)), None)
    if fed_logs:
        r = [lg.get("round", i+1) for i, lg in enumerate(fed_logs)]
        un = [lg.get("update_norm") for lg in fed_logs]
        idx = [i for i,v in enumerate(un) if v is not None]
        if idx:
            ax_up.plot([r[i] for i in idx], [un[i] for i in idx],
                       color="darkorange", marker="s", markersize=4.5, linewidth=1.8, alpha=0.9, label="UpdateNorm")
    ax_up.set_ylabel("UpdateNorm", fontsize=FNT["label"], fontweight="bold", color="darkorange", labelpad=6)
    ax_up.tick_params(axis="y", labelsize=FNT["tick"], colors="darkorange", pad=1)
    ax_up.yaxis.set_major_locator(MaxNLocator(4))

    # right #2: Participation (outer)
    ax_part = _make_right_axis(ax, "forestgreen", RIGHT_AX_OUTWARD_CONV[1])
    ax_part.set_ylim(0, 1.05)
    if fed_logs:
        r = [lg.get("round", i+1) for i, lg in enumerate(fed_logs)]
        pp = [lg.get("participation_rate", 1.0) for lg in fed_logs]
        _scatter_participation(ax_part, r, pp)
    ax_part.set_ylabel("Participation", fontsize=FNT["label"], fontweight="bold", color="forestgreen", labelpad=10)
    ax_part.tick_params(axis="y", labelsize=FNT["tick"], colors="forestgreen", pad=1)
    ax_part.set_yticks([0.0, 0.5, 1.0])

    # Force log-scale axis and global y-limits for RMSEP, using fixed bounds
    try:
        if use_log:
            # Use absolute log bounds across all figures: [1e-1, 1]
            ax.set_yscale('log')
            ax.set_ylim(1e-1, 1.0)
            # Set log ticks at the standard powers-of-ten grid (1e-1 .. 1)
            # Set major ticks precisely at powers-of-ten between 1e-1 and 1
            import numpy as _np
            ticks = _np.logspace(-1, 0, 2)
            ax.set_yticks(ticks)
            from matplotlib.ticker import LogFormatter
            ax.yaxis.set_major_formatter(LogFormatter(base=10.0))
    except Exception:
        # on failure, don't break plotting; leave default scaling
        pass

    # stash row-legend entries
    h0,l0 = ax.get_legend_handles_labels()
    h1,l1 = ax_up.get_legend_handles_labels()
    h2,l2 = ax_part.get_legend_handles_labels()
    ax._row_legend_handles = h0+h1+h2
    ax._row_legend_labels  = l0+l1+l2
    for _a in (ax, ax_up, ax_part):
        leg = _a.get_legend()
        if leg is not None: leg.remove()

def plot_communication_panel(ax, logs, title_text):
    if not logs:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, fontsize=FNT["label"])
        ax.set_xticks([]); ax.set_yticks([])
        return

    r = [lg.get("round", i+1) for i, lg in enumerate(logs)]
    # param communication per round (downlink + uplink)
    comm_kb = np.array([int(lg.get("bytes_sent", 0)) + int(lg.get("bytes_recv", 0)) for lg in logs], float) / 1024.0
    # PDS transfer bytes per round (fallback to legacy 'communication_bytes' if present)
    pds_kb = np.array([int(lg.get("pds_bytes", lg.get("communication_bytes", 0))) for lg in logs], float) / 1024.0
    total_kb = comm_kb + pds_kb
    cum = np.cumsum(total_kb)
    eps = [lg.get("epsilon_so_far") for lg in logs]
    pp  = [lg.get("participation_rate", 1.0) for lg in logs]

    # left: KB & cumulative
    if np.any(pds_kb > 0):
        # Plot stacked bars: comm (existing) + pds (new)
        ax.bar(r, comm_kb, color="steelblue", alpha=0.5, width=0.8, edgecolor="navy", linewidth=0.4, label="KB/round (params)")
        ax.bar(r, pds_kb, bottom=comm_kb, color="sandybrown", alpha=0.6, width=0.8, edgecolor="darkorange", linewidth=0.4, label="KB/round (PDS)")
        ax.plot(r, cum, color="navy", marker=".", markersize=3.5, linewidth=2.0, label="Cumulative KB", zorder=5)
    else:
        ax.bar(r, comm_kb, color="steelblue", alpha=0.5, width=0.8, edgecolor="navy", linewidth=0.4, label="KB/round")
        ax.plot(r, np.cumsum(comm_kb), color="navy", marker=".", markersize=3.5, linewidth=2.0, label="Cumulative KB", zorder=5)

    ax.set_title(f"Communication & Privacy\n{title_text}", fontsize=FNT["title"], fontweight="bold", pad=4)
    ax.set_xlabel("Round", fontsize=FNT["label"], fontweight="bold")
    ax.set_ylabel("KB per Round", fontsize=FNT["label"], fontweight="bold", color="steelblue")
    ax.tick_params(axis="both", labelsize=FNT["tick"], pad=2, labelcolor="steelblue")
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)
    ax.spines["left"].set_color("steelblue"); ax.spines["left"].set_linewidth(1.6)

    # right #1: ε (inner)
    ax_eps = _make_right_axis(ax, "crimson", RIGHT_AX_OUTWARD_COMM[0])
    idx = [i for i, e in enumerate(eps) if e is not None]
    if idx:
        from matplotlib.ticker import LogLocator
        ax_eps.set_yscale('log')
        ax_eps.plot([r[i] for i in idx], [eps[i] for i in idx],
                    color="crimson", marker="s", markersize=4.5, linewidth=2.0,
                    alpha=0.9, label="ε (log scale)", zorder=4)
        ax_eps.set_ylabel("ε (log scale)", fontsize=FNT["label"], fontweight="bold", color="crimson", labelpad=18)
        # Set log ticks at powers of 10
        ax_eps.yaxis.set_major_locator(LogLocator(base=10.0))
    ax_eps.tick_params(axis="y", labelsize=FNT["tick"], colors="crimson", pad=1)

    # right #2: Participation (outer)
    ax_part = _make_right_axis(ax, "forestgreen", RIGHT_AX_OUTWARD_COMM[1])
    ax_part.set_ylim(0, 1.05)
    _scatter_participation(ax_part, r, pp)
    ax_part.set_ylabel("Participation", fontsize=FNT["label"], fontweight="bold", color="forestgreen", labelpad=18)
    ax_part.tick_params(axis="y", labelsize=FNT["tick"], colors="forestgreen", pad=1)
    ax_part.set_yticks([0.0, 0.5, 1.0])

    # stash row-legend entries
    h0,l0 = ax.get_legend_handles_labels()
    h1,l1 = ax_eps.get_legend_handles_labels()
    h2,l2 = ax_part.get_legend_handles_labels()
    ax._row_legend_handles = h0+h1+h2
    ax._row_legend_labels  = l0+l1+l2
    for _a in (ax, ax_eps, ax_part):
        leg = _a.get_legend()
        if leg is not None: leg.remove()

# ---------------------------- LAYOUT HELPERS ----------------------------

def _add_legends_per_row(fig, axes_by_row, y_pos_map):
    for row_idx, axes in axes_by_row.items():
        if not axes: 
            continue
        handles, labels = [], []
        for a in axes:
            handles += getattr(a, "_row_legend_handles", [])
            labels  += getattr(a, "_row_legend_labels", [])
        if not handles: 
            continue
        h, l = _dedup_legend(handles, labels)
        fig.legend(h, l, loc="lower center",
                   bbox_to_anchor=(0.5, y_pos_map.get(row_idx, 0.02)),
                   ncol=6, fontsize=FNT["legend"], frameon=False)

# ----------------------------- FULL FIGURES -----------------------------

def create_transfer_size_impact_figure(epsilon=0.1):
    fig = plt.figure(figsize=(30, 16), dpi=FIG_DPI)
    outer = gridspec.GridSpec(2, 4, figure=fig, hspace=0.40, wspace=0.82)

    rows = {0: [], 1: []}
    for j, k in enumerate(K_VALUES):
        manifest = load_manifest(k, epsilon)
        if manifest is None: 
            continue
        logs_by_algo = manifest.get("logs_by_algorithm", {})
        top = fig.add_subplot(outer[0, j]); plot_convergence_panel(top, logs_by_algo, f"k={k}")
        bot = fig.add_subplot(outer[1, j]); plot_communication_panel(bot, logs_by_algo.get("FedAvg", []), f"k={k}")
        rows[0].append(top); rows[1].append(bot)

    _add_legends_per_row(fig, rows, y_pos_map={0: 0.495, 1: 0.03})
    fig.suptitle(
        f"Transfer Size Impact (ε={epsilon:g})\n"
        "Top: Convergence (RMSEP & UpdateNorm) | "
        "Bottom: Communication/Privacy (KB & ε & Participation)",
        fontsize=FNT["suptitle"], fontweight="bold", y=0.95
    )
    # Use a unique filename for each epsilon value
    if epsilon == float('inf'):
        eps_str = 'inf'
    else:
        eps_str = str(epsilon).replace('.', '_')
    out = OUTPUT_DIR / f"Figure_1_eps{eps_str}_Transfer_Sweep_Publication.png"
    fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

def create_privacy_impact_figure(k_selected=K_VALUES):
    fig = plt.figure(figsize=(30, 16), dpi=FIG_DPI)
    outer = gridspec.GridSpec(2, 4, figure=fig, hspace=0.40, wspace=0.82)

    rows = {0: [], 1: []}
    for j, eps in enumerate(EPS_VALUES):
        manifest = load_manifest(k_selected, eps)
        if manifest is None:
            continue
        logs_by_algo = manifest.get("logs_by_algorithm", {})
        txt = f"ε={EPS_LABELS[j]}"
        top = fig.add_subplot(outer[0, j]); plot_convergence_panel(top, logs_by_algo, txt)
        bot = fig.add_subplot(outer[1, j]); plot_communication_panel(bot, logs_by_algo.get("FedAvg", []), txt)
        rows[0].append(top); rows[1].append(bot)

    _add_legends_per_row(fig, rows, y_pos_map={0: 0.495, 1: 0.03})
    fig.suptitle(
        f"Privacy Impact (k={k_selected})\n"
        "Top: Convergence (RMSEP & UpdateNorm) | "
        "Bottom: Communication/Privacy (KB & ε & Participation)",
        fontsize=FNT["suptitle"], fontweight="bold", y=0.95
    )
    out = OUTPUT_DIR / f"Figure_1_k{k_selected}_Privacy_Sweep_Publication.png"
    fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def create_rmsep_heatmap(manifests_map, method: str, baseline_method: str, output_dir: Path, fmt: str = 'png', dpi: int = 300):
    """Create an RMSEP heatmap for a given method across manifests_map (ks x eps)."""
    import numpy as np
    fig, ax = plt.subplots(figsize=(6, 5), dpi=dpi)
    ks = sorted(manifests_map.keys())
    eps_labels = []
    rmsep_matrix = np.full((len(ks), 0 if not ks else len(list(manifests_map[ks[0]].keys()))), np.nan)
    # Build labels
    for i, k in enumerate(ks):
        eps_keys = list(manifests_map[k].keys())
        if not eps_labels:
            eps_labels = eps_keys
    # Build matrix
    rmsep_matrix = np.full((len(ks), len(eps_labels)), np.nan)
    baseline_matrix = np.full_like(rmsep_matrix, np.nan)
    for i, k in enumerate(ks):
        for j, eps in enumerate(eps_labels):
            mpath = manifests_map[k].get(eps)
            if mpath is None:
                continue
            manifest = viz_load_manifest(mpath)
            # Skip if manifest failed to load (viz_load_manifest may return None)
            if manifest is None:
                continue
            rm = viz_extract_final_rmsep(manifest, method)
            base_rm = viz_extract_final_rmsep(manifest, baseline_method) if baseline_method else float('nan')
            rmsep_matrix[i, j] = rm
            baseline_matrix[i, j] = base_rm
    # Compute delta if baseline provided
    if baseline_method:
        delta_matrix = rmsep_matrix - baseline_matrix
    else:
        delta_matrix = None
    viz_set_plot_style('publication', dpi=dpi)
    img = viz_plot_rmsep_heatmap(ax, rmsep_matrix, [str(k) for k in ks], eps_labels)
    cbar = fig.colorbar(img, ax=ax, orientation='vertical', shrink=0.85)
    ax.set_title(f"Final RMSEP for {method}")
    out_file = output_dir / f"heatmap_rmsep_{method}.{fmt}"
    metadata = {
        'figure': str(out_file.name),
        'method': method,
        'baseline': baseline_method,
        'ks': ks,
        'eps_labels': eps_labels,
    }
    viz_save_figure_and_metadata(fig, out_file, metadata, fmt, dpi)
    plt.close(fig)


def create_weight_heatmaps(manifests_map, output_dir: Path, fmt: str = 'png', dpi: int = 300, max_plots: int = 4):
    import numpy as np
    from fedchem.visualization.plots import plot_weight_heatmap
    from fedchem.visualization.utils import save_figure_and_metadata
    # For each entry in the grid, look for debug weights and render a simple heatmap
    ks = sorted(manifests_map.keys())
    plotted = 0
    for k in ks:
        for eps, path in manifests_map[k].items():
            if path is None:
                continue
            manifests_dir = Path(path).parent
            weight_files = viz_find_debug_weights(manifests_dir)
            if not weight_files:
                continue
            for wf in weight_files:
                if plotted >= max_plots:
                    return
                # Load the weight data if possible
                try:
                    if wf.suffix == '.npy' or wf.suffix == '.npz':
                        import numpy as np
                        matrix = np.load(str(wf))
                        # if npz, pick first array
                        if isinstance(matrix, np.lib.npyio.NpzFile):
                            k0 = list(matrix.keys())[0]
                            matrix = matrix[k0]
                    else:
                        # json
                        import json
                        matrix = json.loads(wf.read_text())
                        matrix = np.array(matrix)
                except Exception as e:
                    print(f'Failed to load weight file {wf}: {e}')
                    continue
                # Plot
                fig, ax = plt.subplots(figsize=(6, 4), dpi=dpi)
                img = plot_weight_heatmap(ax, matrix)
                fig.colorbar(img, ax=ax)
                out_file = output_dir / f"weight_heatmap_k{k}_eps{eps}_{wf.stem}.{fmt}"
                metadata = {
                    'figure': str(out_file.name),
                    'weights_file': str(wf),
                    'k': k,
                    'eps': eps,
                }
                save_figure_and_metadata(fig, out_file, metadata, fmt, dpi)
                plt.close(fig)
                plotted += 1

# --------------------------------- MAIN ---------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Generate publication figures (fig_1 variants)")
    parser.add_argument('--manifest-root', default='final1', help='Root directory where manifests and archives live (default: final1)')
    parser.add_argument('--ks', nargs='+', default=K_VALUES, help='List of k values (transfer samples)')
    parser.add_argument('--eps', nargs='+', default=EPS_VALUES, help='List of epsilons (privacy)')
    parser.add_argument('--plot-types', nargs='+', default=['transfer_sweep', 'privacy_sweep', 'heatmap'], help='Which plot types to generate')
    parser.add_argument('--methods', nargs='+', default=None, help='List of methods to plot (one or more)')
    parser.add_argument('--heatmap-baseline', default=None, help='Baseline method to compute delta heatmaps')
    parser.add_argument('--config', default='config.yaml', help='Path to config.yaml for default baseline')
    parser.add_argument('--output-dir', default=str(OUTPUT_DIR), help='Output directory to write figures')
    parser.add_argument('--format', default='png', choices=['png', 'pdf'], help='Figure output format')
    parser.add_argument('--dpi', default=FIG_DPI, type=int, help='Figure DPI')
    parser.add_argument('--dry-run', action='store_true', help='Do not create figures; only report which manifests would be used')
    parser.add_argument('--skip-missing-manifests', action='store_true', help='Skip missing manifests instead of failing')
    parser.add_argument('--k-for-privacy', default=None, type=int, help='k value to hold fixed for the privacy-sweep figure')
    args = parser.parse_args(argv)
    return args
# Duplicate/incorrect main() removed — the real main that constructs manifests_map,
# ks, eps_values and baseline is implemented below.
def main(argv=None):
    args = parse_args(argv)

    # set evaluate ks and eps
    ks = [int(x) for x in args.ks]
    # bless eps: convert strings 'inf' to float('inf') where appropriate
    def parse_eps(x):
        try:
            if str(x).lower() in {'inf', 'infty', '∞'}:
                return float('inf')
            return float(x)
        except Exception:
            return x
    eps_values = [parse_eps(e) for e in args.eps]

    # Determine baseline
    baseline = args.heatmap_baseline
    baseline_source = 'cli' if baseline else None
    if baseline is None:
        baseline = viz_get_default_baseline(args.config)
        baseline_source = 'config' if baseline else None

    # collect manifests via visualization helper (this returns k -> eps token -> path or None)
    manifests_map = viz_collect_manifests(args.manifest_root, ks, [str(e) for e in args.eps], skip_missing=args.skip_missing_manifests)

    # ensure our local find_manifest_path() searches the manifest root first
    try:
        rootp = Path(args.manifest_root)
        if rootp.exists():
            # insert the candidate path as highest priority so local load_manifest() uses the correct root
            CANDIDATE_MANIFEST_DIRS.insert(0, rootp)
    except Exception:
        pass

    # Convert to a simple map: k -> eps token -> path
    out_dir = Path(args.output_dir)
    # update global output directory used by other helper functions that default to OUTPUT_DIR
    global OUTPUT_DIR, COMPOSITE_SUBDIR
    OUTPUT_DIR = out_dir
    COMPOSITE_SUBDIR = OUTPUT_DIR / "figure_1_composites"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    COMPOSITE_SUBDIR.mkdir(parents=True, exist_ok=True)

    # If dry run, report manifests we would use and exit
    if args.dry_run:
        print("Dry run: manifest map summary:")
        for k, d in manifests_map.items():
            found = {eps: (str(p) if p else None) for eps, p in d.items()}
            print(f"k={k}: {found}")
        print("Done (dry-run). To generate figures run the script without --dry-run and with --manifest-root pointing at your results directory (e.g. --manifest-root final2)")
        return

    # Resolve the list of methods for heatmaps if not provided
    methods = args.methods
    if methods is None:
        # extract candidate methods from the first non-empty manifest
        candidate_methods = None
        for kmap in manifests_map.values():
            for manifest_path in kmap.values():
                if manifest_path:
                    m = viz_load_manifest(manifest_path)
                    if m and m.get('logs_by_algorithm'):
                        candidate_methods = list(m.get('logs_by_algorithm').keys())
                        break
            if candidate_methods:
                break
        if candidate_methods:
            methods = candidate_methods
        else:
            methods = ["FedAvg", "FedProx", "FedAvg_noDP", "Centralized_PLS"]

    # Which k to use for privacy sweep (prefer CLI override, otherwise the hard-coded default)
    k_for_privacy = args.k_for_privacy if args.k_for_privacy is not None else K_FOR_PRIVACY
    # If a user accidentally supplies a list (e.g., from incorrect parsing), handle defensively
    if isinstance(k_for_privacy, (list, tuple)):
        if len(k_for_privacy) >= 1:
            k_for_privacy = int(k_for_privacy[0])
        else:
            k_for_privacy = K_FOR_PRIVACY
    # Coerce to int if provided as a string
    try:
        k_for_privacy = int(k_for_privacy)
    except Exception:
        k_for_privacy = K_FOR_PRIVACY
    # Warn if the requested k is not present in the provided ks list
    if k_for_privacy not in ks:
        print(f"⚠️  Warning: requested k_for_privacy={k_for_privacy} not in ks={ks}; proceeding but results may be empty.")


    # Build requested plot types
    plot_types = [p.lower() for p in args.plot_types]
    if 'transfer_sweep' in plot_types:
        for eps in eps_values:
            try:
                create_transfer_size_impact_figure(epsilon=eps)
                print(f"Created transfer-sweep figure for eps={eps}")
            except Exception as e:
                print(f"Failed to create transfer-sweep eps={eps}: {e}")

    if 'privacy_sweep' in plot_types:
        try:
            create_privacy_impact_figure(k_selected=k_for_privacy)
            print(f"Created privacy-sweep figure for k={k_for_privacy}")
        except Exception as e:
            print(f"Failed to create privacy-sweep for k={k_for_privacy}: {e}")

    if 'heatmap' in plot_types:
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            for method in methods:
                print(f"Creating RMSEP heatmap for method: {method}")
                create_rmsep_heatmap(manifests_map, method, baseline, out_dir, fmt=args.format, dpi=args.dpi)
        except Exception as e:
            print(f"Failed to create heatmap(s): {e}")

    if 'weights' in plot_types or 'weight' in plot_types:
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            create_weight_heatmaps(manifests_map, out_dir, fmt=args.format, dpi=args.dpi)
        except Exception as e:
            print(f"Failed to create weight heatmaps: {e}")

    print("Done: figure generation script completed.")


if __name__ == "__main__":
    main()
