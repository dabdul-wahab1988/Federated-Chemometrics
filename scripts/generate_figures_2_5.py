"""
Script to generate Figures 2, 3, 4, and 5 from the master JSON database.
Saves high-res PNGs (300 DPI) and handles missing values gracefully.

Usage:
    python scripts/generate_figures_2_5.py --input master_database_20251120_200830.json --outdir generated_figures_tables

This script uses pandas, matplotlib, and seaborn.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional, Sequence, Dict, Any

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("generate_figures_2_5")

# Seaborn style
sns.set(style="whitegrid")

# The list of expected methods
METHODS = [
    "Centralized_PLS",
    "Site_Specific",
    "PDS",
    "SBC",
    "FedPLS",
    "FedProx",
    "Global_calibrate_after_fed",
]

# Colors (a consistent palette)
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


def load_json_to_df(path: Path) -> pd.DataFrame:
    """Load a JSON file into a Pandas DataFrame while handling nested top-level keys.

    The function tries common structures: list-of-dicts, or top-level dict with 'results'/'data'/'rows'.
    """
    logger.info(f"Loading data from {path}")
    raw = None
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)

    # Common patterns
    if isinstance(raw, list):
        df = pd.DataFrame(raw)
    elif isinstance(raw, dict):
        # Try discover key
        for k in ("results", "data", "rows", "items"):  # heuristics
            if k in raw and isinstance(raw[k], list):
                df = pd.DataFrame(raw[k])
                break
        else:
            # If dict of lists or mapping of instrument->list, try flatten
            try:
                df = pd.DataFrame(raw)
            except Exception:
                raise ValueError("Unsupported JSON structure. Expecting a top-level list-of-dicts or a key like 'results'.")
    else:
        raise ValueError("Unsupported JSON structure; expected dict or list")

    logger.info(f"Loaded DataFrame with shape: {df.shape}")
    return df


def validate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure required columns exist. If not, try light mapping or raise informative error."""
    required = [
        "method",
        "rmsep",
        "dp_epsilon",
        "transfer_k",
        "instrument_code",
        "Global_Coverage",
        "Global_MeanWidth",
    ]

    missing = [c for c in required if c not in df.columns]
    if not missing:
        return df

    # Try simple remappings that occur in our repo
    colmap = {}
    if "metrics.RMSEP" in df.columns and "rmsep" not in df.columns:
        colmap["metrics.RMSEP"] = "rmsep"
    if "design_factors.DP_Target_Eps" in df.columns and "dp_epsilon" not in df.columns:
        colmap["design_factors.DP_Target_Eps"] = "dp_epsilon"
    if "design_factors.Transfer_Samples" in df.columns and "transfer_k" not in df.columns:
        colmap["design_factors.Transfer_Samples"] = "transfer_k"
    if "instrument_code" not in df.columns and "instrument" in df.columns:
        colmap["instrument"] = "instrument_code"
    if "metrics.Global_Coverage" in df.columns and "Global_Coverage" not in df.columns:
        colmap["metrics.Global_Coverage"] = "Global_Coverage"
    if "metrics.Global_MeanWidth" in df.columns and "Global_MeanWidth" not in df.columns:
        colmap["metrics.Global_MeanWidth"] = "Global_MeanWidth"

    if colmap:
        df = df.rename(columns=colmap)
        missing = [c for c in required if c not in df.columns]

    if missing:
        logger.warning(f"Missing columns after attempts to map: {missing}. Some figures may be incomplete.")

    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess dataset: types and dp_epsilon conversions."""
    df = validate_columns(df)

    # Convert dp_epsilon 'inf' string to np.inf and numeric conversion otherwise
    if "dp_epsilon" in df.columns:
        def parse_eps(x):
            if pd.isna(x):
                return np.nan
            try:
                if isinstance(x, str) and x.lower() in {"inf", "infinity", "∞"}:
                    return np.inf
                return float(x)
            except Exception:
                try:
                    return float(str(x))
                except Exception:
                    return np.nan
        df["dp_epsilon"] = df["dp_epsilon"].apply(parse_eps)

    # Ensure transfer_k numeric
    if "transfer_k" in df.columns:
        df["transfer_k"] = pd.to_numeric(df["transfer_k"], errors="coerce")

    # Ensure rmsep numeric
    if "rmsep" in df.columns:
        df["rmsep"] = pd.to_numeric(df["rmsep"], errors="coerce")

    # Convert method into categorical and filter to known methods where possible
    if "method" in df.columns:
        # map common variants
        df["method"] = df["method"].astype(str)

    # Remove rows without instrument or method or rmsep
    df = df.dropna(subset=[c for c in ("instrument_code", "method", "rmsep") if c in df.columns])

    logger.info(f"After preprocessing, DataFrame shape: {df.shape}")
    return df


def collect_round_logs(manifest_root: Path) -> pd.DataFrame:
    """Collect per-round logs from generated manifests in `final2/generated_figures_tables_archive`.

    Returns a DataFrame with columns: method, round, rmsep, transfer_k, dp_epsilon, manifest_path
    """
    manifest_root = Path(manifest_root)
    if not manifest_root.exists():
        return pd.DataFrame()

    rows = []
    for mfile in manifest_root.rglob('manifest_*.json'):
        try:
            with open(mfile, 'r', encoding='utf-8') as fh:
                manifest = json.load(fh)
        except Exception:
            continue

        # Metadata
        cfg = manifest.get('config', {})
        transfer_k = cfg.get('transfer_samples_used') or cfg.get('transfer_samples_requested') or cfg.get('max_transfer_samples')
        dp_epsilon = cfg.get('target_epsilon') if 'target_epsilon' in cfg else None
        # Some manifests encode epsilon into strings or under DIFF_PRIV settings
        if dp_epsilon is None and 'DIFFERENTIAL_PRIVACY' in cfg:
            dp_epsilon = cfg.get('DIFFERENTIAL_PRIVACY', {}).get('target_eps') or cfg.get('target_epsilon')

        logs_by_algo = manifest.get('logs_by_algorithm') or manifest.get('logs') or {}
        if isinstance(logs_by_algo, dict):
            for method, logs in logs_by_algo.items():
                for l in logs:
                    row = {
                        'method': method,
                        'round': l.get('round'),
                        'rmsep': l.get('rmsep'),
                        'transfer_k': transfer_k,
                        'dp_epsilon': dp_epsilon,
                        'manifest': str(mfile)
                    }
                    rows.append(row)
    if not rows:
        return pd.DataFrame()

    df_rounds = pd.DataFrame(rows)
    # Clean types
    df_rounds['round'] = pd.to_numeric(df_rounds['round'], errors='coerce')
    df_rounds['rmsep'] = pd.to_numeric(df_rounds['rmsep'], errors='coerce')
    df_rounds['transfer_k'] = pd.to_numeric(df_rounds['transfer_k'], errors='coerce')
    # If dp_epsilon None, try to parse from manifest path tokens like 'eps_inf' or 'eps_1_0'
    def eps_from_str(x):
        if pd.isna(x):
            return np.nan
        try:
            return float(x)
        except Exception:
            s = str(x)
            if 'inf' in s or '∞' in s:
                return np.inf
            return np.nan
    df_rounds['dp_epsilon'] = df_rounds['dp_epsilon'].apply(eps_from_str)
    # If dp_epsilon is still NaN, try from manifest path with fillna to avoid dtype issues
    df_rounds['dp_epsilon'] = df_rounds['dp_epsilon'].fillna(df_rounds['manifest'].apply(lambda p: _parse_eps_from_path(p)))
    logger.info(f"Collected {len(df_rounds)} round-log entries from manifests; methods: {df_rounds['method'].unique()}")
    return df_rounds


def _parse_eps_from_path(path_str: str):
    """Parse epsilon values from path tokens like eps_0_1, eps_inf, eps_10_0"""
    p = Path(path_str)
    for part in p.parts:
        if part.startswith('eps_'):
            token = part.split('eps_')[-1]
            token = token.replace('_', '.')
            if token == 'inf' or token == '∞':
                return np.inf
            try:
                return float(token)
            except Exception:
                return np.nan
    return np.nan


# ----------------------- Figure 2: Baselines & CT -----------------------

def figure_2_baselines_ct(df: pd.DataFrame, outpath: Path, dpi: int = 300, show: bool = False) -> None:
    """Generate Figure 2 with Panel A (bar by instrument) and Panel B (learning curves).

    Per instruction: Panel A compares Centralized_PLS (gold), Site_Specific, FedPLS (inf,k=200), PDS (k=200), SBC (k=200) grouped by instrument.
    Panel B shows rmsep vs transfer_k for Site_Specific, PDS, SBC.
    """
    required_methods = ["Centralized_PLS", "Site_Specific", "FedPLS", "PDS", "SBC"]

    # Panel A: group by instrument and method -> mean rmsep
    df_panel = df.copy()

    # Fetch FedPLS (non-private baseline at dp_epsilon inf and k=200)
    fed_baseline = df_panel[(df_panel["method"] == "FedPLS") & (df_panel["dp_epsilon"] == np.inf) & (df_panel["transfer_k"] == 200)]
    pds_200 = df_panel[(df_panel["method"] == "PDS") & (df_panel["transfer_k"] == 200)]
    sbc_200 = df_panel[(df_panel["method"] == "SBC") & (df_panel["transfer_k"] == 200)]

    # We'll create a combined DataFrame for all methods for k=200 where applicable
    # For Centralized_PLS and Site_Specific, we take the overall mean RMSEP (don't depend on k)
    central = df_panel[df_panel["method"] == "Centralized_PLS"].groupby("instrument_code")["rmsep"].mean().rename("Centralized_PLS")
    site = df_panel[df_panel["method"] == "Site_Specific"].groupby("instrument_code")["rmsep"].mean().rename("Site_Specific")
    fed = fed_baseline.groupby("instrument_code")["rmsep"].mean().rename("FedPLS") if not fed_baseline.empty else pd.Series(dtype=float)
    pds = pds_200.groupby("instrument_code")["rmsep"].mean().rename("PDS") if not pds_200.empty else pd.Series(dtype=float)
    sbc = sbc_200.groupby("instrument_code")["rmsep"].mean().rename("SBC") if not sbc_200.empty else pd.Series(dtype=float)

    combined = pd.concat([central, site, fed, pds, sbc], axis=1)

    # Panel B: Learning curves (mean across instruments)
    lc_methods = ["Site_Specific", "PDS", "SBC"]
    lc_df = df_panel[df_panel["method"].isin(lc_methods)]
    curve = lc_df.groupby(["method", "transfer_k"])["rmsep"].mean().reset_index()

    # Setup figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Panel A: grouped bars per instrument
    ax = axes[0]
    combined_plot = combined.copy()
    combined_plot = combined_plot.dropna(how="all")  # drop instruments where all NaN

    if combined_plot.empty:
        ax.text(0.5, 0.5, "No data for requested baselines", ha="center", va="center")
        ax.set_axis_off()
    else:
        instruments = combined_plot.index.tolist()
        x = np.arange(len(instruments))
        width = 0.14
        n_methods = len(combined_plot.columns)
        offsets = (np.arange(n_methods) - (n_methods - 1) / 2) * width

        for i, col in enumerate(combined_plot.columns):
            ax.bar(x + offsets[i], combined_plot[col].fillna(np.nan), width=width, label=col,
                   color=METHOD_COLORS.get(col, None))

        ax.set_xticks(x)
        ax.set_xticklabels(instruments, rotation=45, ha="right")
        ax.set_ylabel("RMSEP")
        ax.set_title("Panel A: Baseline & CT per Instrument (k=200 for Transfer/CT)")
        ax.legend(fontsize=8)

    # Panel B: Learning curves
    ax2 = axes[1]
    # For central line (dashed), compute Centralized_PLS mean RMSEP (aggregate across instruments)
    if not central.empty:
        central_mean = central.mean()
        ax2.axhline(central_mean, linestyle='--', color=METHOD_COLORS.get('Centralized_PLS'), label='Centralized_PLS mean')

    if not curve.empty:
        for method in lc_methods:
            sub = curve[curve['method'] == method].sort_values('transfer_k')
            if not sub.empty:
                ax2.plot(sub['transfer_k'], sub['rmsep'], marker='o', label=method)

    ax2.set_xlabel('Transfer samples (k)')
    ax2.set_ylabel('Mean RMSEP')
    ax2.set_title('Panel B: Learning curves (Site-specific, PDS, SBC)')
    ax2.legend()

    plt.tight_layout()
    outpath.mkdir(parents=True, exist_ok=True)
    fname = outpath / 'Figure_2_Baselines_CT.png'
    fig.savefig(fname, dpi=dpi, bbox_inches='tight')
    logger.info(f"Saved Figure 2 to {fname}")
    if show:
        plt.show()
    plt.close(fig)


# ----------------------- Figure 3: Training Dynamics -----------------------

def figure_3_training_dynamics(df: pd.DataFrame, outpath: Path, dpi: int = 300, show: bool = False, rounds_df: Optional[pd.DataFrame] = None) -> None:
    """Produce Figure 3: federated training dynamics (rmsep vs round) and comparison of FedPLS vs FedProx.

    If round-level training data is not available, show placeholder text.
    """
    df_panel = df.copy()

    # Look for columns commonly used for training dynamics
    round_cols = [c for c in df_panel.columns if c in ("round", "num_round", "round_num")]

    # Check a separate file 'training_dynamics.csv' in workspace if present
    training_file = Path('training_dynamics.csv')
    df_rounds = rounds_df

    if round_cols and df_rounds is None:
        rc = round_cols[0]
        df_rounds = df_panel.dropna(subset=[rc, 'rmsep'])
        df_rounds[rc] = df_rounds[rc].astype(int)
    elif training_file.exists() and df_rounds is None:
        try:
            df_rounds = pd.read_csv(training_file)
            # try to ensure common columns exist
            if 'round' not in df_rounds.columns and 'num_round' in df_rounds.columns:
                df_rounds.rename(columns={'num_round': 'round'}, inplace=True)
        except Exception:
            df_rounds = None

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    if df_rounds is None or df_rounds.empty:
        # Placeholder
        axes[0].text(0.5, 0.5, 'No training dynamics (rounds) data available', ha='center', va='center')
        axes[0].set_axis_off()
        axes[1].text(0.5, 0.5, 'No training dynamics (rounds) data available', ha='center', va='center')
        axes[1].set_axis_off()
    else:
        # Panel A: rmsep vs round for Federated methods (mean across instruments)
        fed_methods = ['FedPLS', 'FedProx', 'FedAvg']
        ax = axes[0]
        for method in fed_methods:
            sub = df_rounds[df_rounds['method'] == method]
            if not sub.empty:
                grouped = sub.groupby('round')['rmsep'].agg(['mean'])
                ax.plot(grouped.index, grouped['mean'], marker='o', label=method)

        ax.set_xlabel('Round')
        ax.set_ylabel('RMSEP')
        ax.set_title('Panel A: Global RMSEP vs Round')
        ax.legend()

        # Panel B: FedPLS vs FedProx stability (mean ± std across instruments per round) at epsilon=inf
        ax2 = axes[1]
        for method in ['FedPLS', 'FedProx']:
            sub = df_rounds[(df_rounds['method'] == method) & (df_rounds['dp_epsilon'] == np.inf)]
            if not sub.empty:
                stats = sub.groupby('round')['rmsep'].agg(['mean', 'std']).reset_index()
                ax2.errorbar(stats['round'], stats['mean'], yerr=stats['std'], label=method, marker='o', capsize=3)

        ax2.set_xlabel('Round')
        ax2.set_ylabel('RMSEP (mean ± std)')
        ax2.set_title('Panel B: FedPLS vs FedProx Stability (ε = inf)')
        ax2.legend()

    plt.tight_layout()
    outpath.mkdir(parents=True, exist_ok=True)
    fname = outpath / 'Figure_3_Training_Dynamics.png'
    fig.savefig(fname, dpi=dpi, bbox_inches='tight')
    logger.info(f"Saved Figure 3 to {fname}")
    if show:
        plt.show()
    plt.close(fig)


# ----------------------- Figure 4: Privacy-Utility -----------------------

def figure_4_privacy_utility(df: pd.DataFrame, outpath: Path, method: str = 'FedPLS', dpi: int = 300, show: bool = False) -> None:
    """Generate Figure 4: privacy-utility heatmap and Pareto frontier for a single federated method (default FedPLS).
    """
    df_panel = df.copy()
    df_panel = df_panel[df_panel['method'] == method]

    if df_panel.empty:
        logger.warning(f"No rows found for method {method}. Skipping Figure 4.")
        # Generate placeholder
        fig, axs = plt.subplots(1, 2, figsize=(16, 6))
        axs[0].text(0.5,0.5, f'No data for {method}', ha='center', va='center')
        axs[1].text(0.5,0.5, f'No data for {method}', ha='center', va='center')
        for ax in axs:
            ax.set_axis_off()
        outpath.mkdir(parents=True, exist_ok=True)
        fname = outpath / 'Figure_4_Privacy_Utility.png'
        fig.savefig(fname, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        return

    # Build pivot for heatmap: rows: dp_epsilon, cols: transfer_k
    pivot = df_panel.groupby(['dp_epsilon', 'transfer_k'])['rmsep'].mean().reset_index()
    pivot_table = pivot.pivot(index='dp_epsilon', columns='transfer_k', values='rmsep')
    # Sort dp_epsilon so finite desc then inf
    eps_sorted = sorted([e for e in pivot_table.index if not np.isinf(e)])
    # Put np.inf at top or bottom; we want "  ,  " perhaps at bottom labelled 'inf'
    if np.inf in pivot_table.index:
        eps_sorted.append(np.inf)
    pivot_table = pivot_table.reindex(index=eps_sorted)

    # Panel A: heatmap
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    ax_heat = axs[0]

    sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax_heat, cbar_kws={'label': 'RMSEP'})
    ax_heat.set_title(f'Panel A: {method} RMSEP Heatmap (dp_epsilon x transfer_k)')
    ax_heat.set_xlabel('transfer_k')
    ax_heat.set_ylabel('dp_epsilon')
    ax_heat.set_yticklabels(["∞" if np.isinf(x) else f"{x}" for x in pivot_table.index])

    # Panel B: Pareto frontier (rmsep vs dp_epsilon) lines for k=20,80,200
    ax_pf = axs[1]
    ks = sorted(df_panel['transfer_k'].dropna().unique())
    ks_to_plot = [k for k in (20, 80, 200) if k in ks]

    if not ks_to_plot:
        # if none of the standard ks available, pick up to 3 unique ks
        ks_to_plot = list(ks)[:3]

    for k in ks_to_plot:
        sub = df_panel[df_panel['transfer_k'] == k]
        # mean rmsep per dp_epsilon
        pf = sub.groupby('dp_epsilon')['rmsep'].mean().reset_index()
        if pf.empty:
            continue
        # For plotting on log scale, transform np.inf to a large value
        pf_plot = pf.copy()
        pf_plot['eps_plot'] = pf_plot['dp_epsilon'].apply(lambda x: (1e6 if np.isinf(x) else x))
        ax_pf.plot(pf_plot['eps_plot'], pf_plot['rmsep'], marker='o', label=f'k={k}')
        # annotate sweet spot if dp_epsilon 1.0 or 10.0 present
        for target in (1.0, 10.0):
            sp = pf[pf['dp_epsilon'] == target]
            if not sp.empty:
                ax_pf.scatter([target], sp['rmsep'], color='black', s=90, zorder=5)
                ax_pf.annotate('Sweet spot' if target == 1.0 else '', (target, sp['rmsep'].iloc[0]), xytext=(8, -10), textcoords='offset points', fontsize=9)

    ax_pf.set_xscale('log')
    ax_pf.set_xlabel('dp_epsilon (log scale)')
    ax_pf.set_ylabel('RMSEP')
    ax_pf.set_title('Panel B: Privacy-Utility Frontier')
    ax_pf.legend(fontsize=9)

    plt.tight_layout()
    outpath.mkdir(parents=True, exist_ok=True)
    fname = outpath / 'Figure_4_Privacy_Utility.png'
    fig.savefig(fname, dpi=dpi, bbox_inches='tight')
    logger.info(f"Saved Figure 4 to {fname}")
    if show:
        plt.show()
    plt.close(fig)


# ----------------------- Figure 5: Conformal Prediction -----------------------

def figure_5_conformal(df: pd.DataFrame, outpath: Path, method: Optional[str] = None, dpi: int = 300, show: bool = False) -> None:
    """Generate Figure 5: Conformal prediction coverage and width vs dp_epsilon.

    If 'method' is provided, filter to it; otherwise plot aggregated across methods.
    """
    df_panel = df.copy()
    if method:
        df_panel = df_panel[df_panel['method'] == method]

    if df_panel.empty:
        logger.warning("No data for conformal plot.")
        fig, axs = plt.subplots(1, 2, figsize=(16, 6))
        for ax in axs: ax.set_axis_off()
        outpath.mkdir(parents=True, exist_ok=True)
        fname = outpath / 'Figure_5_Conformal.png'
        fig.savefig(fname, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        return

    # Prepare aggregated metrics per dp_epsilon
    agg = df_panel.groupby('dp_epsilon')[['Global_Coverage', 'Global_MeanWidth']].mean().reset_index()
    agg = agg.sort_values('dp_epsilon')

    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    # Panel A: coverage vs dp_epsilon
    ax1 = axs[0]
    agg_plot = agg.copy()
    agg_plot['eps_plot'] = agg_plot['dp_epsilon'].apply(lambda x: (1e6 if np.isinf(x) else x))
    ax1.plot(agg_plot['eps_plot'], agg_plot['Global_Coverage'], marker='o')
    ax1.axhline(0.95, color='red', linestyle='--', label='Target coverage (0.95)')
    ax1.set_xscale('log')
    ax1.set_xlabel('dp_epsilon (log scale)')
    ax1.set_ylabel('Global_Coverage')
    ax1.set_title('Panel A: Coverage vs Privacy')
    ax1.legend()

    # Panel B: mean width vs dp_epsilon
    ax2 = axs[1]
    ax2.plot(agg_plot['eps_plot'], agg_plot['Global_MeanWidth'], marker='o')
    ax2.set_xscale('log')
    ax2.set_xlabel('dp_epsilon (log scale)')
    ax2.set_ylabel('Global_MeanWidth')
    ax2.set_title('Panel B: Interval Width vs Privacy')

    plt.tight_layout()
    outpath.mkdir(parents=True, exist_ok=True)
    fname = outpath / 'Figure_5_Conformal.png'
    fig.savefig(fname, dpi=dpi, bbox_inches='tight')
    logger.info(f"Saved Figure 5 to {fname}")
    if show:
        plt.show()
    plt.close(fig)


# ----------------------- Runner -----------------------

def main():
    parser = argparse.ArgumentParser(description='Generate Figures 2-5 from a master JSON DB')
    parser.add_argument('--input', '-i', type=str, required=False, help='Path to master json file or to directory containing the json (defaults to final2/publication_database/json/)')
    parser.add_argument('--outdir', '-o', type=str, default='generated_figures_tables', help='Output directory for figures')
    parser.add_argument('--manifests', '-m', type=str, required=False, help='Path to generated_figures_tables_archive if you want round-level logs')
    parser.add_argument('--dpi', type=int, default=300, help='PNG DPI')
    parser.add_argument('--show', action='store_true', help='Show figures interactively (blocking)')
    args = parser.parse_args()

    # Default dataset folder in the workspace
    default_json_dir = Path('final2') / 'publication_database' / 'json'
    if args.input:
        path = Path(args.input)
    else:
        # Auto-discover: pick master_database_*.json in default_json_dir
        if default_json_dir.exists():
            candidates = sorted(list(default_json_dir.glob('master_database*.json')))
            if candidates:
                path = candidates[-1]
            else:
                # fallback to first .json in the folder
                c_all = sorted(list(default_json_dir.glob('*.json')))
                path = c_all[-1] if c_all else default_json_dir
        else:
            path = Path('.')

    # If path is a directory, attempt to find a master_*.json file within it
    if path.is_dir():
        candidates = sorted(list(path.glob('master_database*.json')))
        if candidates:
            path = candidates[-1]
        else:
            candidates = sorted(list(path.glob('*.json')))
            if candidates:
                path = candidates[-1]
            else:
                raise FileNotFoundError(f"No JSON files found in provided directory: {path}")

    df = load_json_to_df(path)
    df = preprocess(df)

    outdir = Path(args.outdir)

    figure_2_baselines_ct(df, outdir, dpi=args.dpi, show=args.show)
    # Collect round logs from manifests if available
    default_manifest_dir = Path('final2') / 'generated_figures_tables_archive'
    manifest_root = Path(args.manifests) if args.manifests else default_manifest_dir
    rounds_df = collect_round_logs(manifest_root) if manifest_root.exists() else pd.DataFrame()
    figure_3_training_dynamics(df, outdir, dpi=args.dpi, show=args.show, rounds_df=rounds_df)
    figure_4_privacy_utility(df, outdir, method='FedPLS', dpi=args.dpi, show=args.show)
    figure_5_conformal(df, outdir, method='FedPLS', dpi=args.dpi, show=args.show)

    logger.info("All figures generated successfully.")


if __name__ == '__main__':
    main()
