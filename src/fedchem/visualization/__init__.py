"""
Visualization package for fedchem
Expose helper APIs for plotting and I/O
"""
from .io import collect_manifests, load_manifest, find_debug_weights, get_default_baseline
from .table_utils import manifest_to_table, extract_final_rmsep
from .plots import (
    convergence_plot_panel,
    communication_plot_panel,
    plot_rmsep_heatmap,
    plot_weight_heatmap,
    plot_participation_vs_rmsep,
)
from .utils import save_figure_and_metadata, get_git_short_hash, human_readable_bytes
from .styles import set_plot_style

__all__ = [
    "collect_manifests",
    "load_manifest",
    "find_debug_weights",
    "get_default_baseline",
    "manifest_to_table",
    "extract_final_rmsep",
    "convergence_plot_panel",
    "communication_plot_panel",
    "plot_rmsep_heatmap",
    "plot_weight_heatmap",
    "plot_participation_vs_rmsep",
    "save_figure_and_metadata",
    "get_git_short_hash",
    "human_readable_bytes",
    "set_plot_style",
]
