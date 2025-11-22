from matplotlib import pyplot as plt
import numpy as np
from typing import List


def plot_rmsep_heatmap(ax, rmsep_matrix: np.ndarray, k_labels: List[str], eps_labels: List[str], show_annotations: bool = True, cmap: str = 'viridis'):
    """Plot a heatmap for RMSEP values.
    rmsep_matrix should be shape (len(k_labels), len(eps_labels))
    """
    img = ax.imshow(rmsep_matrix, aspect='auto', interpolation='nearest', cmap=cmap)
    ax.set_xticks(np.arange(len(eps_labels)))
    ax.set_yticks(np.arange(len(k_labels)))
    ax.set_xticklabels(eps_labels)
    ax.set_yticklabels(k_labels)
    ax.set_xlabel('epsilon')
    ax.set_ylabel('transfer samples k')
    for edge, spine in ax.spines.items():
        spine.set_visible(False)
    if show_annotations and rmsep_matrix.size <= 100:
        for i in range(rmsep_matrix.shape[0]):
            for j in range(rmsep_matrix.shape[1]):
                v = rmsep_matrix[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.3f}", ha='center', va='center', color='white', fontsize=7)
    return img


def plot_weight_heatmap(ax, weight_matrix: np.ndarray, xlabels=None, ylabels=None, cmap: str = 'RdBu_r'):
    img = ax.imshow(weight_matrix, aspect='auto', interpolation='nearest', cmap=cmap)
    # axis labels
    if xlabels is not None:
        ax.set_xticks(np.arange(len(xlabels)))
        ax.set_xticklabels(xlabels, rotation=45, ha='right', fontsize=7)
    if ylabels is not None:
        ax.set_yticks(np.arange(len(ylabels)))
        ax.set_yticklabels(ylabels, fontsize=7)
    ax.set_xlabel('feature')
    ax.set_ylabel('client / round')
    return img


def convergence_plot_panel(ax, df_by_method, method_order, palette, log_y=False, **kwargs):
    # simple RMSEP vs round
    for method in method_order:
        df = df_by_method.get(method)
        if df is None or df.empty:
            continue
        ax.plot(df['round'], df['rmsep'], label=method, color=palette.get(method))
    if log_y:
        ax.set_yscale('log')
    ax.set_xlabel('Round')
    ax.set_ylabel('RMSEP')
    ax.legend()


def communication_plot_panel(ax, df_by_method, method_order, palette, show_cumulative=True, **kwargs):
    for method in method_order:
        df = df_by_method.get(method)
        if df is None or df.empty:
            continue
        ax.plot(df['round'], df['bytes_sent'].cumsum()/1024.0, label=method, color=palette.get(method))
    ax.set_xlabel('Round')
    ax.set_ylabel('Cumulative KB Sent')
    ax.legend()


def plot_participation_vs_rmsep(ax, df_by_method, method_order, palette, **kwargs):
    for method in method_order:
        df = df_by_method.get(method)
        if df is None or df.empty:
            continue
        # scatter participation (x) vs rmsep (y) using final round
        last = df.groupby('method').last().reset_index()
        ax.scatter(last['participation'], last['rmsep'], label=method, color=palette.get(method))
    ax.set_xlabel('Participation')
    ax.set_ylabel('Final RMSEP')
    ax.legend()
