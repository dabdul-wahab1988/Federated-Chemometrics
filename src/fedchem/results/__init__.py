"""Helpers for managing experiment outputs under the canonical ``results/`` tree."""

from .aggregator import (
    build_methods_summary,
    export_federated_results,
    export_methods_summary,
    load_raw_records,
    summarize_metrics,
)
from .manager import (
    ResultsTree,
    ensure_results_tree,
    manifest_to_raw_records,
    write_raw_records,
)

__all__ = [
    "build_methods_summary",
    "export_federated_results",
    "export_methods_summary",
    "load_raw_records",
    "summarize_metrics",
    "ResultsTree",
    "ensure_results_tree",
    "manifest_to_raw_records",
    "write_raw_records",
]
