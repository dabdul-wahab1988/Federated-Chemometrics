from __future__ import annotations

from typing import Dict, Iterable, List, Sequence
import numpy as np

from fedchem.metrics.metrics import mae, rmsep, r2, cvrmsep

# Default metric ordering used in publication tables
DEFAULT_METRICS: Sequence[str] = ("rmsep", "cvrmsep", "mae", "r2")

_METRIC_FUNCS = {
    "rmsep": rmsep,
    "cvrmsep": cvrmsep,
    "mae": mae,
    "r2": r2,
}


def build_method_rows(
    site_id: str,
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    metrics: Iterable[str] | None = None,
) -> List[Dict[str, float | str]]:
    """Return per-method metric rows for a single site.

    Parameters
    ----------
    site_id: identifier for the local site (e.g., ``site_1``)
    y_true: ground-truth responses for the site's local test split
    predictions: mapping of method name -> prediction vector
    metrics: iterable of metric names (subset of DEFAULT_METRICS)
    """
    metric_list = list(DEFAULT_METRICS if metrics is None else metrics)
    rows: List[Dict[str, float | str]] = []
    y = np.asarray(y_true).ravel()
    for method, preds in predictions.items():
        preds_arr = np.asarray(preds).ravel()
        row: Dict[str, float | str] = {"site_id": site_id, "method": method}
        for metric in metric_list:
            func = _METRIC_FUNCS.get(metric)
            if func is None:
                raise ValueError(f"Unknown metric '{metric}'")
            row[metric] = float(func(y, preds_arr))
        rows.append(row)
    return rows

