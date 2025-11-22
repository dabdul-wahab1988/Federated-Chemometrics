from __future__ import annotations

from typing import Dict
import numpy as np


def summarize_interval(
    site_id: str,
    label: str,
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> Dict[str, float | str]:
    """Return coverage + width summary for a conformal interval."""
    y = np.asarray(y_true).ravel()
    lo = np.asarray(lower).ravel()
    hi = np.asarray(upper).ravel()
    coverage = float(np.mean((y >= lo) & (y <= hi)))
    width = float(np.mean(hi - lo))
    return {"site_id": site_id, "label": label, "coverage": coverage, "mean_width": width}
