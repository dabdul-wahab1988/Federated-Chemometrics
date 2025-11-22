from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple
import numpy as np
from ..types import FloatArray


def load_local_csv(path: str, *, has_header: bool = True, target_col: Optional[int] = None) -> Tuple[FloatArray, Optional[FloatArray]]:
    """Load a local CSV as spectra matrix (and optional target y).

    - If target_col is provided, that column is returned as y and removed from X.
    - Otherwise, returns (X, None).
    - Expects numeric values; header skipped if has_header.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    data = np.genfromtxt(p, delimiter=",", skip_header=1 if has_header else 0)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    y = None
    if target_col is not None:
        y = data[:, target_col]
        X = np.delete(data, target_col, axis=1)
    else:
        X = data
    return X.astype(np.float32), None if y is None else y.astype(np.float32)
