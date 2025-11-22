from __future__ import annotations

from typing import Tuple
import numpy as np
from ..types import FloatArray


class ConformalPredictor:
    """Simple conformal predictor stub.

    Provides split-conformal style calibration producing symmetric intervals
    around point predictions using MAD-based scale.
    """

    def __init__(self) -> None:
        self._resid_q: float | None = None
        self._median: float | None = None

    def fit(self, X: FloatArray, y: FloatArray, method: str = "split", condition_by: str | None = None) -> "ConformalPredictor":
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same number of samples")
        # Stub: Use a simple central tendency on y to estimate residual scale
        self._median = float(np.median(y))
        resid = np.abs(y - self._median)
        self._resid_q = float(np.quantile(resid, 0.9))  # rough scale
        return self

    def predict_interval(self, X: FloatArray, alpha: float = 0.1) -> Tuple[FloatArray, FloatArray]:
        if self._resid_q is None or self._median is None:
            raise RuntimeError("ConformalPredictor not fitted")
        # Symmetric interval around median (placeholder)
        width = self._resid_q * (1.0 / max(1e-6, (1.0 - alpha)))
        lo = np.full((X.shape[0],), self._median - width, dtype=float)
        hi = np.full((X.shape[0],), self._median + width, dtype=float)
        from typing import cast
        return cast(FloatArray, lo), cast(FloatArray, hi)
