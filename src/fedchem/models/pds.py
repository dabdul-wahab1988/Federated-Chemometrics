from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass
class PDSMapper:
    """Piecewise Direct Standardization transfer mapper.

    Learns a blockwise linear mapping T such that X_target @ T ≈ X_source
    using a paired transfer set. Optionally supports block overlap.
    """

    window: int = 16
    overlap: int = 0
    ridge: float = 1e-6

    def fit(self, X_source: np.ndarray, X_target: np.ndarray) -> "PDSMapper":
        if X_source.shape != X_target.shape:
            raise ValueError("Source/Target shapes must match for paired transfer set")
        n, d = X_source.shape
        W = self.window
        overlap_amt = max(0, self.overlap)
        self._blocks: list[Tuple[int, int, np.ndarray]] = []
        start = 0
        while start < d:
            stop = min(d, start + W)
            Xs = X_source[:, start:stop]
            Xt = X_target[:, start:stop]
            # Solve Xt @ T ≈ Xs => T = argmin ||Xt T - Xs|| + ridge ||T||^2
            A = Xt.T @ Xt + self.ridge * np.eye(Xt.shape[1])
            T = np.linalg.solve(A, Xt.T @ Xs)
            self._blocks.append((start, stop, T))
            if stop == d:
                break
            start = stop - overlap_amt if overlap_amt > 0 else stop
        return self

    def transform(self, X_target: np.ndarray) -> np.ndarray:
        if not hasattr(self, "_blocks"):
            raise RuntimeError("PDSMapper not fitted")
        out = np.zeros_like(X_target)
        weight = np.zeros_like(X_target)
        for start, stop, T in self._blocks:
            Xt = X_target[:, start:stop]
            Xs_hat = Xt @ T
            out[:, start:stop] += Xs_hat
            weight[:, start:stop] += 1.0
        weight[weight == 0] = 1.0
        return out / weight
