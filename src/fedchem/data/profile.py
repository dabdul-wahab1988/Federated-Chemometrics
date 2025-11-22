from __future__ import annotations

from typing import Dict, List
import numpy as np
from ..types import FloatArray


def compute_profile(X: FloatArray, wavelengths: List[float]) -> Dict[str, object]:
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-9
    x0 = X[0] - X[0].mean()
    ac1 = float(np.corrcoef(x0[:-1], x0[1:])[0, 1]) if x0.size > 1 else 0.0
    kernel = int(max(3, min(21, round(5 + 10 * max(0.0, ac1)))))
    return {
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        "wavelengths": list(wavelengths),
        "noise_std": float(np.median(std) * 0.05),
        "smooth_kernel": kernel,
    }
