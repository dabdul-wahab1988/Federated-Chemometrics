from __future__ import annotations

import numpy as np
from ..types import FloatArray


def rmsep(y_true: FloatArray, y_pred: FloatArray) -> float:
    if y_true.shape != y_pred.shape:
        raise ValueError("Shapes must match")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: FloatArray, y_pred: FloatArray) -> float:
    if y_true.shape != y_pred.shape:
        raise ValueError("Shapes must match")
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true: FloatArray, y_pred: FloatArray) -> float:
    if y_true.shape != y_pred.shape:
        raise ValueError("Shapes must match")
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - (ss_res / (ss_tot + 1e-12))


def coverage(y_true: FloatArray, lo: FloatArray, hi: FloatArray) -> float:
    if not (y_true.shape == lo.shape == hi.shape):
        raise ValueError("Shapes must match")
    inside = (y_true >= lo) & (y_true <= hi)
    return float(np.mean(inside))


def interval_width(lo: FloatArray, hi: FloatArray) -> float:
    if lo.shape != hi.shape:
        raise ValueError("Shapes must match")
    return float(np.mean(hi - lo))
