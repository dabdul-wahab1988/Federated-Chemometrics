from __future__ import annotations

import numpy as np


class SlopeBiasCalibrator:
    """Post-hoc slope/bias calibrator for any base model.

    Fit on a transfer set: predictions vs true y, then calibrate predictions.
    """

    def __init__(self) -> None:
        self.slope_: float | None = None
        self.bias_: float | None = None

    def fit(self, y_pred: np.ndarray, y_true: np.ndarray) -> "SlopeBiasCalibrator":
        A = np.c_[y_pred, np.ones_like(y_pred)]
        self.slope_, self.bias_ = np.linalg.lstsq(A, y_true, rcond=None)[0]
        return self

    def transform(self, y_pred: np.ndarray) -> np.ndarray:
        if self.slope_ is None or self.bias_ is None:
            raise RuntimeError("Calibrator not fitted")
        return float(self.slope_) * y_pred + float(self.bias_)

