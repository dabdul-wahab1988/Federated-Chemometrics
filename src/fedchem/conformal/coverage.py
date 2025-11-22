from typing import Tuple
import math
import numpy as np


class ConformalPredictor:
    """Simple split-conformal predictor using absolute residuals.

    Usage: fit on calibration residuals (abs errors) then call predict_interval
    on new point predictions.
    """

    def __init__(self, alpha: float = 0.1):
        if not (0 < alpha < 1):
            raise ValueError("alpha must be in (0,1)")
        self.alpha = float(alpha)
        self.q_hat: float | None = None

    def fit(self, residuals: np.ndarray) -> "ConformalPredictor":
        """Compute quantile from calibration residuals.

        Uses the classic split-conformal quantile: q = quantile(residuals, ceil((n+1)*(1-alpha))/n).
        """
        residuals = np.asarray(residuals)
        if residuals.ndim != 1:
            residuals = residuals.ravel()
        n = residuals.shape[0]
        if n <= 0:
            raise ValueError("residuals must contain at least one value for calibration")
        k = math.ceil((n + 1) * (1.0 - self.alpha))
        q = k / n
        # np.quantile expects a float in [0,1]
        self.q_hat = float(np.quantile(residuals, q))
        return self

    def predict_interval(self, predictions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return lower and upper bounds for given predictions."""
        if self.q_hat is None:
            raise RuntimeError("ConformalPredictor not fitted")
        preds = np.asarray(predictions)
        lower = preds - self.q_hat
        upper = preds + self.q_hat
        return lower, upper

    def compute_coverage(self, y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
        y = np.asarray(y_true)
        return float(np.mean((y >= lower) & (y <= upper)))
