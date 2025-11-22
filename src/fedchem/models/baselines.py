from __future__ import annotations

import numpy as np
from .base import BaseModel
from ._common import ridge_lstsq


class PDSBaseline(BaseModel):
    """Piecewise Direct Standardization-inspired baseline.

    Splits the spectrum into windows and fits per-window linear models to y;
    predictions are averaged across windows. This mimics localized calibration
    and often improves robustness to shifts localized in the spectrum.
    """

    def __init__(self, window: int = 16, ridge: float = 1e-6) -> None:
        self.window = int(max(1, window))
        self.ridge = float(ridge)
        self.coefs_: list[np.ndarray] | None = None
        self.biases_: list[float] | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PDSBaseline":
        n, d = X.shape
        W = self.window
        self.coefs_ = []
        self.biases_ = []
        for start in range(0, d, W):
            stop = min(d, start + W)
            Xi = X[:, start:stop]
            # ridge-regularized least squares per window using shared helper
            w, b = ridge_lstsq(Xi, y, ridge=self.ridge, fit_intercept=True)
            self.coefs_.append(w)
            self.biases_.append(float(b))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coefs_ is None or self.biases_ is None:
            raise RuntimeError("Model not fitted")
        d = X.shape[1]
        W = self.window
        preds = []
        w_idx = 0
        for start in range(0, d, W):
            stop = min(d, start + W)
            Xi = X[:, start:stop]
            coef = self.coefs_[w_idx]
            bias = self.biases_[w_idx]
            preds.append(Xi @ coef + bias)
            w_idx += 1
        # average per-window predictions
        from typing import cast
        return cast(np.ndarray, np.mean(np.vstack(preds), axis=0))


class SlopeBiasCorrection(BaseModel):
    """Slope/Bias correction on top of a base linear fit.

    Fits a linear model and then calibrates the predictions via y ≈ a + b*yhat.
    """

    def __init__(self) -> None:
        self.w_: np.ndarray | None = None
        self.b_: float | None = None
        self.corr_slope_: float | None = None
        self.corr_bias_: float | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SlopeBiasCorrection":
        X_ = np.c_[X, np.ones((X.shape[0], 1))]
        coef = np.linalg.lstsq(X_, y, rcond=None)[0]
        self.w_ = coef[:-1]
        self.b_ = float(coef[-1])
        yhat = X @ self.w_ + self.b_
        A = np.c_[yhat, np.ones_like(yhat)]
        self.corr_slope_, self.corr_bias_ = np.linalg.lstsq(A, y, rcond=None)[0]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if any(v is None for v in (self.w_, self.b_, self.corr_slope_, self.corr_bias_)):
            raise RuntimeError("Model not fitted")
        assert self.w_ is not None and self.b_ is not None
        assert self.corr_slope_ is not None and self.corr_bias_ is not None
        from typing import cast
        yhat = cast(np.ndarray, X @ self.w_ + self.b_)
        return cast(np.ndarray, float(self.corr_slope_) * yhat + float(self.corr_bias_))


class DTWCT(BaseModel):
    """DTW-based calibration transfer.

    Warps each spectrum to a learned template via DTW, then fits a linear model
    on the warped representations.
    """

    def __init__(self, window: int | None = None) -> None:
        self.template_: np.ndarray | None = None
        self.w_: np.ndarray | None = None
        self.b_: float | None = None
        self.window = window  # Sakoe–Chiba band half-width in indices

    @staticmethod
    def _dtw_path(a: np.ndarray, b: np.ndarray, window: int | None = None) -> list[tuple[int, int]]:
        n, m = len(a), len(b)
        D = np.full((n + 1, m + 1), np.inf)
        D[0, 0] = 0.0
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if window is not None and abs(i - j) > window:
                    continue
                cost = (a[i - 1] - b[j - 1]) ** 2
                D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
        # backtrack
        path: list[tuple[int, int]] = []
        i, j = n, m
        while i > 0 and j > 0:
            path.append((i - 1, j - 1))
            steps = [(i - 1, j), (i, j - 1), (i - 1, j - 1)]
            i2, j2 = min(steps, key=lambda ij: D[ij])
            i, j = i2, j2
        path.reverse()
        return path

    @staticmethod
    def _warp_to_template(x: np.ndarray, template: np.ndarray, window: int | None = None) -> np.ndarray:
        path = DTWCT._dtw_path(x, template, window)
        # Map each template index to average of x indices aligned to it
        T = len(template)
        buckets: list[list[float]] = [[] for _ in range(T)]
        for i, j in path:
            buckets[j].append(x[i])
        out = np.zeros(T, dtype=float)
        for j in range(T):
            if buckets[j]:
                out[j] = float(np.mean(buckets[j]))
            else:
                # fallback: use template value proportionally
                out[j] = float(x[min(len(x) - 1, j)])
        return out

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DTWCT":
        self.template_ = X.mean(axis=0)
        # warp all to template
        template = self.template_
        assert template is not None  # for type checkers
        Xw = np.vstack([self._warp_to_template(x, template, self.window) for x in X])
        self.w_, self.b_ = ridge_lstsq(Xw, y, ridge=0.0, fit_intercept=True)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.template_ is None or self.w_ is None or self.b_ is None:
            raise RuntimeError("Model not fitted")
        template = self.template_
        assert template is not None
        Xw = np.vstack([self._warp_to_template(x, template, self.window) for x in X])
        from typing import cast
        return cast(np.ndarray, Xw @ self.w_ + self.b_)
