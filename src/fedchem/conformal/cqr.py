from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from sklearn.linear_model import QuantileRegressor
from ..types import FloatArray


@dataclass
class CQRState:
    q_lo: QuantileRegressor
    q_hi: QuantileRegressor
    calib_scores: FloatArray  # nonconformity scores on calibration set


class CQRConformal:
    """Conformalized Quantile Regression for interval prediction.

    Trains two quantile regressors at tau_lo and tau_hi on a train split.
    Uses a calibration split to compute nonconformity scores:
      s_i = max(q_lo(x_i) - y_i, y_i - q_hi(x_i))
    For a desired alpha, predicts intervals:
      [q_lo(x) - q_alpha, q_hi(x) + q_alpha], where q_alpha is the 1-alpha
    quantile of calibration scores.
    """

    def __init__(self, tau_lo: float = 0.1, tau_hi: float = 0.9, calib_fraction: float = 0.2, alpha_default: float = 0.1) -> None:
        assert 0 < tau_lo < tau_hi < 1
        assert 0 < calib_fraction < 1
        self.tau_lo = tau_lo
        self.tau_hi = tau_hi
        self.calib_fraction = calib_fraction
        self.alpha_default = alpha_default
        self._state: Optional[CQRState] = None

    def fit(self, X: FloatArray, y: FloatArray) -> "CQRConformal":
        n = X.shape[0]
        n_cal = max(1, int(self.calib_fraction * n))
        idx = np.arange(n)
        # simple split: last n_cal rows for calibration
        train_idx, cal_idx = idx[:-n_cal], idx[-n_cal:]
        Xtr, ytr = X[train_idx], y[train_idx]
        Xc, yc = X[cal_idx], y[cal_idx]

        qlo = QuantileRegressor(quantile=self.tau_lo, alpha=0.0, solver="highs").fit(Xtr, ytr)
        qhi = QuantileRegressor(quantile=self.tau_hi, alpha=0.0, solver="highs").fit(Xtr, ytr)
        lo_hat = qlo.predict(Xc)
        hi_hat = qhi.predict(Xc)
        scores = np.maximum(lo_hat - yc, yc - hi_hat)
        self._state = CQRState(q_lo=qlo, q_hi=qhi, calib_scores=scores)
        return self

    def predict_interval(self, X: FloatArray, alpha: float | None = None) -> Tuple[FloatArray, FloatArray]:
        if self._state is None:
            raise RuntimeError("CQR not fitted")
        a = self.alpha_default if alpha is None else alpha
        q_alpha = float(np.quantile(self._state.calib_scores, 1 - a, method="higher"))
        lo = self._state.q_lo.predict(X) - q_alpha
        hi = self._state.q_hi.predict(X) + q_alpha
        from typing import cast
        return cast(FloatArray, lo), cast(FloatArray, hi)


class MondrianCQR(CQRConformal):
    """Mondrian CQR with conditioning by group labels (e.g., site/instrument)."""

    def fit(self, X: FloatArray, y: FloatArray, groups: np.ndarray | None = None) -> "MondrianCQR":
        if groups is None:
            super().fit(X, y)  # fallback to global
            return self
        uniq = np.unique(groups)
        self._group_states: dict[int, CQRState] = {}
        for g in uniq:
            mask = groups == g
            sub = super(MondrianCQR, self).__new__(CQRConformal)
            CQRConformal.__init__(sub, self.tau_lo, self.tau_hi, self.calib_fraction, self.alpha_default)
            sub.fit(X[mask], y[mask])
            assert sub._state is not None
            self._group_states[int(g)] = sub._state
        return self

    def predict_interval(self, X: FloatArray, groups: np.ndarray | None = None, alpha: float | None = None) -> Tuple[FloatArray, FloatArray]:  # type: ignore[override]
        if groups is None or not hasattr(self, "_group_states"):
            return super().predict_interval(X, alpha)
        lo = np.empty((X.shape[0],), dtype=float)
        hi = np.empty((X.shape[0],), dtype=float)
        for g in np.unique(groups):
            mask = groups == g
            state = self._group_states[int(g)]
            a = self.alpha_default if alpha is None else alpha
            q_alpha = float(np.quantile(state.calib_scores, 1 - a, method="higher"))
            lo[mask] = state.q_lo.predict(X[mask]) - q_alpha
            hi[mask] = state.q_hi.predict(X[mask]) + q_alpha
        from typing import cast
        return cast(FloatArray, lo), cast(FloatArray, hi)
