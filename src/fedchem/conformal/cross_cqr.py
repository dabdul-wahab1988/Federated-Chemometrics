from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
from sklearn.linear_model import QuantileRegressor
from ..types import FloatArray


@dataclass
class _FoldModel:
    q_lo: QuantileRegressor
    q_hi: QuantileRegressor


class CrossCQRConformal:
    """K-fold cross-conformalized quantile regression.

    Trains K pairs of quantile models; each fold provides calibration scores
    on its hold-out split. Predictions average fold quantiles and inflate by
    pooled calibration quantile for target alpha.
    """

    def __init__(self, k_folds: int = 5, tau_lo: float = 0.1, tau_hi: float = 0.9, alpha_default: float = 0.1, shuffle: bool = True, seed: Optional[int] = None) -> None:
        assert 1 < k_folds
        assert 0 < tau_lo < tau_hi < 1
        assert 0 < alpha_default < 1
        self.k = int(k_folds)
        self.tau_lo = tau_lo
        self.tau_hi = tau_hi
        self.alpha_default = alpha_default
        self.shuffle = shuffle
        self.seed = seed
        self._folds: List[_FoldModel] | None = None
        self._scores: FloatArray | None = None

    def fit(self, X: FloatArray, y: FloatArray) -> "CrossCQRConformal":
        n = X.shape[0]
        idx = np.arange(n)
        if self.shuffle:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(idx)
        folds = np.array_split(idx, self.k)
        models: List[_FoldModel] = []
        scores = []
        for i in range(self.k):
            val_idx = folds[i]
            tr_idx = np.concatenate([folds[j] for j in range(self.k) if j != i])
            Xtr, ytr = X[tr_idx], y[tr_idx]
            Xv, yv = X[val_idx], y[val_idx]
            qlo = QuantileRegressor(quantile=self.tau_lo, alpha=0.0, solver="highs").fit(Xtr, ytr)
            qhi = QuantileRegressor(quantile=self.tau_hi, alpha=0.0, solver="highs").fit(Xtr, ytr)
            lo_hat = qlo.predict(Xv)
            hi_hat = qhi.predict(Xv)
            s = np.maximum(lo_hat - yv, yv - hi_hat)
            scores.append(s)
            models.append(_FoldModel(q_lo=qlo, q_hi=qhi))
        self._folds = models
        self._scores = np.hstack(scores)
        return self

    def predict_interval(self, X: FloatArray, alpha: float | None = None) -> Tuple[FloatArray, FloatArray]:
        if self._folds is None or self._scores is None:
            raise RuntimeError("Model not fitted")
        a = self.alpha_default if alpha is None else alpha
        q_alpha = float(np.quantile(self._scores, 1 - a, method="higher"))
        lo_preds = [m.q_lo.predict(X) for m in self._folds]
        hi_preds = [m.q_hi.predict(X) for m in self._folds]
        lo = np.mean(np.vstack(lo_preds), axis=0) - q_alpha
        hi = np.mean(np.vstack(hi_preds), axis=0) + q_alpha
        from typing import cast
        return cast(FloatArray, lo), cast(FloatArray, hi)

    @staticmethod
    def coverage_by_group(y: FloatArray, lo: FloatArray, hi: FloatArray, groups: np.ndarray) -> dict:
        cov = {}
        for g in np.unique(groups):
            m = groups == g
            cov[int(g)] = float(np.mean((y[m] >= lo[m]) & (y[m] <= hi[m])))
        return cov
