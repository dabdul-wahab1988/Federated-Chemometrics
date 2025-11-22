from __future__ import annotations

from typing import Optional
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import GridSearchCV

from .base import BaseModel
from ..types import FloatArray
from ..federated.types import ParametricModel
from ._common import build_pls_pipeline, pls_param_grid, make_kfold
import logging


class ParametricPLSModel(BaseModel, ParametricModel):
    """Federated-friendly PLS: exposes linear params (w, b) in original feature space.

    Internally fits a StandardScaler + PLSRegression (with optional CV over n_components),
    then derives the exact linear mapping y ≈ X @ w + b by probing the fitted pipeline on
    the standard basis. This enables parameter averaging in FL orchestrators.
    """

    def __init__(self, n_components: Optional[int] = None, max_components: int = 20, cv: int = 5, random_state: Optional[int] = None, use_global_scaler: bool = False) -> None:
        self.n_components = n_components
        self.max_components = int(max_components)
        self.cv = int(cv)
        self.random_state = random_state
        self.pipeline: Pipeline | None = None
        self.w: np.ndarray | None = None
        self.b: float | None = None
        self.use_global_scaler = bool(use_global_scaler)
        self.global_scaler_mean: np.ndarray | None = None
        self.global_scaler_scale: np.ndarray | None = None

    def _fit_pipeline(self, X: FloatArray, y: FloatArray) -> Pipeline:
        X = np.asarray(X)
        y = np.asarray(y)
        # If provided, use the global scaler to transform X before training.
        if self.use_global_scaler and self.global_scaler_mean is not None and self.global_scaler_scale is not None:
            gm = np.asarray(self.global_scaler_mean, dtype=float)
            gs = np.asarray(self.global_scaler_scale, dtype=float)
            if gm.shape[0] != X.shape[1] or gs.shape[0] != X.shape[1]:
                raise ValueError("Global scaler length does not match number of features.")
            Xp = (X - gm) / gs
        else:
            Xp = X
        if self.n_components is None:
            # If using a global scaler, we've already transformed Xp and should fit
            # a PLSRegression directly (no local scaler in pipeline).
            if self.use_global_scaler and self.global_scaler_mean is not None and self.global_scaler_scale is not None:
                from sklearn.cross_decomposition import PLSRegression
                pls = PLSRegression()
                param_grid = pls_param_grid(self.max_components, X.shape[1], X.shape[0])
                cv = make_kfold(self.cv, self.random_state)
                gs = GridSearchCV(pls, param_grid, scoring="neg_root_mean_squared_error", cv=cv)
                gs.fit(Xp, y)
                best_pls = gs.best_estimator_
                # Build pipeline with an identity scaler (we pre-scaled X)
                pipe = Pipeline([("scaler", FunctionTransformer(lambda x: x, validate=False)), ("pls", best_pls)])
                best = pipe
            else:
                pipe = build_pls_pipeline(None)
                param_grid = pls_param_grid(self.max_components, X.shape[1], X.shape[0])
                cv = make_kfold(self.cv, self.random_state)
                gs = GridSearchCV(pipe, param_grid, scoring="neg_root_mean_squared_error", cv=cv)
                gs.fit(X, y)
                best = gs.best_estimator_
            # Warn if selected n_components is large relative to sample size (may indicate overfit/ill-conditioning)
            try:
                n_sel = int(getattr(best.named_steps['pls'], 'n_components'))
                if X.shape[0] > 0 and n_sel >= max(2, X.shape[0] // 2):
                    logging.warning("PLS selected n_components=%s for n_samples=%s (may be unstable)", n_sel, X.shape[0])
            except Exception:
                pass
            return best
        else:
            if self.use_global_scaler and self.global_scaler_mean is not None and self.global_scaler_scale is not None:
                from sklearn.cross_decomposition import PLSRegression
                pls = PLSRegression(n_components=int(self.n_components))
                pls.fit(Xp, y)
                pipe = Pipeline([("scaler", FunctionTransformer(lambda x: x, validate=False)), ("pls", pls)])
            else:
                pipe = build_pls_pipeline(self.n_components)
                pipe.fit(X, y)
            return pipe

    def _derive_linear_params(self, d: int) -> tuple[np.ndarray, float]:
        if self.pipeline is None:
            raise RuntimeError("Pipeline not fitted")
        # Bias via response to zero vector
        z = np.zeros((1, d), dtype=float)
        b = float(np.asarray(self.pipeline.predict(z)).ravel()[0])
        # Each coordinate weight via response to unit basis
        w = np.zeros(d, dtype=float)
        for i in range(d):
            e = np.zeros((1, d), dtype=float)
            e[0, i] = 1.0
            y_e = float(np.asarray(self.pipeline.predict(e)).ravel()[0])
            w[i] = y_e - b
        return w, b

    def fit(self, X: FloatArray, y: FloatArray) -> "ParametricPLSModel":
        X = np.asarray(X)
        y = np.asarray(y)
        self.pipeline = self._fit_pipeline(X, y)
        self.w, self.b = self._derive_linear_params(X.shape[1])
        return self

    def set_global_scaler(self, mean: list | np.ndarray, scale: list | np.ndarray) -> None:
        self.global_scaler_mean = np.asarray(mean, dtype=float).copy()
        self.global_scaler_scale = np.asarray(scale, dtype=float).copy()

    def set_use_global_scaler(self, val: bool) -> None:
        self.use_global_scaler = bool(val)

    def predict(self, X: FloatArray) -> FloatArray:
        if self.w is None or self.b is None:
            raise RuntimeError("Model not fitted")
        return np.asarray(X) @ self.w + float(self.b)

    def get_params(self) -> dict:
        if self.w is None or self.b is None:
            return {"w": np.zeros(0, dtype=float), "b": np.array([0.0])}
        return {"w": self.w.copy(), "b": np.array([float(self.b)])}

    def set_params(self, params: dict) -> None:
        self.w = np.asarray(params["w"], dtype=float).copy()
        self.b = float(np.asarray(params["b"]).ravel()[0])
