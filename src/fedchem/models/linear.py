from __future__ import annotations

import numpy as np
from .base import BaseModel
from ..types import FloatArray
from ..federated.types import ParametricModel
from ._common import ridge_lstsq


class LinearModel(BaseModel, ParametricModel):
    """Simple linear model y ≈ X w + b with param access for FL.

    Uses least squares fit; params are {"w": (d,), "b": (1,)}.
    """

    def __init__(self) -> None:
        # Small ridge regularization can improve stability for ill-conditioned X
        self.w: np.ndarray | None = None
        self.b: float | None = None
        self.ridge: float = 0.0

    def fit(self, X: FloatArray, y: FloatArray, ridge: float | None = None) -> "LinearModel":
        """Fit linear model with optional ridge regularization using shared helper."""
        if ridge is not None:
            self.ridge = float(ridge)
        self.w, self.b = ridge_lstsq(np.asarray(X), np.asarray(y), ridge=self.ridge, fit_intercept=True)
        return self

    def predict(self, X: FloatArray) -> FloatArray:
        if self.w is None or self.b is None:
            raise RuntimeError("Model not fitted")
        from typing import cast
        return cast(np.ndarray, X @ self.w + self.b)

    def get_params(self) -> dict:
        if self.w is None or self.b is None:
            # expose zeros for uninitialized model
            return {"w": np.zeros(0, dtype=float), "b": np.array([0.0])}
        return {"w": self.w.copy(), "b": np.array([self.b])}

    def set_params(self, params: dict) -> None:
        self.w = params["w"].copy()
        self.b = float(params["b"][0])
