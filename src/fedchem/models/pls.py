from __future__ import annotations

from typing import Optional
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from .base import BaseModel
from ..types import FloatArray
from ._common import build_pls_pipeline, pls_param_grid, make_kfold


class PLSModel(BaseModel):
    """PLS regression wrapper with optional CV to select n_components.

    If n_components is None, fit will perform CV over 1..max_components using
    KFold(cv) and select the number of components minimizing RMSE.
    """

    def __init__(self, n_components: Optional[int] = None, max_components: int = 20, cv: int = 5, random_state: Optional[int] = None) -> None:
        self.n_components = n_components
        self.max_components = int(max_components)
        self.cv = int(cv)
        self.random_state = random_state
        self.pipeline: Pipeline | None = None

    def fit(self, X: FloatArray, y: FloatArray) -> "PLSModel":
        X = np.asarray(X)
        y = np.asarray(y)
        if self.n_components is None:
            pipe = build_pls_pipeline(None)
            param_grid = pls_param_grid(self.max_components, X.shape[1])
            cv = make_kfold(self.cv, self.random_state)
            gs = GridSearchCV(pipe, param_grid, scoring="neg_root_mean_squared_error", cv=cv)
            gs.fit(X, y)
            self.pipeline = gs.best_estimator_
        else:
            pipe = build_pls_pipeline(self.n_components)
            pipe.fit(X, y)
            self.pipeline = pipe
        return self

    def predict(self, X: FloatArray) -> FloatArray:
        if self.pipeline is None:
            raise RuntimeError("PLSModel not fitted")
        yhat = self.pipeline.predict(np.asarray(X))
        # ensure 1-D output
        return np.asarray(yhat).ravel()
