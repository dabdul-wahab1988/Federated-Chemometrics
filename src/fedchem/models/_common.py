from __future__ import annotations

from typing import Optional, Tuple, Dict, Any
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold


def ridge_lstsq(X: np.ndarray, y: np.ndarray, ridge: float = 0.0, fit_intercept: bool = True) -> Tuple[np.ndarray, float]:
    """Solve linear least squares with optional ridge and intercept.

    Returns (w, b) where w has shape (d,) and b is a float.
    If fit_intercept=False, b will be 0.0.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    if fit_intercept:
        X_ = np.c_[X, np.ones((X.shape[0], 1))]
        if ridge == 0.0:
            coef = np.linalg.lstsq(X_, y, rcond=None)[0]
        else:
            D = np.eye(X_.shape[1]) * float(ridge)
            # do not regularize the intercept term
            D[-1, -1] = 0.0
            A = X_.T @ X_ + D
            coef = np.linalg.solve(A, X_.T @ y)
        w = coef[:-1]
        b = float(coef[-1])
        return w, b
    else:
        if ridge == 0.0:
            w = np.linalg.lstsq(X, y, rcond=None)[0]
        else:
            A = X.T @ X + float(ridge) * np.eye(X.shape[1])
            w = np.linalg.solve(A, X.T @ y)
        return np.asarray(w).ravel(), 0.0


def build_pls_pipeline(n_components: Optional[int] = None) -> Pipeline:
    """Create a StandardScaler + PLSRegression pipeline.

    If n_components is None, the PLSRegression will be created without n_components
    so that it can be tuned via grid search.
    """
    if n_components is None:
        pls = PLSRegression()
    else:
        pls = PLSRegression(n_components=int(n_components))
    return Pipeline([("scaler", StandardScaler()), ("pls", pls)])


def pls_param_grid(max_components: int, n_features: int, n_samples: int | None = None) -> Dict[str, Any]:
    """Parameter grid for tuning PLS n_components up to max_components, feature count, or sample size."""
    if n_samples is None:
        max_c = max(1, min(int(max_components), int(n_features)))
    else:
        # PLS components cannot exceed min(n_features, n_samples-1) for meaningful fit
        max_c = max(1, min(int(max_components), int(n_features), max(1, int(n_samples) - 1)))
    return {"pls__n_components": list(range(1, max_c + 1))}


def make_kfold(cv: int, random_state: Optional[int] = None) -> KFold:
    """Consistent KFold setup used across models."""
    return KFold(n_splits=int(cv), shuffle=True, random_state=random_state)
