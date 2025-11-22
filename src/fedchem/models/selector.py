from __future__ import annotations

from typing import Any, Dict, Tuple, Callable, Optional
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import get_scorer
from sklearn.cross_decomposition import PLSRegression

from ..types import FloatArray
from ._common import build_pls_pipeline


def _make_candidates(random_state: int = 0) -> Dict[str, Tuple[Pipeline, Dict[str, Any]]]:
    candidates: Dict[str, Tuple[Pipeline, Dict[str, Any]]] = {}

    # PLS pipeline
    pls_pipe = build_pls_pipeline(None)
    pls_grid = {"pls__n_components": [2, 4, 6, 8, 10]}
    candidates["pls"] = (pls_pipe, pls_grid)

    # NOTE: Other candidate pipelines (Ridge, SVR, RandomForest) removed to
    # enforce using only PLS as requested. If you later want to re-enable
    # them, restore their pipeline/grid definitions here.

    return candidates


def select_best_model(
    X: FloatArray,
    y: FloatArray,
    outer_cv: int = 5,
    inner_cv: int = 5,
    random_state: int = 0,
    scoring: str = "neg_root_mean_squared_error",
    validation_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """Run nested CV over candidate pipelines and return the best fitted estimator and diagnostics.

    Parameters
    - X, y: data
    - outer_cv, inner_cv: integers for nested CV
    - random_state: RNG seed
    - scoring: a sklearn-compatible scoring string (e.g. 'neg_root_mean_squared_error' or 'r2').

    Returns (best_estimator, diagnostics) where diagnostics contains per-candidate outer scores and best params.
    """
    X = np.asarray(X)
    y = np.asarray(y)

    candidates = _make_candidates(random_state=random_state)
    outer = KFold(n_splits=outer_cv, shuffle=True, random_state=random_state)

    results: Dict[str, Any] = {}

    # Use provided scoring string for GridSearchCV. If a validation_transform is
    # provided we run an explicit outer loop so validation folds can be
    # transformed (e.g. to mimic the target distribution) before scoring.
    for name, (pipe, grid) in candidates.items():
        scores = []
        for train_idx, val_idx in outer.split(X):
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            gs = GridSearchCV(pipe, grid, scoring=scoring, cv=inner_cv, n_jobs=-1, refit=True)
            gs.fit(X_train, y_train)
            best_est = gs.best_estimator_

            X_val_t = validation_transform(X_val) if validation_transform is not None else X_val
            scorer = get_scorer(scoring)
            score_val = scorer(best_est, X_val_t, y_val)
            scores.append(score_val)

        scores = np.asarray(scores)

        if isinstance(scoring, str) and scoring.startswith("neg_"):
            metric_vals = -scores
            results[name] = {
                "outer_metric_name": scoring.replace("neg_", ""),
                "outer_metric_mean": float(np.mean(metric_vals)),
                "outer_metric_std": float(np.std(metric_vals)),
                "outer_metric_all": metric_vals,
                "raw_scores": scores,
            }
        else:
            metric_vals = scores
            results[name] = {
                "outer_metric_name": scoring,
                "outer_metric_mean": float(np.mean(metric_vals)),
                "outer_metric_std": float(np.std(metric_vals)),
                "outer_metric_all": metric_vals,
                "raw_scores": scores,
            }

    # pick best model depending on whether metric is 'neg_' (we minimized original metric) or not
    if isinstance(scoring, str) and scoring.startswith("neg_"):
        # for neg_ metrics we converted to positive (e.g. RMSE) and should pick the minimum mean
        best_name = min(results.keys(), key=lambda n: results[n]["outer_metric_mean"])
    else:
        # for metrics where larger is better (e.g. r2), pick the maximum mean
        best_name = max(results.keys(), key=lambda n: results[n]["outer_metric_mean"])
    best_pipe, best_grid = candidates[best_name]

    # Refit best GridSearchCV on full data to obtain final estimator and params
    best_gs = GridSearchCV(best_pipe, best_grid, scoring=scoring, cv=inner_cv, n_jobs=-1, refit=True)
    best_gs.fit(X, y)

    diagnostics = {"candidates": results, "selected": best_name, "selected_best_params": best_gs.best_params_}
    return best_gs.best_estimator_, diagnostics
