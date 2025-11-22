from typing import Any, Dict, List
import numpy as np
from fedchem.models.pls import PLSModel
from fedchem.models.linear import LinearModel


def train_central_pls(X_ref: np.ndarray, y_ref: np.ndarray, n_components: int = 15):
    """Train central PLS model on reference data."""
    m = PLSModel(n_components=n_components)
    return m.fit(X_ref, y_ref)


def train_site_pls(X_site: np.ndarray, y_site: np.ndarray, n_components: int = 15):
    m = PLSModel(n_components=n_components)
    return m.fit(X_site, y_site)


def sbc_bias_correct(central_model: Any, X_paired: np.ndarray, y_paired: np.ndarray):
    """Simple bias correction: compute mean residual on paired set and return a wrapper.

    Returns a callable predict_fn(X) -> predictions
    """
    y_pred = np.asarray(central_model.predict(X_paired)).ravel()
    bias = float(np.mean(y_paired - y_pred))

    def predict_fn(X):
        return np.asarray(central_model.predict(X)).ravel() + bias

    return predict_fn


def fedavg_site_models(site_models: List[Any], X: np.ndarray):
    """Predict by averaging predictions from site-local models."""
    preds = [np.asarray(m.predict(X)).ravel() for m in site_models]
    return np.mean(np.vstack(preds), axis=0)


def fedavg_coef_predict(site_models: List[Any], X: np.ndarray):
    """Attempt coefficient-level FedAvg for linear/Pipeline models.

    If site models are PLSModel instances with a pipeline containing a
    StandardScaler followed by a PLSRegression, we extract the equivalent
    coefficients in the original feature space, average them and return
    linear predictions using the averaged coef and intercept. Falls back
    to prediction averaging if coefficients can't be extracted.
    """
    if len(site_models) == 0:
        return np.zeros((X.shape[0],), dtype=float)
    coefs = []
    intercepts = []
    for m in site_models:
        try:
            pipe = getattr(m, "pipeline", None)
            pls = pipe.named_steps["pls"]
            scaler = pipe.named_steps["scaler"]
            coef_pls = np.asarray(pls.coef_).ravel()
            scale = np.asarray(scaler.scale_)
            mean = np.asarray(scaler.mean_)
            # coefficient in original feature space
            coef_orig = coef_pls / scale
            intercept_orig = float(pls.y_mean_ - np.dot(mean, coef_orig))
            coefs.append(coef_orig)
            intercepts.append(intercept_orig)
        except Exception:
            coefs = []
            break
    if len(coefs) == 0:
        # fallback
        return fedavg_site_models(site_models, X)
    avg_coef = np.mean(np.vstack(coefs), axis=0)
    avg_intercept = float(np.mean(np.array(intercepts)))
    return X.dot(avg_coef) + avg_intercept


def crds_predict(central_model: Any,
                 X_ref: np.ndarray,
                 X_site: np.ndarray,
                 X_local: np.ndarray,
                 window: int = 16,
                 overlap: int = 0,
                 clip_feature: float | None = None,
                 clip_target: float | None = None) -> np.ndarray:
    """Piecewise CRDS: per-window least-squares mapping from site -> ref.

    Parameters
    - window, overlap: define piecewise segments (like PDSTransfer)
    - clip_feature/clip_target: optional L2 clipping applied to rows before solving
    """
    try:
        X_ref = np.asarray(X_ref)
        X_site = np.asarray(X_site)
        X_local = np.asarray(X_local)
        n, d = X_site.shape
        # optional clipping of rows
        if clip_feature is not None:
            norms = np.linalg.norm(X_ref, axis=1)
            too_large = norms > clip_feature
            if np.any(too_large):
                X_ref[too_large] = X_ref[too_large] * (clip_feature / norms[too_large])[:, None]
        if clip_target is not None:
            norms_t = np.linalg.norm(X_site, axis=1)
            too_large_t = norms_t > clip_target
            if np.any(too_large_t):
                X_site[too_large_t] = X_site[too_large_t] * (clip_target / norms_t[too_large_t])[:, None]

        out = np.zeros_like(X_local)
        weight = np.zeros_like(X_local)
        start = 0
        while start < d:
            stop = min(d, start + window)
            Xs = X_site[:, start:stop]
            Xr = X_ref[:, start:stop]
            try:
                T, *_ = np.linalg.lstsq(Xs, Xr, rcond=None)
                Xloc_seg = X_local[:, start:stop] @ T
            except Exception:
                # fallback: identity mapping for this segment
                Xloc_seg = X_local[:, start:stop]
            out[:, start:stop] += Xloc_seg
            weight[:, start:stop] += 1.0
            if stop == d:
                break
            start = stop - overlap if overlap > 0 else stop
        weight[weight == 0] = 1.0
        X_local_mapped = out / weight
        return np.asarray(central_model.predict(X_local_mapped)).ravel()
    except Exception:
        return np.asarray(central_model.predict(X_local)).ravel()
