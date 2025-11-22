from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np
from ..types import FloatArray


def compute_pca_profile(X: FloatArray, wavelengths: List[float], n_components: Optional[int] = None) -> Dict[str, object]:
    """Compute a PCA-based profile to guide simulation.

    Returns a dict with keys:
    - mean: (d,) feature means
    - pca_components: (k, d) top-k right singular vectors (principal axes)
    - pca_variances: (k,) variances (eigenvalues) for each component
    - residual_std: (d,) per-feature std of residual after k components
    - wavelengths: list[float]
    """
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    n, d = X.shape
    k = d if n_components is None else int(max(1, min(d, n_components)))
    mu = X.mean(axis=0)
    Xc = X - mu
    # economy SVD
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    # variances are eigenvalues of covariance = S^2 / (n-1)
    vars_all = (S ** 2) / max(1, (n - 1))
    k = min(k, Vt.shape[0])
    comps = Vt[:k]
    vars_k = vars_all[:k]
    # residual
    Xk = (U[:, :k] * S[:k]) @ Vt[:k]
    R = Xc - Xk
    resid_std = R.std(axis=0) + 1e-9
    return {
        "mean": mu.astype(np.float32),
        "pca_components": comps.astype(np.float32),
        "pca_variances": vars_k.astype(np.float32),
        "residual_std": resid_std.astype(np.float32),
        "wavelengths": list(wavelengths),
    }
