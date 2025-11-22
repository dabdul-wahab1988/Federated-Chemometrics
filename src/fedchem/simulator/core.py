from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
from ..types import FloatArray


@dataclass
class SiteData:
    X: FloatArray
    y: FloatArray
    meta: dict


class SpectralSimulator:
    """Generate synthetic multi-site spectral datasets with simple distortions.

    This stub focuses on deterministic structure; distortion details are minimal
    but follow the A2 contract so tests can target shapes and determinism.
    """

    def __init__(self, seed: Optional[int] = None, profile: Optional[dict] = None) -> None:
        self.rng = np.random.default_rng(seed)
        self._profile: Optional[dict] = profile

    def set_profile(self, profile: dict) -> "SpectralSimulator":
        """Attach a real-data profile to guide simulation statistics.

        Expected keys (all optional, used if present):
        - mean: (n_wv,) mean spectrum
        - std: (n_wv,) per-wavelength std dev (non-negative)
        - wavelengths: list[float]
        - noise_std: float baseline noise level
        - smooth_kernel: int window size for correlation smoothing
        - pca_components: (k, n_wv) principal axes
        - pca_variances: (k,) variances along components
        - residual_std: (n_wv,) per-wavelength residual std
        """
        self._profile = dict(profile)
        return self

    def fit_profile(self, X: FloatArray, wavelengths: List[float]) -> dict:
        """Compute a simple profile from real spectra to guide simulation.

        The profile captures mean curve, per-wavelength std, and a crude
        correlation proxy via smoothing window derived from autocorrelation.
        """
        if X.ndim != 2:
            raise ValueError("X must be 2D")
        mean = X.mean(axis=0)
        std = X.std(axis=0) + 1e-9
        # crude estimate of smoothness from first-lag autocorr
        x0 = X[0] - X[0].mean()
        ac1 = float(np.corrcoef(x0[:-1], x0[1:])[0, 1]) if x0.size > 1 else 0.0
        kernel = int(max(3, min(21, round(5 + 10 * max(0.0, ac1)))))
        prof = {
            "mean": mean.astype(np.float32),
            "std": std.astype(np.float32),
            "wavelengths": list(wavelengths),
            "noise_std": float(np.median(std) * 0.05),
            "smooth_kernel": kernel,
        }
        self._profile = prof
        return prof

    def generate_sites(
        self,
        n_sites: int,
        n_samples_per_site: int,
        wavelengths: List[float],
        modalities: List[str],
    ) -> Dict[str, Dict[str, object]]:
        if n_sites <= 0 or n_samples_per_site <= 0:
            raise ValueError("n_sites and n_samples_per_site must be > 0")
        if not wavelengths:
            raise ValueError("wavelengths must be non-empty")
        if not modalities:
            raise ValueError("modalities must be non-empty")

        n_wv = len(wavelengths)
        data: Dict[str, Dict[str, object]] = {}
        for s in range(n_sites):
            # base spectra guided by profile if available
            if self._profile is not None and "mean" in self._profile and len(self._profile["mean"]) == n_wv:
                mean = np.asarray(self._profile["mean"], dtype=np.float32)
                comps = self._profile.get("pca_components")
                vars_k = self._profile.get("pca_variances")
                resid_std = self._profile.get("residual_std")
                if comps is not None and vars_k is not None:
                    C = np.asarray(comps, dtype=np.float32)  # (k, d)
                    v = np.asarray(vars_k, dtype=np.float32)  # (k,)
                    z = self.rng.normal(0.0, 1.0, size=(n_samples_per_site, v.shape[0])).astype(np.float32)
                    Xk = z @ (C * np.sqrt(v)[:, None])  # (n, d)
                    if resid_std is not None:
                        rs = np.asarray(resid_std, dtype=np.float32)
                        eps = self.rng.normal(0.0, 1.0, size=(n_samples_per_site, n_wv)).astype(np.float32) * rs
                        X = mean + Xk + eps
                    else:
                        X = mean + Xk
                else:
                    std = np.asarray(self._profile.get("std", np.ones(n_wv)), dtype=np.float32)
                    smooth_k = int(self._profile.get("smooth_kernel", 7))
                    if smooth_k % 2 == 0:
                        smooth_k += 1
                    raw = self.rng.normal(0.0, 1.0, size=(n_samples_per_site, n_wv))
                    # impose local correlation by smoothing along wavelength
                    if smooth_k > 1:
                        pad = smooth_k // 2
                        kernel = np.ones(smooth_k, dtype=np.float32) / smooth_k
                        raw_p = np.pad(raw, ((0, 0), (pad, pad)), mode="edge")
                        raw = np.apply_along_axis(lambda v: np.convolve(v, kernel, mode="valid"), 1, raw_p)
                    X = mean + raw * std
            else:
                # fallback: smooth random walk
                X = self.rng.normal(0.0, 1.0, size=(n_samples_per_site, n_wv))
                X = np.cumsum(X, axis=1)
            # simple site bias/scale
            bias = (s + 1) * 0.05
            scale = 1.0 + (s * 0.02)
            X = scale * X + bias
            # target is linear projection + noise
            w = np.linspace(0.1, 1.0, n_wv)
            nstd = float(self._profile.get("noise_std", 0.1)) if self._profile else 0.1
            y = X @ w + self.rng.normal(0.0, nstd, size=(n_samples_per_site,))
            data[f"site_{s}"] = {
                "X": X.astype(np.float32),
                "y": y.astype(np.float32),
                "meta": {
                    "modality": modalities[min(s, len(modalities) - 1)],
                    "wavelengths": wavelengths,
                    "distortions": {},
                },
            }
        return data

    def apply_distortions(self, X: FloatArray, config: dict) -> FloatArray:
        """Apply simple, parameterized distortions.

        Supported keys (minimal stub):
        - scatter: float multiplicative
        - baseline: float additive
        - noise_std: float gaussian noise
        """
        if X.ndim != 2:
            raise ValueError("X must be 2D [n_samples, n_features]")
        Y = X.copy()
        scatter = float(config.get("scatter", 1.0))
        baseline = float(config.get("baseline", 0.0))
        noise_std = float(config.get("noise_std", 0.0))
        Y = scatter * Y + baseline
        if noise_std > 0:
            Y = Y + self.rng.normal(0.0, noise_std, size=Y.shape)
        return Y.astype(np.float32)
