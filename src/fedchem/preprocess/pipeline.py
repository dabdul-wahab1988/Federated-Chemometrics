from __future__ import annotations

from typing import Optional, cast
import numpy as np
from ..types import FloatArray
from scipy.signal import savgol_filter, detrend as sp_detrend
from sklearn.preprocessing import StandardScaler


class PreprocessPipeline:
    """Configurable spectral preprocessing pipeline.

    Steps (order): detrend -> SNV -> MSC -> Savitzky–Golay -> scaling.
    Each step is optional and controlled by constructor flags.
    """

    def __init__(
        self,
        *,
        snv: bool = False,
        msc: bool = False,
        detrend: bool = False,
        savgol: Optional[dict] = None,
        scale: Optional[str] = None,
        drift_cfg: Optional[dict] = None,
        augmentation_seed: int | None = None,
    ) -> None:
        self.use_snv = snv
        self.use_msc = msc
        self.use_detrend = detrend
        self.savgol_cfg = savgol
        self.scale_mode = scale

        self._fitted = False
        self._msc_ref: Optional[np.ndarray] = None
        self._scaler: Optional[StandardScaler] = None
        self._drift_cfg = drift_cfg or {}
        self._augmentation_seed = augmentation_seed

    def fit(self, X: FloatArray, y: FloatArray | None = None) -> "PreprocessPipeline":
        if X.ndim != 2:
            raise ValueError("X must be 2D [n_samples, n_features]")
        Xp = X
        # Optionally apply augmentation during training if configured
        if isinstance(self._drift_cfg, dict) and bool(self._drift_cfg.get('apply_augmentation_during_training', False)):
            Xp = self._apply_drift(Xp, is_training=True)
        if self.use_detrend:
            Xp = sp_detrend(Xp, axis=1, type="linear")
        if self.use_snv:
            Xp = self._snv(Xp)
        if self.use_msc:
            # reference spectrum: mean across samples (post-SNV/detrend if enabled)
            self._msc_ref = Xp.mean(axis=0)
        if self.scale_mode == "standard":
            self._scaler = StandardScaler(with_mean=True, with_std=True)
            self._scaler.fit(Xp)
        self._fitted = True
        return self

    def transform(self, X: FloatArray) -> FloatArray:
        if X.ndim != 2:
            raise ValueError("X must be 2D [n_samples, n_features]")
        Xp = X
        # Optionally apply test-time drift augmentation
        if isinstance(self._drift_cfg, dict) and bool(self._drift_cfg.get('apply_test_shifts', False)):
            Xp = self._apply_drift(Xp, is_training=False)
        if self.use_detrend:
            Xp = sp_detrend(Xp, axis=1, type="linear")
        if self.use_snv:
            Xp = self._snv(Xp)
        if self.use_msc:
            if self._msc_ref is None:
                raise RuntimeError("MSC enabled but pipeline not fitted")
            Xp = self._msc(Xp, self._msc_ref)
        if self.savgol_cfg:
            cfg = {"window_length": 7, "polyorder": 2, "deriv": 0}
            cfg.update(self.savgol_cfg)
            wl = int(cfg["window_length"]) | 1  # ensure odd
            po = int(cfg["polyorder"])  # polyorder
            dv = int(cfg.get("deriv", 0))
            if wl <= po:
                wl = po + 1 if (po + 1) % 2 == 1 else po + 2
            Xp = savgol_filter(Xp, window_length=wl, polyorder=po, deriv=dv, axis=1)
        if self.scale_mode == "standard":
            if self._scaler is None:
                raise RuntimeError("Scaler not fitted")
            Xp = cast(np.ndarray, self._scaler.transform(Xp))
        return Xp

    def fit_transform(self, X: FloatArray, y: FloatArray | None = None) -> FloatArray:
        return self.fit(X, y).transform(X)

    def _apply_drift(self, X: FloatArray, is_training: bool = False) -> FloatArray:
        """Apply spectral drift/augmentation in-place and return augmented X.

        Supports jitter (wavelength shift), multiplicative scatter, baseline offset,
        and white noise.
        """
        cfg = self._drift_cfg or {}
        n, m = X.shape[0], X.shape[1]
        if n == 0 or m == 0:
            return X
        rng_seed = self._augmentation_seed if self._augmentation_seed is not None else None
        rng = np.random.default_rng(rng_seed)
        Xm = X.astype(float).copy()
        # jitter in fraction of pixel (stdev = jitter_fraction * m)
        jitter_fraction = float(cfg.get('jitter_wavelength_px', 0.0) or 0.0)
        mult_scatter = float(cfg.get('multiplicative_scatter', 0.0) or 0.0)
        baseline_offset = float(cfg.get('baseline_offset', 0.0) or 0.0)
        white_noise_sigma = float(cfg.get('white_noise_sigma', 0.0) or 0.0)

        xs = np.arange(m, dtype=float)
        for i in range(n):
            row = Xm[i]
            # multiplicative scatter
            if mult_scatter > 0.0:
                mfac = 1.0 + rng.normal(0.0, mult_scatter)
                row = row * mfac
            # baseline offset (fraction of mean)
            if baseline_offset > 0.0:
                offs = rng.normal(0.0, baseline_offset) * np.mean(np.abs(row))
                row = row + offs
            # white noise
            if white_noise_sigma > 0.0:
                noise = rng.normal(0.0, white_noise_sigma * np.mean(np.abs(row)), size=(m,))
                row = row + noise
            # jitter (wavelength pixel shift)
            if jitter_fraction > 0.0:
                shift = float(rng.normal(0.0, jitter_fraction * float(m)))
                new_x = xs + shift
                # Interpolate to original grid
                try:
                    row = np.interp(xs, new_x, row, left=row[0], right=row[-1])
                except Exception:
                    # Fallback: no jitter when interpolation fails
                    pass
            Xm[i] = row
        return Xm

    @staticmethod
    def _snv(X: FloatArray) -> FloatArray:
        # Standard Normal Variate per sample
        mu = X.mean(axis=1, keepdims=True)
        sd = X.std(axis=1, keepdims=True) + 1e-12
        return cast(np.ndarray, (X - mu) / sd)

    @staticmethod
    def _msc(X: FloatArray, ref: FloatArray) -> FloatArray:
        # Multiplicative Scatter Correction using reference spectrum
        # For each sample s: solve s ~ a + b*ref; corrected = (s - a) / b
        n = X.shape[0]
        out = np.empty_like(X)
        for i in range(n):
            s = X[i]
            A = np.c_[ref, np.ones_like(ref)]
            # Solve for [b, a] in least squares: s ≈ b*ref + a
            b, a = np.linalg.lstsq(A, s, rcond=None)[0]
            out[i] = (s - a) / (b + 1e-12)
        return out
