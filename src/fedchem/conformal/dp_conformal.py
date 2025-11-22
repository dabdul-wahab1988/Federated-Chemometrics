from __future__ import annotations

from typing import Optional, Tuple
import numpy as np

from ..privacy.dp_accountant import DPAccountant
from .cqr import CQRConformal
from .cross_cqr import CrossCQRConformal
from ..types import FloatArray


class DPConformal:
    """Differentially private conformal quantile release wrapper.

    Approach: Fit an underlying conformal method (CQR/CrossCQR/MondrianCQR)
    and obtain the calibration nonconformity scores. Clip them to a bound
    `clip_bound`. Compute the empirical (1 - alpha) quantile on the clipped
    scores. Apply a Gaussian mechanism to the released quantile value
    (noise drawn from N(0, sigma^2)) where sigma is chosen to meet the
    requested privacy budget with sensitivity `clip_bound / n_calib`.

    This yields a DP quantile that can be injected into the standard
    (q_lo, q_hi) interval outputs.
    """

    def __init__(
        self,
        base: Optional[object] = None,
        *,
        alpha: float = 0.1,
        clip_bound: float = 1.0,
        dp_config: Optional[dict] = None,
        rng_seed: Optional[int] = None,
    ) -> None:
        self.base = base if base is not None else CQRConformal()
        self.alpha = float(alpha)
        self.clip_bound = float(clip_bound)
        self.dp_config = dict(dp_config) if dp_config is not None else None
        self.rng = np.random.default_rng(rng_seed)
        self._q_alpha_noisy: Optional[float] = None
        self._dp: Optional[dict] = None

    def fit(self, X: FloatArray, y: FloatArray, groups: Optional[np.ndarray] = None) -> "DPConformal":
        # Fit base conformal and capture calibration scores
        if isinstance(self.base, CrossCQRConformal):
            # CrossCQR handles splitting internally
            self.base.fit(X, y)
            calib_scores = getattr(self.base, "_scores", None)
        else:
            if groups is None:
                # Expect CQR-like behavior
                if hasattr(self.base, "fit"):
                    self.base.fit(X, y)
                calib_scores = getattr(self.base, "_state", None)
                if calib_scores is not None:
                    calib_scores = getattr(calib_scores, "calib_scores", None)
            else:
                # Mondrian or grouped fit
                try:
                    self.base.fit(X, y, groups=groups)  # type: ignore[arg-type]
                    # For mondrian, group states exist; aggregate scores
                    if hasattr(self.base, "_group_states"):
                        # concat all group calib scores
                        all_scores = []
                        for s in self.base._group_states.values():
                            all_scores.append(s.calib_scores)
                        calib_scores = np.hstack(all_scores) if all_scores else np.empty((0,))
                    else:
                        calib_scores = getattr(self.base, "_state", None)
                        if calib_scores is not None:
                            calib_scores = getattr(calib_scores, "calib_scores", None)
                except Exception:
                    # fallback: try base.fit(X,y)
                    self.base.fit(X, y)
                    calib_scores = getattr(self.base, "_state", None)
                    if calib_scores is not None:
                        calib_scores = getattr(calib_scores, "calib_scores", None)

        if calib_scores is None:
            raise RuntimeError("Base conformal did not expose calibration scores; ensure base is CQR/CrossCQR/MondrianCQR")
        calib_scores = np.asarray(calib_scores, dtype=float)
        # Clip scores to bound
        if self.clip_bound is not None and self.clip_bound > 0:
            calib_scores = np.clip(calib_scores, 0.0, float(self.clip_bound))
        ncal = calib_scores.size
        if ncal <= 0:
            raise ValueError("No calibration samples available for DP conformalization")
        # compute empirical quantile
        q_emp = float(np.quantile(calib_scores, 1.0 - self.alpha, method="higher"))

        dp = None
        noise_std = None
        if self.dp_config:
            dp = dict(self.dp_config)
            target_eps = dp.get("target_epsilon") or dp.get("epsilon")
            delta = dp.get("delta", 1e-5)
            noise_std_cfg = dp.get("noise_std") or dp.get("sigma")
            sensitivity = float(self.clip_bound) / max(1, ncal)
            acct = DPAccountant()
            if noise_std_cfg is None and target_eps is not None and not np.isinf(float(target_eps)):
                # Solve sigma for a single release
                sigma = acct.solve_sigma_for_target_epsilon(float(target_eps), float(delta), rounds=1, sensitivity=sensitivity)
                noise_std = float(sigma)
            elif noise_std_cfg is not None:
                noise_std = float(noise_std_cfg)
            else:
                # No DP applied
                noise_std = None
            # Apply Gaussian noise to the quantile value
            if noise_std is not None and noise_std > 0:
                q_noisy = q_emp + float(self.rng.normal(0.0, noise_std))
            else:
                q_noisy = q_emp
            # Compute epsilon for reporting
            if noise_std is not None and noise_std > 0:
                res = acct.compute("gaussian", {"sigma": float(noise_std), "delta": float(delta)}, rounds=1, sensitivity=sensitivity)
                eps = float(res.get("epsilon", 0.0))
            else:
                eps = float("inf")
            dp = {"noise_std": noise_std, "epsilon": eps, "delta": float(delta), "sensitivity": sensitivity}
        else:
            q_noisy = q_emp

        self._q_alpha_noisy = float(max(0.0, q_noisy))
        self._dp = dp
        return self

    def predict_interval(self, X: FloatArray, groups: Optional[np.ndarray] = None, alpha: Optional[float] = None) -> Tuple[FloatArray, FloatArray]:
        # If base has q_lo/q_hi regressors we can compute intervals from them and our noisy quantile
        a = self.alpha if alpha is None else alpha
        if self._q_alpha_noisy is None:
            raise RuntimeError("DPConformal has not been fitted")
        q_alpha = float(self._q_alpha_noisy)
        # CQR/Mondrian: base should have q_lo and q_hi regressors
        if hasattr(self.base, "_state") and getattr(self.base, "_state") is not None:
            qlo = getattr(self.base, "_state").q_lo
            qhi = getattr(self.base, "_state").q_hi
            lo = qlo.predict(X) - q_alpha
            hi = qhi.predict(X) + q_alpha
            return lo, hi
        # CrossCQR: average fold predictions
        if hasattr(self.base, "_folds") and getattr(self.base, "_folds") is not None:
            lo_preds = [m.q_lo.predict(X) for m in self.base._folds]
            hi_preds = [m.q_hi.predict(X) for m in self.base._folds]
            lo = np.mean(np.vstack(lo_preds), axis=0) - q_alpha
            hi = np.mean(np.vstack(hi_preds), axis=0) + q_alpha
            return lo, hi
        raise RuntimeError("DPConformal base not recognized for prediction")

    @property
    def dp(self) -> Optional[dict]:
        return self._dp
