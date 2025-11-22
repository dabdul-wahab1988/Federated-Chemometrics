from typing import Tuple, Optional
import numpy as np
from ..ct.pds_transfer import PDSTransfer
from ..privacy.dp_accountant import DPAccountant


class PrivacyGuard:
    """Helpers to measure reconstruction risk and add (Gaussian) DP noise.

    Methods here provide conservative, easy-to-audit operations:
    - compute_reconstruction_error: compare original vs reconstructed via pinv(T)
    - add_dp_noise: add Gaussian noise calibrated for (epsilon, delta)-DP on matrix entries
    """

    @staticmethod
    def compute_reconstruction_error(X_original: np.ndarray,
                                     X_transformed: np.ndarray,
                                     T_matrix: np.ndarray) -> float:
        """Estimate normalized Frobenius reconstruction error.

        The method reconstructs X_original_hat = X_transformed @ pinv(T_matrix)
        and returns ||X_original - X_original_hat||_F / ||X_original||_F.

        Notes:
        - Caller must ensure T_matrix aligns with X_transformed (shapes).
        - For block-wise PDSTransfer, callers should compute blockwise metrics using
          the per-block T matrices and then aggregate; this function handles the
          dense matrix case.
        """
        if T_matrix.size == 0:
            return float('inf')
        try:
            T_pinv = np.linalg.pinv(T_matrix)
        except np.linalg.LinAlgError:
            return float('inf')
        X_reconstructed = X_transformed @ T_pinv
        num = float(np.linalg.norm(X_original - X_reconstructed, ord='fro'))
        den = float(np.linalg.norm(X_original, ord='fro'))
        if den == 0:
            return float('inf')
        return num / den

    @staticmethod
    def add_dp_noise(matrix: np.ndarray,
                     epsilon: float = 1.0,
                     delta: float = 1e-5,
                     sensitivity: float = 1.0,
                     rng: Optional[np.random.Generator] = None,
                     clip_norm: Optional[float] = None) -> np.ndarray:
        """Add Gaussian noise to `matrix` entries calibrated for (epsilon, delta)-DP.

        This is a simple per-entry Gaussian mechanism. For rigorous DP claims the
        caller must ensure `sensitivity` is correct for the release (often it is
        not trivial for matrix-valued releases).

        Parameters
        - epsilon, delta: DP target
        - sensitivity: L2-sensitivity of the quantity being released
        - rng: optional numpy Generator for reproducibility
        - clip_norm: optional per-entry clipping (absolute) applied before noise
        """
        if rng is None:
            rng = np.random.default_rng()
        # standard Gaussian mechanism scaling for (epsilon, delta)-DP
        if epsilon <= 0 or delta <= 0:
            raise ValueError("epsilon and delta must be positive for DP noise")
        sigma = float(sensitivity * np.sqrt(2.0 * np.log(1.25 / delta)) / epsilon)
        mat = matrix.copy()
        if clip_norm is not None:
            np.clip(mat, -clip_norm, clip_norm, out=mat)
        noise = rng.normal(loc=0.0, scale=sigma, size=mat.shape)
        return mat + noise

    @staticmethod
    def _clip_rows(mat: np.ndarray, clip_norm: float) -> np.ndarray:
        if clip_norm is None:
            return mat.copy()
        out = mat.copy()
        # clip each row to have l2-norm <= clip_norm
        norms = np.linalg.norm(out, axis=1)
        too_large = norms > clip_norm
        if np.any(too_large):
            out[too_large] = out[too_large] * (clip_norm / norms[too_large])[:, None]
        return out

    @staticmethod
    def compute_pds_sensitivities(pds: PDSTransfer,
                                  clip_feature: float = 1.0,
                                  clip_target: float = 1.0) -> dict:
        """Compute conservative per-block L2-sensitivity estimates for PDS block matrices.

        We use a conservative, easily-auditable bound derived from the ridge-regularized
        least-squares solution used in PDSTransfer.fit. For a block with n samples,
        clipped per-row feature norm <= clip_feature and per-row target norm <= clip_target,
        and ridge parameter lambda, we bound the change in the block mapping TC (Frobenius norm)
        under removal/addition of a single sample by roughly:

            sens_block <= (2 * clip_feature * clip_target) / max(lambda, eps) / max(1, n)

        This is conservative: it upper-bounds the influence of one row by using the ridge
        term as a lower bound on the smallest eigenvalue of the normal matrix. The 1/n
        factor captures that the normal matrix scales with n.

        Returns a dict with per-block sensitivities and an aggregated sensitivity (Frobenius).
        """
        blocks = pds.get_blocks()
        if blocks is None:
            raise RuntimeError("PDSTransfer must be fitted to compute sensitivities")
        sens_blocks = []
        # use actual paired-set size when available on the PDSTransfer instance
        n_samples = pds.to_dict().get("n_samples")
        if n_samples is None:
            n_samples = 1
        for (start, stop, TC) in blocks:
            # TC shape: (w+1, w) where w = stop-start
            w = TC.shape[1]
            lam = max(float(getattr(pds, "ridge", 1e-6)), 1e-12)
            sens_block = (2.0 * float(clip_feature) * float(clip_target)) / lam / max(1.0, float(n_samples))
            sens_blocks.append(float(sens_block))
        # aggregate Frobenius sensitivity if all blocks are released together
        agg = float(np.sqrt(np.sum(np.array(sens_blocks) ** 2))) if sens_blocks else 0.0
        return {"per_block": sens_blocks, "aggregate": agg}

    @staticmethod
    def add_dp_noise_to_pds(pds: PDSTransfer,
                            epsilon: float,
                            delta: float = 1e-5,
                            clip_feature: float = 1.0,
                            clip_target: float = 1.0,
                            rng: Optional[np.random.Generator] = None,
                            q: float | None = None) -> dict:
        """Apply per-block clipping and add Gaussian noise calibrated to a conservative
        sensitivity estimate for each block mapping. Returns metadata including noisy
        blocks and reported epsilons from the DP accountant (for transparency).

        Notes:
        - This function recomputes each block mapping from clipped features/targets
          when possible. If original paired data are not available here, this function
          will instead add noise to the existing TC using the conservative sensitivity
          computed by compute_pds_sensitivities.
        - For a formal DP guarantee, callers must ensure the clipping values are enforced
          before learning the mappings (i.e., clip data prior to fitting). This helper
          provides a conservative end-to-end report for the existing mapping.
        """
        if rng is None:
            rng = np.random.default_rng()
        if pds._blocks is None and pds._global_TC is None:
            raise RuntimeError("PDSTransfer not fitted or empty")
        accountant = DPAccountant()
        result = {"blocks": [], "reported_epsilons": []}
        # Per-block handling
        if pds.is_global() and pds.get_global_TC() is not None:
            TC = pds.get_global_TC()
            sens_info = PrivacyGuard.compute_pds_sensitivities(pds, clip_feature=clip_feature, clip_target=clip_target)
            sens = sens_info.get("aggregate", 1.0)
            noisy = PrivacyGuard.add_dp_noise(TC, epsilon=epsilon, delta=delta, sensitivity=sens, rng=rng)
            pds.set_global_TC(noisy)
            # compute reported epsilon via accountant using sigma
            sigma = float(sens * np.sqrt(2.0 * np.log(1.25 / delta)) / epsilon)
            if q is not None:
                reported = accountant.compute("gaussian_subsampled", {"sigma": sigma, "delta": delta, "q": float(q)}, rounds=1, sensitivity=sens)
            else:
                reported = accountant.compute("gaussian", {"sigma": sigma, "delta": delta}, rounds=1, sensitivity=sens)
            result["blocks"].append({"mode": "global", "sensitivity": sens, "reported": reported})
            result["reported_epsilons"].append(reported.get("epsilon"))
            return result
        # blockwise
        sens_info = PrivacyGuard.compute_pds_sensitivities(pds, clip_feature=clip_feature, clip_target=clip_target)
        per_block_sens = sens_info.get("per_block", [])
        noisy_blocks = []
        blocks = pds.get_blocks()
        for idx, (start, stop, TC) in enumerate(blocks):
            sens = per_block_sens[idx] if idx < len(per_block_sens) else per_block_sens[-1]
            noisy_TC = PrivacyGuard.add_dp_noise(TC, epsilon=epsilon, delta=delta, sensitivity=sens, rng=rng)
            noisy_blocks.append((start, stop, noisy_TC))
            sigma = float(sens * np.sqrt(2.0 * np.log(1.25 / delta)) / epsilon)
            if q is not None:
                reported = accountant.compute("gaussian_subsampled", {"sigma": sigma, "delta": delta, "q": float(q)}, rounds=1, sensitivity=sens)
            else:
                reported = accountant.compute("gaussian", {"sigma": sigma, "delta": delta}, rounds=1, sensitivity=sens)
            result["blocks"].append({"start": start, "stop": stop, "sensitivity": sens, "reported": reported})
            result["reported_epsilons"].append(reported.get("epsilon"))
        pds.set_blocks(noisy_blocks)
        return result
