from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from ..types import FloatArray


@dataclass
class PDSDiagnostics:
    block_ranges: List[Tuple[int, int]]
    cond_numbers: List[float]
    block_rmse: List[float]
    mean_rmse: float


class PDSTransfer:
    """Strong PDS transfer using paired source/target transfer sets.

    Learns blockwise linear maps with optional overlap and ridge regularization.
    Provides diagnostics (condition numbers and block errors).
    """

    def __init__(self, window: int = 16, overlap: int = 0, ridge: float = 1e-6, use_global_affine: bool = True) -> None:
        self.window = int(max(1, window))
        self.overlap = int(max(0, overlap))
        self.ridge = float(ridge)
        self.use_global_affine = bool(use_global_affine)
        self._blocks: List[Tuple[int, int, np.ndarray]] | None = None
        self._diags: PDSDiagnostics | None = None
        self._global_TC: np.ndarray | None = None
        self._mode: str = "blocks"

    def fit(self, X_source: FloatArray, X_target: FloatArray,
            clip_feature: float | None = None, clip_target: float | None = None) -> "PDSTransfer":
        """Fit blockwise (and optional global) PDS mappings.

        Parameters
        - clip_feature: optional L2 clip bound applied to rows of X_source (reference)
        - clip_target: optional L2 clip bound applied to rows of X_target (site)
        """
        if X_source.shape != X_target.shape:
            raise ValueError("Source and Target transfer sets must have same shape")
        n, d = X_source.shape
        # record number of paired samples for downstream sensitivity calculations
        self._n_samples = int(n)

        # optional clipping-before-learning (per-row L2 clipping)
        if clip_feature is not None or clip_target is not None:
            # defensive copy
            X_source = np.asarray(X_source).copy()
            X_target = np.asarray(X_target).copy()
            if clip_feature is not None:
                norms = np.linalg.norm(X_source, axis=1)
                too_large = norms > clip_feature
                if np.any(too_large):
                    X_source[too_large] = X_source[too_large] * (clip_feature / norms[too_large])[:, None]
            if clip_target is not None:
                norms_t = np.linalg.norm(X_target, axis=1)
                too_large_t = norms_t > clip_target
                if np.any(too_large_t):
                    X_target[too_large_t] = X_target[too_large_t] * (clip_target / norms_t[too_large_t])[:, None]
        W = self.window
        overlap_amt = self.overlap
        blocks: List[Tuple[int, int, np.ndarray]] = []
        ranges: List[Tuple[int, int]] = []
        conds: List[float] = []
        rmses: List[float] = []
        start = 0
        while start < d:
            stop = min(d, start + W)
            Xs = X_source[:, start:stop]
            Xt = X_target[:, start:stop]
            # Include intercept: solve [Xt | 1] @ [T; c^T] ≈ Xs
            X_aug = np.c_[Xt, np.ones((Xt.shape[0], 1))]
            A = X_aug.T @ X_aug + self.ridge * np.eye(X_aug.shape[1])
            # diagnostics: condition number
            conds.append(float(np.linalg.cond(A)))
            TC = np.linalg.solve(A, X_aug.T @ Xs)  # shape [(w+1), w]
            T = TC[:-1, :]
            C = TC[-1, :]  # intercept per feature in block
            ranges.append((start, stop))
            # store bias as last row appended to T with marker
            blocks.append((start, stop, np.vstack([T, C])))
            Xs_hat = Xt @ T + C
            rmses.append(float(np.sqrt(np.mean((Xs_hat - Xs) ** 2))))
            if stop == d:
                break
            start = stop - overlap_amt if overlap_amt > 0 else stop
        self._blocks = blocks
        self._diags = PDSDiagnostics(
            block_ranges=ranges,
            cond_numbers=conds,
            block_rmse=rmses,
            mean_rmse=float(np.mean(rmses) if rmses else 0.0),
        )
        # Optional global affine map selection (preferred when enabled)
        if self.use_global_affine:
            # Learn a global affine map from target -> source directly by solving
            # (X_target^T X_target + ridge I) T_t2s = X_target^T X_source.
            # This avoids inverting a potentially ill-conditioned source->target map.
            At = X_target.T @ X_target + self.ridge * np.eye(d)
            # Solve for target->source mapping directly (d, d)
            T_t2s = np.linalg.solve(At, X_target.T @ X_source)
            # Evaluate global mapping quality on the transfer set
            Xs_hat_global = X_target @ T_t2s
            global_rmse = float(np.sqrt(np.mean((Xs_hat_global - X_source) ** 2)))
            block_mean_rmse = float(self._diags.mean_rmse if self._diags is not None else np.inf)
            # Accept global mapping only if its RMSE is comparable to block mean RMSE
            # Use a tolerance factor to allow slight degradation (e.g., 2x)
            tol_factor = 2.0
            if global_rmse <= tol_factor * block_mean_rmse:
                TC = np.vstack([T_t2s, np.zeros((1, d))])
                self._mode = "global"
                self._global_TC = TC
            else:
                # Keep blockwise mappings if global mapping is poor
                self._mode = "blocks"
        else:
            self._mode = "blocks"
        return self

    def transform(self, X_target: FloatArray) -> FloatArray:
        if self._mode == "global":
            if self._global_TC is None:
                raise RuntimeError("PDSTransfer global mapping missing")
            TC = self._global_TC
            T = TC[:-1, :]
            C = TC[-1, :]
            from typing import cast
            return cast(np.ndarray, X_target @ T + C)
        if self._blocks is None:
            raise RuntimeError("PDSTransfer not fitted")
        out = np.zeros_like(X_target)
        weight = np.zeros_like(X_target)
        for start, stop, TC in self._blocks:
            Xt = X_target[:, start:stop]
            T = TC[:-1, :]
            C = TC[-1, :]
            Xs_hat = Xt @ T + C
            out[:, start:stop] += Xs_hat
            weight[:, start:stop] += 1.0
        weight[weight == 0] = 1.0
        from typing import cast
        return cast(np.ndarray, out / weight)

    @property
    def diagnostics(self) -> PDSDiagnostics:
        if self._diags is None:
            raise RuntimeError("PDSTransfer not fitted")
        return self._diags

    # Public convenience helpers -------------------------------------------------
    def estimated_bytes(self) -> int:
        """Return a conservative estimate of bytes required to transmit the
        learned mapping(s). Uses the underlying numpy arrays' `nbytes` when
        available. Returns 0 if not fitted.
        """
        if getattr(self, "_global_TC", None) is not None:
            return int(getattr(self._global_TC, "nbytes", 0) or 0)
        blocks = getattr(self, "_blocks", None)
        if not blocks:
            return 0
        total = 0
        for (_, _, tc) in blocks:
            total += int(getattr(tc, "nbytes", 0) or 0)
        return total

    def is_global(self) -> bool:
        """Return True when the PDSTransfer selected the global affine mapping."""
        return getattr(self, "_mode", "blocks") == "global"

    def to_dict(self) -> dict:
        """Return a small serializable summary of the fitted transfer.

        Includes mode, number of samples (if recorded), estimated bytes and
        diagnostics summary. Useful for manifests and logging without
        accessing private attributes directly.
        """
        d = {
            "mode": getattr(self, "_mode", None),
            "n_samples": int(getattr(self, "_n_samples", -1)) if getattr(self, "_n_samples", None) is not None else None,
            "estimated_bytes": int(self.estimated_bytes()),
        }
        try:
            di = self.diagnostics
            d["diagnostics"] = {
                "block_ranges": list(di.block_ranges),
                "cond_numbers": list(di.cond_numbers),
                "block_rmse": list(di.block_rmse),
                "mean_rmse": float(di.mean_rmse),
            }
        except Exception:
            d["diagnostics"] = None
        return d

    # Safe accessors / mutators for external code --------------------------------
    def get_blocks(self):
        """Return the fitted block list or None."""
        return getattr(self, "_blocks", None)

    def get_global_TC(self):
        """Return the fitted global TC matrix or None."""
        return getattr(self, "_global_TC", None)

    def set_global_TC(self, matrix: np.ndarray):
        """Replace the global TC mapping and set mode to 'global'.

        Intended for controlled use-cases (e.g., adding DP noise). Caller must
        ensure the provided matrix has appropriate shape.
        """
        self._global_TC = matrix
        self._mode = "global"

    def set_blocks(self, blocks_list):
        """Replace the internal block list and set mode to 'blocks'.

        `blocks_list` must be a list of `(start, stop, TC)` tuples.
        """
        self._blocks = blocks_list
        self._mode = "blocks"
