"""Small demo script used by tests.

Exposes `run_demo(seed, n, d, window, q)` returning a dict with 'dp_meta'
containing 'reported_epsilons'. This is intentionally small and independent of
the rest of the tools to ease test coverage.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
from typing import Dict, Any, Optional

from fedchem.ct.pds_transfer import PDSTransfer
from fedchem.privacy.guards import PrivacyGuard


def run_demo(seed: int = 0, n: int = 30, d: int = 16, window: int = 8, q: float = 0.2) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    # generate synthetic reference and target paired set
    X_ref = rng.normal(size=(n, d))
    X_site = 1.1 * X_ref + 0.1 + rng.normal(scale=0.01, size=X_ref.shape)
    # random target; in real usage you'd ensure y, but we only apply PDS to X
    pds = PDSTransfer(window=window, overlap=0, ridge=1e-6, use_global_affine=True)
    pds.fit(X_ref, X_site)
    # apply DP to pds using a simple epsilon schedule and return metadata
    dp_eps = [0.1, 1.0, 10.0]
    dp_meta = {"reported_epsilons": []}
    for eps in dp_eps:
        meta = PrivacyGuard.add_dp_noise_to_pds(pds, epsilon=float(eps), delta=1e-5, clip_feature=1.0, clip_target=1.0, rng=rng, q=q)
        dp_meta["reported_epsilons"].append(meta.get("reported_epsilons") or meta.get("reported_epsilons"))
    return {"dp_meta": dp_meta}


if __name__ == "__main__":
    out = run_demo(seed=0, n=30, d=16, window=8, q=0.2)
    print(out)
