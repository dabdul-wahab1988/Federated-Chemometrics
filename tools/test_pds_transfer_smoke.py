"""Quick smoke test for PDSTransfer implementation.

This script fits PDSTransfer on synthetic paired data and reports diagnostics
and estimated byte sizes for the learned mappings. Intended for quick local
validation; not a unit test harness.
"""
import numpy as np
from fedchem.ct.pds_transfer import PDSTransfer


def make_pair(n=200, d=64, noise=1e-3):
    rng = np.random.default_rng(0)
    # generate a random linear map and bias
    A = rng.normal(size=(d, d)) * 0.1
    b = rng.normal(size=(d,)) * 0.01
    X_target = rng.normal(size=(n, d))
    X_source = X_target @ A + b
    X_source += rng.normal(scale=noise, size=X_source.shape)
    return X_source, X_target


def bytes_from_pds(pds: PDSTransfer) -> int:
    try:
        return int(pds.estimated_bytes())
    except Exception:
        g = pds.get_global_TC() if hasattr(pds, "get_global_TC") else None
        if g is not None:
            return int(getattr(g, "nbytes", 0) or 0)
        blocks = pds.get_blocks() if hasattr(pds, "get_blocks") else None
        if not blocks:
            return 0
        total = 0
        for (_, _, tc) in blocks:
            total += int(getattr(tc, "nbytes", 0) or 0)
        return total


def main():
    Xs, Xt = make_pair(n=300, d=64, noise=1e-4)

    # test with global affine allowed
    p = PDSTransfer(window=16, overlap=0, ridge=1e-6, use_global_affine=True)
    p.fit(Xs, Xt)
    print("mode:", p._mode)
    try:
        diags = p.diagnostics
        print("diag mean_rmse:", diags.mean_rmse)
        print("cond numbers (first 3):", diags.cond_numbers[:3])
    except Exception as e:
        print("diagnostics access error:", e)
    print("bytes estimate:", bytes_from_pds(p))

    # test transform
    Y = p.transform(Xt)
    print("transform shape:", Y.shape)
    print("reconstruction rmse:", float(np.sqrt(np.mean((Y - Xs) ** 2))))

    # test without global affine
    p2 = PDSTransfer(window=8, overlap=2, ridge=1e-6, use_global_affine=False)
    p2.fit(Xs, Xt)
    print("mode (p2):", p2._mode)
    print("bytes estimate (p2):", bytes_from_pds(p2))
    Y2 = p2.transform(Xt)
    print("transform2 rmse:", float(np.sqrt(np.mean((Y2 - Xs) ** 2))))


if __name__ == '__main__':
    main()
