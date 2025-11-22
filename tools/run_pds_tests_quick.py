"""Quick non-pytest runner for PDSTransfer checks.

This avoids invoking the repo's pytest/coverage configuration and performs
targeted assertions analogous to our unit tests.
"""
import numpy as np
import sys
import pathlib

# Ensure local `src/` is on sys.path so we import the workspace package, not
# an installed `fedchem` from site-packages (if present in the environment).
repo_root = pathlib.Path(__file__).resolve().parents[1]
src_dir = repo_root / "src"
sys.path.insert(0, str(src_dir))

from fedchem.ct.pds_transfer import PDSTransfer


def make_pair(n=200, d=64, noise=1e-3, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(d, d)) * 0.1
    b = rng.normal(size=(d,)) * 0.01
    X_target = rng.normal(size=(n, d))
    X_source = X_target @ A + b
    X_source += rng.normal(scale=noise, size=X_source.shape)
    return X_source, X_target


def main():
    print("Running PDSTransfer quick checks...")
    Xs, Xt = make_pair(n=300, d=64, noise=1e-4)
    p = PDSTransfer(window=16, overlap=0, ridge=1e-6, use_global_affine=True)
    p.fit(Xs, Xt)
    Y = p.transform(Xt)
    assert Y.shape == Xs.shape, "transform shape mismatch"
    di = p.diagnostics
    assert hasattr(di, "mean_rmse")
    eb = p.estimated_bytes()
    assert isinstance(eb, int) and eb >= 0
    assert isinstance(p.is_global(), bool)
    summary = p.to_dict()
    assert "mode" in summary and "estimated_bytes" in summary
    print("Synthetic checks passed. mode=", summary.get("mode"), "estimated_bytes=", summary.get("estimated_bytes"))

    # mismatched shapes
    try:
        Xs2 = np.zeros((10, 20))
        Xt2 = np.zeros((9, 20))
        p2 = PDSTransfer()
        try:
            p2.fit(Xs2, Xt2)
            print("ERROR: mismatched shapes did not raise ValueError")
            sys.exit(2)
        except ValueError:
            print("mismatched shapes correctly raised ValueError")
    except Exception as e:
        print("mismatched shapes test failed:", e)

    # transform before fit
    try:
        p3 = PDSTransfer()
        try:
            p3.transform(np.zeros((5, 8)))
            print("ERROR: transform before fit did not raise")
            sys.exit(2)
        except RuntimeError:
            print("transform-before-fit correctly raised RuntimeError")
    except Exception as e:
        print("transform-before-fit test failed:", e)

    # real-data smoke check (best-effort)
    try:
        from fedchem.utils.real_data import load_idrc_wheat_shootout_site_dict
        from fedchem.utils.config import load_and_seed_config, get_experimental_sites
        print("Real-data loader available; attempting quick smoke run (may take a moment)...")
        cfg = load_and_seed_config()
        config_sites = get_experimental_sites(cfg)
        resample_cfg = cfg.get('RESAMPLE_SPECTRA') if isinstance(cfg, dict) else None
        site_data, ds_meta = load_idrc_wheat_shootout_site_dict(sites=config_sites, data_dir="data", n_wavelengths=64, quick_cap_per_site=50, seed=0, max_transfer_samples=100, resample=resample_cfg)
        sites = list(site_data.items())
        if len(sites) < 2:
            print("Not enough sites found; skipping real-data smoke.")
        else:
            tgt_name, tgt = sites[0]
            tgt_X = tgt["cal"][0]
            ref_list = []
            for name, v in sites[1:5]:
                try:
                    ref_list.append(v["cal"][0][:50])
                except Exception:
                    continue
            if not ref_list:
                print("Not enough pooled reference samples; skipping real-data smoke.")
            else:
                ref_X = np.vstack(ref_list)
                p_real = PDSTransfer(window=16, overlap=0, ridge=1e-6, use_global_affine=True)
                p_real.fit(ref_X, tgt_X[: ref_X.shape[0]])
                print("Real-data PDSTransfer summary:", p_real.to_dict())
                Y = p_real.transform(tgt_X[: ref_X.shape[0]])
                rmse = float(np.sqrt(np.mean((Y - ref_X) ** 2)))
                print("Real-data transform RMSE:", rmse)
    except Exception as e:
        print("Real-data smoke skipped or failed (loader/data missing):", e)

    print("All quick checks complete.")


if __name__ == '__main__':
    main()
