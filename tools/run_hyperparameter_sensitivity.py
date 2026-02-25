"""
Final Hyperparameter Sensitivity Analysis Script.

Generates definitive data for:
1. PDS w x lambda Grid (RMSEP)
2. DP Clip Norm C (RMSEP & Clipped Fraction)
3. Rounds R (Convergence)
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from fedchem.utils.config import load_and_seed_config
from fedchem.utils.real_data import resample_spectra
from fedchem.federated.orchestrator import FederatedOrchestrator
from fedchem.utils.model_registry import instantiate_model
from fedchem.metrics.metrics import rmsep
from fedchem.ct.pds_transfer import PDSTransfer

# Setup
cfg = load_and_seed_config()
output_dir = Path(cfg.get("OUTPUT_DIR", "generated_figures_tables"))
output_dir.mkdir(parents=True, exist_ok=True)

def load_data():
    df_a = pd.read_csv("data/MA_A2.csv")
    df_b = pd.read_csv("data/MB_B2.csv")
    def prep(df):
        y = df['Protein'].to_numpy().astype(float)
        spec_cols = [c for c in df.columns if c not in ['ID', 'Protein']]
        X = df[spec_cols].to_numpy().astype(float)
        return X, y, spec_cols
    Xa, ya, ca = prep(df_a)
    Xb, yb, cb = prep(df_b)
    Xra, _ = resample_spectra(Xa, col_names=ca, n_wavelengths=128)
    Xrb, _ = resample_spectra(Xb, col_names=cb, n_wavelengths=128)
    return Xra, ya, Xrb, yb

Xra, ya, Xrb, yb = load_data()

def run_pds_grid():
    results = []
    for w in [16, 32, 64]:
        for lam in [1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
            pds = PDSTransfer(window=w, ridge=lam, use_global_affine=False).fit(Xra[:40], Xrb[:40])
            X_t = pds.transform(Xrb[40:])
            master = instantiate_model("PLSModel", n_components=10).fit(Xra, ya)
            err = rmsep(yb[40:], master.predict(X_t))
            results.append({"w": w, "lambda": lam, "RMSEP": float(err)})
    return pd.DataFrame(results)

def run_fl_grid():
    results = []
    # Test C and R simultaneously under moderate DP (eps=1.0)
    split = int(len(Xra) * 0.8)
    X_val = np.vstack([Xra[split:], Xrb[split:]])
    y_val = np.hstack([ya[split:], yb[split:]])
    clients = [{"X": Xra[:split], "y": ya[:split]}, {"X": Xrb[:split], "y": yb[:split]}]
    
    for C in [0.5, 1.0, 2.0]:
        for R in [1, 5, 10]:
            orch = FederatedOrchestrator()
            res = orch.run_rounds(
                clients=clients, model=instantiate_model("PLSModel", n_components=5),
                rounds=R, algo="fedavg", dp_config={"delta": 1e-5, "target_epsilon": 1.0},
                clip_norm=C, eval_fn=lambda m: {"rmsep": rmsep(y_val, m.predict(X_val))}, seed=42
            )
            final = res["logs"][-1]
            results.append({
                "C": C, "rounds": R, "RMSEP": final.get("rmsep"),
                "avg_clip_frac": np.mean([l.get("clip_fraction", 0) or 0 for l in res["logs"]])
            })
    return pd.DataFrame(results)

print("Regenerating grids...")
df_pds = run_pds_grid()
df_fl = run_fl_grid()
df_pds.to_csv(output_dir / "hparam_pds_grid.csv", index=False)
df_fl.to_csv(output_dir / "hparam_fl_grid.csv", index=False)
print("Done.")
