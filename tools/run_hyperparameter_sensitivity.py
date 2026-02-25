"""
Hyperparameter Sensitivity Analysis Script (Fixed & Enhanced).

Evaluates impact of:
- PDS Window Size (w) - explicitly testing block-wise resolution.
- Ridge Penalty (lambda)
- DP Clip Norm (C) - capturing actual utility (RMSEP).
- Communication Rounds (R) - verifying convergence.

Outputs structured CSVs for the Supplementary Information.
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
    
    # Resample to canonical 128
    Xra, _ = resample_spectra(Xa, col_names=ca, n_wavelengths=128)
    Xrb, _ = resample_spectra(Xb, col_names=cb, n_wavelengths=128)
    return Xra, ya, Xrb, yb

Xra, ya, Xrb, yb = load_data()

# Shared eval helper
def make_eval_fn(X_val, y_val):
    def eval_fn(model):
        yhat = model.predict(X_val)
        return {"rmsep": rmsep(y_val, yhat)}
    return eval_fn

def run_pds_grid():
    results = []
    windows = [16, 32, 64]
    ridges = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    
    k = 40
    X_ref, y_ref = Xra[:k], ya[:k]
    X_site, y_site = Xrb[:k], yb[:k]
    X_test, y_test = Xrb[k:], yb[k:]
    
    master = instantiate_model("PLSModel", n_components=10).fit(Xra, ya)
    
    for w in windows:
        for lam in ridges:
            # IMPORTANT: Disable global affine to force window-size sensitivity testing
            pds = PDSTransfer(window=w, ridge=lam, use_global_affine=False).fit(X_ref, X_site)
            X_trans = pds.transform(X_test)
            err = rmsep(y_test, master.predict(X_trans))
            results.append({"w": w, "lambda": lam, "RMSEP": err})
            
    return pd.DataFrame(results)

def run_fl_sensitivity():
    """Combined Clip and Round sensitivity."""
    results = []
    clips = [0.5, 1.0, 2.0]
    rounds_list = [1, 5, 10]
    epsilons = [0.1, 1.0, 10.0]
    
    # Use 20% pooled data for validation
    split = int(len(Xra) * 0.8)
    X_val = np.vstack([Xra[split:], Xrb[split:]])
    y_val = np.hstack([ya[split:], yb[split:]])
    eval_fn = make_eval_fn(X_val, y_val)
    
    clients = [
        {"X": Xra[:split], "y": ya[:split]},
        {"X": Xrb[:split], "y": yb[:split]}
    ]
    
    for C in clips:
        for eps in epsilons:
            for R in rounds_list:
                orch = FederatedOrchestrator()
                res = orch.run_rounds(
                    clients=clients,
                    model=instantiate_model("PLSModel", n_components=5),
                    rounds=R,
                    algo="fedavg",
                    dp_config={"delta": 1e-5, "target_epsilon": eps},
                    clip_norm=C,
                    eval_fn=eval_fn,
                    seed=42
                )
                final_rmsep = res["logs"][-1].get("rmsep")
                # Handle None values in clip_fraction log
                clip_fracs = [l.get("clip_fraction") for l in res["logs"]]
                clean_clip_fracs = [f if f is not None else 0.0 for f in clip_fracs]
                avg_clip_frac = np.mean(clean_clip_fracs)
                
                results.append({
                    "C": C, 
                    "eps": eps, 
                    "rounds": R, 
                    "RMSEP": final_rmsep,
                    "avg_clipped_fraction": avg_clip_frac
                })
            
    return pd.DataFrame(results)

print("Running Fixed PDS Grid...")
df_pds = run_pds_grid()
df_pds.to_csv(output_dir / "hparam_pds_grid.csv", index=False)

print("Running Enhanced FL Sensitivity (Clips & Rounds)...")
df_fl = run_fl_sensitivity()
df_fl.to_csv(output_dir / "hparam_fl_grid.csv", index=False)

print("\n--- RESULTS PREVIEW (PDS) ---")
print(df_pds.groupby('w')['RMSEP'].mean())

print("\n--- RESULTS PREVIEW (ROUNDS) ---")
# filter for typical DP setting to show convergence
df_sub = df_fl[df_fl['C']==1.0]
print(df_sub.groupby('rounds')['RMSEP'].mean())

print("\nAnalysis complete.")
