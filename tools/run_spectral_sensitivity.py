"""
Spectral Sensitivity Analysis Script (CSV Edition - Robust).

This script runs federated experiments using specific CSV files (MA_A2.csv, MB_B2.csv)
across different spectral channel counts (n_wavelengths) to evaluate compression impact.
"""

import os
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from fedchem.utils.config import load_and_seed_config
from fedchem.utils.real_data import resample_spectra
from fedchem.federated.orchestrator import FederatedOrchestrator
from fedchem.utils.model_registry import instantiate_model
from fedchem.metrics.metrics import rmsep

# Load base config
cfg = load_and_seed_config()

# Channel counts to test
CHANNEL_COUNTS = [32, 64, 128, 256, 512, 1024]

def load_csv_site(path):
    """Load IDRC CSV format: ID, Protein, [Spectra...]"""
    print(f"Loading {path}...")
    df = pd.read_csv(path, low_memory=False)
    df['Protein'] = pd.to_numeric(df['Protein'], errors='coerce')
    df = df.dropna(subset=['Protein'])
    y = df['Protein'].to_numpy().astype(float)
    spec_cols = [c for c in df.columns if c not in ['ID', 'Protein']]
    X_df = df[spec_cols].apply(pd.to_numeric, errors='coerce')
    mask = X_df.isna().any(axis=1)
    if mask.any():
        print(f"Warning: Dropping {mask.sum()} rows with non-numeric spectral data in {path}")
        X_df = X_df.dropna()
        y = y[~mask]
    X = X_df.to_numpy().astype(float)
    return X, y, spec_cols

def run_experiment(n_wavelengths, seed=42):
    print(f"\n>>> Running experiment with n_wavelengths = {n_wavelengths}")
    
    # Load raw data
    X_a, y_a, cols_a = load_csv_site("data/MA_A2.csv")
    X_b, y_b, cols_b = load_csv_site("data/MB_B2.csv")
    
    # Resample both to common n_wavelengths using linear interpolation
    Xr_a, _ = resample_spectra(X_a, col_names=cols_a, n_wavelengths=n_wavelengths, method="interpolate")
    Xr_b, _ = resample_spectra(X_b, col_names=cols_b, n_wavelengths=n_wavelengths, method="interpolate")
    
    clients = [
        {"X": Xr_a, "y": y_a, "name": "MA_A2"},
        {"X": Xr_b, "y": y_b, "name": "MB_B2"}
    ]

    # Simple 80/20 split for internal validation
    def make_eval_fn(clients):
        X_val_list = []
        y_val_list = []
        for c in clients:
            Xc, yc = c["X"], c["y"]
            n = Xc.shape[0]
            split = max(1, int(n * 0.8))
            X_val_list.append(Xc[split:])
            y_val_list.append(yc[split:])
        X_val = np.vstack(X_val_list)
        y_val = np.hstack(y_val_list)
        def eval_fn(model):
            yhat = model.predict(X_val)
            return {"rmsep": rmsep(y_val, yhat)}
        return eval_fn

    eval_fn = make_eval_fn(clients)
    
    # Run Federated Learning (FedAvg)
    orch = FederatedOrchestrator()
    # Force PLSModel
    model = instantiate_model("PLSModel", n_components=10) 
    
    rounds = cfg.get("ROUNDS", 10)
    
    # Use DP if enabled in config
    dp_target_eps = cfg.get("DP_TARGET_EPS", 2.0)
    dp_config = None
    if dp_target_eps is not None and str(dp_target_eps).lower() != 'inf':
        dp_config = {
            "delta": cfg.get("DP_DELTA", 1e-5),
            "target_epsilon": float(dp_target_eps),
        }
    
    # Robustly handle clip_norm
    clip_norm_val = cfg.get("CLIP_NORM")
    if clip_norm_val is None or clip_norm_val == "" or str(clip_norm_val).lower() == "null":
        clip_norm_val = 1.0
    else:
        clip_norm_val = float(clip_norm_val)

    res = orch.run_rounds(
        clients=clients,
        model=model,
        rounds=rounds,
        algo="fedavg",
        dp_config=dp_config,
        clip_norm=clip_norm_val,
        eval_fn=eval_fn,
        seed=seed,
    )
    
    # Extract final metrics
    logs = res.get("logs", [])
    final_rmsep = logs[-1].get("rmsep") if logs else None
    
    total_bytes = sum(int(log.get("bytes_sent", 0)) + int(log.get("bytes_recv", 0)) for log in logs)
    
    return {
        "n_wavelengths": n_wavelengths,
        "final_rmsep": final_rmsep,
        "total_kb": total_bytes / 1024.0
    }

def main():
    results = []
    for count in CHANNEL_COUNTS:
        try:
            res = run_experiment(count)
            if res:
                results.append(res)
        except Exception as e:
            print(f"Failed count {count}: {e}")
            import traceback
            traceback.print_exc()
    
    if not results:
        print("No results collected.")
        return

    # Save results
    df = pd.DataFrame(results)
    output_dir = Path(cfg.get("OUTPUT_DIR", "generated_figures_tables"))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_dir / "spectral_sensitivity_results.csv", index=False)
    print(f"\nResults saved to {output_dir / 'spectral_sensitivity_results.csv'}")
    print(df)
    
    # Generate a simple plot
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:red'
    ax1.set_xlabel('Number of Wavelengths')
    ax1.set_ylabel('Final RMSEP', color=color)
    ax1.plot(df['n_wavelengths'], df['final_rmsep'], marker='o', color=color, label='RMSEP')
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Total Communication (KB)', color=color)
    ax2.plot(df['n_wavelengths'], df['total_kb'], marker='s', color=color, label='Comm (KB)')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Spectral Sensitivity Analysis (CSV Data): RMSEP vs Communication')
    fig.tight_layout()
    plt.savefig(output_dir / "spectral_sensitivity_plot.png")
    print(f"Plot saved to {output_dir / 'spectral_sensitivity_plot.png'}")

if __name__ == "__main__":
    main()
