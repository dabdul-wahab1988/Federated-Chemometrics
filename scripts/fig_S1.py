
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from pathlib import Path

def parse_spectral_data(df):
    # Identify spectral columns: those that can be converted to float
    spec_cols = [c for c in df.columns if c.replace('.', '', 1).isdigit()]
    # Return as list of floats and corresponding values
    wavelengths = np.array([float(c) for c in spec_cols])
    spectra = df[spec_cols].values
    return wavelengths, spectra

def main():
    root = Path('data')
    out_dir = Path('generated_figures_tables')
    out_dir.mkdir(exist_ok=True, parents=True)
    
    files = {'MA_A2': root / 'MA_A2.csv', 'MB_B2': root / 'MB_B2.csv'}
    colors = {'MA_A2': 'blue', 'MB_B2': 'orange'}
    
    # Check files exist
    for name, fpath in files.items():
        if not fpath.exists():
            print(f"Error: {fpath} not found.")
            return

    # Load data
    data_spectra = {}
    data_wavelengths = {}
    
    for name, fpath in files.items():
        df = pd.read_csv(fpath)
        wl, spectra = parse_spectral_data(df)
        data_wavelengths[name] = wl
        data_spectra[name] = spectra
        print(f"{name}: {len(wl)} channels, {wl[0]} - {wl[-1]} nm, step ~{wl[1]-wl[0]:.2f}")

    # Find intersection for PCA
    wl_A = data_wavelengths['MA_A2']
    wl_B = data_wavelengths['MB_B2']
    
    # Use simple float intersection with tolerance
    common_wl = np.intersect1d(wl_A, wl_B)
    
    if len(common_wl) == 0:
        print("No exact common wavelengths found using intersect1d. checking closer.")
        # Only keep A that are in B
        common_wl = [w for w in wl_A if np.min(np.abs(wl_B - w)) < 1e-5]
        common_wl = np.array(common_wl)
        
    print(f"Common channels for PCA: {len(common_wl)}")
    if len(common_wl) > 0:
        print(f"Common range: {common_wl[0]} - {common_wl[-1]} nm")

    # PCA Data Prep
    pca_inputs = []
    
    if len(common_wl) > 0:
        for name in ['MA_A2', 'MB_B2']:
            wl = data_wavelengths[name]
            spec = data_spectra[name]
            
            # Select columns corresponding to common_wl
            mask = np.isin(wl, common_wl)
            subset_spec = spec[:, mask]
            pca_inputs.append(subset_spec)
            
        pca_matrix = np.vstack(pca_inputs)
        
        # PCA
        pca = PCA(n_components=2)
        scores = pca.fit_transform(pca_matrix)
        expl_var = pca.explained_variance_ratio_ * 100
    else:
        print("Skipping PCA due to no overlap")
        scores = None
    
    # Compute stats on raw data (full range)
    stats = {}
    for name, spectra in data_spectra.items():
        stats[name] = {
            'mean': np.mean(spectra, axis=0),
            'std': np.std(spectra, axis=0),
            'wl': data_wavelengths[name]
        }
    
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. Spectral Appearance (Mean +/- SD)
    ax = axes[0]
    for name, stat in stats.items():
        mu = stat['mean']
        sigma = stat['std']
        wl = stat['wl']
        color = colors[name]
        
        ax.plot(wl, mu, label=name, color=color)
        ax.fill_between(wl, mu - sigma, mu + sigma, color=color, alpha=0.2)
        
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Absorbance (log(1/R))")
    ax.set_title("Representative Spectra (Mean ± SD)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. PCA
    ax = axes[1]
    if scores is not None:
        # Split scores back
        n_a = len(data_spectra['MA_A2'])
        scores_a = scores[:n_a]
        scores_b = scores[n_a:]
        
        ax.scatter(scores_a[:, 0], scores_a[:, 1], c=colors['MA_A2'], label='MA_A2', alpha=0.6, s=15)
        ax.scatter(scores_b[:, 0], scores_b[:, 1], c=colors['MB_B2'], label='MB_B2', alpha=0.6, s=15)
        
        ax.set_xlabel(f"PC1 ({expl_var[0]:.1f}%)")
        ax.set_ylabel(f"PC2 ({expl_var[1]:.1f}%)")
        ax.set_title("PCA Projection of Combined Dataset")
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No overlapping wavelengths for PCA", ha='center')
    
    plt.tight_layout()
    out_path = out_dir / 'Figure_S1.png'
    plt.savefig(out_path, dpi=300)
    print(f"Saved figure to {out_path}")

if __name__ == "__main__":
    main()
