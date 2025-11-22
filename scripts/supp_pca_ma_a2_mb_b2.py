#!/usr/bin/env python
"""
Supplementary PCA and spectral diagnostics for MA_A2 and MB_B2.

- Loads two CSV files: MA_A2.csv and MB_B2.csv (rows = samples, columns = wavelengths).
- Auto-detects numeric spectral columns.
- Runs PCA on mean-centred spectra (optionally standardised).
- Produces a 4-panel figure:

  (A) PC1 vs PC2 scores, coloured by site (MA_A2 vs MB_B2).
  (B) Mean ± SD spectra for each site.
  (C) Difference spectrum (MB_B2 mean - MA_A2 mean).
  (D) Explained variance ratio for first PCs.

Usage (from the directory containing MA_A2.csv and MB_B2.csv):

    python supp_pca_ma_a2_mb_b2.py \
        --ma_csv MA_A2.csv \
        --mb_csv MB_B2.csv \
        --outdir figs_supp \
        --standardize 0

Requirements:
    pip install pandas numpy matplotlib scikit-learn
"""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


def load_spectra(csv_path: Path, site_label: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load spectra from a CSV file.

    Assumptions:
    - Each row is a sample.
    - Numeric columns are spectral intensities at different wavelengths.
    - Non-numeric columns (e.g., IDs) are ignored for PCA.

    Returns
    -------
    spectra : DataFrame
        Numeric spectral matrix (n_samples x n_wavelengths).
    labels : Series
        Site label for each row.
    """
    df = pd.read_csv(csv_path)

    # Keep only numeric columns (float/int)
    numeric_df = df.select_dtypes(include=["number"]).copy()
    if numeric_df.empty:
        raise ValueError(f"No numeric columns found in {csv_path}.")

    labels = pd.Series([site_label] * len(numeric_df), name="site")
    return numeric_df, labels


def run_pca(X: np.ndarray, n_components: int = 3, standardize: bool = False):
    """
    Run PCA on spectral matrix X.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Spectral matrix.
    n_components : int
        Number of principal components.
    standardize : bool
        If True, standardise each variable to unit variance before PCA.

    Returns
    -------
    scores : np.ndarray, shape (n_samples, n_components)
        PCA scores.
    pca : PCA object
        Fitted sklearn PCA instance.
    """
    # Handle missing values
    imputer = SimpleImputer(strategy="mean")
    X = imputer.fit_transform(X)

    # Mean-centre (and optionally standardise)
    if standardize:
        scaler = StandardScaler(with_mean=True, with_std=True)
        X_proc = scaler.fit_transform(X)
    else:
        X_proc = X - X.mean(axis=0, keepdims=True)

    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X_proc)
    return scores, pca


def make_supplementary_figure(
    spectra_ma: pd.DataFrame,
    spectra_mb: pd.DataFrame,
    labels: pd.Series,
    scores: np.ndarray,
    pca,
    out_path: Path,
) -> None:
    """
    Build and save the 4-panel supplementary figure.

    Panels:
      (A) PC1 vs PC2 scores coloured by site.
      (B) Mean ± SD spectra per site.
      (C) Difference of mean spectra (MB_B2 - MA_A2).
      (D) Explained variance ratio of PCs.
    """
    # Combine for PCA scores DataFrame
    scores_df = pd.DataFrame(
        scores, columns=[f"PC{i+1}" for i in range(scores.shape[1])]
    )
    scores_df["site"] = labels.values

    # Wavelength axis from column names (assumed numeric or convertible)
    # If column names are not numeric, just use index as "wavenumber index"
    def _get_wavelengths(df):
        try:
            return np.array(df.columns, dtype=float)
        except ValueError:
            return np.arange(df.shape[1])

    wl_ma = _get_wavelengths(spectra_ma)
    wl_mb = _get_wavelengths(spectra_mb)

    # Sanity check: require same wavelength grid
    if spectra_ma.shape[1] != spectra_mb.shape[1]:
        raise ValueError(
            f"MA_A2 and MB_B2 have different number of spectral channels "
            f"({spectra_ma.shape[1]} vs {spectra_mb.shape[1]})."
        )

    if not np.array_equal(wl_ma, wl_mb):
        # We can still plot, but we will use MA_A2 wavelengths as reference
        wavelengths = wl_ma
    else:
        wavelengths = wl_ma

    # Compute mean and std per site
    mean_ma = spectra_ma.mean(axis=0).values
    std_ma = spectra_ma.std(axis=0).values
    mean_mb = spectra_mb.mean(axis=0).values
    std_mb = spectra_mb.std(axis=0).values

    diff_mb_minus_ma = mean_mb - mean_ma

    # Explained variance
    evr = pca.explained_variance_ratio_

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axA, axB = axes[0]
    axC, axD = axes[1]

    # Panel A – PC1 vs PC2 scores
    for site, marker in [("MA_A2", "o"), ("MB_B2", "s")]:
        mask = scores_df["site"] == site
        axA.scatter(
            scores_df.loc[mask, "PC1"],
            scores_df.loc[mask, "PC2"],
            label=site,
            alpha=0.7,
            marker=marker,
        )
    axA.set_xlabel("PC1 score")
    axA.set_ylabel("PC2 score")
    axA.set_title("(A) PCA score plot (PC1 vs PC2)")
    axA.axhline(0, color="grey", linestyle="--", linewidth=0.5)
    axA.axvline(0, color="grey", linestyle="--", linewidth=0.5)
    axA.grid(True, linestyle="--", alpha=0.3)
    axA.legend()

    # Panel B – Mean ± SD spectra per site
    axB.plot(wavelengths, mean_ma, label="MA_A2 mean", linestyle="-")
    axB.fill_between(
        wavelengths, mean_ma - std_ma, mean_ma + std_ma, alpha=0.2, label="MA_A2 ± SD"
    )
    axB.plot(wavelengths, mean_mb, label="MB_B2 mean", linestyle="--")
    axB.fill_between(
        wavelengths, mean_mb - std_mb, mean_mb + std_mb, alpha=0.2, label="MB_B2 ± SD"
    )
    axB.set_xlabel("Wavelength (a.u.)")
    axB.set_ylabel("Intensity (a.u.)")
    axB.set_title("(B) Mean ± SD spectra (MA_A2 vs MB_B2)")
    axB.grid(True, linestyle="--", alpha=0.3)
    axB.legend(fontsize=8)

    # Panel C – Difference spectrum (MB_B2 - MA_A2)
    axC.plot(wavelengths, diff_mb_minus_ma)
    axC.axhline(0, color="grey", linestyle="--", linewidth=0.5)
    axC.set_xlabel("Wavelength (a.u.)")
    axC.set_ylabel("MB_B2 mean - MA_A2 mean")
    axC.set_title("(C) Difference spectrum (systematic offsets / scatter)")
    axC.grid(True, linestyle="--", alpha=0.3)

    # Panel D – Explained variance ratio
    pcs = np.arange(1, len(evr) + 1)
    axD.bar(pcs, evr * 100.0)
    axD.set_xticks(pcs)
    axD.set_xlabel("Principal component")
    axD.set_ylabel("Explained variance (%)")
    axD.set_title("(D) PCA explained variance")
    axD.grid(True, axis="y", linestyle="--", alpha=0.3)

    fig.suptitle(
        "PCA and spectral diagnostics for MA_A2 vs MB_B2",
        fontsize=14,
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def load_and_align_spectra(ma_csv: Path, mb_csv: Path):
    """
    Load and align spectra from two CSV files based on common wavelengths.

    Parameters
    ----------
    ma_csv : Path
        Path to MA_A2 spectra CSV.
    mb_csv : Path
        Path to MB_B2 spectra CSV.

    Returns
    -------
    aligned_ma : pd.DataFrame
        Aligned spectral data for MA_A2.
    aligned_mb : pd.DataFrame
        Aligned spectral data for MB_B2.
    common_wavelengths : np.ndarray
        Common wavelengths shared by both datasets.
    """
    def load_and_filter(csv_path):
        df = pd.read_csv(csv_path)
        numeric_df = df.select_dtypes(include=["number"])
        numeric_df.columns = pd.to_numeric(numeric_df.columns, errors="coerce")
        numeric_df = numeric_df.dropna(axis=1, how="any")
        return numeric_df

    spectra_ma = load_and_filter(ma_csv)
    spectra_mb = load_and_filter(mb_csv)

    common_wavelengths = np.intersect1d(spectra_ma.columns, spectra_mb.columns)
    if common_wavelengths.size == 0:
        raise ValueError("No common wavelengths found between MA_A2 and MB_B2.")

    aligned_ma = spectra_ma[common_wavelengths]
    aligned_mb = spectra_mb[common_wavelengths]

    return aligned_ma, aligned_mb, common_wavelengths


def main():
    parser = argparse.ArgumentParser(
        description="PCA + spectral supplementary figure for MA_A2 and MB_B2."
    )
    parser.add_argument(
        "--ma_csv",
        type=str,
        default="data/MA_A2.csv",
        help="Path to MA_A2 spectra CSV.",
    )
    parser.add_argument(
        "--mb_csv",
        type=str,
        default="data/MB_B2.csv",
        help="Path to MB_B2 spectra CSV.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="figs_supp",
        help="Directory to save supplementary figure into.",
    )
    parser.add_argument(
        "--standardize",
        type=int,
        default=0,
        help=(
            "If 1, standardise variables before PCA (mean=0, std=1). "
            "If 0, only mean-centre (default)."
        ),
    )
    args = parser.parse_args()

    ma_path = Path(args.ma_csv)
    mb_path = Path(args.mb_csv)
    outdir = Path(args.outdir)

    aligned_ma, aligned_mb, common_wavelengths = load_and_align_spectra(ma_path, mb_path)

    # Combine
    spectra_all = pd.concat([aligned_ma, aligned_mb], axis=0, ignore_index=True)
    labels_all = pd.Series(
        ["MA_A2"] * len(aligned_ma) + ["MB_B2"] * len(aligned_mb), name="site"
    )

    # Run PCA
    scores, pca = run_pca(
        spectra_all.values,
        n_components=5,
        standardize=bool(args.standardize),
    )

    # Build figure
    out_path = outdir / "supp_fig_pca_ma_a2_mb_b2_common_window.png"
    make_supplementary_figure(
        spectra_ma=aligned_ma,
        spectra_mb=aligned_mb,
        labels=labels_all,
        scores=scores,
        pca=pca,
        out_path=out_path,
    )

    print(f"Supplementary PCA figure saved to: {out_path}")
    print(f"Common wavelength range: {common_wavelengths.min()} to {common_wavelengths.max()} ({len(common_wavelengths)} channels)")

if __name__ == "__main__":
    main()
