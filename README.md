# Federated Chemometrics

A small research toolkit for evaluating federated chemometric approaches (FedPLS, FedProx, calibration transfer) and conformal prediction methods on multi-instrument spectral data.

This repository contains scripts and modules used to:
- Prepare and validate spectral datasets (MA_A2 / MB_B2 instrument data)
- Run federated learning experiments and baselines (centralized, site-specific, PDS, SBC)
- Evaluate privacy-utility trade-offs (differential privacy and communication costs)
- Produce figures and tables for manuscript results
- Generate and evaluate conformal prediction intervals

Project highlights
- Reproducible experiment pipeline for figure/table generation
- Data loaders for real (ManufacturerA/ManufacturerB) CSVs and example datasets
- Visualization and figure generation scripts under `scripts/` and `fig_Tab/`
- Support for differential privacy, conformal prediction, and transfer budget studies

Quick start
----------
Prerequisites:
- Python 3.10+ (project uses Python 3.10 in CI)
- Conda or virtual environment (recommended)

Install dependencies (conda + pip recommended):

```powershell
conda create -n fedchem-dev python=3.10 -y
conda activate fedchem-dev
pip install -r requirements.txt
```

Run the canonical figure pipeline (small interpretative runs used for figures):

```powershell
python scripts/run_for_figures.py --data_root data --out_dir results/figures --seed 42
```

Aggregate the results into paper tables:

```powershell
python scripts/generate_core_tables.py --config config.yaml --results_dirs results/figures/figure3_rmsep_methods,results/figures/figure4_conformal_coverage --output_dir final2/generated_figures_tables
```

Run all canonical objectives (large, full factorial runs; caution — this can be expensive):

```powershell
python scripts/run_all_objectives.py --rounds 50 --results_dir results
```

Spectral exploratory scripts and diagnostics
- The `scripts/supp_pca_ma_a2_mb_b2.py` script:
  - Loads and aligns spectral datasets by wavelength.
  - Runs PCA on the combined MA_A2 + MB_B2 spectra.
  - Saves a 4-panel supplementary figure showing PC scores, mean ± SD spectra, difference spectrum, and explained variance.

Usage example (run PCA and save supplementary figure):

```powershell
python scripts/supp_pca_ma_a2_mb_b2.py --ma_csv data/MA_A2.csv --mb_csv data/MB_B2.csv --outdir figs_supp --standardize 0
```

Notes & Best Practices
---------------------
- Data: The `data/` directory included sample/real instrument files in `ManufacturerA` and `ManufacturerB` subfolders. The scripts expect typical spectral CSVs where numeric column names are wavelengths (e.g., `730.0`, `730.5`, ...). If column names that cannot be converted to floats exist, those columns will be ignored for alignment/PCA.
- Wavelength alignment: The `supp_pca` script computes and uses the intersection of wavelengths shared by both instruments to ensure PCA is performed on a consistent common grid.
- Missing data: The PCA script includes a simple imputer to handle NaNs (mean imputation per wavelength column). Remove or replace this behavior if you prefer stricter filtering.

Development & running tests
---------------------------
Run unit tests with pytest (project currently includes several tests and smoke tests):

```powershell
pytest -q
```

Developer Tools & Utilities
---------------------------
- `tools/` contains helper scripts for debugging and dataset preparation.
- `scripts/` contains experiment wrappers and figure generation scripts.
- `src/fedchem/` is the Python package that contains models, federated orchestrators, evaluation code, and conformal prediction utilities.

Contributing
------------
Please consult `CONTRIBUTING.md` for contribution guidelines. In brief:
- Use small feature branches and open PRs
- Include tests for non-trivial changes
- Keep data and heavy artifacts out of the main repo

License
-------
This project is released under the `LICENSE` included in the repository.

Contact / Support
-----------------
If you need help or want me to add more automation (e.g., GitHub Actions to auto-run canonical experiments or deploy docs), tell me what you'd like automated and I can add it.

---

Thanks for using Federated Chemometrics — good luck with your experiments!