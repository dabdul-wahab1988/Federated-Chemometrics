Running canonical figure & table experiments
=========================================

This directory contains helper scripts to run representative experiment combos used to generate the main figures and tables for the paper.

Prerequisites
- Python environment with the project's dependencies (e.g., create and activate conda environment `fedchem-dev`).
- The `data/` directory should contain real data (the script checks for `data/ManufacturerA`, `data/ManufacturerB`, ...).

Quick start (single machine sequential)
---------------------------------------

1) Run the `run_for_figures.py` wrapper to execute canonical combos for each figure and output results into `results/figures` (default):

```powershell
python .\scripts\run_for_figures.py --data_root data --out_dir results/figures --seed 42
```

2) After the runs complete, aggregate results into tables suitable for manuscript tables using `generate_core_tables.py`:

```powershell
python .\scripts\generate_core_tables.py --config config.yaml --results_dirs results/figures/figure3_rmsep_methods,results/figures/figure4_conformal_coverage --output_dir generated_figures_tables
```

Notes
- Use `scripts/run_all_objectives.py` for large batch runs (full factorial enumeration); `run_for_figures.py` is designed for small, interpretative runs for figures.
- The tools assume output formats in `tools/run_real_site_experiment.py` (e.g., `methods_summary.csv`, `federated_results.json`, `coverage_summary.json`).
- If a run fails, check the printed error logs and the returned stderr from the called process.

Advanced
- To run `run_all_objectives.py` and produce a large set of experiments for publication, use:

```powershell
python .\scripts\run_all_objectives.py --rounds 50
```

Tip: If your `EXPERIMENTAL_DESIGN` contains many factor levels, the full-factorial enumeration may be large. To avoid excessive memory or extremely long runs, limit the number of enumerated combos or preview them with `scripts/list_experimental_combos.py`.

Example: preview and limit the number of printed combos
```powershell
python .\scripts\list_experimental_combos.py --max-combos 100
```

And when running full experiments, avoid enumerating too many combos without explicit intent by setting `--max-combos` to a reasonable number (or reduce factor lists in `config.yaml`):
```powershell
python .\scripts\run_all_objectives.py --max-combos 1000
If you want to run exactly 122 combos that preserve a core set of factors (e.g., `DP_Target_Eps`, `Federated_Method`, and `Spectral_Drift`), you can randomly sample 122 combos from the full design like this:
```powershell
python .\scripts\run_all_objectives.py --sample-combos 122 --results-dir results
```
This uses reservoir sampling to select 122 representative combinations while respecting the distribution across the full-factorial space.

Note: By default, if the full-factorial enumeration exceeds 122 combos, `run_all_objectives.py` will automatically apply stratified sampling to preserve coverage across DP × Federated_Method × Spectral_Drift (48 strata), ensuring at least 2 combos per stratum (96 combos) and then sampling the remaining 26 combos randomly to reach 122 total combos. You can override this behavior with `--no-stratify` (use random reservoir sampling or full factorial depending on other flags) or force the full factorial via `--full-factorial` (dangerous for large designs).
```

This enumerates the `EXPERIMENTAL_DESIGN` specified in `config.yaml` and launches child runs accordingly.

Contact
- If you want me to add plotting scripts to auto-generate Figures 3–6 from the aggregated outputs, tell me the preferred output format (PNG, PDF, or matplotlib figures saved as pickles) and I will add a `scripts/plot_for_figures.py`.

