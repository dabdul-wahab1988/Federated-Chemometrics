"""
Script for Objective 5: Benchmark against baselines on the real 5-site dataset.

Generates Figure 5 and Table 5 comparing Centralized, Site-specific, and PDS transfer.
Defaults to the persistent 5-site data (Tecator fallback) and respects the
global site-count override. Also writes a small manifest for traceability.

- Env toggles:
- FEDCHEM_USE_TECATOR=1: force Tecator dataset (requires FEDCHEM_ALLOW_DOWNLOAD=1 first run)
- FEDCHEM_NUM_SITES: override site count (default 5)
- FEDCHEM_QUICK=1: faster run (reduces per-site samples; doesn’t cap sites if override set)
- FEDCHEM_PDS_TRANSFER_N: transfer set size per-site (default 40)
"""

import inspect
import os
import platform
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fedchem.ct.pds_transfer import PDSTransfer
from fedchem.metrics.metrics import mae, r2, rmsep
from fedchem.utils.config import load_and_seed_config, get_experimental_sites, get_data_config
from fedchem.utils.manifest_utils import resolve_design_version
from fedchem.utils.model_registry import MODEL_REGISTRY, instantiate_model
from fedchem.utils.real_data import load_real_site_dict, resample_spectra
from fedchem.federated.orchestrator import FederatedOrchestrator
from fedchem.models.parametric_pls import ParametricPLSModel

cfg = load_and_seed_config()
config = cfg
config_sites = get_experimental_sites(cfg)
data_config = get_data_config(cfg)

# Default visual configuration (tweak these values to change figure appearance)
VISUAL = {
    # overall base font size (used for fallback)
    "base_font_size": 12,
    # figure sizing (width, height) in inches for the entire canvas
    "figsize": (18, 14),
    # top-row height ratio vs bottom-row
    "height_ratios": (1, 2),
    # subplot spacing (passed to fig.subplots_adjust)
    "left": 0.06,
    "right": 0.98,
    "top": 0.96,
    "bottom": 0.06,
    "wspace": 0.32,
    "hspace": 0.36,
    # text sizes
    "title_size": 14,
    "axis_label_size": 14,
    "tick_label_size": 14,
    "legend_fontsize": 14,
    # marker sizes
    "line_marker_size": 6,
    "parity_marker_size": 18,
}
plt.rcParams.update({"font.size": VISUAL["base_font_size"]})

output_dir = Path(str(config.get('OUTPUT_DIR', 'generated_figures_tables')))
output_dir.mkdir(exist_ok=True)

def _coerce_int(value: Any, default: int | None = None) -> int | None:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_float(value: Any, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
        if lowered == "":
            return default
    if value is None:
        return default
    return bool(value)


def _resolve_bool_from_env(env_name: str, cfg_value: Any, default: bool = False) -> bool:
    env_val = os.environ.get(env_name)
    if env_val is not None:
        return _coerce_bool(env_val, default)
    return _coerce_bool(cfg_value, default)


def _resolve_fedpls_enabled() -> bool:
    method = _resolve_fedpls_method()
    if not method:
        return False
    fedpls_cfg_val = config.get('USE_FEDPLS')
    return _resolve_bool_from_env("FEDCHEM_USE_FEDPLS", fedpls_cfg_val, False)


def _resolve_fedpls_method() -> str | None:
    env_method = os.environ.get("FEDCHEM_FEDPLS_METHOD")
    if env_method:
        return env_method
    if 'FEDPLS_METHOD' in config:
        return config.get('FEDPLS_METHOD')
    return None


DEFAULT_N_WAVELENGTHS = _coerce_int(config.get('DEFAULT_N_WAVELENGTHS'), 256) or 256


def _parse_optional_int(name: str, default: int | None = None) -> int | None:
    return _coerce_int(os.environ.get(name), default)


def _resolve_model_label() -> str:
    """Resolve the pipeline model label.

    Priority (highest -> lowest):
      1. Environment `FEDCHEM_PIPELINE_MODEL`
      2. `config["PIPELINE"]["model"]` (if present)
      3. Legacy env `FEDCHEM_USE_LINEAR` (keeps previous behavior)
      4. Default to `PLSModel` to preserve previous default behavior
    """
    env_label = os.environ.get("FEDCHEM_PIPELINE_MODEL")
    if env_label:
        return env_label
    # Safely resolve pipeline config to a dict to avoid calling .get on None
    pipeline_cfg = config.get("PIPELINE")
    if not isinstance(pipeline_cfg, dict):
        pipeline_cfg = {}
    cfg_label = pipeline_cfg.get("model")
    if isinstance(cfg_label, str) and cfg_label:
        return cfg_label
    # fallback to legacy toggle
    if os.environ.get("FEDCHEM_USE_LINEAR", "0") == "1":
        return "LinearModel"
    return "PLSModel"


def _resolve_ct_federated_variant() -> str | None:
    env = os.environ.get("FEDCHEM_CT_FEDERATED_VARIANT")
    if env:
        return str(env).strip()
    cfg_val = config.get("CT_FEDERATED_VARIANT")
    if isinstance(cfg_val, str) and cfg_val:
        return cfg_val.strip()
    return None


def _align_feature_matrices(arrays: list[np.ndarray], target_n_wavelengths: int | None = None) -> list[np.ndarray]:
    """Trim all arrays in `arrays` to the minimum number of columns present.

    This helps when datasets have differing wavelength counts; we conservatively
    trim to the minimum width to allow concatenation/stacking.
    """
    if not arrays:
        return arrays
    try:
        arrs = [np.asarray(a) for a in arrays]
    except Exception:
        return arrays
    widths = [a.shape[1] for a in arrs if a.ndim == 2 and a.shape[1] > 0]
    if not widths:
        return arrs
    min_w = min(widths)
    # Prefer resampling to `target_n_wavelengths` (if provided) to preserve domain structure
    if target_n_wavelengths is not None and target_n_wavelengths > 0:
        # Ensure we only request downsampling when target <= current widths; pick min width across arrays
        effective_target = min(min_w, int(target_n_wavelengths)) if min_w is not None else int(target_n_wavelengths)
        # If effective_target equals existing min_w, we will downsample/trim all to min_w
        if effective_target <= 0:
            effective_target = min_w
        # Call resample_spectra per array; method resolution via env var
        method = os.environ.get("FEDCHEM_RESAMPLE_METHOD", "interpolate")
        resampled = []
        for arr in arrs:
            try:
                Xr, _ = resample_spectra(np.asarray(arr), col_names=None, n_wavelengths=effective_target, method=method)
                resampled.append(Xr)
            except Exception:
                # fallback to trimming if resample fails
                if arr.ndim == 2 and arr.shape[1] > min_w:
                    resampled.append(arr[:, :min_w])
                else:
                    resampled.append(arr)
        return resampled
    if any(a.shape[1] != min_w for a in arrs if a.ndim == 2):
        # Trim to min_w
        trimmed = [a[:, :min_w] if (a.ndim == 2 and a.shape[1] > min_w) else a for a in arrs]
        return trimmed
    return arrs


def _get_base_model(*args, **kwargs):
    """Instantiate the selected base model via the registry."""
    label = _resolve_model_label()
    pipeline_cfg = config.get("PIPELINE") if isinstance(config.get("PIPELINE"), dict) else {}
    # Ensure pipeline_cfg is a dict
    pipeline_cfg = pipeline_cfg or {}
    # Attempt to find the class in the registry for signature introspection
    cls = MODEL_REGISTRY.get(label)
    ctor_kwargs = {}
    if cls is not None:
        try:
            sig = inspect.signature(cls.__init__)
            # allowed params excluding 'self' and varargs
            allowed = [p for p in sig.parameters.keys() if p not in ("self", "args", "kwargs")]
            for k, v in pipeline_cfg.items():
                if k in allowed:
                    ctor_kwargs[k] = v
        except Exception:
            # best-effort: if introspection fails, avoid passing any kwargs
            ctor_kwargs = {}
    else:
        # Unknown class; let instantiate_model raise the clearer error
        ctor_kwargs = {}

    # Merge explicit kwargs provided at call time (they override pipeline cfg)
    ctor_kwargs.update(kwargs)
    return instantiate_model(label, *args, **ctor_kwargs)


def _prepare_data(
    n_sites: int = 5,
    seed: int = 42,
    quick: bool = False,
    force_tecator: bool = False,
    n_wavelengths: int | None = DEFAULT_N_WAVELENGTHS,
    max_transfer_samples: int | None = None,
    config_sites: list[str] | None = None,
):
    """Load the real multi-site dataset (5-site persistent preferred, Tecator fallback)."""
    data, meta = load_real_site_dict(
        n_sites=n_sites,
        seed=seed,
        quick=quick,
        force_tecator=force_tecator,
        n_wavelengths=n_wavelengths,
        max_transfer_samples=max_transfer_samples,
        config_sites=config_sites,
    )
    return data, meta

def _compute_benchmarks(data: dict[str, dict[str, np.ndarray]], transfer_n: int = 40):
    methods = ['Centralized', 'Site-specific', 'PDS', 'SBC']
    sites = sorted(list(data.keys()))
    method_rmse = {m: [] for m in methods}
    per_site_preds = {}
    # Determine common minimum number of wavelengths across sites and trim per-site arrays
    try:
        min_waves = min([int(data[s]["X"].shape[1]) for s in sites if data[s]["X"].ndim == 2])
    except Exception:
        min_waves = None
    # Pooled train for centralized: use trimmed features to ensure models align
    if min_waves is not None:
        pool_X_list = [data[s]["X"][:80, :min_waves] for s in sites if data[s]["X"].shape[0] >= 100]
    else:
        pool_X_list = [data[s]["X"][:80] for s in sites if data[s]["X"].shape[0] >= 100]
    pool_y_list = [data[s]["y"][:80] for s in sites if data[s]["y"].shape[0] >= 100]
    if len(pool_X_list) > 0:
        # Align/resample features across sites to pipeline width if needed
        pool_X_list = _align_feature_matrices(pool_X_list, target_n_wavelengths=DEFAULT_N_WAVELENGTHS)
        X_pool = np.vstack(pool_X_list)
        y_pool = np.hstack(pool_y_list)
    else:
        # fallback: allow shorter splits
        pool_X_list = [data[s]["X"][: max(1,int(0.6*data[s]["X"].shape[0]))] for s in sites]
        pool_y_list = [data[s]["y"][: max(1,int(0.6*data[s]["y"].shape[0]))] for s in sites]
        pool_X_list = _align_feature_matrices(pool_X_list, target_n_wavelengths=DEFAULT_N_WAVELENGTHS)
        X_pool = np.vstack(pool_X_list)
        y_pool = np.hstack(pool_y_list)
    # instantiate and fit with any accepted fit kwargs (e.g., ridge for LinearModel)
    central_model = _get_base_model()
    fit_kwargs = {}
    pipeline_cfg = config.get("PIPELINE") if isinstance(config.get("PIPELINE"), dict) else {}
    if pipeline_cfg:
        sig = inspect.signature(central_model.fit)
        for k, v in pipeline_cfg.items():
            if k in sig.parameters and k not in ("X", "y"):
                fit_kwargs[k] = v
    central = central_model.fit(X_pool, y_pool, **fit_kwargs)
    for s in sites:
        X_orig = data[s]["X"]; y = data[s]["y"]
        X = X_orig[:, :min_waves] if min_waves is not None else X_orig
        n = X.shape[0]
        n_tr = max(1, int(0.8 * n))
        X_tr, y_tr, X_te, y_te = X[:n_tr], y[:n_tr], X[n_tr:], y[n_tr:]
        model_site = _get_base_model()
        # reuse fit_kwargs logic per-site (signature may accept e.g., ridge)
        site_fit_kwargs = {}
        if pipeline_cfg:
            sig_site = inspect.signature(model_site.fit)
            for k, v in pipeline_cfg.items():
                if k in sig_site.parameters and k not in ("X", "y"):
                    site_fit_kwargs[k] = v
        model_site = model_site.fit(X_tr, y_tr, **site_fit_kwargs)
        yhat_site = model_site.predict(X_te)
        method_rmse['Site-specific'].append(rmsep(y_te, yhat_site))
        # Trim X_te to pooled model width before predicting with central model (central model trained on pool width)
        try:
            pool_w = int(X_pool.shape[1])
            if X_te.shape[1] != pool_w:
                X_te_central = X_te[:, :pool_w]
                X_tr_central = X_tr[:, :pool_w]
            else:
                X_te_central = X_te
                X_tr_central = X_tr
        except Exception:
            X_te_central = X_te
            X_tr_central = X_tr
        yhat_c = central.predict(X_te_central)
        method_rmse['Centralized'].append(rmsep(y_te, yhat_c))
        # More robust PDS defaults for stability on synthetic spectra
        k = min(transfer_n, n_tr, X_pool.shape[0])
        # For PDS transform, ensure features align between pool and target training
        try:
            pool_for_pds = X_pool
            if X_tr.shape[1] != pool_for_pds.shape[1]:
                # make sure to trim to min width
                minw = min(X_tr.shape[1], pool_for_pds.shape[1])
                pool_for_pds = pool_for_pds[:, :minw]
                X_tr_for_pds = X_tr[:, :minw]
            else:
                X_tr_for_pds = X_tr
        except Exception:
            pool_for_pds = X_pool
            X_tr_for_pds = X_tr
        pds = PDSTransfer(window=32, overlap=16, ridge=1e-1).fit(pool_for_pds[:k], X_tr_for_pds[:k])
        # attach estimated bytes for this site's transfer mapping
        try:
            pds_bytes = int(pds.estimated_bytes())
        except Exception:
            pds_bytes = 0
        # ensure X_te transformed for PDS is trimmed to pds width
        try:
            minw2 = int(pool_for_pds.shape[1])
            X_te_pds = X_te[:, :minw2] if X_te.shape[1] != minw2 else X_te
        except Exception:
            X_te_pds = X_te
        yhat_pds = central.predict(pds.transform(X_te_pds))
        method_rmse['PDS'].append(rmsep(y_te, yhat_pds))
        # Simple slope/bias correction (SBC) on centralized predictions using site train split
        # for SBC, use central prediction on trimmed training split
        yhat_tr = central.predict(X_tr_central)
        A = np.vstack([yhat_tr, np.ones_like(yhat_tr)]).T
        coeffs, *_ = np.linalg.lstsq(A, y_tr, rcond=None)
        a, b = coeffs
        yhat_adj = a * yhat_c + b
        method_rmse['SBC'].append(rmsep(y_te, yhat_adj))
        per_site_preds[s] = {
            "y_te": y_te,
            "yhat_pds": yhat_pds,
            "yhat_central": yhat_c,
            "yhat_site": yhat_site,
            "pds_bytes": int(pds_bytes),
        }
    # Additional CT federated variants
    ct_variant = _resolve_ct_federated_variant()
    if ct_variant:
        # pooled training set used for calibration in local_then_pooled & global calibration
        X_pool = X_pool if 'X_pool' in locals() else np.vstack(pool_X_list)
        y_pool = y_pool if 'y_pool' in locals() else np.hstack(pool_y_list)
        if ct_variant == "local_then_pooled":
            methods.append("Local_then_pooled")
            local_rmse = []
            local_preds = {}
            for idx, s in enumerate(sites):
                X = data[s]["X"]; y = data[s]["y"]
                n = X.shape[0]
                n_tr = max(1, int(0.8 * n))
                X_tr, y_tr, X_te, y_te = X[:n_tr], y[:n_tr], X[n_tr:], y[n_tr:]
                model_site = _get_base_model()
                try:
                    model_site.fit(X_tr, y_tr)
                    y_pool_pred = model_site.predict(X_pool)
                    A = np.vstack([y_pool_pred, np.ones_like(y_pool_pred)]).T
                    coeffs, *_ = np.linalg.lstsq(A, y_pool, rcond=None)
                    a, b = coeffs
                    yhat_adj = a * model_site.predict(X_te) + b
                except Exception:
                    # fallback to site-specific predictions if calibration or fit fails
                    yhat_adj = model_site.predict(X_te)
                local_rmse.append(rmsep(y_te, yhat_adj))
                local_preds[s] = {"y_te": y_te, "yhat_local_then_pooled": yhat_adj}
            method_rmse["Local_then_pooled"] = [float(r) for r in local_rmse]
            # add to per-site preds for downstream plotting
            for k, v in local_preds.items():
                per_site_preds[k].update(v)

        if ct_variant == "local_then_secure_aggregate":
            methods.append("Local_then_secure_aggregate")
            # Fit local models, extract linear params and average into a global model
            weights, biases = [], []
            # Determine a parametric model class to use for local parameter extraction
            base_model_sample = _get_base_model()
            parametric_cls = None
            try:
                # If the class exposes a params API (get_params/set_params), prefer it
                if hasattr(base_model_sample, "get_params") and hasattr(base_model_sample, "set_params"):
                    parametric_cls = type(base_model_sample)
                else:
                    # Prefer ParametricPLSModel if available, otherwise fall back to LinearModel
                    parametric_cls = ParametricPLSModel
            except Exception:
                parametric_cls = ParametricPLSModel
            for s in sites:
                X = data[s]["X"]; y = data[s]["y"]
                n = X.shape[0]
                n_tr = max(1, int(0.8 * n))
                X_tr, y_tr, X_te, y_te = X[:n_tr], y[:n_tr], X[n_tr:], y[n_tr:]
                try:
                    m = parametric_cls()
                except Exception:
                    m = _get_base_model()
                try:
                    m.fit(X_tr, y_tr)
                    params = None
                    try:
                        params = m.get_params()
                    except Exception:
                        params = None
                    if params and "w" in params:
                        weights.append(np.asarray(params.get("w", np.zeros(0))).astype(float))
                        biases.append(float(np.asarray(params.get("b", np.array([0.0]))).ravel()[0]))
                except Exception:
                    continue
            if weights:
                w_avg = np.mean(np.vstack(weights), axis=0)
                b_avg = float(np.mean(np.asarray(biases))) if biases else 0.0
                # Evaluate aggregated model
                from fedchem.models.linear import LinearModel
                global_model = LinearModel()
                try:
                    global_model.set_params({"w": w_avg, "b": np.array([b_avg], dtype=float)})
                except Exception:
                    pass
                # use pooled validation as test pool for aggregated model
                yhat_pool = global_model.predict(np.vstack([data[s]['X'][int(0.8*data[s]['X'].shape[0]):] for s in sites]))
                y_pool_test = np.hstack([data[s]['y'][int(0.8*data[s]['y'].shape[0]):] for s in sites])
                method_rmse["Local_then_secure_aggregate"] = [float(rmsep(y_pool_test, yhat_pool))]
            else:
                # ensure method key exists even when aggregation couldn't be performed
                method_rmse["Local_then_secure_aggregate"] = [float(np.nan)]

        if ct_variant == "global_calibrate_after_fed":
            methods.append("Global_calibrate_after_fed")
            # Train a federated global model and then calibrate on pooled transfer set
            try:
                # build clients list for orchestrator
                clients_list = []
                for s in sites:
                    X = data[s]["X"]; y = data[s]["y"]
                    clients_list.append({"X": X, "y": y})
                orch = FederatedOrchestrator()
                fed_res = orch.run_rounds(clients=clients_list, model=_get_base_model(), rounds=10, algo="fedavg")
                fed_model = fed_res.get("model") if isinstance(fed_res, dict) else None
                if fed_model is not None:
                    # pooled calibration: use first transfer_n pooled samples from sites
                    pool_X_list2 = [data[s]["X"][:transfer_n] for s in sites if data[s]["X"].shape[0] >= transfer_n]
                    pool_y_list2 = [data[s]["y"][:transfer_n] for s in sites if data[s]["y"].shape[0] >= transfer_n]
                    if pool_X_list2:
                        X_pool2 = np.vstack(pool_X_list2)
                        y_pool2 = np.hstack(pool_y_list2)
                    else:
                        X_pool2 = X_pool
                        y_pool2 = y_pool
                    y_pred_pool = fed_model.predict(X_pool2)
                    A = np.vstack([y_pred_pool, np.ones_like(y_pred_pool)]).T
                    coeffs, *_ = np.linalg.lstsq(A, y_pool2, rcond=None)
                    a, b = coeffs
                    # Evaluate on per-site test sets similar to central benchmark
                    y_pred_adj = a * fed_model.predict(np.vstack([data[s]["X"][int(0.8*data[s]["X"].shape[0]):] for s in sites])) + b
                    y_test_all = np.hstack([data[s]["y"][int(0.8*data[s]["y"].shape[0]):] for s in sites])
                    method_rmse["Global_calibrate_after_fed"] = [float(rmsep(y_test_all, y_pred_adj))]
                else:
                    method_rmse["Global_calibrate_after_fed"] = [float(np.nan)]
            except Exception:
                method_rmse["Global_calibrate_after_fed"] = [float(np.nan)]
    return methods, sites, method_rmse, per_site_preds

def generate_figure_5(
    data=None,
    ds_meta=None,
    transfer_sizes: list[int] | None = None,
    rep_site: str | None = None,
    precomputed: dict | None = None,
):
    """Figure 5: Benchmarks.

    (a) Mean RMSEP per method
    (b) RMSEP vs transfer set size for PDS (averaged over sites)
    (c) Parity plot for representative site using PDS
    """
    if data is None or ds_meta is None:
        data, ds_meta = _prepare_data(config_sites=config_sites)
    if transfer_sizes is None:
        transfer_sizes = [10, 20, 40]

    # Baseline with default transfer size (for panel a). Allow precomputed to ensure consistency across artifacts.
    if precomputed is not None:
        methods = precomputed["methods"]
        sites = precomputed["sites"]
        method_rmse = precomputed["method_rmse"]
        per_site_preds = precomputed.get("per_site_preds", {})
    else:
        methods, sites, method_rmse, per_site_preds = _compute_benchmarks(data, transfer_n=max(transfer_sizes))

    # Layout: top row = panels 5a and 5b (1x2), bottom = panel 5c as 2x3 parity subplots for sites
    from matplotlib import gridspec
    fig = plt.figure(figsize=VISUAL["figsize"])
    # Two rows: top for (a,b), bottom for grid of parity plots
    outer = gridspec.GridSpec(2, 1, height_ratios=[1, 2], hspace=0.3)

    # Top row: 1x2 subplots for 5a and 5b
    top = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0], wspace=0.3)
    ax_a = fig.add_subplot(top[0, 0])
    ax_b = fig.add_subplot(top[0, 1])

    # (a) Mean RMSEP per method
    means = [float(np.mean(method_rmse[m])) for m in methods]
    # (a) Use log scale for y-axis to better show differences across orders of magnitude
    means_arr = np.array(means, dtype=float)
    # avoid non-positive values on log scale
    means_plot = means_arr.copy()
    means_plot[means_plot <= 0] = 1e-12
    ax_a.bar(methods, means_plot)
    ax_a.set_yscale('log')
    ax_a.set_ylabel("RMSEP", fontsize=VISUAL["axis_label_size"])
    ax_a.set_title("5(a) Benchmark (mean RMSEP, log scale)", fontsize=VISUAL["title_size"]) 
    ax_a.tick_params(axis='both', which='major', labelsize=VISUAL["tick_label_size"]) 

    # (b) PDS sensitivity to transfer size
    avg_rmse = []
    for k in transfer_sizes:
        _, _, mr, _ = _compute_benchmarks(data, transfer_n=k)
        avg_rmse.append(float(np.mean(mr['PDS'])))
    ax_b.plot(transfer_sizes, avg_rmse, marker='o', label='PDS', markersize=VISUAL["line_marker_size"]) 
    ax_b.set_xlabel("Transfer set size", fontsize=VISUAL["axis_label_size"]) 
    ax_b.set_ylabel("Mean RMSEP (PDS)", fontsize=VISUAL["axis_label_size"]) 
    ax_b.set_title("5(b) Transfer size sensitivity", fontsize=VISUAL["title_size"]) 
    ax_b.grid(True, linestyle='--', alpha=0.4)
    ax_b.legend(fontsize=VISUAL["legend_fontsize"]) 

    # Bottom: 2x3 grid for parity plots (panel 5c)
    bottom = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=outer[1], hspace=0.4, wspace=0.35)
    axes_c = []
    for i in range(2):
        for j in range(3):
            axes_c.append(fig.add_subplot(bottom[i, j]))

    # Determine which sites to plot: if user provided rep_site and it matches, prioritize that one
    plot_sites = list(sites)
    # If more than 6 sites, only show first 6 and note in caption/meta
    max_plots = len(axes_c)
    if len(plot_sites) > max_plots:
        plot_sites = plot_sites[:max_plots]

    # Gather R^2 values for meta
    r2_map: dict[str, float] = {}
    for ax_idx, site_key in enumerate(plot_sites):
        ax = axes_c[ax_idx]
        y_te = np.asarray(per_site_preds[site_key]["y_te"], dtype=float)
        yhat_pds = np.asarray(per_site_preds[site_key]["yhat_pds"], dtype=float)
        ax.scatter(y_te, yhat_pds, alpha=0.6, s=VISUAL["parity_marker_size"]) 
        lims = [float(min(np.min(y_te), np.min(yhat_pds))), float(max(np.max(y_te), np.max(yhat_pds)))]
        ax.plot(lims, lims, 'r--', linewidth=1)
        # compute R^2 (safety: handle constant y_te)
        ss_res = float(np.sum((y_te - yhat_pds) ** 2))
        ss_tot = float(np.sum((y_te - float(np.mean(y_te))) ** 2))
        r2_val = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        r2_map[site_key] = float(r2_val)
        ax.set_xlabel("True", fontsize=VISUAL["axis_label_size"]) 
        ax.set_ylabel("Predicted", fontsize=VISUAL["axis_label_size"]) 
        ax.set_title(f"{site_key} — R\u00b2={r2_val:.3f}", fontsize=VISUAL["title_size"]) 
        ax.tick_params(axis='both', which='major', labelsize=VISUAL["tick_label_size"]) 

    # Hide any unused axes
    for k in range(len(plot_sites), len(axes_c)):
        axes_c[k].axis('off')

    fig_path = output_dir / "figure_5.png"
    # Use manual subplots_adjust (avoid tight_layout to keep GridSpec placement stable)
    fig.subplots_adjust(left=VISUAL["left"], right=VISUAL["right"], top=VISUAL["top"], bottom=VISUAL["bottom"],
                        wspace=VISUAL["wspace"], hspace=VISUAL["hspace"]) 
    plt.savefig(fig_path, dpi=300)
    plt.close()

    # Write caption and meta for traceability
    # Representative site(s) shown in panel 5c
    # Representative site(s) shown in panel 5c — choose for meta, not used as 'rep' variable
    # Keep the chosen list in `plot_sites` for reference.
    mean_rmse = {m: float(np.mean(method_rmse[m])) for m in methods}
    # reflect the resolved label from the registry/config/env
    model_label = _resolve_model_label()
    quick = _resolve_bool_from_env("FEDCHEM_QUICK", config.get('QUICK'), False)
    ds_name = (ds_meta or {}).get("dataset", "real")
    parts = [
        f"Figure 5. Benchmarks across Centralized ({model_label}), Site-specific ({model_label}), PDS transfer, and SBC recalibration on {ds_name} data. ",
        "(a) Mean RMSEP across sites; (b) PDS sensitivity to transfer set size; ",
        f"(c) Parity plots for selected sites: {', '.join(plot_sites)}. ",
        f"Mean RMSEP - Centralized: {mean_rmse['Centralized']:.3f}, ",
        f"Site-specific: {mean_rmse['Site-specific']:.3f}, PDS: {mean_rmse['PDS']:.3f}. ",
    ]
    if 'SBC' in mean_rmse:
        parts.append(f"SBC: {mean_rmse['SBC']:.3f}. ")
    parts.extend([
        f"Transfer sizes: {transfer_sizes}; Sites: {len(sites)}. ",
        "Units: RMSEP scale=raw. ",
        "PDS params: window=32, overlap=16, ridge=1e-1. ",
    ])
    if quick:
        parts.append("Quick mode enabled; baseline tuning limited.")
    caption = "".join(parts)
    (output_dir / "figure_5_caption.txt").write_text(caption, encoding="utf-8")
    import json
    meta = {
        "figure": "figure_5.png",
        "panels": {
            "5a": "Mean RMSEP per method",
            "5b": {"title": "Transfer size sensitivity", "transfer_sizes": transfer_sizes},
            "5c": {"title": "Parity plots (PDS)", "plotted_sites": plot_sites, "r2": r2_map},
        },
        "methods": methods,
        "dataset": ds_name,
        "mean_rmse": mean_rmse,
        "site_count": len(sites),
    }
    (output_dir / "figure_5_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

def generate_table_5(
    data=None,
    transfer_n: int = 40,
    precomputed: dict | None = None,
):
    """Table 5: Benchmark metrics per site (Centralized, Site-specific, PDS).

    Uses precomputed metrics when provided to keep table in sync with figure/manifest.
    """
    if precomputed is not None:
        methods = precomputed["methods"]
        sites = precomputed["sites"]
        method_rmse = precomputed["method_rmse"]
    else:
        if data is None:
            data, _ = _prepare_data(config_sites=config_sites)
        methods, sites, method_rmse, _ = _compute_benchmarks(data, transfer_n=transfer_n)
    rows = []
    used_model = _resolve_model_label()
    has_sbc = 'SBC' in methods and 'SBC' in method_rmse
    for i, s in enumerate(sites):
        row = {
            "Site": s,
            "Centralized": float(method_rmse['Centralized'][i]),
            "Site-specific": float(method_rmse['Site-specific'][i]),
            "PDS": float(method_rmse['PDS'][i]),
            "UsedModel": used_model,
        }
        if has_sbc:
            row["SBC"] = float(method_rmse['SBC'][i])
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "table_5.csv", index=False)

def _write_benchmark_summary(per_site_preds: dict[str, dict[str, np.ndarray]], dataset_label: str) -> None:
    if not per_site_preds:
        return
    site_names = sorted(per_site_preds.keys())
    y_true = np.hstack([per_site_preds[s]["y_te"] for s in site_names])
    summary_rows = []
    for label, key in [("Centralized", "yhat_central"), ("Site-specific", "yhat_site"), ("PDS", "yhat_pds")]:
        preds = np.hstack([per_site_preds[s][key] for s in site_names])
        summary_rows.append({
            "Method": label,
            "RMSEP": float(rmsep(y_true, preds)),
            "R2": float(r2(y_true, preds)),
            "MAE": float(mae(y_true, preds)),
        })
    df = pd.DataFrame(summary_rows)
    df["Dataset"] = dataset_label
    df.to_csv(output_dir / "table_5_summary.csv", index=False)

def _write_manifest(cfg: dict, summary: dict):
    """Write a lightweight manifest for Objective 5.

    Parameter renamed to avoid shadowing the module-level 'config'.
    """
    import json
    from fedchem.utils.manifest_utils import compute_combo_id
    manifest_cfg = cfg or {}
    # Include reproducibility and config blocks if present
    for sec in ('DIFFERENTIAL_PRIVACY', 'CONFORMAL', 'SECURE_AGGREGATION', 'DRIFT_AUGMENT', 'REPRODUCIBILITY'):
        if sec in globals().get('config', {}):
            manifest_cfg[sec] = globals()['config'].get(sec)
    # allow values of different types (e.g., strings) to be assigned later
    manifest_obj: dict[str, Any] = {"config": manifest_cfg, "summary": summary}
    # ensure compute_combo_id always receives a dict and store the resulting identifier
    manifest_obj["combo_id"] = compute_combo_id(manifest_obj.get("config", {}))
    (output_dir / "manifest_5.json").write_text(json.dumps(manifest_obj, indent=2))

def main(
    data: dict[str, dict[str, np.ndarray]] | None = None,
    ds_meta: dict[str, Any] | None = None,
):
    start_time = time.perf_counter()
    cfg = config if isinstance(config, dict) else {}
    n_sites_default = _coerce_int(cfg.get('NUM_SITES'), 5) or 5
    n_sites = _parse_optional_int("FEDCHEM_NUM_SITES", n_sites_default) or n_sites_default
    n_sites = max(1, n_sites)

    quick = _resolve_bool_from_env("FEDCHEM_QUICK", cfg.get('QUICK'), False)

    transfer_default = _coerce_int(cfg.get('PDS_TRANSFER_N'), 40) or 40
    transfer_n = _parse_optional_int("FEDCHEM_PDS_TRANSFER_N", transfer_default) or transfer_default

    # Reproducibility seed
    seed_default = _coerce_int(cfg.get('SEED'), 42) or 42
    seed = _parse_optional_int("FEDCHEM_SEED", seed_default) or seed_default
    import random as _random
    _random.seed(int(seed))
    np.random.seed(int(seed))

    force_tecator = _resolve_bool_from_env("FEDCHEM_USE_TECATOR", cfg.get('USE_TECATOR'), False)
    n_wavelengths_default = _coerce_int(cfg.get('DEFAULT_N_WAVELENGTHS'), DEFAULT_N_WAVELENGTHS) or DEFAULT_N_WAVELENGTHS
    n_wavelengths = _parse_optional_int("FEDCHEM_N_WAVELENGTHS", n_wavelengths_default)
    max_transfer_default = _coerce_int(cfg.get('MAX_TRANSFER_SAMPLES'))
    max_transfer_samples = _parse_optional_int("FEDCHEM_MAX_TRANSFER_SAMPLES", max_transfer_default)
    if data is None or ds_meta is None:
        data, ds_meta = _prepare_data(
            n_sites=n_sites,
            seed=seed,
            quick=quick,
            force_tecator=force_tecator,
            n_wavelengths=n_wavelengths,
            max_transfer_samples=max_transfer_samples,
            config_sites=config_sites,
        )
    ds_meta = ds_meta or {}
    ds_name = ds_meta.get("dataset", "real")
    # Compute once and reuse across figure, table, manifest for consistency
    methods, sites, method_rmse, per_site_preds = _compute_benchmarks(data, transfer_n=transfer_n)
    precomputed = {"methods": methods, "sites": sites, "method_rmse": method_rmse, "per_site_preds": per_site_preds}
    generate_figure_5(
        data,
        ds_meta=ds_meta,
        transfer_sizes=[10, 20, max(10, transfer_n)],
        precomputed=precomputed,
    )
    generate_table_5(data, transfer_n=transfer_n, precomputed=precomputed)
    _write_benchmark_summary(per_site_preds, ds_name)

    # Manifest from the same precomputed metrics
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    pipeline_cfg = config.get("PIPELINE") if isinstance(config.get("PIPELINE"), dict) else {}
    # Ensure resolved_pipeline is always a dict to satisfy type checkers
    if isinstance(pipeline_cfg, dict):
        resolved_pipeline = dict(pipeline_cfg)
    else:
        resolved_pipeline = {}
    resolved_pipeline["model"] = _resolve_model_label()
    fedpls_enabled = _resolve_fedpls_enabled()
    run_config = {
        "dataset": ds_name,
        "dataset_meta": ds_meta,
        "force_tecator": force_tecator,
        "n_sites": n_sites,
        "transfer_n": transfer_n,
        "n_wavelengths_requested": ds_meta.get("n_wavelengths_requested"),
        "n_wavelengths_actual": ds_meta.get("n_wavelengths_actual"),
        "transfer_samples_requested": ds_meta.get("transfer_samples_requested", max_transfer_samples),
        "transfer_samples_used": ds_meta.get("transfer_samples_used"),
        "max_transfer_samples": max_transfer_samples,
        "standard_design": True,
        "design_version": resolve_design_version(config),
        "pipeline": resolved_pipeline,
        "fedpls_enabled": fedpls_enabled,
        "fedpls_method": _resolve_fedpls_method() if fedpls_enabled else None,
        "seed": seed,
        "pds_params": {"window": 32, "overlap": 16, "ridge": 1e-1},
        "quick_mode": bool(quick),
        "secure_aggregation": False,
        "ct_federated_variant": _resolve_ct_federated_variant(),
        "runtime": {"wall_time_total_sec": float(time.perf_counter() - start_time)},
    }
    summary = {"mean_rmse": {m: float(np.mean(method_rmse[m])) for m in methods}}
    manifest = {
        "config": run_config,
        "summary": summary,
        "metrics": {"metric_scale": "raw"},
        "logs_by_algorithm": {},  # PDS Transfer doesn't use orchestrator, but keep for consistency
        "used_models": {m: resolved_pipeline.get("model") for m in methods},
        "versions": versions
    }
    # For consistency with other objective manifests, expose runtime at top-level as well
    # (some tools expect `manifest["runtime"]` to exist).
    manifest["runtime"] = run_config.get("runtime")
    import json
    # Add pipeline hash for traceability
    try:
        import hashlib
        from pathlib import Path
        h = hashlib.sha1()
        base = Path(__file__).parent
        for i in range(1, 8):
            p = base / f"generate_objective_{i}.py"
            if p.exists():
                h.update(p.read_bytes())
        manifest["pipeline_hash"] = h.hexdigest()[:12]
    except Exception:
        manifest["pipeline_hash"] = None
    (output_dir / "manifest_5.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Objective 5 completed.")

if __name__ == "__main__":
    main()
