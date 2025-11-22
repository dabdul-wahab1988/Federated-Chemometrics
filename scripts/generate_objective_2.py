"""
Script for Objective 2: Integrate conformal prediction on the real 5-site dataset.

Generates Figure 2 and Table 2 showing coverage and interval widths across
multiple sites, including a held-out site. Real data is the default; set
FEDCHEM_USE_TECATOR=1 only to force the Tecator fallback.

Env toggles:
- FEDCHEM_USE_TECATOR=1 to force Tecator (requires FEDCHEM_ALLOW_DOWNLOAD=1 on first run)
- FEDCHEM_HELD_OUT_SITE: name like "site_0" (default), excluded from pooled training.
- FEDCHEM_REP_SITE: site used for panel 2(a); defaults to the first non-held-out.
- FEDCHEM_ALPHAS: comma-separated e.g. "0.1,0.05"; first drives Figure 2; all go to Table 2.
- FEDCHEM_CQR_USE_PLS_FEATS=1: project X to PLS latent scores (CV-selected components) before CQR/Mondrian.
"""

from typing import Any
from pathlib import Path
from fedchem.utils.config import load_and_seed_config, load_config, get_experimental_sites, get_data_config

# Seed environment variables from `config.yaml` for generators run directly
cfg = load_and_seed_config()
# Load configuration via centralized helper
# `cfg` already contains config; keep a `config` alias for consistency
config = cfg
config_sites = get_experimental_sites(cfg)
data_config = get_data_config(cfg)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fedchem.conformal.cqr import CQRConformal, MondrianCQR
from fedchem.conformal.cross_cqr import CrossCQRConformal
from fedchem.models.linear import LinearModel
from fedchem.models.pls import PLSModel
from fedchem.metrics.metrics import coverage, rmsep, r2, mae
import time, platform, os
from fedchem.federated.orchestrator import FederatedOrchestrator
from fedchem.utils.real_data import load_real_site_dict
from fedchem.utils.logging_utils import extract_logs_for_manifest, create_log_summary
from fedchem.utils.model_registry import instantiate_model
from fedchem.ct.pds_transfer import PDSTransfer

plt.rcParams.update({'font.size': 14})
sns.set_style("whitegrid")

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


def _parse_optional_int(name: str, default: int | None = None) -> int | None:
    return _coerce_int(os.environ.get(name), default)


def _parse_optional_float(name: str, default: float | None = None) -> float | None:
    return _coerce_float(os.environ.get(name), default)


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

def _train_cqr_global(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    *,
    use_cross: bool = False,
    k_folds: int = 5,
    tau_lo: float = 0.1,
    tau_hi: float = 0.9,
    calib_fraction: float = 0.2,
    seed: int | None = 42,
) -> CQRConformal:
    if use_cross:
        return CrossCQRConformal(k_folds=int(k_folds), tau_lo=tau_lo, tau_hi=tau_hi, alpha_default=0.1, seed=seed).fit(Xtr, ytr)  # type: ignore[return-value]
    c = CQRConformal(tau_lo=tau_lo, tau_hi=tau_hi, calib_fraction=calib_fraction, alpha_default=0.1)
    return c.fit(Xtr, ytr)

def _train_cqr_mondrian(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> MondrianCQR:
    return MondrianCQR().fit(X, y, groups=groups)

def _split_site(X: np.ndarray, y: np.ndarray, train_frac=0.6, calib_frac=0.2):
    n = X.shape[0]
    n_tr = max(1, int(train_frac * n))
    n_cal = max(1, int(calib_frac * n))
    n_te = n - n_tr - n_cal
    if n_te <= 0:
        n_te = max(1, n - n_tr - n_cal)
    X_tr, y_tr = X[:n_tr], y[:n_tr]
    X_cal, y_cal = X[n_tr:n_tr+n_cal], y[n_tr:n_tr+n_cal]
    X_te, y_te = X[n_tr+n_cal:], y[n_tr+n_cal:]
    return (X_tr, y_tr), (X_cal, y_cal), (X_te, y_te)

def _prepare_pooled_train(data: dict[str, dict[str, np.ndarray]], held_out: str | None = None):
    """Create pooled train data across all sites except held_out.

    Returns pooled X,y and per-site test splits.
    """
    Xtr_list, ytr_list = [], []
    per_site_splits: dict[str, dict[str, np.ndarray]] = {}
    group_tr: list[int] = []
    group_id_map = {name: i for i, name in enumerate(sorted(data.keys()))}
    for name, site in data.items():
        X, y = site["X"], site["y"]
        (X_tr, y_tr), (X_cal, y_cal), (X_te, y_te) = _split_site(X, y)
        # For training classifiers, we feed combined (train+calib) to the learner
        Xtr_site = np.vstack([X_tr, X_cal])
        ytr_site = np.hstack([y_tr, y_cal])
        if name != held_out:
            Xtr_list.append(Xtr_site)
            ytr_list.append(ytr_site)
            group_tr.extend([group_id_map[name]] * Xtr_site.shape[0])
        per_site_splits[name] = {
            "X_train": X_tr,
            "y_train": y_tr,
            "X_cal": X_cal,
            "y_cal": y_cal,
            "X_test": X_te,
            "y_test": y_te,
            "group_id": np.full(X_te.shape[0], group_id_map[name], dtype=int),
        }
    Xtr_all = np.vstack(Xtr_list) if Xtr_list else np.zeros((0, data[next(iter(data))]["X"].shape[1]))
    ytr_all = np.hstack(ytr_list) if ytr_list else np.zeros((0,))
    groups_all = np.array(group_tr, dtype=int) if group_tr else np.zeros((0,), dtype=int)
    return Xtr_all, ytr_all, groups_all, per_site_splits

def _build_pls_feature_map(Xtr: np.ndarray, ytr: np.ndarray):
    """Fit a PLSModel with CV and return a callable to map X -> PLS scores."""
    pls = PLSModel(n_components=None, max_components=20, cv=5, random_state=0)
    pls.fit(Xtr, ytr)
    if pls.pipeline is None:
        return (lambda X: X)
    scaler = pls.pipeline.named_steps.get("scaler")
    pls_step = pls.pipeline.named_steps.get("pls")
    def transform(X: np.ndarray) -> np.ndarray:
        Xs = scaler.transform(X) if scaler is not None else X
        if pls_step is not None and hasattr(pls_step, "transform"):
            return pls_step.transform(Xs)
        return Xs
    return transform

def generate_figure_2(
    data: dict[str, dict[str, np.ndarray]],
    held_out_site: str | None = None,
    rep_site: str | None = None,
    alpha: float = 0.1,
    out_name: str = "figure_2.png",
    use_fed_baseline: bool = False,
    use_pls_baseline: bool = False,
    use_pls_feats: bool = False,
    *,
    use_cross_cqr: bool = False,
    cross_k_folds: int = 5,
    tau_lo: float = 0.1,
    tau_hi: float = 0.9,
    calib_fraction: float = 0.2,
    seed: int = 42,
):
    """Figure 2: Conformal Prediction across sites with held-out evaluation."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Prepare pooled training (excluding held-out) and per-site tests
    Xtr, ytr, groups_tr, per_site = _prepare_pooled_train(data, held_out=held_out_site)

    # Optional PLS feature projection for CQR/Mondrian
    if use_pls_feats:
        feat_map = _build_pls_feature_map(Xtr, ytr)
        Xtr_feat = feat_map(Xtr)
    else:
        feat_map = lambda X: X
        Xtr_feat = Xtr
    # Train global and Mondrian CQR on features
    cqr = _train_cqr_global(
        Xtr_feat,
        ytr,
        use_cross=use_cross_cqr,
        k_folds=cross_k_folds,
        tau_lo=tau_lo,
        tau_hi=tau_hi,
        calib_fraction=calib_fraction,
    )
    mond = _train_cqr_mondrian(Xtr_feat, ytr, groups_tr)
    # Resolve calibration mode from env/config in table generation too
    env_calib = os.environ.get("FEDCHEM_CONFORMAL_CALIBRATION")
    config_calib = (config.get('CONFORMAL') or {}).get('per_site_or_pooled') if isinstance(config.get('CONFORMAL'), dict) else None
    if env_calib:
        calib_token = str(env_calib).strip().lower()
    elif config_calib:
        calib_token = str(config_calib).strip().lower()
    else:
        calib_token = "per_site"
    use_mondrian = calib_token in {"per_site", "per-site", "mondrian", "per_site"}
    # Resolve calibration mode from env/config: 'per_site' (Mondrian) or 'pooled' (global)
    env_calib = os.environ.get("FEDCHEM_CONFORMAL_CALIBRATION")
    config_calib = (config.get('CONFORMAL') or {}).get('per_site_or_pooled') if isinstance(config.get('CONFORMAL'), dict) else None
    if env_calib:
        calib_token = str(env_calib).strip().lower()
    elif config_calib:
        calib_token = str(config_calib).strip().lower()
    else:
        calib_token = "per_site"
    use_mondrian = calib_token in {"per_site", "per-site", "mondrian", "per_site"}

    # Resolve calibration mode from env/config: 'per_site' (Mondrian) or 'pooled' (global)
    env_calib = os.environ.get("FEDCHEM_CONFORMAL_CALIBRATION")
    config_calib = (config.get('CONFORMAL') or {}).get('per_site_or_pooled') if isinstance(config.get('CONFORMAL'), dict) else None
    if env_calib:
        calib_token = str(env_calib).strip().lower()
    elif config_calib:
        calib_token = str(config_calib).strip().lower()
    else:
        calib_token = "per_site"
    use_mondrian = calib_token in {"per_site", "per-site", "mondrian", "per_site"}

    # Choose a representative site for panel (a)
    rep_name = rep_site if (rep_site in data) else next((n for n in sorted(data.keys()) if n != held_out_site), next(iter(data.keys())))
    rep = per_site[rep_name]
    X_rep, y_rep, g_rep = rep["X_test"], rep["y_test"], rep["group_id"]

    # Build baseline predictor for plotting predicted vs true centers
    if use_fed_baseline:
        # Quick federated global model using non-held-out clients
        clients = [
            {"X": site["X"], "y": site["y"]}
            for name, site in data.items() if name != held_out_site
        ]
        # Best-effort: estimate per-client PDS communication bytes and attach to client dicts
        try:
            # Use pooled training Xtr as reference for PDSTransfer
            if Xtr is not None and Xtr.shape[0] > 0:
                d = Xtr.shape[1]
                window = min(32, max(1, d))
                overlap = min(16, max(0, window // 2))
                for idx, c in enumerate(clients):
                    try:
                        Xc = c.get("X")
                        if Xc is None or Xc.size == 0:
                            c["communication_bytes"] = 0
                            continue
                        p = PDSTransfer(window=window, overlap=overlap, ridge=1e-1)
                        # fit on a limited number of rows to keep it fast
                        k = min(50, Xtr.shape[0], Xc.shape[0])
                        p.fit(Xtr[:k], Xc[:k])
                        c["communication_bytes"] = int(p.estimated_bytes())
                    except Exception:
                        c["communication_bytes"] = 0
        except Exception:
            # don't fail the entire figure generation on estimation errors
            pass
        orch = FederatedOrchestrator()
        res = orch.run_rounds(
            clients=clients,
            model=instantiate_model(config.get("PIPELINE", {}).get("model", "LinearModel")),
            rounds=2,
            algo="fedavg",
            dp_config=None,
            clip_norm=None,
            seed=seed,
        )
        base_model = res.get("model", instantiate_model(config.get("PIPELINE", {}).get("model", "LinearModel")).fit(Xtr, ytr))
        y_pred = base_model.predict(X_rep)
    elif use_pls_baseline:
        pls = PLSModel(n_components=None, max_components=20, cv=5, random_state=0)
        pls.fit(Xtr, ytr)
        y_pred = pls.predict(X_rep)
    else:
        base = instantiate_model(config.get("PIPELINE", {}).get("model", "LinearModel")).fit(Xtr, ytr)
        y_pred = base.predict(X_rep)
    X_rep_feat = feat_map(X_rep)
    lo, hi = cqr.predict_interval(X_rep_feat, alpha=alpha)

    # (a) Prediction intervals on representative site
    err = np.maximum((hi - lo) / 2, 0)
    label_global = f"Global {'Cross-CQR' if use_cross_cqr else 'CQR'} α={alpha}"
    label_mond = "Mondrian CQR" if use_mondrian else "Pooled CQR"
    axes[0,0].errorbar(y_rep, y_pred, yerr=err, fmt='o', alpha=0.6, label=label_global)
    if use_mondrian:
        lo_m, hi_m = mond.predict_interval(X_rep_feat, groups=g_rep, alpha=alpha)
    else:
        lo_m, hi_m = cqr.predict_interval(X_rep_feat, alpha=alpha)
    err_m = np.maximum((hi_m - lo_m) / 2, 0)
    axes[0,0].errorbar(y_rep, y_pred, yerr=err_m, fmt='x', alpha=0.4, label=label_mond)
    lims = [float(np.min(y_rep)), float(np.max(y_rep))]
    axes[0,0].plot(lims, lims, 'r--', linewidth=1)
    axes[0,0].set_xlabel("True")
    axes[0,0].set_ylabel("Predicted (baseline)")
    axes[0,0].set_title("2(a) Intervals (representative site)")
    # Place legend below subplot (centered)
    axes[0,0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
    axes[0,0].tick_params(axis='both', which='major', labelsize=14)

    # (b) Coverage per site (including held-out)
    names, cov_g_list, cov_m_list = [], [], []
    for name in sorted(per_site.keys()):
        X_te, y_te, g_te = per_site[name]["X_test"], per_site[name]["y_test"], per_site[name]["group_id"]
        X_te_feat = feat_map(X_te)
        lo_g, hi_g = cqr.predict_interval(X_te_feat, alpha=alpha)
        if held_out_site is None or name != held_out_site:
            if use_mondrian:
                lo_mm, hi_mm = mond.predict_interval(X_te_feat, groups=g_te, alpha=alpha)
            else:
                lo_mm, hi_mm = cqr.predict_interval(X_te_feat, alpha=alpha)
        else:
            lo_mm, hi_mm = cqr.predict_interval(X_te, alpha=alpha)
        names.append(name if name != held_out_site else f"{name} (held-out)")
        cov_g_list.append(coverage(y_te, lo_g, hi_g))
        cov_m_list.append(coverage(y_te, lo_mm, hi_mm))
    x = np.arange(len(names))
    axes[0,1].bar(x - 0.2, cov_g_list, width=0.4, label=("Global Cross-CQR" if use_cross_cqr else "Global CQR"))
    axes[0,1].bar(x + 0.2, cov_m_list, width=0.4, label=label_mond)
    axes[0,1].axhline(1 - alpha, color='r', linestyle='--', label=f"Nominal {1-alpha:.0%}")
    # Pretty-print site labels: replace underscores, capitalize, and put held-out on a second line
    pretty_names = []
    for name in names:
        if name.endswith(" (held-out)"):
            base = name.replace(" (held-out)", "")
            pretty_names.append(base.replace("site_", "Site ").replace("_", " ") + "\n(held-out)")
        else:
            pretty_names.append(name.replace("site_", "Site ").replace("_", " "))
    axes[0,1].set_xticks(x)
    axes[0,1].set_xticklabels(pretty_names, rotation=30, ha='right')
    axes[0,1].set_ylim(0.0, 1.05)
    axes[0,1].set_ylabel("Coverage")
    axes[0,1].set_title("2(b) Coverage by site")
    # Place legend below subplot (centered)
    axes[0,1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False)
    axes[0,1].tick_params(axis='both', which='major', labelsize=14)

    # (c) Mean interval width per site
    width_g, width_m = [], []
    for name in sorted(per_site.keys()):
        X_te, y_te, g_te = per_site[name]["X_test"], per_site[name]["y_test"], per_site[name]["group_id"]
        X_te_feat = feat_map(X_te)
        lo_g, hi_g = cqr.predict_interval(X_te_feat, alpha=alpha)
        if held_out_site is None or name != held_out_site:
            if use_mondrian:
                lo_mm, hi_mm = mond.predict_interval(X_te_feat, groups=g_te, alpha=alpha)
            else:
                lo_mm, hi_mm = cqr.predict_interval(X_te_feat, alpha=alpha)
        else:
            lo_mm, hi_mm = cqr.predict_interval(X_te, alpha=alpha)
        width_g.append(float(np.mean(hi_g - lo_g)))
        width_m.append(float(np.mean(hi_mm - lo_mm)))
    axes[1,0].bar(x - 0.2, width_g, width=0.4, label="Global CQR")
    axes[1,0].bar(x + 0.2, width_m, width=0.4, label=label_mond)
    # Reuse pretty names for interval width subplot for consistent presentation
    axes[1,0].set_xticks(x)
    axes[1,0].set_xticklabels(pretty_names, rotation=30, ha='right')
    axes[1,0].set_ylabel("Mean width")
    axes[1,0].set_title("2(c) Interval width by site")
    # Place legend below subplot (centered)
    axes[1,0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False)
    axes[1,0].tick_params(axis='both', which='major', labelsize=14)

    # (d) Calibration curve (expected vs observed) on pooled test
    # Build pooled test across all sites (held-out included)
    Xpool = np.vstack([per_site[n]["X_test"] for n in per_site])
    ypool = np.hstack([per_site[n]["y_test"] for n in per_site])
    Xpool_feat = feat_map(Xpool)
    alphas = np.linspace(0.01, 0.3, 15)
    obs = []
    for a in alphas:
        lo_g, hi_g = cqr.predict_interval(Xpool_feat, alpha=float(a))
        obs.append(coverage(ypool, lo_g, hi_g))
    axes[1,1].plot(1 - alphas, obs, marker='o', label=('Global Cross-CQR' if use_cross_cqr else 'Global CQR'))
    axes[1,1].plot([0,1], [0,1], 'r--', label='Ideal')
    axes[1,1].set_xlabel("Nominal coverage")
    axes[1,1].set_ylabel("Observed coverage (pooled test)")
    axes[1,1].set_title("2(d) Calibration curve")
    # Place legend below subplot (centered)
    axes[1,1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
    axes[1,1].tick_params(axis='both', which='major', labelsize=14)

    # Increase bottom margin so legends below subplots are visible
    plt.subplots_adjust(bottom=0.12, hspace=0.35, wspace=0.25)
    plt.tight_layout(rect=(0, 0.04, 1, 1))
    plt.savefig(output_dir / out_name, dpi=300, bbox_inches='tight')
    plt.close()

def generate_table_2(
    data: dict[str, dict[str, np.ndarray]],
    held_out_site: str | None = None,
    alphas: list[float] | None = None,
    use_pls_feats: bool = False,
    *,
    use_cross_cqr: bool = False,
    cross_k_folds: int = 5,
    tau_lo: float = 0.1,
    tau_hi: float = 0.9,
    calib_fraction: float = 0.2,
):
    """Table 2: Coverage and width per site for Global and Mondrian CQR.

    Trains on pooled train (excluding held-out), then evaluates per-site tests.
    """
    # Read conformal calibration mode from environment variable
    conformal_calibration_env = os.getenv("FEDCHEM_CONFORMAL_CALIBRATION", "per_site").lower()
    use_mondrian = conformal_calibration_env == "per_site"
    Xtr, ytr, groups_tr, per_site = _prepare_pooled_train(data, held_out=held_out_site)
    if use_pls_feats:
        feat_map = _build_pls_feature_map(Xtr, ytr)
        Xtr_feat = feat_map(Xtr)
    else:
        feat_map = lambda X: X
        Xtr_feat = Xtr
    cqr = _train_cqr_global(
        Xtr_feat,
        ytr,
        use_cross=use_cross_cqr,
        k_folds=cross_k_folds,
        tau_lo=tau_lo,
        tau_hi=tau_hi,
        calib_fraction=calib_fraction,
    )
    mond = _train_cqr_mondrian(Xtr_feat, ytr, groups_tr)

    rows = []
    used_model = _resolve_model_label()
    if not alphas:
        alphas = [0.1]
    for a in alphas:
        for name in sorted(per_site.keys()):
            X_te, y_te, g_te = per_site[name]["X_test"], per_site[name]["y_test"], per_site[name]["group_id"]
            X_te_feat = feat_map(X_te)
            lo_g, hi_g = cqr.predict_interval(X_te_feat, alpha=a)
            if held_out_site is None or name != held_out_site:
                if use_mondrian:
                    lo_m, hi_m = mond.predict_interval(X_te_feat, groups=g_te, alpha=a)
                else:
                    lo_m, hi_m = cqr.predict_interval(X_te_feat, alpha=a)
            else:
                lo_m, hi_m = cqr.predict_interval(X_te, alpha=a)
            rows.append({
                "Site": name if name != held_out_site else f"{name} (held-out)",
                "Alpha": float(a),
                "Nominal": 1 - float(a),
                "Global_Coverage": float(coverage(y_te, lo_g, hi_g)),
                "Global_MeanWidth": float(np.mean(hi_g - lo_g)),
                "Mondrian_Coverage": float(coverage(y_te, lo_m, hi_m)),
                "Mondrian_MeanWidth": float(np.mean(hi_m - lo_m)),
                "UsedModel": used_model,
            })
    pd.DataFrame(rows).to_csv(output_dir / "table_2.csv", index=False)

def _write_global_summary_metrics(
    data: dict[str, dict[str, np.ndarray]],
    *,
    held_out_site: str | None,
    alpha: float,
    tau_lo: float,
    tau_hi: float,
    calib_fraction: float,
    use_pls_feats: bool,
    use_cross_cqr: bool,
    cross_k_folds: int,
    dataset_label: str,
    seed: int,
) -> None:
    Xtr, ytr, groups_tr, per_site = _prepare_pooled_train(data, held_out=held_out_site)
    if Xtr.size == 0 or ytr.size == 0:
        return
    if use_pls_feats:
        feat_map = _build_pls_feature_map(Xtr, ytr)
        Xtr_feat = feat_map(Xtr)
    else:
        feat_map = lambda X: X
        Xtr_feat = Xtr
    cqr = _train_cqr_global(
        Xtr_feat,
        ytr,
        use_cross=use_cross_cqr,
        k_folds=cross_k_folds,
        tau_lo=tau_lo,
        tau_hi=tau_hi,
        calib_fraction=calib_fraction,
        seed=seed,
    )
    total_cal = sum(per_site[name]["X_cal"].shape[0] for name in per_site)
    alpha_adj = max(1e-6, float(alpha) * max(1.0, float(total_cal)) / (float(total_cal) + 1.0))
    site_names = sorted(per_site.keys())
    Xpool = np.vstack([per_site[name]["X_test"] for name in site_names])
    ypool = np.hstack([per_site[name]["y_test"] for name in site_names])
    pooled_feat = feat_map(Xpool)
    lo, hi = cqr.predict_interval(pooled_feat, alpha=alpha_adj)
    coverage_val = coverage(ypool, lo, hi)
    width_val = float(np.mean(hi - lo))
    global_model = PLSModel(n_components=None, max_components=20, cv=5, random_state=0)
    global_model.fit(Xtr, ytr)
    preds = global_model.predict(Xpool)
    rows = [
        {"Metric": "RMSEP", "Value": float(rmsep(ypool, preds))},
        {"Metric": "R2", "Value": float(r2(ypool, preds))},
        {"Metric": "MAE", "Value": float(mae(ypool, preds))},
        {"Metric": "Coverage", "Value": float(coverage_val)},
        {"Metric": "MeanWidth", "Value": float(width_val)},
    ]
    df = pd.DataFrame(rows)
    df["Alpha"] = float(alpha)
    df["AlphaAdjusted"] = float(alpha_adj)
    df["Dataset"] = dataset_label
    df["HeldOutSite"] = held_out_site or "None"
    df.to_csv(output_dir / "table_2_summary.csv", index=False)

def _parse_alphas(s: str | None) -> list[float]:
    if not s:
        return [0.1]
    try:
        vals = [float(x.strip()) for x in s.split(',') if x.strip()]
        return [v for v in vals if 0 < v < 1] or [0.1]
    except Exception:
        return [0.1]


def _resolve_model_label() -> str:
    env_label = os.environ.get("FEDCHEM_PIPELINE_MODEL")
    if env_label:
        return env_label
    pipeline_cfg = config.get("PIPELINE") if isinstance(config.get("PIPELINE"), dict) else {}
    cfg_label = pipeline_cfg.get('model') if isinstance(pipeline_cfg.get('model'), str) else None
    if cfg_label:
        return cfg_label
    # Default model for this set of experiments
    return "PLSModel"

def main(
    data: dict[str, dict[str, np.ndarray]] | None = None,
    ds_meta: dict[str, Any] | None = None,
):
    import os
    # Reproducibility: read seed from env
    seed_env = os.environ.get("FEDCHEM_SEED")
    try:
        seed = int(seed_env) if seed_env is not None else 42
    except Exception:
        seed = 42
    import random as _random
    _random.seed(int(seed))
    np.random.seed(int(seed))
    start_time = time.perf_counter()
    # Held-out site/env toggles: environment overrides config
    held_out = os.environ.get("FEDCHEM_HELD_OUT_SITE") or config.get('HELD_OUT_SITE')
    rep_site = os.environ.get("FEDCHEM_REP_SITE")
    alphas = _parse_alphas(os.environ.get("FEDCHEM_ALPHAS"))
    # If not provided via env, check CONFORMAL config block for target_coverages
    if (not alphas or len(alphas) == 0) and isinstance(config.get('CONFORMAL'), dict):
        cv = config.get('CONFORMAL', {}).get('target_coverages')
        if isinstance(cv, (list, tuple)) and cv:
            try:
                # convert coverage to alpha values
                alphas = [1.0 - float(v) for v in cv if v is not None]
            except Exception:
                alphas = [0.1]
    use_fed_baseline = os.environ.get("FEDCHEM_USE_FED_BASELINE", "0") == "1"
    use_pls_baseline = os.environ.get("FEDCHEM_USE_PLS_BASELINE", "0") == "1"
    second_fig_alpha_override = os.environ.get("FEDCHEM_SECOND_FIG_ALPHA")
    use_pls_feats = os.environ.get("FEDCHEM_CQR_USE_PLS_FEATS", "0") == "1"
    use_cross_cqr = os.environ.get("FEDCHEM_USE_CROSS_CQR", "0") == "1"
    if not use_cross_cqr and isinstance(config.get('CONFORMAL'), dict):
        method = config.get('CONFORMAL', {}).get('method')
        if isinstance(method, str) and method.strip().lower() in {"cross", "cross_cqr", "cross_conformal", "cross-cqr"}:
            use_cross_cqr = True
    cross_k = int(os.environ.get("FEDCHEM_CROSS_CQR_FOLDS", "5"))
    tau_lo = float(os.environ.get("FEDCHEM_CQR_TAU_LO", "0.1"))
    tau_hi = float(os.environ.get("FEDCHEM_CQR_TAU_HI", "0.9"))
    calib_fraction = float(os.environ.get("FEDCHEM_CQR_CALIB_FRAC", "0.2"))
    # Respect global site count override if provided
    num_sites_env = os.environ.get("FEDCHEM_NUM_SITES")
    try:
        n_sites = max(1, int(num_sites_env)) if num_sites_env is not None else int(config.get('NUM_SITES', 5))
    except Exception:
        n_sites = int(config.get('NUM_SITES', 5))
    quick = _resolve_bool_from_env("FEDCHEM_QUICK", config.get('QUICK'), False)
    force_tecator = _resolve_bool_from_env("FEDCHEM_USE_TECATOR", config.get('USE_TECATOR'), False)
    n_wavelengths_default = _coerce_int(config.get('DEFAULT_N_WAVELENGTHS'), DEFAULT_N_WAVELENGTHS) or DEFAULT_N_WAVELENGTHS
    n_wavelengths = _parse_optional_int("FEDCHEM_N_WAVELENGTHS", n_wavelengths_default)
    max_transfer_default = _coerce_int(config.get('MAX_TRANSFER_SAMPLES'))
    max_transfer_samples = _parse_optional_int("FEDCHEM_MAX_TRANSFER_SAMPLES", max_transfer_default)
    if data is None or ds_meta is None:
        # Read Site codes and per-manufacturer data limits from config.yaml so non-coders
        # can control which site codes are used.
        exp_design = config.get("EXPERIMENTAL_DESIGN", {}) or {}
        factors = exp_design.get("FACTORS", {}) if isinstance(exp_design.get("FACTORS"), dict) else {}
        config_sites = factors.get("Site")
        if isinstance(config_sites, str):
            config_sites = [s.strip() for s in config_sites.split(",") if s.strip()]
        data_config = config.get("DATA_CONFIG") or {}
        data, ds_meta = load_real_site_dict(
            n_sites=n_sites,
            seed=seed,
            quick=quick,
            force_tecator=force_tecator,
            n_wavelengths=n_wavelengths,
            max_transfer_samples=max_transfer_samples,
            config_sites=config_sites,
            data_config=data_config,
        )
    ds_meta = ds_meta or {}
    ds_name = ds_meta.get("dataset", "real")
    # Figure uses the first alpha; table evaluates all
    generate_figure_2(
        data,
        held_out_site=held_out,
        rep_site=rep_site,
        alpha=alphas[0],
        out_name="figure_2.png",
        use_fed_baseline=use_fed_baseline,
        use_pls_baseline=False if use_fed_baseline else use_pls_baseline,
        use_pls_feats=use_pls_feats,
        use_cross_cqr=use_cross_cqr,
        cross_k_folds=cross_k,
        tau_lo=tau_lo,
        tau_hi=tau_hi,
        calib_fraction=calib_fraction,
        seed=seed,
    )
    # Optional second figure for another alpha
    second_alpha: float | None = None
    if second_fig_alpha_override:
        try:
            v = float(second_fig_alpha_override)
            if 0 < v < 1:
                second_alpha = v
        except Exception:
            second_alpha = None
    elif len(alphas) > 1:
        second_alpha = alphas[1]
    if second_alpha is not None:
        out_nm = f"figure_2_alpha{second_alpha:.2f}.png"
        generate_figure_2(
            data,
            held_out_site=held_out,
            rep_site=rep_site,
            alpha=second_alpha,
            out_name=out_nm,
            use_fed_baseline=use_fed_baseline,
            use_pls_baseline=False if use_fed_baseline else use_pls_baseline,
            use_pls_feats=use_pls_feats,
            use_cross_cqr=use_cross_cqr,
            cross_k_folds=cross_k,
            tau_lo=tau_lo,
            tau_hi=tau_hi,
            calib_fraction=calib_fraction,
            seed=seed,
        )
    generate_table_2(
        data,
        held_out_site=held_out,
        alphas=alphas,
        use_pls_feats=use_pls_feats,
        use_cross_cqr=use_cross_cqr,
        cross_k_folds=cross_k,
        tau_lo=tau_lo,
        tau_hi=tau_hi,
        calib_fraction=calib_fraction,
    )
    _write_global_summary_metrics(
        data,
        held_out_site=held_out,
        alpha=alphas[0],
        tau_lo=tau_lo,
        tau_hi=tau_hi,
        calib_fraction=calib_fraction,
        use_pls_feats=use_pls_feats,
        use_cross_cqr=use_cross_cqr,
        cross_k_folds=cross_k,
        dataset_label=ds_name,
        seed=seed,
    )
    # Write a caption + manifest for traceability (ASCII only)
    import json
    from fedchem.utils.manifest_utils import resolve_design_version
    caption = (
        f"Figure 2. Conformal prediction across multi-site {ds_name} data. "
        f"Panels: (a) intervals on representative site (alpha={alphas[0]:.2f}); (b) coverage by site; (c) mean interval width by site; (d) calibration curve on pooled test. "
        f"Held-out site: {held_out}. Sites: {n_sites}. Alphas evaluated: {alphas}. "
        + ("Quick mode enabled; coverage may exceed ±0.02 tolerance." if quick else "")
        + (" Using cross-conformal (k-fold) calibration." if use_cross_cqr else "")
    )
    (output_dir / "figure_2_caption.txt").write_text(caption, encoding="utf-8")
    # Coverage diagnostics (primary alpha)
    try:
        tbl = pd.read_csv(output_dir / "table_2.csv")
        primary_rows = tbl[tbl["Alpha"] == alphas[0]]
        cov_global_mean = float(primary_rows["Global_Coverage"].mean()) if not primary_rows.empty else None
    except Exception:
        cov_global_mean = None
    # Extract federated logs if available
    fed_logs = {}
    res = None  # Initialize res to avoid undefined errors
    if 'res' in locals() and res:
        fed_logs = extract_logs_for_manifest({"FedAvg_baseline": res})
    
    # Resolve pipeline configuration from config.yaml (fallbacks provided)
    cfg_pipeline = config.get("PIPELINE") if isinstance(config.get("PIPELINE"), dict) else {}
    pipeline = {
        "model": cfg_pipeline.get("model", "LinearModel"),
        "x_scaler": cfg_pipeline.get("x_scaler", None),
        "y_scaler": cfg_pipeline.get("y_scaler", None),
    }
    # Determine whether FedPLS is enabled via config or env; prefer explicit config
    use_fedpls = _resolve_fedpls_enabled()
    fedpls_method = _resolve_fedpls_method() if use_fedpls else None

    manifest = {
        "config": {
            "dataset": ds_name,
            "dataset_meta": ds_meta,
            "force_tecator": force_tecator,
            "n_sites": n_sites,
            "held_out": held_out,
            "alphas": alphas,
            "pls_features": use_pls_feats,
            "quick_mode": bool(quick),
            "standard_design": True,
            "design_version": resolve_design_version(config),
            "pipeline": pipeline,
            "fedpls_method": fedpls_method,
            "seed": seed,
            "cross_cqr": use_cross_cqr,
            "cross_cqr_folds": cross_k,
        "tau_lo": tau_lo,
        "tau_hi": tau_hi,
        "calib_fraction": calib_fraction,
        "n_wavelengths_requested": ds_meta.get("n_wavelengths_requested"),
        "n_wavelengths_actual": ds_meta.get("n_wavelengths_actual"),
        "transfer_samples_requested": ds_meta.get("transfer_samples_requested"),
        "transfer_samples_used": ds_meta.get("transfer_samples_used"),
    },
        "diagnostics": {
            "global_mean_coverage_alpha_primary": cov_global_mean,
            "nominal_primary": (1 - alphas[0]) if alphas else None,
            "coverage_deviation": (cov_global_mean - (1 - alphas[0])) if cov_global_mean is not None and alphas else None,
            "tolerance_pass": (abs(cov_global_mean - (1 - alphas[0])) <= 0.02) if cov_global_mean is not None and alphas else None,
        },
        "logs_by_algorithm": fed_logs,
        "used_models": {"Global": pipeline.get("model"), "FedAvg_baseline": (res.get("used_model") if isinstance(res, dict) else None)},
    }
    # Add reproducibility blocks and DP/CONFORMAL keys in config for traceability
    manifest_cfg = manifest.get('config', {}) or {}
    for sec in ('DIFFERENTIAL_PRIVACY', 'CONFORMAL', 'SECURE_AGGREGATION', 'DRIFT_AUGMENT', 'REPRODUCIBILITY'):
        if sec in config:
            manifest_cfg[sec] = config.get(sec)
    manifest['config'] = manifest_cfg
    # Enrich manifest with runtime and versions for reproducibility
    manifest["runtime"] = {"wall_time_total_sec": float(time.perf_counter() - start_time)}
    manifest["versions"] = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__}
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
    from fedchem.utils.manifest_utils import compute_combo_id
    manifest["combo_id"] = compute_combo_id(manifest.get("config"))
    (output_dir / "manifest_2.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Objective 2 completed.")

if __name__ == "__main__":
    main()
