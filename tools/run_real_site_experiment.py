"""Orchestrator for the site real data federated experiment.

This script uses the local `fedchem` modules:
 - RealSiteLoader
 - PDSTransfer
 - PLSModel
 - PreprocessPipeline (if available)
 - PrivacyGuard
 - ConformalPredictor

It saves per-site results using `np.savez_compressed` for portability.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional
import argparse
import json
import csv
import logging
import numpy as np

from fedchem.data.real_loader import RealSiteLoader
from fedchem.utils.config import load_and_seed_config
from fedchem.ct.pds_transfer import PDSTransfer
from fedchem.models.pls import PLSModel
from fedchem.privacy.guards import PrivacyGuard
from fedchem.conformal.coverage import ConformalPredictor
from fedchem.conformal.jackknife import jackknife_plus_intervals
from fedchem.privacy.attacks import exfiltration_reconstruction_error, membership_inference_auc
from fedchem.methods import (
    train_site_pls,
    sbc_bias_correct,
    fedavg_site_models,
    fedavg_coef_predict,
    crds_predict,
    train_central_pls,
)
from fedchem.telemetry.logger import TelemetryLogger
from fedchem.conformal.report import summarize_interval
from fedchem.benchmarks import build_method_rows, DEFAULT_METRICS
from fedchem.privacy.report import DPConfig, build_privacy_entry
from fedchem.metrics.metrics import rmsep, r2, cvrmsep

try:
    # Pipeline is optional in some setups; fall back to identity
    from fedchem.preprocess.pipeline import PreprocessPipeline
    from fedchem.utils.config import load_and_seed_config
except Exception:  # pragma: no cover - optional
    class PreprocessPipeline:
        def __init__(self, **kwargs):
            pass

        def fit_transform(self, X):
            return X

        def transform(self, X):
            return X


def _estimate_pds_bytes(pds: PDSTransfer) -> int:
    """Rough estimate of bytes transferred when sharing the PDS mapping."""
    try:
        return int(pds.estimated_bytes())
    except Exception:
        # Fallback: attempt to access via public getters
        matrix = pds.get_global_TC() if hasattr(pds, "get_global_TC") else None
        if matrix is not None:
            return int(getattr(matrix, "nbytes", 0) or 0)
        blocks = pds.get_blocks() if hasattr(pds, "get_blocks") else None
        if not blocks:
            return 0
        total = 0
        for _, _, TC in blocks:
            total += int(getattr(TC, "nbytes", 0) or 0)
        return total


def run_federated_protocol(data_root: Path, k: int = 200,
                           preproc: Optional[str] = "standard",
                           dp_epsilon: Optional[float] = None,
                           dp_delta: float = 1e-5,
                           output_dir: Path = Path("results"),
                           seed: Optional[int] = None,
                           conformal_method: str = "split",
                           clip_feature: float | None = None,
                           clip_target: float | None = None,
                           subsample_q: float | None = None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    loader = RealSiteLoader(data_root)
    # Dynamically enumerate available site directories of the form 'site_<i>' in the data root
    sites = sorted([p.name for p in Path(data_root).iterdir() if p.is_dir() and p.name.startswith("site_")])
    if not sites:
        raise FileNotFoundError(f"No site directories found in {data_root}")

    # Phase 1: Reference site
    logging.info("Phase 1: Training central model at reference site...")
    X_ref, y_ref = loader.load_paired_subset(sites[0], k=k)

    # Configure preprocessing pipeline according to `preproc` string
    # Supported: 'none', 'standard', 'snv', 'msc'
    preproc = (preproc or "standard").lower()
    cfg = load_and_seed_config()
    drift_cfg = cfg.get('DRIFT_AUGMENT') if isinstance(cfg.get('DRIFT_AUGMENT'), dict) else None

    # Allow environment overrides for DRIFT_AUGMENT parameters using flattened env vars.
    # e.g., FEDCHEM_DRIFT_AUGMENT_JITTER_WAVELENGTH_PX=0.01
    def _merge_drift_env_overrides(dcfg):
        import os
        if not isinstance(dcfg, dict):
            dcfg = {}
        mapping = {
            'FEDCHEM_DRIFT_AUGMENT_JITTER_WAVELENGTH_PX': 'jitter_wavelength_px',
            'FEDCHEM_DRIFT_AUGMENT_MULTIPLICATIVE_SCATTER': 'multiplicative_scatter',
            'FEDCHEM_DRIFT_AUGMENT_BASELINE_OFFSET': 'baseline_offset',
            'FEDCHEM_DRIFT_AUGMENT_WHITE_NOISE_SIGMA': 'white_noise_sigma',
            'FEDCHEM_DRIFT_AUGMENT_AUGMENTATION_SEED': 'augmentation_seed',
            'FEDCHEM_DRIFT_AUGMENT_APPLY_AUGMENTATION_DURING_TRAINING': 'apply_augmentation_during_training',
            'FEDCHEM_DRIFT_AUGMENT_APPLY_TEST_SHIFTS': 'apply_test_shifts',
        }
        for env_key, cfg_key in mapping.items():
            if env_key in os.environ:
                val = os.environ.get(env_key)
                if val is None:
                    continue
                # Boolean flags
                if cfg_key.startswith('apply_'):
                    dcfg[cfg_key] = val in {'1', 'true', 'True', 'yes', 'on'}
                    continue
                # Try numeric parsing, fallback to string
                try:
                    if '.' in val:
                        dcfg[cfg_key] = float(val)
                    else:
                        dcfg[cfg_key] = int(val)
                except Exception:
                    try:
                        dcfg[cfg_key] = float(val)
                    except Exception:
                        dcfg[cfg_key] = val
        return dcfg

    drift_cfg = _merge_drift_env_overrides(drift_cfg)
    if preproc == "none":
        pipeline = PreprocessPipeline(drift_cfg=drift_cfg, augmentation_seed=seed)
    elif preproc == "snv":
        pipeline = PreprocessPipeline(scale=None, snv=True, drift_cfg=drift_cfg, augmentation_seed=seed)
    elif preproc == "msc":
        pipeline = PreprocessPipeline(scale=None, msc=True, drift_cfg=drift_cfg, augmentation_seed=seed)
    else:
        pipeline = PreprocessPipeline(scale="standard", snv=False, drift_cfg=drift_cfg, augmentation_seed=seed)
    X_ref_proc = pipeline.fit_transform(X_ref)

    # Limit n_components to a safe upper bound (min(n,p)) to avoid PLS errors
    n_comp = min(15, int(X_ref_proc.shape[1]), int(max(1, X_ref_proc.shape[0])))
    central_model = PLSModel(n_components=n_comp).fit(X_ref_proc, y_ref)

    y_ref_pred = central_model.predict(X_ref_proc)
    conformal = ConformalPredictor(alpha=0.1)
    conformal.fit(np.abs(y_ref - y_ref_pred))

    results = {"sites": [], "privacy_audit": [], "coverage_validation": []}
    interval_summaries: List[dict] = []
    method_rows: List[dict] = []
    telemetry = TelemetryLogger(output_dir / "telemetry.jsonl")

    rng = np.random.default_rng(seed)

    for site_id in sites[1:]:
        logging.info(f"Processing {site_id}...")
        X_site, y_site = loader.load_paired_subset(site_id, k=k)
        X_local_test, y_local_test = loader.load_local_test(site_id)

        X_site_proc = pipeline.transform(X_site)
        X_local_proc = pipeline.transform(X_local_test)

        # choose PDSTransfer window/overlap adaptively to data dimensionality
        d = X_ref_proc.shape[1]
        window = min(32, max(1, d))
        overlap = min(16, max(0, window // 2))
        pds = PDSTransfer(window=window, overlap=overlap, ridge=1e-1).fit(X_ref_proc, X_site_proc,
                                                                         clip_feature=clip_feature,
                                                                         clip_target=clip_target)

        # Optionally apply DP to the transfer matrix(s) using per-block clipping and
        # conservative sensitivity estimates. This returns accounting metadata for
        # transparency and reproducibility. Pass subsampling q (if user used subsampling)
        dp_meta: dict | None = None
        if dp_epsilon is not None:
            try:
                dp_meta = PrivacyGuard.add_dp_noise_to_pds(
                    pds,
                    epsilon=dp_epsilon,
                    delta=dp_delta,
                    clip_feature=(clip_feature if clip_feature is not None else 1.0),
                    clip_target=(clip_target if clip_target is not None else 1.0),
                    rng=rng,
                    q=subsample_q,
                )
                logging.info(f"DP applied to PDS (site={site_id}): meta={dp_meta}")
            except Exception:
                logging.exception("DP noise application failed")
        communication_bytes = _estimate_pds_bytes(pds)
        paired_samples = pds.to_dict().get("n_samples")

        # Compute privacy metric: reconstruction error (exfiltration estimate)
        try:
            X_site_trans = pds.transform(X_site_proc)
            if pds.is_global() and pds.get_global_TC() is not None:
                T = pds.get_global_TC()
                recon_error = exfiltration_reconstruction_error(X_site_proc, X_site_trans, T)
            else:
                T_est = np.linalg.lstsq(X_site_trans, X_site_proc, rcond=None)[0]
                recon_error = exfiltration_reconstruction_error(X_site_proc, X_site_trans, T_est)
        except Exception:
            recon_error = float('inf')

        X_local_trans = pds.transform(X_local_proc)
        y_pred_pds = central_model.predict(X_local_trans)
        y_pred_central_naive = central_model.predict(X_local_proc)

        # ---- Baselines and alternative methods ----
        # Site-specific model (train on site paired)
        site_model = train_site_pls(X_site_proc, y_site)
        y_pred_site_specific = site_model.predict(X_local_proc)

        # SBC: bias-corrected central predictions using paired set
        sbc_predict = sbc_bias_correct(central_model, X_site_proc, y_site)
        y_pred_sbc = sbc_predict(X_local_proc)

        # FedAvg: train site models on each site's paired set and average
        site_models = []
        for sid in sites:
            try:
                Xs_tmp, ys_tmp = loader.load_paired_subset(sid, k=k)
                Xs_tmp_proc = pipeline.transform(Xs_tmp)
                site_models.append(train_site_pls(Xs_tmp_proc, ys_tmp))
            except Exception:
                continue
        # prefer coefficient-level FedAvg when possible
        try:
            y_pred_fedavg = fedavg_coef_predict(site_models, X_local_proc)
        except Exception:
            y_pred_fedavg = fedavg_site_models(site_models, X_local_proc)

        # CRDS-like transform predictions (piecewise)
        try:
            y_pred_crds = crds_predict(central_model, X_ref_proc, X_site_proc, X_local_proc,
                                       window=window, overlap=overlap,
                                       clip_feature=clip_feature, clip_target=clip_target)
        except Exception:
            y_pred_crds = None

        # Conformal intervals on PDS predictions by default
        if conformal_method == "jackknife":
            try:
                # build a factory for the PLSModel used for jackknife
                model_factory = lambda: PLSModel(n_components=15)
                lower_pds, upper_pds = jackknife_plus_intervals(model_factory, X_ref_proc, y_ref, X_local_proc, alpha=0.1)
            except Exception:
                lower_pds, upper_pds = conformal.predict_interval(y_pred_pds)
        else:
            lower_pds, upper_pds = conformal.predict_interval(y_pred_pds)
        coverage_pds = conformal.compute_coverage(y_local_test, lower_pds, upper_pds)

        lower_central, upper_central = conformal.predict_interval(y_pred_central_naive)
        coverage_central = conformal.compute_coverage(y_local_test, lower_central, upper_central)

        interval_summaries.append(summarize_interval(site_id, "pds", y_local_test, lower_pds, upper_pds))
        interval_summaries.append(summarize_interval(site_id, "central", y_local_test, lower_central, upper_central))

        rmsep_pds = float(rmsep(y_local_test, y_pred_pds))
        cvrmsep_pds = float(cvrmsep(y_local_test, y_pred_pds))
        rmsep_site_specific = float(rmsep(y_local_test, y_pred_site_specific))
        cvrmsep_site_specific = float(cvrmsep(y_local_test, y_pred_site_specific))
        rmsep_central_naive = float(rmsep(y_local_test, y_pred_central_naive))
        cvrmsep_central_naive = float(cvrmsep(y_local_test, y_pred_central_naive))
        rmsep_sbc = float(rmsep(y_local_test, y_pred_sbc))
        cvrmsep_sbc = float(cvrmsep(y_local_test, y_pred_sbc))
        rmsep_fedavg = float(rmsep(y_local_test, y_pred_fedavg))
        cvrmsep_fedavg = float(cvrmsep(y_local_test, y_pred_fedavg))
        r2_pds = float(r2(y_local_test, y_pred_pds))
        rmsep_crds = float(rmsep(y_local_test, y_pred_crds)) if y_pred_crds is not None else None
        cvrmsep_crds = float(cvrmsep(y_local_test, y_pred_crds)) if y_pred_crds is not None else None

        method_predictions = {
            "pds": y_pred_pds,
            "central": y_pred_central_naive,
            "site_specific": y_pred_site_specific,
            "sbc": y_pred_sbc,
            "fedavg": y_pred_fedavg,
        }
        if y_pred_crds is not None:
            method_predictions["crds"] = y_pred_crds
        method_rows.extend(build_method_rows(site_id, y_local_test, method_predictions, metrics=DEFAULT_METRICS))

        results_entry = {
            "site_id": site_id,
            "rmsep_pds": rmsep_pds,
            "cvrmsep_pds": cvrmsep_pds,
            "rmsep_site_specific": rmsep_site_specific,
            "cvrmsep_site_specific": cvrmsep_site_specific,
            "rmsep_central_naive": rmsep_central_naive,
            "cvrmsep_central_naive": cvrmsep_central_naive,
            "rmsep_sbc": rmsep_sbc,
            "cvrmsep_sbc": cvrmsep_sbc,
            "rmsep_fedavg": rmsep_fedavg,
            "cvrmsep_fedavg": cvrmsep_fedavg,
            "rmsep_crds": rmsep_crds,
            "cvrmsep_crds": cvrmsep_crds,
            "r2_pds": r2_pds,
            "coverage_pds": coverage_pds,
            "coverage_central": coverage_central,
            "interval_width_pds": float(np.mean(upper_pds - lower_pds)),
            "interval_width_central": float(np.mean(upper_central - lower_central)),
            "privacy_recon_error": recon_error,
            "pds_mode": pds.to_dict().get("mode"),
            "communication_bytes": int(communication_bytes),
            "pds_bytes": int(communication_bytes),
            "paired_samples": None if paired_samples is None else int(paired_samples),
            "dp_reported_eps": None if dp_meta is None else dp_meta.get("reported_epsilons"),
        }
        # Cond numbers if available
        try:
            results_entry["max_cond"] = float(np.max(pds.diagnostics.cond_numbers))
        except Exception:
            results_entry["max_cond"] = None

        results["sites"].append(results_entry)

        # Privacy audit: membership inference (paired vs local held-out)
        try:
            # positive = paired site samples, negative = local held-out
            X_pos = X_site_proc
            X_neg = X_local_proc
            auc = membership_inference_auc(X_pos, X_neg, classifier='logreg')
        except Exception:
            auc = None
        dp_config = DPConfig(
            epsilon=dp_epsilon,
            delta=dp_delta,
            clip_feature=clip_feature,
            clip_target=clip_target,
            subsample_q=subsample_q,
        )
        privacy_entry = build_privacy_entry(
            site_id,
            reconstruction_error=recon_error,
            membership_auc=auc,
            dp_config=dp_config,
            dp_metadata=dp_meta,
            communication_bytes=communication_bytes,
            paired_samples=paired_samples,
        )
        results["privacy_audit"].append(privacy_entry)

        telemetry.log_round(
            site_id=site_id,
            method="pds_transfer",
            communication_bytes=int(communication_bytes),
            pds_bytes=int(communication_bytes),
            paired_samples=None if paired_samples is None else int(paired_samples),
            dp_epsilon=dp_epsilon,
            dp_reported_eps=None if dp_meta is None else dp_meta.get("reported_epsilons"),
            recon_error=float(recon_error),
            membership_auc=None if auc is None else float(auc),
            window=window,
            overlap=overlap,
            pds_mode=pds.to_dict().get("mode"),
        )

        # Save per-site predictions using portable npz
        intervals = np.stack([lower_pds, upper_pds], axis=1)
        np.savez_compressed(output_dir / f"{site_id}_predictions.npz",
                            y_true=y_local_test, y_pred=y_pred_pds, intervals=intervals)

    results["coverage_validation"] = interval_summaries

    with open(output_dir / "federated_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Also write a small per-site communication map to simplify merging later
    try:
        per_site_map = {r.get('site_id'): int(r.get('communication_bytes', 0) or 0) for r in results.get('sites', [])}
        with open(output_dir / "per_site_communication.json", "w", encoding='utf-8') as fh:
            json.dump(per_site_map, fh, indent=2)
    except Exception:
        logging.exception("Failed to write per_site_communication.json")

    csv_path = output_dir / "5site_real_experiment.csv"
    keys = [
        "site_id",
        "rmsep_pds",
        "cvrmsep_pds",
        "rmsep_central_naive",
        "cvrmsep_central_naive",
        "rmsep_site_specific",
        "cvrmsep_site_specific",
        "rmsep_sbc",
        "cvrmsep_sbc",
        "rmsep_fedavg",
        "cvrmsep_fedavg",
        "rmsep_crds",
        "cvrmsep_crds",
        "coverage_pds",
        "coverage_central",
        "interval_width_pds",
        "interval_width_central",
        "privacy_recon_error",
        "communication_bytes",
        "max_cond",
    ]
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for row in results["sites"]:
            writer.writerow({k: row.get(k) for k in keys})

    privacy_path = output_dir / "privacy_metadata.json"
    with open(privacy_path, "w") as fh:
        json.dump(results["privacy_audit"], fh, indent=2)

    coverage_path = output_dir / "coverage_summary.json"
    with open(coverage_path, "w") as fh:
        json.dump(results["coverage_validation"], fh, indent=2)

    methods_csv = output_dir / "methods_summary.csv"
    fieldnames = ["site_id", "method", *DEFAULT_METRICS]
    with open(methods_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in method_rows:
            writer.writerow(row)

    logging.info(f"Experiment complete. Results saved to {output_dir}")
    return results


def _cli():
    # Load config for default CLI values (env overrides remain authoritative)
    cfg = load_and_seed_config()
    conf_def = cfg.get('CONFORMAL', {}).get('method') if isinstance(cfg.get('CONFORMAL'), dict) else None
    default_conformal = conf_def if isinstance(conf_def, str) else "split"
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=Path, required=True)
    p.add_argument("--k", type=int, default=None,
                   help="single k value (ignored if --k_values provided)")
    p.add_argument("--k_values", type=str, default=None,
                   help="comma-separated k values for grid runs, e.g. 20,40,80")
    p.add_argument("--preproc", type=str, default="standard",
                   help="preprocessing: none, standard, snv, msc")
    p.add_argument("--dp_epsilon", type=float, default=None)
    p.add_argument("--dp_delta", type=float, default=1e-5)
    p.add_argument("--output_dir", type=Path, default=Path("results/real_experiment"))
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--conformal", type=str, default=default_conformal, choices=["split", "jackknife"],
                   help="conformal method: split (default) or jackknife")
    p.add_argument("--clip_feature", type=float, default=None,
                   help="L2 clip bound for reference rows before fitting PDS (None=disabled)")
    p.add_argument("--clip_target", type=float, default=None,
                   help="L2 clip bound for site rows before fitting PDS (None=disabled)")
    p.add_argument("--subsample_q", type=float, default=None,
                   help="Poisson subsampling fraction q for DP accounting (None=disabled)")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO)

    # determine k grid
    if args.k_values:
        k_list = [int(x) for x in args.k_values.split(",") if x.strip()]
    elif args.k is not None:
        k_list = [args.k]
    else:
        k_list = [200]

    for k in k_list:
        out_dir = args.output_dir / f"k_{k}_preproc_{args.preproc}"
        out_dir.mkdir(parents=True, exist_ok=True)
        run_federated_protocol(
            args.data_root,
            k=k,
            preproc=args.preproc,
            dp_epsilon=args.dp_epsilon,
            dp_delta=args.dp_delta,
            output_dir=out_dir,
            seed=args.seed,
            conformal_method=args.conformal,
            clip_feature=args.clip_feature,
            clip_target=args.clip_target,
            subsample_q=args.subsample_q,
        )


if __name__ == "__main__":
    _cli()
