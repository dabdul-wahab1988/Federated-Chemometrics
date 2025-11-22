"""
Script for Objective 1: Build an end-to-end federated CT framework on the real site data.

Generates Figure 1 and Table 1 using actual federation logs:
- Per-round convergence metrics (RMSEP/R^2/MAE) via orchestrator eval_fn
- Real communication bytes sent/received per round
- Optional DP epsilon progression and participation rate
"""

import os
import time
import platform
from collections.abc import Callable, Iterable
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from pathlib import Path
from fedchem.federated.orchestrator import FederatedOrchestrator
from fedchem.federated.secure_aggregation import SimulatedSecureAggregator
from fedchem.models.linear import LinearModel
from fedchem.models.pls import PLSModel
from fedchem.models.parametric_pls import ParametricPLSModel
from fedchem.metrics.metrics import rmsep, r2, mae
from fedchem.utils.real_data import load_idrc_wheat_shootout_site_dict
from fedchem.utils.logging_utils import extract_logs_for_manifest, create_log_summary
from fedchem.experimental_design import generate_experimental_design
from fedchem.utils.manifest_utils import resolve_design_version
from fedchem.utils.model_registry import instantiate_model
from fedchem.utils.real_data import resample_spectra
from fedchem.utils.config import load_and_seed_config, get_experimental_sites, get_data_config

# Seed environment variables from `config.yaml` for generators run directly
cfg = load_and_seed_config()
# Load config via centralized helper (still provide 'config' for legacy name)
config: Dict[str, Any] = cfg
config_sites = get_experimental_sites(cfg)
data_config = get_data_config(cfg)


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

# Resolve if resampling of spectra is enabled (env/config) using helper _coerce_bool
resample_cfg = _coerce_bool(config.get('RESAMPLE_SPECTRA'), True)

plt.rcParams.update({'font.size': 12})

output_dir = Path(str(config.get('OUTPUT_DIR', 'generated_figures_tables')))
output_dir.mkdir(parents=True, exist_ok=True)

ClientDict = Dict[str, Any]


DEFAULT_N_WAVELENGTHS = _coerce_int(config.get('DEFAULT_N_WAVELENGTHS'), 256) or 256


def _parse_optional_int(name: str, default: int | None = None) -> int | None:
    return _coerce_int(os.environ.get(name), default)


def _parse_optional_float(name: str, default: float | None = None) -> float | None:
    return _coerce_float(os.environ.get(name), default)


def _resolve_bool_from_env(env_name: str, cfg_value: Any, default: bool = False) -> bool:
    env_val = os.environ.get(env_name)
    if env_val is not None:
        return _coerce_bool(env_val, default)
    return _coerce_bool(cfg_value, default)


def _resolve_fedpls_enabled(default: bool = False) -> bool:
    """Resolve whether FedPLS should be enabled.

    FedPLS is only enabled when both the method is specified (via env or config) and
    the `FEDCHEM_USE_FEDPLS` flag resolves to true. If the method is missing (None/"")
    FedPLS is disabled regardless of `FEDCHEM_USE_FEDPLS`.
    """
    method = _resolve_fedpls_method()
    if not method:
        return False
    fedpls_cfg_val = config.get('USE_FEDPLS')
    return _resolve_bool_from_env("FEDCHEM_USE_FEDPLS", fedpls_cfg_val, default)


def _resolve_fedpls_method() -> Optional[str]:
    env_method = os.environ.get("FEDCHEM_FEDPLS_METHOD")
    if env_method:
        return env_method
    # Prefer unprefixed key (`FEDPLS_METHOD`) in config; fall back to legacy `FEDCHEM_FEDPLS_METHOD` if present
    if 'FEDPLS_METHOD' in config:
        return config.get('FEDPLS_METHOD')
    return None


def _resolve_rounds(rounds_arg: int | None) -> int:
    env_rounds = _parse_optional_int("FEDCHEM_ROUNDS")
    if env_rounds is not None:
        return env_rounds
    if rounds_arg is not None:
        return int(rounds_arg)
    cfg_rounds = _coerce_int(config.get('ROUNDS'), 20)
    return cfg_rounds if cfg_rounds is not None else 20


def _resolve_ct_federated_variant() -> str | None:
    """Resolve the CT federated variant token from env or config.

    Priority: env->config['CT_FEDERATED_VARIANT']->None
    Allowed values that this script recognizes: 'local_then_pooled', 'local_then_secure_aggregate',
    'global_calibrate_after_fed'.
    """
    env = os.environ.get("FEDCHEM_CT_FEDERATED_VARIANT")
    if env:
        return str(env).strip()
    cfg_val = config.get("CT_FEDERATED_VARIANT")
    if isinstance(cfg_val, str) and cfg_val:
        return cfg_val.strip()
    return None


def _align_feature_matrices(arrays: list[np.ndarray], target_n_wavelengths: int | None = None) -> list[np.ndarray]:
    """Trim all arrays in `arrays` to the minimum number of columns present.

    Like generate_objective_5, this avoids vstack errors when datasets don't have
    uniform numbers of wavelengths across sites.
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
        effective_target = min(min_w, int(target_n_wavelengths)) if min_w is not None else int(target_n_wavelengths)
        if effective_target <= 0:
            effective_target = min_w
        method = os.environ.get("FEDCHEM_RESAMPLE_METHOD", "interpolate")
        resampled = []
        for arr in arrs:
            try:
                Xr, _ = resample_spectra(np.asarray(arr), col_names=None, n_wavelengths=effective_target, method=method)
                resampled.append(Xr)
            except Exception:
                if arr.ndim == 2 and arr.shape[1] > min_w:
                    resampled.append(arr[:, :min_w])
                else:
                    resampled.append(arr)
        return resampled
    if any(a.shape[1] != min_w for a in arrs if a.ndim == 2):
        trimmed = [a[:, :min_w] if (a.ndim == 2 and a.shape[1] > min_w) else a for a in arrs]
        return trimmed
    return arrs


def _parse_target_epsilon(default: float = 2.0) -> float:
    val = os.environ.get("FEDCHEM_DP_TARGET_EPS")
    try:
        return float(val) if val else default
    except Exception:
        return default

def _make_eval_fn(clients: List[ClientDict]) -> Callable[[Any], Dict[str, float]]:
    """Create an eval_fn that computes pooled validation metrics across sites.

    Uses last 20% of each site's data as validation; trains are handled by FL.
    """
    # Precompute validation splits
    X_val_list = []
    y_val_list = []
    for c in clients:
        Xc, yc = c["X"], c["y"]
        n = Xc.shape[0]
        split = max(1, int(n * 0.8))
        X_val_list.append(Xc[split:])
        y_val_list.append(yc[split:])
    # Align features across validation splits before stacking
    try:
        X_val_list = [np.asarray(x) for x in X_val_list]
        X_val_list = _align_feature_matrices(X_val_list)
    except Exception:
        pass
    X_val = np.vstack(X_val_list)
    y_val = np.hstack(y_val_list)

    def eval_fn(model):
        yhat = model.predict(X_val)
        return {
            "rmsep": rmsep(y_val, yhat),
            "r2": r2(y_val, yhat),
            "mae": mae(y_val, yhat),
        }

    return eval_fn


def run_federated_experiment(
    clients: List[ClientDict],
    rounds: int | None = None,
    with_dp: bool = True,
    seed: int = 42,
    *,
    target_epsilon: float | None = None,
    participation_rate: float | None = None,
    participation_schedule: Optional[List[float]] = None,
    compression_schedule: Optional[List[float]] = None,
    dp_delta: float | None = None,
):
    """Run federated learning for FedAvg/FedProx variants and collect logs."""
    eval_fn = _make_eval_fn(clients)
    results: Dict[str, Dict[str, Any]] = {}
    # Shared DP config (optional) to demonstrate epsilon tracking
    dp_config: Dict[str, Any] | None = None
    clip_norm: float | None = None
    effective_rounds = _resolve_rounds(rounds)
    # Resolve DP target epsilon with precedence: explicit arg -> env -> config -> default
    if target_epsilon is not None:
        target_eps = float(target_epsilon)
    else:
        cfg_eps_float = _coerce_float(config.get('DP_TARGET_EPS'), 2.0) or 2.0
        target_eps = _parse_target_epsilon(default=cfg_eps_float)
    if with_dp:
        # Clip norm precedence: env -> config -> default (1.0)
        clip_norm_default = _coerce_float(config.get('CLIP_NORM'), 1.0) or 1.0
        clip_norm = _parse_optional_float("FEDCHEM_CLIP_NORM", clip_norm_default)
        if clip_norm is None:
            clip_norm = 1.0
        # Helper parsers keep precedence: env > explicit arg > config > fallback defaults
        part_sched_env = os.environ.get("FEDCHEM_PARTICIPATION_SCHEDULE")
        comp_sched_env = os.environ.get("FEDCHEM_COMPRESSION_SCHEDULE")

        def _parse_sched(raw: Optional[str]) -> Optional[List[float]]:
            s = raw or ""
            if not s:
                return None
            try:
                vals = [float(x.strip()) for x in s.split(',') if x.strip()]
                return vals if vals else None
            except Exception:
                return None

        def _coerce_schedule(values: Optional[Iterable[float]]) -> Optional[List[float]]:
            if values is None:
                return None
            try:
                seq = [float(v) for v in values]
            except Exception:
                return None
            max_rounds = max(1, int(effective_rounds))
            seq = seq[:max_rounds]
            return seq

        part_sched = _parse_sched(part_sched_env)
        if part_sched is None:
            part_sched = _coerce_schedule(participation_schedule)
        if part_sched is None:
            part_sched = _coerce_schedule(config.get('PARTICIPATION_SCHEDULE'))
        # No hard-coded default participation schedule: leave as None if not provided

        comp_sched = _parse_sched(comp_sched_env)
        if comp_sched is None:
            comp_sched = _coerce_schedule(compression_schedule)
        if comp_sched is None:
            comp_sched = _coerce_schedule(config.get('COMPRESSION_SCHEDULE'))
        # No hard-coded default compression schedule: leave as None if not provided

        rate_env = os.environ.get("FEDCHEM_PARTICIPATION_RATE")
        effective_participation_rate = None
        if rate_env is not None:
            try:
                effective_participation_rate = float(rate_env)
            except Exception:
                effective_participation_rate = None
        if effective_participation_rate is None and participation_rate is not None:
            try:
                effective_participation_rate = float(participation_rate)
            except Exception:
                effective_participation_rate = None
        if effective_participation_rate is None:
            cfg_rate = config.get('PARTICIPATION_RATE')
            if cfg_rate is not None:
                try:
                    effective_participation_rate = float(cfg_rate)
                except Exception:
                    effective_participation_rate = None

        delta_env = os.environ.get("FEDCHEM_DP_DELTA")
        if delta_env is not None:
            delta_value = _coerce_float(delta_env, None)
        elif dp_delta is not None:
            delta_value = float(dp_delta)
        else:
            delta_value = _coerce_float(config.get('DP_DELTA'), 1e-5)
        if delta_value is None:
            delta_value = 1e-5

        dp_config = {
            "delta": delta_value,
            "target_epsilon": target_eps,
        }
        if effective_participation_rate is not None:
            dp_config["participation_rate"] = effective_participation_rate
        if part_sched is not None:
            dp_config["participation_schedule"] = part_sched
        if comp_sched is not None:
            dp_config["compression_schedule"] = comp_sched
        clip_norm = float(clip_norm)
        # Allow DIFF PRIV block in config to specify a noise_multiplier map (convenience)
        dp_section = config.get('DIFFERENTIAL_PRIVACY') if isinstance(config.get('DIFFERENTIAL_PRIVACY'), dict) else None
        if dp_section is not None and dp_config.get('noise_std') is None:
            noise_map = dp_section.get('noise_multiplier_map') or {}
            # Keys in map can be strings like 'inf', '10.0', '1.0', '0.1'
            key_candidate = str(target_eps) if target_eps is not None else str(config.get('DP_TARGET_EPS'))
            key_candidate = key_candidate.replace(' ', '')
            if key_candidate in noise_map:
                nm = noise_map.get(key_candidate)
                # Guard against None values to avoid passing None to float()
                if nm is None:
                    dp_config['noise_multiplier'] = nm
                else:
                    try:
                        dp_config['noise_multiplier'] = float(nm)
                    except Exception:
                        dp_config['noise_multiplier'] = nm

    server_eta = _parse_optional_float("FEDCHEM_SERVER_ETA", 1.0) or 1.0

    # Determine whether to use secure aggregation based on config or env
    use_secure = _resolve_bool_from_env("FEDCHEM_USE_SECURE_AGGREGATION", config.get('SECURE_AGGREGATION', {}).get('enabled') if isinstance(config.get('SECURE_AGGREGATION'), dict) else config.get('SECURE_AGGREGATION', False), False)
    secure_aggregator = None
    if use_secure:
        # Use Simulated secure aggregator for now; can be replaced with real implementation
        try:
            secure_aggregator = SimulatedSecureAggregator(rng_seed=int(seed or 0))
        except Exception:
            secure_aggregator = None

    # FedAvg
    orch = FederatedOrchestrator()
    results["FedAvg"] = orch.run_rounds(
        clients=clients,
    model=instantiate_model(config.get("PIPELINE", {}).get("model", "LinearModel")),
        rounds=effective_rounds,
        algo="fedavg",
        dp_config=dp_config,
        clip_norm=clip_norm,
        eval_fn=eval_fn,
        seed=seed,
        secure_aggregator=secure_aggregator,
    )

    # FedProx
    orch2 = FederatedOrchestrator()
    results["FedProx"] = orch2.run_rounds(
        clients=clients,
    model=instantiate_model(config.get("PIPELINE", {}).get("model", "LinearModel")),
        rounds=effective_rounds,
        algo="fedprox",
        prox_mu=0.1,
        dp_config=dp_config,
        clip_norm=clip_norm,
        eval_fn=eval_fn,
        server_eta=server_eta,
        seed=seed + 1,
        secure_aggregator=secure_aggregator,
    )
    # FedAvg without DP for comparison (privacy-utility impact)
    orch3 = FederatedOrchestrator()
    results["FedAvg_noDP"] = orch3.run_rounds(
        clients=clients,
    model=instantiate_model(config.get("PIPELINE", {}).get("model", "LinearModel")),
        rounds=effective_rounds,
        algo="fedavg",
        dp_config=None,
        clip_norm=None,
        eval_fn=eval_fn,
        server_eta=server_eta,
        seed=seed + 2,
        secure_aggregator=secure_aggregator,
    )
    # Optional: Federated PLS. When enabled, run federated PLS using
    # ParametricPLSModel so that clients produce linear (w,b) parameters
    # which the orchestrator can aggregate using the existing FedAvg logic.
    if _resolve_fedpls_enabled():
        orch4 = FederatedOrchestrator()
        # Build a ParametricPLSModel factory for federated usage.
        pls_model = ParametricPLSModel(n_components=None, max_components=20, cv=5, random_state=0)
        # Use the same effective rounds as other algos so comparison is fair.
        results["FedPLS"] = orch4.run_rounds(
            clients=clients,
            model=pls_model,
            rounds=effective_rounds,
            algo="fedpls",
            dp_config=dp_config,
            clip_norm=clip_norm,
            eval_fn=eval_fn,
            seed=seed + 3,
            secure_aggregator=secure_aggregator,
        )
    return results

def generate_figure_1(results_by_algo):
    """Figure 1: Convergence (FedAvg vs FedProx, with/without DP) and Communication/Privacy."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # (b) Convergence (RMSEP per round for each algorithm)
    for algo, res in results_by_algo.items():
        logs = res.get("logs", [])
        rounds = [log.get("round", i + 1) for i, log in enumerate(logs)]
        rmseps = [log.get("rmsep") for log in logs]
        style = '-' if 'noDP' not in algo else '--'
        axes[0].plot(rounds, rmseps, marker='o', linewidth=2.0, label=algo, linestyle=style)
    axes[0].set_xlabel("Rounds", labelpad=8)
    axes[0].set_ylabel("RMSEP (validation)", labelpad=10)
    axes[0].set_title("1(a) Convergence: FedAvg vs FedProx", pad=14)
    # Place legend below the subplot (centered) to avoid overlap with data
    axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=False)
    axes[0].tick_params(axis='both', which='major', labelsize=14)

    # Optional debug overlay: UpdateNorm and Participation for FedAvg
    if os.environ.get("FEDCHEM_PLOT_DEBUG", "1") == "1":
        fa = results_by_algo.get("FedAvg")
        if fa:
            flog = fa.get("logs", [])
            rnds = [log.get("round", i + 1) for i, log in enumerate(flog)]
            upd = [log.get("update_norm") for log in flog]
            prt = [log.get("participation_rate") for log in flog]
            ax0b = axes[0].twinx()
            ax0b.plot(rnds, upd, color='crimson', linestyle=':', linewidth=1.8, label='UpdateNorm (right)')
            ax0b.set_ylabel("UpdateNorm")
            # Right y-axis on 1(a): show one decimal place
            ax0b.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            ax0c = axes[0].twinx()
            # Move the third axis further right to avoid overlap with the second y-axis
            ax0c.spines.right.set_position(("axes", 1.19))
            ax0c.scatter(rnds, prt, color='darkgreen', marker='x', label='Participation')
            ax0c.set_ylim(0.0, 1.05)
            ax0c.set_ylabel("Participation")
            ax0c.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            # Merge legends
            l0, lab0 = axes[0].get_legend_handles_labels()
            l1, lab1 = ax0b.get_legend_handles_labels()
            l2, lab2 = ax0c.get_legend_handles_labels()
            # Place merged legend below the subplot (centered)
            axes[0].legend(l0 + l1 + l2, lab0 + lab1 + lab2, loc='upper center', bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=False)

    # (c) Communication and privacy (bytes per round and epsilon progression)
    # Use FedAvg run as reference for bytes; overlay epsilon if available
    ref = results_by_algo.get("FedAvg") or next(iter(results_by_algo.values()))
    logs = ref.get("logs", [])
    rounds = [log.get("round", i + 1) for i, log in enumerate(logs)]
    bytes_total = [int(log.get("bytes_sent", 0)) + int(log.get("bytes_recv", 0)) for log in logs]
    # Convert to KB for readability
    kb_total = np.array(bytes_total, dtype=float) / 1024.0
    cum_kb = np.cumsum(kb_total)
    ax1 = axes[1]
    ax1.bar(rounds, kb_total, alpha=0.6, label="KB/round")
    ax1.set_xlabel("Rounds", labelpad=8)
    ax1.set_ylabel("KB/round", labelpad=10)
    ax1.tick_params(axis='both', which='major', labelsize=14)
    ax1.set_title("1(b) Communication & Privacy")
    # Add cumulative KB as a line on primary axis
    ax1.plot(rounds, cum_kb, color='navy', marker='.', linewidth=2.0, label='Cumulative KB')
    # Secondary axis for epsilon
    ax2 = ax1.twinx()
    eps_prog = [log.get("epsilon_so_far") for log in logs]
    if any(e is not None for e in eps_prog):
        ax2.plot(rounds, eps_prog, color='crimson', marker='s', linewidth=2.0, label='Epsilon')
        ax2.set_ylabel("Epsilon", labelpad=12)
    # Third axis for participation (to avoid mixing scales)
    ax3 = ax1.twinx()
    # Move the third axis further right to provide more space between the two right y-axes
    ax3.spines.right.set_position(("axes", 1.28))
    parts = [log.get("participation_rate") for log in logs]
    if any(p is not None for p in parts):
        ax3.scatter(rounds, parts, color='darkgreen', marker='x', label='Participation')
        ax3.set_ylim(0.0, 1.05)
    # Add slightly more padding so the label doesn't run into the axis spine
    ax3.set_ylabel("Participation", labelpad=18)
    # Add legend combining handles
    lines, labels = ax1.get_legend_handles_labels()
    if any(e is not None for e in eps_prog):
        l2, lab2 = ax2.get_legend_handles_labels()
        lines += l2; labels += lab2
    l3, lab3 = ax3.get_legend_handles_labels()
    lines += l3; labels += lab3
    # Place legend below the subplot (centered)
    axes[1].legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=False)
    # Increase left/right margins to reduce crowding
    plt.subplots_adjust(top=0.88, left=0.08, right=0.92, wspace=0.25)
    plt.tight_layout(rect=(0,0,1,0.96))
    plt.savefig(output_dir / "figure_1.png", dpi=300)
    plt.close()

def generate_table_1(results_by_algo):
    """Table 1: Federation Summary from logs (all algorithms)."""
    rows = []
    for algo, res in results_by_algo.items():
        logs = res.get("logs", [])
        dp = res.get("dp", {}) or {}
        cum = 0
        # For centralized baselines like PLS, emit a single summary row
        if algo.lower().startswith("centralized_pls") or algo == "Centralized_PLS":
            if logs:
                log = logs[-1]
                rows.append({
                    "Algo": algo,
                    "Round": 0,
                    "BytesSent": 0,
                    "BytesRecv": 0,
                    "BytesTotal": 0,
                    "CumulativeBytes": 0,
                    "DurationSec": None,
                    "RMSEP": log.get("rmsep"),
                    "R2": log.get("r2"),
                    "MAE": log.get("mae"),
                    "ParticipationRate": None,
                    "EpsilonSoFar": None,
                    "DP_Epsilon_Final": None,
                    "DP_Delta": None,
                    "DP_NoiseStd": None,
                    "UsedModel": res.get("used_model") if isinstance(res, dict) else None,
                })
            continue
        for log in logs:
            bt = int(log.get("bytes_sent", 0)) + int(log.get("bytes_recv", 0))
            cum += bt
            rows.append({
                "Algo": algo,
                "Round": log.get("round"),
                "BytesSent": log.get("bytes_sent"),
                "BytesRecv": log.get("bytes_recv"),
                "BytesTotal": bt,
                "CumulativeBytes": cum,
                "DurationSec": log.get("duration_sec"),
                "RMSEP": log.get("rmsep"),
                "R2": log.get("r2"),
                "MAE": log.get("mae"),
                "ParticipationRate": log.get("participation_rate"),
                "EpsilonSoFar": log.get("epsilon_so_far"),
                # New instrumentation fields from the orchestrator
                "UpdateNorm": log.get("update_norm"),
                "WeightNorm": log.get("weight_norm"),
                "Participants": log.get("participants"),
                "ClipNormUsed": log.get("clip_norm_used"),
                "CompressionRatio": log.get("compression_ratio"),
                "EvalError": log.get("eval_error"),
                "ServerEta": log.get("server_eta"),
                "DP_Epsilon_Final": dp.get("epsilon"),
                "DP_Delta": dp.get("delta"),
                # Prefer per-round dp_noise_std if present; fall back to run-level
                "DP_NoiseStd": log.get("dp_noise_std", dp.get("noise_std")),
                "UsedModel": res.get("used_model") if isinstance(res, dict) else None,
            })
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "table_1.csv", index=False)
def main(
    clients: Optional[List[ClientDict]] = None,
    ds_meta: Optional[Dict[str, Any]] = None,
):
    start_time = time.perf_counter()
    seed_env = os.environ.get("FEDCHEM_SEED")
    try:
        seed = int(seed_env) if seed_env is not None else 42
    except Exception:
        seed = 42
    import random as _random
    _random.seed(int(seed))
    np.random.seed(int(seed))

    # Use top-level config values (fallback to env if present)
    num_sites = _parse_optional_int("FEDCHEM_NUM_SITES", _coerce_int(config.get('NUM_SITES'), 3)) or 3
    rounds_cfg_default = _coerce_int(config.get('ROUNDS'), 20) or 20
    rounds_cfg = _parse_optional_int("FEDCHEM_ROUNDS", rounds_cfg_default) or rounds_cfg_default
    participation_rate = _parse_optional_float(
        "FEDCHEM_PARTICIPATION_RATE",
        _coerce_float(config.get('PARTICIPATION_RATE'), 1.0),
    )
    participation_rate = participation_rate if participation_rate is not None else 1.0
    participation_schedule = config.get('PARTICIPATION_SCHEDULE') or None
    compression_schedule = config.get('COMPRESSION_SCHEDULE') or None

    quick_env = os.environ.get("FEDCHEM_QUICK")
    quick = quick_env == "1" if quick_env is not None else _coerce_bool(config.get('QUICK'), False)
    force_env = os.environ.get("FEDCHEM_USE_TECATOR")
    force_tecator = force_env == "1" if force_env is not None else _coerce_bool(config.get('USE_TECATOR'), False)
    dp_target_default = _coerce_float(config.get('DP_TARGET_EPS'), 2.0) or 2.0
    target_eps = _parse_target_epsilon(dp_target_default)
    n_wavelengths_default = _coerce_int(config.get('DEFAULT_N_WAVELENGTHS'), DEFAULT_N_WAVELENGTHS) or DEFAULT_N_WAVELENGTHS
    n_wavelengths = _parse_optional_int("FEDCHEM_N_WAVELENGTHS", n_wavelengths_default)
    if n_wavelengths is None:
        n_wavelengths = n_wavelengths_default
    max_transfer_default = _coerce_int(config.get('MAX_TRANSFER_SAMPLES'))
    max_transfer_samples = _parse_optional_int("FEDCHEM_MAX_TRANSFER_SAMPLES", max_transfer_default)
    dp_delta_override: float | None = None
    if clients is None or ds_meta is None:
        exp_design = config.get("EXPERIMENTAL_DESIGN", {}) or {}
        factors = exp_design.get("FACTORS", {}) if isinstance(exp_design.get("FACTORS"), dict) else {}
        config_sites = factors.get("Site", None)
        data_config = config.get("DATA_CONFIG") or {}
        site_data, ds_meta = load_idrc_wheat_shootout_site_dict(
            sites=config_sites,
            data_dir="data",
            n_wavelengths=n_wavelengths,
            quick_cap_per_site=None,
            seed=seed,
            max_transfer_samples=max_transfer_samples,
            data_config=data_config,
            resample=resample_cfg,
        )
        clients = [{"X": v["cal"][0], "y": v["cal"][1]} for v in site_data.values()]
    if not clients:
        raise ValueError("No clients available for Objective 1 experiment.")
    ds_meta = ds_meta or {}

    # Generate experimental design if specified in config and apply first combo overrides
    design = generate_experimental_design(config)
    applied_design = design[0] if design else {}
    design_cfg = config.get('EXPERIMENTAL_DESIGN', {}) or {}
    design_type = design_cfg.get('DESIGN_TYPE')
    if applied_design:
        # Map known factor names to runtime parameters
        if 'Rounds' in applied_design:
            try:
                rounds_cfg = int(applied_design['Rounds'])
            except Exception:
                pass
        if 'DP_Target_Eps' in applied_design:
            try:
                target_eps = float(applied_design['DP_Target_Eps']) if applied_design['DP_Target_Eps'] != '∞' else float('inf')
            except Exception:
                pass
        if 'Participation_Rate' in applied_design:
            try:
                participation_rate = float(applied_design['Participation_Rate'])
            except Exception:
                pass
        if 'DP_Delta' in applied_design:
            try:
                dp_delta_override = float(applied_design['DP_Delta'])
            except Exception:
                pass
        if 'Transfer_Samples' in applied_design:
            maybe_transfer = _coerce_int(applied_design['Transfer_Samples'], max_transfer_samples)
            if maybe_transfer is not None:
                max_transfer_samples = maybe_transfer
        # Parse participation/compression schedules if provided in the design
        if 'Participation_Schedule' in applied_design:
            try:
                val = applied_design['Participation_Schedule']
                if isinstance(val, str):
                    participation_schedule = [float(x.strip()) for x in val.split(',') if x.strip()]
                elif isinstance(val, list):
                    participation_schedule = [float(x) for x in val]
            except Exception:
                pass
        if 'Compression_Schedule' in applied_design:
            try:
                val = applied_design['Compression_Schedule']
                if isinstance(val, str):
                    compression_schedule = [float(x.strip()) for x in val.split(',') if x.strip()]
                elif isinstance(val, list):
                    compression_schedule = [float(x) for x in val]
            except Exception:
                pass
        # Use canonical `DP_Target_Eps` if present in the applied design
        if 'DP_Target_Eps' in applied_design:
            try:
                target_eps = float(applied_design['DP_Target_Eps']) if applied_design['DP_Target_Eps'] != '∞' else float('inf')
            except Exception:
                pass
        # No fallback on legacy key; use canonical `DP_Target_Eps` in experimental design.
    try:
        import json as _json
        _applied = _json.dumps(applied_design, ensure_ascii=True)
    except Exception:
        _applied = str(applied_design)
    design_kw = f" ({design_type})" if design_type else ""
    print(f"Experimental design combos loaded: {len(design)}{design_kw}; using first: {_applied}")

    # Best-effort: estimate PDSTransfer communication footprint per client and attach
    # as `client["communication_bytes"]`. This instrumentation is non-fatal: any
    # failure falls back to leaving/setting zero so experiment runs are unaffected.
    try:
        from fedchem.ct.pds_transfer import PDSTransfer
    except Exception:
        PDSTransfer = None

    if PDSTransfer is not None:
        try:
            def _estimate_pds_bytes(pds):
                try:
                    return int(pds.estimated_bytes())
                except Exception:
                    # best-effort fallback
                    try:
                        g = pds.get_global_TC() if hasattr(pds, "get_global_TC") else None
                        if g is not None:
                            return int(getattr(g, "nbytes", 0) or 0)
                        blocks = pds.get_blocks() if hasattr(pds, "get_blocks") else None
                        if blocks:
                            total = 0
                            for _, _, tc in blocks:
                                total += int(getattr(tc, "nbytes", 0) or 0)
                            return total
                    except Exception:
                        return 0
                return 0

            # Build a pooled reference sample bounded by `max_transfer_samples`.
            try:
                ref_n = int(max(1, min(max_transfer_samples or 100, 100)))
            except Exception:
                ref_n = 100
            ref_list = []
            for c in clients:
                try:
                    n = c["X"].shape[0]
                    take = min(ref_n, n)
                    if take > 0:
                        ref_list.append(c["X"][:take])
                except Exception:
                    continue
            if ref_list:
                try:
                    ref_X = np.vstack(ref_list)
                except Exception:
                    ref_X = None

                if ref_X is not None:
                    for idx, c in enumerate(clients):
                        try:
                            n = c["X"].shape[0]
                            take = min(ref_n, n)
                            tgt_X = c["X"][:take] if take > 0 else c["X"]
                            p = PDSTransfer()
                            p.fit(ref_X, tgt_X)
                            cb = _estimate_pds_bytes(p)
                            c["communication_bytes"] = int(cb or 0)
                        except Exception:
                            c["communication_bytes"] = int(c.get("communication_bytes", 0) or 0)
                else:
                    for c in clients:
                        c["communication_bytes"] = int(c.get("communication_bytes", 0) or 0)
            else:
                # No reference samples available: set zeros conservatively
                for c in clients:
                    c["communication_bytes"] = int(c.get("communication_bytes", 0) or 0)
            print("PDSTransfer per-client communication_bytes estimated and attached to clients (approx).")
        except Exception:
            for c in clients:
                c["communication_bytes"] = int(c.get("communication_bytes", 0) or 0)
    else:
        for c in clients:
            c["communication_bytes"] = int(c.get("communication_bytes", 0) or 0)

    results = run_federated_experiment(
        clients,
        rounds=rounds_cfg,
        seed=seed,
        target_epsilon=target_eps,
        participation_rate=participation_rate,
        participation_schedule=participation_schedule,
        compression_schedule=compression_schedule,
        dp_delta=dp_delta_override,
    )

    X_tr_list, y_tr_list, X_val_list, y_val_list = [], [], [], []
    for c in clients:
        Xc, yc = c["X"], c["y"]
        n = Xc.shape[0]
        split = max(1, int(n * 0.8))
        X_tr_list.append(Xc[:split])
        y_tr_list.append(yc[:split])
        X_val_list.append(Xc[split:])
        y_val_list.append(yc[split:])
    # Align all stacked train/val matrices to common width before vstack
    try:
        X_tr_list = [np.asarray(x) for x in X_tr_list]
        X_tr_list = _align_feature_matrices(X_tr_list, target_n_wavelengths=DEFAULT_N_WAVELENGTHS)
        X_val_list = [np.asarray(x) for x in X_val_list]
        X_val_list = _align_feature_matrices(X_val_list, target_n_wavelengths=DEFAULT_N_WAVELENGTHS)
    except Exception:
        pass
    X_tr = np.vstack(X_tr_list)
    y_tr = np.hstack(y_tr_list)
    X_val = np.vstack(X_val_list)
    y_val = np.hstack(y_val_list)

    pls = PLSModel(n_components=None, max_components=20, cv=5, random_state=0)
    pls.fit(X_tr, y_tr)
    yhat = pls.predict(X_val)
    pls_rmsep = rmsep(y_val, yhat)

    ref_any = next(iter(results.values()))
    n_rounds = len(ref_any.get("logs", [])) or rounds_cfg
    pls_logs = [{"round": i + 1, "rmsep": float(pls_rmsep)} for i in range(n_rounds)]
    results["Centralized_PLS"] = {"logs": pls_logs, "dp": {}}

    # Compute additional CT Federated variant metrics if a variant token is present
    ct_variant = _resolve_ct_federated_variant()
    # Helper: collect pooled calibration/training set (reuse pooled training X_tr/y_tr)
    X_pool = X_tr
    y_pool = y_tr
    if ct_variant:
        try:
            print(f"Applying CT federated variant: {ct_variant}")
        except Exception:
            pass
        # (1) local_then_pooled: site-local models then pooled calibration mapping
        if ct_variant == "local_then_pooled":
            local_rmse = []
            per_site_preds_local = {}
            for idx, c in enumerate(clients):
                Xc = c["X"]; yc = c["y"]
                n = Xc.shape[0]
                split = max(1, int(n * 0.8))
                X_tr_site, y_tr_site = Xc[:split], yc[:split]
                X_te_site, y_te_site = Xc[split:], yc[split:]
                # local model: use PLS by default
                local_model = PLSModel(n_components=None, max_components=20, cv=5, random_state=0)
                try:
                    local_model.fit(X_tr_site, y_tr_site)
                    y_pool_pred = local_model.predict(X_pool)
                    # pooled calibration: linear mapping from local predictions -> pooled y
                    A = np.vstack([y_pool_pred, np.ones_like(y_pool_pred)]).T
                    coeffs, *_ = np.linalg.lstsq(A, y_pool, rcond=None)
                    a, b = coeffs
                    yhat_adj = a * local_model.predict(X_te_site) + b
                    local_rmse.append(float(rmsep(y_te_site, yhat_adj)))
                    per_site_preds_local[idx] = {"y_te": y_te_site, "yhat_local_then_pooled": yhat_adj}
                except Exception:
                    # fallback to site-specific if local fails
                    site_model = PLSModel(n_components=None, max_components=20, cv=5, random_state=0).fit(X_tr_site, y_tr_site)
                    yhat = site_model.predict(X_te_site)
                    local_rmse.append(float(rmsep(y_te_site, yhat)))
                    per_site_preds_local[idx] = {"y_te": y_te_site, "yhat_local_then_pooled": yhat}
            mean_rmse = float(np.mean(local_rmse)) if local_rmse else None
            logs = [{"round": 1, "rmsep": mean_rmse, "r2": None, "mae": None}]
            results["Local_then_pooled"] = {"logs": logs, "dp": {}, "preds": per_site_preds_local}

        # (2) local_then_secure_aggregate: aggregate site-local models via secure aggregator equivalent
        if ct_variant == "local_then_secure_aggregate":
            weights = []
            biases = []
            for idx, c in enumerate(clients):
                Xc = c["X"]; yc = c["y"]
                n = Xc.shape[0]
                split = max(1, int(n * 0.8))
                X_tr_site, y_tr_site = Xc[:split], yc[:split]
                X_te_site, y_te_site = Xc[split:], yc[split:]
                m = PLSModel(n_components=None, max_components=20, cv=5, random_state=0)
                try:
                    m.fit(X_tr_site, y_tr_site)
                    # Try common attributes for extracting linear parameters from fitted PLS models.
                    w = None
                    b = None
                    # Prefer scikit-learn-like attributes if present
                    try:
                        if hasattr(m, "coef_"):
                            coef = np.asarray(getattr(m, "coef_"))
                            w = coef.ravel()
                        if hasattr(m, "intercept_"):
                            b = float(np.asarray(getattr(m, "intercept_")).ravel()[0])
                    except Exception:
                        w = None
                        b = None
                    # Fallback to a params dict or other model-specific attributes
                    if w is None:
                        try:
                            params = getattr(m, "params", None)
                            if isinstance(params, dict) and "w" in params:
                                w = np.asarray(params.get("w")).astype(float)
                                b = float(params.get("b", 0.0))
                        except Exception:
                            w = None
                            b = None
                    # If we have a weight vector, record it for aggregation
                    if w is not None:
                        weights.append(np.asarray(w).astype(float))
                        biases.append(float(b) if b is not None else 0.0)
                except Exception:
                    continue
            if weights:
                # average params as a simple aggregator (secure aggregator simulates aggregation in FL)
                w_avg = np.mean(np.vstack(weights), axis=0)
                b_avg = float(np.mean(np.asarray(biases))) if biases else 0.0
                # Build a simple LinearModel to evaluate
                global_model = LinearModel()
                try:
                    global_model.set_params({"w": w_avg, "b": np.array([b_avg], dtype=float)})
                except Exception:
                    pass
                # Evaluate on pooled validation set constructed earlier
                yhat_pool = global_model.predict(X_val)
                pooled_rmsep = float(rmsep(y_val, yhat_pool))
            else:
                pooled_rmsep = None
            logs = [{"round": 1, "rmsep": pooled_rmsep, "r2": None, "mae": None}]
            results["Local_then_secure_aggregate"] = {"logs": logs, "dp": {}, "model_params": {"w": w_avg.tolist() if weights else None, "b": b_avg if weights else None}}

        # (3) global_calibrate_after_fed: take final federated global model and calibrate on pooled transfer set
        if ct_variant == "global_calibrate_after_fed":
            fed_avg_res = results.get("FedAvg")
            if fed_avg_res and isinstance(fed_avg_res, dict):
                fed_model = fed_avg_res.get("model")
                if fed_model is not None:
                    try:
                        y_pred_pool = fed_model.predict(X_pool)
                        A = np.vstack([y_pred_pool, np.ones_like(y_pred_pool)]).T
                        coeffs, *_ = np.linalg.lstsq(A, y_pool, rcond=None)
                        a, b = coeffs
                        yhat_calibrated = a * fed_model.predict(X_val) + b
                        cal_rmsep = float(rmsep(y_val, yhat_calibrated))
                        logs = [{"round": 1, "rmsep": cal_rmsep, "r2": None, "mae": None}]
                        results["Global_calibrate_after_fed"] = {"logs": logs, "dp": {}, "calibration": {"a": float(a), "b": float(b)}}
                    except Exception:
                        results["Global_calibrate_after_fed"] = {"logs": [{"round": 1, "rmsep": None}], "dp": {}}
                else:
                    results["Global_calibrate_after_fed"] = {"logs": [{"round": 1, "rmsep": None}], "dp": {}}
            else:
                results["Global_calibrate_after_fed"] = {"logs": [{"round": 1, "rmsep": None}], "dp": {}}

    generate_figure_1(results)
    generate_table_1(results)
    ds_name = ds_meta.get("dataset", "real")
    caption = (
        f"Figure 1. (a) Validation RMSEP per round for FedAvg, FedProx, and FedAvg without DP on {ds_name} data; flat line shows centralized PLS baseline. "
        "(b) Communication bytes per round (parameter payloads, float64) with cumulative bytes (line), epsilon progression and participation rate (q). "
        "Units: RMSEP scale=raw; epsilon computed with delta=1e-5, clip_norm=1.0, variable q schedule. "
        + ("Quick mode enabled (reduced samples); numbers are for smoke testing, not final reporting." if quick else "")
    )
    (output_dir / "figure_1_caption.txt").write_text(caption, encoding="utf-8")

    summary = {}
    for algo, res in results.items():
        logs = res.get("logs", [])
        if logs:
            last = logs[-1]
            summary[algo] = {
                "rmsep": last.get("rmsep"),
                "r2": last.get("r2"),
                "mae": last.get("mae"),
                "epsilon": last.get("epsilon_so_far"),
                "used_model": res.get("used_model") if isinstance(res, dict) else None,
            }

    # Extract communication metrics from logs
    ref_result = next(iter(results.values())) if results else {}
    ref_logs = ref_result.get("logs", [])
    total_bytes = sum(
        int(log.get("bytes_sent", 0)) + int(log.get("bytes_recv", 0))
        for log in ref_logs
    )
    total_bytes_mb = total_bytes / (1024 * 1024)

    import json
    total_time_sec = float(time.perf_counter() - start_time)
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    # Extract logs for each algorithm using helper function
    logs_by_algo = extract_logs_for_manifest(results)
    
    # Create summary statistics from logs
    log_summary = create_log_summary(logs_by_algo)
    # Resolve pipeline configuration from config.yaml (fallbacks provided)
    cfg_pipeline = config.get("PIPELINE")
    if not isinstance(cfg_pipeline, dict):
        cfg_pipeline = {}
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
            "n_sites": num_sites,
            "rounds": n_rounds,
            "quick_mode": bool(quick),
            "secure_aggregation": False,
            "pipeline": pipeline,
            "seed": seed,
            "standard_design": True,
            "design_version": resolve_design_version(config),
            "force_tecator": force_tecator,
            "n_wavelengths_requested": ds_meta.get("n_wavelengths_requested"),
            "n_wavelengths_actual": ds_meta.get("n_wavelengths_actual"),
            "transfer_samples_requested": ds_meta.get("transfer_samples_requested"),
            "transfer_samples_used": ds_meta.get("transfer_samples_used"),
            "participation_rate": participation_rate,
            "participation_schedule": participation_schedule,
            "compression_schedule": compression_schedule,
            "test_samples_per_site": 30,
            "target_epsilon": target_eps,
            # Record which federated PLS mode was requested/used (if any).
            # Expected values: 'simpls', 'parametric' or None when FedPLS not used.
            "fedpls_method": fedpls_method,
        },
        "summary": summary,
        "metrics": {"metric_scale": "raw"},
        "logs_by_algorithm": logs_by_algo,
        "log_summary": log_summary,
        # Map of algorithm -> used_model to report fallbacks explicitly
        "used_models": {algo: (res.get("used_model") if isinstance(res, dict) else None) for algo, res in results.items()},
        "runtime": {
            "wall_time_total_sec": total_time_sec,
            "total_bytes": total_bytes,
            "total_bytes_mb": total_bytes_mb,
        },
        "versions": versions,
    }
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
    # Update manifest config with reproducibility blocks and secure_aggregation flag
    manifest_cfg = manifest.get('config', {}) or {}
    manifest_cfg['DIFFERENTIAL_PRIVACY'] = config.get('DIFFERENTIAL_PRIVACY')
    manifest_cfg['CONFORMAL'] = config.get('CONFORMAL')
    manifest_cfg['SECURE_AGGREGATION'] = config.get('SECURE_AGGREGATION')
    manifest_cfg['DRIFT_AUGMENT'] = config.get('DRIFT_AUGMENT')
    manifest_cfg['REPRODUCIBILITY'] = config.get('REPRODUCIBILITY')
    manifest_cfg['secure_aggregation'] = bool(config.get('SECURE_AGGREGATION', False))
    # CT-specific federated variant token requested for this run
    ct_variant = _resolve_ct_federated_variant()
    manifest_cfg['ct_federated_variant'] = ct_variant
    manifest['config'] = manifest_cfg

    # Include an explicit combo id derived from the run config for easy correlation
    from fedchem.utils.manifest_utils import compute_combo_id
    manifest["combo_id"] = compute_combo_id(manifest.get("config"))
    (output_dir / "manifest_1.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print("Objective 1 completed.")

if __name__ == "__main__":
    main()
