"""
Script for Objective 6: Privacy and communication on the real 5-site dataset.

Generates Figure 6 and Table 6 using real federated telemetry:
- 6(a) Epsilon progression vs RMSEP (pooled test)
- 6(b) Communication bytes per round
- 6(c) Participation and compression heatmap

Env toggles:
- FEDCHEM_NUM_SITES: number of clients/sites (default 5)
- FEDCHEM_ROUNDS: federated rounds (default 10)
- FEDCHEM_DP_TARGET_EPS: target epsilon for DP (default 2.0)
- FEDCHEM_DP_DELTA: delta for DP (default 1e-5)
- FEDCHEM_PARTICIPATION_RATE: fixed client participation rate (0-1)
- FEDCHEM_PARTICIPATION_SCHEDULE: comma-separated participation per round (overrides rate)
- FEDCHEM_COMPRESSION_SCHEDULE: comma-separated compression ratio per round (e.g., 1,0.5,0.5,...)
- FEDCHEM_CLIP_NORM: gradient/update clip norm (default None)
- FEDCHEM_SERVER_ETA: server damping factor (default 1.0)
- FEDCHEM_QUICK=1: reduce samples for speed
- FEDCHEM_USE_TECATOR=1: force Tecator fallback instead of the persistent 5-site design
"""

import inspect
import json
import os
import platform
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

from fedchem.ct.pds_transfer import PDSTransfer
from fedchem.federated.orchestrator import FederatedOrchestrator
from fedchem.federated.secure_aggregation import SimulatedSecureAggregator
from fedchem.metrics.metrics import rmsep
from fedchem.models.pls import PLSModel
from fedchem.utils.config import load_and_seed_config, load_config, get_experimental_sites, get_data_config
from fedchem.utils.logging_utils import create_log_summary
from fedchem.utils.manifest_utils import resolve_design_version
from fedchem.utils.model_registry import MODEL_REGISTRY, instantiate_model
from fedchem.utils.real_data import load_real_site_dict

# Seed environment variables from `config.yaml` for generators run directly
cfg = load_and_seed_config()
config = cfg

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


def _parse_optional_int(name: str, default: int | None = None) -> int | None:
    return _coerce_int(os.environ.get(name), default)


def _parse_optional_float(name: str, default: float | None = None) -> float | None:
    return _coerce_float(os.environ.get(name), default)


def _resolve_model_label() -> str:
    env_label = os.environ.get("FEDCHEM_PIPELINE_MODEL")
    if env_label:
        return env_label
    pipeline_cfg = config.get("PIPELINE") if isinstance(config.get("PIPELINE"), dict) else {}
    cfg_label = pipeline_cfg.get("model") if isinstance(pipeline_cfg, dict) else None
    if isinstance(cfg_label, str) and cfg_label:
        return cfg_label
    if os.environ.get("FEDCHEM_USE_LINEAR", "0") == "1":
        return "LinearModel"
    return "PLSModel"


def _instantiate_pipeline_model(*args, **kwargs):
    label = _resolve_model_label()
    pipeline_cfg = config.get("PIPELINE") if isinstance(config.get("PIPELINE"), dict) else {}
    pipeline_cfg = pipeline_cfg or {}
    ctor_kwargs: dict[str, Any] = {}
    cls = MODEL_REGISTRY.get(label)
    if cls is not None:
        try:
            sig = inspect.signature(cls.__init__)
            allowed = [p for p in sig.parameters.keys() if p not in ("self", "args", "kwargs")]
            for key, value in pipeline_cfg.items():
                if key in allowed:
                    ctor_kwargs[key] = value
        except Exception:
            ctor_kwargs = {}
    ctor_kwargs.update(kwargs)
    return instantiate_model(label, *args, **ctor_kwargs)


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


# Visual configuration: easy to tweak for publication-quality figures
DEFAULT_FONT_SIZE = 14
LEGEND_FONTSIZE = 14
AXIS_LABEL_SIZE = 14
TICK_LABEL_SIZE = 14
TITLE_SIZE = 14
LEGEND_KWARGS = {"loc": "best", "fontsize": LEGEND_FONTSIZE}

plt.rcParams.update({'font.size': DEFAULT_FONT_SIZE})
sns.set_style("whitegrid")

output_dir = Path(str(config.get('OUTPUT_DIR', 'generated_figures_tables')))
output_dir.mkdir(exist_ok=True)

DEFAULT_N_WAVELENGTHS = _coerce_int(config.get('DEFAULT_N_WAVELENGTHS'), 256) or 256

def _safe_float(val):
    try:
        return float(val) if val is not None else None
    except Exception:
        return None

def _parse_schedule(s: str | None) -> list[float] | None:
    if not s:
        return None
    try:
        vals = [float(x.strip()) for x in s.split(',') if x.strip()]
        return vals if vals else None
    except Exception:
        return None


def _normalize_schedule(value: str | Sequence[float] | None) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return _parse_schedule(value)
    try:
        return [float(x) for x in value]
    except Exception:
        return None

def _make_clients(
    n_sites: int = 5,
    seed: int = 42,
    quick: bool = False,
    force_tecator: bool = False,
    n_wavelengths: int | None = None,
    max_transfer_samples: int | None = None,
    config_sites: list[str] | None = None,
    data_config: dict | None = None,
):
    cfg = load_config()
    if n_wavelengths is None:
        n_wavelengths_default = _coerce_int(cfg.get('DEFAULT_N_WAVELENGTHS'), DEFAULT_N_WAVELENGTHS) or DEFAULT_N_WAVELENGTHS
        n_wavelengths = _parse_optional_int("FEDCHEM_N_WAVELENGTHS", n_wavelengths_default)
    if max_transfer_samples is None:
        max_transfer_default = _coerce_int(cfg.get('MAX_TRANSFER_SAMPLES'))
        max_transfer_samples = _parse_optional_int("FEDCHEM_MAX_TRANSFER_SAMPLES", max_transfer_default)
        data, meta = load_real_site_dict(
        n_sites=n_sites,
        seed=seed,
        quick=quick,
        force_tecator=force_tecator,
        n_wavelengths=n_wavelengths,
        max_transfer_samples=max_transfer_samples,
        config_sites=config_sites,
    )
    clients = []
    Xte_list, yte_list = [], []
    for name in sorted(data.keys()):
        site = data[name]
        X, y = site["X"], site["y"]
        n = X.shape[0]
        n_tr = max(1, int(0.8 * n))
        clients.append({"X": X[:n_tr], "y": y[:n_tr]})
        Xte_list.append(X[n_tr:])
        yte_list.append(y[n_tr:])
    # Best-effort: attach per-client communication_bytes estimated via PDSTransfer
    try:
        pooled_ref = np.vstack([data[n]["X"] for n in sorted(data.keys())]) if data else None
        if pooled_ref is not None and pooled_ref.shape[0] > 0:
            d = pooled_ref.shape[1]
            window = min(32, max(1, d))
            overlap = min(16, max(0, window // 2))
            for c in clients:
                try:
                    Xc = c.get("X")
                    if Xc is None or Xc.size == 0:
                        c["communication_bytes"] = 0
                        continue
                    k = min(50, pooled_ref.shape[0], Xc.shape[0])
                    p = PDSTransfer(window=window, overlap=overlap, ridge=1e-1)
                    p.fit(pooled_ref[:k], Xc[:k])
                    c["communication_bytes"] = int(p.estimated_bytes())
                except Exception:
                    c["communication_bytes"] = 0
    except Exception:
        # non-fatal if estimation fails
        pass
    X_test = np.vstack(Xte_list)
    y_test = np.hstack(yte_list)
    return clients, X_test, y_test, meta

def _default_participation_schedule(rounds: int) -> list[float]:
    # Step-down schedule to show effect of subsampling
    sched = []
    for r in range(1, rounds + 1):
        if r <= max(1, rounds // 5):
            sched.append(1.0)
        elif r <= max(1, 2 * rounds // 5):
            sched.append(0.8)
        elif r <= max(1, 3 * rounds // 5):
            sched.append(0.6)
        elif r <= max(1, 4 * rounds // 5):
            sched.append(0.5)
        else:
            sched.append(0.4)
    return sched

def _default_compression_schedule(rounds: int) -> list[float]:
    sched = []
    for r in range(1, rounds + 1):
        if r <= max(1, rounds // 5):
            sched.append(1.0)
        elif r <= max(1, 2 * rounds // 5):
            sched.append(0.8)
        else:
            sched.append(0.6)
    return sched

def _plot_figure_6_from_logs(logs: list[dict], baseline_rmsep: float):
    """Plot Figure 6 using provided logs and baseline RMSEP (no re-run)."""
    rounds_idx = list(range(1, len(logs) + 1))
    eps_series = [l.get("epsilon_so_far") for l in logs]
    rmsep_series = [l.get("rmsep") for l in logs]
    # simple smoothing to reduce noise
    def _smooth(xs, k=2):
        if xs is None:
            return None
        out = []
        for i in range(len(xs)):
            win = xs[max(0, i - k + 1): i + 1]
            try:
                out.append(float(np.mean([v for v in win if v is not None])))
            except Exception:
                out.append(xs[i])
        return out
    rmsep_smooth = _smooth(rmsep_series, k=2)
    bytes_sent = [int(l.get("bytes_sent", 0)) for l in logs]
    bytes_recv = [int(l.get("bytes_recv", 0)) for l in logs]
    bytes_round = [s + r for s, r in zip(bytes_sent, bytes_recv)]
    participation = [l.get("participation_rate") for l in logs]
    compression = [l.get("compression_ratio") for l in logs]
    # Uncompressed bytes approximate
    uncomp = []
    for b, c in zip(bytes_round, compression):
        cc = float(c) if c is not None and float(c) > 1e-9 else 1.0
        uncomp.append(int(b / cc))
    cum_bytes = list(np.cumsum(bytes_round))

    # Baseline (non-DP pooled)
    base_rmsep = float(baseline_rmsep)

    # Create a 2x2 layout but make the bottom row a single axis spanning both columns
    fig = plt.figure(figsize=(16, 12), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    # (a) RMSEP over rounds with epsilon on secondary axis (top-left)
    ax_a = fig.add_subplot(gs[0, 0])
    rmsep_plot = [float(x) if x is not None else float('nan') for x in (rmsep_smooth or [])]
    ax_a.plot(rounds_idx, rmsep_plot, marker='o', label='RMSEP (smoothed)')
    ax_a.axhline(base_rmsep, color='gray', linestyle='--', linewidth=1, label='Baseline (no DP)')
    ax2 = ax_a.twinx()
    eps_plot = [float(x) if x is not None else float('nan') for x in eps_series]
    ax2.plot(rounds_idx, eps_plot, color='tab:orange', marker='x', label='Epsilon')
    ax_a.set_xlabel("Round", fontsize=AXIS_LABEL_SIZE)
    ax_a.set_ylabel("RMSEP (pooled)", fontsize=AXIS_LABEL_SIZE)
    ax2.set_ylabel("Epsilon", fontsize=AXIS_LABEL_SIZE)
    ax_a.set_title("6(a) Privacy–utility over rounds", fontsize=TITLE_SIZE)
    # combined legend
    lines, labels = ax_a.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax_a.legend(lines + lines2, labels + labels2, **LEGEND_KWARGS)
    ax_a.grid(True, linestyle='--', alpha=0.3)
    # Force integer round ticks and readable tick label sizes
    ax_a.set_xticks(rounds_idx)
    try:
        from matplotlib.ticker import MaxNLocator
        ax_a.xaxis.set_major_locator(MaxNLocator(integer=True))
    except Exception:
        pass
    ax_a.tick_params(axis='both', which='major', labelsize=TICK_LABEL_SIZE)
    ax2.tick_params(axis='y', which='major', labelsize=TICK_LABEL_SIZE)

    # (b) Bytes per round: actual vs uncompressed (log scale), with cumulative on secondary axis (log scale) (top-right)
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.plot(rounds_idx, bytes_round, marker='s', label='Actual bytes')
    ax_b.plot(rounds_idx, uncomp, marker='^', linestyle='--', label='Uncompressed (approx)')
    ax2b = ax_b.twinx()
    ax2b.plot(rounds_idx, cum_bytes, color='tab:green', alpha=0.5, label='Cumulative bytes')
    ax_b.set_xlabel("Round", fontsize=AXIS_LABEL_SIZE)
    ax_b.set_ylabel("Bytes (round, log)", fontsize=AXIS_LABEL_SIZE)
    ax2b.set_ylabel("Bytes (cumulative, log)", fontsize=AXIS_LABEL_SIZE)
    ax_b.set_yscale('log')
    ax2b.set_yscale('log')
    ax_b.set_title("6(b) Communication efficiency (log-scale)")
    lines_b, labels_b = ax_b.get_legend_handles_labels()
    lines_b2, labels_b2 = ax2b.get_legend_handles_labels()
    ax_b.legend(lines_b + lines_b2, labels_b + labels_b2, **LEGEND_KWARGS)
    ax_b.grid(True, linestyle='--', alpha=0.3)
    # Force integer round ticks and readable tick label sizes
    ax_b.set_xticks(rounds_idx)
    try:
        from matplotlib.ticker import MaxNLocator
        ax_b.xaxis.set_major_locator(MaxNLocator(integer=True))
    except Exception:
        pass
    ax_b.tick_params(axis='both', which='major', labelsize=TICK_LABEL_SIZE)
    ax2b.tick_params(axis='y', which='major', labelsize=TICK_LABEL_SIZE)

    # (c) Participation/Compression/EpsilonFrac heatmap (spans full bottom row)
    eps_final = float(eps_series[-1]) if eps_series and eps_series[-1] is not None else None
    eps_frac = [float(e)/eps_final if eps_final and e is not None else 0.0 for e in eps_series]
    mat = np.vstack([
        np.array(participation, dtype=float) if participation else np.zeros(len(logs)),
        np.array(compression, dtype=float) if compression else np.ones(len(logs)),
        np.array(eps_frac, dtype=float),
    ])
    ax_c = fig.add_subplot(gs[1, :])
    sns.heatmap(mat, ax=ax_c, annot=True, fmt='.2f', cmap='YlGnBu', cbar=True,
                vmin=0.0, vmax=1.0,
                yticklabels=["Participation", "Compression", "Epsilon frac"], xticklabels=[str(r) for r in rounds_idx])
    ax_c.set_xlabel("Round", fontsize=AXIS_LABEL_SIZE)
    ax_c.set_title("6(c) Protocol schedules", fontsize=TITLE_SIZE)
    ax_c.tick_params(axis='both', which='major', labelsize=TICK_LABEL_SIZE)

    fig_path = output_dir / "figure_6.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()

def generate_table_6(logs: list[dict], used_model: str | None = None):
    """Table 6: Round-by-round privacy and communication telemetry."""
    rows = []
    cum = 0
    for l in logs:
        br = int(l.get("bytes_sent", 0)) + int(l.get("bytes_recv", 0))
        comp_raw = l.get("compression_ratio")
        comp = _safe_float(comp_raw)
        if comp is None:
            comp = 1.0
        uncomp = int(br / max(comp, 1e-9))
        cum += br
        r_idx = int(l.get("round", 0))
        part = _safe_float(l.get("participation_rate"))
        eps = _safe_float(l.get("epsilon_so_far"))
        sens_r = _safe_float(l.get("sensitivity_round"))
        noise_std = _safe_float(l.get("dp_noise_std"))
        noise_mult_eff = (float(noise_std) / float(sens_r)) if (noise_std is not None and sens_r not in (None, 0.0)) else None
        rows.append({
            "Round": int(l.get("round", 0)),
            "Epsilon": eps,
            "Participation": part,
            "BytesSent": int(l.get("bytes_sent", 0)),
            "BytesRecv": int(l.get("bytes_recv", 0)),
            "BytesRound": br,
            "CumulativeBytes": cum,
            "UncompressedBytes": uncomp,
            "Compression": comp,
            "ClipNorm": _safe_float(l.get("clip_norm_used")),
            "NoiseStd": noise_std,
            "NoiseMultiplierEff": _safe_float(noise_mult_eff),
            "ServerEta": _safe_float(l.get("server_eta")),
            "UpdateNorm": _safe_float(l.get("update_norm")),
            "WeightNorm": _safe_float(l.get("weight_norm")),
            "ClipFraction": _safe_float(l.get("clip_fraction")),
            "MeanUpdateNormRaw": _safe_float(l.get("mean_update_norm_raw")),
            "MeanUpdateNormClipped": _safe_float(l.get("mean_update_norm_clipped")),
            "SensitivityUsed": sens_r,
            "RMSEP": _safe_float(l.get("rmsep")),
            "DurationSec": _safe_float(l.get("duration_sec")),
            "EffectiveRound": float(part) * r_idx if part is not None else None,
            "UsedModel": used_model if used_model is not None else l.get("used_model"),
        })
    pd.DataFrame(rows).to_csv(output_dir / "table_6.csv", index=False)

def _write_caption_meta(logs: list[dict], config: dict, used_models: dict | None = None):
    eps = [l.get("epsilon_so_far") for l in logs if l.get("epsilon_so_far") is not None]
    rm = [l.get("rmsep") for l in logs if l.get("rmsep") is not None]
    final_eps = _safe_float(eps[-1]) if eps else None
    final_rm = _safe_float(rm[-1]) if rm else None
    quick = bool(config.get("quick_mode", False))
    eps_str = f"{final_eps:.3f}" if final_eps is not None else "NA"
    rm_str = f"{final_rm:.3f}" if final_rm is not None else "NA"
    caption = (
        "Figure 6. Privacy and communication telemetry from a federated run. "
        "Panels: (a) RMSEP vs epsilon (LinearModel vs centralized PLS baseline); "
        "(b) bytes per round vs uncompressed estimate with cumulative bytes (both y-axes log scale); "
        "(c) participation, compression, epsilon fraction schedules. "
        f"Final epsilon={eps_str}; Final RMSEP={rm_str}; rounds={config.get('rounds')}; sites={config.get('n_sites')}; clip_norm={config.get('clip_norm')}; "
        f"dataset={config.get('dataset', 'real')}. "
        "Units: RMSEP scale=raw. "
        + ("Quick mode enabled (reduced samples); values are illustrative." if quick else "")
    )
    (output_dir / "figure_6_caption.txt").write_text(caption, encoding="utf-8")
    meta = {
        "figure": "figure_6.png",
        "config": config,
        "final": {"epsilon": final_eps, "rmsep": final_rm},
        "panels": {
            "6a": "Epsilon vs RMSEP (pooled)",
            "6b": "Bytes per round",
            "6c": "Participation & compression heatmap",
        },
    }
    (output_dir / "figure_6_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    # Additional manifest file (lightweight)
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    # Ensure canonical reproducibility keys exist on config
    seed_env = os.environ.get("FEDCHEM_SEED")
    try:
        seed_val = int(seed_env) if seed_env is not None else 42
    except Exception:
        seed_val = 42
    if "seed" not in config:
        config["seed"] = seed_val
    if "standard_design" not in config:
        config["standard_design"] = True
    # Resolve design version deterministically when not provided
    if "design_version" not in config:
        config["design_version"] = resolve_design_version(config)

    # Resolve pipeline configuration and FedPLS selection from config (fallback to env)
    pipeline_section = config.get("pipeline") if isinstance(config.get("pipeline"), dict) else None
    if pipeline_section is None:
        cfg_pipeline = config.get("PIPELINE")
        if not isinstance(cfg_pipeline, dict):
            cfg_pipeline = {}
        # Safely extract values from cfg_pipeline without calling .get on None
        model_val = cfg_pipeline.get("model") if isinstance(cfg_pipeline, dict) else None
        x_scaler_val = cfg_pipeline.get("x_scaler") if isinstance(cfg_pipeline, dict) else None
        y_scaler_val = cfg_pipeline.get("y_scaler") if isinstance(cfg_pipeline, dict) else None
        pipeline_section = {
            "model": model_val if model_val is not None else _resolve_model_label(),
            "x_scaler": x_scaler_val,
            "y_scaler": y_scaler_val,
        }
    else:
        pipeline_section = dict(pipeline_section)
        pipeline_section.setdefault("model", _resolve_model_label())
    config["pipeline"] = pipeline_section

    if "fedpls_enabled" not in config:
        config["fedpls_enabled"] = _resolve_fedpls_enabled()
    if config.get("fedpls_enabled"):
        if "fedpls_method" not in config:
            config["fedpls_method"] = _resolve_fedpls_method()
    else:
        config["fedpls_method"] = None

    manifest = {
        "config": config,
        "summary": {
            "final_epsilon": final_eps,
            "final_rmsep": final_rm,
            "rounds": config.get('rounds'),
        },
        "metrics": {"metric_scale": "raw"},
        "logs_by_algorithm": {"FedAvg": logs},
        # Map algorithm -> used_model (useful when FedPLS falls back)
        "used_models": {"FedAvg": (used_models.get("FedAvg") if used_models and "FedAvg" in used_models else None)},
        "log_summary": create_log_summary({"FedAvg": logs}),
        "versions": versions,
    }
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
    # Place a top-level runtime record for manifest checker if provided in config
    if isinstance(config.get("runtime"), dict):
        manifest["runtime"] = config.get("runtime")
    from fedchem.utils.manifest_utils import compute_combo_id
    manifest["combo_id"] = compute_combo_id(manifest.get("config"))
    (output_dir / "manifest_6.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

def main(
    clients: Optional[list[dict[str, np.ndarray]]] = None,
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    ds_meta: Optional[Dict[str, Any]] = None,
):
    # Generate and capture logs for both figure and table (single run shared across artifacts)
    start_time = time.perf_counter()
    cfg = config if isinstance(config, dict) else {}

    n_sites_default = _coerce_int(cfg.get('NUM_SITES'), 5) or 5
    n_sites = _parse_optional_int("FEDCHEM_NUM_SITES", n_sites_default) or n_sites_default
    n_sites = max(1, n_sites)

    rounds_default = _coerce_int(cfg.get('ROUNDS'), 20) or 20
    rounds = _parse_optional_int("FEDCHEM_ROUNDS", rounds_default) or rounds_default
    rounds = max(1, rounds)

    quick = _resolve_bool_from_env("FEDCHEM_QUICK", cfg.get('QUICK'), False)

    target_eps_default = _coerce_float(cfg.get('DP_TARGET_EPS'), 2.0) or 2.0
    target_eps = _parse_optional_float("FEDCHEM_DP_TARGET_EPS", target_eps_default) or target_eps_default

    delta_default = _coerce_float(cfg.get('DP_DELTA'), 1e-5) or 1e-5
    delta = _parse_optional_float("FEDCHEM_DP_DELTA", delta_default) or delta_default

    clip_norm_default = _coerce_float(cfg.get('CLIP_NORM'), 1.0) or 1.0
    clip_norm = _parse_optional_float("FEDCHEM_CLIP_NORM", clip_norm_default) or clip_norm_default

    server_eta_default = _coerce_float(cfg.get('SERVER_ETA'), 1.0) or 1.0
    server_eta = _parse_optional_float("FEDCHEM_SERVER_ETA", server_eta_default) or server_eta_default

    part_sched = _parse_schedule(os.environ.get("FEDCHEM_PARTICIPATION_SCHEDULE"))
    part_rate = None
    part_rate_env = os.environ.get("FEDCHEM_PARTICIPATION_RATE")
    if part_sched is None:
        part_sched = _parse_schedule(part_rate_env)
    if part_sched is None:
        part_rate = _safe_float(part_rate_env)
    cfg_part_sched = cfg.get('PARTICIPATION_SCHEDULE')
    if part_sched is None and cfg_part_sched is not None:
        part_sched = _parse_schedule(cfg_part_sched) if isinstance(cfg_part_sched, str) else _normalize_schedule(cfg_part_sched)
    if part_sched is None and part_rate is None:
        cfg_part_rate = cfg.get('PARTICIPATION_RATE')
        if cfg_part_rate is not None:
            part_rate = _safe_float(cfg_part_rate)

    comp_sched = _parse_schedule(os.environ.get("FEDCHEM_COMPRESSION_SCHEDULE"))
    if comp_sched is None:
        cfg_comp_sched = cfg.get('COMPRESSION_SCHEDULE')
        if cfg_comp_sched is not None:
            comp_sched = _parse_schedule(cfg_comp_sched) if isinstance(cfg_comp_sched, str) else _normalize_schedule(cfg_comp_sched)

    part_sched = _normalize_schedule(part_sched)
    comp_sched = _normalize_schedule(comp_sched)

    if part_sched is None:
        part_sched = _default_participation_schedule(rounds)
    if comp_sched is None:
        comp_sched = _default_compression_schedule(rounds)

    part_sched = part_sched[:rounds]
    comp_sched = comp_sched[:rounds]

    seed_default = _coerce_int(cfg.get('SEED'), 42) or 42
    seed = _parse_optional_int("FEDCHEM_SEED", seed_default) or seed_default
    import random as _random
    _random.seed(int(seed))
    np.random.seed(int(seed))

    force_tecator = _resolve_bool_from_env("FEDCHEM_USE_TECATOR", cfg.get('USE_TECATOR'), False)

    n_wavelengths_default = _coerce_int(cfg.get('DEFAULT_N_WAVELENGTHS'), DEFAULT_N_WAVELENGTHS) or DEFAULT_N_WAVELENGTHS
    n_wavelengths_requested = _parse_optional_int("FEDCHEM_N_WAVELENGTHS", n_wavelengths_default)
    max_transfer_default = _coerce_int(cfg.get('MAX_TRANSFER_SAMPLES'))
    max_transfer_requested = _parse_optional_int("FEDCHEM_MAX_TRANSFER_SAMPLES", max_transfer_default)
    # Derive config-based site codes and per-manufacturer instrument config
    exp_design = cfg.get("EXPERIMENTAL_DESIGN", {}) or {}
    factors = exp_design.get("FACTORS", {}) if isinstance(exp_design.get("FACTORS"), dict) else {}
    config_sites = get_experimental_sites(cfg)
    data_config = get_data_config(cfg)

    if clients is None or X_test is None or y_test is None or ds_meta is None:
        clients, X_test, y_test, ds_meta = _make_clients(
            n_sites=n_sites,
            seed=seed,
            quick=quick,
            force_tecator=force_tecator,
            n_wavelengths=n_wavelengths_requested,
            max_transfer_samples=max_transfer_requested,
            config_sites=config_sites,
        )
    ds_meta = ds_meta or {}

    def _eval_fn(model: Any) -> dict:
        yhat = model.predict(X_test)
        return {"rmsep": rmsep(y_test, yhat)}

    dp_config: dict[str, Any] = {"delta": delta, "target_epsilon": target_eps}
    dp_config["participation_schedule"] = [float(x) for x in part_sched]
    if part_rate is not None:
        dp_config["participation_rate"] = float(part_rate)
    dp_config["compression_schedule"] = [float(x) for x in comp_sched]
    # Map DIFF PRIV mapping from config if present (noise_multiplier_map), if explicit noise_std not set
    dp_section = config.get('DIFFERENTIAL_PRIVACY') if isinstance(config.get('DIFFERENTIAL_PRIVACY'), dict) else None
    if dp_section is not None and dp_config.get('noise_std') is None:
        noise_map = dp_section.get('noise_multiplier_map') or {}
        key_candidate = str(target_eps) if target_eps is not None else str(config.get('DP_TARGET_EPS'))
        key_candidate = key_candidate.replace(' ', '')
        if key_candidate in noise_map:
            # Safely extract the value first to avoid passing None to float()
            val = noise_map.get(key_candidate)
            if val is None:
                dp_config['noise_multiplier'] = None
            else:
                try:
                    dp_config['noise_multiplier'] = float(val)
                except Exception:
                    # Keep original value if it cannot be converted to float
                    dp_config['noise_multiplier'] = val

    # Secure aggregator (if requested via config or env)
    use_secure = _resolve_bool_from_env("FEDCHEM_USE_SECURE_AGGREGATION", config.get('SECURE_AGGREGATION', {}).get('enabled') if isinstance(config.get('SECURE_AGGREGATION'), dict) else config.get('SECURE_AGGREGATION', False), False)
    secure_aggregator = None
    if use_secure:
        try:
            secure_aggregator = SimulatedSecureAggregator(rng_seed=int(seed or 0))
        except Exception:
            secure_aggregator = None

    orch = FederatedOrchestrator()
    res = orch.run_rounds(
        clients=clients,
        model=_instantiate_pipeline_model(),
        rounds=rounds,
        algo="fedavg",
        dp_config=dp_config,
        clip_norm=clip_norm,
        server_eta=server_eta,
        seed=seed,
        eval_fn=_eval_fn,
        secure_aggregator=secure_aggregator,
    )
    logs = res.get("logs", [])
    # Baseline (non-DP pooled) for reference line in panel (a)
    base_model = PLSModel(n_components=None, max_components=15, cv=5, random_state=0).fit(
        np.vstack([c["X"] for c in clients]), np.hstack([c["y"] for c in clients])
    )
    baseline_rmsep = float(rmsep(y_test, base_model.predict(X_test)))
    # Figure + Table + Caption/Meta using the same logs
    _plot_figure_6_from_logs(logs, baseline_rmsep)
    generate_table_6(logs, used_model=(res.get("used_model") if isinstance(res, dict) else None))
    total_time_sec = float(time.perf_counter() - start_time)
    pipeline_cfg = config.get("PIPELINE") if isinstance(config.get("PIPELINE"), dict) else {}
    # Build a safe dict with str keys and non-bytes values to avoid __setitem__ type issues
    resolved_pipeline = dict(pipeline_cfg) if isinstance(pipeline_cfg, dict) else {}
    resolved_pipeline = {
        str(k): (v.decode() if isinstance(v, (bytes, bytearray)) else v)
        for k, v in resolved_pipeline.items()
    }
    resolved_pipeline["model"] = _resolve_model_label()
    fedpls_enabled = _resolve_fedpls_enabled()
    run_config = {
        "dataset": ds_meta.get("dataset", "real"),
        "dataset_meta": ds_meta,
        "force_tecator": force_tecator,
        "n_sites": n_sites,
        "rounds": rounds,
        "target_epsilon": target_eps,
        "delta": delta,
        "clip_norm": clip_norm,
        "server_eta": server_eta,
        "participation_rate": part_rate,
        "participation_schedule": [float(x) for x in part_sched] if part_sched else None,
        "compression_schedule": [float(x) for x in comp_sched] if comp_sched else None,
        "quick_mode": bool(quick),
        "secure_aggregation": False,
        "runtime": {"wall_time_total_sec": total_time_sec},
        "standard_design": True,
        "design_version": resolve_design_version(config),
        "seed": seed,
        "n_wavelengths_requested": n_wavelengths_requested,
        "n_wavelengths_actual": ds_meta.get("n_wavelengths_actual"),
        "transfer_samples_requested": ds_meta.get("transfer_samples_requested", max_transfer_requested),
        "transfer_samples_used": ds_meta.get("transfer_samples_used"),
        "max_transfer_samples": max_transfer_requested,
        "test_samples_per_site": 30,
        "pipeline": resolved_pipeline,
        "fedpls_enabled": fedpls_enabled,
        "fedpls_method": _resolve_fedpls_method() if fedpls_enabled else None,
    }
    run_config['DIFFERENTIAL_PRIVACY'] = config.get('DIFFERENTIAL_PRIVACY')
    run_config['CONFORMAL'] = config.get('CONFORMAL')
    run_config['SECURE_AGGREGATION'] = config.get('SECURE_AGGREGATION')
    run_config['DRIFT_AUGMENT'] = config.get('DRIFT_AUGMENT')
    run_config['REPRODUCIBILITY'] = config.get('REPRODUCIBILITY')
    run_config['secure_aggregation'] = bool(use_secure)
    _write_caption_meta(logs, run_config, used_models={"FedAvg": (res.get("used_model") if isinstance(res, dict) else None)})
    print("Objective 6 completed.")

if __name__ == "__main__":
    main()
