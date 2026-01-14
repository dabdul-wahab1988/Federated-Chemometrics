from __future__ import annotations

import time
import logging
import json
from pathlib import Path
import math
from dataclasses import dataclass
import os
from typing import Any, Callable, Dict, List, Optional
import numpy as np

from .types import ClientBatch, ParametricModel
from .algorithms import FedAvgAlgorithm, FedProxAlgorithm
from ..privacy.dp_accountant import DPAccountant
from .secure_aggregation import SecureAggregator


# Small adapter/wrappers for DPAccountant internals so calling code
# doesn't scatter private method usage everywhere. If the accountant
# implementation changes, update these wrappers accordingly.
def _acc_orders(acc: DPAccountant):
    try:
        return list(acc._orders())
    except Exception:
        # Fallback reasonable sweep
        return [1.25, 1.5, 2.0, 2.5, 3.0, 3.5, 5.0, 8.0, 10.0, 16.0, 32.0, 64.0, 128.0, 256.0]


def _acc_rdp_gaussian(acc: DPAccountant, alpha: float, sigma: float, sensitivity: float = 1.0) -> float:
    try:
        return acc._rdp_gaussian(alpha, sigma, sensitivity=sensitivity)
    except Exception:
        # Use analytic fallback
        if alpha <= 1:
            raise
        return alpha * (sensitivity ** 2) / (2.0 * (sigma ** 2))


def _acc_rdp_gaussian_subsampled(acc: DPAccountant, alpha: float, q: float, sigma: float) -> float:
    try:
        return acc._rdp_gaussian_subsampled(alpha, q, sigma)
    except Exception:
        # As a last resort fall back to the gaussian approximation at this alpha
        # after scaling sigma by sensitivity (not exact for subsampling).
        return _acc_rdp_gaussian(acc, alpha, sigma / max(1e-20, 1.0), sensitivity=1.0)


@dataclass
class RoundLog:
    round: int
    bytes_sent: int
    bytes_recv: int
    duration_sec: float
    seed: Optional[int] = None
    # Optional evaluation metrics (filled if eval_fn provided)
    rmsep: Optional[float] = None
    cvrmsep: Optional[float] = None
    r2: Optional[float] = None
    mae: Optional[float] = None
    # Optional privacy/participation telemetry
    participation_rate: Optional[float] = None
    epsilon_so_far: Optional[float] = None
    # Optional instrumentation for debugging DP/aggregation behaviour
    update_norm: Optional[float] = None
    weight_norm: Optional[float] = None
    participants: Optional[int] = None
    dp_noise_std: Optional[float] = None
    clip_norm_used: Optional[float] = None
    compression_ratio: Optional[float] = None
    eval_error: Optional[str] = None
    server_eta: Optional[float] = None
    # New DP diagnostics
    clip_fraction: Optional[float] = None
    mean_update_norm_raw: Optional[float] = None
    mean_update_norm_clipped: Optional[float] = None
    sensitivity_round: Optional[float] = None
    # PDS communication bytes (total bytes transferred due to PDS mapping during this round)
    pds_bytes: Optional[int] = None
    # Aggregation diagnostics
    agg_method: Optional[str] = None
    agg_trim_frac: Optional[float] = None
    # Which model was used for the round (e.g., 'FedPLS', 'pooled_parametric_pls', 'fedavg')
    used_model: Optional[str] = None


class FederatedOrchestrator:
    """Federated loop with pluggable aggregation algorithms and DP/clipping."""

    def run_rounds(
        self,
        clients: List[Dict[str, Any]],
        model: ParametricModel,
        rounds: int,
        algo: str = "fedavg",
        prox_mu: float = 0.0,
        dp_config: Optional[dict] = None,
        clip_norm: Optional[float] = None,
        seed: Optional[int] = None,
        eval_fn: Optional[Callable[[ParametricModel], Dict[str, float]]] = None,
        server_eta: float = 1.0,
        secure_aggregator: Optional[SecureAggregator] = None,
    ) -> Dict[str, Any]:
        if rounds <= 0:
            raise ValueError("rounds must be > 0")
        # Fast path for Federated PLS MVP (k=1, one-shot sufficient statistics)
        if algo.lower() == "fedpls":
            return self.run_fedpls(clients, model=model, rounds=rounds, dp_config=dp_config, clip_norm=clip_norm, seed=seed, eval_fn=eval_fn)  # type: ignore[attr-defined]

        rng = np.random.default_rng(seed)
        logs: List[RoundLog] = []
        global_model = model
        has_param_api = hasattr(global_model, "get_params") and hasattr(global_model, "set_params")
        param_keys: Optional[tuple[str, ...]] = None

        # choose algorithm (provide RNG for reproducible noise in algorithms)
        algo_impl = FedAvgAlgorithm(rng=rng) if algo.lower() == "fedavg" else FedProxAlgorithm(rng=rng)

        # DP/accounting setup
        dp_noise_std: Optional[float] = None
        epsilon: Optional[float] = None
        delta: Optional[float] = None
        log_inv_delta: Optional[float] = None
        configured_participation_rate: Optional[float] = None
        participation_schedule: Optional[tuple[float, ...]] = None
        compression_schedule: Optional[tuple[float, ...]] = None
        sensitivity: float = 1.0
        acc = DPAccountant()
        orders_all: tuple[float, ...] = tuple(_acc_orders(acc))
        # Algorithm aggregation diagnostics (always initialize so logs can reference them)
        agg_method: Optional[str] = None
        agg_trim_frac: Optional[float] = None
        # Track which model was used for this run (defaults to algorithm name)
        used_model: Optional[str] = algo if algo is not None else None

        def _normalize_schedule(values: Any, *, clip01: bool = False, floor: float | None = None) -> Optional[tuple[float, ...]]:
            """Normalize a schedule value into a tuple of floats.

            Accepts:
            - None -> None
            - list/tuple of numbers/strings
            - comma/whitespace-separated string like "1.0,0.8,0.6"
            Returns None if no valid floats parsed.
            """
            if values is None:
                return None
            # If it's a string, split on common separators
            if isinstance(values, str):
                raw = [s.strip() for s in values.replace(';', ',').split(',') if s.strip()]
            elif isinstance(values, (list, tuple)):
                raw = list(values)
            else:
                # Unknown type: try to coerce to string and split
                try:
                    raw = [s.strip() for s in str(values).replace(';', ',').split(',') if s.strip()]
                except Exception:
                    return None
            seq: List[float] = []
            for v in raw:
                try:
                    f = float(v)
                except Exception:
                    # skip invalid tokens
                    continue
                if clip01:
                    f = min(max(f, 0.0), 1.0)
                elif floor is not None and f < floor:
                    f = floor
                seq.append(f)
            return tuple(seq) if seq else None

        if dp_config:
            delta_val = _safe_float(dp_config.get("delta"))
            delta = delta_val if delta_val is not None else 1e-5
            log_inv_delta = math.log(1.0 / max(1e-20, float(delta)))
            dp_noise_std = _safe_float(dp_config.get("noise_std"))
            # Accept a 'noise_multiplier' config (sigma multiplier) in dp_config as a convenience
            # If provided, derive dp_noise_std as multiplier * sensitivity
            noise_mult = _safe_float(dp_config.get("noise_multiplier"))
            if noise_mult is not None:
                try:
                    dp_noise_std = float(noise_mult) * float(sensitivity)
                except Exception:
                    dp_noise_std = float(noise_mult)
            target_eps = _safe_float(dp_config.get("target_epsilon"))
            # If the target epsilon is infinite (user requested 'no DP'), do not attempt
            # to compute a finite `dp_noise_std` via the solver — treat DP as disabled.
            if target_eps is not None and math.isinf(float(target_eps)):
                # silence DP for this run; downstream code respects dp_noise_std None
                dp_noise_std = None
                target_eps = None
            configured_participation_rate = _safe_float(dp_config.get("participation_rate"))
            participation_schedule = _normalize_schedule(dp_config.get("participation_schedule"), clip01=True)
            compression_schedule = _normalize_schedule(dp_config.get("compression_schedule"), floor=0.0)
            # Effective sensitivity: average under secure aggregation over N clients when clipping is used
            if clip_norm is not None and len(clients) > 0:
                N = float(len(clients))
                sensitivity = float(clip_norm) / N
            else:
                sensitivity = 1.0
            # Solve sigma when target epsilon is given
            if dp_noise_std is None and target_eps is not None:
                # If a schedule is provided, solve using variable composition across the schedule
                if participation_schedule is not None and len(participation_schedule) > 0:
                    sched = list(participation_schedule[:rounds])
                    orders = orders_all

                    # helper to compute final epsilon for a candidate sigma under schedule
                    def _final_eps_for_sigma(sig: float) -> float:
                        rdp_cum: dict[float, float] = {a: 0.0 for a in orders}
                        Nloc = max(1, int(len(clients)))
                        for r in range(rounds):
                            q_r = float(sched[r]) if r < len(sched) else float(sched[-1])
                            # expected participants for solving; at runtime, actual may differ
                            m_exp = max(1, int(round(q_r * Nloc)))
                            if clip_norm is not None:
                                sens_r = float(clip_norm) / float(m_exp)
                            else:
                                sens_r = 1.0
                            if 0.0 < q_r < 1.0:
                                # subsampled Gaussian: scale sigma by sensitivity internally
                                sigma_eff = float(sig) / max(1e-20, sens_r)
                                for a in orders:
                                    if a < 2.0:
                                        continue
                                    rdp_cum[a] += _acc_rdp_gaussian_subsampled(acc, a, q_r, sigma_eff)
                            else:
                                # full participation Gaussian
                                for a in orders:
                                    if a <= 1.0:
                                        continue
                                    rdp_cum[a] += _acc_rdp_gaussian(acc, a, float(sig), sensitivity=sens_r)

                        # convert to (eps,delta)
                        best = math.inf
                        for a in orders:
                            if a <= 1.0:
                                continue
                            eps_a = rdp_cum[a] + (log_inv_delta if log_inv_delta is not None else math.log(1.0 / max(1e-20, float(delta)))) / (a - 1.0)
                            if eps_a < best:
                                best = eps_a
                        return float(best)

                    # binary search sigma
                    lo, hi = 1e-3, 1e3
                    for _ in range(60):
                        mid = math.sqrt(lo * hi)
                        eps_mid = _final_eps_for_sigma(mid)
                        if eps_mid > float(target_eps):
                            lo = mid
                        else:
                            hi = mid
                        if abs(hi - lo) / max(1.0, lo) < 1e-4:
                            break
                    dp_noise_std = float(hi)
                else:
                    # fall back to existing solver for constant q (or none)
                    sigma = acc.solve_sigma_for_target_epsilon(
                        float(target_eps), float(delta), rounds, sensitivity=sensitivity,
                        q=float(configured_participation_rate) if configured_participation_rate else None
                    )
                    dp_noise_std = float(sigma)
            # Compute final epsilon for reporting
            if dp_noise_std is not None:
                if configured_participation_rate is not None and 0 < float(configured_participation_rate) < 1:
                    q = float(configured_participation_rate)
                    res = acc.compute(
                        "gaussian_subsampled",
                        {"sigma": float(dp_noise_std), "delta": float(delta), "q": q},
                        rounds,
                        sensitivity=sensitivity,
                    )
                    eff_rounds = max(1, int(round(q * rounds)))
                    res_eff = acc.compute(
                        "gaussian",
                        {"sigma": float(dp_noise_std), "delta": float(delta)},
                        eff_rounds,
                        sensitivity=sensitivity,
                    )
                    epsilon = min(res["epsilon"], res_eff["epsilon"])
                else:
                    res = acc.compute(
                        "gaussian",
                        {"sigma": float(dp_noise_std), "delta": float(delta)},
                        rounds,
                        sensitivity=sensitivity,
                    )
                    epsilon = res["epsilon"]

        # Estimate per-round communication payload size: parameter vector downlink + uplink per participating client
        # Try to infer parameter size from model params; if empty, infer from first client's feature dimension
        param_elems = 0
        params_snapshot: Any = {}
        if has_param_api:
            try:
                params_snapshot = global_model.get_params()
            except Exception:
                params_snapshot = {}
            else:
                try:
                    param_elems = int(sum(np.asarray(v).size for v in params_snapshot.values()))
                except Exception:
                    param_elems = 0
                if isinstance(params_snapshot, dict) and np.asarray(params_snapshot.get("w", np.zeros(0, dtype=float))).size == 0 and len(clients) > 0:
                    d = int(clients[0]["X"].shape[1])
                    try:
                        global_model.set_params({"w": np.zeros(d, dtype=float), "b": np.array([0.0], dtype=float)})
                        params_snapshot = global_model.get_params()
                        param_elems = int(sum(np.asarray(v).size for v in params_snapshot.values()))
                    except Exception:
                        param_elems = 0
            if hasattr(params_snapshot, "keys"):
                param_keys = tuple(sorted(params_snapshot.keys()))
        if param_elems == 0 and len(clients) > 0:
            d = int(clients[0]["X"].shape[1])
            param_elems = d + 1  # weights + bias
            # Ensure the global model carries a correctly shaped parameter vector
            # so that update-based algorithms can compute deltas consistently.
            if has_param_api:
                try:
                    global_model.set_params({"w": np.zeros(d, dtype=float), "b": np.array([0.0], dtype=float)})
                    params_snapshot = global_model.get_params()
                    if hasattr(params_snapshot, "keys"):
                        param_keys = tuple(sorted(params_snapshot.keys()))
                except Exception:
                    pass
        # assume float64 for numpy arrays
        BYTES_PER_ELEM = 8

        # Helper to produce a deterministic vectorization of model params for
        # computing norms (match algorithms' sorted-key behaviour)
        empty_vec = np.zeros(0, dtype=float)

        def _vec_params(params: Dict[str, Any], keys: Optional[tuple[str, ...]] = None) -> np.ndarray:
            if not hasattr(params, "items"):
                return empty_vec
            key_iter = keys if keys is not None else tuple(sorted(params.keys()))
            vecs: List[np.ndarray] = []
            for k in key_iter:
                v = np.asarray(params[k]).ravel()
                vecs.append(v)
            try:
                return np.concatenate(vecs) if vecs else empty_vec
            except Exception as e:
                raise RuntimeError("Inconsistent parameter shapes when vectorizing params in orchestrator") from e

        def _devectorize(vec: np.ndarray, template_params: Dict[str, Any], keys: Optional[tuple[str, ...]] = None) -> Dict[str, Any]:
            """Reconstruct a params dict from a flat vector `vec` using shapes in `template_params`.
            Keys order follows `keys` if provided, otherwise sorted(template_params.keys()).
            """
            if not hasattr(template_params, "items"):
                return {}
            key_iter = keys if keys is not None else tuple(sorted(template_params.keys()))
            out: Dict[str, Any] = {}
            idx = 0
            total = int(vec.size) if isinstance(vec, np.ndarray) else 0
            for k in key_iter:
                if k not in template_params:
                    continue
                tmpl = np.asarray(template_params[k])
                size = int(tmpl.size)
                # Safely handle zero-sized templates
                if size == 0:
                    out[k] = np.zeros_like(tmpl)
                    continue
                if idx + size > total:
                    raise RuntimeError("Vector length does not match template parameter sizes when devectorizing")
                seg = vec[idx : idx + size]
                try:
                    out[k] = seg.reshape(tmpl.shape)
                except Exception:
                    # Fallback: return flat 1-D array for this key if reshape fails
                    out[k] = seg.copy()
                idx += size
            return out

        per_round_q: List[float] = []
        # RDP cumulative tracker for epsilon progression (variable schedule aware)
        rdp_cum: Dict[float, float] = {a: 0.0 for a in orders_all}
        for r in range(1, rounds + 1):
            t0 = time.perf_counter()
            sent = 0
            recv = 0
            batches: List[ClientBatch] = []
            # Choose participating clients for this round
            if dp_config and (configured_participation_rate is not None or participation_schedule is not None):
                N = len(clients)
                if participation_schedule is not None and 1 <= r <= len(participation_schedule):
                    q_used = float(participation_schedule[r - 1])
                else:
                    q_used = float(configured_participation_rate) if configured_participation_rate is not None else 1.0
                m = max(1, int(rng.binomial(N, min(max(q_used, 0.0), 1.0))))
                idx = np.arange(N)
                chosen = rng.choice(idx, size=m, replace=False)
                chosen_clients = [clients[i] for i in chosen]
            else:
                chosen_clients = clients
            actual_participation = len(chosen_clients) / max(1, len(clients))
            per_round_q.append(float(actual_participation))
            # Build batches
            for c in chosen_clients:
                Xc, yc = c["X"], c["y"]
                batches.append(ClientBatch(Xc, yc))
            # Communication accounting (apply optional compression)
            comp = 1.0
            if compression_schedule is not None and 0 <= r - 1 < len(compression_schedule):
                comp = float(compression_schedule[r - 1])
            # Explicit accounting: downlink (server -> clients) and uplink (clients -> server)
            downlink_bytes = int(param_elems * BYTES_PER_ELEM * len(chosen_clients) * comp)
            uplink_bytes = int(param_elems * BYTES_PER_ELEM * len(chosen_clients) * comp)
            sent += downlink_bytes
            recv += uplink_bytes
            # Aggregate or refit
            if has_param_api:
                prev_params = global_model.get_params()
                if param_keys is None and hasattr(prev_params, "keys"):
                    param_keys = tuple(sorted(prev_params.keys()))
                prev_vec = _vec_params(prev_params, param_keys)
                # If a secure aggregator is provided, compute per-client updates
                # explicitly and use the aggregator for privacy-preserving masking
                if secure_aggregator is not None:
                    # vectorize global params
                    prev_vec = _vec_params(prev_params, param_keys)
                    updates = []
                    update_keys = param_keys
                    for i, b in enumerate(batches):
                        try:
                            m = type(global_model)()
                            try:
                                if hasattr(global_model, 'set_use_global_scaler') and getattr(global_model, 'use_global_scaler', False):
                                    if getattr(global_model, 'global_scaler_mean', None) is not None and getattr(global_model, 'global_scaler_scale', None) is not None:
                                        set_use = getattr(m, "set_use_global_scaler", None)
                                        set_scaler = getattr(m, "set_global_scaler", None)
                                        if callable(set_use):
                                            try:
                                                set_use(True)
                                            except Exception:
                                                # avoid failing the client creation for non-conforming implementations
                                                pass
                                        if callable(set_scaler):
                                            try:
                                                set_scaler(getattr(global_model, 'global_scaler_mean'), getattr(global_model, 'global_scaler_scale'))
                                            except Exception:
                                                pass
                            except Exception:
                                pass
                        except Exception:
                            raise RuntimeError("Failed to create per-client model for secure aggregation")
                        m.fit(b.X, b.y)
                        # Debug dump of per-client fitted params if enabled
                        try:
                            if os.environ.get("FEDCHEM_DEBUG_DUMP_WEIGHTS") == "1":
                                out_dir = Path(os.environ.get("FEDCHEM_OUTPUT_DIR", "generated_figures_tables")) / "lca_artifacts" / "debug_weights"
                                out_dir.mkdir(parents=True, exist_ok=True)
                                params = m.get_params()
                                scaler = None
                                n_samples = int(np.asarray(b.X).shape[0]) if b is not None and hasattr(b, 'X') else None
                                try:
                                    pipe = getattr(m, 'pipeline', None)
                                    if pipe is not None and hasattr(pipe, 'named_steps') and 'scaler' in pipe.named_steps:
                                        sc = pipe.named_steps['scaler']
                                        # Use temporaries so static analyzers know we checked for None before calling .tolist()
                                        mean_attr = getattr(sc, 'mean_', None)
                                        scale_attr = getattr(sc, 'scale_', None)
                                        mean_val = mean_attr.tolist() if mean_attr is not None else None
                                        scale_val = scale_attr.tolist() if scale_attr is not None else None
                                        scaler = {'mean': mean_val, 'scale': scale_val}
                                        if 'pls' in pipe.named_steps and hasattr(pipe.named_steps['pls'], 'n_components'):
                                            scaler['n_components'] = int(getattr(pipe.named_steps['pls'], 'n_components'))
                                except Exception:
                                    scaler = None
                                dump = {
                                    'timestamp': time.time(),
                                    'algo': 'secure_aggregator',
                                    'round': r,
                                    'client_index': i,
                                    'params': {k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in params.items()},
                                    'n_samples': n_samples,
                                    'scaler': scaler,
                                }
                                fname = out_dir / f"debug_weight_secure_client_{i}_round_{r}_{int(time.time()*1000)}.json"
                                with open(fname, 'w', encoding='utf-8') as fh:
                                    json.dump(dump, fh)
                        except Exception:
                            pass
                        v_i = _vec_params(m.get_params(), update_keys)
                        u_i = v_i - prev_vec
                        masked_u = secure_aggregator.prepare_client_masked_update(str(i), u_i)
                        updates.append(masked_u)
                    total_masked = secure_aggregator.aggregate_masked_updates(updates)
                    total_unmasked = secure_aggregator.unmask_aggregate(total_masked)
                    avg_update = total_unmasked / max(1, len(updates))
                    if dp_noise_std is not None and dp_noise_std > 0:
                        avg_update = avg_update + rng.normal(0.0, dp_noise_std, size=avg_update.shape)
                    new_vec = prev_vec + avg_update
                    new_params_raw = {}
                    try:
                        new_params_raw = _devectorize(new_vec, prev_params, param_keys)
                    except Exception:
                        new_params_raw = global_model.get_params()
                else:
                    new_params_raw = algo_impl.aggregate(
                        global_model, batches, clip_norm=clip_norm, dp_noise_std=dp_noise_std, prox_mu=prox_mu
                    )
                iter_keys = param_keys if param_keys is not None else tuple(sorted(new_params_raw.keys()))
                # Server damping
                damped_params: Dict[str, Any] = {}
                for k in iter_keys:
                    if k not in prev_params or k not in new_params_raw:
                        continue
                    pv = np.asarray(prev_params[k])
                    nv = np.asarray(new_params_raw[k])
                    damped_params[k] = pv + float(server_eta) * (nv - pv)
                if param_keys is not None:
                    for k in param_keys:
                        if k not in damped_params and k in prev_params:
                            damped_params[k] = np.asarray(prev_params[k])
                global_model.set_params(damped_params)
                new_vec = _vec_params(damped_params, param_keys)
                upd_norm = float(np.linalg.norm(new_vec - prev_vec)) if new_vec.size else None
                w_norm = float(np.linalg.norm(new_vec)) if new_vec.size else None
            else:
                X_all = np.vstack([b.X for b in batches])
                y_all = np.hstack([b.y for b in batches])
                global_model.fit(X_all, y_all)
                # Unknown paramization; skip norm telemetry
                upd_norm = None
                w_norm = None
            dt = time.perf_counter() - t0
            # Evaluate if requested
            m_rmsep: Optional[float] = None
            m_cvrmsep: Optional[float] = None
            m_r2: Optional[float] = None
            m_mae: Optional[float] = None
            eval_err: Optional[str] = None
            if eval_fn is not None:
                try:
                    metrics = eval_fn(global_model)
                    val_rmsep = metrics.get("rmsep")
                    val_cvrmsep = metrics.get("cvrmsep")
                    val_r2 = metrics.get("r2")
                    val_mae = metrics.get("mae")
                    m_rmsep = float(val_rmsep) if val_rmsep is not None else None
                    m_cvrmsep = float(val_cvrmsep) if val_cvrmsep is not None else None
                    m_r2 = float(val_r2) if val_r2 is not None else None
                    m_mae = float(val_mae) if val_mae is not None else None
                except Exception as e:
                    eval_err = str(e)[:200]
            # Epsilon progression using cumulative RDP with actual participation and round-specific sensitivity
            eps_round: Optional[float] = None
            sens_r_logged: Optional[float] = None
            if dp_config and dp_noise_std is not None and delta is not None:
                q_eff = float(per_round_q[-1]) if per_round_q else (
                    float(configured_participation_rate) if configured_participation_rate is not None else 1.0
                )
                # round-specific sensitivity: C/m when clipping; else 1.0
                if clip_norm is not None and len(chosen_clients) > 0:
                    sens_r = float(clip_norm) / float(len(chosen_clients))
                else:
                    sens_r = 1.0
                sens_r_logged = float(sens_r)
                if 0.0 < q_eff < 1.0:
                    sigma_eff = float(dp_noise_std) / max(1e-20, sens_r)
                    for a in orders_all:
                        if a < 2.0:
                            continue
                        rdp_cum[a] += _acc_rdp_gaussian_subsampled(acc, a, q_eff, sigma_eff)
                else:
                    for a in orders_all:
                        if a <= 1.0:
                            continue
                        rdp_cum[a] += _acc_rdp_gaussian(acc, a, float(dp_noise_std), sensitivity=sens_r)
                # Convert to epsilon for the prefix of rounds
                best = math.inf
                for a in orders_all:
                    if a <= 1.0:
                        continue
                    eps_a = rdp_cum[a] + (log_inv_delta if log_inv_delta is not None else math.log(1.0 / max(1e-20, float(delta)))) / (a - 1.0)
                    if eps_a < best:
                        best = eps_a
                eps_round = float(best)
            # Extract algorithm-side diagnostics if available
            clip_fraction = None
            mean_update_norm_raw = None
            mean_update_norm_clipped = None
            if hasattr(algo_impl, "last_stats") and isinstance(algo_impl.last_stats, dict):
                clip_fraction = algo_impl.last_stats.get("clip_fraction")
                mean_update_norm_raw = algo_impl.last_stats.get("mean_update_norm_raw")
                mean_update_norm_clipped = algo_impl.last_stats.get("mean_update_norm_clipped")
                agg_method = algo_impl.last_stats.get("agg_method")
                agg_trim_frac = algo_impl.last_stats.get("agg_trim_frac")
            # compute PDS bytes contributed by chosen clients if present in client dicts
            pds_bytes_round = 0
            try:
                pds_bytes_round = int(sum(int(c.get("communication_bytes", 0)) for c in chosen_clients))
            except Exception:
                pds_bytes_round = 0

            logs.append(
                RoundLog(
                    round=r,
                    bytes_sent=sent,
                    bytes_recv=recv,
                    duration_sec=dt,
                    seed=int(rng.integers(0, 2**31)),
                    rmsep=m_rmsep,
                    cvrmsep=m_cvrmsep,
                    r2=m_r2,
                    mae=m_mae,
                    participation_rate=float(actual_participation),
                    epsilon_so_far=eps_round,
                    update_norm=upd_norm,
                    weight_norm=w_norm,
                    participants=int(len(chosen_clients)),
                    dp_noise_std=float(dp_noise_std) if dp_noise_std is not None else None,
                    clip_norm_used=float(clip_norm) if clip_norm is not None else None,
                    compression_ratio=float(comp),
                    eval_error=eval_err,
                    server_eta=float(server_eta),
                    clip_fraction=_safe_float(clip_fraction),
                    mean_update_norm_raw=_safe_float(mean_update_norm_raw),
                    mean_update_norm_clipped=_safe_float(mean_update_norm_clipped),
                    sensitivity_round=_safe_float(sens_r_logged),
                    pds_bytes=int(pds_bytes_round),
                    agg_method=str(agg_method) if agg_method is not None else None,
                    agg_trim_frac=_safe_float(agg_trim_frac),
                    used_model=used_model,
                )
            )
        out = {"model": global_model, "logs": [log.__dict__ for log in logs]}
        # Final epsilon from cumulative RDP (variable actual participation) when DP is enabled
        if dp_config and dp_noise_std is not None and delta is not None:
            if len([v for v in rdp_cum.values() if v > 0]) > 0:
                best = math.inf
                for a in orders_all:
                    if a <= 1.0:
                        continue
                    eps_a = rdp_cum[a] + (log_inv_delta if log_inv_delta is not None else math.log(1.0 / max(1e-20, float(delta)))) / (a - 1.0)
                    if eps_a < best:
                        best = eps_a
                epsilon = float(best)
            # If fixed participation rate was configured (no schedule), also compute effective-rounds Gaussian and take min (tighter bound)
            if participation_schedule is None and configured_participation_rate is not None:
                # Fixed participation: apply min bound logic (subsampled vs effective rounds) using *configured* rate.
                q = float(configured_participation_rate)
                res_sub = acc.compute(
                    "gaussian_subsampled",
                    {"sigma": float(dp_noise_std), "delta": float(delta), "q": q},
                    rounds,
                    sensitivity=sensitivity,
                )["epsilon"]
                eff_rounds = max(1, int(round(q * rounds)))
                res_eff = acc.compute(
                    "gaussian",
                    {"sigma": float(dp_noise_std), "delta": float(delta)},
                    eff_rounds,
                    sensitivity=sensitivity,
                )["epsilon"]
                tight = float(min(res_sub, res_eff))
                epsilon = tight
        if dp_config:
            out["dp"] = {
                "noise_std": float(dp_noise_std) if dp_noise_std is not None else None,
                "epsilon": float(epsilon) if epsilon is not None else None,
                "delta": float(delta) if delta is not None else None,
                "participation_rate": float(configured_participation_rate) if configured_participation_rate is not None else None,
                "sensitivity": float(sensitivity),
            }
        # Mark which model was used for final model output; default to the algorithm name
        out.setdefault("used_model", algo if algo is not None else None)
        return out

    @staticmethod
    def create_loso_client_sets(
        data: dict[str, dict[str, Any]],
        *,
        include_meta: bool = False,
    ) -> list[Any]:
        """Return leave-one-site-out client plans for a site/instrument data dict.

        Args:
            data: mapping of site/instrument id -> site dict containing at least
                'X' and 'y'. When include_meta=True, optional keys such as
                'test', 'instrument_id', and 'backing_site' are propagated.
            include_meta: when True, return dictionaries with held-out metadata.
                When False (default), return tuples (held_out_id, clients_list)
                for backward compatibility.
        """
        if not isinstance(data, dict):
            return []
        keys = list(data.keys())
        sets: list[Any] = []
        for left in keys:
            clients: list[dict[str, Any]] = []
            for k in keys:
                if k == left:
                    continue
                site = data[k]
                if not isinstance(site, dict):
                    continue
                X = site.get("X")
                y = site.get("y")
                if X is None or y is None:
                    continue
                clients.append({"X": X, "y": y})
            if include_meta:
                held_out_site = data.get(left, {}) or {}
                holdout_meta = {
                    "instrument_id": held_out_site.get("instrument_id", left),
                    "backing_site": held_out_site.get("backing_site") or held_out_site.get("site"),
                    "test": held_out_site.get("test"),
                    "val": held_out_site.get("val"),
                    "cal": held_out_site.get("cal"),
                }
                sets.append(
                    {
                        "held_out_id": left,
                        "clients": clients,
                        "held_out_site": holdout_meta,
                    }
                )
            else:
                sets.append((left, clients))
        return sets

    def run_fedpls(
        self,
        clients: List[Dict[str, Any]],
        model: Optional[Any] = None,
        rounds: int = 2,
        dp_config: Optional[dict] = None,
        clip_norm: Optional[float] = None,
        seed: Optional[int] = None,
        eval_fn: Optional[Callable[[ParametricModel], Dict[str, float]]] = None,
    ) -> Dict[str, Any]:
        """Federated PLS helper.

        This upgraded implementation performs a pooled PLS fit using client-supplied
        data (same data the old MVP aggregated) but fits a multi-component PLS model
        (using the existing PLSModel wrapper). If the caller passes a `model` and
        it exposes PLS-like attributes (`n_components`, `max_components`, `cv`),
        those will be respected; otherwise sensible defaults are used.

        Phases (kept for compatibility/telemetry):
        1) Global standardization statistics are aggregated for accounting.
        2) Pooled PLS fit is performed on stacked client data (can use CV).

        Returns a LinearModel with (w,b), plus two RoundLog entries for the phases.
        """
        if not clients:
            raise ValueError("No clients provided")
        BYTES_PER_ELEM = 8
        d = int(clients[0]["X"].shape[1])
        used_model = "FedPLS"
        # Phase 1: global standardization
        n_total = 0
        sum_x = np.zeros(d, dtype=float)
        sum_x2 = np.zeros(d, dtype=float)
        sum_y = 0.0
        sum_y2 = 0.0
        for c in clients:
            Xc = np.asarray(c["X"])  # (n_i, d)
            yc = np.asarray(c["y"])  # (n_i,)
            n_i = int(Xc.shape[0])
            n_total += n_i
            sum_x += Xc.sum(axis=0)
            sum_x2 += (Xc ** 2).sum(axis=0)
            sum_y += float(yc.sum())
            sum_y2 += float((yc ** 2).sum())
        mu_x = sum_x / max(1, n_total)
        var_x = sum_x2 / max(1, n_total) - mu_x ** 2
        sigma_x = np.sqrt(np.maximum(var_x, 1e-12))
        mu_y = sum_y / max(1, n_total)
        var_y = sum_y2 / max(1, n_total) - mu_y ** 2
        sigma_y = float(np.sqrt(max(var_y, 1e-12)))
        # Bytes accounting for phase 1
        bytes_recv_p1 = len(clients) * BYTES_PER_ELEM * (2 * d + 3)  # per-client: n, sum_x(d), sum_x2(d), sum_y, sum_y2
        bytes_sent_p1 = BYTES_PER_ELEM * (2 * d + 2)  # broadcast mu_x(d), sigma_x(d), mu_y, sigma_y
        log1 = RoundLog(
            round=1,
            bytes_sent=int(bytes_sent_p1),
            bytes_recv=int(bytes_recv_p1),
            duration_sec=0.0,
            participation_rate=1.0,
            pds_bytes=0,
            used_model=used_model,
        )

        # Implement a federated PLS by having each client fit a small
        # ParametricPLSModel locally and averaging the resulting linear
        # parameters (w, b) on the server (FedAvg-style). This avoids
        # sharing raw datapoints and integrates with existing DP/clipping
        # behaviour implemented in the aggregation algorithms.
        try:
            from ..models.parametric_pls import ParametricPLSModel as _ParametricPLSModel
        except Exception:
            _ParametricPLSModel = None

        # If environment requests SIMPLS one-shot, compute from aggregated
        # second-moment sufficient statistics (Sxx, Sxy) and derive PLS
        # components without exchanging raw samples. This path is selected
        # when FEDCHEM_FEDPLS_METHOD environment variable == 'simpls'.
        use_simpls = os.environ.get("FEDCHEM_FEDPLS_METHOD", "").strip().lower() == "simpls"

        if use_simpls:
            # Aggregate standardized sufficient statistics across clients
            Sxx = np.zeros((d, d), dtype=float)
            Sxy = np.zeros(d, dtype=float)
            for c in clients:
                Xc = np.asarray(c["X"])  # (n_i, d)
                yc = np.asarray(c["y"])  # (n_i,)
                Xcs = (Xc - mu_x) / sigma_x
                ycs = (yc - mu_y) / sigma_y
                Sxx += Xcs.T @ Xcs
                Sxy += Xcs.T @ ycs

            # Determine number of components to extract
            n_comp = 1
            try:
                if model is not None and hasattr(model, "n_components") and getattr(model, "n_components") is not None:
                    n_comp = int(getattr(model, "n_components"))
                else:
                    n_comp = min(d, 5)
            except Exception:
                n_comp = min(d, 5)

            # SIMPLS-like iterative solver for PLS1 using Sxx and Sxy
            # W: (d, h), P: (d, h), Q: (h,)
            W = np.zeros((d, n_comp), dtype=float)
            P = np.zeros((d, n_comp), dtype=float)
            Q = np.zeros((n_comp,), dtype=float)
            S_mat = Sxy.reshape((d, 1)).copy()  # (d,1)
            R = Sxx.copy()
            comps_found = 0
            for h in range(n_comp):
                # weight direction from current cross-covariance
                v = S_mat.reshape((d,))
                # solve R w = v
                try:
                    w = np.linalg.solve(R, v)
                except Exception:
                    # fallback to pseudo-inverse
                    w = np.linalg.pinv(R) @ v
                denom = float(np.sqrt(max(1e-20, w.T @ (R @ w))))
                if denom <= 1e-12:
                    break
                w = w / denom
                p = R @ w
                q = float(w.T @ S_mat.reshape((d,)))
                # store
                W[:, h] = w
                P[:, h] = p
                Q[h] = q
                # deflate S_mat
                S_mat = S_mat - np.outer(p, q)
                comps_found += 1

            if comps_found == 0:
                beta_c = np.zeros(d, dtype=float)
            else:
                # compute regression coefficients in standardized space
                # beta = W @ inv(P^T W) @ Q
                PTW = P[:, :comps_found].T @ W[:, :comps_found]
                try:
                    inv_PTW = np.linalg.inv(PTW)
                except Exception:
                    inv_PTW = np.linalg.pinv(PTW)
                beta_c = W[:, :comps_found] @ (inv_PTW @ Q[:comps_found])

            # Map back to original scale
            w_hat = beta_c / sigma_x
            b_hat = float(mu_y - np.sum(beta_c * (mu_x / sigma_x)))
            from ..models.linear import LinearModel  # local import to avoid cycles
            mdl = LinearModel()
            mdl.set_params({"w": w_hat.astype(float), "b": np.array([b_hat], dtype=float)})
            # Provide two-phase logs (standardization + simpls)
            BYTES_PER_ELEM = 8
            bytes_recv_p1 = len(clients) * BYTES_PER_ELEM * (2 * d + 3)
            bytes_sent_p1 = BYTES_PER_ELEM * (2 * d + 2)
            log1 = RoundLog(round=1, bytes_sent=int(bytes_sent_p1), bytes_recv=int(bytes_recv_p1), duration_sec=0.0, participation_rate=1.0, pds_bytes=0, used_model=used_model)
            bytes_recv_p2 = len(clients) * BYTES_PER_ELEM * (d * d + d)
            bytes_sent_p2 = BYTES_PER_ELEM * (d + 1)
            # Evaluate if requested
            m_rmsep = None
            m_r2 = None
            m_mae = None
            if eval_fn is not None:
                try:
                    metrics = eval_fn(mdl)
                    m_rmsep = _safe_float(metrics.get("rmsep"))
                    m_r2 = _safe_float(metrics.get("r2"))
                    m_mae = _safe_float(metrics.get("mae"))
                except Exception:
                    pass
            # If the SIMPLS-derived model is catastrophically poor, fallback to a pooled ParametricPLSModel
            if (m_rmsep is not None and m_rmsep > 5.0) or (m_r2 is not None and m_r2 < -50):
                try:
                    # pooled fit to recover a more stable model
                    from ..models.parametric_pls import ParametricPLSModel as _ParametricPLSModel
                    if _ParametricPLSModel is not None:
                        Xs = [c["X"] for c in clients]
                        ys = [c["y"] for c in clients]
                        X_full = np.vstack(Xs)
                        y_full = np.hstack(ys)
                        fallback_pls = _ParametricPLSModel(n_components=None, max_components=20, cv=5, random_state=0)
                        fallback_pls.fit(X_full, y_full)
                        from ..models.linear import LinearModel
                        fallback_mdl = LinearModel()
                        params = fallback_pls.get_params()
                        fallback_mdl.set_params({"w": params.get("w"), "b": params.get("b")})
                        mdl = fallback_mdl
                        used_model = "pooled_parametric_pls"
                        try:
                            logging.warning("SIMPLS-derived model was poor (rmsep=%s); falling back to pooled ParametricPLSModel", m_rmsep)
                        except Exception:
                            pass
                        # Re-evaluate
                        if eval_fn is not None:
                            try:
                                metrics2 = eval_fn(mdl)
                                val_rmsep = metrics2.get("rmsep")
                                val_r2 = metrics2.get("r2")
                                val_mae = metrics2.get("mae")
                                if val_rmsep is not None:
                                    m_rmsep = float(val_rmsep)
                                if val_r2 is not None:
                                    m_r2 = float(val_r2)
                                if val_mae is not None:
                                    m_mae = float(val_mae)
                            except Exception:
                                pass
                except Exception:
                    # Ignore fallback failures; keep original mdl
                    pass
            log2 = RoundLog(round=2, bytes_sent=int(bytes_sent_p2), bytes_recv=int(bytes_recv_p2), duration_sec=0.0, participation_rate=1.0, rmsep=m_rmsep, r2=m_r2, mae=m_mae, pds_bytes=0, used_model=used_model)
            return {"model": mdl, "logs": [log1.__dict__, log2.__dict__], "dp": {}, "used_model": used_model}

        if _ParametricPLSModel is None:
            # Fall back to previous closed-form MVP one-component implementation
            Sxx = np.zeros((d, d), dtype=float)
            Sxy = np.zeros(d, dtype=float)
            for c in clients:
                Xc = np.asarray(c["X"])  # (n_i, d)
                yc = np.asarray(c["y"])  # (n_i,)
                Xcs = (Xc - mu_x) / sigma_x
                ycs = (yc - mu_y) / sigma_y
                Sxx += Xcs.T @ Xcs
                Sxy += Xcs.T @ ycs
            # One-component PLS1 closed-form using Sxx, Sxy
            w = Sxy.copy()
            denom = float(w.T @ (Sxx @ w))
            num = float(w.T @ Sxy)
            if denom <= 1e-12:
                scale = 0.0
            else:
                scale = num / denom
            beta_c = w * scale
            w_hat = beta_c / sigma_x
            b_hat = float(mu_y - np.sum(beta_c * (mu_x / sigma_x)))
            from ..models.linear import LinearModel  # local import to avoid cycles
            mdl = LinearModel()
            mdl.set_params({"w": w_hat.astype(float), "b": np.array([b_hat], dtype=float)})
        else:
            # Build global ParametricPLSModel and perform FedAvg-style rounds
            global_pls = _ParametricPLSModel(
                n_components=getattr(model, "n_components", None) if model is not None else None,
                max_components=getattr(model, "max_components", 20) if model is not None else 20,
                cv=getattr(model, "cv", 5) if model is not None else 5,
                random_state=getattr(model, "random_state", None) if model is not None else None,
            )
            # If global scaling is configured, propagate global stats computed in phase 1
            try:
                use_global_scaler_env = os.environ.get("FEDCHEM_USE_GLOBAL_SCALER")
                if use_global_scaler_env is None:
                    use_global_scaler = bool(getattr(model, 'use_global_scaler', False) if model is not None else False)
                else:
                    use_global_scaler = str(use_global_scaler_env).strip().lower() in {"1", "true", "t", "yes", "y"}
                if use_global_scaler:
                    global_pls.set_use_global_scaler(True)
                    global_pls.set_global_scaler(mu_x, sigma_x)
            except Exception:
                pass
            # Initialize params to zeros so vectorization works
            try:
                d_local = int(clients[0]["X"].shape[1])
            except Exception:
                d_local = d
            global_pls.set_params({"w": np.zeros(d_local, dtype=float), "b": np.array([0.0], dtype=float)})

            # Setup RNG
            rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
            algo_impl = FedAvgAlgorithm(rng=rng)

            logs: List[RoundLog] = []
            BYTES_PER_ELEM = 8
            dp_noise_std_local: Optional[float] = _safe_float(dp_config.get("noise_std")) if dp_config else None
            for r in range(1, max(1, int(rounds)) + 1):
                # Simple participation sampling (mirrors run_rounds behaviour when dp_config is present)
                if dp_config and "participation_rate" in dp_config:
                    q = float(dp_config.get("participation_rate", 1.0))
                    N = len(clients)
                    m = max(1, int(rng.binomial(N, min(max(q, 0.0), 1.0))))
                    idx = np.arange(0, N)
                    chosen = rng.choice(idx, size=m, replace=False)
                    chosen_clients = [clients[i] for i in chosen]
                else:
                    chosen_clients = clients

                batches: List[ClientBatch] = [ClientBatch(np.asarray(c["X"]), np.asarray(c["y"])) for c in chosen_clients]
                # Allow a runtime override to prevent single-client (or very small) updates
                # to avoid extreme variance and catastrophic rounds. When `FEDCHEM_MIN_PARTICIPANTS`
                # is set > 1, skip aggregation and keep global params unchanged for rounds
                # where fewer than that many clients have participated.
                try:
                    min_participants = int(os.environ.get("FEDCHEM_MIN_PARTICIPANTS", "1"))
                except Exception:
                    # On any failure parsing the env var, default to 1 participant (no skip).
                    min_participants = 1
                if min_participants > 1 and len(chosen_clients) < min_participants:
                    # No aggregation — keep previous params and log a skipped-round entry
                    logs.append(RoundLog(
                        round=r,
                        bytes_sent=0,
                        bytes_recv=0,
                        duration_sec=0.0,
                        participation_rate=float(len(chosen_clients) / max(1, len(clients))),
                        participants=len(chosen_clients),
                        dp_noise_std=float(dp_noise_std_local) if dp_noise_std_local is not None else None,
                        rmsep=None,
                        r2=None,
                        mae=None,
                        pds_bytes=0,
                    ))
                    continue

                # Aggregate using existing FedAvgAlgorithm (which fits fresh model instances per client)
                try:
                    new_params = algo_impl.aggregate(global_pls, batches, clip_norm=clip_norm, dp_noise_std=(dp_config.get("noise_std") if dp_config else None))
                except Exception:
                    # If aggregation fails, keep previous params
                    new_params = global_pls.get_params()
                global_pls.set_params(new_params)
                # Capture algo diagnostics (aggregation method & trimming fraction)
                algo_agg_method = None
                algo_agg_trim_frac = None
                try:
                    if hasattr(algo_impl, "last_stats") and isinstance(algo_impl.last_stats, dict):
                        algo_agg_method = algo_impl.last_stats.get("agg_method")
                        algo_agg_trim_frac = algo_impl.last_stats.get("agg_trim_frac")
                except Exception:
                    algo_agg_method = None
                    algo_agg_trim_frac = None

                # Evaluate, if requested
                m_rmsep = None
                m_r2 = None
                m_mae = None
                if eval_fn is not None:
                    try:
                        metrics = eval_fn(global_pls)
                        _val = metrics.get("rmsep")
                        m_rmsep = float(_val) if _val is not None else None
                        _val = metrics.get("r2")
                        m_r2 = float(_val) if _val is not None else None
                        _val = metrics.get("mae")
                        m_mae = float(_val) if _val is not None else None
                    except Exception:
                        pass

                # Simple bytes accounting: server sends params and receives updates from participants
                param_elems = int(np.asarray(global_pls.get_params().get("w", np.zeros(0))).size + 1)
                sent = BYTES_PER_ELEM * param_elems * len(chosen_clients)
                recv = BYTES_PER_ELEM * param_elems * len(chosen_clients)
                logs.append(RoundLog(round=r, bytes_sent=int(sent), bytes_recv=int(recv), duration_sec=0.0, participation_rate=float(len(chosen_clients) / max(1, len(clients))), rmsep=m_rmsep, r2=m_r2, mae=m_mae, pds_bytes=0, agg_method=str(algo_agg_method) if algo_agg_method is not None else None, agg_trim_frac=_safe_float(algo_agg_trim_frac)))

            # After rounds, wrap into a LinearModel for compatibility with downstream code
            from ..models.linear import LinearModel
            final_params = global_pls.get_params()
            w_final = np.asarray(final_params.get("w", np.zeros(d_local, dtype=float))).astype(float)
            b_final = float(np.asarray(final_params.get("b", np.array([0.0]))).ravel()[0])
            mdl = LinearModel()
            mdl.set_params({"w": w_final, "b": np.array([b_final], dtype=float)})
            # Prepare fallback config
            try:
                fallback_factor = float(os.environ.get("FEDCHEM_FEDPLS_FALLBACK_FACTOR", "2.0"))
            except Exception:
                fallback_factor = 2.0
            try:
                fallback_abs_rmsep = float(os.environ.get("FEDCHEM_FEDPLS_FALLBACK_ABS_RMSEP", "5.0"))
            except Exception:
                fallback_abs_rmsep = 5.0
            fallback_reason = None
            # Attempt pooled fallback when parametric FedPLS seems much worse than pooled fit
            try:
                if _ParametricPLSModel is not None and eval_fn is not None:
                    # build pooled dataset
                    Xs = [c["X"] for c in clients]
                    ys = [c["y"] for c in clients]
                    X_full = np.vstack(Xs)
                    y_full = np.hstack(ys)
                    pooled_pls = _ParametricPLSModel(
                        n_components=getattr(model, "n_components", None) if model is not None else None,
                        max_components=getattr(model, "max_components", 20) if model is not None else 20,
                        cv=getattr(model, "cv", 5) if model is not None else 5,
                        random_state=getattr(model, "random_state", None) if model is not None else None,
                    )
                    pooled_pls.fit(X_full, y_full)
                    pooled_params = pooled_pls.get_params()
                    pooled_w = np.asarray(pooled_params.get("w", np.zeros(d_local, dtype=float))).astype(float)
                    pooled_b = float(np.asarray(pooled_params.get("b", np.array([0.0]))).ravel()[0])
                    pooled_mdl = LinearModel()
                    pooled_mdl.set_params({"w": pooled_w, "b": np.array([pooled_b], dtype=float)})
                    pooled_metrics = None
                    try:
                        pooled_metrics = eval_fn(pooled_mdl)
                    except Exception:
                        pooled_metrics = None
                    pooled_rmsep = _safe_float(pooled_metrics.get("rmsep")) if pooled_metrics is not None else None
                    # Evaluate current federated (converted) model
                    fed_rmsep = None
                    try:
                        fed_metrics = eval_fn(mdl)
                        fed_rmsep = _safe_float(fed_metrics.get("rmsep")) if fed_metrics is not None else None
                    except Exception:
                        fed_rmsep = None
                    # Decide fallback
                    used_model = "FedPLS"
                    if pooled_rmsep is not None:
                        if fed_rmsep is None or fed_rmsep > max(fallback_abs_rmsep, fallback_factor * pooled_rmsep):
                            # use pooled model instead
                            mdl = pooled_mdl
                            used_model = "pooled_parametric_pls"
                            fallback_reason = f"fallback_to_pooled_parametric_pls pooled_rmsep={pooled_rmsep:.6f} fed_rmsep={fed_rmsep}" if fed_rmsep is not None else f"fallback_to_pooled_parametric_pls pooled_rmsep={pooled_rmsep:.6f} fed_rmsep=None"
                            try:
                                logging.warning("FedPLS parametric model is poor (fed_rmsep=%s); falling back to pooled Parametric PLS (pooled_rmsep=%s)", fed_rmsep, pooled_rmsep)
                            except Exception:
                                pass
                    else:
                        # if pooled_rmsep not computable but fed_rmsep is catastrophic, fallback to pooled linear
                        if fed_rmsep is not None and fed_rmsep > fallback_abs_rmsep:
                            # fallback to simple linear model (ridge/OLS) fit on pooled data
                            from ..models.linear import LinearModel as LinearModelFallback
                            # Use ridge_lstsq in _common to create robust linear fit
                            try:
                                from ..models._common import ridge_lstsq
                                w_lin, b_lin = ridge_lstsq(X_full, y_full, ridge=0.1, fit_intercept=True)
                                fallback_lin = LinearModelFallback()
                                fallback_lin.set_params({"w": w_lin, "b": np.array([b_lin], dtype=float)})
                                mdl = fallback_lin
                                used_model = "pooled_linear"
                                fallback_reason = f"fallback_to_pooled_linear fed_rmsep={fed_rmsep:.6f}"
                                try:
                                    logging.warning("FedPLS parametric model is poor (fed_rmsep=%s); falling back to pooled linear ridge (approx) ", fed_rmsep)
                                except Exception:
                                    pass
                            except Exception:
                                # If any failure here, do not crash
                                fallback_reason = fallback_reason or None
            except Exception:
                # best-effort fallback; ignore any exceptions
                fallback_reason = fallback_reason or None
        # end else _ParametricPLSModel

        # Evaluate if requested
        m_rmsep: Optional[float] = None
        m_r2: Optional[float] = None
        m_mae: Optional[float] = None
        if eval_fn is not None:
            try:
                metrics = eval_fn(mdl)
                val_rmsep = metrics.get("rmsep")
                val_r2 = metrics.get("r2")
                val_mae = metrics.get("mae")
                m_rmsep = float(val_rmsep) if val_rmsep is not None else None
                m_r2 = float(val_r2) if val_r2 is not None else None
                m_mae = float(val_mae) if val_mae is not None else None
            except Exception:
                pass

        # Bytes accounting for phase 2
        bytes_recv_p2 = len(clients) * BYTES_PER_ELEM * (d * d + d)
        bytes_sent_p2 = BYTES_PER_ELEM * (d + 1)  # broadcast w_hat(d) and b_hat
        log2 = RoundLog(
            round=2,
            bytes_sent=int(bytes_sent_p2),
            bytes_recv=int(bytes_recv_p2),
            duration_sec=0.0,
            participation_rate=1.0,
            rmsep=m_rmsep,
            r2=m_r2,
            mae=m_mae,
            eval_error=fallback_reason if fallback_reason is not None else None,
            used_model=used_model,
        )
        return {"model": mdl, "logs": [log1.__dict__, log2.__dict__], "dp": {}, "used_model": used_model}

# Helper safe float (avoid circular imports)
def _safe_float(val: Any) -> Optional[float]:
    try:
        return float(val) if val is not None else None
    except Exception:
        return None
