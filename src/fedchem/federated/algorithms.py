from __future__ import annotations

from typing import Dict, List, Protocol, Optional, Any, cast
from pathlib import Path
import os
import json
import time
import numpy as np

from .types import ClientBatch, ParametricModel, FloatArray
from ..core.registry import register_algo
import math


class FederatedAlgorithm(Protocol):
    def aggregate(
        self,
        model: ParametricModel,
        clients: List[ClientBatch],
        *,
        clip_norm: Optional[float] = None,
        dp_noise_std: Optional[float] = None,
        prox_mu: float = 0.0,
    ) -> Dict[str, FloatArray]:
        ...

def _vectorize(params: Dict[str, FloatArray]) -> tuple[np.ndarray, List[tuple[str, slice]]]:
    keys = []
    vecs = []
    start = 0
    for k in sorted(params.keys()):
        v = params[k].ravel()
        vecs.append(v)
        keys.append((k, slice(start, start + v.size)))
        start += v.size
    try:
        return np.concatenate(vecs) if vecs else np.zeros(0), keys
    except Exception as e:
        raise RuntimeError("Inconsistent parameter shapes when vectorizing model params") from e


def _devectorize(vec: np.ndarray, template: Dict[str, FloatArray], keys: List[tuple[str, slice]]) -> Dict[str, FloatArray]:
    out: Dict[str, FloatArray] = {}
    for k, sl in keys:
        shape = template[k].shape
        out[k] = vec[sl].reshape(shape)
    return out


def _coord_trimmed_mean(stack: np.ndarray, frac: float) -> np.ndarray:
    if frac <= 0.0:
        return stack.mean(axis=0)
    n, d = stack.shape
    k = int(math.floor(n * float(frac)))
    if k <= 0 or 2 * k >= n:
        return stack.mean(axis=0)
    s = np.sort(stack, axis=0)
    trimmed = s[k : n - k, :]
    return trimmed.mean(axis=0)


def _coord_median(stack: np.ndarray) -> np.ndarray:
    return np.median(stack, axis=0)


def _robust_aggregate(stack: np.ndarray, method: str = "mean", frac: float = 0.1) -> np.ndarray:
    method = (method or "").strip().lower()
    if method in ("trim_mean", "trimmed_mean", "trimmed-mean"):
        return _coord_trimmed_mean(stack, frac)
    if method == "median":
        return _coord_median(stack)
    return stack.mean(axis=0)
@register_algo("fedavg")
class FedAvgAlgorithm:
    def __init__(self, rng: Optional[np.random.Generator] = None):
        self.rng = rng or np.random.default_rng()

    def aggregate(
        self,
        model: ParametricModel,
        clients: List[ClientBatch],
        *,
        clip_norm: Optional[float] = None,
        dp_noise_std: Optional[float] = None,
        prox_mu: float = 0.0,
    ) -> Dict[str, FloatArray]:
        # DP-correct FedAvg on parameter updates:
        # 1) compute global parameter vector g
        # 2) each client fits locally -> v_i, form update u_i = v_i - g
        # 3) clip each u_i to L2 norm C (if provided)
        # 4) average updates and add Gaussian noise N(0, sigma^2 I)
        # 5) new params = g + noisy_avg
        global_params = model.get_params()
        gvec, keys = _vectorize(global_params)

        updates: List[np.ndarray] = []
        pre_norms: List[float] = []
        post_norms: List[float] = []
        clipped_count = 0
        for idx, c in enumerate(clients):
            try:
                m = type(model)()
            except Exception as e:
                raise RuntimeError(
                    "Failed to construct a fresh model instance via type(model)(). "
                    "Ensure your model has a zero-argument constructor or provide a model factory."
                ) from e
            # If server provided a global scaler on `model`, propagate to the local model
            try:
                if hasattr(model, 'set_use_global_scaler') and getattr(model, 'use_global_scaler', False):
                    if getattr(model, 'global_scaler_mean', None) is not None and getattr(model, 'global_scaler_scale', None) is not None:
                        if hasattr(m, 'set_use_global_scaler') and hasattr(m, 'set_global_scaler'):
                            m_any = cast(Any, m)
                            m_any.set_use_global_scaler(True)
                            m_any.set_global_scaler(getattr(model, 'global_scaler_mean'), getattr(model, 'global_scaler_scale'))
            except Exception:
                pass
            # If server provided a global scaler on `model`, propagate to the local model
            try:
                if hasattr(model, 'set_use_global_scaler') and getattr(model, 'use_global_scaler', False):
                    if getattr(model, 'global_scaler_mean', None) is not None and getattr(model, 'global_scaler_scale', None) is not None:
                        if hasattr(m, 'set_use_global_scaler') and hasattr(m, 'set_global_scaler'):
                            m_any = cast(Any, m)
                            m_any.set_use_global_scaler(True)
                            m_any.set_global_scaler(getattr(model, 'global_scaler_mean'), getattr(model, 'global_scaler_scale'))
            except Exception:
                pass
            m.fit(c.X, c.y)
            # Optionally dump per-client local model params for debugging
            try:
                if os.environ.get("FEDCHEM_DEBUG_DUMP_WEIGHTS") == "1":
                    out_dir = Path(os.environ.get("FEDCHEM_OUTPUT_DIR", "generated_figures_tables")) / "lca_artifacts" / "debug_weights"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    params = m.get_params()
                    scaler = None
                    n_samples = getattr(c, 'X', None)
                    n_samples = int(n_samples.shape[0]) if n_samples is not None else None
                    try:
                        pipe = getattr(m, 'pipeline', None)
                        if pipe is not None and hasattr(pipe, 'named_steps') and 'scaler' in pipe.named_steps:
                            sc = pipe.named_steps['scaler']
                            sc_mean = getattr(sc, 'mean_', None)
                            sc_scale = getattr(sc, 'scale_', None)
                            scaler = {'mean': sc_mean.tolist() if sc_mean is not None and hasattr(sc_mean, 'tolist') else sc_mean,
                                      'scale': sc_scale.tolist() if sc_scale is not None and hasattr(sc_scale, 'tolist') else sc_scale}
                            # Try to extract selected PLS components if present
                            if 'pls' in pipe.named_steps and hasattr(pipe.named_steps['pls'], 'n_components'):
                                scaler['n_components'] = int(getattr(pipe.named_steps['pls'], 'n_components'))
                    except Exception:
                        scaler = None
                    dump = {
                        'timestamp': time.time(),
                        'algo': 'fedavg',
                        'client_index': idx,
                        'params': {k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in params.items()},
                        'n_samples': n_samples,
                        'scaler': scaler,
                    }
                    # Use a unique name
                    fname = out_dir / f"debug_weight_fedavg_client_{idx}_{int(time.time()*1000)}.json"
                    with open(fname, 'w', encoding='utf-8') as fh:
                        json.dump(dump, fh)
            except Exception:
                pass
            v_i, _ = _vectorize(m.get_params())
            u = v_i - gvec
            n_pre = float(np.linalg.norm(u))
            if clip_norm is not None and clip_norm > 0:
                n = np.linalg.norm(u)
                if n > clip_norm:
                    u = (clip_norm / n) * u
                    clipped_count += 1
            n_post = float(np.linalg.norm(u))
            pre_norms.append(n_pre)
            post_norms.append(n_post)
            updates.append(u)

        if not updates:
            return global_params

        # Determine aggregation strategy via ENV var FEDCHEM_AGG_METHOD (mean, trim_mean, median)
        # and trimming fraction via FEDCHEM_AGG_TRIM_FRAC (0.0-0.45 recommended)
        agg_method = os.environ.get("FEDCHEM_AGG_METHOD", "mean").strip().lower()
        try:
            trim_frac = float(os.environ.get("FEDCHEM_AGG_TRIM_FRAC", "0.1"))
        except Exception:
            trim_frac = 0.1
        upd_stack = np.vstack(updates)

        def _coord_trimmed_mean(stack: np.ndarray, frac: float) -> np.ndarray:
            # Fraction applied per-side, e.g. frac=0.1 removes bottom 10% and top 10% per coordinate
            if frac <= 0.0:
                return stack.mean(axis=0)
            n, d = stack.shape
            k = int(math.floor(n * float(frac)))
            if k <= 0 or 2 * k >= n:
                # Not enough clients to trim; fall back to mean
                return stack.mean(axis=0)
            # sort by column
            s = np.sort(stack, axis=0)
            trimmed = s[k : n - k, :]
            return trimmed.mean(axis=0)

        def _coord_median(stack: np.ndarray) -> np.ndarray:
            return np.median(stack, axis=0)

        def _robust_aggregate(stack: np.ndarray, method: str, frac: float) -> np.ndarray:
            if method in ("trim_mean", "trimmed_mean", "trimmed-mean"):
                return _coord_trimmed_mean(stack, frac)
            if method == "median":
                return _coord_median(stack)
            return stack.mean(axis=0)

        avg_update = _robust_aggregate(upd_stack, agg_method, trim_frac)
        if dp_noise_std is not None and dp_noise_std > 0:
            # Use provided RNG for reproducibility in tests
            avg_update = avg_update + self.rng.normal(0.0, dp_noise_std, size=avg_update.shape)

        new_vec = gvec + avg_update
        # Side-channel diagnostics for orchestrator
        try:
            total = max(1, len(clients))
            self.last_stats = {
                "clip_fraction": float(clipped_count) / float(total),
                "mean_update_norm_raw": float(np.mean(pre_norms)) if pre_norms else None,
                "mean_update_norm_clipped": float(np.mean(post_norms)) if post_norms else None,
                "agg_method": agg_method,
                "agg_trim_frac": float(trim_frac),
            }
        except Exception:
            self.last_stats = {}
        return _devectorize(new_vec, global_params, keys)


@register_algo("fedprox")
class FedProxAlgorithm(FedAvgAlgorithm):
    def aggregate(
        self,
        model: ParametricModel,
        clients: List[ClientBatch],
        *,
        clip_norm: Optional[float] = None,
        dp_noise_std: Optional[float] = None,
        prox_mu: float = 0.0,
    ) -> Dict[str, FloatArray]:
        # For this linear closed-form stub, FedProx reduces to a convex combo
        # between local fits and current global params (proximal pull).
        global_params = model.get_params()
        gvec, keys = _vectorize(global_params)
        local_vecs = []
        for idx, c in enumerate(clients):
            try:
                m = type(model)()
            except Exception as e:
                raise RuntimeError(
                    "Failed to construct a fresh model instance via type(model)() for FedProx. "
                    "Ensure your model has a zero-argument constructor or provide a model factory."
                ) from e
            m.fit(c.X, c.y)
            # Optionally dump per-client local model params for debugging
            try:
                if os.environ.get("FEDCHEM_DEBUG_DUMP_WEIGHTS") == "1":
                    out_dir = Path(os.environ.get("FEDCHEM_OUTPUT_DIR", "generated_figures_tables")) / "lca_artifacts" / "debug_weights"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    params = m.get_params()
                    scaler = None
                    n_samples = getattr(c, 'X', None)
                    n_samples = int(n_samples.shape[0]) if n_samples is not None else None
                    try:
                        pipe = getattr(m, 'pipeline', None)
                        if pipe is not None and hasattr(pipe, 'named_steps') and 'scaler' in pipe.named_steps:
                            sc = pipe.named_steps['scaler']
                            sc_mean = getattr(sc, 'mean_', None)
                            sc_scale = getattr(sc, 'scale_', None)
                            scaler = {'mean': sc_mean.tolist() if sc_mean is not None and hasattr(sc_mean, 'tolist') else sc_mean,
                                      'scale': sc_scale.tolist() if sc_scale is not None and hasattr(sc_scale, 'tolist') else sc_scale}
                    except Exception:
                        scaler = None
                    dump = {
                        'timestamp': time.time(),
                        'algo': 'fedprox',
                        'client_index': idx,
                        'params': {k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in params.items()},
                        'n_samples': n_samples,
                        'scaler': scaler,
                    }
                    fname = out_dir / f"debug_weight_fedprox_client_{idx}_{int(time.time()*1000)}.json"
                    with open(fname, 'w', encoding='utf-8') as fh:
                        json.dump(dump, fh)
            except Exception:
                pass
            p = m.get_params()
            v, _ = _vectorize(p)
            local_vecs.append(v)
        if not local_vecs:
            return global_params
        # Use robust aggregator if configured
        agg_method = os.environ.get("FEDCHEM_AGG_METHOD", "mean").strip().lower()
        try:
            trim_frac = float(os.environ.get("FEDCHEM_AGG_TRIM_FRAC", "0.1"))
        except Exception:
            trim_frac = 0.1
        L = _robust_aggregate(np.vstack(local_vecs), agg_method, trim_frac)
        lam = prox_mu
        new = (1.0 / (1.0 + lam)) * (L + lam * gvec)
        if clip_norm is not None:
            n = np.linalg.norm(new)
            if n > clip_norm > 0:
                new = (clip_norm / n) * new
        if dp_noise_std is not None and dp_noise_std > 0:
            # Use RNG from parent (FedAvgAlgorithm) for reproducible noise
            new = new + self.rng.normal(0.0, dp_noise_std, size=new.shape)
        # Provide comparable diagnostics for FedProx (no per-client updates available here)
        try:
            self.last_stats = {
                "clip_fraction": None,
                "mean_update_norm_raw": None,
                "mean_update_norm_clipped": float(np.linalg.norm(new - gvec)),
            }
        except Exception:
            self.last_stats = {}
        return _devectorize(new, global_params, keys)
