"""
Run Objectives 1, 2, 5, and 6 on the shared real 3-site dataset.

- Loads the persistent 3-site design (or Tecator fallback) once.
- Reuses the same site splits across the telemetry, conformal, benchmarking, and privacy runs.
- Honors standard env toggles (FEDCHEM_NUM_SITES, FEDCHEM_SEED, FEDCHEM_QUICK, FEDCHEM_USE_TECATOR).
"""

from __future__ import annotations

import math
import os
import shutil
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np

from fedchem.utils.real_data import load_idrc_wheat_shootout_site_dict
from fedchem.utils.config import (
    load_config,
    get_experimental_sites,
    get_data_config,
    get_instrument_to_site_map,
)

from . import generate_objective_1 as objective1
from . import generate_objective_2 as objective2
from . import generate_objective_5 as objective5
from . import generate_objective_6 as objective6


def _load_env_int(name: str, default: int) -> int:
    val = os.environ.get(name)
    if val is None:
        return default
    try:
        parsed = int(val)
        return parsed if parsed > 0 else default
    except Exception:
        return default


def _coerce_int(value: Any, default: int | None = None) -> int | None:
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    try:
        return int(float(str(value)))
    except Exception:
        return default


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "t", "yes", "y"}:
            return True
        if lowered in {"0", "false", "f", "no", "n"}:
            return False
    return default


def _ensure_list(values: Any) -> list[Any]:
    if values is None:
        return []
    if isinstance(values, (list, tuple, set)):
        return list(values)
    if isinstance(values, str):
        # Split comma-separated strings but keep simple tokens together
        parts = [v.strip() for v in values.split(",") if v.strip()]
        return parts if parts else [values]
    return [values]


def _parse_transfer_values(values: Any, fallback: list[int]) -> list[int]:
    candidates = []
    for item in _ensure_list(values):
        parsed = _coerce_int(item)
        if parsed is not None and parsed > 0:
            candidates.append(parsed)
    if candidates:
        # Preserve order but remove duplicates while keeping stability
        seen: set[int] = set()
        result: list[int] = []
        for v in candidates:
            if v not in seen:
                seen.add(v)
                result.append(v)
        return result
    return fallback


def _parse_privacy_budgets(values: Any, fallback: list[float]) -> list[float]:
    candidates: list[float] = []
    for item in _ensure_list(values):
        if isinstance(item, str) and item.strip() in {"∞", "inf", "infinity", "INF"}:
            candidates.append(float("inf"))
            continue
        try:
            candidates.append(float(item))
        except Exception:
            continue
    if candidates:
        seen: list[float] = []
        for val in candidates:
            if val not in seen:
                seen.append(val)
        return seen
    return fallback


def _site_train_split(site: Dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    # Primary expectation: explicit 'X'/'y' fields present (calibration split)
    if "X" in site and "y" in site:
        X = site["X"]
        y = site["y"]
        # Prefer a non-empty 'cal' tuple when the X/y pair is missing or malformed: try fallbacks
        if X is None or y is None:
            cal = site.get("cal")
            if isinstance(cal, tuple) and len(cal) == 2 and cal[0] is not None and cal[1] is not None:
                X, y = cal
            else:
                # Try to fall back to validation or test sets (useful when cal is empty)
                val = site.get("val")
                test = site.get("test")
                # Look for a val/test with both X and y present
                for candidate in (val, test):
                    if isinstance(candidate, tuple) and len(candidate) == 2:
                        cX, cy = candidate
                        if cX is not None and cy is not None:
                            X, y = cX, cy
                            break
                # If still missing, raise with helpful context
                if X is None or y is None:
                    site_id = site.get("instrument_id") or site.get("backing_site") or site.get("site")
                    extra = f" for site '{site_id}'" if site_id else ""
                    raise ValueError(
                        "Site 'X' or 'y' is None — calibration data seems missing or misconfigured." + extra
                    )
        X = np.asarray(X)
        # Protect against poorly formed scalars (np.asarray(None) yields a 0-dim array)
        if X.ndim == 0:
            raise ValueError("Site 'X' is not an array-like with at least one dimension.")
        y = np.asarray(y)
        if y.ndim == 0:
            raise ValueError("Site 'y' is not an array-like with at least one dimension.")
        # Guard against empty calibration splits (no rows)
        if X.shape[0] == 0 or y.shape[0] == 0:
            site_id = site.get("instrument_id") or site.get("backing_site") or site.get("site")
            extra = f" for site '{site_id}'" if site_id else ""
            raise ValueError("Site 'X' or 'y' is empty — calibration rows missing." + extra)
        return X, y
    cal = site.get("cal")
    if isinstance(cal, tuple) and len(cal) == 2:
        X, y = cal
        if X is None or y is None:
            # Try val/test fallbacks again for 'cal'-only sites
            val = site.get("val")
            test = site.get("test")
            for candidate in (val, test):
                if isinstance(candidate, tuple) and len(candidate) == 2:
                    cX, cy = candidate
                    if cX is not None and cy is not None:
                        X, y = cX, cy
                        break
            if X is None or y is None:
                raise ValueError("Calibration split missing data for site; ensure DATA_CONFIG is configured correctly.")
        # Convert to arrays and validate shapes
        if isinstance(X, np.ndarray) and X.ndim == 0:
            raise ValueError("Site 'X' is not an array-like with at least one dimension.")
        if isinstance(y, np.ndarray) and y.ndim == 0:
            raise ValueError("Site 'y' is not an array-like with at least one dimension.")
        if np.asarray(X).shape[0] == 0 or np.asarray(y).shape[0] == 0:
            raise ValueError("Calibration split missing data for site; ensure DATA_CONFIG is configured correctly.")
        return np.asarray(X), np.asarray(y)
    raise KeyError("Expected site dictionary to provide either 'X'/'y' or 'cal' tuple.")


def _build_clients(site_data: Dict[str, Dict[str, np.ndarray]]) -> list[dict[str, np.ndarray]]:
    clients: list[dict[str, np.ndarray]] = []
    skipped = 0
    for key, site in sorted(site_data.items()):
        try:
            X, y = _site_train_split(site)
        except (ValueError, KeyError) as exc:
            print(f"[Unified] Skipping site '{key}' due to missing calibration data: {exc}")
            skipped += 1
            continue
        clients.append({"X": X, "y": y})
    if not clients:
        # Provide actionable guidance when no sites are found.
        available_keys = list(site_data.keys()) if isinstance(site_data, dict) else []
        msg = (
            "No valid sites with calibration data were found — aborting experiment. "
            "Check that your config's 'DATA_CONFIG' keys or 'INSTRUMENTS' IDs match the directories under 'data/', "
            "that INSTRUMENT_AS_SITE is set consistently, and that instruments are enabled in 'INSTRUMENTS' or 'INSTRUMENT_MAP'. "
            f"Available site keys from loader: {available_keys}"
        )
        raise ValueError(msg)
    if skipped:
        print(f"[Unified] Skipped {skipped} sites with missing calibration data; continuing with {len(clients)} site(s).")
    return clients


def _build_privacy_clients(site_data: Dict[str, Dict[str, np.ndarray]], test_samples_per_site: int = 30):
    """Build privacy clients while capping the per-site test split to the experimental design."""
    clients = []
    Xte_list, yte_list = [], []
    for key, site in sorted(site_data.items()):
        try:
            X, y = _site_train_split(site)
        except (ValueError, KeyError) as exc:
            print(f"[Unified] Skipping site '{key}' in privacy clients due to missing calibration data: {exc}")
            continue
        n = X.shape[0]
        n_test = min(test_samples_per_site, n // 2)
        n_train = max(1, n - n_test)
        clients.append({"X": X[:n_train], "y": y[:n_train]})
        Xte_list.append(X[n_train:n_train + n_test])
        yte_list.append(y[n_train:n_train + n_test])
    # If no test slices were collected, return empty arrays with consistent shapes
    if not Xte_list:
        X_test = np.empty((0, 0))
        y_test = np.array([])
    else:
        X_test = np.vstack(Xte_list)
        y_test = np.hstack(yte_list)
    if not clients:
        # Return empty client list and empty test arrays, consistent with tests
        return clients, X_test, y_test
    return clients, X_test, y_test


OUTPUT_DIR = Path(os.environ.get("FEDCHEM_OUTPUT_DIR", "generated_figures_tables"))
ARCHIVE_ROOT = Path(os.environ.get("FEDCHEM_ARCHIVE_ROOT", "generated_figures_tables_archive"))
TRANSFER_SAMPLE_VALUES_DEFAULT = [20, 40, 80, 200]
EPSILON_VALUES_DEFAULT = [float("inf"), 10.0, 1.0, 0.1]
DEFAULT_N_WAVELENGTHS_FALLBACK = 256


def _reset_output_dir():
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _archive_outputs(label: str) -> Path:
    ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)
    dest = ARCHIVE_ROOT / label
    if dest.exists():
        shutil.rmtree(dest)
    if OUTPUT_DIR.exists():
        shutil.move(str(OUTPUT_DIR), str(dest))
    else:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return dest


def main():
    start = time.perf_counter()
    cfg = load_config()
    config = cfg
    config_sites = get_experimental_sites(cfg)
    instrument_map = get_instrument_to_site_map(cfg)
    # Build a loader-friendly instrument map keyed by sheet names where possible.
    # Some Excel sheets in the IDRC dataset have names like 'CalSetA1', which don't
    # match the instrument IDs in `INSTRUMENTS` (e.g., 'MA_A1'). To avoid filtering out
    # available sheets, create alternate keys: instrument id, suffix, and 'CalSet' + suffix.
    instrument_map_for_loader = {}
    if instrument_map:
        for iid, meta in instrument_map.items():
            # original key
            instrument_map_for_loader[iid] = meta
            # suffix: part after last underscore (e.g., 'A1' in 'MA_A1')
            suffix = str(iid).split("_")[-1]
            if suffix not in instrument_map_for_loader:
                instrument_map_for_loader[suffix] = meta
            calset_key = f"CalSet{suffix}"
            if calset_key not in instrument_map_for_loader:
                instrument_map_for_loader[calset_key] = meta
    data_config = get_data_config(cfg)
    # Convert config 'Site' entries to loader-expected manufacturer directories when using INSTRUMENT_AS_SITE.
    # When the experimental design uses instrument IDs (e.g., 'MA_A1') and we also set
    # INSTRUMENT_AS_SITE=True, the loader expects manufacturer paths (e.g., 'ManufacturerA'),
    # otherwise it will attempt to open 'data/MA_A1' which doesn't exist. Map instrument IDs
    # to their 'site' (manufacturer) via the instrument map.
    instrument_map = get_instrument_to_site_map(cfg)
    instrument_as_site_flag = bool(cfg.get("INSTRUMENT_AS_SITE"))
    loader_sites = config_sites
    if instrument_as_site_flag and config_sites:
        # Map any instrument ID in `config_sites` to its backing manufacturer site name
        mapped: set[str] = set()
        for s in _ensure_list(config_sites):
            if s in instrument_map and instrument_map[s].get("site"):
                mapped.add(instrument_map[s]["site"])  # Manufacturer name
            else:
                # If the provided site isn't an instrument ID (already manufacturer), keep as-is
                mapped.add(s)
        loader_sites = sorted(mapped)
        print(f"[Unified] Mapping experimental instrument sites {config_sites} to loader directories {loader_sites}")
    else:
        loader_sites = config_sites
    exp_design = config.get("EXPERIMENTAL_DESIGN", {})
    factors = exp_design.get("FACTORS", {})
    raw_sites = factors.get("Site")
    # Ensure we honor any explicit site code selections in the config
    config_sites_list = _ensure_list(raw_sites) or (_ensure_list(config_sites) if config_sites else None)
    config_sites = config_sites_list if config_sites_list else None
    data_config = config.get("DATA_CONFIG") or {}

    seed_default = _coerce_int(config.get("SEED"), 42) or 42
    seed = _load_env_int("FEDCHEM_SEED", seed_default)

    n_sites_default = _coerce_int(config.get("NUM_SITES"), 3) or 3
    n_sites = _load_env_int("FEDCHEM_NUM_SITES", n_sites_default)

    quick_default = _coerce_bool(config.get("QUICK"), False)
    quick_env = os.environ.get("FEDCHEM_QUICK")
    quick = _coerce_bool(quick_env, quick_default) if quick_env is not None else quick_default

    force_default = _coerce_bool(config.get("USE_TECATOR"), False)
    force_env = os.environ.get("FEDCHEM_USE_TECATOR")
    force_tecator = _coerce_bool(force_env, force_default) if force_env is not None else force_default

    n_wavelengths_default = _coerce_int(config.get("DEFAULT_N_WAVELENGTHS"), DEFAULT_N_WAVELENGTHS_FALLBACK)
    if n_wavelengths_default is None or n_wavelengths_default <= 0:
        n_wavelengths_default = DEFAULT_N_WAVELENGTHS_FALLBACK
    n_wavelengths_env = os.environ.get("FEDCHEM_N_WAVELENGTHS")
    if n_wavelengths_env is not None:
        env_override = _coerce_int(n_wavelengths_env, n_wavelengths_default)
        if env_override is not None and env_override > 0:
            n_wavelengths_default = env_override
    n_wavelengths = n_wavelengths_default

    # If a higher-level runner (e.g., run_all_objectives.py) sets
    # FEDCHEM_SKIP_INTERNAL_DESIGN=1 we accept env-provided single combo values
    # and avoid re-enumerating the entire experimental design here.
    skip_internal = os.environ.get("FEDCHEM_SKIP_INTERNAL_DESIGN") == "1"
    if skip_internal:
        # Read transfer values, privacy epsilon and delta from env (singletons expected)
        transfer_values = _parse_transfer_values(os.environ.get("FEDCHEM_TRANSFER_SAMPLES"), TRANSFER_SAMPLE_VALUES_DEFAULT)
        privacy_values = _parse_privacy_budgets(os.environ.get("FEDCHEM_DP_TARGET_EPS"), EPSILON_VALUES_DEFAULT)
        try:
            delta_fallback = [float(os.environ.get('FEDCHEM_DP_DELTA', config.get('DP_DELTA', 1e-5)))]
        except Exception:
            delta_fallback = [1e-5]
        delta_values = _parse_privacy_budgets(os.environ.get("FEDCHEM_DP_DELTA"), delta_fallback)
    else:
        transfer_values = _parse_transfer_values(factors.get("Transfer_Samples"), TRANSFER_SAMPLE_VALUES_DEFAULT)
        # Use canonical `DP_Target_Eps` as the privacy factor key in the experimental design
        privacy_raw = None
        if isinstance(factors, dict):
                privacy_raw = factors.get("DP_Target_Eps") or factors.get("DP_TARGET_EPS")
        privacy_values = _parse_privacy_budgets(privacy_raw, EPSILON_VALUES_DEFAULT)
        # Also enumerate delta (DP_DELTA) factor values so we can run full ε × δ grid when requested
        try:
            delta_fallback = [float(config.get('DP_DELTA', 1e-5))]
        except Exception:
            delta_fallback = [1e-5]
        delta_values = _parse_privacy_budgets(factors.get("DP_Delta"), delta_fallback)

    max_transfer_limit = _coerce_int(config.get("MAX_TRANSFER_SAMPLES"))
    test_samples_per_site = _coerce_int(config.get("TEST_SAMPLES_PER_SITE"), 30) or 30

    if isinstance(config_sites, list) and len(config_sites) > n_sites:
        config_sites = config_sites[:n_sites]

    os.environ["FEDCHEM_SEED"] = str(seed)
    os.environ["FEDCHEM_NUM_SITES"] = str(n_sites)
    os.environ["FEDCHEM_QUICK"] = "1" if quick else "0"
    os.environ["FEDCHEM_USE_TECATOR"] = "1" if force_tecator else "0"
    os.environ["FEDCHEM_N_WAVELENGTHS"] = str(n_wavelengths)
    resample_enabled = _coerce_bool(config.get("RESAMPLE_SPECTRA"), True)

    ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)
    _reset_output_dir()

    for k in transfer_values:
        effective_k = k
        if max_transfer_limit is not None:
            effective_k = min(k, max_transfer_limit)
        if effective_k is None or effective_k <= 0:
            continue
        if effective_k != k:
            print(
                f"[Unified] Loading transfer-efficient dataset with k={k} (capped to {effective_k} by MAX_TRANSFER_SAMPLES)..."
            )
        else:
            print(f"[Unified] Loading transfer-efficient dataset with k={k}...")
        k_label = effective_k
        site_data, ds_meta = load_idrc_wheat_shootout_site_dict(
            sites=loader_sites,
            data_dir="data",
            n_wavelengths=n_wavelengths,
            quick_cap_per_site=None,
            seed=seed,
            max_transfer_samples=effective_k,
            resample=resample_enabled,
            data_config=data_config,
            instrument_map=instrument_map_for_loader if instrument_map_for_loader else instrument_map,
            instrument_as_site=bool(cfg.get("INSTRUMENT_AS_SITE")),
        )
        # Debugging/logging: print available site keys and summary counts to help diagnose mismatches
        try:
            loaded_keys = list(site_data.keys()) if isinstance(site_data, dict) else []
            print(f"[Unified] Loader returned {len(loaded_keys)} site(s): {loaded_keys}")
            if isinstance(ds_meta, dict) and ds_meta.get("site_summaries"):
                print(f"[Unified] site_summaries: {ds_meta.get('site_summaries')}")
        except Exception:
            pass
        clients = _build_clients(site_data)
        try:
            privacy_clients, X_test, y_test = _build_privacy_clients(
                site_data, test_samples_per_site=test_samples_per_site
            )
        except ValueError as exc:
            print(f"[Unified] Error building privacy clients for k={k_label}: {exc}")
            raise
        test_samples_used = len(X_test) // len(site_data) if site_data else 0
        print(
            f"[Unified] Assigning {test_samples_used} local test samples per site (target={test_samples_per_site})."
        )
        ds_meta = ds_meta or {}
        print(
            f"[Unified] Dataset {ds_meta.get('dataset', 'real')} loaded (.meta transfer_used="
            f"{ds_meta.get('transfer_samples_used', 'unknown')})."
        )

        # Run DP-sensitive objectives per epsilon
        # Iterate over ε and δ factor levels to support full factorial experiments.
        for eps in privacy_values:
            for delta in delta_values:
                eps_label = "inf" if math.isinf(eps) else str(eps).replace(".", "_")
                # Use a compact label for delta (e.g., 1e-5 -> 1e-05, 1e-6 -> 1e-06)
                try:
                    delta_label = f"delta_{str(delta).replace('.', '_').replace('-', 'neg')}"
                except Exception:
                    delta_label = f"delta_{delta}"
                os.environ["FEDCHEM_DP_TARGET_EPS"] = str(eps)
                os.environ["FEDCHEM_DP_DELTA"] = str(delta)
                print(f"[Unified] Running Objective 1 (Federated telemetry, eps={eps_label}, delta={delta}) for k={k_label}...")
                objective1.main(clients=list(clients), ds_meta=dict(ds_meta))
                _archive_outputs(f"transfer_k_{k_label}/eps_{eps_label}/{delta_label}/objective_1")

                print(f"[Unified] Running Objective 6 (Privacy telemetry, eps={eps_label}, delta={delta}) for k={k_label}...")
                objective6.main(clients=privacy_clients, X_test=X_test, y_test=y_test, ds_meta=dict(ds_meta))
                _archive_outputs(f"transfer_k_{k_label}/eps_{eps_label}/{delta_label}/objective_6")

        os.environ["FEDCHEM_PDS_TRANSFER_N"] = str(k_label)
        print(f"[Unified] Running Objective 2 (Conformal intervals) for k={k_label}...")
        objective2.main(data=site_data, ds_meta=dict(ds_meta))
        _archive_outputs(f"transfer_k_{k_label}/objective_2")

        print(f"[Unified] Running Objective 5 (Benchmarks) for k={k_label}...")
        objective5.main(data=site_data, ds_meta=dict(ds_meta))
        _archive_outputs(f"transfer_k_{k_label}/objective_5")

    elapsed = time.perf_counter() - start
    print(f"[Unified] Completed shared experiment in {elapsed:.1f} seconds.")


if __name__ == "__main__":
    main()
