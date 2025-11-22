from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterator

import numpy as np
import pandas as pd

from fedchem import load_dataset

try:  # pragma: no cover - optional dependency
    from .persistent_dataset import load_persistent_site
except Exception:  # pragma: no cover - import guard for packaging
    load_persistent_site = None

RealSiteDict = Dict[str, Dict[str, np.ndarray]]
MetaDict = Dict[str, Any]

__all__ = ["load_real_site_dict", "load_real_site_clients"]


def _feature_target_split(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray | None]:
    """Extract feature matrix and optional target vector from a dataframe."""
    target_col = next(
        (c for c in df.columns if isinstance(c, str) and c.lower() in {"protein", "target", "y"}),
        None,
    )
    if target_col is None:
        target_col = next(
            (c for c in df.columns if str(c).lower() in {"protein", "target", "y"}),
            None,
        )

    feature_cols = [c for c in df.columns if c != target_col and str(c).lower() != "id"]
    X = df[feature_cols].to_numpy()
    y = df[target_col].to_numpy() if target_col is not None else None
    return X, y


def _cap_rows(X: np.ndarray, y: np.ndarray | None, cap: int | None) -> tuple[np.ndarray, np.ndarray | None]:
    if cap is None or X.shape[0] <= cap:
        return X, y
    X = X[:cap]
    if y is not None:
        y = y[:cap]
    return X, y


def _random_subset(
    X: np.ndarray,
    y: np.ndarray | None,
    limit: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    if limit is None or X.shape[0] <= limit:
        return X, y
    rng = np.random.RandomState(seed)
    idx = np.sort(rng.choice(X.shape[0], size=limit, replace=False))
    X = X[idx]
    if y is not None:
        y = y[idx]
    return X, y


def _trim_wavelengths(X: np.ndarray, n_wavelengths: int | None) -> np.ndarray:
    if n_wavelengths is None or X.shape[1] <= n_wavelengths:
        return X
    return X[:, :n_wavelengths]


def _iter_frames(path: Path, multi_sheet: bool) -> Iterator[tuple[str | None, pd.DataFrame]]:
    if not path.exists():
        return iter(())
    if multi_sheet:
        sheets = pd.read_excel(path, sheet_name=None)
        return ((name, df) for name, df in sheets.items())
    frame = pd.read_excel(path)
    return iter([(None, frame)])


def _load_excel_tables(
    path: Path,
    *,
    multi_sheet: bool,
    n_wavelengths: int | None,
    quick_cap_per_site: int | None,
    max_transfer_samples: int | None,
    base_seed: int,
    resample: bool | None = None,
    return_sheet_names: bool = False,
) -> list[tuple[np.ndarray, np.ndarray | None]]:
    results: list[tuple[np.ndarray, np.ndarray | None]] = []
    for offset, (sheet_name, df) in enumerate(_iter_frames(path, multi_sheet)):
        X, y = _feature_target_split(df)
        X, y = _cap_rows(X, y, quick_cap_per_site)
        X, y = _random_subset(X, y, max_transfer_samples, base_seed + offset)
        # Attempt resampling to canonical `n_wavelengths` (interpolation or even subsampling)
        try:
            col_names = list(df.columns) if df is not None else None
            if _is_resampling_enabled(resample):
                # Resolve method from env/arg; prefer explicit method arg present in strings
                method_arg = os.environ.get("FEDCHEM_RESAMPLE_METHOD")
                effective_method = _get_resample_method(resample, method_arg)
                X, _ = resample_spectra(X, col_names=col_names, n_wavelengths=n_wavelengths, method=effective_method)
            else:
                X = _trim_wavelengths(X, n_wavelengths)
        except Exception:
            X = _trim_wavelengths(X, n_wavelengths)
        if return_sheet_names:
            results.append((sheet_name, X, y))
        else:
            results.append((X, y))
    return results


def _limit_tables(
    tables: list[tuple[np.ndarray, np.ndarray | None]],
    limit: int | None,
) -> list[tuple[np.ndarray, np.ndarray | None]]:
    if limit is None or limit >= len(tables):
        return tables
    return tables[:limit]


def _combine_tables(
    tables: list[tuple[np.ndarray, np.ndarray | None]],
    *,
    allow_empty: bool = False,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if not tables:
        return (np.empty((0, 0)), None) if allow_empty else (None, None)
    valid = [(X, y) for X, y in tables if X.size]
    if not valid:
        return (np.empty((0, 0)), None) if allow_empty else (None, None)
    Xs = [np.asarray(X) for X, _ in valid]
    X = np.vstack(Xs) if len(Xs) > 1 else Xs[0]
    if any(y is None for _, y in valid):
        return X, None
    ys = [np.asarray(y) for _, y in valid]
    y = np.hstack(ys) if len(ys) > 1 else ys[0]
    return X, y


def _compute_wavelength_indices(n_features: int, n_wavelengths: int | None) -> np.ndarray | None:
    if n_wavelengths is None or n_wavelengths <= 0 or n_wavelengths >= n_features:
        return None


    # parse helper is now defined at module level
    return np.linspace(0, n_features - 1, n_wavelengths, dtype=int)


def _parse_wavelength_names(col_names: list[str] | None) -> np.ndarray | None:
    """Parse column names for numeric wavelength values where possible.

    Returns an ndarray of floats representing wavelengths or None if parsing fails.
    """
    if not col_names:
        return None
    vals = []
    for c in col_names:
        try:
            v = float(str(c).strip().replace("wl_", "").replace("w", ""))
            vals.append(v)
        except Exception:
            # not a numeric column name: fail gracefully
            return None
    try:
        arr = np.asarray(vals, dtype=float)
        if arr.size and np.all(np.isfinite(arr)):
            return arr
    except Exception:
        pass
    return None


def resample_spectra(
    X: np.ndarray,
    col_names: list[str] | None = None,
    n_wavelengths: int | None = None,
    method: str = "interpolate",
) -> tuple[np.ndarray, np.ndarray | None]:
    """Resample / trim spectral matrix X to `n_wavelengths`.

        - If `n_wavelengths` is None or less than or equal to current width and neither numeric
            wavelengths nor indices are available, returns X unchanged.
        - If numeric wavelength values are available in `col_names`, resampling will perform
            interpolation and may upsample or downsample to `n_wavelengths`.
    - If column names contain numeric wavelength values, resample using numeric domain and interpolation.
    - Otherwise, compute evenly spaced indices across existing features and subsample.

    Returns (X_resampled, wavelengths_selected) where wavelengths_selected are the numeric wavelengths if available, else None.
    """
    if n_wavelengths is None or n_wavelengths <= 0:
        return X, None
    n_feat = X.shape[1]
    # Try to parse col_names as numeric wavelengths
    wl = _parse_wavelength_names(col_names) if col_names is not None else None
    if wl is not None and wl.size == n_feat:
        # original numeric wavelengths present: resample onto evenly spaced grid in index space
        new_wl = np.linspace(wl.min(), wl.max(), n_wavelengths)
        Xr = np.empty((X.shape[0], n_wavelengths), dtype=X.dtype)
        # Linear interpolation across columns for each row
        if method == "interpolate":
            for i in range(X.shape[0]):
                Xr[i, :] = np.interp(new_wl, wl, X[i, :])
        else:
            # Subsample indices (nearest) from original spectral values
            idx = _compute_wavelength_indices(n_feat, n_wavelengths)
            if idx is None:
                for i in range(X.shape[0]):
                    Xr[i, :] = np.interp(new_wl, wl, X[i, :])
            else:
                Xr = X[:, idx]
        return Xr, new_wl.tolist()
    # Fall back to evenly spaced indices selection (no interpolation)
    indices = _compute_wavelength_indices(n_feat, n_wavelengths)
    if indices is None:
        return X, None
    Xr = X[:, indices]
    return Xr, None


def _is_resampling_enabled(resample: bool | None) -> bool:
    """Decide whether loaders should resample spectra.

    Priority: explicit `resample` argument wins; otherwise check environment
    `FEDCHEM_RESAMPLE_SPECTRA` (1/0 booleans), defaulting to True for backward compatibility.
    """
    if resample is not None:
        return bool(resample)
    v = os.environ.get("FEDCHEM_RESAMPLE_SPECTRA")
    if v is None:
        return True
    return str(v).strip() not in {"0", "false", "False", "FALSE"}


def _get_resample_method(resample: bool | None, method: str | None) -> str | None:
    """Resolve the effective resample method name.

    Priority:
    - If `resample` is explicitly False -> return None (no resampling)
    - If `method` provided explicitly, validate and return it
    - Otherwise examine env `FEDCHEM_RESAMPLE_METHOD` or default to 'interpolate' when resampling enabled
    """
    if resample is not None and not bool(resample):
        return None
    # If a method is explicitly provided, prioritize it (validate values)
    if method is not None:
        m = str(method).strip().lower()
        if m in {"interpolate", "linear", "subsample"}:
            return "interpolate" if m in {"interpolate", "linear"} else "subsample"
        return None
    v = os.environ.get("FEDCHEM_RESAMPLE_METHOD")
    if v is not None:
        m = str(v).strip().lower()
        if m in {"interpolate", "linear", "subsample"}:
            return "interpolate" if m in {"interpolate", "linear"} else "subsample"
    # Default behavior: interpolate (when resample enabled) or None
    if _is_resampling_enabled(resample):
        return "interpolate"
    return None


def _trim_transfer_samples(
    X: np.ndarray,
    y: np.ndarray,
    keep_transfer: int | None,
    transfer_boundary: int | None,
    seed: int | None = None,
    k_value: int | None = None,
) -> tuple[np.ndarray, np.ndarray, int | None]:
    """Keep at most `keep_transfer` paired rows while preserving site-specific extras.

    If keep_transfer exceeds transfer_boundary, randomly sample from the full dataset
    using a K-dependent seed to enable different sample selection for each K value.
    This implements the experimental design where different K values get different subsets.
    """

    if keep_transfer is None:
        return X, y, transfer_boundary
    n_total = X.shape[0]
    if keep_transfer <= 0:
        return X[:0], y[:0], transfer_boundary

    # If transfer_boundary is None or >= n_total, or if keep_transfer > transfer_boundary,
    # use random sampling with K-dependent seed to select diverse subsets for different K values
    if transfer_boundary is None or transfer_boundary >= n_total or keep_transfer > transfer_boundary:
        keep_transfer = min(keep_transfer, n_total)
        # Use K-dependent seed: combine base seed with K value to get different samples per K
        if k_value is not None and seed is not None:
            effective_seed = seed + k_value  # Different K -> different seed -> different samples
        else:
            effective_seed = seed
        rng = np.random.RandomState(effective_seed)
        selected_idx = np.sort(rng.choice(n_total, size=keep_transfer, replace=False))
        return X[selected_idx], y[selected_idx], keep_transfer

    # Original behavior: take first keep_transfer rows, then append extras
    extras = X[transfer_boundary:]
    extras_y = y[transfer_boundary:]
    keep_transfer = min(keep_transfer, transfer_boundary)
    if keep_transfer <= 0:
        X_trimmed = extras.copy()
        y_trimmed = extras_y.copy()
    else:
        X_trimmed = np.vstack([X[:keep_transfer], extras]) if extras.size else X[:keep_transfer]
        y_trimmed = np.hstack([y[:keep_transfer], extras_y]) if extras_y.size else y[:keep_transfer]
    return X_trimmed, y_trimmed, keep_transfer

def load_idrc_wheat_shootout_site_dict(
    *,
    sites: list[str] | None = None,
    data_dir: str = "data",
    n_wavelengths: int | None = None,
    resample: bool | None = None,
    quick_cap_per_site: int | None = None,
    seed: int = 42,
    max_transfer_samples: int | None = None,
    data_config: dict[str, Any] | None = None,
    instrument_map: dict[str, dict[str, Any]] | None = None,
    instrument_as_site: bool = False,
    loso_skip_instrument: str | None = None,
) -> tuple[RealSiteDict, MetaDict]:
    """Load IDRC wheat shootout .xlsx files, split by manufacturer/site.

    When `instrument_as_site` is true, each worksheet (instrument) is returned
    as its own entry enriched with `instrument_id`/`backing_site` metadata.
    Set `loso_skip_instrument` to skip a specific instrument id (LOSO support).
    """
    if sites is None:
        if data_config:
            sites = list(data_config.keys())
        else:
            sites = ["ManufacturerA", "ManufacturerB"]
    data: RealSiteDict = {}
    base_dir = Path(data_dir)
    site_summaries: dict[str, dict[str, Any]] = {}
    skip_instruments = (
        {str(loso_skip_instrument)}
        if instrument_as_site and loso_skip_instrument is not None
        else set()
    )
    for site_idx, site in enumerate(sites):
        site_path = base_dir / site
        cal_path = site_path / f"Cal_{site}.xlsx"
        val_path = site_path / f"Val_{site}.xlsx"
        test_path = site_path / f"Test_{site}.xlsx"

        site_cfg = data_config.get(site, {}) if data_config else {}
        cal_limit = site_cfg.get("cal_instruments")
        val_limit = site_cfg.get("val_instruments")
        test_limit = site_cfg.get("test_instruments")

        cal_results = _load_excel_tables(
            cal_path,
            multi_sheet=True,
            n_wavelengths=n_wavelengths,
            quick_cap_per_site=quick_cap_per_site,
            max_transfer_samples=max_transfer_samples,
            base_seed=seed + site_idx * 100,
            resample=resample,
            return_sheet_names=bool(instrument_as_site),
        )
        if instrument_as_site:
            # cal_results is now list of (sheet_name, X, y)
            cal_results = [(str(name), X if X is not None else np.empty((0, 0)), y) for name, X, y in cal_results]
            # If an instrument_map (from config) was provided, filter and respect enabled flags
            if instrument_map is not None:
                cal_results = [
                    t for t in cal_results if t[0] in instrument_map and instrument_map[t[0]].get("enabled", True)
                ]
            if skip_instruments:
                cal_results = [t for t in cal_results if t[0] not in skip_instruments]
            cal_instruments_loaded = len(cal_results)
            # For per-instrument entries, we'll pipeline below after val/test sets are loaded
        else:
            cal_results = _limit_tables(cal_results, cal_limit)
            cal_X, cal_y = _combine_tables(cal_results, allow_empty=True)

        val_tables = _load_excel_tables(
            val_path,
            multi_sheet=False,
            n_wavelengths=n_wavelengths,
            quick_cap_per_site=quick_cap_per_site,
            max_transfer_samples=max_transfer_samples,
            base_seed=seed + site_idx * 1000,
            resample=resample,
            return_sheet_names=False,
        )
        val_tables = _limit_tables(val_tables, val_limit)
        val_X, val_y = _combine_tables(val_tables)

        test_tables = _load_excel_tables(
            test_path,
            multi_sheet=False,
            n_wavelengths=n_wavelengths,
            quick_cap_per_site=quick_cap_per_site,
            max_transfer_samples=max_transfer_samples,
            base_seed=seed + site_idx * 2000,
            resample=resample,
            return_sheet_names=False,
        )
        test_tables = _limit_tables(test_tables, test_limit)
        test_X, test_y = _combine_tables(test_tables)

        if instrument_as_site:
            # Build per-instrument entries using sheet_name as instrument identifier
            for sheet_name, X, y in cal_results:
                instr_id = str(sheet_name)
                if skip_instruments and instr_id in skip_instruments:
                    continue
                data[instr_id] = {
                    "cal": (X, y),
                    "val": (val_X, val_y),
                    "test": (test_X, test_y),
                    "X": X,
                    "y": y,
                    "instrument_id": instr_id,
                }
                # Add original backing site (manufacturer) if instrument map provided
                backing_site = instrument_map.get(instr_id, {}).get("site") if instrument_map is not None else site
                data[instr_id]["backing_site"] = backing_site
                site_summaries[instr_id] = {
                    "instrument_id": instr_id,
                    "backing_site": backing_site,
                    "cal_instruments_requested": 1,
                    "cal_instruments_loaded": 1,
                    "val_instruments_requested": val_limit,
                    "val_instruments_loaded": len(val_tables),
                    "test_instruments_requested": test_limit,
                    "test_instruments_loaded": len(test_tables),
                    "cal_rows": int(X.shape[0]) if isinstance(X, np.ndarray) else 0,
                    "val_rows": int(val_X.shape[0]) if isinstance(val_X, np.ndarray) else 0,
                    "test_rows": int(test_X.shape[0]) if isinstance(test_X, np.ndarray) else 0,
                }
        else:
            data[site] = {
                "cal": (cal_X, cal_y),
                "val": (val_X, val_y),
                "test": (test_X, test_y),
                "X": cal_X,
                "y": cal_y,
            }
            site_summaries[site] = {
                "cal_instruments_requested": cal_limit,
                "cal_instruments_loaded": len(cal_results),
                "val_instruments_requested": val_limit,
                "val_instruments_loaded": len(val_tables),
                "test_instruments_requested": test_limit,
                "test_instruments_loaded": len(test_tables),
                "cal_rows": int(cal_X.shape[0]) if isinstance(cal_X, np.ndarray) else 0,
                "val_rows": int(val_X.shape[0]) if isinstance(val_X, np.ndarray) else 0,
                "test_rows": int(test_X.shape[0]) if isinstance(test_X, np.ndarray) else 0,
            }
    meta_out: MetaDict = {
        "dataset": "idrc_wheat_shootout",
        "sites": sites,
        "data_dir": data_dir,
        "data_config": data_config,
        "site_summaries": site_summaries,
    }
    meta_out["instrument_ids"] = sorted(data.keys()) if instrument_as_site else None
    meta_out["loso_skip_instrument"] = str(loso_skip_instrument) if loso_skip_instrument else None
    return data, meta_out



def load_shootout_site_dict(
    *,
    n_sites: int = 3,
    seed: int = 42,
    quick: bool = False,
    quick_cap_per_site: int | None = None,
    n_wavelengths: int | None = 256,
    resample: bool | None = None,
    max_transfer_samples: int | None = None,
    config_sites: list[str] | None = None,
    data_dir: str | Path = "data/corrected2_20180719",
) -> tuple[RealSiteDict, MetaDict]:
    """Load shootout NIR dataset, split by site/experiment/instrument as per config.yaml."""
    # Read CalibrationRaw.txt
    calib_path = Path(data_dir) / "CalibrationRaw.txt"
    df = pd.read_csv(calib_path, sep="\t")
    spec_cols = [c for c in df.columns if c.startswith("w")]
    # Use config_sites to select sites/instruments if provided
    if config_sites:
        # Map config site names to shootout Site/Instrument_Type
        # Example: config_sites = ["m5_Site1", "mp5_Site2", "mp6_Site3"]
        site_map = {}
        for cs in config_sites:
            parts = cs.split("_")
            if len(parts) == 2:
                site_map[cs] = {"Instrument_Type": parts[0], "Site": parts[1]}
        # Filter df for each site
        site_dfs = []
        for cs, filt in site_map.items():
            mask = (df["Instrument_Type"] == filt["Instrument_Type"]) & (df["Site"] == filt["Site"])
            site_dfs.append((cs, df[mask].copy()))
    else:
        # Fallback: split by unique Site or Instrument_Type
        unique_sites = df["Site"].unique()
        site_dfs = [(str(s), df[df["Site"] == s].copy()) for s in unique_sites[:n_sites]]
    # For each site, extract X (spectra) and y (solute or target)
    data: RealSiteDict = {}
    wavelengths_selected = spec_cols[:n_wavelengths] if n_wavelengths is not None else spec_cols
    for name, sdf in site_dfs:
        X = sdf[wavelengths_selected].to_numpy()
        y = sdf["C_Solute"].to_numpy()  # or other target column
        # K-dependent random sampling for transfer samples
        if max_transfer_samples is not None:
            rng = np.random.RandomState(seed + max_transfer_samples)
            idx = rng.choice(len(X), size=min(max_transfer_samples, len(X)), replace=False)
            X = X[idx]
            y = y[idx]
        # Cap samples for quick mode
        quick_cap = quick_cap_per_site if quick_cap_per_site is not None else (80 if quick else None)
        if quick_cap is not None and len(X) > quick_cap:
            X = X[:quick_cap]
            y = y[:quick_cap]
        # Resample / trim wavelengths to canonical length when requested
        if n_wavelengths is not None and X.shape[1] != n_wavelengths:
            if _is_resampling_enabled(resample):
                effective_method = _get_resample_method(resample, os.environ.get("FEDCHEM_RESAMPLE_METHOD"))
                try:
                    X, wl_sel = resample_spectra(X, col_names=wavelengths_selected, n_wavelengths=n_wavelengths, method=effective_method)
                except Exception:
                    X = _trim_wavelengths(X, n_wavelengths)
                else:
                    # If resample returned numeric wavelength values, use them as metadata
                    if wl_sel is not None:
                        wavelengths_selected = [str(v) for v in wl_sel]
            else:
                X = _trim_wavelengths(X, n_wavelengths)
        data[name] = {"X": X, "y": y}
    meta_out: MetaDict = {
        "dataset": "shootout",
        "wavelengths": wavelengths_selected,
        "source_meta": {"data_dir": str(data_dir)},
        "requested_sites": n_sites,
        "actual_sites": len(data),
        "quick_cap": quick_cap_per_site,
        "n_wavelengths_requested": n_wavelengths,
        "n_wavelengths_actual": len(wavelengths_selected) if wavelengths_selected is not None else None,
        "transfer_samples_requested": max_transfer_samples,
        "site_names": list(data.keys()),
    }
    return data, meta_out

def load_real_site_dict(
    *,
    n_sites: int = 3,
    seed: int = 42,
    allow_download: bool | None = None,
    quick: bool = False,
    quick_cap_per_site: int | None = None,
    force_tecator: bool = False,
    n_wavelengths: int | None = 256,
    resample: bool | None = None,
    max_transfer_samples: int | None = None,
    config_sites: list[str] | None = None,
) -> tuple[RealSiteDict, MetaDict]:
    """Load the canonical site dataset, shootout, or Tecator fallback as a site->data dict."""
    if n_sites < 1:
        raise ValueError("n_sites must be >= 1")
    quick_cap = quick_cap_per_site if quick_cap_per_site is not None else (80 if quick else None)
    if allow_download is None:
        allow_download = os.environ.get("FEDCHEM_ALLOW_DOWNLOAD", "0") == "1"

    use_tecator_env = os.environ.get("USE_TECATOR", "false").lower() == "true"

    if not force_tecator and not use_tecator_env and load_persistent_site is not None:
        try:
            res = load_persistent_site()
        except Exception:  # pragma: no cover - defensive against optional dependency errors
            res = None
        if res is not None:
            data_raw, wavelengths, meta = res
            site_names = sorted(data_raw.keys())
            if len(site_names) < n_sites:
                raise ValueError(
                    f"Persistent site dataset has {len(site_names)} sites; cannot satisfy request for {n_sites}."
                )
            # If the caller provided a list of desired site codes (config_sites), prefer them.
            # Keep order from config_sites when possible and fall back to the persistent order
            # to fill the requested `n_sites` slots.
            if config_sites:
                # normalize config_sites into a list of present site names
                requested = [s for s in config_sites if s in site_names]
                # If insufficient sites in `requested`, fill from `site_names` while avoiding duplicates
                if len(requested) < n_sites:
                    for s in site_names:
                        if s not in requested:
                            requested.append(s)
                        if len(requested) >= n_sites:
                            break
                selected = requested[:n_sites]
                # Warn if the user requested specific sites but some were not available in
                # the persistent dataset. This helps with debugging misconfigured site codes.
                missing = [s for s in config_sites if s not in site_names]
                if missing:
                    import warnings

                    warnings.warn(
                        f"Some requested config_sites were not present in persistent dataset: {missing}. Falling back to available sites.",
                        UserWarning,
                    )
            else:
                selected = site_names[:n_sites]
            data: RealSiteDict = {}
            manifest = meta.get("manifest", {}) if isinstance(meta, dict) else {}
            raw_transfer_boundary = manifest.get("n_transfer")
            transfer_boundary = int(raw_transfer_boundary) if raw_transfer_boundary is not None else None
            available_transfer = transfer_boundary
            requested_transfer = max_transfer_samples
            wavelength_indices: np.ndarray | None = None
            wavelengths_selected: list[float] | None = None
            if wavelengths is not None:
                try:
                    wavelengths_arr = np.asarray(wavelengths)
                    wavelengths_selected = list(wavelengths_arr.flatten())
                except Exception:
                    wavelengths_selected = None
            actual_transfer_used: int | None = None
            for name in selected:
                site = data_raw[name]
                X = np.asarray(site["X"])
                y = np.asarray(site["y"])
                # If loaded wavelength metadata does not match data shape, drop it so we don't misreport metadata
                if wavelengths_selected is not None and len(wavelengths_selected) != X.shape[1]:
                    wavelengths_selected = None
                if wavelength_indices is None:
                    wavelength_indices = _compute_wavelength_indices(X.shape[1], n_wavelengths)
                    if wavelength_indices is not None and wavelengths is not None:
                        try:
                            wavelengths_selected = np.asarray(wavelengths)[wavelength_indices].tolist()
                        except Exception:
                            wavelengths_selected = list(np.asarray(wavelengths).flatten())
                # If original wavelength values are available, perform resampling to
                # `n_wavelengths` via interpolation for consistency across datasets.
                if wavelengths_selected is not None:
                    if _is_resampling_enabled(resample):
                        try:
                            effective_method = _get_resample_method(resample, os.environ.get("FEDCHEM_RESAMPLE_METHOD"))
                            X, wl_sel = resample_spectra(
                                X, col_names=wavelengths_selected, n_wavelengths=n_wavelengths, method=effective_method
                            )
                            # update wavelengths_selected for meta if wl_sel provided
                            if wl_sel is not None:
                                wavelengths_selected = list(wl_sel)
                        except Exception:
                            # Fall back to indices if resampling fails
                            if wavelength_indices is not None:
                                X = X[:, wavelength_indices]
                    else:
                        if wavelength_indices is not None:
                            X = X[:, wavelength_indices]
                elif wavelength_indices is not None:
                    X = X[:, wavelength_indices]
                X_trim, y_trim, kept_transfer = _trim_transfer_samples(
                    X,
                    y,
                    requested_transfer,
                    transfer_boundary,
                    seed=seed,
                )
                X_cap, y_cap = _cap_rows(X_trim, y_trim, quick_cap)
                data[name] = {"X": X_cap, "y": y_cap}
                if actual_transfer_used is None and kept_transfer is not None:
                    actual_transfer_used = kept_transfer
            meta_out: MetaDict = {
                "dataset": "site_persistent",
                "wavelengths": wavelengths_selected,
                "source_meta": meta,
                "requested_sites": n_sites,
                "actual_sites": len(selected),
                "quick_cap": quick_cap,
                "force_tecator": False,
                "n_wavelengths_requested": n_wavelengths,
                "n_wavelengths_actual": len(wavelengths_selected) if wavelengths_selected is not None else None,
                "transfer_samples_requested": requested_transfer,
                "transfer_samples_available": available_transfer,
                "transfer_samples_used": actual_transfer_used,
            }
            meta_out["site_names"] = selected
            return data, meta_out

    use_tecator = force_tecator or use_tecator_env
    if not use_tecator:
        return load_shootout_site_dict(
            n_sites=n_sites,
            seed=seed,
            quick=quick,
            quick_cap_per_site=quick_cap_per_site,
            n_wavelengths=n_wavelengths,
            resample=resample,
            max_transfer_samples=max_transfer_samples,
            config_sites=config_sites,
        )

    X, y, wavelengths, ds_meta = load_dataset("tecator", allow_download=allow_download)
    if y is None:
        raise ValueError("Tecator dataset returned without targets (y); cannot proceed.")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(X.shape[0])
    X = np.asarray(X[idx])
    y = np.asarray(y[idx])
    if n_sites > X.shape[0]:
        raise ValueError(
            f"Tecator dataset has {X.shape[0]} samples, which is fewer than requested sites ({n_sites})."
        )
    wavelength_indices: np.ndarray | None = None
    wavelengths_selected: list[float] | None = None
    if wavelengths is not None:
        try:
            wavelengths_arr = np.asarray(wavelengths)
            wavelengths_selected = list(wavelengths_arr.flatten())
        except Exception:
            wavelengths_selected = None
    data: RealSiteDict = {}
    splits = np.array_split(np.arange(X.shape[0]), n_sites)
    for i, part in enumerate(splits):
        Xi = np.asarray(X[part])
        yi = np.asarray(y[part])
        if wavelength_indices is None:
            wavelength_indices = _compute_wavelength_indices(Xi.shape[1], n_wavelengths)
            if wavelength_indices is not None and wavelengths is not None:
                try:
                    wavelengths_selected = np.asarray(wavelengths)[wavelength_indices].tolist()
                except Exception:
                    wavelengths_selected = list(np.asarray(wavelengths).flatten())
        if wavelength_indices is not None:
            Xi = Xi[:, wavelength_indices]
        Xi_cap, yi_cap = _cap_rows(Xi, yi, quick_cap)
        data[f"site_{i}"] = {"X": Xi_cap, "y": yi_cap}
    meta_out: MetaDict = {
        "dataset": "tecator",
        "wavelengths": wavelengths_selected,
        "source_meta": ds_meta,
        "requested_sites": n_sites,
        "actual_sites": len(data),
        "quick_cap": quick_cap,
        "force_tecator": force_tecator,
        "n_wavelengths_requested": n_wavelengths,
        "n_wavelengths_actual": len(wavelengths_selected) if wavelengths_selected is not None else None,
        "transfer_samples_requested": max_transfer_samples,
        "transfer_samples_available": None,
        "transfer_samples_used": None,
    }
    meta_out["site_names"] = sorted(data.keys())
    return data, meta_out



def load_real_site_clients(
    *,
    n_sites: int = 3,
    seed: int = 42,
    allow_download: bool | None = None,
    quick: bool = False,
    quick_cap_per_site: int | None = None,
    force_tecator: bool = False,
    n_wavelengths: int | None = 256,
    resample: bool | None = None,
    max_transfer_samples: int | None = None,
    config_sites: list[str] | None = None,
) -> tuple[list[Dict[str, np.ndarray]], MetaDict]:
    """Return a list of per-site client dictionaries suitable for FL along with metadata."""
    data, meta = load_real_site_dict(
        n_sites=n_sites,
        seed=seed,
        allow_download=allow_download,
        quick=quick,
        quick_cap_per_site=quick_cap_per_site,
        force_tecator=force_tecator,
        n_wavelengths=n_wavelengths,
        resample=resample,
        max_transfer_samples=max_transfer_samples,
        config_sites=config_sites,
    )
    clients: list[Dict[str, np.ndarray]] = []
    for name in sorted(data.keys()):
        site = data[name]
        clients.append({"X": site["X"], "y": site["y"]})
    return clients, meta
