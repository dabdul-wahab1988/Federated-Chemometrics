"""Centralized helper for loading `config.yaml` and deriving commonly used values.

This module provides a single `load_config()` function which returns a normalized
dictionary representing configuration from `config.yaml` and convenience helpers
for deriving common pieces (e.g. experimental site codes, per-manufacturer data config,
and the pipeline configuration).

This centralizes the YAML loading logic so scripts and components can import
and reuse the same parsing logic instead of calling `yaml.safe_load` directly.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple
import yaml
from fedchem.utils.config_env_clean import seed_env_from_config, override_config_from_env

DEFAULT_PATH = Path("config.yaml")


def _parse_sites_from_factor(factors: Dict[str, Any]) -> List[str] | None:
    if not isinstance(factors, dict):
        return None
    raw = factors.get("Site")
    if raw is None:
        return None
    if isinstance(raw, list):
        return [str(s).strip() for s in raw if str(s).strip()]
    if isinstance(raw, str):
        # allow comma-separated site codes in a single string
        parts = [s.strip() for s in raw.split(",") if s.strip()]
        return parts if parts else None
    return None


def _parse_loso_mode_value(raw_value: Any) -> Tuple[str, str | None]:
    """Normalize LOSO_MODE values and return (mode, warning)."""
    if raw_value is None:
        return "none", None
    # bool/int treated as toggle: True -> instrument LOSO, False -> none
    if isinstance(raw_value, bool):
        return ("instrument" if raw_value else "none"), None
    if isinstance(raw_value, (int, float)):
        return ("instrument" if float(raw_value) != 0.0 else "none"), None
    if isinstance(raw_value, str):
        token = raw_value.strip().lower()
        if not token:
            return "none", None
        if token in {"instrument", "instruments", "instrumental", "instrument_as_site", "instrument-los", "loso_instrument"}:
            return "instrument", None
        if token in {"site", "sites", "manufacturer", "manufacturers", "site-los"}:
            return "site", None
        if token in {"none", "off", "disable", "disabled"}:
            return "none", None
        return "none", f"Unrecognized LOSO_MODE value '{raw_value}' (expected 'instrument', 'site', or 'none')."
    return "none", f"Unsupported LOSO_MODE type {type(raw_value).__name__}; expected string, bool, or int."


def get_loso_mode(cfg: Dict[str, Any]) -> str:
    """Return normalized LOSO mode ('none', 'instrument', or 'site')."""
    if not isinstance(cfg, dict):
        return "none"
    mode, _ = _parse_loso_mode_value(cfg.get("LOSO_MODE"))
    return mode


def _collect_instrument_entries(cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Return raw instrument metadata from either INSTRUMENTS or INSTRUMENT_MAP."""
    entries: Dict[str, Dict[str, Any]] = {}
    if not isinstance(cfg, dict):
        return entries
    instruments = cfg.get("INSTRUMENTS")
    if isinstance(instruments, dict):
        for site, items in instruments.items():
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                iid = item.get("id")
                if not iid:
                    continue
                entries[str(iid)] = {
                    "site": str(site),
                    "enabled": bool(item.get("enabled", True)),
                    "role": item.get("role"),
                    **{k: v for k, v in item.items() if k not in {"id", "role", "enabled"}},
                }
    flat = cfg.get("INSTRUMENT_MAP")
    if isinstance(flat, dict):
        for iid, meta in flat.items():
            if not isinstance(meta, dict):
                continue
            site = meta.get("site") or meta.get("manufacturer") or meta.get("mfr")
            entries[str(iid)] = {
                "site": str(site) if site is not None else None,
                "enabled": bool(meta.get("enabled", True)),
                "role": meta.get("role"),
                **{k: v for k, v in meta.items() if k not in {"site", "manufacturer", "mfr", "enabled", "role"}},
            }
    return entries


def get_instruments_from_config(cfg: Dict[str, Any]) -> List[str] | None:
    """Return a list of enabled instrument IDs from INSTRUMENTS/INSTRUMENT_MAP."""
    entries = _collect_instrument_entries(cfg)
    enabled = [iid for iid, meta in entries.items() if meta.get("enabled", True)]
    return enabled if enabled else None


def get_instrument_to_site_map(cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Return mapping of instrument ID -> metadata (site/manufacturer, role, enabled).

    Supports both nested `INSTRUMENTS` blocks (manufacturer -> list of instruments)
    and flat `INSTRUMENT_MAP` dictionaries keyed by instrument id.
    """
    raw_entries = _collect_instrument_entries(cfg)
    out: Dict[str, Dict[str, Any]] = {}
    for iid, meta in raw_entries.items():
        entry = {
            "site": meta.get("site"),
            "enabled": bool(meta.get("enabled", True)),
            "role": meta.get("role"),
        }
        # Preserve any additional metadata fields for downstream consumers.
        for key, value in meta.items():
            if key not in entry:
                entry[key] = value
        out[iid] = entry
    return out


def load_config(path: str | Path = DEFAULT_PATH) -> Dict[str, Any]:
    """Load YAML config from the provided file path and return a dict.

    Falls back to an empty dict on parse errors or missing file.
    """
    p = Path(path)
    if not p.exists():
        return {}
    try:
        with p.open("r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
    except Exception:
        return {}
    return cfg if isinstance(cfg, dict) else {}


def get_experimental_sites(cfg: Dict[str, Any]) -> List[str] | None:
    """Return list of site codes derived from `EXPERIMENTAL_DESIGN.FACTORS.Site` or
   , when not present, from `DATA_CONFIG` keys (manufacturer site names).
    """
    if not isinstance(cfg, dict):
        return None
    exp_design = cfg.get("EXPERIMENTAL_DESIGN")
    factors = exp_design.get("FACTORS") if isinstance(exp_design, dict) else None
    sites = _parse_sites_from_factor(factors) if factors else None
    if sites:
        return sites
    # If instruments are enumerated and we wish to treat them as independent sites,
    # return the instrument IDs instead of manufacturer-level keys
    if cfg.get("INSTRUMENT_AS_SITE"):
        instruments = get_instruments_from_config(cfg)
        if instruments:
            return instruments
    data_cfg = cfg.get("DATA_CONFIG")
    if isinstance(data_cfg, dict) and data_cfg:
        return list(data_cfg.keys())
    return None


def get_data_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Extract DATA_CONFIG section with defaults applied."""
    if not isinstance(cfg, dict):
        return {}
    data_cfg = cfg.get("DATA_CONFIG") or {}
    
    # Apply defaults for any site missing instrument limits
    DEFAULT_INSTRUMENTS = {
        "cal_instruments": None,  # None = no limit (load all)
        "val_instruments": None,
        "test_instruments": None,
        "enabled": True,
    }
    
    normalized = {}
    for site, site_cfg in data_cfg.items():
        if not isinstance(site_cfg, dict):
            site_cfg = {}
        normalized[site] = {**DEFAULT_INSTRUMENTS, **site_cfg}
    
    return normalized


def validate_site_config(cfg: Dict[str, Any]) -> List[str]:
    """Validate that experimental sites have corresponding DATA_CONFIG entries.
    
    Returns list of warning/error messages.
    """
    warnings = []
    sites = get_experimental_sites(cfg)
    data_cfg = cfg.get("DATA_CONFIG") or {}
    instrument_map = get_instrument_to_site_map(cfg)
    instrument_as_site = bool(cfg.get("INSTRUMENT_AS_SITE"))
    enabled_instrs = {iid for iid, meta in instrument_map.items() if meta.get("enabled", True)}
    if instrument_as_site and not instrument_map:
        warnings.append("INSTRUMENT_AS_SITE is true but no instruments found in INSTRUMENTS or INSTRUMENT_MAP.")
    
    if not sites:
        warnings.append("No experimental sites defined in EXPERIMENTAL_DESIGN.FACTORS.Site or DATA_CONFIG")
        return warnings
    
    for site in sites:
        if instrument_as_site:
            instrs = enabled_instrs
            if site not in instrs:
                warnings.append(f"Instrument '{site}' referenced in EXPERIMENTAL_DESIGN.FACTORS.Site is not declared in INSTRUMENTS")
        else:
            if site not in data_cfg:
                warnings.append(f"Site '{site}' in EXPERIMENTAL_DESIGN.FACTORS.Site has no DATA_CONFIG entry")
                continue
            if not isinstance(data_cfg[site], dict):
                warnings.append(f"DATA_CONFIG['{site}'] is not a dictionary")
                continue
            site_cfg = data_cfg[site]
            # Check if disabled
            if not site_cfg.get("enabled", True):
                warnings.append(f"Site '{site}' is disabled (enabled=false) but still in experiment factors")
            
            # Warn about missing instrument limits (None is valid, but should be intentional)
            for key in ["cal_instruments", "val_instruments", "test_instruments"]:
                if key not in site_cfg:
                    warnings.append(f"Site '{site}' missing '{key}' limit (will load all available)")
    
    # Check for DATA_CONFIG or INSTRUMENTS entries not used in experiments
    if sites:
        if instrument_as_site:
            declared_instrs = enabled_instrs
            unused_instrs = declared_instrs - set(sites)
            if unused_instrs:
                warnings.append(f"INSTRUMENTS has unused instrument(s): {sorted(unused_instrs)}")
        else:
            unused_sites = set(data_cfg.keys()) - set(sites)
            if unused_sites:
                warnings.append(f"DATA_CONFIG has unused site(s): {sorted(unused_sites)}")
    
    return warnings


def get_pipeline(cfg: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(cfg, dict):
        return {}
    return cfg.get("PIPELINE") or {}


def load_and_seed_config(path: str | Path = DEFAULT_PATH, prefix: str = "FEDCHEM_") -> Dict[str, Any]:
    """Convenience helper: seed environment variables from `path` and return the loaded config.

    This mirrors common script bootstrapping, where we seed environment variables from `config.yaml`
    (without overriding existing env var) then return the parsed config dict for further use.
    """
    seed_env_from_config(path=path, prefix=prefix)
    cfg = load_config(path)
    # Apply any flattened `FEDCHEM_*` env overrides back into the config dict
    try:
        cfg = override_config_from_env(cfg, prefix=prefix)
    except Exception:
        # Best-effort: don't fail if overrides cannot be applied
        pass
    # Note: scripts may choose to merge env overrides themselves for nested
    # structures (e.g., DRIFT_AUGMENT); `seed_env_from_config` still seeds env
    # variables for traceability and child process overrides.
    # Log resolved wavelengths and resample flag to assist debugging: check env override or fall back to config
    import os
    env_name = "FEDCHEM_N_WAVELENGTHS"
    env_val = os.environ.get(env_name) or os.environ.get("FEDCHEM_DEFAULT_N_WAVELENGTHS")
    if env_val is not None:
        print(f"[config] Resolved {env_name} from environment: {env_val}")
    else:
        cfg_val = cfg.get("DEFAULT_N_WAVELENGTHS")
        print(f"[config] Using DEFAULT_N_WAVELENGTHS from config: {cfg_val}")
    # Also report if resampling is enabled via env or config
    resample_env = os.environ.get("FEDCHEM_RESAMPLE_SPECTRA")
    if resample_env is not None:
        print(f"[config] Resolved FEDCHEM_RESAMPLE_SPECTRA from environment: {resample_env}")
    else:
        print(f"[config] Using RESAMPLE_SPECTRA from config: {cfg.get('RESAMPLE_SPECTRA')}")
    # Also provide resolved resample method
    method_env = os.environ.get("FEDCHEM_RESAMPLE_METHOD")
    if method_env is not None:
        print(f"[config] Resolved FEDCHEM_RESAMPLE_METHOD from environment: {method_env}")
    else:
        print(f"[config] Using RESAMPLE_METHOD from config: {cfg.get('RESAMPLE_METHOD')}")
    return cfg


def validate_config(cfg: Dict[str, Any], strict: bool = True) -> list[str]:
    """Validate config values and return a list of warning messages.

    Detect common misconfigurations (duplicated `FEDCHEM_*` keys, inconsistent types, missing defaults)
    and return human readable warnings. When `strict` is True the presence of legacy `FEDCHEM_*` keys
    raises a ValueError; otherwise it merely appends a warning.
    """
    warnings: list[str] = []
    if not isinstance(cfg, dict):
        warnings.append("Config appears to be empty or invalid (not a mapping)")
        return warnings
    
    # Add site configuration validation
    site_warnings = validate_site_config(cfg)
    warnings.extend(site_warnings)

    # Validate LOSO_MODE
    loso_mode, loso_warning = _parse_loso_mode_value(cfg.get("LOSO_MODE"))
    if loso_warning:
        warnings.append(loso_warning)
    if loso_mode == "instrument" and not cfg.get("INSTRUMENT_AS_SITE"):
        warnings.append("LOSO_MODE='instrument' requires INSTRUMENT_AS_SITE=true to enumerate instrument clients.")
    if loso_mode == "site" and not cfg.get("DATA_CONFIG"):
        warnings.append("LOSO_MODE='site' requires DATA_CONFIG entries for sites.")
    
    # Detect legacy prefixed top-level `FEDCHEM_*` keys (no longer supported).
    # Fail by default when such keys are detected, to remove backward compatibility.
    found_legacy = [k for k in cfg.keys() if isinstance(k, str) and k.startswith("FEDCHEM_")]
    if found_legacy:
        # Prepare a helpful message and either raise (strict) or warn (non-strict)
        message = (
            f"Legacy top-level FEDCHEM_* keys detected in config: {found_legacy}."
            " Please migrate these to unprefixed keys (e.g., USE_FEDPLS, FEDPLS_METHOD) and retry."
        )
        if strict:
            raise ValueError(message)
        warnings.append(message)

    # Detect duplicate prefixed keys that may get seeded with double prefix
    for k in list(cfg.keys()):
        if k.startswith("FEDCHEM_"):
            base = k[len("FEDCHEM_"):]
            if base in cfg:
                warnings.append(
                    f"Config contains both '{k}' and '{base}'. Prefer using one key (recommend: '{base}'), to avoid duplicate env var prefixes."
                )
    # Check numeric values for defaults
    if cfg.get("DEFAULT_N_WAVELENGTHS") is None:
        warnings.append("DEFAULT_N_WAVELENGTHS is not set; model loading may yield inconsistent input shapes.")
    else:
        try:
            v = int(cfg.get("DEFAULT_N_WAVELENGTHS"))
            if v <= 0:
                warnings.append("DEFAULT_N_WAVELENGTHS must be a positive integer.")
        except Exception:
            warnings.append("DEFAULT_N_WAVELENGTHS is not an integer.")
    # Validate resample method
    if cfg.get("RESAMPLE_SPECTRA") and cfg.get("RESAMPLE_METHOD") not in {"interpolate", "subsample", None}:
        warnings.append("RESAMPLE_METHOD must be 'interpolate' or 'subsample' when RESAMPLE_SPECTRA is enabled.")

    # Validate participation/compression schedule lengths equal ROUNDS
    rounds_val = None
    try:
        rounds_val = int(cfg.get("ROUNDS")) if cfg.get("ROUNDS") is not None else None
    except Exception:
        rounds_val = None

    def _parse_schedule_length(item) -> int | None:
        """Return number of elements in a schedule representation (string, list, or scalar)."""
        if item is None:
            return None
        if isinstance(item, (list, tuple)):
            # If it's a list of strings and those strings may themselves be CSV schedule representations,
            # treat the list as either a single schedule (list of numbers) or a list of schedules (list of strings)
            if len(item) > 0 and all(isinstance(x, (int, float)) for x in item):
                return len(item)
            # If it's a list of strings, we consider it as a list of schedule options; return first one's length
            if len(item) > 0 and all(isinstance(x, str) for x in item):
                try:
                    return len([t for t in item[0].split(',') if t.strip()])
                except Exception:
                    return None
            # mixed types: try best effort
            try:
                return len(item)
            except Exception:
                return None
        if isinstance(item, str):
            try:
                parts = [t for t in item.strip().replace(';', ',').split(',') if t.strip()]
                return len(parts)
            except Exception:
                return None
        # If it's numeric, consider it a single-element schedule
        if isinstance(item, (int, float)):
            return 1
        return None

    if rounds_val is not None:
        # Check top-level schedules
        for key in ("PARTICIPATION_SCHEDULE", "COMPRESSION_SCHEDULE"):
            if key in cfg and cfg.get(key) is not None:
                sched_len = _parse_schedule_length(cfg.get(key))
                if sched_len is not None and sched_len != rounds_val:
                    message = f"Top-level {key} has length {sched_len}, expected {rounds_val} (ROUNDS)."
                    if strict:
                        raise ValueError(message)
                    warnings.append(message)
        # Check experimental-design schedule factors
        exp = cfg.get("EXPERIMENTAL_DESIGN") if isinstance(cfg.get("EXPERIMENTAL_DESIGN"), dict) else None
        if exp is not None:
            facts = exp.get("FACTORS") if isinstance(exp.get("FACTORS"), dict) else None
            if facts:
                for fld in ("Participation_Schedule", "Compression_Schedule"):
                    if fld in facts and facts.get(fld) is not None:
                        options = facts.get(fld)
                        if isinstance(options, (list, tuple)):
                            # Each option could be a schedule string; validate each
                            for opt in options:
                                opt_len = _parse_schedule_length(opt)
                                if opt_len is not None and opt_len != rounds_val:
                                    message = f"Experimental factor {fld} option '{opt}' has length {opt_len}, expected {rounds_val} (ROUNDS)."
                                    if strict:
                                        raise ValueError(message)
                                    warnings.append(message)
    return warnings
