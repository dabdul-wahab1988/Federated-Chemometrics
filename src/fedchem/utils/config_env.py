"""Shim module that exports the clean config env helpers.

This shim keeps the public module path `fedchem.utils.config_env` but delegates
implementation to `config_env_clean.py` for maintainability while the original
file was being fixed.
"""
from fedchem.utils.config_env_clean import *



 


def _flatten_items(obj: Any, parent_keys: list[str] | None = None):
    parent_keys = parent_keys or []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if not isinstance(k, str):
                continue
            yield from _flatten_items(v, parent_keys + [k])
    else:
        yield parent_keys, obj


def seed_env_from_config(path: str | Path = "config.yaml", prefix: str = "FEDCHEM_") -> None:
    p = Path(path)
    if not p.exists():
        return
    try:
        with p.open("r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
    except Exception:
        return
    if not isinstance(cfg, dict):
        return
    for top_k, top_v in cfg.items():
        if not isinstance(top_k, str):
            continue
        if isinstance(top_v, dict):
            for key_path, val in _flatten_items(top_v, parent_keys=[top_k]):
                parts = [_sanitize_key(part) for part in key_path if isinstance(part, str)]
                env_key = prefix + "_".join(parts)
                if env_key in os.environ:
                    continue
                os.environ[env_key] = _coerce_to_str(val)
        else:
            env_key = prefix + _sanitize_key(top_k)
            if env_key in os.environ:
                continue
            os.environ[env_key] = _coerce_to_str(top_v)


def _parse_env_value(v: str):
    if v is None:
        return None
    if not isinstance(v, str):
        return v
    s = v.strip()
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
        try:
            return json.loads(s)
        except Exception:
            pass
    low = s.lower()
    if low in {"true", "1", "yes", "y"}:
        return True
    if low in {"false", "0", "no", "n"}:
        return False
    try:
        if "." in s:
            return float(s)
        return int(s)
    except Exception:
        pass
    return s


def override_config_from_env(cfg: dict, prefix: str = "FEDCHEM_") -> dict:
    if not isinstance(cfg, dict):
        return cfg
    cfg_top_keys = {"".join(ch if ch.isalnum() else "_" for ch in k).upper(): k for k in cfg.keys() if isinstance(k, str)}
    pat = re.compile(rf"^{re.escape(prefix)}(?P<rest>.+)$")
    for k, v in os.environ.items():
        m = pat.match(k)
        if not m:
            continue
        rest = m.group("rest")
        rest_upper = rest.upper()
        matched = None
        for top in sorted(cfg_top_keys.keys(), key=lambda x: -len(x)):
            if rest_upper == top or rest_upper.startswith(top + "_"):
                matched = top
                break
        if matched is None:
            parts = [p for p in rest.split("_") if p]
            if not parts:
                continue
            top_key = parts[0]
            nested = parts[1:]
        else:
            top_key = cfg_top_keys[matched]
            tail = rest_upper[len(matched):].lstrip("_")
            nested = [p.lower() for p in tail.split("_") if p]
        if top_key not in cfg or not isinstance(cfg[top_key], dict):
            cfg[top_key] = {}
        cur = cfg[top_key]
        for name in nested[:-1]:
            if name not in cur or not isinstance(cur[name], dict):
                cur[name] = {}
            cur = cur[name]
        if nested:
            cur[nested[-1]] = _parse_env_value(v)
        else:
            cfg[top_key] = _parse_env_value(v)
    return cfg
"""Small utilities to export a config into environment variables and to apply
flattened environment variables back into a nested config dict.

Intended usage patterns:
 - `seed_env_from_config(path)` loads a YAML config and exports keys as
    `FEDCHEM_<UPPER_CASE_KEYS>` environment variables (non-destructively).
 - `override_config_from_env(cfg)` applies existing `FEDCHEM_*` env vars back into
    a loaded config dict (in-place) so runtime overrides are honored.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import yaml


def _coerce_to_str(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, (list, dict)):
        try:
            return json.dumps(v)
        except Exception:
            return str(v)
    return str(v)


def _sanitize_key(k: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in k).upper()


def _flatten_items(obj: Any, parent_keys: list[str] | None = None):
    parent_keys = parent_keys or []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if not isinstance(k, str):
                continue
            yield from _flatten_items(v, parent_keys + [k])
    else:
        yield parent_keys, obj


def seed_env_from_config(path: str | Path = "config.yaml", prefix: str = "FEDCHEM_") -> None:
    p = Path(path)
    if not p.exists():
        return
    try:
        with p.open("r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
    except Exception:
        return

    if not isinstance(cfg, dict):
        return

    for top_k, top_v in cfg.items():
        if not isinstance(top_k, str):
            continue
        if isinstance(top_v, dict):
            for key_path, val in _flatten_items(top_v, parent_keys=[top_k]):
                parts = [_sanitize_key(part) for part in key_path if isinstance(part, str)]
                env_key = prefix + "_".join(parts)
                if env_key in os.environ:
                    continue
                os.environ[env_key] = _coerce_to_str(val)
        else:
            env_key = prefix + _sanitize_key(top_k)
            if env_key in os.environ:
                continue
            os.environ[env_key] = _coerce_to_str(top_v)


def _parse_env_value(v: str):
    if v is None:
        return None
    if not isinstance(v, str):
        return v
    s = v.strip()
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
        try:
            return json.loads(s)
        except Exception:
            pass
    low = s.lower()
    if low in {"true", "1", "yes", "y"}:
        return True
    if low in {"false", "0", "no", "n"}:
        return False
    try:
        if "." in s:
            return float(s)
        return int(s)
    except Exception:
        pass
    return s


def override_config_from_env(cfg: dict, prefix: str = "FEDCHEM_") -> dict:
    if not isinstance(cfg, dict):
        return cfg
    cfg_top_keys = {"".join(ch if ch.isalnum() else "_" for ch in k).upper(): k for k in cfg.keys() if isinstance(k, str)}
    pat = re.compile(rf"^{re.escape(prefix)}(?P<rest>.+)$")
    for k, v in os.environ.items():
        m = pat.match(k)
        if not m:
            continue
        rest = m.group("rest")
        rest_upper = rest.upper()
        matched = None
        for top in sorted(cfg_top_keys.keys(), key=lambda x: -len(x)):
            if rest_upper == top or rest_upper.startswith(top + "_"):
                matched = top
                break
        if matched is None:
            parts = [p for p in rest.split("_") if p]
            if not parts:
                continue
            top_key = parts[0]
            nested = parts[1:]
        else:
            top_key = cfg_top_keys[matched]
            tail = rest_upper[len(matched):].lstrip("_")
            nested = [p.lower() for p in tail.split("_") if p]
        if top_key not in cfg or not isinstance(cfg[top_key], dict):
            cfg[top_key] = {}
        cur = cfg[top_key]
        for name in nested[:-1]:
            if name not in cur or not isinstance(cur[name], dict):
                cur[name] = {}
            cur = cur[name]
        if nested:
            cur[nested[-1]] = _parse_env_value(v)
        else:
            cfg[top_key] = _parse_env_value(v)
    return cfg
"""Configuration seeding and env->config override helpers.

This module provides two complementary helpers:
- `seed_env_from_config(path, prefix)`: flatten `config.yaml` keys to `FEDCHEM_*` env vars
- `override_config_from_env(cfg, prefix)`: apply flattened `FEDCHEM_*` env vars back into cfg

The env variable names follow the convention used elsewhere in the project (uppercased
keys with underscores). This module converts key names to lower/underscore form when
merging back into the config mapping.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import yaml


def _coerce_to_str(v: Any) -> str:
    """Convert a python value to a compact string suitable for an env var."""
    if v is None:
        return ""
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, (list, dict)):
        try:
            return json.dumps(v)
        except Exception:
            return str(v)
    return str(v)


def _sanitize_key(k: str) -> str:
    """Convert strings to a FEDCHEM_* friendly upper-case underscore string."""
    return "".join(ch if ch.isalnum() else "_" for ch in k).upper()


def _flatten_items(obj: Any, parent_keys: list[str] | None = None):
    """Yield (key_path_list, value) for dictionary leaf nodes.

    Lists are treated as leaf values and are serialized by callers.
    """
    parent_keys = parent_keys or []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if not isinstance(k, str):
                continue
            yield from _flatten_items(v, parent_keys + [k])
    else:
        # treat non-dict (including list) as leaf
        yield parent_keys, obj


def seed_env_from_config(path: str | Path = "config.yaml", prefix: str = "FEDCHEM_") -> None:
    """Load keys from a YAML file and export as flattened FEDCHEM_* environment variables.

    Example: DRIFT_AUGMENT.jitter_wavelength_px -> FEDCHEM_DRIFT_AUGMENT_JITTER_WAVELENGTH_PX
    Existing env vars are not overwritten.
    """
    p = Path(path)
    if not p.exists():
        return
    try:
        with p.open("r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
    except Exception:
        return

    if not isinstance(cfg, dict):
        return

    for top_k, top_v in cfg.items():
        if not isinstance(top_k, str):
            continue
        if isinstance(top_v, dict):
            for key_path, val in _flatten_items(top_v, parent_keys=[top_k]):
                parts = [_sanitize_key(part) for part in key_path if isinstance(part, str)]
                env_key = prefix + "_".join(parts)
                if env_key in os.environ:
                    continue
                os.environ[env_key] = _coerce_to_str(val)
        else:
            env_key = prefix + _sanitize_key(top_k)
            if env_key in os.environ:
                continue
            os.environ[env_key] = _coerce_to_str(top_v)


def _parse_env_value(v: str):
    """Attempt to parse an env string into a sensible python object.

    Try JSON, bool-like values, int, float, then fallback to string.
    """
    if v is None:
        return None
    if not isinstance(v, str):
        return v
    s = v.strip()
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
        try:
            return json.loads(s)
        except Exception:
            pass
    low = s.lower()
    if low in {"true", "1", "yes", "y"}:
        return True
    if low in {"false", "0", "no", "n"}:
        return False
    try:
        if "." in s:
            return float(s)
        return int(s)
    except Exception:
        pass
    return s


def override_config_from_env(cfg: dict, prefix: str = "FEDCHEM_") -> dict:
    """Apply flattened FEDCHEM_* env variables back into a nested config dict.

    This update is in-place and returns the updated cfg. Nested dicts are created
    as-needed when assignment requires them.
    """
    if not isinstance(cfg, dict):
        return cfg
    # Build upper-case sanitized map of top-level cfg keys -> original key
    cfg_top_keys = {"".join(ch if ch.isalnum() else "_" for ch in k).upper(): k for k in cfg.keys() if isinstance(k, str)}
    pat = re.compile(rf"^{re.escape(prefix)}(?P<rest>.+)$")
    for k, v in os.environ.items():
        m = pat.match(k)
        if not m:
            continue
        rest = m.group("rest")
        rest_upper = rest.upper()
        matched = None
        for top in sorted(cfg_top_keys.keys(), key=lambda x: -len(x)):
            if rest_upper == top or rest_upper.startswith(top + "_"):
                matched = top
                break
        if matched is None:
            parts = [p for p in rest.split("_") if p]
            if not parts:
                continue
            top_key = parts[0]
            nested = parts[1:]
        else:
            top_key = cfg_top_keys[matched]
            tail = rest_upper[len(matched):].lstrip("_")
            nested = [p.lower() for p in tail.split("_") if p]
        # Ensure dict exists at top_key
        if top_key not in cfg or not isinstance(cfg[top_key], dict):
            cfg[top_key] = {}
        cur = cfg[top_key]
        for name in nested[:-1]:
            if name not in cur or not isinstance(cur[name], dict):
                cur[name] = {}
            cur = cur[name]
        if nested:
            cur[nested[-1]] = _parse_env_value(v)
        else:
            cfg[top_key] = _parse_env_value(v)
    return cfg
"""Configuration seeding and env->config override helpers.

This module provides two complementary helpers:
- `seed_env_from_config(path, prefix)`: flatten `config.yaml` keys to `FEDCHEM_*` env vars
- `override_config_from_env(cfg, prefix)`: apply flattened `FEDCHEM_*` env vars back into cfg

The env variable names follow the convention used elsewhere in the project (uppercased
keys with underscores). This module converts key names to lower/underscore form when
merging back into the config mapping.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import yaml


def _coerce_to_str(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, (list, dict)):
        try:
            return json.dumps(v)
        except Exception:
            return str(v)
    return str(v)


def _sanitize_key(k: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in k).upper()


def _flatten_items(obj: Any, parent_keys: list[str] | None = None):
    parent_keys = parent_keys or []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if not isinstance(k, str):
                continue
            yield from _flatten_items(v, parent_keys + [k])
    else:
        yield parent_keys, obj


def seed_env_from_config(path: str | Path = "config.yaml", prefix: str = "FEDCHEM_") -> None:
    p = Path(path)
    if not p.exists():
        return
    try:
        with p.open("r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
    except Exception:
        return

    if not isinstance(cfg, dict):
        return

    for top_k, top_v in cfg.items():
        if not isinstance(top_k, str):
            continue
        if isinstance(top_v, dict):
            for key_path, val in _flatten_items(top_v, parent_keys=[top_k]):
                parts = [_sanitize_key(part) for part in key_path if isinstance(part, str)]
                env_key = prefix + "_".join(parts)
                if env_key in os.environ:
                    continue
                os.environ[env_key] = _coerce_to_str(val)
        else:
            env_key = prefix + _sanitize_key(top_k)
            if env_key in os.environ:
                continue
            os.environ[env_key] = _coerce_to_str(top_v)


def _parse_env_value(v: str):
    if v is None:
        return None
    if not isinstance(v, str):
        return v
    s = v.strip()
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
        try:
            return json.loads(s)
        except Exception:
            pass
    low = s.lower()
    if low in {"true", "1", "yes", "y"}:
        return True
    if low in {"false", "0", "no", "n"}:
        return False
    try:
        if "." in s:
            return float(s)
        return int(s)
    except Exception:
        pass
    return s


def override_config_from_env(cfg: dict, prefix: str = "FEDCHEM_") -> dict:
    """Apply FEDCHEM_* env variables back into a config dict.

    This handles nested keys like `DRIFT_AUGMENT_JITTER_WAVELENGTH_PX` -> cfg['DRIFT_AUGMENT']['jitter_wavelength_px']
    It is best-effort and non-destructive (it creates nested dict blocks when needed).
    """
    if not isinstance(cfg, dict):
        return cfg
    # Build map of sanitized top-level config keys
    cfg_top_keys = {"".join(ch if ch.isalnum() else "_" for ch in k).upper(): k for k in cfg.keys() if isinstance(k, str)}
    pat = re.compile(rf"^{re.escape(prefix)}(?P<rest>.+)$")
    for k, v in os.environ.items():
        m = pat.match(k)
        if not m:
            continue
        rest = m.group("rest")
        rest_upper = rest.upper()
        matched = None
        for top in sorted(cfg_top_keys.keys(), key=lambda x: -len(x)):
            if rest_upper == top or rest_upper.startswith(top + "_"):
                matched = top
                break
        if matched is None:
            # fallback: first token
            parts = [p for p in rest.split("_") if p]
            if not parts:
                continue
            top_key = parts[0]
            nested = parts[1:]
        else:
            top_key = cfg_top_keys[matched]
            tail = rest_upper[len(matched):].lstrip("_")
            nested = [p.lower() for p in tail.split("_") if p]
        # Ensure dict exists
        if top_key not in cfg or not isinstance(cfg[top_key], dict):
            cfg[top_key] = {}
        # Walk nested
        cur = cfg[top_key]
        for name in nested[:-1]:
            if name not in cur or not isinstance(cur[name], dict):
                cur[name] = {}
            cur = cur[name]
        if nested:
            cur[nested[-1]] = _parse_env_value(v)
        else:
            # No nested, set top-level
            cfg[top_key] = _parse_env_value(v)
    return cfg
"""Helpers to seed environment variables from `config.yaml`.

This module provides `seed_env_from_config()` which loads a YAML
config and exports top-level keys as `FEDCHEM_<KEY>` environment
variables when not already set. Values are coerced to simple strings.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def _coerce_to_str(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, (list, dict)):
        # store compact JSON-like repr for traceability
        try:
            import json

            return json.dumps(v)
        except Exception:
            return str(v)
    return str(v)


def _sanitize_key(k: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in k).upper()


def _flatten_items(obj: Any, parent_keys: list[str] | None = None):
    """Yield (key_path_list, value) for non-dict/list leaves.

    For lists, treat the whole list as a leaf value (serialize to JSON).
    For dicts, recurse into items.
    """
    parent_keys = parent_keys or []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if not isinstance(k, str):
                continue
            yield from _flatten_items(v, parent_keys + [k])
    else:
        # treat non-dict (including list) as leaf
        yield parent_keys, obj


def seed_env_from_config(path: str | Path = "config.yaml", prefix: str = "FEDCHEM_") -> None:
    """Load keys from `path` and export flattened env vars.

    Nested dict keys become joined with underscores. Example:
      PIPELINE: { model: 'PLSModel' }
    becomes env var `FEDCHEM_PIPELINE_MODEL=PLSModel`.

    Existing environment variables are not overwritten.
    """
    p = Path(path)
    if not p.exists():
        return
    try:
        with p.open("r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
    except Exception:
        return

    if not isinstance(cfg, dict):
        return

    for top_k, top_v in cfg.items():
        if not isinstance(top_k, str):
            continue
        # If top_v is a dict, flatten; else treat as single value
        if isinstance(top_v, dict):
            for key_path, val in _flatten_items(top_v, parent_keys=[top_k]):
                # key_path is list like ["PIPELINE","model"]
                parts = [_sanitize_key(part) for part in key_path if isinstance(part, str)]
                env_key = prefix + "_".join(parts)
                if env_key in os.environ:
                    continue
                os.environ[env_key] = _coerce_to_str(val)
        else:
            env_key = prefix + _sanitize_key(top_k)
            if env_key in os.environ:
                continue
            os.environ[env_key] = _coerce_to_str(top_v)
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml


def _coerce_to_str(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, (list, dict)):
        try:
            return json.dumps(v)
        except Exception:
            return str(v)
    return str(v)


def _sanitize_key(k: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in k).upper()


def _flatten_items(obj: Any, parent_keys: list[str] | None = None):
    parent_keys = parent_keys or []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if not isinstance(k, str):
                continue
            yield from _flatten_items(v, parent_keys + [k])
    else:
        yield parent_keys, obj


def seed_env_from_config(path: str | Path = "config.yaml", prefix: str = "FEDCHEM_") -> None:
    p = Path(path)
    if not p.exists():
        return
    try:
        with p.open("r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
    except Exception:
        return

    if not isinstance(cfg, dict):
        return

    for top_k, top_v in cfg.items():
        if not isinstance(top_k, str):
            continue
        if isinstance(top_v, dict):
            for key_path, val in _flatten_items(top_v, parent_keys=[top_k]):
                parts = [_sanitize_key(part) for part in key_path if isinstance(part, str)]
                env_key = prefix + "_".join(parts)
                if env_key in os.environ:
                    continue
                os.environ[env_key] = _coerce_to_str(val)
        else:
            env_key = prefix + _sanitize_key(top_k)
            if env_key in os.environ:
                continue
            os.environ[env_key] = _coerce_to_str(top_v)
"""Helpers to seed and override environment variables from `config.yaml`.

Provides simple utilities:
- seed_env_from_config(config_path, prefix='FEDCHEM_'): export config keys to env vars
- override_config_from_env(cfg, prefix='FEDCHEM_'): update config dict from env overrides

This keeps the seeding and override logic symmetric by using flattened env
variables with uppercased tokens joined by underscores.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import yaml


def _coerce_to_str(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, (list, dict)):
        try:
            return json.dumps(v)
        except Exception:
            return str(v)
    return str(v)


def _sanitize_key(k: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in k).upper()


def _flatten_items(obj: Any, parent_keys: list[str] | None = None):
    parent_keys = parent_keys or []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if not isinstance(k, str):
                continue
            yield from _flatten_items(v, parent_keys + [k])
    else:
        yield parent_keys, obj


def seed_env_from_config(path: str | Path = "config.yaml", prefix: str = "FEDCHEM_") -> None:
    p = Path(path)
    if not p.exists():
        return
    try:
        with p.open("r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
    except Exception:
        return

    if not isinstance(cfg, dict):
        return

    for top_k, top_v in cfg.items():
        if not isinstance(top_k, str):
            continue
        if isinstance(top_v, dict):
            for key_path, val in _flatten_items(top_v, parent_keys=[top_k]):
                parts = [_sanitize_key(part) for part in key_path if isinstance(part, str)]
                env_key = prefix + "_".join(parts)
                if env_key in os.environ:
                    continue
                os.environ[env_key] = _coerce_to_str(val)
        else:
            env_key = prefix + _sanitize_key(top_k)
            if env_key in os.environ:
                continue
            os.environ[env_key] = _coerce_to_str(top_v)


def _parse_env_value(v: str):
    if v is None:
        return None
    if not isinstance(v, str):
        return v
    s = v.strip()
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
        try:
            return json.loads(s)
        except Exception:
            pass
    low = s.lower()
    if low in {"true", "1", "yes", "y"}:
        return True
    if low in {"false", "0", "no", "n"}:
        return False
    try:
        if "." in s:
            return float(s)
        return int(s)
    except Exception:
        pass
    return s


def override_config_from_env(cfg: dict, prefix: str = "FEDCHEM_") -> dict:
    if not isinstance(cfg, dict):
        return cfg
    pat = re.compile(rf"^{re.escape(prefix)}(?P<rest>.+)$")
    cfg_top_keys = {}
    for k in list(cfg.keys()):
        if not isinstance(k, str):
            continue
        sanitized = "".join(ch if ch.isalnum() else "_" for ch in k).upper()
        cfg_top_keys[sanitized] = k
    for k, v in os.environ.items():
        m = pat.match(k)
        if not m:
            continue
        rest = m.group("rest")
        rest_upper = rest.upper()
        matched_top = None
        for top_key in sorted(cfg_top_keys.keys(), key=lambda x: -len(x)):
            if rest_upper.startswith(top_key + "_") or rest_upper == top_key:
                matched_top = top_key
                break
        if matched_top is None:
            parts = [p for p in rest.split("_") if p]
            if not parts:
                continue
            top = parts[0]
            nested = parts[1:]
        else:
            top = cfg_top_keys[matched_top]
            tail = rest_upper[len(matched_top):].lstrip("_")
            nested = [p.lower() for p in tail.split("_") if p]
        key_name = top
        if key_name not in cfg or not isinstance(cfg[key_name], dict):
            cfg[key_name] = {}
        current = cfg[key_name]
        for sub_k in nested[:-1]:
            if sub_k not in current or not isinstance(current[sub_k], dict):
                current[sub_k] = {}
            current = current[sub_k]
        if nested:
            last = nested[-1]
            current[last] = _parse_env_value(v)
        else:
            cfg[key_name] = _parse_env_value(v)
    return cfg
"""Helpers to seed and override environment variables from `config.yaml`.

Provides:
- seed_env_from_config(config_path, prefix='FEDCHEM_'): export config keys to env vars
- override_config_from_env(cfg, prefix='FEDCHEM_'): update config dict from env overrides

The module uses flattened env var representations (upper-cased, underscores) to map
between nested config keys and env variable names. For example:
`DRIFT_AUGMENT.jitter_wavelength_px` -> `FEDCHEM_DRIFT_AUGMENT_JITTER_WAVELENGTH_PX`.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import yaml


def _coerce_to_str(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, (list, dict)):
        try:
            return json.dumps(v)
        except Exception:
            return str(v)
    return str(v)


def _sanitize_key(k: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in k).upper()


def _flatten_items(obj: Any, parent_keys: list[str] | None = None):
    parent_keys = parent_keys or []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if not isinstance(k, str):
                continue
            yield from _flatten_items(v, parent_keys + [k])
    else:
        yield parent_keys, obj


def seed_env_from_config(path: str | Path = "config.yaml", prefix: str = "FEDCHEM_") -> None:
    p = Path(path)
    if not p.exists():
        return
    try:
        with p.open("r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
    except Exception:
        return

    if not isinstance(cfg, dict):
        return

    for top_k, top_v in cfg.items():
        if not isinstance(top_k, str):
            continue
        if isinstance(top_v, dict):
            for key_path, val in _flatten_items(top_v, parent_keys=[top_k]):
                parts = [_sanitize_key(part) for part in key_path if isinstance(part, str)]
                env_key = prefix + "_".join(parts)
                if env_key in os.environ:
                    continue
                os.environ[env_key] = _coerce_to_str(val)
        else:
            env_key = prefix + _sanitize_key(top_k)
            if env_key in os.environ:
                continue
            os.environ[env_key] = _coerce_to_str(top_v)


def _parse_env_value(v: str):
    if v is None:
        return None
    if not isinstance(v, str):
        return v
    s = v.strip()
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
        try:
            return json.loads(s)
        except Exception:
            pass
    low = s.lower()
    if low in {"true", "1", "yes", "y"}:
        return True
    if low in {"false", "0", "no", "n"}:
        return False
    try:
        if "." in s:
            return float(s)
        return int(s)
    except Exception:
        pass
    return s


def override_config_from_env(cfg: dict, prefix: str = "FEDCHEM_") -> dict:
    if not isinstance(cfg, dict):
        return cfg
    pat = re.compile(rf"^{re.escape(prefix)}(?P<rest>.+)$")
    cfg_top_keys = {}
    for k in list(cfg.keys()):
        if not isinstance(k, str):
            continue
        sanitized = "".join(ch if ch.isalnum() else "_" for ch in k).upper()
        cfg_top_keys[sanitized] = k
    for k, v in os.environ.items():
        m = pat.match(k)
        if not m:
            continue
        rest = m.group("rest")
        rest_upper = rest.upper()
        matched_top = None
        for top_key in sorted(cfg_top_keys.keys(), key=lambda x: -len(x)):
            if rest_upper.startswith(top_key + "_") or rest_upper == top_key:
                matched_top = top_key
                break
        if matched_top is None:
            parts = [p for p in rest.split("_") if p]
            if not parts:
                continue
            top = parts[0]
            nested = parts[1:]
        else:
            top = cfg_top_keys[matched_top]
            tail = rest_upper[len(matched_top):].lstrip("_")
            nested = [p.lower() for p in tail.split("_") if p]
        key_name = top
        if key_name not in cfg or not isinstance(cfg[key_name], dict):
            cfg[key_name] = {}
        current = cfg[key_name]
        for sub_k in nested[:-1]:
            if sub_k not in current or not isinstance(current[sub_k], dict):
                current[sub_k] = {}
            current = current[sub_k]
        if nested:
            last = nested[-1]
            current[last] = _parse_env_value(v)
        else:
            cfg[key_name] = _parse_env_value(v)
    return cfg
"""Helpers to seed and override environment variables from `config.yaml`.

Provides:
- seed_env_from_config(config_path, prefix='FEDCHEM_'): export config keys to env vars
- override_config_from_env(cfg, prefix='FEDCHEM_'): update config dict from env overrides

These functions use a flattened env var representation for nested keys that mirrors
the seeding function: top-level keys become `FEDCHEM_<TOP>` and nested keys become
`FEDCHEM_<TOP>_<SUB>_...`.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import yaml


def _coerce_to_str(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, (list, dict)):
        try:
            return json.dumps(v)
        except Exception:
            return str(v)
    return str(v)


def _sanitize_key(k: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in k).upper()


def _flatten_items(obj: Any, parent_keys: list[str] | None = None):
    parent_keys = parent_keys or []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if not isinstance(k, str):
                continue
            yield from _flatten_items(v, parent_keys + [k])
    else:
        yield parent_keys, obj


def seed_env_from_config(path: str | Path = "config.yaml", prefix: str = "FEDCHEM_") -> None:
    p = Path(path)
    if not p.exists():
        return
    try:
        with p.open("r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
    except Exception:
        return

    if not isinstance(cfg, dict):
        return

    for top_k, top_v in cfg.items():
        if not isinstance(top_k, str):
            continue
        if isinstance(top_v, dict):
            for key_path, val in _flatten_items(top_v, parent_keys=[top_k]):
                parts = [_sanitize_key(part) for part in key_path if isinstance(part, str)]
                env_key = prefix + "_".join(parts)
                if env_key in os.environ:
                    continue
                os.environ[env_key] = _coerce_to_str(val)
        else:
            env_key = prefix + _sanitize_key(top_k)
            if env_key in os.environ:
                continue
            os.environ[env_key] = _coerce_to_str(top_v)


def _parse_env_value(v: str):
    if v is None:
        return None
    if not isinstance(v, str):
        return v
    s = v.strip()
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
        try:
            return json.loads(s)
        except Exception:
            pass
    low = s.lower()
    if low in {"true", "1", "yes", "y"}:
        return True
    if low in {"false", "0", "no", "n"}:
        return False
    try:
        if "." in s:
            return float(s)
        return int(s)
    except Exception:
        pass
    return s


def override_config_from_env(cfg: dict, prefix: str = "FEDCHEM_") -> dict:
    if not isinstance(cfg, dict):
        return cfg
    pat = re.compile(rf"^{re.escape(prefix)}(?P<rest>.+)$")
    cfg_top_keys = {}
    for k in list(cfg.keys()):
        if not isinstance(k, str):
            continue
        sanitized = "".join(ch if ch.isalnum() else "_" for ch in k).upper()
        cfg_top_keys[sanitized] = k
    for k, v in os.environ.items():
        m = pat.match(k)
        if not m:
            continue
        rest = m.group("rest")
        rest_upper = rest.upper()
        matched_top = None
        for top_key in sorted(cfg_top_keys.keys(), key=lambda x: -len(x)):
            if rest_upper.startswith(top_key + "_") or rest_upper == top_key:
                matched_top = top_key
                break
        if matched_top is None:
            parts = [p for p in rest.split("_") if p]
            if not parts:
                continue
            top = parts[0]
            nested = parts[1:]
        else:
            top = cfg_top_keys[matched_top]
            tail = rest_upper[len(matched_top):].lstrip("_")
            nested = [p.lower() for p in tail.split("_") if p]
        key_name = top
        if key_name not in cfg or not isinstance(cfg[key_name], dict):
            cfg[key_name] = {}
        current = cfg[key_name]
        for sub_k in nested[:-1]:
            if sub_k not in current or not isinstance(current[sub_k], dict):
                current[sub_k] = {}
            current = current[sub_k]
        if nested:
            last = nested[-1]
            current[last] = _parse_env_value(v)
        else:
            cfg[key_name] = _parse_env_value(v)
    return cfg
"""Helpers to seed environment variables from `config.yaml`.

This module provides `seed_env_from_config()` which loads a YAML
config and exports top-level keys as `FEDCHEM_<KEY>` environment
variables when not already set. Values are coerced to simple strings.

It also provides `override_config_from_env()` to apply flattened
`FEDCHEM_*` environment variables back into a loaded config dictionary,
allowing runtime overrides for nested config blocks (e.g., DRIFT_AUGMENT).
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import yaml


def _coerce_to_str(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, (list, dict)):
        # store compact JSON-like repr for traceability
        try:
            return json.dumps(v)
        except Exception:
            return str(v)
    return str(v)


def _sanitize_key(k: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in k).upper()


def _flatten_items(obj: Any, parent_keys: list[str] | None = None):
    """Yield (key_path_list, value) for non-dict/list leaves.

    For lists, treat the whole list as a leaf value (serialize to JSON).
    For dicts, recurse into items.
    """
    parent_keys = parent_keys or []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if not isinstance(k, str):
                continue
            yield from _flatten_items(v, parent_keys + [k])
    else:
        # treat non-dict (including list) as leaf
        yield parent_keys, obj


def seed_env_from_config(path: str | Path = "config.yaml", prefix: str = "FEDCHEM_") -> None:
    """Load keys from `path` and export flattened env vars.

    Nested dict keys become joined with underscores. Example:
      PIPELINE: { model: 'PLSModel' }
    becomes env var `FEDCHEM_PIPELINE_MODEL=PLSModel`.

    Existing environment variables are not overwritten.
    """
    p = Path(path)
    if not p.exists():
        return
    try:
        with p.open("r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
    except Exception:
        return

    if not isinstance(cfg, dict):
        return

    for top_k, top_v in cfg.items():
        if not isinstance(top_k, str):
            continue
        # If top_v is a dict, flatten; else treat as single value
        if isinstance(top_v, dict):
            for key_path, val in _flatten_items(top_v, parent_keys=[top_k]):
                # key_path is list like ["PIPELINE","model"]
                parts = [_sanitize_key(part) for part in key_path if isinstance(part, str)]
                env_key = prefix + "_".join(parts)
                if env_key in os.environ:
                    continue
                os.environ[env_key] = _coerce_to_str(val)
        else:
            env_key = prefix + _sanitize_key(top_k)
            if env_key in os.environ:
                continue
            os.environ[env_key] = _coerce_to_str(top_v)


def _parse_env_value(v: str):
    """Attempt to parse an environment value into a sensible Python object.

    Order of attempts: JSON, bool-like, int, float, fallback to string.
    """
    if v is None:
        return None
    if not isinstance(v, str):
        return v
    s = v.strip()
    # Try JSON decode for lists/objects
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
        try:
            return json.loads(s)
        except Exception:
            pass
    # Bool-like values
    low = s.lower()
    if low in {"true", "1", "yes", "y"}:
        return True
    if low in {"false", "0", "no", "n"}:
        return False
    # Try numeric
    try:
        if "." in s:
            return float(s)
        return int(s)
    except Exception:
        pass
    return s


def override_config_from_env(cfg: dict, prefix: str = "FEDCHEM_") -> dict:
    """Apply applicable FEDCHEM_* environment variables to the provided config dict.

    This function recognizes flattened env keys like `FEDCHEM_DRIFT_AUGMENT_JITTER_WAVELENGTH_PX`
    and updates nested keys in `cfg` accordingly, converting key names to lower snake case.
    """
    if not isinstance(cfg, dict):
        return cfg
    # Regex to match prefix and capture the remainder
    pat = re.compile(rf"^{re.escape(prefix)}(?P<rest>.+)$")
    # Build sanitized map of top-level cfg keys for prefix resolution (upper+underscores)
    # Example: 'DRIFT_AUGMENT' -> 'DRIFT_AUGMENT'
    cfg_top_keys = {}
    for k in list(cfg.keys()):
        if not isinstance(k, str):
            continue
        sanitized = "".join(ch if ch.isalnum() else "_" for ch in k).upper()
        cfg_top_keys[sanitized] = k
    for k, v in os.environ.items():
        m = pat.match(k)
        if not m:
            continue
        rest = m.group("rest")
        # We expect rest to be like 'DRIFT_AUGMENT_JITTER_WAVELENGTH_PX' or 'PIPELINE_MODEL'
        # Find the longest top-level key match in the rest string
        rest_upper = rest.upper()
        # Find the longest matching prefix key
        matched_top = None
        for top_key in sorted(cfg_top_keys.keys(), key=lambda x: -len(x)):
            if rest_upper.startswith(top_key + "_") or rest_upper == top_key:
                matched_top = top_key
                break
        if matched_top is None:
            # Fallback: take the first token as top-level
            parts = [p for p in rest.split("_") if p]
            if not parts:
                continue
            top = parts[0]
            nested = parts[1:]
        else:
            # slice the rest after the matched top and treat remainder as nested keys
            top = cfg_top_keys[matched_top]
            tail = rest_upper[len(matched_top):].lstrip("_")
            nested = [p.lower() for p in tail.split("_") if p]
        key_name = top
        # Ensure top-level dict exists
        if key_name not in cfg or not isinstance(cfg[key_name], dict):
            # If top-level non-dict exists, override with a dict
            cfg[key_name] = {}
        current = cfg[key_name]
        # Walk nested keys except last
        for sub_k in nested[:-1]:
            if sub_k not in current or not isinstance(current[sub_k], dict):
                current[sub_k] = {}
            current = current[sub_k]
        if nested:
            last = nested[-1]
            current[last] = _parse_env_value(v)
        else:
            # No nested keys; set top-level key to parsed value
            cfg[key_name] = _parse_env_value(v)
    return cfg
"""Helpers to seed environment variables from `config.yaml`.

This module provides `seed_env_from_config()` which loads a YAML
config and exports top-level keys as `FEDCHEM_<KEY>` environment
variables when not already set. Values are coerced to simple strings.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def _coerce_to_str(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, (list, dict)):
        # store compact JSON-like repr for traceability
        try:
            import json

            return json.dumps(v)
        except Exception:
            return str(v)
    return str(v)


def _sanitize_key(k: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in k).upper()


def _flatten_items(obj: Any, parent_keys: list[str] | None = None):
    """Yield (key_path_list, value) for non-dict/list leaves.

    For lists, treat the whole list as a leaf value (serialize to JSON).
    For dicts, recurse into items.
    """
    parent_keys = parent_keys or []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if not isinstance(k, str):
                continue
            yield from _flatten_items(v, parent_keys + [k])
    else:
        # treat non-dict (including list) as leaf
        yield parent_keys, obj


def seed_env_from_config(path: str | Path = "config.yaml", prefix: str = "FEDCHEM_") -> None:
    """Load keys from `path` and export flattened env vars.

    Nested dict keys become joined with underscores. Example:
      PIPELINE: { model: 'PLSModel' }
    becomes env var `FEDCHEM_PIPELINE_MODEL=PLSModel`.

    Existing environment variables are not overwritten.
    """
    p = Path(path)
    if not p.exists():
        return
    try:
        with p.open("r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
    except Exception:
        return

    if not isinstance(cfg, dict):
        return

    for top_k, top_v in cfg.items():
        if not isinstance(top_k, str):
            continue
        # If top_v is a dict, flatten; else treat as single value
        if isinstance(top_v, dict):
            for key_path, val in _flatten_items(top_v, parent_keys=[top_k]):
                # key_path is list like ["PIPELINE","model"]
                parts = [_sanitize_key(part) for part in key_path if isinstance(part, str)]
                env_key = prefix + "_".join(parts)
                if env_key in os.environ:
                    continue
                os.environ[env_key] = _coerce_to_str(val)

def _parse_env_value(v: str):
    """Attempt to parse an environment value into a sensible Python object.

    Order of attempts: JSON, bool-like, int, float, fallback to string.
    """
    if v is None:
        return None
    if not isinstance(v, str):
        return v
    s = v.strip()
    # Try JSON decode for lists/objects
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
        try:
            import json

            return json.loads(s)
        except Exception:
            pass
    # Bool-like values
    low = s.lower()
    if low in {"true", "1", "yes", "y"}:
        return True
    if low in {"false", "0", "no", "n"}:
        return False
    # Try numeric
    try:
        if "." in s:
            return float(s)
        return int(s)
    except Exception:
        pass
    return s


    def override_config_from_env(cfg: dict, prefix: str = "FEDCHEM_") -> dict:
        """Apply applicable FEDCHEM_* environment variables to the provided config dict.

        This function recognizes flattened env keys like `FEDCHEM_DRIFT_AUGMENT_JITTER_WAVELENGTH_PX`
        and updates nested keys in `cfg` accordingly, converting key names to lower snake case.
        """
    import os
    import re
        if not isinstance(cfg, dict):
            return cfg
        # Regex to match prefix and capture the remainder
        pat = re.compile(rf"^{re.escape(prefix)}(?P<rest>.+)$")
        # Build sanitized map of top-level cfg keys for prefix resolution (upper+underscores)
        # Example: 'DRIFT_AUGMENT' -> 'DRIFT_AUGMENT'
        cfg_top_keys = {}
        for k in list(cfg.keys()):
            if not isinstance(k, str):
                continue
            sanitized = "".join(ch if ch.isalnum() else "_" for ch in k).upper()
            cfg_top_keys[sanitized] = k
    for k, v in os.environ.items():
            m = pat.match(k)
            if not m:
                continue
            rest = m.group("rest")
            # We expect rest to be like 'DRIFT_AUGMENT_JITTER_WAVELENGTH_PX' or 'PIPELINE_MODEL'
            # Find the longest top-level key match in the rest string
            rest_upper = rest.upper()
            # Find the longest matching prefix key
            matched_top = None
            for top_key in sorted(cfg_top_keys.keys(), key=lambda x: -len(x)):
                if rest_upper.startswith(top_key + "_") or rest_upper == top_key:
                    matched_top = top_key
                    break
            if matched_top is None:
                # Fallback: take the first token as top-level
                parts = [p for p in rest.split("_") if p]
                if not parts:
                    continue
                top = parts[0]
                nested = parts[1:]
            else:
                # slice the rest after the matched top and treat remainder as nested keys
                top = cfg_top_keys[matched_top]
                tail = rest_upper[len(matched_top):].lstrip("_")
                nested = [p.lower() for p in tail.split("_") if p]
            key_name = top
            # Ensure top-level dict exists
            if key_name not in cfg or not isinstance(cfg[key_name], dict):
                # If top-level non-dict exists, override with a dict
                cfg[key_name] = {}
            current = cfg[key_name]
            # Walk nested keys except last
            for sub_k in nested[:-1]:
                if sub_k not in current or not isinstance(current[sub_k], dict):
                    current[sub_k] = {}
                current = current[sub_k]
            if nested:
                last = nested[-1]
                current[last] = _parse_env_value(v)
            else:
                # No nested keys; set top-level key to parsed value
                cfg[key_name] = _parse_env_value(v)
        return cfg
