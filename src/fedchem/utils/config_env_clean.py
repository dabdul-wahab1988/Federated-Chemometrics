"""Minimal, clean environment seed/override helpers to replace the broken module.

This is intentionally small and easy to maintain; the goal is to keep testable
behavior stable while we avoid modifying the corrupted `config_env.py` directly.
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
