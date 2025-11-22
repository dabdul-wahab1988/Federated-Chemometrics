"""Helpers for creating manifest fields used across objective generators.

Provides a deterministic `design_version` resolver: prefer explicit
`DESIGN_VERSION` in the loaded config, otherwise compute a short SHA1 of
the `EXPERIMENTAL_DESIGN` block (JSON sorted) and return a compact id.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Optional


def resolve_design_version(config: Optional[Dict[str, Any]] = None) -> str:
    """Return a string identifying the experimental design.

    Strategy:
    - If `config` contains `DESIGN_VERSION`, return that value (truthy string).
    - Else, if `EXPERIMENTAL_DESIGN` found, compute SHA1 of its JSON (sorted keys)
      and return `hash-<8hex>` where <8hex> are the first 8 hex digits.
    - Otherwise return the static fallback 'v1'.

    This gives projects an explicit override while ensuring a deterministic
    identifier when users forget to set a version.
    """
    if not config or not isinstance(config, dict):
        return "v1"
    # explicit override
    dv = config.get("DESIGN_VERSION") or config.get("design_version")
    if isinstance(dv, str) and dv.strip():
        return dv.strip()
    # compute from experimental design block if present
    exp = config.get("EXPERIMENTAL_DESIGN") or config.get("experimental_design")
    if exp is None:
        return "v1"
    try:
        j = json.dumps(exp, sort_keys=True, separators=(",", ":"))
        h = hashlib.sha1(j.encode("utf-8")).hexdigest()[:8]
        return f"hash-{h}"
    except Exception:
        return "v1"


def compute_combo_id(run_config: Optional[Dict[str, Any]] = None) -> str:
    """Return a short identifier for a single experimental combo (hash of run_config).

    This is meant to unambiguously identify the exact configuration used for a run
    (DP values, transfer sizes, schedule strings, site selection, etc.).
    """
    if not run_config or not isinstance(run_config, dict):
        return "combo-unknown"
    try:
        # Sort keys to ensure deterministic ordering and compact JSON separators.
        j = json.dumps(run_config, sort_keys=True, separators=(",", ":"))
        h = hashlib.sha1(j.encode("utf-8")).hexdigest()[:8]
        return f"combo-{h}"
    except Exception:
        return "combo-unknown"
