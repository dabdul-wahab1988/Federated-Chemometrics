"""Experimental design helpers for the FedChem package."""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path
from random import shuffle
from typing import Any, Dict, Iterable, List

import logging
logger = logging.getLogger(__name__)

try:
    import pyDOE2 as _pydoe2
except ImportError:  # pragma: no cover - optional dependency
    _pydoe2 = None


def _normalize_factors(raw_factors: Any) -> Dict[str, List[Any]]:
    if not raw_factors or not isinstance(raw_factors, dict):
        return {}
    normalized: Dict[str, List[Any]] = {}
    for key, values in raw_factors.items():
        if values is None:
            continue
        if isinstance(values, Iterable) and not isinstance(values, (str, bytes)):
            normalized[key] = [x for x in values]
        else:
            normalized[key] = [values]
    return normalized


def _full_factorial(factors: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    keys = list(factors.keys())
    if not keys:
        return []
    combos: List[Dict[str, Any]] = []
    for values in product(*(factors[key] for key in keys)):
        combos.append({key: value for key, value in zip(keys, values)})
    return combos


def _randomized(factors: Dict[str, List[Any]], sample_size: int = 10) -> List[Dict[str, Any]]:
    combos = _full_factorial(factors)
    if not combos:
        return []
    shuffle(combos)
    return combos[: min(sample_size, len(combos))]


def _fractional_factorial(factors: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    combos = _full_factorial(factors)
    if not combos:
        return []
    if _pydoe2 is None:
        fallback_count = max(1, len(combos) // 2)
        logger.info(
            "pyDOE2 not installed; falling back to simplified fractional selection (half the full factorial set: %d combos of %d full combos)",
            fallback_count,
            len(combos),
        )
        return combos[:fallback_count]

    keys = list(factors.keys())
    for levels in factors.values():
        if len(levels) < 2:
            return combos[: max(1, len(combos) // 2)]

    letters = [chr(ord('a') + idx) for idx in range(len(keys))]
    alias = " ".join(letters)
    try:
        matrix = _pydoe2.fracfact(alias)
        # Compute number of rows (combinations) produced by fracfact
        try:
            num_fractional = len(matrix)
        except Exception:
            num_fractional = None
        if isinstance(num_fractional, int):
            logger.info(
                "Using pyDOE2.fracfact to generate fractional factorial design: %d fractional combos produced (from %d full combos)",
                num_fractional,
                len(combos),
            )
        else:
            logger.info("Using pyDOE2.fracfact to generate fractional factorial design")
    except ValueError:
        logger.warning("pyDOE2.fracfact raised ValueError; falling back to simplified fractional selection (half full factorial set)")
        return combos[: max(1, len(combos) // 2)]

    fractional: List[Dict[str, Any]] = []
    for row in matrix:
        mapping: Dict[str, Any] = {}
        for key, level_values, sign in zip(keys, factors.values(), row):
            mapping[key] = level_values[0] if sign < 0 else level_values[1]
        fractional.append(mapping)

    if not fractional:
        return combos[: max(1, len(combos) // 2)]
    return fractional


def _load_custom_design(design_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    custom_file = design_cfg.get('CUSTOM_FILE')
    if not custom_file:
        return []
    path = Path(custom_file)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(data, list) and all(isinstance(entry, dict) for entry in data):
        return data
    return []


def generate_experimental_design(config_obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    design_cfg = (config_obj or {}).get('EXPERIMENTAL_DESIGN') or {}
    design_type = (design_cfg.get('DESIGN_TYPE') or 'full_factorial').lower()
    factors = _normalize_factors(design_cfg.get('FACTORS'))
    if not factors:
        return []

    if design_type == 'full_factorial':
        return _full_factorial(factors)
    if design_type == 'fractional_factorial':
        return _fractional_factorial(factors)
    if design_type == 'randomized':
        return _randomized(factors)
    if design_type == 'custom':
        return _load_custom_design(design_cfg)

    return []