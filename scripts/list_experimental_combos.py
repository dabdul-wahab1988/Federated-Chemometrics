#!/usr/bin/env python3
"""Dry-run: list experimental-design combos derived from `config.yaml`.
This mirrors the enumeration logic in `scripts/run_all_objectives.py` but only prints combos.
"""
import sys
from pathlib import Path
import random
try:
    import fedchem  # noqa: E402
except Exception:
    _proj_root = Path(__file__).resolve().parents[1]
    _src_dir = _proj_root / "src"
    if _src_dir.exists():
        sys.path.insert(0, str(_src_dir))
from fedchem.utils.config import load_config
import itertools
import argparse
import math
from pathlib import Path
import json

cfg = load_config()
if not cfg:
    print("config.yaml not found or empty; exiting")
    raise SystemExit(1)

design = (cfg.get('EXPERIMENTAL_DESIGN') or {}) if isinstance(cfg.get('EXPERIMENTAL_DESIGN'), dict) else {}
factors = design.get('FACTORS', {}) if isinstance(design.get('FACTORS', {}), dict) else {}

# Enumerate all factor keys, including schedule-like complex factors
enum_keys = list(factors.keys())

def _factor_values(v):
    if isinstance(v, (list, tuple)):
        return list(v)
    return [v]

value_lists = [_factor_values(factors.get(k)) for k in enum_keys]

# Quick CLI for limiting printed combos so large enumerations won't explode your terminal
parser = argparse.ArgumentParser(description="List experimental-design combos from config.yaml")
parser.add_argument("--max-combos", dest="max_combos", type=int, help="Maximum combos to print (default: 5000, set 0 for unlimited)")
parser.add_argument("--sample-combos", dest="sample_combos", type=int, help="If provided, sample N combos (memory-safe reservoir sampling)")
ns, _ = parser.parse_known_args()
DEFAULT_MAX = 5000

if not value_lists:
    print("No factors to enumerate; single default combo")
    combos = [()]
else:
    combo_counts = [len(vals) for vals in value_lists]
    num_combos = math.prod(combo_counts) if combo_counts else 1
    if ns.sample_combos:
        max_allowed = max(0, ns.max_combos or DEFAULT_MAX)
    else:
        if ns.max_combos is None:
            max_allowed = DEFAULT_MAX
        else:
            max_allowed = ns.max_combos
        if max_allowed != 0 and num_combos > max_allowed:
            print(f"Refusing to print {num_combos} combos (over --max-combos/{DEFAULT_MAX}={max_allowed}). Specify `--max-combos {num_combos}` to override or reduce your design.")
            raise SystemExit(2)
    if ns.sample_combos:
        # reservoir sample from the iterator
        k = ns.sample_combos
        rnd = random.Random()
        reservoir = []
        for i, item in enumerate(itertools.product(*value_lists)):
            if i < k:
                reservoir.append(item)
            else:
                j = rnd.randrange(i + 1)
                if j < k:
                    reservoir[j] = item
        combos = reservoir
    else:
        combos = list(itertools.product(*value_lists))

print(f"Found {len(enum_keys)} factor keys: {enum_keys}")
print(f"Total combos: {len(combos)}")
print("Combos (showing as dicts):\n")
for combo in combos:
    mapping = dict(zip(enum_keys, combo))
    # Also compute a combo_id for reproducibility and easy referencing
    from fedchem.utils.manifest_utils import compute_combo_id
    print(json.dumps({"combo": mapping, "combo_id": compute_combo_id(mapping)}, ensure_ascii=False))

# Also print parsed privacy epsilon/delta lists using same parser semantics from run_real_site_experiment

def _ensure_list(values):
    if values is None:
        return []
    if isinstance(values, (list, tuple, set)):
        return list(values)
    if isinstance(values, str):
        parts = [v.strip() for v in values.split(",") if v.strip()]
        return parts if parts else [values]
    return [values]


def _parse_privacy_budgets(values, fallback):
    candidates = []
    for item in _ensure_list(values):
        if isinstance(item, str) and item.strip() in {"∞", "inf", "infinity", "INF"}:
            candidates.append(float('inf'))
            continue
        try:
            candidates.append(float(item))
        except Exception:
            continue
    if candidates:
        seen = []
        for val in candidates:
            if val not in seen:
                seen.append(val)
        return seen
    return fallback

print("\nInterpreted privacy lists:")
dp_target_raw = factors.get('DP_Target_Eps')
privacy_raw = None
dp_delta_raw = factors.get('DP_Delta')
print(f"  FACTORS.DP_Target_Eps (raw): {dp_target_raw}")
print(f"  FACTORS.DP_Delta (raw): {dp_delta_raw}")

EPSILON_VALUES_DEFAULT = [float('inf'), 10.0, 1.0, 0.1]

parsed_dp_target = _parse_privacy_budgets(dp_target_raw, EPSILON_VALUES_DEFAULT)
try:
    fallback_delta = [float(cfg.get('DP_DELTA', 1e-5))]
except Exception:
    fallback_delta = [1e-5]
parsed_delta = _parse_privacy_budgets(dp_delta_raw, fallback_delta)

print(f"  Parsed DP_Target_Eps -> {parsed_dp_target}")
print(f"  Parsed DP_Delta -> {parsed_delta}")
