"""
Master script to run all objective generators.
"""

import argparse
import math
import os
import random
import subprocess
import sys
from pathlib import Path

# If the project package (`fedchem`) isn't installed in the environment, allow running
# scripts directly from a checkout by ensuring `src/` is on `sys.path`.
try:
    # Try a lightweight import to let normal execution proceed when installed
    import fedchem  # noqa: E402
except Exception:
    _proj_root = Path(__file__).resolve().parents[1]
    _src_dir = _proj_root / "src"
    if _src_dir.exists():
        # Prepend to ensure we take local checkout over any installed package
        sys.path.insert(0, str(_src_dir))

from fedchem.results import ensure_results_tree, manifest_to_raw_records, write_raw_records
from fedchem.utils.config import load_config, load_and_seed_config, validate_config, get_instruments_from_config

# Updated references from '5site' to 'site' to reflect the three-site configuration.
# Updated script names, paths, and environment variables accordingly.

scripts = [
    "run_real_site_experiment.py",
]
DEFAULT_SAMPLE_COMBOS = 122
DEFAULT_MIN_PER_STRATUM = 2
DEFAULT_STRATIFY_KEYS = ["DP_Target_Eps", "Federated_Method", "Spectral_Drift"]

# Experimental factor Drift_Type (isolated drift type testing) mapping helper
def _apply_drift_type(combo_env: dict, drift_type: str, cfg: dict):
    """Enable only one of the drift augmentations depending on Drift_Type.

    Accepted types: 'jitter', 'scatter', 'baseline', 'noise', 'combined', 'none'
    """
    # Use DRIFT_AUGMENT defaults and set one augmentation non-zero for isolated types
    base = cfg.get('DRIFT_AUGMENT')
    if not isinstance(base, dict):
        base = {}

    def _safe_float_from_base(key, default):
        try:
            # Prefer explicit config value when present
            raw = base.get(key, default) if isinstance(base, dict) else default
            if raw is None:
                return float(default)
            return float(raw)
        except Exception:
            # Fall back to environment or the provided default
            try:
                return float(os.environ.get(f"FEDCHEM_DRIFT_AUGMENT_{key.upper()}") or default)
            except Exception:
                return float(default)

    jitter = _safe_float_from_base('jitter_wavelength_px', 0.01)
    scatter = _safe_float_from_base('multiplicative_scatter', 0.05)
    baseline = _safe_float_from_base('baseline_offset', 0.005)
    noise = _safe_float_from_base('white_noise_sigma', 0.005)
    # default reset
    combo_env['FEDCHEM_DRIFT_AUGMENT_JITTER_WAVELENGTH_PX'] = '0.0'
    combo_env['FEDCHEM_DRIFT_AUGMENT_MULTIPLICATIVE_SCATTER'] = '0.0'
    combo_env['FEDCHEM_DRIFT_AUGMENT_BASELINE_OFFSET'] = '0.0'
    combo_env['FEDCHEM_DRIFT_AUGMENT_WHITE_NOISE_SIGMA'] = '0.0'
    combo_env['FEDCHEM_DRIFT_AUGMENT_APPLY_AUGMENTATION_DURING_TRAINING'] = '0'
    combo_env['FEDCHEM_DRIFT_AUGMENT_APPLY_TEST_SHIFTS'] = '0'
    t = (drift_type or '').lower()
    if t == 'none':
        return
    combo_env['FEDCHEM_DRIFT_AUGMENT_APPLY_AUGMENTATION_DURING_TRAINING'] = '1'
    combo_env['FEDCHEM_DRIFT_AUGMENT_APPLY_TEST_SHIFTS'] = '1'
    if t == 'jitter':
        combo_env['FEDCHEM_DRIFT_AUGMENT_JITTER_WAVELENGTH_PX'] = str(jitter)
    elif t == 'scatter':
        combo_env['FEDCHEM_DRIFT_AUGMENT_MULTIPLICATIVE_SCATTER'] = str(scatter)
    elif t == 'baseline':
        combo_env['FEDCHEM_DRIFT_AUGMENT_BASELINE_OFFSET'] = str(baseline)
    elif t == 'noise':
        combo_env['FEDCHEM_DRIFT_AUGMENT_WHITE_NOISE_SIGMA'] = str(noise)
    elif t == 'combined':
        combo_env['FEDCHEM_DRIFT_AUGMENT_JITTER_WAVELENGTH_PX'] = str(jitter)
        combo_env['FEDCHEM_DRIFT_AUGMENT_MULTIPLICATIVE_SCATTER'] = str(scatter)
        combo_env['FEDCHEM_DRIFT_AUGMENT_BASELINE_OFFSET'] = str(baseline)
        combo_env['FEDCHEM_DRIFT_AUGMENT_WHITE_NOISE_SIGMA'] = str(noise)

def _format_eps_label(value):
    if value is None:
        return None
    try:
        numeric = float(value)
        if math.isinf(numeric):
            return "inf"
        formatted = str(numeric)
    except Exception:
        formatted = str(value)
    return formatted.replace(".", "_")


def _format_delta_label(value):
    if value is None:
        return None
    try:
        label = str(value)
    except Exception:
        return None
    return f"delta_{label.replace('.', '_').replace('-', 'neg')}"


def _apply_spectral_drift_level(combo_env: dict, level: str, cfg: dict):
    """Set FEDCHEM_DRIFT_AUGMENT_* env vars in combo_env based on a given level.

    Levels: 'none', 'low', 'moderate', 'high'. Values are multiplicative factors
    of config defaults or sensible fallbacks.
    """
    # base defaults from config
    base = cfg.get('DRIFT_AUGMENT') if isinstance(cfg.get('DRIFT_AUGMENT'), dict) else {}
    def _get_float(k, default):
        try:
            # Ensure `base` is a dict before calling .get to avoid AttributeError if it's None.
            if isinstance(base, dict):
                val = base.get(k, default)
            else:
                val = os.environ.get(f"FEDCHEM_DRIFT_AUGMENT_{k.upper()}") or default
            return float(val)
        except Exception:
            try:
                return float(os.environ.get(f"FEDCHEM_DRIFT_AUGMENT_{k.upper()}") or default)
            except Exception:
                return default

    jitter_default = _get_float('jitter_wavelength_px', 0.01)
    scatter_default = _get_float('multiplicative_scatter', 0.05)
    baseline_default = _get_float('baseline_offset', 0.005)
    noise_default = _get_float('white_noise_sigma', 0.005)
    seed_default = int(_get_float('augmentation_seed', 99) or 99)

    mapping = {
        'none': 0.0,
        'low': 0.25,
        'moderate': 1.0,
        'high': 2.0,
    }
    factor = mapping.get(str(level).lower(), 0.0)

    # Compose env vars as strings
    combo_env['FEDCHEM_DRIFT_AUGMENT_JITTER_WAVELENGTH_PX'] = str(jitter_default * factor)
    combo_env['FEDCHEM_DRIFT_AUGMENT_MULTIPLICATIVE_SCATTER'] = str(scatter_default * factor)
    combo_env['FEDCHEM_DRIFT_AUGMENT_BASELINE_OFFSET'] = str(baseline_default * factor)
    combo_env['FEDCHEM_DRIFT_AUGMENT_WHITE_NOISE_SIGMA'] = str(noise_default * factor)
    combo_env['FEDCHEM_DRIFT_AUGMENT_AUGMENTATION_SEED'] = str(seed_default)
    # Apply augmentation flags: enable if not none
    combo_env['FEDCHEM_DRIFT_AUGMENT_APPLY_AUGMENTATION_DURING_TRAINING'] = '1' if factor > 0.0 else '0'
    combo_env['FEDCHEM_DRIFT_AUGMENT_APPLY_TEST_SHIFTS'] = '1' if factor > 0.0 else '0'


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Run all objectives with config-driven combos")
    p.add_argument("--config", dest="config", default="config.yaml", help="Path to configuration YAML (default: config.yaml)")
    p.add_argument("--resample-method", choices=["interpolate", "subsample"], help="Override RESAMPLE_METHOD for resampling")
    p.add_argument("--no-resample", dest="no_resample", action="store_true", help="Disable resampling for all runs")
    p.add_argument("--dp-default", dest="dp_default", type=float, help="Override top-level DP_TARGET_EPS default")
    p.add_argument("--rounds", dest="rounds", type=int, help="Override number of federated rounds (ROUNDS) for child runs")
    p.add_argument("--num-sites", dest="num_sites", type=int, help="Override number of federated sites (NUM_SITES) for child runs")
    p.add_argument("--n-wavelengths", dest="n_wavelengths", type=int, help="Override DEFAULT_N_WAVELENGTHS for child runs")
    p.add_argument("--seed", dest="seed", type=int, help="Override PRNG seed for child runs")
    p.add_argument("--max-transfer-samples", dest="max_transfer_samples", type=int, help="Override MAX_TRANSFER_SAMPLES (cap on transfer samples)")
    p.add_argument("--quick", dest="quick", action="store_true", help="Force quick mode (FEDCHEM_QUICK=1)")
    p.add_argument("--strict-config", dest="strict_config", action="store_true", help="Halt on configuration validation errors (strict) ")
    p.add_argument(
        "--results-dir",
        dest="results_dir",
        default="results",
        help="Root directory for the canonical results tree (defaults to ./results)",
    )
    p.add_argument(
        "--max-combos",
        dest="max_combos",
        type=int,
        help="Maximum number of enumerated experimental-design combos to allow (prevents OOM)",
    )
    p.add_argument(
        "--target-combos",
        dest="target_combos",
        type=int,
        help="Target number of combos to sample/execute (default: 122 when the full design is large)",
    )
    p.add_argument(
        "--min-per-stratum",
        dest="min_per_stratum",
        type=int,
        default=2,
        help="Minimum number of combos to sample per DP × Method × Drift stratum (default: 2)",
    )
    p.add_argument(
        "--no-stratify",
        dest="no_stratify",
        action="store_true",
        help="Disable default stratified sampling and fall back to reservoir or full enumeration.",
    )
    p.add_argument(
        "--full-factorial",
        dest="full_factorial",
        action="store_true",
        help="Force full factorial enumeration even if large (dangerous; can OOM).",
    )
    p.add_argument(
        "--sample-combos",
        dest="sample_combos",
        type=int,
        help="If provided, randomly sample N combos from the full factorial design (memory-safe reservoir sampling)",
    )
    # Accept and ignore unknown args (e.g., pytest) to avoid parsing errors within test runner
    ns, _ = p.parse_known_args(argv)
    return ns


def main(argv=None):
    args = parse_args(argv)
    # Load config.yaml to seed environment for scripts (env overrides still win)
    cfg = load_and_seed_config(args.config)
    results_tree = ensure_results_tree(args.results_dir or "results")
    # Ensure child scripts write their outputs into the chosen results tree.
    generated_tables_dir = results_tree.base / "generated_figures_tables"
    generated_archive_dir = results_tree.base / "generated_figures_tables_archive"
    # Make sure directories exist so child scripts can write to them
    generated_tables_dir.mkdir(parents=True, exist_ok=True)
    generated_archive_dir.mkdir(parents=True, exist_ok=True)
    print(f"Child script outputs (generated_figures_tables) will be under: {generated_tables_dir}")
    print(f"Child script archive outputs (generated_figures_tables_archive) will be under: {generated_archive_dir}")
    
    # Validate config; honor --strict-config flag by re-raising errors when present.
    print("\n" + "="*70)
    print("Configuration Validation")
    print("="*70)
    try:
        warnings = validate_config(cfg, strict=args.strict_config)
    except Exception as e:
        if args.strict_config:
            print(f"\n❌ CRITICAL: Configuration validation failed (strict mode)")
            print(f"   Error: {e}")
            raise
        print(f"⚠️  Configuration validation error: {e}")
        warnings = [str(e)]
    
    if warnings:
        print(f"\n⚠️  Found {len(warnings)} configuration warning(s):")
        for i, w in enumerate(warnings, 1):
            print(f"  {i}. {w}")
        print("\nℹ️  Warnings are informational and won't block execution.")
        print("   Use --strict-config to treat warnings as errors.\n")
    else:
        print("✅ Configuration is valid!\n")
    print("="*70 + "\n")
    # If instrument-as-site is enabled, use number of instruments as NUM_SITES unless CLI overrides
    if cfg.get("INSTRUMENT_AS_SITE") and not cfg.get("NUM_SITES"):
        instruments = get_instruments_from_config(cfg) or []
        if instruments:
            cfg['NUM_SITES'] = len(instruments)

    # Apply CLI overrides to config (non-destructive)
    if args.resample_method:
        cfg['RESAMPLE_METHOD'] = args.resample_method
    if args.no_resample:
        cfg['RESAMPLE_SPECTRA'] = False
    if args.dp_default is not None:
        cfg['DP_TARGET_EPS'] = args.dp_default
    if args.rounds is not None:
        cfg['ROUNDS'] = args.rounds
    if args.num_sites is not None:
        cfg['NUM_SITES'] = args.num_sites
    if args.n_wavelengths is not None:
        cfg['DEFAULT_N_WAVELENGTHS'] = args.n_wavelengths
    if args.seed is not None:
        cfg['SEED'] = args.seed
    if args.max_transfer_samples is not None:
        cfg['MAX_TRANSFER_SAMPLES'] = args.max_transfer_samples
    if args.quick:
        cfg['QUICK'] = 1
    corn_mat_path = Path("data") / "raw" / "corn.mat"
    prepare_corn_script = Path("tools") / "prepare_corn_for_real.py"
    gen_synthetic_script = Path("tools") / "generate_site_design.py"
    
    # Check for real IDRC wheat data
    idrc_data_exists = any((Path("data") / f"Manufacturer{chr(65+i)}").exists() for i in range(3))

    if corn_mat_path.exists() and prepare_corn_script.exists():
        print(f"Real corn data found at {corn_mat_path}. Preparing real multi-site dataset...")
        # ... existing corn logic ...
    elif idrc_data_exists:
        print("Real IDRC wheat shootout data found. Using real data for objectives.")
        os.environ.setdefault("FEDCHEM_USE_SITE_DATA", "1")
        os.environ.setdefault("FEDCHEM_NUM_SITES", str(cfg.get('NUM_SITES', 2)))
    elif gen_synthetic_script.exists():
        print(f"Real data not found. Using synthetic data from {gen_synthetic_script}...")
        # ... existing synthetic logic ...
        print(f"Real corn data found at {corn_mat_path}. Preparing real multi-site dataset...")
        try:
            result_corn = subprocess.run(
                [sys.executable, str(prepare_corn_script), 
                 "--mat", str(corn_mat_path),
                 "--out", "data/site_design",
                 "--paired-size", "20",
                 "--seed", "0"],
                capture_output=True, 
                text=True, 
                env=os.environ.copy()
            )
            if result_corn.returncode != 0:
                print(f"Warning: {prepare_corn_script} failed:\n{result_corn.stderr}")
                print("Falling back to synthetic data generation...")
                if gen_synthetic_script.exists():
                    result_synth = subprocess.run([sys.executable, str(gen_synthetic_script)], capture_output=True, text=True, env=os.environ.copy())
                    if result_synth.returncode != 0:
                        print(f"Warning: {gen_synthetic_script} also failed:\n{result_synth.stderr}")
                    else:
                        print(f"{gen_synthetic_script} completed (synthetic fallback).")
                        os.environ.setdefault("FEDCHEM_NUM_SITES", "3")
            else:
                print(f"{prepare_corn_script} completed; using REAL corn data for objectives.")
                os.environ.setdefault("FEDCHEM_USE_SITE_DATA", "1")
                os.environ.setdefault("FEDCHEM_NUM_SITES", "3")
        except Exception as e:
            print(f"Warning: could not run {prepare_corn_script}: {e}")
            print("Falling back to synthetic data generation...")
            if gen_synthetic_script.exists():
                try:
                    result_synth = subprocess.run([sys.executable, str(gen_synthetic_script)], capture_output=True, text=True, env=os.environ.copy())
                    if result_synth.returncode != 0:
                        print(f"Warning: {gen_synthetic_script} failed:\n{result_synth.stderr}")
                    else:
                        print(f"{gen_synthetic_script} completed (synthetic fallback).")
                        os.environ.setdefault("FEDCHEM_NUM_SITES", "3")
                except Exception as e2:
                    print(f"Warning: synthetic fallback also failed: {e2}")
    elif gen_synthetic_script.exists():
        print(f"Real corn data not found. Using synthetic data from {gen_synthetic_script}...")
        try:
            result_synth = subprocess.run([sys.executable, str(gen_synthetic_script)], capture_output=True, text=True, env=os.environ.copy())
            if result_synth.returncode != 0:
                print(f"Warning: {gen_synthetic_script} failed:\n{result_synth.stderr}")
            else:
                print(f"{gen_synthetic_script} completed; enabling FEDCHEM_USE_SITE_DATA for objectives.")
                os.environ.setdefault("FEDCHEM_USE_SITE_DATA", "1")
                os.environ.setdefault("FEDCHEM_NUM_SITES", "3")
        except Exception as e:
            print(f"Warning: could not run {gen_synthetic_script}: {e}")
    else:
        print(f"Note: Neither real corn data nor synthetic generator found; skipping multi-site dataset preparation.")

    # If an experimental design is defined, enumerate its full factorial combos at this higher level
    design = (cfg.get('EXPERIMENTAL_DESIGN') or {}) if isinstance(cfg.get('EXPERIMENTAL_DESIGN'), dict) else {}
    factors = design.get('FACTORS', {}) if isinstance(design.get('FACTORS', {}), dict) else {}

    # Helper: convert factor value to list for enumeration. Accept aliases for DP factors.
    def _factor_values(v):
        if isinstance(v, (list, tuple)):
            return list(v)
        return [v]

    # Keep simple factors in enumeration; participation/compression schedules are enumerated
    skip_keys = set()
    # Canonicalization mapping for common experimental factor aliases
    alias_map = {
        'DP_Target_Eps': 'DP_Target_Eps',
        'DP_TARGET_EPS': 'DP_Target_Eps',
    }
    enum_keys = [k for k in factors.keys() if k not in skip_keys]
    # Normalize keys with known aliases so code that maps envs can be consistent
    normalized_keys = []
    for k in enum_keys:
        normalized = alias_map.get(k, k)
        # preserve original key for env naming if not DP_Target_Eps; otherwise use canonical key
        normalized_keys.append(normalized)

    # Build value lists for each key (preserve order)
    value_lists = [ _factor_values(factors.get(k)) for k in enum_keys ]

    from itertools import product

    # Avoid building an in-memory list of every combination when the design explodes
    # in size. Use a generator and compute the total count via product of lengths.
    combos = product(*value_lists) if value_lists else iter([()])
    # Compute the number of combos without materializing the full list
    if value_lists:
        try:
            combo_counts = [len(vals) for vals in value_lists]
            num_combos = math.prod(combo_counts) if combo_counts else 1
        except Exception:
            # Fall back to a conservative value if we cannot require len() for a value
            num_combos = float("inf")
    else:
        num_combos = 1
    # Allow caller to cap the number of enumerated combos to avoid OOM/long runs
    if args.max_combos is not None and num_combos > args.max_combos:
        raise RuntimeError(f"Enumerated {num_combos} combos exceeds --max-combos={args.max_combos}. Reduce your experimental design or increase --max-combos.")

    # Decide on effective target_n
    if args.target_combos is None and args.sample_combos is not None:
        args.target_combos = args.sample_combos

    target_n = args.target_combos
    if target_n is None and not args.no_stratify and not args.full_factorial and (isinstance(num_combos, (int, float)) and not math.isinf(num_combos) and num_combos > DEFAULT_SAMPLE_COMBOS):
        target_n = DEFAULT_SAMPLE_COMBOS

    # If requested, sample a fixed number of combos from the product using reservoir sampling.
    def _reservoir_sample(iterator, k: int, seed: int | None = None):
        if k <= 0:
            return []
        if seed is not None:
            rnd = random.Random(seed)
        else:
            rnd = random.Random()
        reservoir = []
        for i, item in enumerate(iterator):
            if i < k:
                reservoir.append(item)
            else:
                j = rnd.randrange(i + 1)
                if j < k:
                    reservoir[j] = item
        return reservoir

    # Determine sampling behavior. Use explicit `--sample-combos` or the default
    # stratified sampling when the full design exceeds `DEFAULT_SAMPLE_COMBOS`.
    should_sample = False
    sample_n = None
    if getattr(args, 'target_combos', None) is not None:
        sample_n = int(args.target_combos)
        should_sample = True
    elif getattr(args, 'sample_combos', None) is not None:
        sample_n = int(args.sample_combos)
        should_sample = True
    else:
        # Default stratified sampling when design grows too large
        if isinstance(num_combos, (int, float)) and not math.isinf(num_combos) and num_combos > DEFAULT_SAMPLE_COMBOS:
            sample_n = DEFAULT_SAMPLE_COMBOS
            should_sample = True

    if should_sample and sample_n and sample_n > 0:
        seeds = cfg.get('SEEDS')
        seed_val = args.seed if getattr(args, 'seed', None) is not None else (seeds[0] if isinstance(seeds, (list, tuple)) and seeds else None)
        # Stratify on vital keys when present. If not, fall back to uniform reservoir sampling.
        stratify_keys = DEFAULT_STRATIFY_KEYS
        stratify_indices = [i for i, k in enumerate(enum_keys) if k in stratify_keys]
        # Fallback to uniform sampling if we don't have the stratify keys
        if len(stratify_indices) < len([k for k in stratify_keys if k in enum_keys]):
            if isinstance(num_combos, (int, float)) and not math.isinf(num_combos) and num_combos <= sample_n:
                combos = list(product(*value_lists)) if value_lists else [()]
            else:
                combos = _reservoir_sample(product(*value_lists), sample_n, seed=seed_val)
            print(f"Sampling enabled: selected {len(combos)} combos via reservoir sampling (seed={seed_val}).")
        else:
            k_per_stratum = int(getattr(args, 'min_per_stratum', DEFAULT_MIN_PER_STRATUM) or DEFAULT_MIN_PER_STRATUM)
            # Build all strata from the specified stratify indices without enumerating the entire space.
            stratum_value_lists = [value_lists[i] for i in stratify_indices]
            all_strata = list(product(*stratum_value_lists))
            rnd = random.Random(seed_val)
            selected = []
            selected_set = set()
            # For each stratum, construct k_per_stratum combos by sampling other factor levels at random
            non_stratify_indices = [i for i in range(len(enum_keys)) if i not in stratify_indices]
            for s in all_strata:
                attempts = 0
                while len([1 for t in selected if tuple(t[i] for i in stratify_indices) == s]) < k_per_stratum and attempts < k_per_stratum * 10:
                    # Build a random combo for this stratum
                    combo = [None] * len(enum_keys)
                    for idx, v in zip(stratify_indices, s):
                        combo[idx] = v
                    for idx in non_stratify_indices:
                        combo[idx] = rnd.choice(value_lists[idx])
                    combo_t = tuple(combo)
                    if combo_t not in selected_set:
                        selected.append(combo_t)
                        selected_set.add(combo_t)
                    attempts += 1
            # If we got more than required via the per-stratum minima, trim randomly
            if len(selected) > sample_n:
                rnd.shuffle(selected)
                selected = selected[:sample_n]
            # If we need more combos, sample uniformly across the full factor space until we reach sample_n.
            if len(selected) < sample_n:
                needed = sample_n - len(selected)
                fallback_attempts = 0
                # Draw random combos by sampling each factor independently
                while len(selected) < sample_n and fallback_attempts < needed * 50:
                    combo = tuple(rnd.choice(value_lists[i]) for i in range(len(enum_keys)))
                    if combo not in selected_set:
                        selected_set.add(combo)
                        selected.append(combo)
                    fallback_attempts += 1
                # As a final fallback, do a controlled iteration over product to fill any remaining slots (bounded)
                if len(selected) < sample_n:
                    it = product(*value_lists)
                    for combo in it:
                        if combo in selected_set:
                            continue
                        selected_set.add(combo)
                        selected.append(combo)
                        if len(selected) >= sample_n:
                            break
            combos = selected
            print(f"Sampling enabled: selected {len(combos)} combos via stratified sampling (seed={seed_val}).")

    # Surface the experimental design metadata for observability
    design_type = design.get('DESIGN_TYPE', 'unspecified') if design else 'unspecified'
    if design_type or enum_keys:
        factor_desc = []
        for key in enum_keys:
            vals = _factor_values(factors.get(key))
            preview = ', '.join(str(v) for v in vals[:5])
            if len(vals) > 5:
                preview += ', ...'
            factor_desc.append(f"{key} ({len(vals)} levels: {preview})")
        factor_text = '; '.join(factor_desc) if factor_desc else 'No enumerated factors'
        printable_count = int(num_combos) if isinstance(num_combos, (int, float)) and not math.isinf(num_combos) else 'unknown'
        print(f"Experimental design type: {design_type}. Enumerated {printable_count} combo(s). {factor_text}.")

    # Run one script invocation per experimental-design combo (higher-level orchestration)
    for script in scripts:
        for combo in combos:
            design_mapping = {key: value for key, value in zip(enum_keys, combo)}
            # Prepare environment for this combo
            combo_env = os.environ.copy()
            repo_root = Path(__file__).resolve().parents[1]
            src_dir = repo_root / "src"
            # Ensure both the project root and src/ are in PYTHONPATH for module imports
            env_py_path = combo_env.get("PYTHONPATH", "")
            parts = []
            if src_dir.exists():
                parts.append(str(src_dir))
            parts.append(str(repo_root))
            if env_py_path:
                parts.append(env_py_path)
            combo_env["PYTHONPATH"] = os.pathsep.join(parts)

            # Seed common parameters from config (override shell env when provided)
            if 'NUM_SITES' in cfg:
                combo_env["FEDCHEM_NUM_SITES"] = str(cfg.get('NUM_SITES'))
            if 'ROUNDS' in cfg:
                combo_env["FEDCHEM_ROUNDS"] = str(cfg.get('ROUNDS'))
            if 'DP_TARGET_EPS' in cfg:
                combo_env["FEDCHEM_DP_TARGET_EPS"] = str(cfg.get('DP_TARGET_EPS'))
            if 'DP_DELTA' in cfg:
                combo_env["FEDCHEM_DP_DELTA"] = str(cfg.get('DP_DELTA'))
            if 'PARTICIPATION_RATE' in cfg and cfg.get('PARTICIPATION_RATE') is not None:
                combo_env["FEDCHEM_PARTICIPATION_RATE"] = str(cfg.get('PARTICIPATION_RATE'))
            # Ensure child scripts output to the runner-specified results tree
            combo_env["FEDCHEM_OUTPUT_DIR"] = str(generated_tables_dir)
            combo_env["FEDCHEM_ARCHIVE_ROOT"] = str(generated_archive_dir)
            # Allow config to override but be explicit about the location we want
            if 'OUTPUT_DIR' in cfg and cfg.get('OUTPUT_DIR'):
                combo_env.update({
                    "FEDCHEM_OUTPUT_DIR": str(Path(cfg.get('OUTPUT_DIR')).resolve()),
                })
            if 'DEFAULT_N_WAVELENGTHS' in cfg:
                combo_env["FEDCHEM_N_WAVELENGTHS"] = str(cfg.get('DEFAULT_N_WAVELENGTHS'))
            # Respect top-level RESAMPLE_SPECTRA flag: ensure child env knows whether to resample
            if 'RESAMPLE_SPECTRA' in cfg:
                combo_env["FEDCHEM_RESAMPLE_SPECTRA"] = "1" if cfg.get('RESAMPLE_SPECTRA') else "0"
            # Respect top-level RESAMPLE_METHOD: set child env if provided
            if 'RESAMPLE_METHOD' in cfg:
                combo_env["FEDCHEM_RESAMPLE_METHOD"] = str(cfg.get('RESAMPLE_METHOD'))

            # Map combo values to FEDCHEM_<KEY> env vars. Use single value per combo entry.
            for key, val in zip(enum_keys, combo):
                # Map aliased keys to canonical env names first
                mapped_key = alias_map.get(key, key)
                # Special handling for Spectral_Drift: set the DRIFT_AUGMENT env vars accordingly
                if key == 'Spectral_Drift':
                    try:
                        _apply_spectral_drift_level(combo_env, val, cfg)
                    except Exception:
                        pass
                    # Also set the generic flag for the spectral drift factor so manifests can record it
                    combo_env['FEDCHEM_SPECTRAL_DRIFT'] = str(val)
                    continue
                if key == 'Drift_Type':
                    try:
                        _apply_drift_type(combo_env, val, cfg)
                    except Exception:
                        pass
                    combo_env['FEDCHEM_DRIFT_TYPE'] = str(val)
                    continue
                # For schedule-like factors, convert lists to CSV strings; accept either a list or a string token
                if key in {"Participation_Schedule", "Compression_Schedule"}:
                    # If the design provided a Python list/tuple, join with commas
                    if isinstance(val, (list, tuple)):
                        value_str = ",".join(str(x) for x in val)
                    else:
                        # Strip possible brackets from textual list representation
                        value_str = str(val).strip()
                        if value_str.startswith("[") and value_str.endswith("]"):
                            value_str = value_str[1:-1].strip()
                else:
                    value_str = str(val)
                # For privacy-specific keys map to canonical env names
                if mapped_key == "DP_Target_Eps":
                    combo_env["FEDCHEM_DP_TARGET_EPS"] = value_str
                elif mapped_key in {"DP_Delta", "DP_Delta", "DP_DELTA", "DP_DELTA"}:
                    combo_env["FEDCHEM_DP_DELTA"] = value_str
                else:
                    # Safe fallback for arbitrary keys
                    if mapped_key:
                        safe_env_key = f"FEDCHEM_{mapped_key.upper()}"
                        combo_env[safe_env_key] = value_str
                        # Special-case mapping: map Conformal_Calibration factor to env var
                        if mapped_key == "Conformal_Calibration":
                            combo_env["FEDCHEM_CONFORMAL_CALIBRATION"] = value_str

            # For participation/compression schedules, if present at top-level use them
            ps = cfg.get('PARTICIPATION_SCHEDULE') if isinstance(cfg.get('PARTICIPATION_SCHEDULE'), list) else None
            if ps:
                try:
                    sched = ",".join([str(x) for x in ps])
                    combo_env.setdefault("FEDCHEM_PARTICIPATION_SCHEDULE", sched)
                except Exception:
                    pass
            cs = cfg.get('COMPRESSION_SCHEDULE') if isinstance(cfg.get('COMPRESSION_SCHEDULE'), list) else None
            if cs:
                try:
                    sched = ",".join([str(x) for x in cs])
                    combo_env.setdefault("FEDCHEM_COMPRESSION_SCHEDULE", sched)
                except Exception:
                    pass

            combo_env.setdefault("FEDCHEM_ROUNDS", "20")
            combo_env.setdefault("FEDCHEM_QUICK", "1" if cfg.get('QUICK') else "0")
            # Honor the top-level config value for FedPLS methods and whether to use it.
            # `setdefault` preserves an existing shell-level env var; prefer the config value
            # for generated runs so `config.yaml` is the truth of the experiment when invoked
            # via `run_all_objectives.py`.
            # Set the USE_FEDPLS flag from config if present (support unprefixed or prefixed keys), otherwise default to '1'
            if 'USE_FEDPLS' in cfg:
                combo_env["FEDCHEM_USE_FEDPLS"] = "1" if cfg.get("USE_FEDPLS") else "0"
            elif 'FEDCHEM_USE_FEDPLS' in cfg:
                combo_env["FEDCHEM_USE_FEDPLS"] = "1" if cfg.get("FEDCHEM_USE_FEDPLS") else "0"
            else:
                combo_env["FEDCHEM_USE_FEDPLS"] = "1"
            # Always set or unset the FEDPLS_METHOD in the child env based on config
            # so it cannot be inherited from the running shell.
            # Always set or unset the FEDPLS_METHOD in the child env based on config keys
            if 'FEDPLS_METHOD' in cfg:
                combo_env["FEDCHEM_FEDPLS_METHOD"] = str(cfg["FEDPLS_METHOD"])
            elif 'FEDCHEM_FEDPLS_METHOD' in cfg:
                combo_env["FEDCHEM_FEDPLS_METHOD"] = str(cfg["FEDCHEM_FEDPLS_METHOD"])
            else:
                combo_env.pop("FEDCHEM_FEDPLS_METHOD", None)

            # Signal to downstream scripts to skip their internal experimental-design enumeration
            combo_env.setdefault("FEDCHEM_SKIP_INTERNAL_DESIGN", "1")

            # Resolve script path relative to the scripts/ folder
            script_path = Path(script)
            if not script_path.is_absolute():
                script_path = Path(__file__).resolve().parent / script_path

            # Log the resolved FedPLS settings for the combo so the output is
            # explicit about whether 'parametric' or 'simpls' was chosen.
            fedpls_enabled = combo_env.get("FEDCHEM_USE_FEDPLS")
            fedpls_method = combo_env.get("FEDCHEM_FEDPLS_METHOD")
            print(
                f"Running {script} for design combo: {design_mapping}... "
                f"[FEDCHEM_USE_FEDPLS={fedpls_enabled} FEDCHEM_FEDPLS_METHOD={fedpls_method}]"
            )
            # If the target script is inside the `scripts/` package, run as a module to allow
            # relative imports (e.g., `from . import generate_objective_1`). Otherwise, run the
            # script file directly.
            try:
                if script_path.parent.name == "scripts":
                    module_name = f"scripts.{script_path.stem}"
                    print(f"Invoking as module: python -m {module_name}")
                    result = subprocess.run([sys.executable, "-m", module_name], capture_output=True, text=True, env=combo_env, cwd=str(repo_root))
                else:
                    result = subprocess.run([sys.executable, str(script_path)], capture_output=True, text=True, env=combo_env, cwd=str(repo_root))
            except Exception as e:
                print(f"Failed to run {script_path}: {e}")
                result = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr=str(e))
            if result.returncode != 0:
                print(f"Error in {script}: {result.stderr}")
            else:
                print(f"{script} completed for combo {design_mapping}.")
                # Basic manifest checks to ensure reproducibility metadata and diagnostics are present
                try:
                    import json
                    targets = []
                    if script == "run_real_site_experiment.py":
                        targets = [1, 2, 5, 6]
                    else:
                        import re
                        base = Path(script).stem
                        m = re.search(r"generate_objective_(\d+)$", base)
                        if m:
                            targets = [int(m.group(1))]
                    for idx in targets:
                        manifest_path = generated_tables_dir / f"manifest_{idx}.json"
                        expected_archive = None
                        if script == "run_real_site_experiment.py":
                            k_label = combo_env.get("FEDCHEM_TRANSFER_SAMPLES")
                            if k_label:
                                archive_base = generated_archive_dir / f"transfer_k_{k_label}"
                                if idx in (1, 6):
                                    eps_label = _format_eps_label(combo_env.get("FEDCHEM_DP_TARGET_EPS"))
                                    delta_label = _format_delta_label(combo_env.get("FEDCHEM_DP_DELTA"))
                                    if eps_label and delta_label:
                                        expected_archive = archive_base / f"eps_{eps_label}" / delta_label / f"objective_{idx}" / f"manifest_{idx}.json"
                                else:
                                    expected_archive = archive_base / f"objective_{idx}" / f"manifest_{idx}.json"
                        if manifest_path.exists():
                            pass
                        elif expected_archive and expected_archive.exists():
                            manifest_path = expected_archive
                        else:
                            # If the generator archives outputs (common for run_real_site_experiment),
                            # look for the manifest inside the archive directory as a fallback.
                            archive_matches = list(generated_archive_dir.rglob(f"manifest_{idx}.json"))
                            if archive_matches:
                                # Prefer matches that follow the expected archive layout produced by
                                # run_real_site_experiment: transfer_k_<k>/eps_<eps_label>/objective_<idx>/manifest_<idx>.json
                                import re

                                def _match_expected_archive(p: Path) -> bool:
                                    s = str(p.as_posix())
                                    return bool(re.search(r"/transfer_k_\d+/eps_[^/]+/objective_\d+/manifest_\d+\.json$", s))

                                preferred = [p for p in archive_matches if _match_expected_archive(p)]
                                chosen = preferred if preferred else archive_matches
                                # pick the most recent/path-sorted match as fallback
                                manifest_path = sorted(chosen)[-1]
                            else:
                                print(f"Warning: expected manifest {manifest_path} not found.")
                                continue
                        try:
                            with open(manifest_path, "r", encoding="utf-8") as fh:
                                mf = json.load(fh)
                        except Exception as e:
                            print(f"Warning: could not read {manifest_path}: {e}")
                            continue
                        run_label = f"objective_{idx}"
                        extra_meta = {
                            "dp_target_eps": combo_env.get("FEDCHEM_DP_TARGET_EPS"),
                            "dp_delta": combo_env.get("FEDCHEM_DP_DELTA"),
                            "clip_norm": combo_env.get("FEDCHEM_CLIP_NORM"),
                        }
                        raw_records = manifest_to_raw_records(
                            mf,
                            design_mapping,
                            run_label=run_label,
                            script_name=script,
                            manifest_path=manifest_path,
                            extra_metadata={k: v for k, v in extra_meta.items() if v is not None},
                        )
                        if raw_records:
                            combo_id_value = raw_records[0].get("combo_id", "combo-unknown")
                            dest_path = write_raw_records(raw_records, results_tree, combo_id=combo_id_value, run_label=run_label)
                            print(f"Logged {len(raw_records)} raw record(s) to {dest_path}")
                        missing = []
                        if not mf.get("versions"):
                            missing.append("versions")
                        if not mf.get("runtime"):
                            missing.append("runtime")
                        cfg = mf.get("config", {})
                        for key in ("seed", "standard_design", "design_version"):
                            if key not in cfg:
                                missing.append(f"config.{key}")
                        design_keys = ("n_wavelengths_actual", "transfer_samples_requested", "transfer_samples_used")
                        for key in design_keys:
                            if key not in cfg:
                                missing.append(f"config.{key}")
                        if idx in (1, 6):
                            for key in ("target_epsilon", "test_samples_per_site"):
                                if key not in cfg:
                                    missing.append(f"config.{key}")
                        if missing:
                            print(f"Manifest check WARNING for {manifest_path}: missing {missing}")
                        else:
                            print(f"Manifest {manifest_path} contains required reproducibility fields.")
                except Exception as e:
                    print(f"Unexpected error while checking manifests for {script}: {e}")

if __name__ == "__main__":
    main()
