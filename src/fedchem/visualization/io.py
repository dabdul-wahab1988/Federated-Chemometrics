import json
import yaml
from pathlib import Path
from typing import Sequence, Optional, Dict, Union
import logging

logger = logging.getLogger(__name__)


def _eps_dir_variants(eps: Union[str, float]) -> Sequence[str]:
    raw = str(eps)
    tokens = {raw}
    tokens.add(raw.replace('.', '_'))
    tokens.add(raw.replace('.', ''))
    tokens.add(raw.replace('.', '_').replace(',', '_'))
    if raw.endswith('.0'):
        tokens.add(raw[:-2])
    if raw.isdigit():
        tokens.add(f"{raw}_0")
    return [t for t in tokens if t]


def collect_manifests(root: Union[str, Path], ks: Sequence[int], eps: Sequence[str], pattern_template: Optional[str] = None, skip_missing: bool = True) -> Dict[int, Dict[str, Path]]:
    """
    Collect manifest paths for a grid of ks and eps under root.
    The expected on-disk layout is root/generated_figures_tables_archive/transfer_k_{k}/eps_{eps}/manifest_1.json
    Returns a dict mapping k -> eps -> Path to manifest
    """
    root = Path(root)
    manifests = {}
    for k in ks:
        k_map = {}
        for eps_val in eps:
            eps_variants = _eps_dir_variants(eps_val)
            manifest_path = None
            for variant in eps_variants:
                subdir = f"transfer_k_{k}/eps_{variant}"
                manifest_guess = root / "generated_figures_tables_archive" / subdir / "manifest_1.json"
                if manifest_guess.exists():
                    manifest_path = manifest_guess
                    break
                alt = root / subdir / "manifest_1.json"
                if alt.exists():
                    manifest_path = alt
                    break
            if manifest_path is None:
                candidates = []
                for variant in eps_variants:
                    base_archive = root / "generated_figures_tables_archive" / f"transfer_k_{k}" / f"eps_{variant}"
                    if base_archive.exists():
                        candidates.extend(base_archive.rglob("manifest_1.json"))
                    base_flat = root / f"transfer_k_{k}" / f"eps_{variant}"
                    if base_flat.exists():
                        candidates.extend(base_flat.rglob("manifest_1.json"))
                    if candidates:
                        break
                if candidates:
                    manifest_path = candidates[0]
            if manifest_path is None:
                if not skip_missing:
                    raise FileNotFoundError(f"Manifest not found for k={k}, eps={eps_val} under {root}")
                k_map[str(eps_val)] = None
            else:
                k_map[str(eps_val)] = manifest_path
        manifests[int(k)] = k_map
    return manifests


def load_manifest(path: Union[str, Path]) -> Optional[dict]:
    """Load JSON manifest safely. Returns dict or None if file not present.
    """
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        logger.debug("Manifest not found: %s", str(p))
        return None
    with p.open('r', encoding='utf-8') as fh:
        try:
            data = json.load(fh)
            return data
        except Exception as e:
            logger.exception("Failed to load manifest %s: %s", path, e)
            return None


def find_debug_weights(manifest_dir: Union[str, Path]) -> Sequence[Path]:
    """Search for debug weight dumps under manifest directory.
    Returns a list of Paths; may be empty
    """
    p = Path(manifest_dir)
    candidates = []
    search_dir = p / 'lca_artifacts' / 'debug_weights'
    if search_dir.exists():
        for f in sorted(search_dir.glob('*')):
            if f.suffix in ['.npy', '.npz', '.json']:
                candidates.append(f)
    return candidates


def get_default_baseline(config_path: Optional[Union[str, Path]] = None) -> Optional[str]:
    """Read config YAML and return the first baseline method declared.
    Look under EXPERIMENTAL_DESIGN.FACTORS.Baseline or Baseline in the root.
    """
    if config_path is None:
        config_path = Path('config.yaml')
    else:
        config_path = Path(config_path)
    if not config_path.exists():
        logger.debug("Config file not found: %s", str(config_path))
        return None
    try:
        with open(config_path, 'r', encoding='utf-8') as fh:
            cfg = yaml.safe_load(fh)
    except Exception as e:
        logger.exception("Failed to read config.yaml: %s", e)
        return None

    # check EXPERIMENTAL_DESIGN.FACTORS.Baseline
    exp = cfg.get('EXPERIMENTAL_DESIGN', {}) or {}
    factors = exp.get('FACTORS', {}) or {}
    baseline_list = factors.get('Baseline')
    if baseline_list:
        if isinstance(baseline_list, (list, tuple)):
            return str(baseline_list[0])
        return str(baseline_list)
    # fallback to top-level Baseline
    top_baseline = cfg.get('Baseline')
    if top_baseline:
        if isinstance(top_baseline, (list, tuple)):
            return str(top_baseline[0])
        return str(top_baseline)
    return None
