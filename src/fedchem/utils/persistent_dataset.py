from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np


def load_persistent_site(data_dir: str | Path | None = None) -> Optional[Tuple[Dict[str, Dict[str, Any]], Optional[list], Dict[str, Any]]]:
    """Load the persistent site dataset if present.

    Looks for data/site_design/manifest.json in the current working directory.
    Returns (data_dict, wavelengths, meta) or None if not present or invalid.
    data_dict maps 'site_i' -> {'X': np.ndarray, 'y': np.ndarray}
    """
    root = Path.cwd()
    base = Path(data_dir) if data_dir is not None else root / "data" / "site_design"
    try:
        if not base.exists() or not base.is_dir():
            return None
        manifest_path = base / "manifest.json"
        if not manifest_path.exists():
            return None
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        # load wavelengths from reference meta if present
        ref_meta = base / "reference" / "meta.json"
        wavelengths = None
        if ref_meta.exists():
            try:
                ref = json.loads(ref_meta.read_text(encoding="utf-8"))
                wavelengths = ref.get("wavelengths")
            except Exception:
                wavelengths = None
        # gather per-site files
        data: Dict[str, Dict[str, Any]] = {}
        sites = [p for p in base.iterdir() if p.is_dir() and p.name.startswith("site_")]
        if not sites:
            return None
        for s in sorted(sites, key=lambda p: p.name):
            try:
                Xp = np.load(s / "X.npy")
                yp = np.load(s / "y.npy")
                data[s.name] = {"X": Xp, "y": yp}
            except Exception:
                # skip invalid site
                return None
        meta = {"manifest": manifest, "source": str(base)}
        return data, wavelengths, meta
    except Exception:
        return None
