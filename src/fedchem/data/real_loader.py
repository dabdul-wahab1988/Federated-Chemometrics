from pathlib import Path
from typing import Tuple
import numpy as np


class RealSiteLoader:
    """Load real multi-site spectral data organized by site directories.

    Expected layout per site (site_dir):
        - X_paired.npy: (n_paired, d)
        - y_paired.npy: (n_paired,)
        - X_local.npy: (n_local, d)
        - y_local.npy: (n_local,)

    Notes:
    - Files are loaded with memory-mapping when possible to reduce memory pressure.
    - Caller is responsible for ensuring data is numeric and compatible across sites.
    """

    def __init__(self, data_root: Path, reference_site: str = "site_0"):
        self.data_root = Path(data_root)
        self.ref_site = reference_site

    def _site_dir(self, site_id: str) -> Path:
        sd = self.data_root / site_id
        if not sd.exists():
            raise FileNotFoundError(f"Site directory not found: {sd}")
        return sd

    def load_paired_subset(self, site_id: str, k: int = 200, mmap: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Load up to k paired samples (X_paired, y_paired) from a site.

        Returns copies to avoid accidental memory sharing between callers.
        """
        sd = self._site_dir(site_id)
        X_path = sd / "X_paired.npy"
        y_path = sd / "y_paired.npy"
        if not X_path.exists() or not y_path.exists():
            raise FileNotFoundError(f"Missing paired files in {sd}")
        mmap_mode = 'r' if mmap else None
        X = np.load(X_path, mmap_mode=mmap_mode)
        y = np.load(y_path, mmap_mode=mmap_mode)
        # Defensive copies to avoid exposing mmap objects for accidental writes
        return X[:k].copy(), y[:k].copy()

    def load_local_test(self, site_id: str, mmap: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Load the held-out local test set (X_local, y_local) for a site."""
        sd = self._site_dir(site_id)
        X_path = sd / "X_local.npy"
        y_path = sd / "y_local.npy"
        if not X_path.exists() or not y_path.exists():
            raise FileNotFoundError(f"Missing local test files in {sd}")
        mmap_mode = 'r' if mmap else None
        X = np.load(X_path, mmap_mode=mmap_mode)
        y = np.load(y_path, mmap_mode=mmap_mode)
        return X.copy(), y.copy()
