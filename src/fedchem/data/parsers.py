from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from ..types import FloatArray
from scipy.io import loadmat


def parse_csv_generic(path: Path) -> Tuple[FloatArray, Optional[FloatArray], List[float], Dict[str, object]]:
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    X = data[:, :-1]
    y = data[:, -1]
    wavelengths = [float(i) for i in range(X.shape[1])]
    return X.astype(np.float64), y.astype(np.float64), wavelengths, {"format": "csv_generic"}


def parse_tecator_mat(path: Path) -> Tuple[FloatArray, Optional[FloatArray], List[float], Dict[str, object]]:
    mat = loadmat(path)
    # Try common keys; fall back to any 2D array with matching rows
    candidates = {k: v for k, v in mat.items() if isinstance(v, np.ndarray)}
    X = None
    y = None
    for k in ("absorbance", "nir", "X"):
        if k in candidates and candidates[k].ndim == 2:
            X = candidates[k]
            break
    for k in ("fat", "y", "target"):
        if k in candidates and candidates[k].ndim == 2:
            yy = candidates[k]
            y = yy.squeeze()
            break
    if X is None:
        raise ValueError("Could not find spectra matrix in MAT file")
    wavelengths = [float(i) for i in range(X.shape[1])]
    return X.astype(np.float64), None if y is None else y.astype(np.float64), wavelengths, {"format": "mat"}
