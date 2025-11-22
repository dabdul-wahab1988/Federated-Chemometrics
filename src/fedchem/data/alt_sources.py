from __future__ import annotations

from pathlib import Path
import re
import urllib.request
import numpy as np
from scipy.io import savemat


def fetch_tecator_from_libstat(dest: Path) -> Path:
    """Fetch the Tecator dataset from lib.stat.cmu.edu (ASCII page) and write a .mat file.

    This is a fallback for when the original .mat URL returns HTML or is unavailable.
    The function writes a MAT file with variables 'absorbance' (240x100) and 'fat' (240,).
    Returns the path to the written file.
    """
    url = "http://lib.stat.cmu.edu/datasets/tecator"
    dest.parent.mkdir(parents=True, exist_ok=True)
    # download page
    with urllib.request.urlopen(url) as resp:
        data = resp.read().decode("utf-8", errors="ignore")

    lines = data.splitlines()
    # robust float regex
    float_re = re.compile(r'[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[Ee][+-]?\d+)?')
    tokens = []
    for ln in lines:
        toks = float_re.findall(ln)
        if toks:
            tokens.extend(toks)

    vals = np.array([float(t) for t in tokens], dtype=np.float64)
    expected = 240 * 125
    if vals.size != expected:
        if vals.size > expected:
            vals = vals[-expected:]
        else:
            raise ValueError(f"Unexpected token count when parsing Tecator page: {vals.size}")

    arr = vals.reshape((240, 125))
    X = arr[:, :100].astype(np.float32)
    contents = arr[:, -3:]
    fat = contents[:, 1].astype(np.float32)
    savemat(dest, {"absorbance": X, "fat": fat})
    return dest
