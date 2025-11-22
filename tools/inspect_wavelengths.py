#!/usr/bin/env python3
"""
Inspect Excel files under `data/` and heuristically detect spectral columns (wavelengths).
Prints a summary and recommends DEFAULT_N_WAVELENGTHS.

Usage:
  python tools/inspect_wavelengths.py

This script is a developer utility for quickly validating the number of spectral
channels across files / manufacturers.
"""
from __future__ import annotations

import pandas as pd
from pathlib import Path
import re

BASE = Path("data")

# Heuristics for spectral column names (common formats):
# - numeric column names (e.g. 350, 350.5)
# - start with 'w' or 'wl' followed by digits (e.g. w350, wl350)
# - column values numeric and column name not obviously metadata

WAVELENGTH_NAME_REGEX = re.compile(r"^(?:w|wl)?\s*-?\d+(?:\.\d+)?$", re.IGNORECASE)
METADATA_KEYS = {
    'id', 'sample', 'sampleid', 'sample_id', 'site', 'instrument', 'C_Solute',
    'protein', 'target', 'y', 'label', 'region', 'type'
}


def is_numeric_colname(name: str) -> bool:
    try:
        float(str(name))
        return True
    except Exception:
        return False


def name_matches_wavelength(name: str) -> bool:
    if not isinstance(name, str):
        return False
    name_stripped = name.strip()
    if WAVELENGTH_NAME_REGEX.match(name_stripped):
        return True
    if name_stripped.lower().startswith(('w', 'wl')) and any(d.isdigit() for d in name_stripped):
        return True
    return False


def detect_spectral_columns(df: pd.DataFrame) -> list[str]:
    cols = list(df.columns)
    spectral = []
    for c in cols:
        cstr = str(c)
        # skip obvious metadata keys
        if cstr.lower() in METADATA_KEYS:
            continue
        # check name heuristics
        if name_matches_wavelength(cstr) or is_numeric_colname(cstr):
            spectral.append(c)
            continue
        # otherwise, fallback to dtype check (numeric columns that are not obviously metadata):
        try:
            if pd.api.types.is_numeric_dtype(df[c]):
                # skip columns with small variance or single-valued metadata often loaded as numeric
                if df[c].nunique(dropna=True) > 3:
                    spectral.append(c)
        except Exception:
            pass
    return spectral


def inspect_file(path: Path):
    try:
        df = pd.read_excel(path, header=0)
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None
    cols = list(df.columns)
    spec = detect_spectral_columns(df)
    return {
        'path': str(path),
        'shape': df.shape,
        'total_cols': len(cols),
        'spectral_cols': len(spec),
        'spectral_names': spec[:12],
    }


def inspect_all(base: Path) -> list[dict]:
    xlsx_files = list(base.rglob('*.xlsx'))
    if not xlsx_files:
        print('No .xlsx files found under', base)
        return []
    out = []
    for f in xlsx_files:
        r = inspect_file(f)
        if r is not None:
            out.append(r)
    return out


def recommend_default(counts: list[int]) -> dict:
    if not counts:
        return {'recommended': None, 'reason': 'no counts'}
    unique = sorted(set(counts))
    if len(unique) == 1:
        return {'recommended': unique[0], 'reason': 'all files share the same spectral channel count'}
    # If majority are same (mode), recommend mode; else recommend min as safe value
    from collections import Counter
    c = Counter(counts)
    mode, mode_count = c.most_common(1)[0]
    total = sum(c.values())
    if mode_count / total >= 0.6:
        return {'recommended': mode, 'reason': f'mode (>=60% of files have {mode} channels)'}
    # else recommend min with explanation
    return {'recommended': min(unique), 'reason': 'variable counts; choosing smallest to avoid missing channels across files'}


def main():
    print('Inspecting .xlsx files under', BASE)
    results = inspect_all(BASE)
    if not results:
        return
    counts = [r['spectral_cols'] for r in results]
    print('\nFiles inspected:')
    for r in results:
        print(f" - {r['path']}: shape={r['shape']} total_cols={r['total_cols']} spectral_cols={r['spectral_cols']} sample_names={r['spectral_names']}")
    print('\nSummary of spectral column counts:')
    unique_counts = sorted(set(counts))
    print('Counts (unique):', unique_counts)
    print('Counts distribution:')
    from collections import Counter
    print(Counter(counts))
    rec = recommend_default(counts)
    print('\nRecommended DEFAULT_N_WAVELENGTHS (int):', rec['recommended'])
    print('Recommendation reason:', rec['reason'])


if __name__ == '__main__':
    main()
