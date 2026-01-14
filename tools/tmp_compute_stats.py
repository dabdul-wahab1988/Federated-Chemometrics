import pandas as pd
from pathlib import Path
import sys

def find_protein_col(df):
    cols = list(df.columns)
    cand = [c for c in cols if 'protein' in c.lower()]
    if not cand:
        cand = [c for c in cols if c.lower() in {'ref', 'reference', 'y', 'target'}]
    if not cand:
        return None
    return cand[0]

files = [Path('data/MA_A2.csv'), Path('data/MB_B2.csv')]
series = {}
for f in files:
    df = pd.read_csv(f)
    col = find_protein_col(df)
    if col is None:
        print(f"No protein-like column found in {f}, columns={df.columns[:10].tolist()}")
        sys.exit(1)
    s = pd.to_numeric(df[col], errors='coerce').dropna()
    series[f.name] = s
    print(f"{f.name}: using column '{col}', n={len(s)}, mean={s.mean():.4f}, sd={s.std(ddof=1):.4f}")

pooled = pd.concat(series.values(), ignore_index=True)
print(f"Pooled: n={len(pooled)}, mean={pooled.mean():.4f}, sd={pooled.std(ddof=1):.4f}")
