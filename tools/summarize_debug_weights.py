#!/usr/bin/env python3
"""Summarize debug weights dumped during Federated runs.

Walks a debug_weights dir (by default in generated_figures_tables/lca_artifacts/debug_weights)
and computes L2 norm summaries per client and overall extremes. Prints a short report.
"""
from __future__ import annotations

import json
import math
import argparse
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List


def summarize(dirpath: Path):
    files = sorted(dirpath.glob('debug_weight_*.json'))
    if not files:
        print(f"No debug weight files found in {dirpath}")
        return
    per_client_vals: Dict[int, List[float]] = {}
    per_client_files: Dict[int, List[tuple[str, float]]] = {}
    for p in files:
        try:
            j = json.loads(p.read_text(encoding='utf-8'))
        except Exception as e:
            print(f"Failed to parse {p}: {e}")
            continue
        params = j.get('params', {})
        w = params.get('w')
        if not w:
            continue
        try:
            norm = math.sqrt(sum((float(x) ** 2) for x in w))
        except Exception:
            norm = float('nan')
        client = j.get('client_index') if j.get('client_index') is not None else -1
        per_client_vals.setdefault(int(client), []).append(norm)
        per_client_files.setdefault(int(client), []).append((p.name, norm))

    # Print per-client stats
    print(f"Found {len(files)} debug weight files. Summarizing norms per client:")
    global_list = []
    for client, vals in sorted(per_client_vals.items()):
        clean = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
        if not clean:
            continue
        global_list.extend(clean)
        print(f"  Client {client}: count={len(clean)} mean={mean(clean):.3f} std={(stdev(clean) if len(clean) > 1 else 0.0):.3f} min={min(clean):.3f} max={max(clean):.3f}")
        # Print top N files per client
        topn = 5
        files = sorted(per_client_files.get(client, []), key=lambda x: x[1], reverse=True)
        for fname, fnorm in files[:topn]:
            print(f"    {fname}: norm={fnorm:.3f}")
    if global_list:
        print(f"\nGlobal: count={len(global_list)} mean={mean(global_list):.3f} std={(stdev(global_list) if len(global_list) > 1 else 0.0):.3f} min={min(global_list):.3f} max={max(global_list):.3f}")
        # Top N global
        all_files = []
        for cidx, flist in per_client_files.items():
            for fname, fnorm in flist:
                all_files.append((cidx, fname, fnorm))
        for cidx, fname, fnorm in sorted(all_files, key=lambda x: x[2], reverse=True)[:5]:
            print(f"  client {cidx} {fname}: norm={fnorm:.3f}")
    else:
        print("No numeric norms computed (weights missing or unparsable)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dir', default=None, help='Path to debug_weights dir. If unset, uses env FEDCHEM_OUTPUT_DIR or ./')
    p.add_argument('--top', type=int, default=5, help='Show top N debug files by norm per client and overall')
    args = p.parse_args()
    dirpath = Path(args.dir) if args.dir else Path.cwd() / 'generated_figures_tables' / 'lca_artifacts' / 'debug_weights'
    if not dirpath.exists():
        print(f"Debug weights dir {dirpath} not found")
        return
    summarize(dirpath)


if __name__ == '__main__':
    main()
