#!/usr/bin/env python3
"""Utility to inspect FedPLS failures and debug-dumped weights.

Usage:
  python tools/diagnose_fedpls.py --manifest <manifest.json>
  python tools/diagnose_fedpls.py --archive <generated_figures_tables_archive path>
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Optional


def load_manifest(p: Path) -> Optional[dict]:
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except Exception as e:
        print(f"Failed to load manifest {p}: {e}")
        return None


def analyze_manifest(mf: dict, manifest_path: Path, detailed: bool = False, z_thresh: float = 3.0):
    print(f"\nAnalyzing manifest: {manifest_path}")
    cfg = mf.get('config', {})
    run_info = cfg.get('pipeline', {})
    print(f"Model pipeline: {run_info.get('model')}")
    # Summary
    summary = mf.get('summary', {})
    print("Summary metrics: ")
    for k, v in summary.items():
        if isinstance(v, dict):
            final = v.get('final', {})
            print(f"  {k}: final rmsep={final.get('rmsep')} r2={final.get('r2')} ma={final.get('mae')} rounds={v.get('statistics', {}).get('n_rounds')}")

    # Inspect FedPLS logs
    logs = mf.get('logs_by_algorithm', {}).get('FedPLS', [])
    if not logs:
        print("No FedPLS logs present in manifest.")
    else:
        print(f"Found {len(logs)} FedPLS log(s)")
        for lg in logs:
            eval_err = lg.get('eval_error') or lg.get('eval_err') or None
            agg_method = lg.get('agg_method') or lg.get('aggregation_method') or None
            agg_trim_frac = lg.get('agg_trim_frac') or lg.get('aggregation_trim_frac') or None
            print(f"  Round {lg.get('round')}: rmsep={lg.get('rmsep')} r2={lg.get('r2')} bytes_sent={lg.get('bytes_sent')} bytes_recv={lg.get('bytes_recv')} participants={lg.get('participants')} compression_ratio={lg.get('compression_ratio')} agg_method={agg_method} agg_trim_frac={agg_trim_frac} eval_error={eval_err}")

    # Look for debug weight files
    lca_dir = cfg.get('lca_artifact_dir')
    # fallback to env-sourced path if not present
    possible_dirs = []
    if lca_dir:
        possible_dirs.append(Path(lca_dir))
    # Also consider the manifest's parent dir padding (use generated_figures_tables path next to manifest)
    parent_dir = manifest_path.parent.parent
    possible_dirs.append(parent_dir / 'generated_figures_tables' / 'lca_artifacts')
    possible_dirs.append(manifest_path.parent / 'lca_artifacts')
    found_any = False
    files_by_algo = {}
    for d in possible_dirs:
        if d.exists() and d.is_dir():
            found_any = True
            print(f"Looking into lca_artifacts dir: {d}")
            for p in d.rglob('debug_weight_*.json'):
                try:
                    jd = json.loads(p.read_text(encoding='utf-8'))
                    ts = jd.get('timestamp')
                    algo = jd.get('algo')
                    round_ = jd.get('round', 'N/A')
                    cidx = jd.get('client_index')
                    params = jd.get('params', {})
                    w = params.get('w')
                    b = params.get('b')
                    norm = None
                    if w:
                        try:
                            import numpy as _np
                            norm = float(_np.linalg.norm(_np.asarray(w, dtype=float)))
                        except Exception:
                            norm = None
                    print(f"  debug file: {p.name} algo={algo} round={round_} client={cidx} w_norm={norm} len={len(w) if w else 0}")
                    if detailed and w:
                        files_by_algo.setdefault(algo, []).append((p, jd))
                except Exception as e:
                    print(f"  Could not read weight file {p}: {e}")
    if not found_any:
        print("No lca_artifacts debug weight files found; rerun with FEDCHEM_DEBUG_DUMP_WEIGHTS=1 to enable weight dumps.")

    else:
        # Optionally run detailed per-dimension analysis
        if detailed:
            run_detailed_analysis(files_by_algo, z_thresh=z_thresh, manifest_path=manifest_path)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=str, help='Path to manifest JSON to analyze')
    p.add_argument('--archive', type=str, help='Path to archive (top-level generated_figures_tables_archive) to search for manifests')
    p.add_argument('--detailed', action='store_true', help='Run detailed per-dimension anomaly analysis')
    p.add_argument('--z', type=float, default=3.0, help='Z-score threshold to detect per-dimension anomalies (default: 3.0)')
    args = p.parse_args()
    if args.manifest:
        mp = Path(args.manifest)
        mf = load_manifest(mp)
        if mf:
            analyze_manifest(mf, mp, detailed=args.detailed, z_thresh=args.z)
    elif args.archive:
        root = Path(args.archive)
        if not root.exists():
            print(f"Archive path {root} not found")
            return
        for pth in root.rglob('manifest_*.json'):
            mf = load_manifest(pth)
            if mf:
                analyze_manifest(mf, pth, detailed=args.detailed, z_thresh=args.z)
    else:
        print("Provide either --manifest <path> or --archive <path>")


def run_detailed_analysis(files_by_algo: dict, z_thresh: float = 3.0, manifest_path: Optional[Path] = None):
    """Produce per-dimension statistics and detect anomalous coefficients across clients.

    For each algorithm (e.g., 'fedavg'), find debug weight files, align their parameter vectors and compute
    mean/std per dimension. Report which clients/dimensions have coefficients with absolute z-score > z_thresh.
    """
    import numpy as _np
    import csv

    for algo, pairs in files_by_algo.items():
        print(f"\nDetailed analysis for algorithm: {algo}")
        if not pairs:
            print("  No files to analyze")
            continue
        # Load weights grouped by client index
        w_by_client = {}
        scaler_by_client = {}
        for p, jd in pairs:
            params = jd.get('params', {})
            w = params.get('w')
            if not w:
                continue
            client = int(jd.get('client_index', -1))
            w_arr = _np.asarray(w, dtype=float)
            w_by_client.setdefault(client, []).append((p, w_arr))
            sc = jd.get('scaler')
            if sc is not None:
                scaler_by_client.setdefault(client, sc)
            # collect additional metadata
            n_s = jd.get('n_samples')
            n_c = None
            if sc is not None and isinstance(sc, dict) and sc.get('n_components') is not None:
                n_c = int(sc.get('n_components'))
            if n_s is not None:
                scaler_by_client.setdefault(client, {})['n_samples'] = int(n_s)
            if n_c is not None:
                scaler_by_client.setdefault(client, {})['n_components'] = int(n_c)
        # For each client pick the snapshot with the largest L2 norm (worst-case weight)
        rep_w = {}
        for cidx, wlist in w_by_client.items():
            norms = [_np.linalg.norm(w) for _, w in wlist]
            max_idx = int(_np.argmax(norms))
            rep_w[cidx] = wlist[max_idx][1]
        if not rep_w:
            print("  No weight arrays collected; nothing to analyze")
            continue
        length_set = {w.shape[0] for w in rep_w.values()}
        if len(length_set) != 1:
            print(f"  Inconsistent parameter vector sizes across clients: {sorted(length_set)}")
            continue
        d = next(iter(length_set))
        clients_sorted = sorted(rep_w.keys())
        W = _np.vstack([rep_w[k] for k in clients_sorted])
        mean = _np.mean(W, axis=0)
        std = _np.std(W, axis=0, ddof=1)
        std_safe = _np.where(std < 1e-12, 1.0, std)
        anomalies = []
        # Compute L2 norms per client and detect L2 outliers
        norms = _np.linalg.norm(W, axis=1)
        global_norm_mean = float(_np.mean(norms))
        global_norm_std = float(_np.std(norms, ddof=1)) if len(norms) > 1 else 0.0
        norm_outlier_idxs = [i for i, n in enumerate(norms) if global_norm_std > 0 and (n - global_norm_mean) > (z_thresh * global_norm_std)]
        for i, cidx in enumerate(clients_sorted):
            w = rep_w[cidx]
            z = (w - mean) / std_safe
            idxs = _np.where((_np.abs(z) > z_thresh) | (_np.abs(w) > (5.0 * _np.mean(_np.abs(w)))))[0]
            if idxs.size > 0:
                anomalies.append((cidx, idxs.tolist(), w[idxs].tolist(), z[idxs].tolist()))
        print(f"  Collected weights from {len(clients_sorted)} clients; feature dim={d}")
        print(f"  Per-client L2 norms: mean={global_norm_mean:.3f} std={global_norm_std:.3f} max={float(_np.max(norms)):.3f}")
        if norm_outlier_idxs:
            print("  Clients with outlying L2 norms:")
            for i in norm_outlier_idxs:
                ci = clients_sorted[i]
                print(f"    Client {ci}: norm={float(norms[i]):.3f}")
        print(f"  Dimensions with anomalies (z>{z_thresh} or >5x mean abs coefficient):")
        for cidx, idxs, vals, zs in anomalies:
            print(f"    Client {cidx}: {len(idxs)} anomalous dims. Showing up to 10 dims:")
            for j, (idx, v, zval) in enumerate(zip(idxs[:10], vals[:10], zs[:10])):
                print(f"      dim {idx}: coeff={v:.4f} z={zval:.2f} scaler_mean={_format_scaler_value(scaler_by_client.get(cidx, {}).get('mean', None), idx)} scaler_scale={_format_scaler_value(scaler_by_client.get(cidx, {}).get('scale', None), idx)}")
            # print metadata
            meta = scaler_by_client.get(cidx, {})
            if meta.get('n_samples'):
                print(f"      metadata: n_samples={meta.get('n_samples')}")
            if meta.get('n_components'):
                print(f"      metadata: n_components={meta.get('n_components')}")
        if not anomalies:
            print("    No anomalies detected for this algorithm at current z-threshold.")
        try:
            if manifest_path is not None:
                outdir = manifest_path.parent
            else:
                outdir = Path.cwd()
            csv_path = outdir / f"detailed_fedpls_dump_{algo}.csv"
            with open(csv_path, 'w', newline='', encoding='utf-8') as fh:
                writer = csv.writer(fh)
                header = ['client', *[f'dim_{i}' for i in range(d)]]
                writer.writerow(header)
                for cidx in clients_sorted:
                    writer.writerow([cidx, *rep_w[cidx].tolist()])
            print(f"  Dumped detailed weight vectors to: {csv_path}")
        except Exception as e:
            print(f"  Failed to write CSV report: {e}")


def _format_scaler_value(arr, idx):
    try:
        if arr is None:
            return 'N/A'
        return f"{float(arr[idx]):.6f}"
    except Exception:
        return 'N/A'


if __name__ == '__main__':
    main()
