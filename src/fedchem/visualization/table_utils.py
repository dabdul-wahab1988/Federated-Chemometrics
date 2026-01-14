from typing import Dict, Any, Optional
import pandas as pd
import numpy as np


FIELD_ALIASES = {
    'rmsep': ['RMSEP', 'rmse', 'rmse_p', 'rmsep'],
    'cvrmsep': ['CVRMSEP', 'cvrmsep', 'cv_rmsep', 'cv_rmse'],
    'bytes_sent': ['Bytes_Sent', 'bytes_sent', 'bytesSent'],
    'bytes_recv': ['Bytes_Received', 'bytes_recv', 'bytesRecv'],
    'epsilon': ['EpsilonSoFar', 'epsilon_so_far', 'epsilon'],
    'update_norm': ['UpdateNorm', 'update_norm'],
    'participation': ['ParticipationRate', 'participation_rate', 'participation'],
}


def _find_field(row: Dict[str, Any], aliases: list) -> Optional[float]:
    for a in aliases:
        if a in row:
            return row[a]
    return None


def manifest_to_table(manifest: Dict[str, Any]) -> pd.DataFrame:
    """Convert logs_by_algorithm to a DataFrame with method, round, and key metrics per-row.
    Expected manifest format: manifest['logs_by_algorithm'][method]['logs'] is a list of per-round dicts
    """
    if manifest is None:
        return pd.DataFrame()
    logs_by_alg = manifest.get('logs_by_algorithm') or {}
    rows = []
    for method, logs in logs_by_alg.items():
        method_logs = logs.get('logs') or []
        for l in method_logs:
            row = {
                'method': method,
                'round': l.get('Round', l.get('round', None)),
                'rmsep': _find_field(l, FIELD_ALIASES['rmsep']),
                'cvrmsep': _find_field(l, FIELD_ALIASES['cvrmsep']),
                'bytes_sent': _find_field(l, FIELD_ALIASES['bytes_sent']),
                'bytes_recv': _find_field(l, FIELD_ALIASES['bytes_recv']),
                'epsilon': _find_field(l, FIELD_ALIASES['epsilon']),
                'update_norm': _find_field(l, FIELD_ALIASES['update_norm']),
                'participation': _find_field(l, FIELD_ALIASES['participation']),
            }
            rows.append(row)
    df = pd.DataFrame(rows)
    # ensure types
    if not df.empty:
        for col in ['rmsep', 'cvrmsep', 'bytes_sent', 'bytes_recv', 'epsilon', 'update_norm', 'participation']:
            series = df.get(col)
            if series is None:
                df[col] = pd.Series([None] * len(df))
            else:
                df[col] = pd.to_numeric(series, errors='coerce')
    return df


def extract_final_rmsep(manifest: Dict[str, Any], method: str) -> float:
    """Get final RMSEP for method from manifest. Looks in summary then logs. Returns np.nan if not found.
    """
    if manifest is None:
        return np.nan
    summary = manifest.get('summary') or {}
    if method in summary:
        m = summary[method]
        for possible in ['RMSEP', 'rmsep', 'final_rmsep']:
            if possible in m:
                try:
                    return float(m[possible])
                except Exception:
                    pass
    # fallback: last log
    logs_by_alg = manifest.get('logs_by_algorithm', {})
    method_logs = (logs_by_alg.get(method) or {}).get('logs', [])
    if method_logs:
        last = method_logs[-1]
        for key in FIELD_ALIASES['rmsep']:
            if key in last:
                try:
                    return float(last[key])
                except Exception:
                    continue
    return np.nan
