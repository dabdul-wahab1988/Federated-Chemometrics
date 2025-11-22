import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def plot_rmsep_vs_k(results_root: Path, out_png: Path | None = None):
    results_root = Path(results_root)
    rows = []
    for sub in results_root.iterdir():
        if not sub.is_dir():
            continue
        f = sub / 'federated_results.json'
        if not f.exists():
            continue
        j = json.loads(f.read_text())
        for s in j.get('sites', []):
            rows.append({'run': sub.name, 'site': s.get('site_id'), 'rmsep': s.get('rmsep_pds'), 'privacy': s.get('privacy_recon_error')})
    if not rows:
        print('No results found')
        return
    import pandas as pd
    df = pd.DataFrame(rows)
    # try to extract k from run name pattern k_{k}_preproc_{p}
    df['k'] = df['run'].str.extract(r'k_(\d+)_').astype(float)
    agg = df.groupby('k')['rmsep'].mean().reset_index()
    plt.figure()
    plt.plot(agg['k'], agg['rmsep'], marker='o')
    plt.xlabel('k (transfer samples)')
    plt.ylabel('Mean RMSEP (PDS)')
    plt.title('RMSEP vs k')
    plt.grid(True)
    if out_png:
        plt.savefig(out_png, dpi=200)
    else:
        plt.show()
