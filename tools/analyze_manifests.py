import json
import os
from pathlib import Path
import statistics

def find_manifests(root=None):
    root = Path(os.environ.get("FEDCHEM_ARCHIVE_ROOT", "generated_figures_tables_archive")) if root is None else Path(root)
    return list(root.rglob("manifest_*.json"))


def load_manifest(p):
    try:
        return json.loads(p.read_text(encoding="utf-8")), None
    except Exception as e:
        return None, e


THRESHOLD_MULT = 5.0


def analyze_manifest(manifest_path: Path):
    mf, err = load_manifest(manifest_path)
    if err:
        return {"path": str(manifest_path), "error": str(err)}
    out = {"path": str(manifest_path), "issues": []}
    config = mf.get("config", {})
    ds = config.get("dataset", "unknown")
    out["dataset"] = ds
    # per algorithm analysis
    logs_by_algo = mf.get("logs_by_algorithm", {})
    # check DP/infinite eps inconsistency
    target_eps = config.get("target_epsilon") if config else None
    if target_eps is None or (isinstance(target_eps, str) and target_eps.strip().lower() in {"null", "none"}):
        target_eps = None
    
    # If epsilon infinite but dp_noise present in logs -> flag
    if target_eps and (isinstance(target_eps, str) and target_eps.lower() == "infinity" or target_eps == float("inf") or str(target_eps).lower() == "inf"):
        # scan logs for dp_noise_std or epsilon_so_far
        for algo, logs in logs_by_algo.items():
            for l in logs:
                if l.get("dp_noise_std") is not None or l.get("epsilon_so_far") not in (None, "", 'null'):
                    out["issues"].append({"type": "dp_noise_in_inf_run", "algo": algo, "round": l.get("round"), "dp_noise_std": l.get("dp_noise_std"), "epsilon_so_far": l.get("epsilon_so_far")})

    # per algo round-level anomalies
    for algo, logs in logs_by_algo.items():
        rlist = [l.get("rmsep") for l in logs if l.get("rmsep") is not None]
        if not rlist:
            continue
        median_rmsep = statistics.median(rlist)
        # for each log, find anomalies
        for l in logs:
            rmsep = l.get("rmsep")
            if rmsep is None:
                continue
            # absurdly large
            if median_rmsep > 0 and rmsep > THRESHOLD_MULT * median_rmsep:
                out["issues"].append({"type": "large_rmsep", "algo": algo, "round": l.get("round"), "rmsep": rmsep, "median_rmsep": median_rmsep, "participants": l.get("participants"), "participation_rate": l.get("participation_rate"), "compression_ratio": l.get("compression_ratio"), "dp_noise_std": l.get("dp_noise_std"), "sensitivity_round": l.get("sensitivity_round")})
            if isinstance(l.get("r2"), (int, float)) and l.get("r2") < -100:  # extremely negative r2
                out["issues"].append({"type": "large_negative_r2", "algo": algo, "round": l.get("round"), "r2": l.get("r2"), "rmsep": rmsep})

    # FedPLS check: compare to centralized PLS if both exist
    central = mf.get("summary", {}).get("Centralized_PLS")
    fedpls = mf.get("summary", {}).get("FedPLS")
    if central and fedpls:
        try:
            central_rmsep = float(central.get("rmsep")) if isinstance(central.get("rmsep"), (int, float)) else None
            fedpls_rmsep = float(fedpls.get("rmsep")) if isinstance(fedpls.get("rmsep"), (int, float)) else None
            if central_rmsep is not None and fedpls_rmsep is not None and fedpls_rmsep > max(5.0, 3 * central_rmsep):
                out["issues"].append({"type": "fedpls_bad_vs_central", "central_rmsep": central_rmsep, "fedpls_rmsep": fedpls_rmsep})
        except Exception:
            pass

    return out


if __name__ == "__main__":
    man_paths = find_manifests()
    print(f"Found {len(man_paths)} manifests")
    all_issues = []
    for p in man_paths:
        r = analyze_manifest(p)
        if r.get("issues"):
            all_issues.append(r)
    if not all_issues:
        print("No issues found by automated heuristics.")
    else:
        for r in all_issues:
            print("\nManifest:", r["path"])
            for i in r["issues"]:
                print("-", i)
    
```}{