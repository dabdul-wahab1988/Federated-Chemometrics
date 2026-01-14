"""
Utility function to extract and prepare logs for manifest saving.
This ensures consistent logging across all objectives.
"""

def extract_logs_for_manifest(results_dict):
    """
    Extract logs from FederatedOrchestrator results and convert to JSON-serializable format.
    
    Args:
        results_dict: Dictionary with algorithms as keys, results as values
                     Each result should have a 'logs' key containing list of RoundLog objects
    
    Returns:
        logs_by_algorithm: Dictionary with algorithm names as keys, lists of log dicts as values
    
    Example:
        results = {
            'FedAvg': orchestrator.run_rounds(...),
            'FedProx': orchestrator.run_rounds(...),
        }
        logs_by_algo = extract_logs_for_manifest(results)
        manifest['logs_by_algorithm'] = logs_by_algo
    """
    logs_by_algorithm = {}
    
    for algo_name, result in results_dict.items():
        if not result:
            continue
            
        algo_logs = result.get("logs", [])
        
        # Convert RoundLog objects to dicts for JSON serialization
        logs_list = []
        for log_entry in algo_logs:
            if isinstance(log_entry, dict):
                logs_list.append(log_entry)
            elif hasattr(log_entry, '__dict__'):
                # Convert dataclass to dict
                logs_list.append(log_entry.__dict__)
            else:
                # Fallback: try to convert to dict
                try:
                    logs_list.append(vars(log_entry))
                except:
                    pass
        
        if logs_list:
            logs_by_algorithm[algo_name] = logs_list
    
    return logs_by_algorithm


def create_log_summary(logs_by_algorithm):
    """
    Create a summary of key metrics from logs for quick reference.
    
    Returns:
        Dictionary with aggregated statistics
    """
    summary = {}
    
    for algo_name, logs in logs_by_algorithm.items():
        if not logs:
            continue
        
        # Get final/last values
        final_log = logs[-1] if logs else {}
        
        # Get min/max/mean values across all rounds
        rmseps = [log.get('rmsep') for log in logs if log.get('rmsep') is not None]
        cvrmseps = [log.get('cvrmsep') for log in logs if log.get('cvrmsep') is not None]
        r2s = [log.get('r2') for log in logs if log.get('r2') is not None]
        maes = [log.get('mae') for log in logs if log.get('mae') is not None]
        bytes_sent = [log.get('bytes_sent', 0) for log in logs]
        bytes_recv = [log.get('bytes_recv', 0) for log in logs]
        # Coerce missing or None pds_bytes to 0 to avoid TypeError when summing
        pds_bytes_list = [int(log.get('pds_bytes') or 0) for log in logs]
        
        summary[algo_name] = {
            "final": {
                "round": final_log.get('round'),
                "rmsep": final_log.get('rmsep'),
                "cvrmsep": final_log.get('cvrmsep'),
                "r2": final_log.get('r2'),
                "mae": final_log.get('mae'),
                "epsilon_so_far": final_log.get('epsilon_so_far'),
                "duration_sec": final_log.get('duration_sec'),
            },
            "statistics": {
                "rmsep_final": rmseps[-1] if rmseps else None,
                "rmsep_min": min(rmseps) if rmseps else None,
                "rmsep_max": max(rmseps) if rmseps else None,
                "rmsep_mean": sum(rmseps)/len(rmseps) if rmseps else None,
                "cvrmsep_final": cvrmseps[-1] if cvrmseps else None,
                "cvrmsep_min": min(cvrmseps) if cvrmseps else None,
                "cvrmsep_max": max(cvrmseps) if cvrmseps else None,
                "cvrmsep_mean": sum(cvrmseps)/len(cvrmseps) if cvrmseps else None,
                "r2_final": r2s[-1] if r2s else None,
                "r2_min": min(r2s) if r2s else None,
                "r2_max": max(r2s) if r2s else None,
                "mae_final": maes[-1] if maes else None,
                "mae_min": min(maes) if maes else None,
                "mae_max": max(maes) if maes else None,
                "total_bytes_sent": sum(bytes_sent),
                "total_bytes_recv": sum(bytes_recv),
                "total_pds_bytes": sum(pds_bytes_list),
                "total_bytes": sum(bytes_sent) + sum(bytes_recv),
                "total_bytes_including_pds": sum(bytes_sent) + sum(bytes_recv) + sum(pds_bytes_list),
                "total_bytes_mb": (sum(bytes_sent) + sum(bytes_recv)) / (1024 * 1024),
                "total_bytes_including_pds_mb": (sum(bytes_sent) + sum(bytes_recv) + sum(pds_bytes_list)) / (1024 * 1024),
                "n_rounds": len(logs),
            }
        }
    
    return summary
