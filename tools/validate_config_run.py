from fedchem.utils.config import load_config, validate_config

if __name__ == '__main__':
    cfg = load_config()
    warnings = validate_config(cfg, strict=False)
    print('WARNINGS:', warnings)
    factors = cfg.get('EXPERIMENTAL_DESIGN', {}).get('FACTORS', {})
    print('EXPERIMENTAL_DESIGN_FACTORS_KEYS:', list(factors.keys()))
    # Print a few specific values to confirm presence
    print('CONFORMAL keys:', list(cfg.get('CONFORMAL', {}).keys()))
    print('DIFFERENTIAL_PRIVACY keys:', list(cfg.get('DIFFERENTIAL_PRIVACY', {}).keys()))
    print('SECURE_AGGREGATION keys:', list(cfg.get('SECURE_AGGREGATION', {}).keys()))
    print('DRIFT_AUGMENT keys:', list(cfg.get('DRIFT_AUGMENT', {}).keys()))
    print('REPRODUCIBILITY keys:', list(cfg.get('REPRODUCIBILITY', {}).keys()))
