#!/usr/bin/env python3
"""
Quick configuration validation check.

Run this to validate your config.yaml before running experiments.
"""
import sys
from pathlib import Path

# Add src to path if needed
src_path = Path(__file__).parent.parent / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

from fedchem.utils.config import (
    load_config,
    validate_config,
    validate_site_config,
    get_experimental_sites,
    get_data_config,
)


def main():
    print("\n" + "="*70)
    print("Configuration Validation Report")
    print("="*70 + "\n")
    
    # Load config
    cfg = load_config()
    
    if not cfg:
        print("❌ ERROR: Could not load config.yaml")
        return 1
    
    print("✅ Loaded config.yaml successfully\n")
    
    # Get basic info
    sites = get_experimental_sites(cfg)
    data_cfg = get_data_config(cfg)
    
    print(f"📍 Experimental Sites: {sites}")
    print(f"📊 DATA_CONFIG entries: {list(data_cfg.keys()) if data_cfg else 'None'}\n")
    
    # Run validation
    print("-" * 70)
    print("Running Full Validation...")
    print("-" * 70)
    
    try:
        warnings = validate_config(cfg, strict=False)
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR during validation:")
        print(f"   {e}\n")
        return 1
    
    # Categorize warnings
    critical = []
    info = []
    
    for w in warnings:
        if any(x in w.lower() for x in ["no data_config entry", "missing data_config", "not defined"]):
            critical.append(w)
        else:
            info.append(w)
    
    # Report results
    if not warnings:
        print("\n✅ Configuration is VALID - No warnings detected!\n")
        print("   You can proceed with running experiments.\n")
        return 0
    
    print(f"\n⚠️  Found {len(warnings)} warning(s):\n")
    
    if critical:
        print("🚫 CRITICAL Warnings (fix before running experiments):")
        for i, w in enumerate(critical, 1):
            print(f"   {i}. {w}")
        print()
    
    if info:
        print("ℹ️  Informational Warnings (review recommended):")
        for i, w in enumerate(info, 1):
            print(f"   {i}. {w}")
        print()
    
    # Provide recommendations
    print("-" * 70)
    print("Recommendations:")
    print("-" * 70)
    
    if critical:
        print("❗ Fix critical warnings before running experiments")
        print("   - Ensure all sites in EXPERIMENTAL_DESIGN.FACTORS.Site")
        print("     have corresponding DATA_CONFIG entries")
    
    if info:
        print("💡 Review informational warnings:")
        print("   - Check if unused DATA_CONFIG entries are intentional")
        print("   - Verify schedule lengths match ROUNDS configuration")
        print("   - Confirm disabled sites are removed from factors")
    
    print("\n📖 For detailed guidance, see:")
    print("   docs/DATA_CONFIG_BEST_PRACTICES.md")
    print()
    
    return 1 if critical else 0


if __name__ == "__main__":
    sys.exit(main())
