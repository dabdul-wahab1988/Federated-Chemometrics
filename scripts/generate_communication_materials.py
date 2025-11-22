#!/usr/bin/env python3
"""
Master script to generate all communication materials for experimental results.

This script orchestrates the creation of figures, tables, and reports for stakeholders.
"""

import subprocess
import sys
from pathlib import Path
import os

def run_script(script_name: str, description: str):
    """Run a Python script and report success/failure."""
    print(f"🔄 {description}...")

    try:
        result = subprocess.run([sys.executable, script_name],
                              capture_output=True, text=True, check=True)
        print(f"[OK] {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Error running {script_name}: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False

def main():
    """Generate all communication materials."""
    print("[START] Starting communication materials generation...\n")

    # Check if required files exist
    required_files = [
        "create_communication_figures.py",
        "generate_summary_report.py"
    ]

    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)

    if missing_files:
        print(f"❌ Missing required scripts: {', '.join(missing_files)}")
        return

    # Check if archive directory exists
    archive_root = Path(os.environ.get("FEDCHEM_ARCHIVE_ROOT", "generated_figures_tables_archive"))
    if not archive_root.exists():
        print(f"[ERROR] Archive directory '{archive_root}' not found.")
        print("   Run the experimental pipeline first to generate results.")
        return

    # Run figure generation
    success1 = run_script("create_communication_figures.py",
                         "Creating communication figures and tables")

    if not success1:
        print("[ERROR] Figure generation failed. Stopping.")
        return

    # Run report generation
    success2 = run_script("generate_summary_report.py",
                         "Generating summary report")

    if not success2:
        print("[ERROR] Report generation failed.")
        return

    print("\n[SUCCESS] All communication materials generated successfully!")
    print("\n[INFO] Generated files in 'communication_figures/':")
    print("  [FIGURE] privacy_performance_heatmap.png")
    print("  [FIGURE] transfer_sample_impact.png")
    print("  [FIGURE] communication_efficiency.png")
    print("  [TABLE] experimental_results_summary.csv")
    print("  [TABLE] benchmarking_comparison.csv")
    print("  [TABLE] prediction_intervals.csv")
    print("  [REPORT] experimental_summary_report.md")

    print("\n[TIP] Usage recommendations:")
    print("  - Use the heatmap for quick overview of privacy-performance trade-offs")
    print("  - Use the impact analysis for detailed performance vs sample size analysis")
    print("  - Use the efficiency plot for communication cost analysis")
    print("  - Use the markdown report for stakeholder presentations")
    print("  - Use the CSV files for further analysis or custom visualizations")

if __name__ == "__main__":
    main()