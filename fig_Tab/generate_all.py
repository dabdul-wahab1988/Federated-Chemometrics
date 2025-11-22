from __future__ import annotations

import argparse
from pathlib import Path

from .style import style_from_args
from .tables import generate_all_tables
from .figures import generate_all_figures


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Generate manuscript tables and figures from final results.",
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=Path("final2"),
        help="Root results directory (default: final2 - will use final2/publication_database)",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to config.yaml",
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Root of raw data/metadata (default: ./data)",
    )
    p.add_argument(
        "--tables-dir",
        type=Path,
        default=None,
        help="Output directory for tables (default: <results>/tables_manuscript)",
    )
    p.add_argument(
        "--figures-dir",
        type=Path,
        default=None,
        help="Output directory for figures (default: <results>/figures_manuscript)",
    )
    p.add_argument(
        "--figure-dpi",
        type=int,
        default=300,
        help="DPI for generated figures (default: 300)",
    )
    p.add_argument(
        "--font-size",
        type=float,
        default=10.0,
        help="Base font size for figures (default: 10)",
    )
    p.add_argument(
        "--line-width",
        type=float,
        default=1.5,
        help="Default line width for plots (default: 1.5)",
    )
    p.add_argument(
        "--marker-size",
        type=float,
        default=5.0,
        help="Default marker size for plots (default: 5.0)",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    results_root = args.results_dir
    config_path = args.config
    data_dir = args.data_dir

    tables_dir = args.tables_dir or (results_root / "tables_manuscript")
    figures_dir = args.figures_dir or (results_root / "figures_manuscript")

    style = style_from_args(args)

    generate_all_tables(
        results_root=results_root,
        config_path=config_path,
        tables_dir=tables_dir,
        data_dir=data_dir,
        n_wavelengths=None,
    )
    generate_all_figures(
        results_root=results_root,
        config_path=config_path,
        data_dir=data_dir,
        figures_dir=figures_dir,
        style=style,
    )


if __name__ == "__main__":
    main()

