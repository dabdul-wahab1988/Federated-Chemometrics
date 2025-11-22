#!/usr/bin/env python
"""
Generate manuscript-ready tables as CSV from master_database_*.json.

Tables produced (CSV files):

  table1_instruments.csv
  table2_experimental_design.csv

  # NEW: includes ALL transfer_k values
  table3_baseline_non_dp_all_k.csv
  + optional: table3a_baseline_non_dp_kXX.csv per transfer_k

  table4_privacy_tradeoff_fed.csv
  table5_conformal_summary.csv

Usage (from directory containing the JSON file):

    python make_tables_from_master_db_v2.py \
        --db master_database_20251120_200830.json \
        --outdir tables

Requires:
    pandas, numpy
"""

import json
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from typing import Any, Dict


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def load_master_db(path: Path) -> pd.DataFrame:
    with open(path, "r") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    return df


def add_dp_helpers(df: pd.DataFrame) -> pd.DataFrame:
    """Add dp_label (string) and dp_numeric (float) columns."""
    df = df.copy()

    def _label_eps(x):
        if pd.isna(x):
            return "inf"
        xf = float(x)
        return str(int(xf)) if xf.is_integer() else str(xf)

    df["dp_label"] = df["dp_epsilon"].apply(_label_eps)
    df["dp_numeric"] = df["dp_epsilon"].fillna(np.inf)
    return df


def _uniq_sorted(series):
    vals = [v for v in series.dropna().unique().tolist()]
    try:
        vals_sorted = sorted(vals)
    except TypeError:
        vals_sorted = sorted(vals, key=lambda x: str(x))
    return vals_sorted


# ----------------------------------------------------------------------
# Table 1 – Instrument registry and default splits
# ----------------------------------------------------------------------

def make_table1_instruments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Table 1: Representative instruments and default data splits.

    Columns:
        Instrument_Code
        Manufacturer
        Instrument_ID
        Site_Code
        Used_as_Client    (True if this instrument appears as a client)
        Drift_Levels      (comma-separated unique)
        Drift_Types       (comma-separated unique)
        Uses_DP_Any       (True if any experiment uses DP on this instrument)
        Methods_Used      (semicolon-separated method names)
        Num_Experiments   (# rows in DB for this instrument)
        Train_Frac        (default config: 0.6)
        Cal_Frac          (default config: 0.2)
        Test_Frac         (default config: 0.2)
        Min_Samples_for_Calibration (default config: 20)
    """
    df = df.copy()
    instruments = []

    # default split constants from PER_INSTRUMENT_SPLIT in config.yaml
    TRAIN_FRAC = 0.6
    CAL_FRAC = 0.2
    TEST_FRAC = 0.2
    MIN_SAMPLES_FOR_CAL = 20

    for inst_code, group in df.groupby("instrument_code"):
        manufacturer = group["manufacturer"].iloc[0] if "manufacturer" in group else ""
        instrument_id = group["instrument_id"].iloc[0] if "instrument_id" in group else ""
        site_codes = group["site_code"].dropna().unique()
        site_code = site_codes[0] if len(site_codes) > 0 else ""

        drift_levels = sorted(
            [str(x) for x in group["drift_level"].dropna().unique()]
        )
        drift_types = sorted(
            [str(x) for x in group["drift_type"].dropna().unique()]
        )

        uses_dp_any = bool(group["uses_dp"].any()) if "uses_dp" in group else False

        methods_used = sorted(group["method"].unique().tolist())
        methods_str = "; ".join(methods_used)

        num_experiments = int(len(group))

        instruments.append(
            {
                "Instrument_Code": inst_code,
                "Manufacturer": manufacturer,
                "Instrument_ID": instrument_id,
                "Site_Code": site_code,
                "Used_as_Client": True,  # both instruments are clients in this subset
                "Drift_Levels": ", ".join(drift_levels) if drift_levels else "",
                "Drift_Types": ", ".join(drift_types) if drift_types else "",
                "Uses_DP_Any": uses_dp_any,
                "Methods_Used": methods_str,
                "Num_Experiments": num_experiments,
                "Train_Frac": TRAIN_FRAC,
                "Cal_Frac": CAL_FRAC,
                "Test_Frac": TEST_FRAC,
                "Min_Samples_for_Calibration": MIN_SAMPLES_FOR_CAL,
            }
        )

    table1 = pd.DataFrame(instruments)
    table1 = table1.sort_values("Instrument_Code").reset_index(drop=True)
    return table1


# ----------------------------------------------------------------------
# Table 2 – Experimental-design factors and method families
# ----------------------------------------------------------------------

def make_table2_experimental_design(df: pd.DataFrame) -> pd.DataFrame:
    """
    Table 2: Experimental-design factors and method families.

    Columns:
        Factor_Group
        Factor_Name
        Values
        Notes
    """
    df = df.copy()
    df = add_dp_helpers(df)

    rows = []

    # Design / topology
    rows.append(
        {
            "Factor_Group": "Design / topology",
            "Factor_Name": "Instruments (clients)",
            "Values": ", ".join(_uniq_sorted(df["instrument_code"])),
            "Notes": "Each instrument is treated as a federated site (instrument-as-site).",
        }
    )
    rows.append(
        {
            "Factor_Group": "Design / topology",
            "Factor_Name": "Site codes",
            "Values": ", ".join(_uniq_sorted(df["site_code"])),
            "Notes": "Site codes match instrument codes in this subset.",
        }
    )
    rows.append(
        {
            "Factor_Group": "Design / topology",
            "Factor_Name": "Methods",
            "Values": ", ".join(_uniq_sorted(df["method"])),
            "Notes": "Includes site-specific, CT baselines (PDS/SBC), centralized PLS, and federated variants.",
        }
    )

    # Transfer / CT
    rows.append(
        {
            "Factor_Group": "Transfer / calibration",
            "Factor_Name": "Transfer_k",
            "Values": ", ".join(str(x) for x in _uniq_sorted(df["transfer_k"])),
            "Notes": "Number of transfer samples per direction (e.g. 20, 80, 200).",
        }
    )

    # Federated / optimisation
    rows.append(
        {
            "Factor_Group": "Federated / optimisation",
            "Factor_Name": "num_rounds",
            "Values": ", ".join(str(x) for x in _uniq_sorted(df["num_rounds"])),
            "Notes": "Number of federated communication rounds.",
        }
    )
    rows.append(
        {
            "Factor_Group": "Federated / optimisation",
            "Factor_Name": "local_epochs",
            "Values": ", ".join(str(x) for x in _uniq_sorted(df["local_epochs"])),
            "Notes": "Number of local epochs per round.",
        }
    )
    rows.append(
        {
            "Factor_Group": "Federated / optimisation",
            "Factor_Name": "batch_size",
            "Values": ", ".join(str(x) for x in _uniq_sorted(df["batch_size"])),
            "Notes": "Local mini-batch size.",
        }
    )
    rows.append(
        {
            "Factor_Group": "Federated / optimisation",
            "Factor_Name": "learning_rate",
            "Values": ", ".join(str(x) for x in _uniq_sorted(df["learning_rate"])),
            "Notes": "Client optimiser learning rate.",
        }
    )

    # Privacy
    rows.append(
        {
            "Factor_Group": "Privacy (DP)",
            "Factor_Name": "uses_dp",
            "Values": ", ".join(str(x) for x in _uniq_sorted(df["uses_dp"])),
            "Notes": "Indicates whether DP noise is applied.",
        }
    )
    rows.append(
        {
            "Factor_Group": "Privacy (DP)",
            "Factor_Name": "dp_epsilon",
            "Values": ", ".join(str(x) for x in _uniq_sorted(df["dp_label"])),
            "Notes": "Non-DP runs are labelled 'inf'. Finite ε ∈ {0.1, 1, 10}.",
        }
    )
    rows.append(
        {
            "Factor_Group": "Privacy (DP)",
            "Factor_Name": "dp_delta",
            "Values": ", ".join(str(x) for x in _uniq_sorted(df["dp_delta"])),
            "Notes": "Target δ for (ε, δ)-DP.",
        }
    )
    rows.append(
        {
            "Factor_Group": "Privacy (DP)",
            "Factor_Name": "clip_norm",
            "Values": ", ".join(str(x) for x in _uniq_sorted(df["clip_norm"])),
            "Notes": "Gradient clipping norm used by the DP mechanism.",
        }
    )

    # Conformal / drift
    rows.append(
        {
            "Factor_Group": "Conformal / uncertainty",
            "Factor_Name": "Alpha",
            "Values": ", ".join(str(x) for x in _uniq_sorted(df["Alpha"])),
            "Notes": "Significance level; nominal coverage = 1 − Alpha.",
        }
    )
    rows.append(
        {
            "Factor_Group": "Conformal / uncertainty",
            "Factor_Name": "Nominal",
            "Values": ", ".join(str(x) for x in _uniq_sorted(df["Nominal"])),
            "Notes": "Nominal conformal coverage (e.g. 0.90).",
        }
    )

    rows.append(
        {
            "Factor_Group": "Spectral drift",
            "Factor_Name": "drift_level",
            "Values": ", ".join(str(x) for x in _uniq_sorted(df["drift_level"])),
            "Notes": "E.g., 'moderate'. None indicates runs without artificial drift.",
        }
    )
    rows.append(
        {
            "Factor_Group": "Spectral drift",
            "Factor_Name": "drift_type",
            "Values": ", ".join(str(x) for x in _uniq_sorted(df["drift_type"])),
            "Notes": "E.g., 'jitter'. None indicates baseline (unperturbed) runs.",
        }
    )

    table2 = pd.DataFrame(rows)
    return table2


# ----------------------------------------------------------------------
# Table 3 – Baseline (non-DP) performance by method, instrument AND transfer_k
# ----------------------------------------------------------------------

def make_table3_baseline_non_dp_all_k(df: pd.DataFrame) -> pd.DataFrame:
    """
    Table 3: Baseline non-DP performance, for ALL transfer_k values.

    This avoids arbitrarily fixing transfer_k = 80 and lets the manuscript
    show the full pattern first, then optionally focus the narrative on
    particular k levels.

    Columns:
        Method
        Category
        Subcategory
        Instrument_Code
        Transfer_k
        RMSEP_Mean
        RMSEP_Std
        R2_Mean
        R2_Std
        Global_Coverage_Mean
        Global_MeanWidth_Mean
        N_Experiments
    """
    df = df.copy()
    df = df[df["dp_epsilon"].isna()]  # non-DP only

    if df.empty:
        raise ValueError("No non-DP rows found in the database.")

    grouped = (
        df.groupby(["method", "category", "subcategory",
                    "instrument_code", "transfer_k"])
        .agg(
            RMSEP_Mean=("rmsep", "mean"),
            RMSEP_Std=("rmsep", "std"),
            R2_Mean=("r2", "mean"),
            R2_Std=("r2", "std"),
            Global_Coverage_Mean=("Global_Coverage", "mean"),
            Global_MeanWidth_Mean=("Global_MeanWidth", "mean"),
            N_Experiments=("combo_id", "count"),
        )
        .reset_index()
    )

    grouped = grouped.rename(
        columns={
            "method": "Method",
            "category": "Category",
            "subcategory": "Subcategory",
            "instrument_code": "Instrument_Code",
            "transfer_k": "Transfer_k",
        }
    )
    grouped = grouped.sort_values(
        ["Method", "Instrument_Code", "Transfer_k"]
    ).reset_index(drop=True)
    return grouped

def make_table3_per_k_views(table3_all: pd.DataFrame) -> Dict[Any, pd.DataFrame]:
    """
    Optional: create a dict of {transfer_k: DataFrame} from the all_k Table 3.

    Useful if you want separate CSV files per transfer_k (e.g. for supplement).
    """
    per_k: Dict[Any, pd.DataFrame] = {}
    for k, group in table3_all.groupby("Transfer_k"):
        # Convert numpy/pandas scalar to a plain Python int when possible;
        # if conversion fails, fall back to the original key (e.g. string).
        try:
            key = int(k)
        except (TypeError, ValueError):
            key = k
        per_k[key] = group.reset_index(drop=True)
    return per_k
    return per_k


# ----------------------------------------------------------------------
# Table 4 – Privacy–utility & communication (FedPLS / FedProx)
# ----------------------------------------------------------------------

def make_table4_privacy_tradeoff(df: pd.DataFrame) -> pd.DataFrame:
    """
    Table 4: Privacy–utility and communication trade-offs for federated methods.

    Restricts to methods: FedPLS, FedProx.

    Columns:
        Method
        Transfer_k
        DP_Label
        DP_Numeric
        RMSEP_Mean
        RMSEP_Std
        R2_Mean
        R2_Std
        Global_Coverage_Mean
        Global_MeanWidth_Mean
        Total_Bytes_Mean
        Total_Bytes_Std
        Wall_Time_Sec_Mean
        Wall_Time_Sec_Std
        N_Experiments
    """
    df = add_dp_helpers(df)
    df = df[df["method"].isin(["FedPLS", "FedProx"])].copy()

    grouped = (
        df.groupby(["method", "transfer_k", "dp_label", "dp_numeric"])
        .agg(
            RMSEP_Mean=("rmsep", "mean"),
            RMSEP_Std=("rmsep", "std"),
            R2_Mean=("r2", "mean"),
            R2_Std=("r2", "std"),
            Global_Coverage_Mean=("Global_Coverage", "mean"),
            Global_MeanWidth_Mean=("Global_MeanWidth", "mean"),
            Total_Bytes_Mean=("total_bytes", "mean"),
            Total_Bytes_Std=("total_bytes", "std"),
            Wall_Time_Sec_Mean=("wall_time_sec", "mean"),
            Wall_Time_Sec_Std=("wall_time_sec", "std"),
            N_Experiments=("combo_id", "count"),
        )
        .reset_index()
    )

    grouped = grouped.rename(
        columns={
            "method": "Method",
            "transfer_k": "Transfer_k",
            "dp_label": "DP_Label",
            "dp_numeric": "DP_Numeric",
        }
    )
    grouped = grouped.sort_values(
        ["Method", "Transfer_k", "DP_Numeric"]
    ).reset_index(drop=True)
    return grouped


# ----------------------------------------------------------------------
# Table 5 – Conformal coverage and width summary
# ----------------------------------------------------------------------

def make_table5_conformal_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Table 5: Conformal coverage and interval width summary across methods,
             transfer_k, and DP levels.

    Columns:
        Method
        Transfer_k
        DP_Label
        DP_Numeric
        Global_Coverage_Mean
        Global_Coverage_Std
        Global_MeanWidth_Mean
        Global_MeanWidth_Std
        Mondrian_Coverage_Mean
        Mondrian_MeanWidth_Mean
        Nominal_Mean
        Alpha_Mean
        N_Experiments
    """
    df = add_dp_helpers(df)

    grouped = (
        df.groupby(["method", "transfer_k", "dp_label", "dp_numeric"])
        .agg(
            Global_Coverage_Mean=("Global_Coverage", "mean"),
            Global_Coverage_Std=("Global_Coverage", "std"),
            Global_MeanWidth_Mean=("Global_MeanWidth", "mean"),
            Global_MeanWidth_Std=("Global_MeanWidth", "std"),
            Mondrian_Coverage_Mean=("Mondrian_Coverage", "mean"),
            Mondrian_MeanWidth_Mean=("Mondrian_MeanWidth", "mean"),
            Nominal_Mean=("Nominal", "mean"),
            Alpha_Mean=("Alpha", "mean"),
            N_Experiments=("combo_id", "count"),
        )
        .reset_index()
    )

    grouped = grouped.rename(
        columns={
            "method": "Method",
            "transfer_k": "Transfer_k",
            "dp_label": "DP_Label",
            "dp_numeric": "DP_Numeric",
        }
    )
    grouped = grouped.sort_values(
        ["Method", "Transfer_k", "DP_Numeric"]
    ).reset_index(drop=True)
    return grouped


# ----------------------------------------------------------------------
# Main CLI
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate manuscript tables as CSV from master_database JSON."
    )
    parser.add_argument(
        "--db",
        type=str,
        default="final2/publication_database/json/master_database_20251120_200830.json",
        help="Path to master_database_*.json",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="tables",
        help="Output directory for CSV tables",
    )
    parser.add_argument(
        "--split_table3_per_k",
        action="store_true",
        help="If set, also export one baseline non-DP table per transfer_k "
             "(table3a_baseline_non_dp_kXX.csv).",
    )
    args = parser.parse_args()

    db_path = Path(args.db)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_master_db(db_path)

    # Table 1
    table1 = make_table1_instruments(df)
    table1.to_csv(outdir / "table1_instruments.csv", index=False)

    # Table 2
    table2 = make_table2_experimental_design(df)
    table2.to_csv(outdir / "table2_experimental_design.csv", index=False)

    # Table 3 – ALL transfer_k
    table3_all = make_table3_baseline_non_dp_all_k(df)
    table3_all.to_csv(outdir / "table3_baseline_non_dp_all_k.csv", index=False)

    # Optional per-k views (for supplement if you want)
    if args.split_table3_per_k:
        per_k = make_table3_per_k_views(table3_all)
        for k, df_k in per_k.items():
            fname = outdir / f"table3_baseline_non_dp_k{k}.csv"
            df_k.to_csv(fname, index=False)

    # Table 4
    table4 = make_table4_privacy_tradeoff(df)
    table4.to_csv(outdir / "table4_privacy_tradeoff_fed.csv", index=False)

    # Table 5
    table5 = make_table5_conformal_summary(df)
    table5.to_csv(outdir / "table5_conformal_summary.csv", index=False)


if __name__ == "__main__":
    main()
