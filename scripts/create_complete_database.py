"""
Create a comprehensive database consolidating all experimental results for publication.

This script generates:
1. Master database with all experimental results
2. Normalized tables for relational structure
3. Summary statistics and metadata
4. Export formats: CSV, Excel, JSON, SQLite
"""

import pandas as pd
import json
# Fallback to add local src/ to sys.path when running script from source tree
try:
    import fedchem  # noqa: F401
except Exception:
    import sys
    _proj_root = Path(__file__).resolve().parents[1]
    _src_dir = _proj_root / "src"
    if _src_dir.exists():
        sys.path.insert(0, str(_src_dir))
try:
    from fedchem.results.aggregator import build_methods_summary
except Exception:
    # If the import still fails, keep going without build_methods_summary; exports that rely on it will handle missing function
    build_methods_summary = None
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from datetime import datetime


class PublicationDatabaseBuilder:
    """Build comprehensive publication database from experimental results."""
    
    def __init__(self, results_dir: str, output_dir: str):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data containers
        self.master_data = None
        self.experiments = None
        self.instruments = None
        self.methods = None
        self.privacy_configs = None
        self.performance_metrics = None
        self.conformal_metrics = None
        self.ct_baselines = None
        self.training_dynamics = None
        # Standardized instrument site names used in publication
        self._publication_sites = ['MA_A2', 'MB_B2']

    def _standardize_instrument_code(self, site_name: str) -> str:
        """Standardize site names like 'CalSetA2' to 'MA_A2' or leave existing MA_A2/MB_B2 unchanged."""
        if not isinstance(site_name, str):
            return site_name
        if site_name.startswith('CalSet') and len(site_name) >= 8:
            manufacturer = site_name[6]
            number = site_name[7]
            return f"M{manufacturer}_{manufacturer}{number}"
        # If already in Mx_Xn format like 'MA_A2' or 'MB_B2', return as is
        if len(site_name) >= 4 and site_name[0] == 'M' and '_' in site_name:
            return site_name
        return site_name
        
    def load_all_data(self):
        """Load all available data sources."""
        print("\n" + "="*80)
        print("LOADING DATA SOURCES")
        print("="*80 + "\n")
        
        # 1. Load raw records (federated experiments)
        raw_records_path = self.results_dir / "aggregated" / "raw_records.csv"
        if raw_records_path.exists():
            self.master_data = pd.read_csv(raw_records_path)
            print(f"[OK] Loaded raw records: {len(self.master_data)} rows")
            # Standardize master data and remove metric-like method rows
            self._standardize_master_data()
            # Save a copy of raw records snapshot in the output directory to match artifacts builder behavior
            out_csv = self.output_dir / "csv" / f"raw_records_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            self.master_data.to_csv(out_csv, index=False)
            print(f"[OK] Exported raw records snapshot: {out_csv}")
        else:
            print(f"[ERROR] Raw records not found: {raw_records_path}")
            return False
        
        # 2. Load classical CT baselines from objective_5
        self._load_ct_baselines()
        
        # 3. Load conformal prediction metrics from objective_2
        self._load_conformal_metrics()
        
        # 4. Load training dynamics from manifests
        self._load_training_dynamics()
        
        return True
    
    def _load_ct_baselines(self):
        """Load PDS, SBC, Site-specific, Pooled results."""
        ct_data = []
        
        for k in [20, 80, 200]:
            table_path = (self.results_dir / "generated_figures_tables_archive" / 
                         f"transfer_k_{k}" / "objective_5" / "table_5.csv")
            
            if table_path.exists():
                df = pd.read_csv(table_path)
                df['transfer_k'] = k
                # Standardize site names for CT baseline table
                if 'Site' in df.columns:
                    df['instrument_code'] = df['Site'].astype(str).apply(self._standardize_instrument_code)
                    # Replace the Site column with standardized instrument code
                    df['Site'] = df['instrument_code']
                ct_data.append(df)
                print(f"[OK] Loaded CT baselines for k={k}: {len(df)} rows")
        
        if ct_data:
            self.ct_baselines = pd.concat(ct_data, ignore_index=True)
        else:
            print("[SKIP] No CT baseline data found")
    
    def _load_conformal_metrics(self):
        """Load conformal prediction coverage and interval widths."""
        conformal_data = []
        
        for k in [20, 80, 200]:
            table_path = (self.results_dir / "generated_figures_tables_archive" / 
                         f"transfer_k_{k}" / "objective_2" / "table_2.csv")
            
            if table_path.exists():
                df = pd.read_csv(table_path)
                df['transfer_k'] = k
                # Standardize 'Site' to instrument_code (MA_A2 style)
                if 'Site' in df.columns:
                    df['instrument_code'] = df['Site'].astype(str).apply(self._standardize_instrument_code)
                    # Replace the Site column with standardized instrument code
                    df['Site'] = df['instrument_code']
                conformal_data.append(df)
                print(f"[OK] Loaded conformal metrics for k={k}: {len(df)} rows")
        
        if conformal_data:
            self.conformal_metrics = pd.concat(conformal_data, ignore_index=True)
        else:
            print("[SKIP] No conformal metrics found")
    
    def _load_training_dynamics(self):
        """Load round-by-round training dynamics from manifest files."""
        dynamics_data = []
        
        # Search for manifest files in objective_1
        for k in [20, 80, 200]:
            for eps in ['0p1', '1p0', '10p0', 'inf']:
                manifest_path = (self.results_dir / "generated_figures_tables_archive" / 
                               f"transfer_k_{k}" / f"eps_{eps}" / "delta_1eneg05" / 
                               "objective_1" / "manifest_1.json")
                
                if manifest_path.exists():
                    try:
                        manifest = json.load(open(manifest_path))
                        logs_by_algo = manifest.get('logs_by_algorithm', {})
                        
                        # Map eps string to numeric value
                        eps_map = {'0p1': 0.1, '1p0': 1.0, '10p0': 10.0, 'inf': np.inf}
                        eps_value = eps_map.get(eps, np.inf)
                        
                        for method, rounds in logs_by_algo.items():
                            for round_data in rounds:
                                dynamics_data.append({
                                    'transfer_k': k,
                                    'dp_epsilon': eps_value,
                                    'method': method,
                                    'round': round_data.get('round'),
                                    'rmsep': round_data.get('rmsep'),
                                    'r2': round_data.get('r2'),
                                    'mae': round_data.get('mae'),
                                    'bytes_sent': round_data.get('bytes_sent'),
                                    'bytes_recv': round_data.get('bytes_recv'),
                                    'duration_sec': round_data.get('duration_sec'),
                                    'participation_rate': round_data.get('participation_rate'),
                                    'epsilon_so_far': round_data.get('epsilon_so_far'),
                                    'clip_norm_used': round_data.get('clip_norm_used'),
                                    'compression_ratio': round_data.get('compression_ratio')
                                })
                    except Exception as e:
                        print(f"[SKIP] Failed to load {manifest_path}: {e}")
        
        if dynamics_data:
            self.training_dynamics = pd.DataFrame(dynamics_data)
            print(f"[OK] Loaded training dynamics: {len(self.training_dynamics)} round records")
        else:
            print("[SKIP] No training dynamics found")

    def _is_metric_like(self, value: str) -> bool:
        """Return True if the provided string looks like a metric name instead of a method name."""
        if not isinstance(value, str):
            return False
        v = value.strip().lower()
        metric_keywords = ['rmse', 'rmsep', 'mean', 'mae', 'std', 'median', 'avg', 'meanwidth']
        # Exclude names that are plausible method names (contain letters and not keys)
        return any(k in v for k in metric_keywords)

    def _standardize_master_data(self):
        """Apply consistent instrument and method normalization to master_data and related tables."""
        # If none of our key tables are present, nothing to standardize
        tables_present = any(
            tbl is not None and not (hasattr(tbl, 'empty') and tbl.empty)
            for tbl in [self.master_data, self.conformal_metrics, self.ct_baselines, self.training_dynamics]
        )
        if not tables_present:
            return

        # Normalize instrument codes from 'Site' or existing instrument_code
        if self.master_data is not None and 'Site' in self.master_data.columns:
            self.master_data['instrument_code'] = (
                self.master_data['Site'].astype(str).apply(self._standardize_instrument_code)
            )

        if self.master_data is not None and 'instrument_code' in self.master_data.columns:
            self.master_data['instrument_code'] = (
                self.master_data['instrument_code'].astype(str).apply(self._standardize_instrument_code)
            )

        # Normalize site_code/site_id columns if present
        if self.master_data is not None and 'site_code' in self.master_data.columns:
            self.master_data['site_code'] = (
                self.master_data['site_code'].astype(str).apply(self._standardize_instrument_code)
            )
        if self.master_data is not None and 'site_id' in self.master_data.columns:
            self.master_data['site_id'] = (
                self.master_data['site_id'].astype(str).apply(self._standardize_instrument_code)
            )

        # Remove rows that appear to list metrics as 'method'
        if self.master_data is not None and 'method' in self.master_data.columns:
            self.master_data = self.master_data[~self.master_data['method'].astype(str).apply(self._is_metric_like)].copy()

        # Apply the same mapping to conformal_metrics if loaded (keep only publication sites)
        if self.conformal_metrics is not None and not self.conformal_metrics.empty:
            # Ensure instrument_code is created
            if 'Site' in self.conformal_metrics.columns:
                self.conformal_metrics['instrument_code'] = (
                    self.conformal_metrics['Site'].astype(str).apply(self._standardize_instrument_code)
                )
                self.conformal_metrics['Site'] = self.conformal_metrics['instrument_code']
            if 'instrument_code' in self.conformal_metrics.columns:
                self.conformal_metrics['instrument_code'] = (
                    self.conformal_metrics['instrument_code'].astype(str).apply(self._standardize_instrument_code)
                )
            # Filter conformal metrics to only the publication sites
            self.conformal_metrics = self.conformal_metrics[self.conformal_metrics['instrument_code'].isin(self._publication_sites)].copy()

        # Apply mapping to ct_baselines and filter to publication sites
        if self.ct_baselines is not None and not self.ct_baselines.empty:
            if 'Site' in self.ct_baselines.columns:
                self.ct_baselines['instrument_code'] = (
                    self.ct_baselines['Site'].astype(str).apply(self._standardize_instrument_code)
                )
                self.ct_baselines['Site'] = self.ct_baselines['instrument_code']
            if 'instrument_code' in self.ct_baselines.columns:
                self.ct_baselines['instrument_code'] = (
                    self.ct_baselines['instrument_code'].astype(str).apply(self._standardize_instrument_code)
                )
            self.ct_baselines = self.ct_baselines[self.ct_baselines['instrument_code'].isin(self._publication_sites)].copy()

        # Apply mapping to training_dynamics 'instrument' like columns if present
        if self.training_dynamics is not None and not self.training_dynamics.empty:
            for candidate in ['instrument_code', 'site_id', 'site_code']:
                if candidate in self.training_dynamics.columns:
                    self.training_dynamics[candidate] = (
                        self.training_dynamics[candidate].astype(str).apply(self._standardize_instrument_code)
                    )
            # Filter training dynamics to publication sites if instrument info is present
            if 'instrument_code' in self.training_dynamics.columns:
                self.training_dynamics = self.training_dynamics[self.training_dynamics['instrument_code'].isin(self._publication_sites)].copy()
    
    def create_normalized_tables(self):
        """Create normalized relational tables."""
        print("\n" + "="*80)
        print("CREATING NORMALIZED TABLES")
        print("="*80 + "\n")
        
        # 1. Experiments table
        self._create_experiments_table()
        
        # 2. Instruments table
        self._create_instruments_table()
        
        # 3. Methods table
        self._create_methods_table()
        
        # 4. Privacy configurations table
        self._create_privacy_configs_table()
        
        # 5. Performance metrics table
        self._create_performance_metrics_table()
    
    def _create_experiments_table(self):
        """Create experiments metadata table."""
        exp_cols = [
            'combo_id', 'run_label', 'timestamp', 'script',
            'design_factors.Transfer_Samples', 'design_factors.DP_Target_Eps',
            'design_factors.DP_Delta', 'design_factors.Clip_Norm',
            'design_factors.Rounds', 'design_factors.Local_Epochs',
            'design_factors.Local_Batch_Size', 'design_factors.Learning_Rate',
            'design_factors.Spectral_Drift', 'design_factors.Drift_Type',
            'seed'
        ]
        
        available_cols = [col for col in exp_cols if col in self.master_data.columns]
        self.experiments = self.master_data[available_cols].drop_duplicates(subset=['combo_id'])
        
        # Rename columns for clarity
        rename_map = {
            'design_factors.Transfer_Samples': 'transfer_k',
            'design_factors.DP_Target_Eps': 'dp_epsilon',
            'design_factors.DP_Delta': 'dp_delta',
            'design_factors.Clip_Norm': 'clip_norm',
            'design_factors.Rounds': 'num_rounds',
            'design_factors.Local_Epochs': 'local_epochs',
            'design_factors.Local_Batch_Size': 'batch_size',
            'design_factors.Learning_Rate': 'learning_rate',
            'design_factors.Spectral_Drift': 'drift_level',
            'design_factors.Drift_Type': 'drift_type'
        }
        self.experiments = self.experiments.rename(columns=rename_map)
        
        print(f"[OK] Created experiments table: {len(self.experiments)} unique experiments")
    
    def _create_instruments_table(self):
        """Create instruments/sites metadata table."""
        instruments_data = []
        
        for instrument in self.master_data['instrument_code'].unique():
            # Ensure instrument_code is standardized
            instrument = self._standardize_instrument_code(instrument)
            manufacturer = 'ManufacturerA' if '_A_' in instrument or instrument.startswith('MA_') or 'A' in instrument else 'ManufacturerB'
            site_id = instrument
            
            instruments_data.append({
                'instrument_code': instrument,
                'site_code': instrument,
                'manufacturer': manufacturer,
                'instrument_id': int(instrument[-1]) if instrument[-1].isdigit() else 0
            })
        
        self.instruments = pd.DataFrame(instruments_data)
        print(f"[OK] Created instruments table: {len(self.instruments)} instruments")
    
    def _create_methods_table(self):
        """Create methods metadata table."""
        methods_data = []
        
        # Federated methods from master data
        # Exclude entries that are clearly metrics (e.g., 'mean_rmse') or other non-method
        metric_blacklist_substrings = ['rmse', 'mean', 'mae', 'std', 'median']
        for method in self.master_data['method'].unique():
            # Skip any string that looks like a metric (lowercase + metric keywords)
            if not isinstance(method, str):
                continue
            lowered = method.lower()
            if any(substr in lowered for substr in metric_blacklist_substrings):
                # Skip metric-like entries
                continue

            # Default: Federated category unless it matches classical CT methods
            if method in ['Centralized_PLS', 'Site_Specific', 'PDS', 'SBC']:
                category = 'Classical'
            else:
                category = 'Federated'

            if method in ['FedAvg', 'FedProx', 'FedAvg_noDP']:
                subcategory = 'Standard'
            elif 'FedPLS' in method:
                subcategory = 'Chemometrics-specific'
            else:
                subcategory = 'Other'

            methods_data.append({
                'method_name': method,
                'category': category,
                'subcategory': subcategory,
                'uses_dp': 'noDP' not in method
            })
        
        # Classical CT methods from baselines
        if self.ct_baselines is not None:
            ct_methods = {
                'Centralized_PLS': ('Classical', 'Centralized', False),
                'Site_Specific': ('Classical', 'Local', False),
                'PDS': ('Classical', 'Calibration Transfer', False),
                'SBC': ('Classical', 'Calibration Transfer', False)
            }
            
            for method, (cat, subcat, dp) in ct_methods.items():
                if method not in [m['method_name'] for m in methods_data]:
                    methods_data.append({
                        'method_name': method,
                        'category': cat,
                        'subcategory': subcat,
                        'uses_dp': dp
                    })
        
        self.methods = pd.DataFrame(methods_data).drop_duplicates(subset=['method_name'])
        print(f"[OK] Created methods table: {len(self.methods)} methods")

        # Export a methods summary equivalent to build_results_artifacts
        try:
            methods_summary = build_methods_summary(self.master_data if self.master_data is not None else pd.DataFrame())
            methods_summary_out = self.output_dir / 'csv' / f"methods_summary_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            methods_summary_out.parent.mkdir(parents=True, exist_ok=True)
            methods_summary.to_csv(methods_summary_out, index=False)
            print(f"[OK] Exported methods_summary_all: {methods_summary_out}")
        except Exception as e:
            print(f"[WARN] Failed to export methods_summary_all: {e}")
    
    def _create_privacy_configs_table(self):
        """Create privacy configurations table."""
        privacy_data = []
        
        # Ensure master data exists and contains the DP epsilon column
        if self.master_data is None:
            print("[SKIP] Master data not loaded, skipping privacy configs creation")
            self.privacy_configs = pd.DataFrame()
            return

        if 'design_factors.DP_Target_Eps' not in self.master_data.columns:
            print("[SKIP] DP target epsilon column not found, skipping privacy configs creation")
            self.privacy_configs = pd.DataFrame()
            return

        # Iterate over non-null unique epsilon values
        eps_series = self.master_data['design_factors.DP_Target_Eps'].dropna()
        if eps_series.empty:
            print("[SKIP] No DP epsilon values found, skipping privacy configs creation")
            self.privacy_configs = pd.DataFrame()
            return

        for eps in eps_series.unique():
            subset = self.master_data[self.master_data['design_factors.DP_Target_Eps'] == eps]
            if subset.empty:
                continue

            # Safely extract delta and clip if present
            if 'design_factors.DP_Delta' in subset.columns and not subset['design_factors.DP_Delta'].dropna().empty:
                delta = subset['design_factors.DP_Delta'].dropna().iloc[0]
            else:
                delta = np.nan

            if 'design_factors.Clip_Norm' in subset.columns and not subset['design_factors.Clip_Norm'].dropna().empty:
                clip = subset['design_factors.Clip_Norm'].dropna().iloc[0]
            else:
                clip = np.nan

            privacy_data.append({
                'dp_epsilon': eps,
                'dp_delta': delta,
                'clip_norm': clip,
                'privacy_level': self._categorize_privacy(eps)
            })
        
        self.privacy_configs = pd.DataFrame(privacy_data).drop_duplicates() if privacy_data else pd.DataFrame()
        print(f"[OK] Created privacy configs table: {len(self.privacy_configs)} configurations")
    
    def _create_performance_metrics_table(self):
        """Create comprehensive performance metrics table."""
        # If master data was not loaded, skip creating performance metrics
        if self.master_data is None:
            print("[SKIP] Master data not loaded, skipping performance metrics creation")
            self.performance_metrics = pd.DataFrame()
            return

        # Start with federated methods
        perf_cols = [
            'combo_id', 'method', 'instrument_code',
            'design_factors.Transfer_Samples', 'design_factors.DP_Target_Eps',
            'metrics.RMSEP', 'metrics.R2', 'metrics.Coverage',
            'runtime.total_bytes_mb', 'metrics.Round_Time',
            'metrics.Bytes_Sent', 'metrics.Bytes_Received', 'metrics.Total_Bytes',
            'runtime.wall_time_total_sec', 'metrics.Rounds'
        ]
        
        available_cols = [col for col in perf_cols if col in self.master_data.columns]
        perf_df = self.master_data[available_cols].copy()
        
        # Rename columns
        rename_map = {
            'design_factors.Transfer_Samples': 'transfer_k',
            'design_factors.DP_Target_Eps': 'dp_epsilon',
            'metrics.RMSEP': 'rmsep',
            'metrics.R2': 'r2',
            'metrics.Coverage': 'coverage',
            'runtime.total_bytes_mb': 'bytes_mb',
            'metrics.Round_Time': 'round_time_sec',
            'metrics.Bytes_Sent': 'bytes_sent',
            'metrics.Bytes_Received': 'bytes_received',
            'metrics.Total_Bytes': 'total_bytes',
            'runtime.wall_time_total_sec': 'wall_time_sec',
            'metrics.Rounds': 'num_rounds'
        }
        perf_df = perf_df.rename(columns=rename_map)
        
        # Add CT baselines if available
        if self.ct_baselines is not None:
            ct_perf = []
            # Only include instruments used in experimental design
            target_sites = self._publication_sites
            
            for _, row in self.ct_baselines.iterrows():
                # Prefer the standardized instrument_code field; fallback to original 'Site'
                site_name = row.get('instrument_code') if 'instrument_code' in row and pd.notna(row.get('instrument_code')) else row.get('Site')
                
                # Standardize site name and skip sites not in experimental design
                site_standard = self._standardize_instrument_code(site_name)
                if site_standard not in target_sites:
                    continue
                
                # Map actual column names in table_5.csv to method names
                method_mapping = {
                    'Centralized': 'Centralized_PLS',
                    'Site-specific': 'Site_Specific',
                    'PDS': 'PDS',
                    'SBC': 'SBC'
                }
                
                # Map CalSetXN to MX_XN format (e.g., CalSetA2 -> MA_A2, CalSetB2 -> MB_B2)
                instrument_code = site_standard
                
                for col_name, method_name in method_mapping.items():
                    if col_name in row.index and pd.notna(row[col_name]):
                        ct_perf.append({
                            'combo_id': f"ct_{method_name}_{instrument_code}_{row['transfer_k']}",
                            'method': method_name,
                            'instrument_code': instrument_code,
                            'transfer_k': row['transfer_k'],
                            'dp_epsilon': np.inf,
                            'rmsep': row[col_name],
                            'r2': np.nan,  # R2 not available in table_5
                            'coverage': np.nan,
                            'bytes_mb': 0.0,
                            'round_time_sec': 0.0,
                            'bytes_sent': 0.0,
                            'bytes_received': 0.0,
                            'total_bytes': 0.0,
                            'wall_time_sec': np.nan,
                            'num_rounds': 0
                        })
            
            if ct_perf:
                ct_df = pd.DataFrame(ct_perf)
                perf_df = pd.concat([perf_df, ct_df], ignore_index=True)
                print(f"[OK] Added {len(ct_perf)} CT baseline records (MA_A2 and MB_B2 only)")
        
        self.performance_metrics = perf_df
        print(f"[OK] Created performance metrics table: {len(self.performance_metrics)} records total")
    
    def _categorize_privacy(self, epsilon: float) -> str:
        """Categorize privacy level based on epsilon."""
        if np.isinf(epsilon):
            return 'None'
        elif epsilon <= 1.0:
            return 'Strong'
        elif epsilon <= 5.0:
            return 'Moderate'
        else:
            return 'Weak'
    
    def create_master_database(self):
        """Create master database with all data merged."""
        print("\n" + "="*80)
        print("CREATING MASTER DATABASE")
        print("="*80 + "\n")
        
        # Start with performance metrics
        master = self.performance_metrics.copy()
        
        # Add method metadata
        master = master.merge(
            self.methods,
            left_on='method',
            right_on='method_name',
            how='left'
        )
        
        # Add instrument metadata
        master = master.merge(
            self.instruments,
            on='instrument_code',
            how='left'
        )
        
        # Add experiment metadata (for federated only)
        if 'combo_id' in master.columns:
            master = master.merge(
                self.experiments,
                on='combo_id',
                how='left',
                suffixes=('', '_exp')
            )
        
        # Add conformal metrics if available
        if self.conformal_metrics is not None:
            # Ensure conformal metrics have a standardized instrument_code column
            conformal_mapped = self.conformal_metrics.copy()
            if 'instrument_code' not in conformal_mapped.columns and 'Site' in conformal_mapped.columns:
                conformal_mapped['instrument_code'] = conformal_mapped['Site'].astype(str).apply(self._standardize_instrument_code)

            # Filter to experimental design sites only
            conformal_mapped = conformal_mapped[conformal_mapped['instrument_code'].isin(self._publication_sites)]
            
            conformal_agg = conformal_mapped.groupby(
                ['transfer_k', 'instrument_code']
            ).agg({
                'Global_Coverage': 'mean',
                'Global_MeanWidth': 'mean',
                'Mondrian_Coverage': 'mean',
                'Mondrian_MeanWidth': 'mean',
                'Alpha': 'first',
                'Nominal': 'first'
            }).reset_index()
            
            master = master.merge(
                conformal_agg,
                on=['transfer_k', 'instrument_code'],
                how='left'
            )
            print(f"[OK] Merged conformal metrics: {conformal_agg.shape[0]} configurations")
        
        self.master_database = master
        print(f"[OK] Created master database: {len(self.master_database)} records, {len(self.master_database.columns)} columns")
    
    def generate_summary_statistics(self):
        """Generate summary statistics for publication."""
        print("\n" + "="*80)
        print("GENERATING SUMMARY STATISTICS")
        print("="*80 + "\n")
        
        summaries = {}
        
        # 1. Performance by method
        if 'method' in self.master_database.columns:
            method_summary = self.master_database.groupby('method').agg({
                'rmsep': ['mean', 'std', 'min', 'max', 'count'],
                'r2': ['mean', 'std', 'min', 'max'],
                'bytes_mb': ['mean', 'std'],
                'round_time_sec': ['mean', 'std']
            }).round(4)
            summaries['performance_by_method'] = method_summary
            print(f"[OK] Performance by method: {len(method_summary)} methods")
        
        # 2. Performance by privacy level
        if 'dp_epsilon' in self.master_database.columns:
            privacy_summary = self.master_database.groupby('dp_epsilon').agg({
                'rmsep': ['mean', 'std', 'count'],
                'r2': ['mean', 'std'],
                'bytes_mb': ['mean', 'std']
            }).round(4)
            summaries['performance_by_privacy'] = privacy_summary
            print(f"[OK] Performance by privacy: {len(privacy_summary)} levels")
        
        # 3. Performance by transfer sample size
        if 'transfer_k' in self.master_database.columns:
            transfer_summary = self.master_database.groupby('transfer_k').agg({
                'rmsep': ['mean', 'std', 'count'],
                'r2': ['mean', 'std'],
                'bytes_mb': ['mean', 'std']
            }).round(4)
            summaries['performance_by_transfer_k'] = transfer_summary
            print(f"[OK] Performance by transfer_k: {len(transfer_summary)} sizes")
        
        # 4. Performance by instrument/manufacturer
        if 'manufacturer' in self.master_database.columns:
            instrument_summary = self.master_database.groupby(['manufacturer', 'instrument_code']).agg({
                'rmsep': ['mean', 'std', 'count'],
                'r2': ['mean', 'std']
            }).round(4)
            summaries['performance_by_instrument'] = instrument_summary
            print(f"[OK] Performance by instrument: {len(instrument_summary)} instruments")
        
        # 5. Best performing configurations
        if 'rmsep' in self.master_database.columns:
            best_configs = self.master_database.nsmallest(20, 'rmsep')[
                ['method', 'instrument_code', 'transfer_k', 'dp_epsilon', 'rmsep', 'r2']
            ]
            summaries['best_configurations'] = best_configs
            print(f"[OK] Best configurations: top 20")
        
        self.summaries = summaries
        return summaries
    
    def export_all_formats(self):
        """Export database in multiple formats."""
        print("\n" + "="*80)
        print("EXPORTING DATABASE")
        print("="*80 + "\n")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Export to CSV
        self._export_csv(timestamp)
        
        # 2. Export to Excel with multiple sheets
        self._export_excel(timestamp)
        
        # 3. Export to JSON
        self._export_json(timestamp)
        
        # 4. Export to SQLite
        self._export_sqlite(timestamp)
        
        # 5. Export summary report
        self._export_summary_report(timestamp)
    
    def _export_csv(self, timestamp: str):
        """Export all tables to CSV."""
        csv_dir = self.output_dir / "csv"
        csv_dir.mkdir(exist_ok=True)
        
        # Raw records (snapshot of standardized master data)
        if self.master_data is not None:
            raw_out = csv_dir / f"raw_records_{timestamp}.csv"
            self.master_data.to_csv(raw_out, index=False)
            print(f"[OK] Exported raw records: {raw_out}")

        # Master database
        master_path = csv_dir / f"master_database_{timestamp}.csv"
        self.master_database.to_csv(master_path, index=False)
        print(f"[OK] Exported master database: {master_path}")
        
        # Normalized tables
        tables = {
            'experiments': self.experiments,
            'instruments': self.instruments,
            'methods': self.methods,
            'privacy_configs': self.privacy_configs,
            'performance_metrics': self.performance_metrics
        }
        
        for name, df in tables.items():
            if df is not None:
                path = csv_dir / f"{name}_{timestamp}.csv"
                df.to_csv(path, index=False)
                print(f"[OK] Exported {name}: {path}")

        # Also write a methods_summary CSV similar to build_results_artifacts
        try:
            methods_summary = build_methods_summary(self.master_data if self.master_data is not None else pd.DataFrame())
            ms_out = csv_dir / f"methods_summary_all_{timestamp}.csv"
            methods_summary.to_csv(ms_out, index=False)
            print(f"[OK] Exported methods_summary_all: {ms_out}")
        except Exception as e:
            print(f"[WARN] Could not export methods_summary_all: {e}")
        
        # Conformal metrics - map site names like CalSetA2 -> MA_A2 and filter to experimental sites
        if self.conformal_metrics is not None:
            conformal_out = self.conformal_metrics.copy()

            # If 'Site' is present, map it to 'instrument_code' (MA_A2 style)
            if 'Site' in conformal_out.columns:
                def map_site_to_instrument(site_name: str) -> str:
                    if isinstance(site_name, str) and site_name.startswith('CalSet') and len(site_name) >= 8:
                        manufacturer = site_name[6]  # 'A' or 'B'
                        number = site_name[7]
                        return f"M{manufacturer}_{manufacturer}{number}"
                    return site_name

                conformal_out['instrument_code'] = conformal_out['Site'].apply(map_site_to_instrument)
                # Replace the 'Site' column with the mapped instrument code
                conformal_out['Site'] = conformal_out['instrument_code']

            # Keep only experimental sites used in publication (MA_A2 and MB_B2)
            keep_instruments = ['MA_A2', 'MB_B2']
            if 'instrument_code' in conformal_out.columns:
                conformal_out = conformal_out[conformal_out['instrument_code'].isin(keep_instruments)].copy()
            else:
                # If no instrument_code column then try mapping Site values directly
                conformal_out = conformal_out[conformal_out['Site'].isin(keep_instruments)].copy()

            path = csv_dir / f"conformal_metrics_{timestamp}.csv"
            conformal_out.to_csv(path, index=False)
            print(f"[OK] Exported conformal metrics (mapped & filtered): {path}")
        
        # Training dynamics
        if self.training_dynamics is not None:
            path = csv_dir / f"training_dynamics_{timestamp}.csv"
            self.training_dynamics.to_csv(path, index=False)
            print(f"[OK] Exported training dynamics: {path}")
    
    def _export_excel(self, timestamp: str):
        """Export to Excel with multiple sheets."""
        excel_path = self.output_dir / f"complete_database_{timestamp}.xlsx"
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Master database
            self.master_database.to_excel(writer, sheet_name='Master_Database', index=False)
            
            # Normalized tables
            if self.experiments is not None:
                self.experiments.to_excel(writer, sheet_name='Experiments', index=False)
            if self.instruments is not None:
                self.instruments.to_excel(writer, sheet_name='Instruments', index=False)
            if self.methods is not None:
                self.methods.to_excel(writer, sheet_name='Methods', index=False)
            if self.privacy_configs is not None:
                self.privacy_configs.to_excel(writer, sheet_name='Privacy_Configs', index=False)
            if self.performance_metrics is not None:
                self.performance_metrics.to_excel(writer, sheet_name='Performance_Metrics', index=False)
            if self.conformal_metrics is not None:
                self.conformal_metrics.to_excel(writer, sheet_name='Conformal_Metrics', index=False)
            if self.training_dynamics is not None:
                self.training_dynamics.to_excel(writer, sheet_name='Training_Dynamics', index=False)
            
            # Summary statistics
            for name, df in self.summaries.items():
                sheet_name = name[:31]  # Excel sheet name limit
                df.to_excel(writer, sheet_name=sheet_name)
        
        print(f"[OK] Exported Excel database: {excel_path}")
    
    def _export_json(self, timestamp: str):
        """Export to JSON format."""
        json_dir = self.output_dir / "json"
        json_dir.mkdir(exist_ok=True)
        
        # Master database
        master_path = json_dir / f"master_database_{timestamp}.json"
        self.master_database.to_json(master_path, orient='records', indent=2)
        print(f"[OK] Exported master JSON: {master_path}")
        
        # Metadata structure
        metadata = {
            'export_timestamp': timestamp,
            'total_records': len(self.master_database),
            'num_experiments': len(self.experiments) if self.experiments is not None else 0,
            'num_instruments': len(self.instruments) if self.instruments is not None else 0,
            'num_methods': len(self.methods) if self.methods is not None else 0,
            'columns': list(self.master_database.columns),
            'methods_list': self.methods['method_name'].tolist() if self.methods is not None else [],
            'instruments_list': self.instruments['instrument_code'].tolist() if self.instruments is not None else []
        }
        
        metadata_path = json_dir / f"metadata_{timestamp}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"[OK] Exported metadata JSON: {metadata_path}")
    
    def _export_sqlite(self, timestamp: str):
        """Export to SQLite database."""
        db_path = self.output_dir / f"complete_database_{timestamp}.db"
        
        conn = sqlite3.connect(db_path)
        
        # Create tables
        self.master_database.to_sql('master_database', conn, if_exists='replace', index=False)
        
        if self.experiments is not None:
            self.experiments.to_sql('experiments', conn, if_exists='replace', index=False)
        if self.instruments is not None:
            self.instruments.to_sql('instruments', conn, if_exists='replace', index=False)
        if self.methods is not None:
            self.methods.to_sql('methods', conn, if_exists='replace', index=False)
        if self.privacy_configs is not None:
            self.privacy_configs.to_sql('privacy_configs', conn, if_exists='replace', index=False)
        if self.performance_metrics is not None:
            self.performance_metrics.to_sql('performance_metrics', conn, if_exists='replace', index=False)
        if self.conformal_metrics is not None:
            self.conformal_metrics.to_sql('conformal_metrics', conn, if_exists='replace', index=False)
        if self.training_dynamics is not None:
            self.training_dynamics.to_sql('training_dynamics', conn, if_exists='replace', index=False)
        
        conn.close()
        print(f"[OK] Exported SQLite database: {db_path}")
    
    def _export_summary_report(self, timestamp: str):
        """Export comprehensive summary report."""
        report_path = self.output_dir / f"DATABASE_SUMMARY_{timestamp}.md"
        
        with open(report_path, 'w') as f:
            f.write("# Complete Publication Database Summary\n\n")
            f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Database Overview\n\n")
            f.write(f"- **Total Records**: {len(self.master_database):,}\n")
            f.write(f"- **Total Columns**: {len(self.master_database.columns)}\n")
            f.write(f"- **Experiments**: {len(self.experiments) if self.experiments is not None else 0}\n")
            f.write(f"- **Instruments**: {len(self.instruments) if self.instruments is not None else 0}\n")
            f.write(f"- **Methods**: {len(self.methods) if self.methods is not None else 0}\n\n")
            
            f.write("## Available Tables\n\n")
            f.write("1. **Master Database**: Complete merged dataset with all metrics\n")
            f.write("2. **Experiments**: Experimental configurations and hyperparameters\n")
            f.write("3. **Instruments**: Instrument/site metadata\n")
            f.write("4. **Methods**: Method descriptions and categorizations\n")
            f.write("5. **Privacy Configs**: Differential privacy configurations\n")
            f.write("6. **Performance Metrics**: All performance measurements\n")
            f.write("7. **Conformal Metrics**: Conformal prediction coverage and widths\n\n")
            
            f.write("## Summary Statistics\n\n")
            
            for name, df in self.summaries.items():
                f.write(f"### {name.replace('_', ' ').title()}\n\n")
                f.write(df.to_markdown() + "\n\n")
            
            f.write("## Column Descriptions\n\n")
            f.write("### Master Database Columns\n\n")
            for col in self.master_database.columns:
                f.write(f"- `{col}`: {self._describe_column(col)}\n")
        
        print(f"[OK] Exported summary report: {report_path}")
    
    def _describe_column(self, col: str) -> str:
        """Generate description for column."""
        descriptions = {
            'combo_id': 'Unique experiment identifier',
            'method': 'Federated or classical method name',
            'instrument_code': 'Instrument/site identifier',
            'transfer_k': 'Number of transfer calibration samples',
            'dp_epsilon': 'Differential privacy epsilon parameter',
            'rmsep': 'Root Mean Squared Error of Prediction',
            'r2': 'R-squared coefficient of determination',
            'bytes_mb': 'Communication cost in megabytes',
            'round_time_sec': 'Average time per training round in seconds',
            'manufacturer': 'Instrument manufacturer (A or B)',
            'category': 'Method category (Federated or Classical)',
            'privacy_level': 'Categorized privacy level (None/Weak/Moderate/Strong)',
            'Global_Coverage': 'Global conformal prediction coverage',
            'Mondrian_Coverage': 'Mondrian (per-site) conformal coverage'
        }
        return descriptions.get(col, 'Experimental data field')


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Create complete publication database')
    parser.add_argument('--results-dir', type=str, default='final2',
                       help='Results directory containing aggregated data')
    parser.add_argument('--output-dir', type=str, default='final2/publication_database',
                       help='Output directory for database files')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("COMPLETE PUBLICATION DATABASE BUILDER")
    print("="*80)
    
    # Create builder
    builder = PublicationDatabaseBuilder(args.results_dir, args.output_dir)
    
    # Load all data
    if not builder.load_all_data():
        print("\n[ERROR] Failed to load data. Exiting.")
        return
    
    # Create normalized tables
    builder.create_normalized_tables()
    
    # Create master database
    builder.create_master_database()
    
    # Generate summary statistics
    builder.generate_summary_statistics()
    
    # Export all formats
    builder.export_all_formats()
    
    print("\n" + "="*80)
    print(f"DATABASE CREATION COMPLETE - Output: {args.output_dir}")
    print("="*80 + "\n")
    
    print("Generated files:")
    print("  - CSV files: csv/")
    print("  - Excel workbook: complete_database_*.xlsx")
    print("  - JSON files: json/")
    print("  - SQLite database: complete_database_*.db")
    print("  - Summary report: DATABASE_SUMMARY_*.md")
    print()


if __name__ == '__main__':
    main()
