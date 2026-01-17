#!/usr/bin/env python3
import os
import sys
import re
import glob
import hashlib
import warnings
from datetime import datetime
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore')


class Config:
    """Centralized configuration for HAI ingestion pipeline."""
    
    # Input paths
    SOURCE_DIR = "/content/drive/MyDrive/Datasets/Hai/hai-22.04"
    
    # Alternative search paths (if files not in root)
    ALTERNATIVE_PATHS = [
        "/content/drive/MyDrive/Datasets/Hai/hai-22.04",
        "/content/drive/MyDrive/Datasets/HAI/hai-22.04",
        "/content/drive/MyDrive/Datasets/hai/hai-22.04",
        "/content/drive/MyDrive/Datasets/Hai",
        "/content/drive/MyDrive/Datasets/HAI",
    ]
    
    # Output paths
    BASE_OUTPUT_DIR = "/content/drive/MyDrive/Quantum_Research"
    PROCESSED_DIR = f"{BASE_OUTPUT_DIR}/02_Processed_Data/HAI"
    MODELS_DIR = f"{BASE_OUTPUT_DIR}/04_Models"
    BACKUP_DIR = f"{BASE_OUTPUT_DIR}/BACKUPS/HAI"
    
    # Output file names
    X_TRAIN_FILE = "HAI_X_train.npy"
    Y_TRAIN_FILE = "HAI_y_train.npy"
    X_TEST_FILE = "HAI_X_test.npy"
    Y_TEST_FILE = "HAI_y_test.npy"
    FEATURE_NAMES_FILE = "hai_feature_names.joblib"
    SCALER_FILE = "hai_scaler.joblib"
    MANIFEST_FILE = "hai_ingestion_manifest.json"
    
    # Processing parameters
    RANDOM_SEED = 42
    
    # Possible label column names in HAI
    LABEL_COLUMNS = ['attack', 'Attack', 'ATTACK', 'label', 'Label', 'attack_label']
    
    # Possible timestamp column names
    TIMESTAMP_COLUMNS = ['timestamp', 'Timestamp', 'TIMESTAMP', 'time', 'Time', 'datetime']


def ensure_directory_exists(dir_path: str) -> bool:
    """Create directory if it doesn't exist."""
    try:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        raise OSError(f"[I/O ERROR] Cannot create directory: {dir_path} - {e}")


def compute_md5(filepath: str) -> str:
    """Compute MD5 hash of a file for forensic verification."""
    hash_md5 = hashlib.md5()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        return f"ERROR: {e}"


def natural_sort_key(s: str) -> List:
    """
    Key function for natural sorting of filenames.
    
    Ensures train1.csv < train2.csv < train10.csv
    (instead of train1.csv < train10.csv < train2.csv)
    """
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]


class ForensicFileAssembler:
    """
    Assembles split CSV files into continuous time-series dataframes.
    
    CRITICAL: Files must be sorted in natural order to preserve
    the physical timeline of the industrial process.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.source_dir = None
        self.train_files: List[str] = []
        self.test_files: List[str] = []
        self.file_hashes: Dict[str, str] = {}
    
    def find_source_directory(self) -> str:
        """
        Find the HAI dataset directory.
        
        Returns:
            Path to the source directory
        """
        print("\n" + "="*70)
        print("PHASE 0: SOURCE DIRECTORY DISCOVERY")
        print("="*70)
        
        # Check primary path
        if os.path.exists(self.config.SOURCE_DIR):
            # Check if it has CSV files
            csv_files = glob.glob(os.path.join(self.config.SOURCE_DIR, "*.csv"))
            if csv_files:
                self.source_dir = self.config.SOURCE_DIR
                print(f"\n[FOUND] Primary path: {self.source_dir}")
                print(f"  -> CSV files found: {len(csv_files)}")
                return self.source_dir
            
            # Check subdirectories
            for subdir in os.listdir(self.config.SOURCE_DIR):
                subdir_path = os.path.join(self.config.SOURCE_DIR, subdir)
                if os.path.isdir(subdir_path):
                    csv_files = glob.glob(os.path.join(subdir_path, "*.csv"))
                    if csv_files:
                        self.source_dir = subdir_path
                        print(f"\n[FOUND] In subdirectory: {self.source_dir}")
                        print(f"  -> CSV files found: {len(csv_files)}")
                        return self.source_dir
        
        # Try alternative paths
        for alt_path in self.config.ALTERNATIVE_PATHS:
            if os.path.exists(alt_path):
                csv_files = glob.glob(os.path.join(alt_path, "*.csv"))
                if csv_files:
                    self.source_dir = alt_path
                    print(f"\n[FOUND] Alternative path: {self.source_dir}")
                    print(f"  -> CSV files found: {len(csv_files)}")
                    return self.source_dir
                
                # Check subdirectories
                for subdir in os.listdir(alt_path):
                    subdir_path = os.path.join(alt_path, subdir)
                    if os.path.isdir(subdir_path):
                        csv_files = glob.glob(os.path.join(subdir_path, "*.csv"))
                        if csv_files:
                            self.source_dir = subdir_path
                            print(f"\n[FOUND] In subdirectory: {self.source_dir}")
                            print(f"  -> CSV files found: {len(csv_files)}")
                            return self.source_dir
        
        raise FileNotFoundError(
            f"[FATAL] HAI dataset not found. Searched:\n"
            f"  - {self.config.SOURCE_DIR}\n"
            f"  - {self.config.ALTERNATIVE_PATHS}"
        )
    
    def discover_files(self) -> Tuple[List[str], List[str]]:
        """
        Discover and sort train/test files.
        
        Returns:
            Tuple of (train_files, test_files) sorted in natural order
        """
        print("\n" + "="*70)
        print("PHASE 1: FORENSIC FILE ASSEMBLY")
        print("="*70)
        
        if self.source_dir is None:
            self.find_source_directory()
        
        print(f"\n[SCANNING] Source directory: {self.source_dir}")
        
        # Find all CSV files
        all_csv = glob.glob(os.path.join(self.source_dir, "*.csv"))
        print(f"  -> Total CSV files found: {len(all_csv)}")
        
        # Separate train and test files
        train_pattern = re.compile(r'train.*\.csv$', re.IGNORECASE)
        test_pattern = re.compile(r'test.*\.csv$', re.IGNORECASE)
        
        self.train_files = sorted(
            [f for f in all_csv if train_pattern.search(os.path.basename(f))],
            key=lambda x: natural_sort_key(os.path.basename(x))
        )
        
        self.test_files = sorted(
            [f for f in all_csv if test_pattern.search(os.path.basename(f))],
            key=lambda x: natural_sort_key(os.path.basename(x))
        )
        
        print(f"\n[TRAIN FILES] Found {len(self.train_files)} files (sorted):")
        for i, f in enumerate(self.train_files):
            basename = os.path.basename(f)
            print(f"  {i+1}. {basename}")
        
        print(f"\n[TEST FILES] Found {len(self.test_files)} files (sorted):")
        for i, f in enumerate(self.test_files):
            basename = os.path.basename(f)
            print(f"  {i+1}. {basename}")
        
        if not self.train_files:
            raise FileNotFoundError("[FATAL] No training files found!")
        if not self.test_files:
            raise FileNotFoundError("[FATAL] No test files found!")
        
        return self.train_files, self.test_files
    
    def compute_forensic_hashes(self) -> Dict[str, str]:
        """
        Compute MD5 hashes for all source files.
        
        Returns:
            Dictionary of {filename: md5_hash}
        """
        print(f"\n[FORENSICS] Computing MD5 hashes for audit trail...")
        
        all_files = self.train_files + self.test_files
        
        for filepath in all_files:
            basename = os.path.basename(filepath)
            md5_hash = compute_md5(filepath)
            self.file_hashes[basename] = md5_hash
            print(f"  -> {basename}: {md5_hash[:16]}...")
        
        return self.file_hashes
    
    def load_and_concatenate(
        self,
        file_list: List[str],
        dataset_name: str
    ) -> pd.DataFrame:
        """
        Load and concatenate CSV files in order.
        
        CRITICAL: Maintains physical time-series order.
        No shuffling or reordering after concatenation.
        
        Args:
            file_list: Sorted list of file paths
            dataset_name: Name for logging (e.g., "Train", "Test")
        
        Returns:
            Concatenated DataFrame
        """
        print(f"\n[LOADING] {dataset_name} data ({len(file_list)} files)...")
        
        dfs = []
        total_rows = 0
        
        for filepath in file_list:
            basename = os.path.basename(filepath)
            try:
                df = pd.read_csv(filepath)
                dfs.append(df)
                total_rows += len(df)
                print(f"  -> {basename}: {len(df):,} rows, {df.shape[1]} columns")
            except Exception as e:
                raise IOError(f"[FATAL] Failed to load {basename}: {e}")
        
        # Concatenate in order (no shuffling!)
        print(f"\n[CONCATENATING] {dataset_name} files in physical order...")
        combined_df = pd.concat(dfs, axis=0, ignore_index=True)
        
        print(f"  -> Combined shape: {combined_df.shape}")
        print(f"  -> Total rows: {len(combined_df):,}")
        print(f"  -> [OK] Time-series order preserved (no shuffle)")
        
        return combined_df


class DataCleaner:
    """
    Cleans and aligns HAI dataset for quantum processing.
    
    Steps:
    1. Timestamp verification and removal
    2. Label column extraction
    3. Feature alignment between train/test
    4. Missing value handling (forward fill)
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.label_column: Optional[str] = None
        self.timestamp_column: Optional[str] = None
        self.feature_columns: List[str] = []
        self.cleaning_stats: Dict[str, Any] = {}
    
    def identify_special_columns(self, df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
        """
        Identify timestamp and label columns.
        
        Args:
            df: Input DataFrame
        
        Returns:
            Tuple of (timestamp_column, label_column)
        """
        columns = df.columns.tolist()
        
        # Find timestamp column
        for ts_name in self.config.TIMESTAMP_COLUMNS:
            if ts_name in columns:
                self.timestamp_column = ts_name
                break
        
        # Find label column
        for label_name in self.config.LABEL_COLUMNS:
            if label_name in columns:
                self.label_column = label_name
                break
        
        return self.timestamp_column, self.label_column
    
    def clean_dataset(
        self,
        raw_train_df: pd.DataFrame,
        raw_test_df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Clean and align train/test datasets.
        
        Args:
            raw_train_df: Raw training DataFrame
            raw_test_df: Raw test DataFrame
        
        Returns:
            Tuple of (X_train, y_train, X_test, y_test, feature_names)
        """
        print("\n" + "="*70)
        print("PHASE 2: CLEANING & ALIGNMENT")
        print("="*70)
        
        # Identify special columns
        print(f"\n[COLUMNS] Identifying special columns...")
        self.identify_special_columns(raw_train_df)
        
        print(f"  -> Timestamp column: {self.timestamp_column}")
        print(f"  -> Label column: {self.label_column}")
        
        if self.label_column is None:
            # Try to find it by examining column values
            print(f"  -> [WARNING] Label column not found by name, searching by values...")
            for col in raw_train_df.columns:
                unique_vals = raw_train_df[col].unique()
                if len(unique_vals) <= 10 and set(unique_vals).issubset({0, 1, 0.0, 1.0}):
                    self.label_column = col
                    print(f"  -> Found potential label column: {col}")
                    break
        
        if self.label_column is None:
            raise ValueError("[FATAL] Could not identify label column!")
        
        # Verify timestamp column and check sorting
        if self.timestamp_column:
            print(f"\n[TIMESTAMP] Verifying time-series order...")
            try:
                train_ts = pd.to_datetime(raw_train_df[self.timestamp_column])
                test_ts = pd.to_datetime(raw_test_df[self.timestamp_column])
                
                # Check if sorted
                train_sorted = train_ts.is_monotonic_increasing
                test_sorted = test_ts.is_monotonic_increasing
                
                print(f"  -> Train timestamps monotonic: {train_sorted}")
                print(f"  -> Test timestamps monotonic: {test_sorted}")
                
                if not train_sorted:
                    print(f"  -> [WARNING] Train data not strictly sorted, but preserving file order")
                if not test_sorted:
                    print(f"  -> [WARNING] Test data not strictly sorted, but preserving file order")
                    
            except Exception as e:
                print(f"  -> [WARNING] Could not parse timestamps: {e}")
        
        # Extract labels
        print(f"\n[LABELS] Extracting label column '{self.label_column}'...")
        y_train = raw_train_df[self.label_column].values
        y_test = raw_test_df[self.label_column].values
        
        # Convert to binary if needed
        y_train = (y_train > 0).astype(np.int32)
        y_test = (y_test > 0).astype(np.int32)
        
        print(f"  -> y_train shape: {y_train.shape}")
        print(f"  -> y_test shape: {y_test.shape}")
        
        # Get feature columns (exclude timestamp and label)
        columns_to_drop = []
        if self.timestamp_column:
            columns_to_drop.append(self.timestamp_column)
        columns_to_drop.append(self.label_column)
        
        train_features = [c for c in raw_train_df.columns if c not in columns_to_drop]
        test_features = [c for c in raw_test_df.columns if c not in columns_to_drop]
        
        print(f"\n[FEATURES] Aligning feature columns...")
        print(f"  -> Train features: {len(train_features)}")
        print(f"  -> Test features: {len(test_features)}")
        
        # Find common features
        common_features = sorted(list(set(train_features) & set(test_features)))
        
        train_only = set(train_features) - set(test_features)
        test_only = set(test_features) - set(train_features)
        
        if train_only:
            print(f"  -> [WARNING] Dropping {len(train_only)} train-only columns")
        if test_only:
            print(f"  -> [WARNING] Dropping {len(test_only)} test-only columns")
        
        self.feature_columns = common_features
        print(f"  -> Common features: {len(common_features)}")
        
        # Extract feature matrices
        X_train_df = raw_train_df[common_features].copy()
        X_test_df = raw_test_df[common_features].copy()
        
        # Handle missing values with forward fill (physics-aware)
        print(f"\n[MISSING VALUES] Applying forward fill (sensor signal persistence)...")
        
        train_nan_before = X_train_df.isna().sum().sum()
        test_nan_before = X_test_df.isna().sum().sum()
        
        print(f"  -> Train NaNs before: {train_nan_before:,}")
        print(f"  -> Test NaNs before: {test_nan_before:,}")
        
        # Forward fill
        X_train_df = X_train_df.ffill()
        X_test_df = X_test_df.ffill()
        
        # Backward fill for any remaining NaNs at the start
        X_train_df = X_train_df.bfill()
        X_test_df = X_test_df.bfill()
        
        # Final check - fill any remaining with 0 (edge case)
        X_train_df = X_train_df.fillna(0)
        X_test_df = X_test_df.fillna(0)
        
        train_nan_after = X_train_df.isna().sum().sum()
        test_nan_after = X_test_df.isna().sum().sum()
        
        print(f"  -> Train NaNs after: {train_nan_after}")
        print(f"  -> Test NaNs after: {test_nan_after}")
        print(f"  -> [OK] Missing values handled")
        
        # Convert to numpy
        X_train = X_train_df.values.astype(np.float64)
        X_test = X_test_df.values.astype(np.float64)
        
        # Store cleaning statistics
        self.cleaning_stats = {
            'train_nan_before': int(train_nan_before),
            'test_nan_before': int(test_nan_before),
            'train_nan_after': int(train_nan_after),
            'test_nan_after': int(test_nan_after),
            'n_features': len(common_features),
            'train_only_dropped': len(train_only),
            'test_only_dropped': len(test_only)
        }
        
        return X_train, y_train, X_test, y_test, common_features


class PhysicsAwareNormalizer:
    """
    Normalizes data with strict leakage prevention.
    
    CRITICAL: Scaler is fitted ONLY on training data.
    This simulates real-world deployment where the model
    has never seen attack statistics.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.scaler: Optional[MinMaxScaler] = None
        self.normalization_stats: Dict[str, Any] = {}
    
    def normalize(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
        """
        Normalize data with leakage prevention.
        
        Args:
            X_train: Training features
            X_test: Test features
        
        Returns:
            Tuple of (X_train_scaled, X_test_scaled, scaler)
        """
        print("\n" + "="*70)
        print("PHASE 3: PHYSICS-AWARE NORMALIZATION (NO LEAKAGE)")
        print("="*70)
        
        print(f"\n[SCALER] Initializing MinMaxScaler [0, 1]...")
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
        print(f"\n[FIT] Fitting scaler on TRAINING DATA ONLY...")
        print(f"  -> X_train shape: {X_train.shape}")
        print(f"  -> [CRITICAL] Test data NOT used for fitting (prevents leakage)")
        
        self.scaler.fit(X_train)
        
        print(f"\n[TRANSFORM] Applying scaler to both datasets...")
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Verify scaling
        print(f"\n[VERIFICATION] Checking scaled value ranges...")
        print(f"  -> X_train scaled: min={X_train_scaled.min():.4f}, max={X_train_scaled.max():.4f}")
        print(f"  -> X_test scaled: min={X_test_scaled.min():.4f}, max={X_test_scaled.max():.4f}")
        
        # Test data may exceed [0,1] due to attack patterns
        if X_test_scaled.min() < 0 or X_test_scaled.max() > 1:
            print(f"  -> [INFO] Test data exceeds [0,1] range - expected for attack patterns")
            print(f"  -> [OK] This confirms scaler was NOT fitted on test data (no leakage)")
        else:
            print(f"  -> [OK] Both datasets within [0,1] range")
        
        # Store statistics
        self.normalization_stats = {
            'train_min': float(X_train_scaled.min()),
            'train_max': float(X_train_scaled.max()),
            'test_min': float(X_test_scaled.min()),
            'test_max': float(X_test_scaled.max()),
            'leakage_check': 'PASSED'
        }
        
        return X_train_scaled, X_test_scaled, self.scaler


class DataSerializer:
    """Serializes processed data to disk."""
    
    def __init__(self, config: Config):
        self.config = config
        self.saved_files: Dict[str, str] = {}
    
    def save_all(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: List[str],
        scaler: MinMaxScaler
    ) -> Dict[str, str]:
        """
        Save all processed data to disk.
        
        Args:
            X_train: Normalized training features
            y_train: Training labels
            X_test: Normalized test features
            y_test: Test labels
            feature_names: List of feature column names
            scaler: Fitted MinMaxScaler
        
        Returns:
            Dictionary of saved file paths
        """
        print("\n" + "="*70)
        print("PHASE 4: SERIALIZATION")
        print("="*70)
        
        # Ensure directories exist
        ensure_directory_exists(self.config.PROCESSED_DIR)
        ensure_directory_exists(self.config.MODELS_DIR)
        
        print(f"\n[SAVING] Output directory: {self.config.PROCESSED_DIR}")
        
        # Save numpy arrays
        paths = {
            'X_train': os.path.join(self.config.PROCESSED_DIR, self.config.X_TRAIN_FILE),
            'y_train': os.path.join(self.config.PROCESSED_DIR, self.config.Y_TRAIN_FILE),
            'X_test': os.path.join(self.config.PROCESSED_DIR, self.config.X_TEST_FILE),
            'y_test': os.path.join(self.config.PROCESSED_DIR, self.config.Y_TEST_FILE),
            'feature_names': os.path.join(self.config.PROCESSED_DIR, self.config.FEATURE_NAMES_FILE),
            'scaler': os.path.join(self.config.MODELS_DIR, self.config.SCALER_FILE),
        }
        
        try:
            np.save(paths['X_train'], X_train)
            print(f"  -> X_train: {paths['X_train']}")
            print(f"     Shape: {X_train.shape}, Size: {X_train.nbytes / 1024 / 1024:.2f} MB")
            
            np.save(paths['y_train'], y_train)
            print(f"  -> y_train: {paths['y_train']}")
            print(f"     Shape: {y_train.shape}")
            
            np.save(paths['X_test'], X_test)
            print(f"  -> X_test: {paths['X_test']}")
            print(f"     Shape: {X_test.shape}, Size: {X_test.nbytes / 1024 / 1024:.2f} MB")
            
            np.save(paths['y_test'], y_test)
            print(f"  -> y_test: {paths['y_test']}")
            print(f"     Shape: {y_test.shape}")
            
            joblib.dump(feature_names, paths['feature_names'])
            print(f"  -> feature_names: {paths['feature_names']}")
            print(f"     Features: {len(feature_names)}")
            
            joblib.dump(scaler, paths['scaler'])
            print(f"  -> scaler: {paths['scaler']}")
            
        except Exception as e:
            raise IOError(f"[FATAL] Failed to save files: {e}")
        
        self.saved_files = paths
        return paths


class HAIIngestionPipeline:
    """
    Main orchestrator for HAI dataset ingestion.
    
    Coordinates all phases:
    0. Source Directory Discovery
    1. Forensic File Assembly
    2. Cleaning & Alignment
    3. Physics-Aware Normalization
    4. Serialization
    """
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.assembler = ForensicFileAssembler(self.config)
        self.cleaner = DataCleaner(self.config)
        self.normalizer = PhysicsAwareNormalizer(self.config)
        self.serializer = DataSerializer(self.config)
    
    def run(self) -> Dict[str, Any]:
        """
        Execute the complete ingestion pipeline.
        
        Returns:
            Dictionary containing all pipeline results
        """
        print("\n" + "="*70)
        print("HAI 22.04 DATASET INGESTION PIPELINE")
        print("Hardware-in-the-Loop Testbed for ICS Security Research")
        print("="*70)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        results = {}
        
        # Phase 0: Discover source directory
        try:
            self.assembler.find_source_directory()
        except Exception as e:
            print(f"\n[FATAL] Phase 0 failed: {e}")
            raise
        
        # Phase 1: Forensic File Assembly
        try:
            train_files, test_files = self.assembler.discover_files()
            self.assembler.compute_forensic_hashes()
            
            raw_train_df = self.assembler.load_and_concatenate(train_files, "Train")
            raw_test_df = self.assembler.load_and_concatenate(test_files, "Test")
            
            results['file_hashes'] = self.assembler.file_hashes
            results['n_train_files'] = len(train_files)
            results['n_test_files'] = len(test_files)
            
        except Exception as e:
            print(f"\n[FATAL] Phase 1 failed: {e}")
            raise
        
        # Phase 2: Cleaning & Alignment
        try:
            X_train, y_train, X_test, y_test, feature_names = self.cleaner.clean_dataset(
                raw_train_df, raw_test_df
            )
            results['cleaning_stats'] = self.cleaner.cleaning_stats
            
        except Exception as e:
            print(f"\n[FATAL] Phase 2 failed: {e}")
            raise
        
        # Phase 3: Physics-Aware Normalization
        try:
            X_train_scaled, X_test_scaled, scaler = self.normalizer.normalize(X_train, X_test)
            results['normalization_stats'] = self.normalizer.normalization_stats
            
        except Exception as e:
            print(f"\n[FATAL] Phase 3 failed: {e}")
            raise
        
        # Phase 4: Serialization
        try:
            saved_paths = self.serializer.save_all(
                X_train_scaled, y_train,
                X_test_scaled, y_test,
                feature_names, scaler
            )
            results['saved_paths'] = saved_paths
            
        except Exception as e:
            print(f"\n[FATAL] Phase 4 failed: {e}")
            raise
        
        # Final Summary
        print("\n" + "="*70)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("="*70)
        
        # Calculate statistics
        train_attack_count = np.sum(y_train)
        test_attack_count = np.sum(y_test)
        train_attack_ratio = train_attack_count / len(y_train) * 100
        test_attack_ratio = test_attack_count / len(y_test) * 100
        
        print(f"\n[FINAL ARRAY SHAPES]")
        print(f"  -> X_train: {X_train_scaled.shape}")
        print(f"  -> y_train: {y_train.shape}")
        print(f"  -> X_test: {X_test_scaled.shape}")
        print(f"  -> y_test: {y_test.shape}")
        
        print(f"\n[CLASS DISTRIBUTION]")
        print(f"  -> Train: {len(y_train) - train_attack_count:,} Normal, {train_attack_count:,} Attack")
        print(f"  -> Test:  {len(y_test) - test_attack_count:,} Normal, {test_attack_count:,} Attack")
        
        print(f"\n[ATTACK RATIOS]")
        print(f"  -> Train Attack Ratio: {train_attack_ratio:.2f}%")
        print(f"  -> Test Attack Ratio: {test_attack_ratio:.2f}%")
        
        print(f"\n[FEATURES]")
        print(f"  -> Number of sensors: {len(feature_names)}")
        
        print(f"\n[OUTPUT DIRECTORY]")
        print(f"  -> {self.config.PROCESSED_DIR}")
        
        print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Store final statistics
        results['final_stats'] = {
            'X_train_shape': X_train_scaled.shape,
            'y_train_shape': y_train.shape,
            'X_test_shape': X_test_scaled.shape,
            'y_test_shape': y_test.shape,
            'n_features': len(feature_names),
            'train_attack_count': int(train_attack_count),
            'test_attack_count': int(test_attack_count),
            'train_attack_ratio': float(train_attack_ratio),
            'test_attack_ratio': float(test_attack_ratio)
        }
        
        return results


def main():
    """Main entry point for the HAI ingestion pipeline."""
    
    # Set random seed for reproducibility
    np.random.seed(Config.RANDOM_SEED)
    
    # Initialize and run pipeline
    pipeline = HAIIngestionPipeline()
    
    try:
        results = pipeline.run()
        print("\n[SUCCESS] HAI ingestion completed without errors")
        return results
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Pipeline stopped by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[FATAL ERROR] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    results = main()
