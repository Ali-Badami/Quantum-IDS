#!/usr/bin/env python3


import os
import sys
import hashlib
import shutil
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn imports
from sklearn.preprocessing import MinMaxScaler
import joblib



class Config:
    """Centralized configuration for reproducibility."""
    
    # Input paths (Google Colab/Drive)
    NORMAL_DATA_PATH = "/content/drive/MyDrive/Datasets/swat/normal.csv"
    ATTACK_DATA_PATH = "/content/drive/MyDrive/Datasets/swat/attack.csv"
    
    # Output directories
    BASE_DIR = "/content/drive/MyDrive/Quantum_Research"
    PROCESSED_DIR = f"{BASE_DIR}/02_Processed_Data"
    MODELS_DIR = f"{BASE_DIR}/04_Models"
    BACKUPS_DIR = f"{BASE_DIR}/BACKUPS"
    MANIFEST_PATH = f"{BASE_DIR}/project_manifest.txt"
    
    # SWaT Protocol Parameters
    WARMUP_ROWS_TO_DROP = 21600  # Standard SWaT protocol: 6 hours @ 1Hz = 21600 samples
    TRAIN_SPLIT_RATIO = 0.80    # 80% train, 20% test from Normal data
    
    # Label column
    LABEL_COLUMN = "Normal/Attack"
    LABEL_MAPPING = {"Normal": 0, "Attack": 1}
    
    # Random seed (for any future stochastic operations)
    RANDOM_SEED = 42



def calculate_md5(filepath: str, chunk_size: int = 8192) -> str:
    """
    Calculate MD5 hash of a file for forensic verification.
    
    Args:
        filepath: Path to the file
        chunk_size: Size of chunks to read (default 8KB)
    
    Returns:
        MD5 hash as hexadecimal string
    """
    md5_hash = hashlib.md5()
    try:
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(chunk_size), b''):
                md5_hash.update(chunk)
        return md5_hash.hexdigest()
    except FileNotFoundError:
        raise FileNotFoundError(f"[FORENSIC ERROR] File not found: {filepath}")
    except PermissionError:
        raise PermissionError(f"[FORENSIC ERROR] Permission denied: {filepath}")


def ensure_directory_exists(dir_path: str) -> bool:
    """
    Create directory if it doesn't exist.
    
    Args:
        dir_path: Path to directory
    
    Returns:
        True if created or exists, raises exception on failure
    """
    try:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        return True
    except PermissionError:
        raise PermissionError(f"[I/O ERROR] Cannot create directory: {dir_path}")
    except OSError as e:
        raise OSError(f"[I/O ERROR] Directory creation failed: {dir_path} - {e}")


def log_to_manifest(manifest_path: str, entry: str) -> None:
    """
    Append an entry to the project manifest log.
    
    Args:
        manifest_path: Path to manifest file
        entry: Log entry to append
    """
    try:
        with open(manifest_path, 'a') as f:
            f.write(entry + "\n")
    except PermissionError:
        print(f"[WARNING] Could not write to manifest: {manifest_path}")
    except OSError as e:
        print(f"[WARNING] Manifest write error: {e}")



class ForensicVerifier:
    """Handles forensic verification and backup operations."""
    
    def __init__(self, config: Config):
        self.config = config
        self.verification_results: Dict[str, str] = {}
    
    def verify_and_backup(self) -> Dict[str, str]:
        """
        Perform forensic verification and backup of raw data files.
        
        Returns:
            Dictionary with file paths and their MD5 hashes
        """
        print("\n" + "="*70)
        print("PHASE 1: FORENSIC VERIFICATION & BACKUP")
        print("="*70)
        
        # Ensure backup directory exists
        ensure_directory_exists(self.config.BACKUPS_DIR)
        
        files_to_verify = [
            ("Normal", self.config.NORMAL_DATA_PATH),
            ("Attack", self.config.ATTACK_DATA_PATH)
        ]
        
        for name, filepath in files_to_verify:
            print(f"\n[VERIFYING] {name} Dataset: {filepath}")
            
            # Calculate MD5 hash
            try:
                md5_hash = calculate_md5(filepath)
                self.verification_results[filepath] = md5_hash
                print(f"  -> MD5 Hash: {md5_hash}")
            except Exception as e:
                print(f"  -> [FATAL] Verification failed: {e}")
                raise
            
            # Backup file
            backup_path = os.path.join(
                self.config.BACKUPS_DIR, 
                os.path.basename(filepath)
            )
            
            try:
                if os.path.exists(backup_path):
                    # Verify existing backup matches
                    backup_hash = calculate_md5(backup_path)
                    if backup_hash == md5_hash:
                        print(f"  -> [SKIP] Backup exists and verified: {backup_path}")
                    else:
                        print(f"  -> [WARNING] Backup hash mismatch! Re-copying...")
                        shutil.copy2(filepath, backup_path)
                        print(f"  -> [BACKUP] Re-copied to: {backup_path}")
                else:
                    shutil.copy2(filepath, backup_path)
                    print(f"  -> [BACKUP] Copied to: {backup_path}")
            except Exception as e:
                print(f"  -> [ERROR] Backup failed: {e}")
                # Continue execution - backup failure is non-fatal
        
        return self.verification_results



class StrictDataCleaner:
    """Handles strict data cleaning per SWaT protocol."""
    
    def __init__(self, config: Config):
        self.config = config
        self.cleaning_stats: Dict[str, any] = {}
    
    def load_and_clean(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and clean both Normal and Attack datasets.
        
        Returns:
            Tuple of (cleaned_normal_df, cleaned_attack_df)
        """
        print("\n" + "="*70)
        print("PHASE 2: STRICT DATA CLEANING")
        print("="*70)
        
        # Load Normal dataset
        print(f"\n[LOADING] Normal Dataset...")
        try:
            df_normal = pd.read_csv(
                self.config.NORMAL_DATA_PATH,
                low_memory=False
            )
            print(f"  -> Raw shape: {df_normal.shape}")
            self.cleaning_stats['normal_raw_shape'] = df_normal.shape
        except Exception as e:
            raise IOError(f"[FATAL] Failed to load Normal data: {e}")
        
        # Load Attack dataset
        print(f"\n[LOADING] Attack Dataset...")
        try:
            df_attack = pd.read_csv(
                self.config.ATTACK_DATA_PATH,
                low_memory=False
            )
            print(f"  -> Raw shape: {df_attack.shape}")
            self.cleaning_stats['attack_raw_shape'] = df_attack.shape
        except Exception as e:
            raise IOError(f"[FATAL] Failed to load Attack data: {e}")
        
        # Header Sanitation: Strip whitespace from column names
        print("\n[SANITIZING] Column headers...")
        df_normal.columns = df_normal.columns.str.strip()
        df_attack.columns = df_attack.columns.str.strip()
        print(f"  -> Normal columns: {len(df_normal.columns)}")
        print(f"  -> Attack columns: {len(df_attack.columns)}")
        
        # Verify label column exists
        if self.config.LABEL_COLUMN not in df_normal.columns:
            raise ValueError(f"[FATAL] Label column '{self.config.LABEL_COLUMN}' not found in Normal data")
        if self.config.LABEL_COLUMN not in df_attack.columns:
            raise ValueError(f"[FATAL] Label column '{self.config.LABEL_COLUMN}' not found in Attack data")
        
        # Warm-up Removal (Standard SWaT protocol)
        print(f"\n[WARM-UP REMOVAL] Dropping first {self.config.WARMUP_ROWS_TO_DROP} rows from Normal data...")
        rows_before = len(df_normal)
        df_normal = df_normal.iloc[self.config.WARMUP_ROWS_TO_DROP:].reset_index(drop=True)
        print(f"  -> Before: {rows_before}, After: {len(df_normal)}")
        self.cleaning_stats['normal_after_warmup'] = df_normal.shape
        
        # Consistency Check: Ensure identical feature columns
        print("\n[CONSISTENCY CHECK] Verifying column alignment...")
        normal_cols = set(df_normal.columns)
        attack_cols = set(df_attack.columns)
        
        if normal_cols != attack_cols:
            missing_in_attack = normal_cols - attack_cols
            missing_in_normal = attack_cols - normal_cols
            
            if missing_in_attack:
                print(f"  -> [WARNING] Columns missing in Attack: {missing_in_attack}")
            if missing_in_normal:
                print(f"  -> [WARNING] Columns missing in Normal: {missing_in_normal}")
            
            # Use intersection of columns
            common_cols = list(normal_cols & attack_cols)
            df_normal = df_normal[common_cols]
            df_attack = df_attack[common_cols]
            print(f"  -> Using {len(common_cols)} common columns")
        else:
            print(f"  -> [OK] All {len(normal_cols)} columns match")
        
        # Ensure column order is identical
        col_order = list(df_normal.columns)
        df_attack = df_attack[col_order]
        
        # Handle timestamp column if present
        timestamp_cols = [c for c in df_normal.columns if 'timestamp' in c.lower() or 'time' in c.lower()]
        if timestamp_cols:
            print(f"\n[INFO] Timestamp columns detected: {timestamp_cols}")
            print("  -> Will be excluded from features during splitting")
        
        self.cleaning_stats['final_normal_shape'] = df_normal.shape
        self.cleaning_stats['final_attack_shape'] = df_attack.shape
        
        return df_normal, df_attack



class LeakageProofSplitter:
    """Handles time-series aware train/test splitting."""
    
    def __init__(self, config: Config):
        self.config = config
        self.split_stats: Dict[str, any] = {}
    
    def split_data(
        self, 
        df_normal: pd.DataFrame, 
        df_attack: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Perform leakage-proof time-series splitting.
        
        CRITICAL: NO SHUFFLING - maintains temporal order.
        
        Train Set: First 80% of Normal data
        Test Set: Last 20% of Normal data + ALL Attack data
        
        Args:
            df_normal: Cleaned normal operation data
            df_attack: Cleaned attack data
        
        Returns:
            Tuple of (X_train, y_train, X_test, y_test, feature_names)
        """
        print("\n" + "="*70)
        print("PHASE 3: LEAKAGE-PROOF TIME-SERIES SPLITTING")
        print("="*70)
        
        # Identify feature columns (exclude label and timestamp)
        exclude_cols = [self.config.LABEL_COLUMN]
        timestamp_cols = [c for c in df_normal.columns if 'timestamp' in c.lower() or 'time' in c.lower()]
        exclude_cols.extend(timestamp_cols)
        
        feature_cols = [c for c in df_normal.columns if c not in exclude_cols]
        print(f"\n[FEATURES] Using {len(feature_cols)} feature columns")
        print(f"  -> Excluded: {exclude_cols}")
        
        # Encode labels
        print(f"\n[ENCODING] Label column: '{self.config.LABEL_COLUMN}'")
        df_normal[self.config.LABEL_COLUMN] = df_normal[self.config.LABEL_COLUMN].str.strip()
        df_attack[self.config.LABEL_COLUMN] = df_attack[self.config.LABEL_COLUMN].str.strip()
        
        # Map labels to binary
        df_normal['label_encoded'] = df_normal[self.config.LABEL_COLUMN].map(self.config.LABEL_MAPPING)
        df_attack['label_encoded'] = df_attack[self.config.LABEL_COLUMN].map(self.config.LABEL_MAPPING)
        
        # Handle any unmapped labels
        if df_normal['label_encoded'].isna().any():
            unique_labels = df_normal[self.config.LABEL_COLUMN].unique()
            print(f"  -> [WARNING] Found unmapped labels in Normal: {unique_labels}")
            # Default unmapped to 0 (Normal)
            df_normal['label_encoded'] = df_normal['label_encoded'].fillna(0).astype(int)
        
        if df_attack['label_encoded'].isna().any():
            unique_labels = df_attack[self.config.LABEL_COLUMN].unique()
            print(f"  -> [WARNING] Found unmapped labels in Attack: {unique_labels}")
            # Default unmapped to 1 (Attack)
            df_attack['label_encoded'] = df_attack['label_encoded'].fillna(1).astype(int)
        
        print(f"  -> Normal label distribution:\n{df_normal['label_encoded'].value_counts().to_dict()}")
        print(f"  -> Attack label distribution:\n{df_attack['label_encoded'].value_counts().to_dict()}")
        
        # Calculate split index (NO SHUFFLING - time-series order)
        n_normal = len(df_normal)
        split_idx = int(n_normal * self.config.TRAIN_SPLIT_RATIO)
        
        print(f"\n[SPLITTING] Time-series aware split (NO SHUFFLING)")
        print(f"  -> Total Normal samples: {n_normal}")
        print(f"  -> Split index (80%): {split_idx}")
        
        # Train set: First 80% of Normal data
        train_df = df_normal.iloc[:split_idx]
        
        # Test set: Last 20% of Normal + ALL Attack data
        test_normal = df_normal.iloc[split_idx:]
        test_df = pd.concat([test_normal, df_attack], axis=0, ignore_index=True)
        
        print(f"\n[TRAIN SET] First {self.config.TRAIN_SPLIT_RATIO*100:.0f}% of Normal data")
        print(f"  -> Samples: {len(train_df)}")
        print(f"  -> Label distribution: {train_df['label_encoded'].value_counts().to_dict()}")
        
        print(f"\n[TEST SET] Last {(1-self.config.TRAIN_SPLIT_RATIO)*100:.0f}% Normal + ALL Attack")
        print(f"  -> Normal samples: {len(test_normal)}")
        print(f"  -> Attack samples: {len(df_attack)}")
        print(f"  -> Total samples: {len(test_df)}")
        print(f"  -> Label distribution: {test_df['label_encoded'].value_counts().to_dict()}")
        
        # --- PHYSICS-AWARE NaN HANDLING ---
        # In ICS/SCADA systems, missing sensor values typically mean "sensor hold"
        # (the physical value persists), NOT "value is zero" (tank empty).
        # Forward fill preserves the last known good reading.
        print("\n[PHYSICS-AWARE NaN HANDLING] Applying forward fill for sensor data...")
        
        train_nan_before = train_df[feature_cols].isna().sum().sum()
        test_nan_before = test_df[feature_cols].isna().sum().sum()
        
        if train_nan_before > 0:
            print(f"  -> Train NaNs before ffill: {train_nan_before}")
        if test_nan_before > 0:
            print(f"  -> Test NaNs before ffill: {test_nan_before}")
        
        # Forward fill, then backward fill for edge cases (start of dataset)
        train_df[feature_cols] = train_df[feature_cols].ffill().bfill()
        test_df[feature_cols] = test_df[feature_cols].ffill().bfill()
        
        train_nan_after = train_df[feature_cols].isna().sum().sum()
        test_nan_after = test_df[feature_cols].isna().sum().sum()
        
        print(f"  -> Train NaNs after ffill: {train_nan_after}")
        print(f"  -> Test NaNs after ffill: {test_nan_after}")
        # --- END PHYSICS-AWARE NaN HANDLING ---
        
        # Extract arrays (Now safe from NaNs due to ffill)
        X_train = train_df[feature_cols].values.astype(np.float32)
        y_train = train_df['label_encoded'].values.astype(np.int32)
        X_test = test_df[feature_cols].values.astype(np.float32)
        y_test = test_df['label_encoded'].values.astype(np.int32)
        
        # Final safety net: Handle any remaining Inf values (NaNs should be gone)
        print("\n[DATA QUALITY] Final check for Inf values...")
        for name, arr in [("X_train", X_train), ("X_test", X_test)]:
            nan_count = np.isnan(arr).sum()
            inf_count = np.isinf(arr).sum()
            if nan_count > 0 or inf_count > 0:
                print(f"  -> [WARNING] {name}: {nan_count} NaN, {inf_count} Inf values - applying safety fallback")
                arr = np.nan_to_num(arr, nan=0.0, posinf=1e10, neginf=-1e10)
                if name == "X_train":
                    X_train = arr
                else:
                    X_test = arr
            else:
                print(f"  -> [OK] {name}: Clean (no NaN/Inf)")
        
        self.split_stats = {
            'n_train': len(X_train),
            'n_test': len(X_test),
            'n_features': len(feature_cols),
            'train_attack_ratio': (y_train == 1).sum() / len(y_train),
            'test_attack_ratio': (y_test == 1).sum() / len(y_test)
        }
        
        return X_train, y_train, X_test, y_test, feature_cols



class PhysicsAwareNormalizer:
    """Handles normalization with proper train-only fitting."""
    
    def __init__(self, config: Config):
        self.config = config
        self.scaler: Optional[MinMaxScaler] = None
    
    def normalize(
        self, 
        X_train: np.ndarray, 
        X_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
        """
        Apply physics-aware normalization.
        
        CRITICAL: Scaler is fitted ONLY on training data to prevent data leakage.
        
        Args:
            X_train: Training features
            X_test: Test features
        
        Returns:
            Tuple of (X_train_scaled, X_test_scaled, fitted_scaler)
        """
        print("\n" + "="*70)
        print("PHASE 4: PHYSICS-AWARE NORMALIZATION")
        print("="*70)
        
        print("\n[SCALER] Initializing MinMaxScaler")
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
        # Fit ONLY on training data
        print(f"[FIT] Fitting scaler on X_train ({X_train.shape[0]} samples)")
        try:
            self.scaler.fit(X_train)
        except Exception as e:
            raise ValueError(f"[FATAL] Scaler fitting failed: {e}")
        
        # Transform both sets
        print(f"[TRANSFORM] Transforming X_train...")
        try:
            X_train_scaled = self.scaler.transform(X_train)
        except Exception as e:
            raise ValueError(f"[FATAL] X_train transform failed: {e}")
        
        print(f"[TRANSFORM] Transforming X_test...")
        try:
            X_test_scaled = self.scaler.transform(X_test)
        except Exception as e:
            raise ValueError(f"[FATAL] X_test transform failed: {e}")
        
        # Verify scaling
        print("\n[VERIFICATION] Checking scaled data ranges...")
        train_min, train_max = X_train_scaled.min(), X_train_scaled.max()
        test_min, test_max = X_test_scaled.min(), X_test_scaled.max()
        
        print(f"  -> X_train_scaled: min={train_min:.4f}, max={train_max:.4f}")
        print(f"  -> X_test_scaled: min={test_min:.4f}, max={test_max:.4f}")
        
        if test_min < -0.5 or test_max > 1.5:
            print("  -> [WARNING] X_test has values significantly outside [0,1]")
            print("     This is expected if test data has attack patterns not seen in training")
        
        return X_train_scaled.astype(np.float32), X_test_scaled.astype(np.float32), self.scaler


class DataSerializer:
    """Handles serialization of processed data and artifacts."""
    
    def __init__(self, config: Config):
        self.config = config
        self.saved_files: List[str] = []
    
    def serialize_all(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        scaler: MinMaxScaler,
        feature_names: List[str]
    ) -> Dict[str, str]:
        """
        Serialize all processed data and artifacts.
        
        Args:
            X_train, y_train, X_test, y_test: Processed arrays
            scaler: Fitted MinMaxScaler
            feature_names: List of feature column names
        
        Returns:
            Dictionary mapping artifact names to file paths
        """
        print("\n" + "="*70)
        print("PHASE 5: SERIALIZATION")
        print("="*70)
        
        # Ensure directories exist
        ensure_directory_exists(self.config.PROCESSED_DIR)
        ensure_directory_exists(self.config.MODELS_DIR)
        
        saved_paths = {}
        
        # Save numpy arrays
        arrays_to_save = [
            ("X_train.npy", X_train),
            ("y_train.npy", y_train),
            ("X_test.npy", X_test),
            ("y_test.npy", y_test)
        ]
        
        print("\n[NUMPY ARRAYS]")
        for filename, array in arrays_to_save:
            filepath = os.path.join(self.config.PROCESSED_DIR, filename)
            try:
                np.save(filepath, array)
                saved_paths[filename] = filepath
                self.saved_files.append(filepath)
                print(f"  -> Saved: {filename} | Shape: {array.shape} | Dtype: {array.dtype}")
            except Exception as e:
                print(f"  -> [ERROR] Failed to save {filename}: {e}")
                raise
        
        # Save scaler
        print("\n[SCALER]")
        scaler_path = os.path.join(self.config.MODELS_DIR, "scaler.joblib")
        try:
            joblib.dump(scaler, scaler_path)
            saved_paths["scaler.joblib"] = scaler_path
            self.saved_files.append(scaler_path)
            print(f"  -> Saved: scaler.joblib to {self.config.MODELS_DIR}")
        except Exception as e:
            print(f"  -> [ERROR] Failed to save scaler: {e}")
            raise
        
        # Save feature names
        print("\n[FEATURE NAMES]")
        features_path = os.path.join(self.config.PROCESSED_DIR, "feature_names.joblib")
        try:
            joblib.dump(feature_names, features_path)
            saved_paths["feature_names.joblib"] = features_path
            self.saved_files.append(features_path)
            print(f"  -> Saved: feature_names.joblib ({len(feature_names)} features)")
        except Exception as e:
            print(f"  -> [ERROR] Failed to save feature names: {e}")
            raise
        
        return saved_paths


class ManifestLogger:
    """Handles audit trail logging to manifest file."""
    
    def __init__(self, config: Config):
        self.config = config
    
    def log_pipeline_run(
        self,
        md5_hashes: Dict[str, str],
        shapes: Dict[str, tuple],
        saved_paths: Dict[str, str]
    ) -> None:
        """
        Log a complete pipeline run to the manifest.
        
        Args:
            md5_hashes: Dictionary of file paths to MD5 hashes
            shapes: Dictionary of array names to shapes
            saved_paths: Dictionary of artifact names to paths
        """
        print("\n" + "="*70)
        print("PHASE 6: MANIFEST LOGGING")
        print("="*70)
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        log_entry = f"""
[SWAT DATA INGESTION LOG]

Timestamp: {timestamp}
Pipeline Version: 1.0
Python Version: {sys.version.split()[0]}

[FORENSIC VERIFICATION - MD5 HASHES]
"""
        for filepath, md5_hash in md5_hashes.items():
            log_entry += f"  {os.path.basename(filepath)}: {md5_hash}\n"
        
        log_entry += "\n[PROCESSED DATA SHAPES]\n"
        for name, shape in shapes.items():
            log_entry += f"  {name}: {shape}\n"
        
        log_entry += "\n[SAVED ARTIFACTS]\n"
        for name, path in saved_paths.items():
            log_entry += f"  {name}: {path}\n"
        
        log_entry += f"""
[PROTOCOL PARAMETERS]
  Warm-up Rows Dropped: {self.config.WARMUP_ROWS_TO_DROP}
  Train/Test Split: {self.config.TRAIN_SPLIT_RATIO}/{1-self.config.TRAIN_SPLIT_RATIO}
  Normalization: MinMaxScaler (fitted on train only)
  Time-Series Order: PRESERVED (No Shuffling)
"""
        
        try:
            # Ensure parent directory exists
            ensure_directory_exists(os.path.dirname(self.config.MANIFEST_PATH))
            log_to_manifest(self.config.MANIFEST_PATH, log_entry)
            print(f"[MANIFEST] Log appended to: {self.config.MANIFEST_PATH}")
        except Exception as e:
            print(f"[WARNING] Failed to write manifest: {e}")
        
        # Also print to console
        print(log_entry)



class SWaTIngestionPipeline:
    """
    Main orchestrator for the SWaT data ingestion pipeline.
    
    This class coordinates all pipeline phases in sequence:
    1. Forensic Verification & Backup
    2. Strict Data Cleaning
    3. Leakage-Proof Splitting
    4. Physics-Aware Normalization
    5. Serialization
    6. Manifest Logging
    """
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.forensic_verifier = ForensicVerifier(self.config)
        self.cleaner = StrictDataCleaner(self.config)
        self.splitter = LeakageProofSplitter(self.config)
        self.normalizer = PhysicsAwareNormalizer(self.config)
        self.serializer = DataSerializer(self.config)
        self.logger = ManifestLogger(self.config)
    
    def run(self) -> Dict[str, any]:
        """
        Execute the complete data ingestion pipeline.
        
        Returns:
            Dictionary containing all pipeline results and statistics
        """
        print("\n" + "="*70)
        print("SWAT DATASET FORENSIC INGESTION PIPELINE")
        print("="*70)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Target: Quantum-Resilient IDS for Critical Infrastructure")
        
        results = {}
        
        # Phase 1: Forensic Verification & Backup
        try:
            md5_hashes = self.forensic_verifier.verify_and_backup()
            results['md5_hashes'] = md5_hashes
        except Exception as e:
            print(f"\n[FATAL] Phase 1 failed: {e}")
            raise
        
        # Phase 2: Strict Data Cleaning
        try:
            df_normal, df_attack = self.cleaner.load_and_clean()
            results['cleaning_stats'] = self.cleaner.cleaning_stats
        except Exception as e:
            print(f"\n[FATAL] Phase 2 failed: {e}")
            raise
        
        # Phase 3: Leakage-Proof Splitting
        try:
            X_train, y_train, X_test, y_test, feature_names = self.splitter.split_data(
                df_normal, df_attack
            )
            results['split_stats'] = self.splitter.split_stats
        except Exception as e:
            print(f"\n[FATAL] Phase 3 failed: {e}")
            raise
        
        # Phase 4: Physics-Aware Normalization
        try:
            X_train_scaled, X_test_scaled, scaler = self.normalizer.normalize(
                X_train, X_test
            )
        except Exception as e:
            print(f"\n[FATAL] Phase 4 failed: {e}")
            raise
        
        # Phase 5: Serialization
        try:
            saved_paths = self.serializer.serialize_all(
                X_train_scaled, y_train,
                X_test_scaled, y_test,
                scaler, feature_names
            )
            results['saved_paths'] = saved_paths
        except Exception as e:
            print(f"\n[FATAL] Phase 5 failed: {e}")
            raise
        
        # Phase 6: Manifest Logging
        shapes = {
            'X_train': X_train_scaled.shape,
            'y_train': y_train.shape,
            'X_test': X_test_scaled.shape,
            'y_test': y_test.shape,
            'n_features': len(feature_names)
        }
        results['shapes'] = shapes
        
        self.logger.log_pipeline_run(md5_hashes, shapes, saved_paths)
        
        # Final Summary
        print("\n" + "="*70)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("="*70)
        print(f"\nFinal Array Shapes:")
        print(f"  X_train: {X_train_scaled.shape}")
        print(f"  y_train: {y_train.shape}")
        print(f"  X_test:  {X_test_scaled.shape}")
        print(f"  y_test:  {y_test.shape}")
        print(f"\nFeatures: {len(feature_names)}")
        print(f"Train Attack Ratio: {(y_train == 1).sum() / len(y_train):.4f}")
        print(f"Test Attack Ratio:  {(y_test == 1).sum() / len(y_test):.4f}")
        print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return results



def main():
    """Main entry point for the SWaT ingestion pipeline."""
    
    # Set random seed for reproducibility
    np.random.seed(Config.RANDOM_SEED)
    
    # Initialize and run pipeline
    pipeline = SWaTIngestionPipeline()
    
    try:
        results = pipeline.run()
        print("\n[SUCCESS] Pipeline completed without errors")
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
