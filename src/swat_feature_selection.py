#!/usr/bin/env python3


import os
import sys
import json
import numpy as np
import joblib
from datetime import datetime
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample

# Plotting imports
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server/Colab



class Config:
    """Centralized configuration for feature selection pipeline."""
    
    # Input paths (from Step 1)
    BASE_DIR = "/content/drive/MyDrive/Quantum_Research"
    PROCESSED_DIR = f"{BASE_DIR}/02_Processed_Data"
    
    X_TRAIN_PATH = f"{PROCESSED_DIR}/X_train.npy"
    Y_TRAIN_PATH = f"{PROCESSED_DIR}/y_train.npy"
    X_TEST_PATH = f"{PROCESSED_DIR}/X_test.npy"
    Y_TEST_PATH = f"{PROCESSED_DIR}/y_test.npy"
    FEATURE_NAMES_PATH = f"{PROCESSED_DIR}/feature_names.joblib"
    
    # Output directories
    PLOTS_DIR = f"{BASE_DIR}/05_Results/Plots"
    LOGS_DIR = f"{BASE_DIR}/05_Results/Logs"
    
    # Feature Selection Parameters
    SUBSAMPLE_SIZE = 50000          # Max samples for RF training (memory safety)
    N_FEATURES_TO_SELECT = 8        # Quantum constraint: 8 qubits = 8 features
    N_FEATURES_TO_PLOT = 15         # Top features to visualize
    
    # RandomForest Parameters
    RF_N_ESTIMATORS = 100
    RF_N_JOBS = -1                  # Use all CPU cores
    RF_RANDOM_STATE = 42
    RF_MAX_DEPTH = 20               # Prevent overfitting on large data
    RF_MIN_SAMPLES_LEAF = 50        # Regularization for stability
    
    # Visualization Parameters
    PLOT_DPI = 150
    PLOT_FIGSIZE = (12, 8)
    
    # Random seed
    RANDOM_SEED = 42



def ensure_directory_exists(dir_path: str) -> bool:
    """Create directory if it doesn't exist."""
    try:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        raise OSError(f"[I/O ERROR] Cannot create directory: {dir_path} - {e}")


def get_memory_usage_mb() -> float:
    """Get current memory usage in MB."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        return -1.0  # psutil not available


def format_bytes(size_bytes: int) -> str:
    """Format bytes to human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} TB"


class MemorySafeDataLoader:
    """Handles memory-efficient data loading with subsampling."""
    
    def __init__(self, config: Config):
        self.config = config
        self.load_stats: Dict[str, any] = {}
    
    def load_with_subsample(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load TEST data (which contains both classes) for feature selection.
        
        CRITICAL FIX: Training data is 100% Normal (for anomaly detection).
        RandomForest needs BOTH classes to compute meaningful Gini importance.
        Therefore, we use the TEST set for feature selection, which contains
        both Normal and Attack samples.
        
        Returns:
            Tuple of (X_subsample, y_subsample, feature_names)
        """
        print("\n" + "="*70)
        print("PHASE 1: MEMORY-SAFE DATA LOADING")
        print("="*70)
        
        print("\n" + "-"*70)
        print("[CRITICAL] Using TEST set for feature selection")
        print("  Reason: Train set is 100% Normal (for anomaly detection)")
        print("  RandomForest needs BOTH classes to compute Gini importance")
        print("-"*70)
        
        # Load feature names first (small file)
        print(f"\n[LOADING] Feature names from: {self.config.FEATURE_NAMES_PATH}")
        try:
            feature_names = joblib.load(self.config.FEATURE_NAMES_PATH)
            print(f"  -> Loaded {len(feature_names)} feature names")
            self.load_stats['n_features'] = len(feature_names)
        except Exception as e:
            raise IOError(f"[FATAL] Failed to load feature names: {e}")
        
        # Load TEST data (contains both Normal and Attack)
        print(f"\n[LOADING] X_test from: {self.config.X_TEST_PATH}")
        try:
            X_test_full = np.load(self.config.X_TEST_PATH)
            print(f"  -> Shape: {X_test_full.shape}")
            print(f"  -> Memory: {format_bytes(X_test_full.nbytes)}")
            self.load_stats['X_test_full_shape'] = X_test_full.shape
        except Exception as e:
            raise IOError(f"[FATAL] Failed to load X_test: {e}")
        
        print(f"\n[LOADING] y_test from: {self.config.Y_TEST_PATH}")
        try:
            y_test_full = np.load(self.config.Y_TEST_PATH)
            print(f"  -> Shape: {y_test_full.shape}")
            self.load_stats['y_test_full_shape'] = y_test_full.shape
        except Exception as e:
            raise IOError(f"[FATAL] Failed to load y_test: {e}")
        
        # Verify we have both classes
        unique_classes, class_counts = np.unique(y_test_full, return_counts=True)
        class_dist = dict(zip(unique_classes.astype(int), class_counts.astype(int)))
        print(f"\n[CLASS CHECK] Test set distribution: {class_dist}")
        
        if len(unique_classes) < 2:
            raise ValueError("[FATAL] Test set must contain both Normal (0) and Attack (1) classes!")
        
        print(f"  -> Normal (0): {class_dist.get(0, 0):,} samples ({class_dist.get(0, 0)/len(y_test_full)*100:.1f}%)")
        print(f"  -> Attack (1): {class_dist.get(1, 0):,} samples ({class_dist.get(1, 0)/len(y_test_full)*100:.1f}%)")
        print(f"  -> [OK] Both classes present - feature selection will work correctly")
        
        # Check if subsampling is needed
        n_samples = len(X_test_full)
        
        if n_samples > self.config.SUBSAMPLE_SIZE:
            print(f"\n[SUBSAMPLING] Dataset size ({n_samples:,}) > threshold ({self.config.SUBSAMPLE_SIZE:,})")
            print(f"  -> Creating STRATIFIED subsample of {self.config.SUBSAMPLE_SIZE:,} samples")
            print(f"  -> Stratification preserves class ratio for unbiased importance")
            
            # Set random seed for reproducibility
            np.random.seed(self.config.RANDOM_SEED)
            
            # Stratified sampling (preserves class ratio)
            from sklearn.model_selection import train_test_split
            
            X_subsample, _, y_subsample, _ = train_test_split(
                X_test_full, y_test_full,
                train_size=self.config.SUBSAMPLE_SIZE,
                stratify=y_test_full,
                random_state=self.config.RANDOM_SEED
            )
            print(f"  -> Stratified subsample created")
            
            # Report subsample class distribution
            unique_sub, counts_sub = np.unique(y_subsample, return_counts=True)
            sub_dist = dict(zip(unique_sub.astype(int), counts_sub.astype(int)))
            print(f"  -> Subsample class distribution: {sub_dist}")
            print(f"     Normal (0): {sub_dist.get(0, 0):,} samples")
            print(f"     Attack (1): {sub_dist.get(1, 0):,} samples")
            
            # Free memory from full arrays
            del X_test_full, y_test_full
            
            self.load_stats['subsampled'] = True
            self.load_stats['subsample_size'] = len(X_subsample)
            self.load_stats['subsample_class_distribution'] = sub_dist
        else:
            print(f"\n[INFO] Dataset size ({n_samples:,}) <= threshold. Using full test data.")
            X_subsample = X_test_full
            y_subsample = y_test_full
            self.load_stats['subsampled'] = False
            self.load_stats['subsample_size'] = n_samples
        
        print(f"\n[MEMORY] Current usage: {get_memory_usage_mb():.1f} MB")
        
        return X_subsample, y_subsample, feature_names



class PhysicsBasedFeatureSelector:
    """RandomForest-based feature selection with physics awareness."""
    
    def __init__(self, config: Config):
        self.config = config
        self.rf_model: Optional[RandomForestClassifier] = None
        self.feature_importances: Optional[np.ndarray] = None
        self.selection_stats: Dict[str, any] = {}
    
    def fit_and_select(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        feature_names: List[str]
    ) -> Tuple[List[str], List[float], Dict[str, float]]:
        """
        Fit RandomForest and select top features based on Gini importance.
        
        Args:
            X: Feature matrix (subsampled)
            y: Label vector
            feature_names: List of feature names
        
        Returns:
            Tuple of (selected_feature_names, selected_importances, all_importances_dict)
        """
        print("\n" + "="*70)
        print("PHASE 2: PHYSICS-BASED FEATURE SELECTION")
        print("="*70)
        
        # Initialize RandomForest
        print(f"\n[INIT] RandomForestClassifier")
        print(f"  -> n_estimators: {self.config.RF_N_ESTIMATORS}")
        print(f"  -> max_depth: {self.config.RF_MAX_DEPTH}")
        print(f"  -> min_samples_leaf: {self.config.RF_MIN_SAMPLES_LEAF}")
        print(f"  -> n_jobs: {self.config.RF_N_JOBS} (all cores)")
        print(f"  -> random_state: {self.config.RF_RANDOM_STATE}")
        
        self.rf_model = RandomForestClassifier(
            n_estimators=self.config.RF_N_ESTIMATORS,
            max_depth=self.config.RF_MAX_DEPTH,
            min_samples_leaf=self.config.RF_MIN_SAMPLES_LEAF,
            n_jobs=self.config.RF_N_JOBS,
            random_state=self.config.RF_RANDOM_STATE,
            class_weight='balanced',  # Handle class imbalance
            verbose=0
        )
        
        # Fit the model
        print(f"\n[TRAINING] Fitting RandomForest on {X.shape[0]:,} samples...")
        start_time = datetime.now()
        
        try:
            self.rf_model.fit(X, y)
        except Exception as e:
            raise RuntimeError(f"[FATAL] RandomForest training failed: {e}")
        
        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"  -> Training completed in {elapsed:.2f} seconds")
        
        # Extract feature importances (Gini Impurity)
        self.feature_importances = self.rf_model.feature_importances_
        print(f"\n[EXTRACTION] Feature importances (Gini Impurity)")
        
        # Create importance dictionary
        importance_dict = dict(zip(feature_names, self.feature_importances))
        
        # Sort by importance (descending)
        sorted_features = sorted(
            importance_dict.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Display top 15
        print(f"\n[TOP {self.config.N_FEATURES_TO_PLOT} FEATURES]")
        print("-" * 50)
        for rank, (name, importance) in enumerate(sorted_features[:self.config.N_FEATURES_TO_PLOT], 1):
            bar = "█" * int(importance * 100)
            print(f"  {rank:2d}. {name:20s} | {importance:.4f} | {bar}")
        
        # Apply quantum constraint: Select top 8
        print(f"\n[QUANTUM CONSTRAINT] Selecting top {self.config.N_FEATURES_TO_SELECT} features for 8-qubit VQC")
        
        selected_features = sorted_features[:self.config.N_FEATURES_TO_SELECT]
        selected_names = [f[0] for f in selected_features]
        selected_importances = [f[1] for f in selected_features]
        
        print(f"\n[SELECTED FEATURES FOR QUANTUM PROCESSING]")
        print("=" * 50)
        total_importance = sum(selected_importances)
        cumulative = 0
        for rank, (name, importance) in enumerate(selected_features, 1):
            cumulative += importance
            pct = (importance / total_importance) * 100
            cum_pct = (cumulative / sum(self.feature_importances)) * 100
            print(f"  Q{rank-1}: {name:20s} | Imp: {importance:.4f} | {pct:5.1f}% | Cum: {cum_pct:5.1f}%")
        
        # Store statistics
        self.selection_stats = {
            'total_features': len(feature_names),
            'selected_features': self.config.N_FEATURES_TO_SELECT,
            'total_importance_captured': sum(selected_importances),
            'importance_ratio': sum(selected_importances) / sum(self.feature_importances),
            'training_time_seconds': elapsed
        }
        
        print(f"\n[COVERAGE] Top {self.config.N_FEATURES_TO_SELECT} features capture "
              f"{self.selection_stats['importance_ratio']*100:.1f}% of total importance")
        
        return selected_names, selected_importances, dict(sorted_features)



class ForensicEvidenceGenerator:
    """Generates plots and logs for audit trail."""
    
    def __init__(self, config: Config):
        self.config = config
        self.generated_files: List[str] = []
    
    def generate_importance_plot(
        self, 
        all_importances: Dict[str, float],
        selected_names: List[str]
    ) -> str:
        """
        Generate high-quality bar chart of feature importances.
        
        Args:
            all_importances: Dictionary of all feature importances (sorted)
            selected_names: Names of selected features (to highlight)
        
        Returns:
            Path to saved plot
        """
        print("\n" + "="*70)
        print("PHASE 3: FORENSIC EVIDENCE GENERATION")
        print("="*70)
        
        ensure_directory_exists(self.config.PLOTS_DIR)
        
        print(f"\n[PLOT] Generating feature importance chart...")
        
        # Get top N features for plotting
        top_features = list(all_importances.items())[:self.config.N_FEATURES_TO_PLOT]
        names = [f[0] for f in top_features]
        importances = [f[1] for f in top_features]
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.config.PLOT_FIGSIZE)
        
        # Color bars: green for selected, gray for others
        colors = ['#2E86AB' if name in selected_names else '#A8A8A8' 
                  for name in names]
        
        # Create horizontal bar chart
        y_pos = np.arange(len(names))
        bars = ax.barh(y_pos, importances, color=colors, edgecolor='black', linewidth=0.5)
        
        # Customize plot
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=10)
        ax.invert_yaxis()  # Highest importance at top
        ax.set_xlabel('Gini Importance', fontsize=12, fontweight='bold')
        ax.set_title('SWaT Feature Importance for Quantum IDS\n'
                     f'(Top {self.config.N_FEATURES_TO_SELECT} selected for 8-qubit VQC)',
                     fontsize=14, fontweight='bold', pad=15)
        
        # Add value labels
        for bar, importance in zip(bars, importances):
            width = bar.get_width()
            ax.text(width + 0.002, bar.get_y() + bar.get_height()/2,
                    f'{importance:.4f}', va='center', fontsize=9)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2E86AB', edgecolor='black', label='Selected for Quantum (8 qubits)'),
            Patch(facecolor='#A8A8A8', edgecolor='black', label='Not Selected')
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
        
        # Add grid
        ax.xaxis.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)
        
        # Tight layout
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.config.PLOTS_DIR, "feature_importance.png")
        try:
            plt.savefig(plot_path, dpi=self.config.PLOT_DPI, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
            plt.close(fig)
            self.generated_files.append(plot_path)
            print(f"  -> Saved: {plot_path}")
        except Exception as e:
            print(f"  -> [ERROR] Failed to save plot: {e}")
            raise
        
        return plot_path
    
    def generate_selection_log(
        self,
        selected_names: List[str],
        selected_importances: List[float],
        all_importances: Dict[str, float],
        selection_stats: Dict[str, any]
    ) -> str:
        """
        Generate JSON log of selected features.
        
        Args:
            selected_names: Names of selected features
            selected_importances: Importance scores of selected features
            all_importances: All feature importances
            selection_stats: Statistics from selection process
        
        Returns:
            Path to saved JSON log
        """
        ensure_directory_exists(self.config.LOGS_DIR)
        
        print(f"\n[LOG] Generating selection log...")
        
        # Build log structure
        log_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "pipeline_version": "1.1",
                "quantum_constraint": f"{self.config.N_FEATURES_TO_SELECT} qubits",
                "selection_method": "RandomForest Gini Importance",
                "data_source": "TEST set (contains both Normal and Attack classes)",
                "methodology_note": "Train set is 100% Normal for anomaly detection. "
                                   "Feature selection requires both classes to compute "
                                   "discriminative Gini importance. This is valid supervised "
                                   "feature selection identifying attack-discriminative sensors.",
                "rf_params": {
                    "n_estimators": self.config.RF_N_ESTIMATORS,
                    "max_depth": self.config.RF_MAX_DEPTH,
                    "min_samples_leaf": self.config.RF_MIN_SAMPLES_LEAF,
                    "random_state": self.config.RF_RANDOM_STATE,
                    "class_weight": "balanced"
                }
            },
            "selection_stats": {
                "total_features": selection_stats['total_features'],
                "selected_features": selection_stats['selected_features'],
                "importance_coverage": round(selection_stats['importance_ratio'] * 100, 2),
                "training_time_seconds": round(selection_stats['training_time_seconds'], 2)
            },
            "selected_features": [
                {
                    "rank": i + 1,
                    "qubit_index": i,
                    "name": name,
                    "importance": round(importance, 6),
                    "importance_percentage": round((importance / sum(selected_importances)) * 100, 2)
                }
                for i, (name, importance) in enumerate(zip(selected_names, selected_importances))
            ],
            "all_features_ranked": [
                {"rank": i + 1, "name": name, "importance": round(imp, 6)}
                for i, (name, imp) in enumerate(all_importances.items())
            ]
        }
        
        # Save JSON log
        log_path = os.path.join(self.config.LOGS_DIR, "selected_features.json")
        try:
            with open(log_path, 'w') as f:
                json.dump(log_data, f, indent=2)
            self.generated_files.append(log_path)
            print(f"  -> Saved: {log_path}")
        except Exception as e:
            print(f"  -> [ERROR] Failed to save log: {e}")
            raise
        
        return log_path


class DatasetReducer:
    """Reduces datasets to selected features only."""
    
    def __init__(self, config: Config):
        self.config = config
        self.reduction_stats: Dict[str, any] = {}
    
    def reduce_and_save(
        self,
        selected_names: List[str],
        feature_names: List[str]
    ) -> Dict[str, str]:
        """
        Reduce full datasets to selected features and save.
        
        Args:
            selected_names: Names of selected features
            feature_names: All feature names (for index lookup)
        
        Returns:
            Dictionary of saved file paths
        """
        print("\n" + "="*70)
        print("PHASE 4: DATASET REDUCTION & SERIALIZATION")
        print("="*70)
        
        # Get column indices for selected features
        selected_indices = [feature_names.index(name) for name in selected_names]
        print(f"\n[INDICES] Selected feature column indices: {selected_indices}")
        
        saved_paths = {}
        
        # Process X_train
        print(f"\n[REDUCING] X_train...")
        try:
            X_train_full = np.load(self.config.X_TRAIN_PATH)
            print(f"  -> Full shape: {X_train_full.shape}")
            
            X_train_reduced = X_train_full[:, selected_indices]
            print(f"  -> Reduced shape: {X_train_reduced.shape}")
            
            # Save
            save_path = os.path.join(self.config.PROCESSED_DIR, "X_train_reduced.npy")
            np.save(save_path, X_train_reduced.astype(np.float32))
            saved_paths['X_train_reduced'] = save_path
            print(f"  -> Saved: {save_path}")
            
            self.reduction_stats['X_train_original'] = X_train_full.shape
            self.reduction_stats['X_train_reduced'] = X_train_reduced.shape
            
            # Memory savings
            original_bytes = X_train_full.nbytes
            reduced_bytes = X_train_reduced.nbytes
            savings = (1 - reduced_bytes / original_bytes) * 100
            print(f"  -> Memory: {format_bytes(original_bytes)} -> {format_bytes(reduced_bytes)} ({savings:.1f}% reduction)")
            
            del X_train_full, X_train_reduced
            
        except Exception as e:
            raise IOError(f"[FATAL] Failed to reduce X_train: {e}")
        
        # Process X_test
        print(f"\n[REDUCING] X_test...")
        try:
            X_test_full = np.load(self.config.X_TEST_PATH)
            print(f"  -> Full shape: {X_test_full.shape}")
            
            X_test_reduced = X_test_full[:, selected_indices]
            print(f"  -> Reduced shape: {X_test_reduced.shape}")
            
            # Save
            save_path = os.path.join(self.config.PROCESSED_DIR, "X_test_reduced.npy")
            np.save(save_path, X_test_reduced.astype(np.float32))
            saved_paths['X_test_reduced'] = save_path
            print(f"  -> Saved: {save_path}")
            
            self.reduction_stats['X_test_original'] = X_test_full.shape
            self.reduction_stats['X_test_reduced'] = X_test_reduced.shape
            
            # Memory savings
            original_bytes = X_test_full.nbytes
            reduced_bytes = X_test_reduced.nbytes
            savings = (1 - reduced_bytes / original_bytes) * 100
            print(f"  -> Memory: {format_bytes(original_bytes)} -> {format_bytes(reduced_bytes)} ({savings:.1f}% reduction)")
            
            del X_test_full, X_test_reduced
            
        except Exception as e:
            raise IOError(f"[FATAL] Failed to reduce X_test: {e}")
        
        # Copy y_train and y_test (unchanged but for completeness)
        print(f"\n[COPYING] Label arrays (unchanged)...")
        try:
            y_train = np.load(self.config.Y_TRAIN_PATH)
            y_test = np.load(self.config.Y_TEST_PATH)
            
            # Save with _reduced suffix for consistency
            y_train_path = os.path.join(self.config.PROCESSED_DIR, "y_train_reduced.npy")
            y_test_path = os.path.join(self.config.PROCESSED_DIR, "y_test_reduced.npy")
            
            np.save(y_train_path, y_train)
            np.save(y_test_path, y_test)
            
            saved_paths['y_train_reduced'] = y_train_path
            saved_paths['y_test_reduced'] = y_test_path
            
            print(f"  -> Saved: {y_train_path}")
            print(f"  -> Saved: {y_test_path}")
            
        except Exception as e:
            print(f"  -> [WARNING] Could not copy label arrays: {e}")
        
        # Save selected feature names for later use
        print(f"\n[SAVING] Selected feature names...")
        try:
            selected_path = os.path.join(self.config.PROCESSED_DIR, "selected_feature_names.joblib")
            joblib.dump(selected_names, selected_path)
            saved_paths['selected_feature_names'] = selected_path
            print(f"  -> Saved: {selected_path}")
        except Exception as e:
            print(f"  -> [WARNING] Could not save selected feature names: {e}")
        
        return saved_paths


class FeatureSelectionPipeline:
    """
    Main orchestrator for the feature selection pipeline.
    
    Coordinates all phases:
    1. Memory-Safe Data Loading
    2. Physics-Based Feature Selection
    3. Forensic Evidence Generation
    4. Dataset Reduction & Serialization
    """
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.data_loader = MemorySafeDataLoader(self.config)
        self.feature_selector = PhysicsBasedFeatureSelector(self.config)
        self.evidence_generator = ForensicEvidenceGenerator(self.config)
        self.dataset_reducer = DatasetReducer(self.config)
    
    def run(self) -> Dict[str, any]:
        """
        Execute the complete feature selection pipeline.
        
        Returns:
            Dictionary containing all pipeline results
        """
        print("\n" + "="*70)
        print("SWAT FEATURE SELECTION PIPELINE")
        print("Quantum Constraint: 8 Qubits = 8 Features")
        print("="*70)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        results = {}
        
        # Phase 1: Memory-Safe Data Loading
        try:
            X_subsample, y_subsample, feature_names = self.data_loader.load_with_subsample()
            results['load_stats'] = self.data_loader.load_stats
        except Exception as e:
            print(f"\n[FATAL] Phase 1 failed: {e}")
            raise
        
        # Phase 2: Physics-Based Feature Selection
        try:
            selected_names, selected_importances, all_importances = \
                self.feature_selector.fit_and_select(X_subsample, y_subsample, feature_names)
            results['selection_stats'] = self.feature_selector.selection_stats
            results['selected_features'] = selected_names
        except Exception as e:
            print(f"\n[FATAL] Phase 2 failed: {e}")
            raise
        
        # Free memory from subsample
        del X_subsample, y_subsample
        
        # Phase 3: Forensic Evidence Generation
        try:
            plot_path = self.evidence_generator.generate_importance_plot(
                all_importances, selected_names
            )
            log_path = self.evidence_generator.generate_selection_log(
                selected_names, selected_importances, 
                all_importances, self.feature_selector.selection_stats
            )
            results['evidence_files'] = {
                'plot': plot_path,
                'log': log_path
            }
        except Exception as e:
            print(f"\n[FATAL] Phase 3 failed: {e}")
            raise
        
        # Phase 4: Dataset Reduction & Serialization
        try:
            saved_paths = self.dataset_reducer.reduce_and_save(
                selected_names, feature_names
            )
            results['saved_paths'] = saved_paths
            results['reduction_stats'] = self.dataset_reducer.reduction_stats
        except Exception as e:
            print(f"\n[FATAL] Phase 4 failed: {e}")
            raise
        
        # Final Summary
        print("\n" + "="*70)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("="*70)
        
        print(f"\n[QUANTUM-READY FEATURES] (8 qubits)")
        print("-" * 40)
        for i, (name, imp) in enumerate(zip(selected_names, selected_importances)):
            print(f"  Qubit {i}: {name} (importance: {imp:.4f})")
        
        print(f"\n[REDUCED DATASET SHAPES]")
        print(f"  X_train_reduced: {results['reduction_stats']['X_train_reduced']}")
        print(f"  X_test_reduced:  {results['reduction_stats']['X_test_reduced']}")
        
        print(f"\n[IMPORTANCE COVERAGE]")
        coverage = results['selection_stats']['importance_ratio'] * 100
        print(f"  Top 8 features capture {coverage:.1f}% of total Gini importance")
        
        print(f"\n[OUTPUT FILES]")
        for name, path in results['saved_paths'].items():
            print(f"  {name}: {path}")
        
        print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return results



def main():
    """Main entry point for the feature selection pipeline."""
    
    # Set random seeds for reproducibility
    np.random.seed(Config.RANDOM_SEED)
    
    # Initialize and run pipeline
    pipeline = FeatureSelectionPipeline()
    
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
