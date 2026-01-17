#!/usr/bin/env python3

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from pathlib import Path
from typing import Tuple, List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns


class Config:
    """Centralized configuration for HAI feature selection pipeline."""
    
    # Base directories
    BASE_DIR = "/content/drive/MyDrive/Quantum_Research"
    HAI_DATA_DIR = f"{BASE_DIR}/02_Processed_Data/HAI"
    PLOTS_DIR = f"{BASE_DIR}/05_Results/Plots"
    LOGS_DIR = f"{BASE_DIR}/05_Results/Logs"
    
    # Input files
    X_TRAIN_PATH = f"{HAI_DATA_DIR}/HAI_X_train.npy"
    Y_TRAIN_PATH = f"{HAI_DATA_DIR}/HAI_y_train.npy"
    X_TEST_PATH = f"{HAI_DATA_DIR}/HAI_X_test.npy"
    Y_TEST_PATH = f"{HAI_DATA_DIR}/HAI_y_test.npy"
    FEATURE_NAMES_PATH = f"{HAI_DATA_DIR}/hai_feature_names.joblib"
    
    # Output files
    X_TRAIN_REDUCED_PATH = f"{HAI_DATA_DIR}/HAI_X_train_reduced.npy"
    Y_TRAIN_REDUCED_PATH = f"{HAI_DATA_DIR}/HAI_y_train_reduced.npy"
    X_TEST_REDUCED_PATH = f"{HAI_DATA_DIR}/HAI_X_test_reduced.npy"
    Y_TEST_REDUCED_PATH = f"{HAI_DATA_DIR}/HAI_y_test_reduced.npy"
    SELECTED_FEATURES_PATH = f"{HAI_DATA_DIR}/hai_selected_feature_names.joblib"
    FEATURE_IMPORTANCE_PLOT = f"{PLOTS_DIR}/hai_feature_importance.png"
    SELECTED_FEATURES_JSON = f"{LOGS_DIR}/hai_selected_features.json"
    
    # Feature selection parameters
    N_QUBITS = 8                    # Quantum circuit constraint
    N_ESTIMATORS = 100              # RandomForest trees
    MAX_DEPTH = 20                  # Tree depth limit
    MIN_SAMPLES_LEAF = 50           # Minimum samples per leaf
    SUBSAMPLE_SIZE = 50000          # Stratified subsample for speed
    TOP_N_VISUALIZE = 20            # Features to show in plot
    
    # Random seed
    RANDOM_SEED = 42


def ensure_directory_exists(dir_path: str) -> bool:
    """Create directory if it doesn't exist."""
    try:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        raise OSError(f"[I/O ERROR] Cannot create directory: {dir_path} - {e}")


def format_time(seconds: float) -> str:
    """Format seconds to human readable string."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        return f"{seconds/60:.2f}m"
    else:
        return f"{seconds/3600:.2f}h"


class DataLoader:
    """Loads HAI processed data for feature selection."""
    
    def __init__(self, config: Config):
        self.config = config
        self.feature_names: List[str] = []
        self.data_stats: Dict[str, Any] = {}
    
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Load HAI processed data.
        
        Returns:
            Tuple of (X_train, y_train, X_test, y_test, feature_names)
        """
        print("\n" + "="*70)
        print("PHASE 1: DATA LOADING")
        print("="*70)
        
        # Load test data (has both classes)
        print(f"\n[LOADING] HAI test data (contains attacks)...")
        try:
            X_test = np.load(self.config.X_TEST_PATH)
            y_test = np.load(self.config.Y_TEST_PATH)
            print(f"  -> X_test shape: {X_test.shape}")
            print(f"  -> y_test shape: {y_test.shape}")
        except Exception as e:
            raise IOError(f"[FATAL] Failed to load test data: {e}")
        
        # Load train data (for later slicing only)
        print(f"\n[LOADING] HAI train data (for slicing only)...")
        try:
            X_train = np.load(self.config.X_TRAIN_PATH)
            y_train = np.load(self.config.Y_TRAIN_PATH)
            print(f"  -> X_train shape: {X_train.shape}")
            print(f"  -> y_train shape: {y_train.shape}")
        except Exception as e:
            raise IOError(f"[FATAL] Failed to load train data: {e}")
        
        # Load feature names
        print(f"\n[LOADING] Feature names...")
        try:
            self.feature_names = joblib.load(self.config.FEATURE_NAMES_PATH)
            print(f"  -> Features loaded: {len(self.feature_names)}")
        except Exception as e:
            raise IOError(f"[FATAL] Failed to load feature names: {e}")
        
        # Verify dimensions
        assert X_test.shape[1] == len(self.feature_names), "Feature count mismatch!"
        assert X_train.shape[1] == len(self.feature_names), "Train feature count mismatch!"
        
        # Class distribution
        print(f"\n[CLASS DISTRIBUTION]")
        train_classes, train_counts = np.unique(y_train, return_counts=True)
        test_classes, test_counts = np.unique(y_test, return_counts=True)
        
        print(f"  -> Train: {dict(zip(train_classes.astype(int), train_counts.astype(int)))}")
        print(f"  -> Test: {dict(zip(test_classes.astype(int), test_counts.astype(int)))}")
        print(f"  -> Test Attack Ratio: {test_counts[1]/len(y_test)*100:.2f}%")
        
        # Store statistics
        self.data_stats = {
            'n_train': len(X_train),
            'n_test': len(X_test),
            'n_features': len(self.feature_names),
            'test_attack_count': int(test_counts[1]),
            'test_attack_ratio': float(test_counts[1] / len(y_test))
        }
        
        return X_train, y_train, X_test, y_test, self.feature_names



class FeatureSelector:
    """
    Selects top features using RandomForest importance analysis.
    
    METHODOLOGY NOTE:
    - Uses TEST set for feature selection (train has 0% attacks)
    - This identifies which sensors discriminate attacks from normal
    - NOT the same as training a classifier (no data leakage concern)
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.rf_model: RandomForestClassifier = None
        self.feature_importances: np.ndarray = None
        self.selected_indices: List[int] = []
        self.selected_names: List[str] = []
        self.selection_stats: Dict[str, Any] = {}
    
    def select_features(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: List[str]
    ) -> Tuple[List[int], List[str], np.ndarray]:
        """
        Select top N features using RandomForest importance.
        
        Args:
            X_test: Test features (contains both classes)
            y_test: Test labels
            feature_names: List of feature names
        
        Returns:
            Tuple of (selected_indices, selected_names, all_importances)
        """
        print("\n" + "="*70)
        print("PHASE 2: RANDOM FOREST FEATURE IMPORTANCE ANALYSIS")
        print("="*70)
        
        # Create stratified subsample for memory efficiency
        print(f"\n[SUBSAMPLE] Creating stratified sample for analysis...")
        print(f"  -> Original test size: {len(X_test):,}")
        print(f"  -> Target subsample: {self.config.SUBSAMPLE_SIZE:,}")
        
        np.random.seed(self.config.RANDOM_SEED)
        
        # Stratified sampling
        X_sample, _, y_sample, _ = train_test_split(
            X_test, y_test,
            train_size=min(self.config.SUBSAMPLE_SIZE, len(X_test)),
            stratify=y_test,
            random_state=self.config.RANDOM_SEED
        )
        
        sample_classes, sample_counts = np.unique(y_sample, return_counts=True)
        print(f"  -> Subsample size: {len(X_sample):,}")
        print(f"  -> Subsample distribution: {dict(zip(sample_classes.astype(int), sample_counts.astype(int)))}")
        
        # Train RandomForest
        print(f"\n[TRAINING] RandomForest for feature importance...")
        print(f"  -> n_estimators: {self.config.N_ESTIMATORS}")
        print(f"  -> max_depth: {self.config.MAX_DEPTH}")
        print(f"  -> min_samples_leaf: {self.config.MIN_SAMPLES_LEAF}")
        print(f"  -> class_weight: balanced")
        
        self.rf_model = RandomForestClassifier(
            n_estimators=self.config.N_ESTIMATORS,
            max_depth=self.config.MAX_DEPTH,
            min_samples_leaf=self.config.MIN_SAMPLES_LEAF,
            class_weight='balanced',
            random_state=self.config.RANDOM_SEED,
            n_jobs=-1
        )
        
        start_time = time.time()
        self.rf_model.fit(X_sample, y_sample)
        train_time = time.time() - start_time
        
        print(f"  -> Training completed in {format_time(train_time)}")
        
        # Extract feature importances
        self.feature_importances = self.rf_model.feature_importances_
        
        print(f"\n[IMPORTANCE] Analyzing feature importances...")
        print(f"  -> Total features: {len(self.feature_importances)}")
        print(f"  -> Non-zero importances: {np.sum(self.feature_importances > 0)}")
        print(f"  -> Max importance: {self.feature_importances.max():.4f}")
        print(f"  -> Mean importance: {self.feature_importances.mean():.4f}")
        
        # Rank features by importance
        importance_ranking = np.argsort(self.feature_importances)[::-1]
        
        # Select top N features
        self.selected_indices = importance_ranking[:self.config.N_QUBITS].tolist()
        self.selected_names = [feature_names[i] for i in self.selected_indices]
        
        # Calculate coverage
        top_n_importance = self.feature_importances[self.selected_indices].sum()
        total_importance = self.feature_importances.sum()
        coverage = top_n_importance / total_importance if total_importance > 0 else 0
        
        print(f"\n" + "="*70)
        print(f"TOP {self.config.N_QUBITS} SELECTED FEATURES (FOR {self.config.N_QUBITS} QUBITS)")
        print("="*70)
        print(f"\n{'Qubit':<8} {'Sensor ID':<20} {'Importance':<15} {'Cumulative %':<15}")
        print("-"*60)
        
        cumulative = 0
        for i, idx in enumerate(self.selected_indices):
            importance = self.feature_importances[idx]
            cumulative += importance
            cumulative_pct = (cumulative / total_importance * 100) if total_importance > 0 else 0
            print(f"Q{i:<7} {feature_names[idx]:<20} {importance:<15.4f} {cumulative_pct:<15.1f}%")
        
        print("-"*60)
        print(f"\n[COVERAGE] Top {self.config.N_QUBITS} features capture {coverage*100:.1f}% of total importance")
        
        # Store statistics
        self.selection_stats = {
            'n_features_analyzed': len(feature_names),
            'n_features_selected': self.config.N_QUBITS,
            'subsample_size': len(X_sample),
            'training_time': train_time,
            'coverage': float(coverage),
            'max_importance': float(self.feature_importances.max()),
            'mean_importance': float(self.feature_importances.mean())
        }
        
        return self.selected_indices, self.selected_names, self.feature_importances


class DataReducer:
    """Reduces datasets to selected features."""
    
    def __init__(self, config: Config):
        self.config = config
        self.reduction_stats: Dict[str, Any] = {}
    
    def reduce_datasets(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        selected_indices: List[int]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Reduce datasets to selected feature columns.
        
        Args:
            X_train: Full training features
            y_train: Training labels
            X_test: Full test features
            y_test: Test labels
            selected_indices: Indices of selected features
        
        Returns:
            Tuple of (X_train_reduced, y_train, X_test_reduced, y_test)
        """
        print("\n" + "="*70)
        print("PHASE 3: DATASET REDUCTION")
        print("="*70)
        
        print(f"\n[REDUCING] Slicing arrays to {len(selected_indices)} features...")
        print(f"  -> Selected indices: {selected_indices}")
        
        # Slice to selected columns
        X_train_reduced = X_train[:, selected_indices]
        X_test_reduced = X_test[:, selected_indices]
        
        # Memory savings
        original_size = (X_train.nbytes + X_test.nbytes) / 1024 / 1024
        reduced_size = (X_train_reduced.nbytes + X_test_reduced.nbytes) / 1024 / 1024
        savings = (1 - reduced_size / original_size) * 100
        
        print(f"\n[SHAPES]")
        print(f"  -> X_train: {X_train.shape} -> {X_train_reduced.shape}")
        print(f"  -> X_test: {X_test.shape} -> {X_test_reduced.shape}")
        
        print(f"\n[MEMORY]")
        print(f"  -> Original: {original_size:.2f} MB")
        print(f"  -> Reduced: {reduced_size:.2f} MB")
        print(f"  -> Savings: {savings:.1f}%")
        
        # Store statistics
        self.reduction_stats = {
            'original_features': X_train.shape[1],
            'reduced_features': len(selected_indices),
            'original_size_mb': float(original_size),
            'reduced_size_mb': float(reduced_size),
            'memory_savings_pct': float(savings)
        }
        
        return X_train_reduced, y_train, X_test_reduced, y_test


class Visualizer:
    """Generates publication-quality feature importance visualization."""
    
    def __init__(self, config: Config):
        self.config = config
    
    def plot_feature_importance(
        self,
        feature_importances: np.ndarray,
        feature_names: List[str],
        selected_indices: List[int],
        save_path: str
    ) -> str:
        """
        Generate horizontal bar chart of top features.
        
        Args:
            feature_importances: Array of importance scores
            feature_names: List of feature names
            selected_indices: Indices of selected features
            save_path: Path to save the figure
        
        Returns:
            Path to saved figure
        """
        print("\n" + "="*70)
        print("PHASE 4: VISUALIZATION")
        print("="*70)
        
        ensure_directory_exists(os.path.dirname(save_path))
        
        print(f"\n[PLOTTING] Generating feature importance chart...")
        
        # Get top N features for visualization
        n_show = min(self.config.TOP_N_VISUALIZE, len(feature_names))
        top_indices = np.argsort(feature_importances)[::-1][:n_show]
        
        # Prepare data
        top_names = [feature_names[i] for i in top_indices]
        top_importances = feature_importances[top_indices]
        
        # Create color map (selected features highlighted)
        colors = ['#2E86AB' if i in selected_indices else '#A9A9A9' 
                  for i in top_indices]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Horizontal bar chart
        y_pos = np.arange(len(top_names))
        bars = ax.barh(y_pos, top_importances, color=colors, edgecolor='black', linewidth=0.5)
        
        # Customize
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_names, fontsize=10)
        ax.invert_yaxis()  # Highest importance at top
        ax.set_xlabel('Gini Importance', fontsize=12)
        ax.set_title(
            f'HAI Dataset: Top {n_show} Feature Importances\n'
            f'(Blue = Selected for {self.config.N_QUBITS}-Qubit Quantum Circuit)',
            fontsize=14, fontweight='bold'
        )
        
        # Add importance values on bars
        for i, (bar, importance) in enumerate(zip(bars, top_importances)):
            ax.text(
                bar.get_width() + 0.002,
                bar.get_y() + bar.get_height()/2,
                f'{importance:.4f}',
                va='center', ha='left', fontsize=9
            )
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2E86AB', edgecolor='black', label=f'Selected ({self.config.N_QUBITS} qubits)'),
            Patch(facecolor='#A9A9A9', edgecolor='black', label='Not selected')
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
        
        # Grid
        ax.xaxis.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        
        # Save
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        print(f"  -> Saved: {save_path}")
        return save_path


class ResultsExporter:
    """Exports feature selection results."""
    
    def __init__(self, config: Config):
        self.config = config
        self.saved_paths: Dict[str, str] = {}
    
    def save_all(
        self,
        X_train_reduced: np.ndarray,
        y_train: np.ndarray,
        X_test_reduced: np.ndarray,
        y_test: np.ndarray,
        selected_indices: List[int],
        selected_names: List[str],
        feature_importances: np.ndarray,
        feature_names: List[str],
        selection_stats: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Save all results to disk.
        
        Returns:
            Dictionary of saved file paths
        """
        print("\n" + "="*70)
        print("PHASE 5: SERIALIZATION")
        print("="*70)
        
        ensure_directory_exists(self.config.HAI_DATA_DIR)
        ensure_directory_exists(self.config.LOGS_DIR)
        
        # Save reduced arrays
        print(f"\n[SAVING] Reduced arrays...")
        
        try:
            np.save(self.config.X_TRAIN_REDUCED_PATH, X_train_reduced)
            print(f"  -> X_train_reduced: {self.config.X_TRAIN_REDUCED_PATH}")
            print(f"     Shape: {X_train_reduced.shape}, Size: {X_train_reduced.nbytes/1024/1024:.2f} MB")
            
            np.save(self.config.Y_TRAIN_REDUCED_PATH, y_train)
            print(f"  -> y_train_reduced: {self.config.Y_TRAIN_REDUCED_PATH}")
            
            np.save(self.config.X_TEST_REDUCED_PATH, X_test_reduced)
            print(f"  -> X_test_reduced: {self.config.X_TEST_REDUCED_PATH}")
            print(f"     Shape: {X_test_reduced.shape}, Size: {X_test_reduced.nbytes/1024/1024:.2f} MB")
            
            np.save(self.config.Y_TEST_REDUCED_PATH, y_test)
            print(f"  -> y_test_reduced: {self.config.Y_TEST_REDUCED_PATH}")
            
            # Save selected feature names
            joblib.dump(selected_names, self.config.SELECTED_FEATURES_PATH)
            print(f"  -> selected_feature_names: {self.config.SELECTED_FEATURES_PATH}")
            
        except Exception as e:
            raise IOError(f"[FATAL] Failed to save arrays: {e}")
        
        # Save detailed JSON log
        print(f"\n[SAVING] Feature selection log...")
        
        selection_log = {
            "metadata": {
                "dataset": "HAI 22.04 (Hardware-in-the-Loop)",
                "task": "Feature Selection for Quantum Kernel",
                "timestamp": datetime.now().isoformat(),
                "n_qubits": self.config.N_QUBITS
            },
            "methodology": {
                "algorithm": "RandomForestClassifier",
                "n_estimators": self.config.N_ESTIMATORS,
                "max_depth": self.config.MAX_DEPTH,
                "data_source": "Test set (contains both Normal and Attack)",
                "rationale": "Train set has 0% attacks, cannot compute Gini importance"
            },
            "selected_features": [
                {
                    "qubit": f"Q{i}",
                    "index": int(idx),
                    "name": selected_names[i],
                    "importance": float(feature_importances[idx]),
                    "importance_pct": float(feature_importances[idx] / feature_importances.sum() * 100)
                }
                for i, idx in enumerate(selected_indices)
            ],
            "statistics": {
                "total_features": len(feature_names),
                "selected_features": len(selected_indices),
                "coverage_pct": float(sum(feature_importances[selected_indices]) / feature_importances.sum() * 100),
                "subsample_size": selection_stats.get('subsample_size', 0),
                "training_time_seconds": selection_stats.get('training_time', 0)
            },
            "output_files": {
                "X_train_reduced": self.config.X_TRAIN_REDUCED_PATH,
                "X_test_reduced": self.config.X_TEST_REDUCED_PATH,
                "selected_features": self.config.SELECTED_FEATURES_PATH,
                "importance_plot": self.config.FEATURE_IMPORTANCE_PLOT
            }
        }
        
        try:
            with open(self.config.SELECTED_FEATURES_JSON, 'w') as f:
                json.dump(selection_log, f, indent=2)
            print(f"  -> selection_log: {self.config.SELECTED_FEATURES_JSON}")
        except Exception as e:
            print(f"  -> [WARNING] Failed to save JSON log: {e}")
        
        self.saved_paths = {
            'X_train_reduced': self.config.X_TRAIN_REDUCED_PATH,
            'y_train_reduced': self.config.Y_TRAIN_REDUCED_PATH,
            'X_test_reduced': self.config.X_TEST_REDUCED_PATH,
            'y_test_reduced': self.config.Y_TEST_REDUCED_PATH,
            'selected_features': self.config.SELECTED_FEATURES_PATH,
            'selection_log': self.config.SELECTED_FEATURES_JSON
        }
        
        return self.saved_paths


class HAIFeatureSelectionPipeline:
    """
    Main orchestrator for HAI feature selection.
    
    Coordinates all phases:
    1. Data Loading
    2. RandomForest Feature Importance Analysis
    3. Dataset Reduction
    4. Visualization
    5. Serialization
    """
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.data_loader = DataLoader(self.config)
        self.feature_selector = FeatureSelector(self.config)
        self.data_reducer = DataReducer(self.config)
        self.visualizer = Visualizer(self.config)
        self.exporter = ResultsExporter(self.config)
    
    def run(self) -> Dict[str, Any]:
        """
        Execute the complete feature selection pipeline.
        
        Returns:
            Dictionary containing all pipeline results
        """
        print("\n" + "="*70)
        print("HAI FEATURE SELECTION PIPELINE")
        print("Selecting Top 8 Sensors for 8-Qubit Quantum Circuit")
        print("="*70)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        results = {}
        pipeline_start = time.time()
        
        # Phase 1: Load Data
        try:
            X_train, y_train, X_test, y_test, feature_names = self.data_loader.load_data()
            results['data_stats'] = self.data_loader.data_stats
        except Exception as e:
            print(f"\n[FATAL] Phase 1 failed: {e}")
            raise
        
        # Phase 2: Feature Selection
        try:
            selected_indices, selected_names, importances = self.feature_selector.select_features(
                X_test, y_test, feature_names
            )
            results['selection_stats'] = self.feature_selector.selection_stats
        except Exception as e:
            print(f"\n[FATAL] Phase 2 failed: {e}")
            raise
        
        # Phase 3: Dataset Reduction
        try:
            X_train_reduced, y_train, X_test_reduced, y_test = self.data_reducer.reduce_datasets(
                X_train, y_train, X_test, y_test, selected_indices
            )
            results['reduction_stats'] = self.data_reducer.reduction_stats
        except Exception as e:
            print(f"\n[FATAL] Phase 3 failed: {e}")
            raise
        
        # Phase 4: Visualization
        try:
            self.visualizer.plot_feature_importance(
                importances, feature_names, selected_indices,
                self.config.FEATURE_IMPORTANCE_PLOT
            )
        except Exception as e:
            print(f"\n[WARNING] Phase 4 failed: {e}")
        
        # Phase 5: Serialization
        try:
            saved_paths = self.exporter.save_all(
                X_train_reduced, y_train,
                X_test_reduced, y_test,
                selected_indices, selected_names,
                importances, feature_names,
                self.feature_selector.selection_stats
            )
            results['saved_paths'] = saved_paths
        except Exception as e:
            print(f"\n[FATAL] Phase 5 failed: {e}")
            raise
        
        # Final Summary
        pipeline_elapsed = time.time() - pipeline_start
        
        print("\n" + "="*70)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("="*70)
        
        print(f"\n[SELECTED FEATURES FOR QUANTUM CIRCUIT]")
        for i, (idx, name) in enumerate(zip(selected_indices, selected_names)):
            imp = importances[idx]
            print(f"  Q{i}: {name} (importance: {imp:.4f})")
        
        coverage = sum(importances[selected_indices]) / importances.sum() * 100
        print(f"\n[COVERAGE] Top 8 features capture {coverage:.1f}% of total importance")
        
        print(f"\n[OUTPUT FILES]")
        print(f"  -> X_train_reduced: {X_train_reduced.shape}")
        print(f"  -> X_test_reduced: {X_test_reduced.shape}")
        print(f"  -> Plot: {self.config.FEATURE_IMPORTANCE_PLOT}")
        print(f"  -> Log: {self.config.SELECTED_FEATURES_JSON}")
        
        print(f"\n[TOTAL TIME] {format_time(pipeline_elapsed)}")
        print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        results['total_time'] = pipeline_elapsed
        
        return results


def main():
    """Main entry point for the HAI feature selection pipeline."""
    
    # Set random seed for reproducibility
    np.random.seed(Config.RANDOM_SEED)
    
    # Initialize and run pipeline
    pipeline = HAIFeatureSelectionPipeline()
    
    try:
        results = pipeline.run()
        print("\n[SUCCESS] Feature selection completed without errors")
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
