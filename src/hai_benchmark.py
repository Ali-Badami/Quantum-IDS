#!/usr/bin/env python3
import os
import sys
import json
import time
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict, Any, List
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report
)

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns



class Config:
    """Centralized configuration for HAI benchmark pipeline."""
    
    # Base directories
    BASE_DIR = "/content/drive/MyDrive/Quantum_Research"
    HAI_DATA_DIR = f"{BASE_DIR}/02_Processed_Data/HAI"
    PLOTS_DIR = f"{BASE_DIR}/05_Results/Plots"
    LOGS_DIR = f"{BASE_DIR}/05_Results/Logs"
    
    # Input files - Quantum Kernel Matrices
    GRAM_TRAIN_PATH = f"{HAI_DATA_DIR}/gram_matrix_train.npy"
    GRAM_TEST_PATH = f"{HAI_DATA_DIR}/gram_matrix_test.npy"
    
    # Input files - Feature Subsets (for Classical SVM)
    X_Q_TRAIN_PATH = f"{HAI_DATA_DIR}/HAI_X_q_train.npy"
    Y_Q_TRAIN_PATH = f"{HAI_DATA_DIR}/HAI_y_q_train.npy"
    X_Q_TEST_PATH = f"{HAI_DATA_DIR}/HAI_X_q_test.npy"
    Y_Q_TEST_PATH = f"{HAI_DATA_DIR}/HAI_y_q_test.npy"
    
    # Output files
    ROC_PLOT_PATH = f"{PLOTS_DIR}/hai_roc_comparison.png"
    CONFUSION_MATRIX_PATH = f"{PLOTS_DIR}/hai_confusion_matrices.png"
    BENCHMARK_LOG_PATH = f"{LOGS_DIR}/hai_final_benchmark.json"
    
    # SVM parameters
    SVM_C = 1.0
    SVM_GAMMA = 'scale'
    
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


def to_native_type(obj):
    """
    Convert numpy types to Python native types for JSON serialization.
    
    Handles: np.int64, np.float64, np.ndarray, np.bool_, etc.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: to_native_type(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_native_type(item) for item in obj]
    else:
        return obj

class DataLoader:
    """Loads HAI quantum kernels and feature subsets."""
    
    def __init__(self, config: Config):
        self.config = config
        self.data_stats: Dict[str, Any] = {}
    
    def load_all(self) -> Tuple[np.ndarray, ...]:
        """
        Load all required data files.
        
        Returns:
            Tuple of (K_train, K_test, X_train, y_train, X_test, y_test)
        """
        print("\n" + "="*70)
        print("PHASE 1: DATA LOADING")
        print("="*70)
        
        # Load quantum kernel matrices
        print(f"\n[LOADING] Quantum kernel matrices...")
        try:
            K_train = np.load(self.config.GRAM_TRAIN_PATH)
            K_test = np.load(self.config.GRAM_TEST_PATH)
            print(f"  -> K_train (Gram matrix): {K_train.shape}")
            print(f"  -> K_test: {K_test.shape}")
        except Exception as e:
            raise IOError(f"[FATAL] Failed to load kernel matrices: {e}")
        
        # Load feature subsets
        print(f"\n[LOADING] Feature subsets...")
        try:
            X_train = np.load(self.config.X_Q_TRAIN_PATH)
            y_train = np.load(self.config.Y_Q_TRAIN_PATH)
            X_test = np.load(self.config.X_Q_TEST_PATH)
            y_test = np.load(self.config.Y_Q_TEST_PATH)
            print(f"  -> X_train: {X_train.shape}")
            print(f"  -> y_train: {y_train.shape}")
            print(f"  -> X_test: {X_test.shape}")
            print(f"  -> y_test: {y_test.shape}")
        except Exception as e:
            raise IOError(f"[FATAL] Failed to load feature subsets: {e}")
        
        # Verification
        print(f"\n[VERIFICATION] Checking data alignment...")
        
        assert K_train.shape[0] == len(y_train), \
            f"K_train rows ({K_train.shape[0]}) != y_train length ({len(y_train)})"
        assert K_train.shape[1] == len(y_train), \
            f"K_train cols ({K_train.shape[1]}) != y_train length ({len(y_train)})"
        assert K_test.shape[0] == len(y_test), \
            f"K_test rows ({K_test.shape[0]}) != y_test length ({len(y_test)})"
        assert K_test.shape[1] == len(y_train), \
            f"K_test cols ({K_test.shape[1]}) != y_train length ({len(y_train)})"
        assert X_train.shape[0] == len(y_train), \
            f"X_train rows ({X_train.shape[0]}) != y_train length ({len(y_train)})"
        assert X_test.shape[0] == len(y_test), \
            f"X_test rows ({X_test.shape[0]}) != y_test length ({len(y_test)})"
        
        print(f"  -> [OK] All dimensions verified and aligned")
        
        # Class distribution
        train_classes, train_counts = np.unique(y_train, return_counts=True)
        test_classes, test_counts = np.unique(y_test, return_counts=True)
        
        train_dist = {int(k): int(v) for k, v in zip(train_classes, train_counts)}
        test_dist = {int(k): int(v) for k, v in zip(test_classes, test_counts)}
        
        print(f"\n[CLASS DISTRIBUTION]")
        print(f"  -> Train: {train_dist} (Attack: {train_counts[1]/len(y_train)*100:.2f}%)")
        print(f"  -> Test: {test_dist} (Attack: {test_counts[1]/len(y_test)*100:.2f}%)")
        
        # Kernel statistics
        print(f"\n[KERNEL STATISTICS]")
        print(f"  -> K_train mean: {K_train.mean():.4f}")
        print(f"  -> K_test mean: {K_test.mean():.4f}")
        
        # Store statistics
        self.data_stats = {
            'n_train': int(len(y_train)),
            'n_test': int(len(y_test)),
            'n_features': int(X_train.shape[1]),
            'train_distribution': train_dist,
            'test_distribution': test_dist,
            'train_attack_ratio': float(train_counts[1] / len(y_train)),
            'test_attack_ratio': float(test_counts[1] / len(y_test)),
            'k_train_mean': float(K_train.mean()),
            'k_test_mean': float(K_test.mean())
        }
        
        return K_train, K_test, X_train, y_train, X_test, y_test

class ModelTrainer:
    """Trains and evaluates QSVM and Classical SVM models."""
    
    def __init__(self, config: Config):
        self.config = config
        self.qsvm_model: SVC = None
        self.classical_model: SVC = None
        self.training_stats: Dict[str, Any] = {}
    
    def train_qsvm(
        self,
        K_train: np.ndarray,
        y_train: np.ndarray
    ) -> SVC:
        """
        Train Quantum SVM with precomputed kernel.
        
        Args:
            K_train: Precomputed training Gram matrix
            y_train: Training labels
        
        Returns:
            Trained SVC model
        """
        print("\n" + "="*70)
        print("PHASE 2A: QUANTUM SVM TRAINING")
        print("="*70)
        
        print(f"\n[QSVM] Initializing SVC with precomputed kernel...")
        print(f"  -> Kernel: precomputed (ZZFeatureMap quantum)")
        print(f"  -> C: {self.config.SVM_C}")
        print(f"  -> probability: True")
        
        self.qsvm_model = SVC(
            kernel='precomputed',
            C=self.config.SVM_C,
            probability=True,
            random_state=self.config.RANDOM_SEED,
            class_weight='balanced'  # Handle class imbalance
        )
        
        print(f"\n[QSVM] Training on Gram matrix {K_train.shape}...")
        start_time = time.time()
        
        self.qsvm_model.fit(K_train, y_train)
        
        train_time = time.time() - start_time
        print(f"  -> Training completed in {format_time(train_time)}")
        print(f"  -> Support vectors: {self.qsvm_model.n_support_}")
        print(f"  -> Total SVs: {sum(self.qsvm_model.n_support_)}")
        
        self.training_stats['qsvm'] = {
            'train_time': float(train_time),
            'n_support_vectors': int(sum(self.qsvm_model.n_support_)),
            'support_per_class': [int(x) for x in self.qsvm_model.n_support_]
        }
        
        return self.qsvm_model
    
    def train_classical_svm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> SVC:
        """
        Train Classical RBF SVM as baseline.
        
        Args:
            X_train: Training features
            y_train: Training labels
        
        Returns:
            Trained SVC model
        """
        print("\n" + "="*70)
        print("PHASE 2B: CLASSICAL RBF SVM TRAINING (BASELINE)")
        print("="*70)
        
        print(f"\n[CLASSICAL] Initializing SVC with RBF kernel...")
        print(f"  -> Kernel: rbf (Gaussian)")
        print(f"  -> C: {self.config.SVM_C}")
        print(f"  -> gamma: {self.config.SVM_GAMMA}")
        print(f"  -> probability: True")
        
        self.classical_model = SVC(
            kernel='rbf',
            C=self.config.SVM_C,
            gamma=self.config.SVM_GAMMA,
            probability=True,
            random_state=self.config.RANDOM_SEED,
            class_weight='balanced'  # Handle class imbalance
        )
        
        print(f"\n[CLASSICAL] Training on raw features {X_train.shape}...")
        start_time = time.time()
        
        self.classical_model.fit(X_train, y_train)
        
        train_time = time.time() - start_time
        print(f"  -> Training completed in {format_time(train_time)}")
        print(f"  -> Support vectors: {self.classical_model.n_support_}")
        print(f"  -> Total SVs: {sum(self.classical_model.n_support_)}")
        
        self.training_stats['classical'] = {
            'train_time': float(train_time),
            'n_support_vectors': int(sum(self.classical_model.n_support_)),
            'support_per_class': [int(x) for x in self.classical_model.n_support_]
        }
        
        return self.classical_model

class MetricsCalculator:
    """Calculates and compares performance metrics."""
    
    def __init__(self, config: Config):
        self.config = config
        self.qsvm_metrics: Dict[str, Any] = {}
        self.classical_metrics: Dict[str, Any] = {}
        self.qsvm_predictions: Dict[str, np.ndarray] = {}
        self.classical_predictions: Dict[str, np.ndarray] = {}
    
    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        model_name: str
    ) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities (positive class)
            model_name: Name of the model
        
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, zero_division=0)),
            'f1_score': float(f1_score(y_true, y_pred, zero_division=0)),
            'roc_auc': float(roc_auc_score(y_true, y_prob)),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
        
        return metrics
    
    def evaluate_models(
        self,
        qsvm_model: SVC,
        classical_model: SVC,
        K_test: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Tuple[Dict, Dict]:
        """
        Evaluate both models and compare metrics.
        
        Args:
            qsvm_model: Trained QSVM model
            classical_model: Trained Classical SVM model
            K_test: Test kernel matrix (for QSVM)
            X_test: Test features (for Classical)
            y_test: Test labels
        
        Returns:
            Tuple of (qsvm_metrics, classical_metrics)
        """
        print("\n" + "="*70)
        print("PHASE 3: MODEL EVALUATION")
        print("="*70)
        
        # QSVM Predictions
        print(f"\n[QSVM] Generating predictions...")
        qsvm_pred = qsvm_model.predict(K_test)
        qsvm_prob = qsvm_model.predict_proba(K_test)[:, 1]
        
        self.qsvm_predictions = {
            'y_pred': qsvm_pred,
            'y_prob': qsvm_prob
        }
        
        self.qsvm_metrics = self.calculate_metrics(
            y_test, qsvm_pred, qsvm_prob, "QSVM"
        )
        
        # Classical SVM Predictions
        print(f"[CLASSICAL] Generating predictions...")
        classical_pred = classical_model.predict(X_test)
        classical_prob = classical_model.predict_proba(X_test)[:, 1]
        
        self.classical_predictions = {
            'y_pred': classical_pred,
            'y_prob': classical_prob
        }
        
        self.classical_metrics = self.calculate_metrics(
            y_test, classical_pred, classical_prob, "Classical"
        )
        
        # Print comparison table
        print(f"\n" + "="*70)
        print("PERFORMANCE COMPARISON: HAI DATASET")
        print("="*70)
        print(f"\n{'Metric':<20} {'QSVM (Quantum)':<20} {'Classical (RBF)':<20} {'Delta':<15}")
        print("-"*75)
        
        metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        qsvm_wins = 0
        
        for metric in metrics_to_compare:
            q_val = self.qsvm_metrics[metric]
            c_val = self.classical_metrics[metric]
            delta = q_val - c_val
            
            if delta > 0:
                delta_str = f"+{delta:.4f}"
                winner = "QSVM"
                qsvm_wins += 1
            elif delta < 0:
                delta_str = f"{delta:.4f}"
                winner = "Classical"
            else:
                delta_str = "0.0000"
                winner = "Tie"
            
            print(f"{metric:<20} {q_val:<20.4f} {c_val:<20.4f} {delta_str:<15} [{winner}]")
        
        print("-"*75)
        
        # Final verdict
        print(f"\n" + "="*70)
        print("FINAL VERDICT")
        print("="*70)
        
        f1_delta = self.qsvm_metrics['f1_score'] - self.classical_metrics['f1_score']
        auc_delta = self.qsvm_metrics['roc_auc'] - self.classical_metrics['roc_auc']
        
        if f1_delta > 0 and auc_delta > 0:
            verdict = "QUANTUM ADVANTAGE CONFIRMED"
            verdict_detail = f"QSVM outperforms on F1 (+{f1_delta:.4f}) and AUC (+{auc_delta:.4f})"
        elif f1_delta > 0:
            verdict = "QUANTUM ADVANTAGE (F1)"
            verdict_detail = f"QSVM wins on F1 (+{f1_delta:.4f}), Classical wins on AUC ({auc_delta:.4f})"
        elif auc_delta > 0:
            verdict = "QUANTUM ADVANTAGE (AUC)"
            verdict_detail = f"QSVM wins on AUC (+{auc_delta:.4f}), Classical wins on F1 ({f1_delta:.4f})"
        elif f1_delta == 0 and auc_delta == 0:
            verdict = "TIE"
            verdict_detail = "Both models perform identically"
        else:
            verdict = "CLASSICAL ADVANTAGE"
            verdict_detail = f"Classical outperforms on F1 ({f1_delta:.4f}) and AUC ({auc_delta:.4f})"
        
        print(f"\n  {verdict}")
        print(f"  {verdict_detail}")
        print(f"  QSVM won {qsvm_wins}/5 metrics")
        
        return self.qsvm_metrics, self.classical_metrics

class Visualizer:
    """Generates publication-quality visualizations."""
    
    def __init__(self, config: Config):
        self.config = config
    
    def plot_roc_curves(
        self,
        y_test: np.ndarray,
        qsvm_prob: np.ndarray,
        classical_prob: np.ndarray,
        qsvm_auc: float,
        classical_auc: float,
        save_path: str
    ) -> str:
        """
        Plot ROC curves for both models on a single graph.
        
        Args:
            y_test: True labels
            qsvm_prob: QSVM prediction probabilities
            classical_prob: Classical SVM prediction probabilities
            qsvm_auc: QSVM AUC-ROC score
            classical_auc: Classical SVM AUC-ROC score
            save_path: Path to save the figure
        
        Returns:
            Path to saved figure
        """
        print("\n" + "="*70)
        print("PHASE 4: VISUALIZATION")
        print("="*70)
        
        print(f"\n[PLOTTING] Generating ROC curve comparison...")
        
        ensure_directory_exists(os.path.dirname(save_path))
        
        # Compute ROC curves
        fpr_qsvm, tpr_qsvm, _ = roc_curve(y_test, qsvm_prob)
        fpr_classical, tpr_classical, _ = roc_curve(y_test, classical_prob)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot QSVM ROC (Blue)
        ax.plot(
            fpr_qsvm, tpr_qsvm,
            color='#2E86AB',
            linewidth=3,
            label=f'QSVM (ZZFeatureMap) - AUC = {qsvm_auc:.4f}'
        )
        
        # Plot Classical ROC (Red)
        ax.plot(
            fpr_classical, tpr_classical,
            color='#E94F37',
            linewidth=3,
            linestyle='--',
            label=f'Classical SVM (RBF) - AUC = {classical_auc:.4f}'
        )
        
        # Random classifier baseline
        ax.plot(
            [0, 1], [0, 1],
            color='gray',
            linewidth=2,
            linestyle=':',
            label='Random Classifier (AUC = 0.5000)'
        )
        
        # Formatting
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=14)
        ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=14)
        ax.set_title(
            'ROC Curve Comparison: Quantum vs Classical SVM\n'
            'HAI 22.04 Dataset - Cross-Dataset Validation',
            fontsize=16, fontweight='bold'
        )
        ax.legend(loc='lower right', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add AUC difference annotation
        auc_diff = qsvm_auc - classical_auc
        if auc_diff >= 0:
            annotation = f'QSVM advantage: +{auc_diff:.4f} AUC'
            color = '#2E86AB'
        else:
            annotation = f'Classical advantage: +{-auc_diff:.4f} AUC'
            color = '#E94F37'
        
        ax.annotate(
            annotation,
            xy=(0.6, 0.2),
            fontsize=12,
            fontweight='bold',
            color=color,
            bbox=dict(boxstyle='round', facecolor='white', edgecolor=color, alpha=0.8)
        )
        
        plt.tight_layout()
        
        # Save
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        print(f"  -> Saved: {save_path}")
        return save_path
    
    def plot_confusion_matrices(
        self,
        qsvm_cm: np.ndarray,
        classical_cm: np.ndarray,
        save_path: str
    ) -> str:
        """
        Generate side-by-side confusion matrices.
        
        Args:
            qsvm_cm: QSVM confusion matrix
            classical_cm: Classical SVM confusion matrix
            save_path: Path to save the figure
        
        Returns:
            Path to saved figure
        """
        print(f"\n[PLOTTING] Generating confusion matrices...")
        
        ensure_directory_exists(os.path.dirname(save_path))
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Class labels
        class_names = ['Normal (0)', 'Attack (1)']
        
        # QSVM Confusion Matrix
        sns.heatmap(
            qsvm_cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            ax=axes[0],
            annot_kws={'size': 14, 'weight': 'bold'}
        )
        axes[0].set_title('Quantum SVM (ZZFeatureMap Kernel)', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Predicted Label', fontsize=12)
        axes[0].set_ylabel('True Label', fontsize=12)
        
        # Classical SVM Confusion Matrix
        sns.heatmap(
            classical_cm,
            annot=True,
            fmt='d',
            cmap='Oranges',
            xticklabels=class_names,
            yticklabels=class_names,
            ax=axes[1],
            annot_kws={'size': 14, 'weight': 'bold'}
        )
        axes[1].set_title('Classical SVM (RBF Kernel)', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Predicted Label', fontsize=12)
        axes[1].set_ylabel('True Label', fontsize=12)
        
        plt.suptitle(
            'Confusion Matrix Comparison: QSVM vs Classical SVM\n'
            'HAI 22.04 Dataset - Cross-Dataset Validation',
            fontsize=16, fontweight='bold', y=1.02
        )
        plt.tight_layout()
        
        # Save
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        print(f"  -> Saved: {save_path}")
        return save_path

class ForensicLogger:
    """Exports comprehensive benchmark log for IEEE Access."""
    
    def __init__(self, config: Config):
        self.config = config
    
    def export_benchmark_log(
        self,
        data_stats: Dict,
        training_stats: Dict,
        qsvm_metrics: Dict,
        classical_metrics: Dict,
        total_time: float
    ) -> str:
        """
        Export final benchmark summary to JSON.
        
        Args:
            data_stats: Data loading statistics
            training_stats: Training time and SV counts
            qsvm_metrics: QSVM performance metrics
            classical_metrics: Classical SVM performance metrics
            total_time: Total pipeline execution time
        
        Returns:
            Path to saved JSON file
        """
        print("\n" + "="*70)
        print("PHASE 5: FORENSIC LOGGING")
        print("="*70)
        
        ensure_directory_exists(self.config.LOGS_DIR)
        
        # Calculate deltas
        f1_delta = qsvm_metrics['f1_score'] - classical_metrics['f1_score']
        auc_delta = qsvm_metrics['roc_auc'] - classical_metrics['roc_auc']
        
        # Determine verdict
        if f1_delta > 0 and auc_delta > 0:
            verdict = "Quantum Advantage Confirmed"
        elif f1_delta > 0 or auc_delta > 0:
            verdict = "Partial Quantum Advantage"
        elif f1_delta == 0 and auc_delta == 0:
            verdict = "Tie"
        else:
            verdict = "Classical Advantage"
        
        # Build comprehensive log (all values converted to native types)
        benchmark_log = {
            "metadata": {
                "experiment_name": "HAI_QSVM_vs_Classical_Benchmark",
                "dataset": "HAI 22.04 (Hardware-in-the-Loop)",
                "physical_system": "Thermal/Pumped Storage",
                "task": "Intrusion Detection (Binary Classification)",
                "purpose": "Cross-Dataset Validation for IEEE Access",
                "timestamp": datetime.now().isoformat(),
                "total_execution_time_seconds": float(total_time),
                "random_seed": self.config.RANDOM_SEED
            },
            "quantum_configuration": {
                "feature_map": "ZZFeatureMap",
                "n_qubits": 8,
                "repetitions": 2,
                "entanglement": "linear",
                "kernel_type": "FidelityStatevectorKernel"
            },
            "data_summary": to_native_type(data_stats),
            "training_summary": {
                "qsvm": to_native_type(training_stats.get('qsvm', {})),
                "classical_svm": to_native_type(training_stats.get('classical', {}))
            },
            "performance_metrics": {
                "qsvm": to_native_type(qsvm_metrics),
                "classical_svm": to_native_type(classical_metrics)
            },
            "comparison": {
                "accuracy_delta": float(qsvm_metrics['accuracy'] - classical_metrics['accuracy']),
                "precision_delta": float(qsvm_metrics['precision'] - classical_metrics['precision']),
                "recall_delta": float(qsvm_metrics['recall'] - classical_metrics['recall']),
                "f1_delta": float(f1_delta),
                "auc_delta": float(auc_delta),
                "verdict": verdict,
                "qsvm_metrics_won": sum([
                    qsvm_metrics['accuracy'] > classical_metrics['accuracy'],
                    qsvm_metrics['precision'] > classical_metrics['precision'],
                    qsvm_metrics['recall'] > classical_metrics['recall'],
                    qsvm_metrics['f1_score'] > classical_metrics['f1_score'],
                    qsvm_metrics['roc_auc'] > classical_metrics['roc_auc']
                ])
            },
            "output_files": {
                "roc_curve_plot": self.config.ROC_PLOT_PATH,
                "confusion_matrix_plot": self.config.CONFUSION_MATRIX_PATH,
                "benchmark_log": self.config.BENCHMARK_LOG_PATH
            }
        }
        
        # Save to JSON
        try:
            with open(self.config.BENCHMARK_LOG_PATH, 'w') as f:
                json.dump(benchmark_log, f, indent=2)
            
            print(f"\n[SAVED] Benchmark log: {self.config.BENCHMARK_LOG_PATH}")
            
        except Exception as e:
            raise IOError(f"[FATAL] Failed to save benchmark log: {e}")
        
        return self.config.BENCHMARK_LOG_PATH

class HAIBenchmarkPipeline:
    """
    Main orchestrator for HAI QSVM vs Classical SVM benchmark.
    
    Coordinates all phases:
    1. Data Loading
    2. Model Training (QSVM + Classical)
    3. Evaluation and Metrics
    4. Visualization
    5. Forensic Logging
    """
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.data_loader = DataLoader(self.config)
        self.trainer = ModelTrainer(self.config)
        self.metrics_calculator = MetricsCalculator(self.config)
        self.visualizer = Visualizer(self.config)
        self.logger = ForensicLogger(self.config)
    
    def run(self) -> Dict[str, Any]:
        """
        Execute the complete benchmark pipeline.
        
        Returns:
            Dictionary containing all benchmark results
        """
        print("\n" + "="*70)
        print("HAI QSVM vs CLASSICAL SVM BENCHMARK")
        print("Cross-Dataset Validation for IEEE Access")
        print("="*70)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        results = {}
        pipeline_start = time.time()
        
        # Phase 1: Load Data
        try:
            K_train, K_test, X_train, y_train, X_test, y_test = self.data_loader.load_all()
            results['data_stats'] = self.data_loader.data_stats
        except Exception as e:
            print(f"\n[FATAL] Phase 1 failed: {e}")
            raise
        
        # Phase 2A: Train QSVM
        try:
            qsvm_model = self.trainer.train_qsvm(K_train, y_train)
        except Exception as e:
            print(f"\n[FATAL] Phase 2A failed: {e}")
            raise
        
        # Phase 2B: Train Classical SVM
        try:
            classical_model = self.trainer.train_classical_svm(X_train, y_train)
            results['training_stats'] = self.trainer.training_stats
        except Exception as e:
            print(f"\n[FATAL] Phase 2B failed: {e}")
            raise
        
        # Phase 3: Evaluate Models
        try:
            qsvm_metrics, classical_metrics = self.metrics_calculator.evaluate_models(
                qsvm_model, classical_model, K_test, X_test, y_test
            )
            results['qsvm_metrics'] = qsvm_metrics
            results['classical_metrics'] = classical_metrics
        except Exception as e:
            print(f"\n[FATAL] Phase 3 failed: {e}")
            raise
        
        # Phase 4: Generate Visualizations
        try:
            # ROC Curves
            self.visualizer.plot_roc_curves(
                y_test,
                self.metrics_calculator.qsvm_predictions['y_prob'],
                self.metrics_calculator.classical_predictions['y_prob'],
                qsvm_metrics['roc_auc'],
                classical_metrics['roc_auc'],
                self.config.ROC_PLOT_PATH
            )
            
            # Confusion Matrices
            self.visualizer.plot_confusion_matrices(
                np.array(qsvm_metrics['confusion_matrix']),
                np.array(classical_metrics['confusion_matrix']),
                self.config.CONFUSION_MATRIX_PATH
            )
            
        except Exception as e:
            print(f"\n[WARNING] Phase 4 failed: {e}")
        
        # Phase 5: Export Forensic Log
        try:
            pipeline_elapsed = time.time() - pipeline_start
            
            self.logger.export_benchmark_log(
                self.data_loader.data_stats,
                self.trainer.training_stats,
                qsvm_metrics,
                classical_metrics,
                pipeline_elapsed
            )
            
        except Exception as e:
            print(f"\n[WARNING] Phase 5 failed: {e}")
        
        # Final Summary
        pipeline_elapsed = time.time() - pipeline_start
        
        print("\n" + "="*70)
        print("BENCHMARK COMPLETED SUCCESSFULLY")
        print("="*70)
        
        print(f"\n[FINAL RESULTS - HAI VALIDATION]")
        print(f"  -> QSVM F1-Score: {qsvm_metrics['f1_score']:.4f}")
        print(f"  -> Classical F1-Score: {classical_metrics['f1_score']:.4f}")
        print(f"  -> QSVM AUC-ROC: {qsvm_metrics['roc_auc']:.4f}")
        print(f"  -> Classical AUC-ROC: {classical_metrics['roc_auc']:.4f}")
        print(f"  -> Total time: {format_time(pipeline_elapsed)}")
        
        print(f"\n[OUTPUT FILES]")
        print(f"  -> ROC curves: {self.config.ROC_PLOT_PATH}")
        print(f"  -> Confusion matrices: {self.config.CONFUSION_MATRIX_PATH}")
        print(f"  -> Benchmark log: {self.config.BENCHMARK_LOG_PATH}")
        
        print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        results['total_time'] = pipeline_elapsed
        
        return results

def main():
    """Main entry point for the HAI benchmark pipeline."""
    
    # Set random seeds for reproducibility
    np.random.seed(Config.RANDOM_SEED)
    
    # Initialize and run pipeline
    pipeline = HAIBenchmarkPipeline()
    
    try:
        results = pipeline.run()
        print("\n[SUCCESS] HAI benchmark completed without errors")
        return results
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Benchmark stopped by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[FATAL ERROR] Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    results = main()
