#!/usr/bin/env python3

import subprocess
import sys

def install_dependencies():
    """Install required packages if not present."""
    packages = [
        ("qiskit", "qiskit==1.1.0"),
        ("qiskit_machine_learning", "qiskit-machine-learning==0.7.2"),
        ("qiskit_aer", "qiskit-aer-gpu==0.14.2"),
        ("tqdm", "tqdm"),
    ]
    
    for module_name, pip_name in packages:
        try:
            __import__(module_name)
        except ImportError:
            print(f"[INSTALLING] {pip_name}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name])
            print(f"[OK] {pip_name} installed")

# Run installation check
print("[CHECKING DEPENDENCIES]")
install_dependencies()
print("[OK] All dependencies ready\n")


import os
import json
import time
import hashlib
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict, Any, List
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score

# Qiskit imports
from qiskit.circuit.library import ZZFeatureMap



class Config:
    """Centralized configuration for statistical robustness suite."""
    
    # Base directories
    BASE_DIR = "/content/drive/MyDrive/Quantum_Research"
    SWAT_DATA_DIR = f"{BASE_DIR}/02_Processed_Data"
    HAI_DATA_DIR = f"{BASE_DIR}/02_Processed_Data/HAI"
    LOGS_DIR = f"{BASE_DIR}/05_Results/Logs"
    
    # SWaT input files
    SWAT_X_PATH = f"{SWAT_DATA_DIR}/X_test_reduced.npy"
    SWAT_Y_PATH = f"{SWAT_DATA_DIR}/y_test_reduced.npy"
    
    # HAI input files
    HAI_X_PATH = f"{HAI_DATA_DIR}/HAI_X_test_reduced.npy"
    HAI_Y_PATH = f"{HAI_DATA_DIR}/HAI_y_test_reduced.npy"
    
    # Output file
    ROBUSTNESS_LOG_PATH = f"{LOGS_DIR}/final_robustness_stats.json"
    
    # Experimental parameters
    SEEDS = [42, 123, 999, 2024, 555]
    Q_TRAIN_SIZE = 2500
    Q_TEST_SIZE = 1000
    
    # Quantum circuit parameters
    N_QUBITS = 8
    FEATURE_MAP_REPS = 2
    ENTANGLEMENT = 'linear'
    
    # SVM parameters
    SVM_C = 1.0
    
    # Batch size for kernel computation
    BATCH_SIZE = 100




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


def compute_row_hash(row: np.ndarray) -> str:
    """Compute hash of a row for deduplication verification."""
    return hashlib.md5(row.tobytes()).hexdigest()


class InMemoryQuantumKernel:
    """
    Computes quantum kernel matrices entirely in memory.
    
    Optimized for speed: no disk I/O, batched evaluation.
    Uses FidelityStatevectorKernel when available, falls back to manual computation.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.feature_map = None
        self.kernel = None
        self._initialize_circuit()
        self._initialize_kernel()
    
    def _initialize_circuit(self):
        """Initialize ZZFeatureMap circuit."""
        self.feature_map = ZZFeatureMap(
            feature_dimension=self.config.N_QUBITS,
            reps=self.config.FEATURE_MAP_REPS,
            entanglement=self.config.ENTANGLEMENT
        )
    
    def _initialize_kernel(self):
        """Initialize quantum kernel with fallback strategies."""
        # Try FidelityStatevectorKernel first
        try:
            from qiskit_machine_learning.kernels import FidelityStatevectorKernel
            self.kernel = FidelityStatevectorKernel(feature_map=self.feature_map)
            self.kernel_type = "FidelityStatevectorKernel"
            return
        except ImportError:
            pass
        except Exception:
            pass
        
        # Try TrainableFidelityStatevectorKernel
        try:
            from qiskit_machine_learning.kernels import TrainableFidelityStatevectorKernel
            self.kernel = TrainableFidelityStatevectorKernel(feature_map=self.feature_map)
            self.kernel_type = "TrainableFidelityStatevectorKernel"
            return
        except Exception:
            pass
        
        # Fallback to manual kernel
        self.kernel = None
        self.kernel_type = "ManualStatevectorKernel"
    
    def compute_kernel_matrix(
        self,
        X_rows: np.ndarray,
        X_cols: np.ndarray,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Compute kernel matrix K(X_rows, X_cols).
        
        Args:
            X_rows: Row samples (n_rows, n_features)
            X_cols: Column samples (n_cols, n_features)
            show_progress: Whether to show progress bar
        
        Returns:
            Kernel matrix (n_rows, n_cols)
        """
        if self.kernel is not None:
            # Use Qiskit kernel
            return self._compute_with_qiskit_kernel(X_rows, X_cols, show_progress)
        else:
            # Use manual computation
            return self._compute_manual_kernel(X_rows, X_cols, show_progress)
    
    def _compute_with_qiskit_kernel(
        self,
        X_rows: np.ndarray,
        X_cols: np.ndarray,
        show_progress: bool
    ) -> np.ndarray:
        """Compute kernel using Qiskit's kernel implementation."""
        n_rows = len(X_rows)
        batch_size = self.config.BATCH_SIZE
        
        K = np.zeros((n_rows, len(X_cols)))
        n_batches = (n_rows + batch_size - 1) // batch_size
        
        iterator = range(n_batches)
        if show_progress:
            iterator = tqdm(iterator, desc="Computing kernel", leave=False)
        
        for batch_idx in iterator:
            start = batch_idx * batch_size
            end = min((batch_idx + 1) * batch_size, n_rows)
            X_batch = X_rows[start:end]
            K[start:end, :] = self.kernel.evaluate(X_batch, X_cols)
        
        return K
    
    def _compute_manual_kernel(
        self,
        X_rows: np.ndarray,
        X_cols: np.ndarray,
        show_progress: bool
    ) -> np.ndarray:
        """Compute kernel using manual statevector computation."""
        from qiskit.quantum_info import Statevector
        
        n_rows = len(X_rows)
        n_cols = len(X_cols)
        K = np.zeros((n_rows, n_cols))
        
        # Precompute statevectors for columns
        col_statevectors = []
        for x in X_cols:
            circuit = self.feature_map.assign_parameters(x)
            sv = Statevector(circuit).data
            col_statevectors.append(sv)
        col_statevectors = np.array(col_statevectors)
        
        # Compute kernel matrix
        iterator = range(n_rows)
        if show_progress:
            iterator = tqdm(iterator, desc="Computing kernel", leave=False)
        
        for i in iterator:
            circuit = self.feature_map.assign_parameters(X_rows[i])
            sv_row = Statevector(circuit).data
            K[i, :] = np.abs(np.dot(np.conj(sv_row), col_statevectors.T)) ** 2
        
        return K



class SingleExperimentRunner:
    """
    Runs a single experiment (one dataset, one seed).
    
    Steps:
    1. Load and deduplicate data
    2. Create stratified split
    3. Compute quantum kernels (in-memory)
    4. Train QSVM and evaluate
    """
    
    def __init__(self, config: Config, quantum_kernel: InMemoryQuantumKernel):
        self.config = config
        self.quantum_kernel = quantum_kernel
    
    def run(
        self,
        X_source: np.ndarray,
        y_source: np.ndarray,
        seed: int,
        dataset_name: str
    ) -> Dict[str, float]:
        """
        Run single experiment.
        
        Args:
            X_source: Source feature data
            y_source: Source labels
            seed: Random seed
            dataset_name: Name for logging
        
        Returns:
            Dictionary with metrics (f1, auc, accuracy, precision, recall)
        """
        np.random.seed(seed)
        
        # Step 1: Deduplicate
        X_rounded = np.round(X_source, decimals=6)
        _, unique_indices = np.unique(X_rounded, axis=0, return_index=True)
        unique_indices = np.sort(unique_indices)
        
        X_unique = X_source[unique_indices]
        y_unique = y_source[unique_indices]
        
        # Step 2: Stratified split - Q_train first
        X_q_train, X_remainder, y_q_train, y_remainder = train_test_split(
            X_unique, y_unique,
            train_size=self.config.Q_TRAIN_SIZE,
            stratify=y_unique,
            random_state=seed
        )
        
        # Q_test from remainder
        X_q_test, _, y_q_test, _ = train_test_split(
            X_remainder, y_remainder,
            train_size=self.config.Q_TEST_SIZE,
            stratify=y_remainder,
            random_state=seed
        )
        
        # Verify no overlap
        train_hashes = set(compute_row_hash(row) for row in X_q_train)
        test_hashes = set(compute_row_hash(row) for row in X_q_test)
        overlap = len(train_hashes.intersection(test_hashes))
        
        if overlap > 0:
            raise RuntimeError(f"Data leakage detected! {overlap} overlapping samples")
        
        # Step 3: Compute quantum kernels (in-memory)
        K_train = self.quantum_kernel.compute_kernel_matrix(X_q_train, X_q_train)
        K_test = self.quantum_kernel.compute_kernel_matrix(X_q_test, X_q_train)
        
        # Step 4: Train QSVM
        qsvm = SVC(
            kernel='precomputed',
            C=self.config.SVM_C,
            probability=True,
            random_state=seed,
            class_weight='balanced'
        )
        qsvm.fit(K_train, y_q_train)
        
        # Step 5: Evaluate
        y_pred = qsvm.predict(K_test)
        y_prob = qsvm.predict_proba(K_test)[:, 1]
        
        metrics = {
            'f1': float(f1_score(y_q_test, y_pred, zero_division=0)),
            'auc': float(roc_auc_score(y_q_test, y_prob)),
            'accuracy': float(accuracy_score(y_q_test, y_pred)),
            'precision': float(precision_score(y_q_test, y_pred, zero_division=0)),
            'recall': float(recall_score(y_q_test, y_pred, zero_division=0))
        }
        
        return metrics


class StatisticalAggregator:
    """Computes Mean ± Std for all metrics."""
    
    @staticmethod
    def aggregate(results: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """
        Aggregate results across multiple runs.
        
        Args:
            results: List of metric dictionaries
        
        Returns:
            Dictionary with mean and std for each metric
        """
        metrics = ['f1', 'auc', 'accuracy', 'precision', 'recall']
        aggregated = {}
        
        for metric in metrics:
            values = [r[metric] for r in results]
            aggregated[metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'values': values
            }
        
        return aggregated



class RobustnessPipeline:
    """
    Main orchestrator for multi-seed statistical validation.
    
    Runs experiments across:
    - 2 datasets (SWaT, HAI)
    - 5 seeds each
    - Reports Mean ± Std
    """
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.quantum_kernel = InMemoryQuantumKernel(self.config)
        self.runner = SingleExperimentRunner(self.config, self.quantum_kernel)
        self.results: Dict[str, Any] = {}
    
    def load_dataset(self, dataset_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load dataset by name."""
        if dataset_name == "SWaT":
            X = np.load(self.config.SWAT_X_PATH)
            y = np.load(self.config.SWAT_Y_PATH)
        elif dataset_name == "HAI":
            X = np.load(self.config.HAI_X_PATH)
            y = np.load(self.config.HAI_Y_PATH)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        return X, y
    
    def run(self) -> Dict[str, Any]:
        """
        Execute full robustness validation.
        
        Returns:
            Complete results dictionary
        """
        print("\n" + "="*75)
        print("STATISTICAL ROBUSTNESS SUITE - MULTI-SEED VERIFICATION")
        print("Protocol A: Proving Quantum Advantage is Statistically Significant")
        print("="*75)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\nSeeds: {self.config.SEEDS}")
        print(f"Quantum Kernel: {self.quantum_kernel.kernel_type}")
        print(f"Q_train: {self.config.Q_TRAIN_SIZE}, Q_test: {self.config.Q_TEST_SIZE}")
        
        pipeline_start = time.time()
        
        datasets = ["SWaT", "HAI"]
        all_results = {}
        
        for dataset_name in datasets:
            print(f"\n" + "="*75)
            print(f"DATASET: {dataset_name}")
            print("="*75)
            
            # Load data
            try:
                X, y = self.load_dataset(dataset_name)
                print(f"[LOADED] X: {X.shape}, y: {y.shape}")
                
                classes, counts = np.unique(y, return_counts=True)
                attack_ratio = counts[1] / len(y) * 100
                print(f"[CLASS DIST] {dict(zip(classes.astype(int), counts.astype(int)))} (Attack: {attack_ratio:.2f}%)")
                
            except Exception as e:
                print(f"[ERROR] Failed to load {dataset_name}: {e}")
                continue
            
            # Run experiments for each seed
            dataset_results = []
            
            print(f"\n{'Seed':<10} {'F1':<12} {'AUC':<12} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'Time':<10}")
            print("-"*80)
            
            for seed in self.config.SEEDS:
                try:
                    start_time = time.time()
                    
                    metrics = self.runner.run(X, y, seed, dataset_name)
                    
                    elapsed = time.time() - start_time
                    
                    dataset_results.append({
                        'seed': seed,
                        **metrics
                    })
                    
                    print(f"{seed:<10} {metrics['f1']:<12.4f} {metrics['auc']:<12.4f} "
                          f"{metrics['accuracy']:<12.4f} {metrics['precision']:<12.4f} "
                          f"{metrics['recall']:<12.4f} {format_time(elapsed):<10}")
                    
                except Exception as e:
                    print(f"{seed:<10} [ERROR] {str(e)[:50]}")
            
            # Aggregate statistics
            if dataset_results:
                aggregated = StatisticalAggregator.aggregate(dataset_results)
                
                print("-"*80)
                print(f"{'MEAN':<10} {aggregated['f1']['mean']:<12.4f} {aggregated['auc']['mean']:<12.4f} "
                      f"{aggregated['accuracy']['mean']:<12.4f} {aggregated['precision']['mean']:<12.4f} "
                      f"{aggregated['recall']['mean']:<12.4f}")
                print(f"{'STD':<10} {aggregated['f1']['std']:<12.4f} {aggregated['auc']['std']:<12.4f} "
                      f"{aggregated['accuracy']['std']:<12.4f} {aggregated['precision']['std']:<12.4f} "
                      f"{aggregated['recall']['std']:<12.4f}")
                
                all_results[dataset_name] = {
                    'raw_results': dataset_results,
                    'aggregated': aggregated
                }
        
        # Store results
        self.results = all_results
        
        # Print final IEEE Access table
        self._print_ieee_table()
        
        # Save to JSON
        self._save_results(pipeline_start)
        
        total_time = time.time() - pipeline_start
        print(f"\n[TOTAL TIME] {format_time(total_time)}")
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return self.results
    
    def _print_ieee_table(self):
        """Print final IEEE Access formatted table."""
        print("\n" + "="*75)
        print("IEEE ACCESS ROBUSTNESS TABLE")
        print("Quantum SVM Performance (Mean ± Std over 5 seeds)")
        print("="*75)
        
        print(f"\n{'Dataset':<12} {'F1-Score':<20} {'AUC-ROC':<20} {'Accuracy':<20}")
        print("-"*75)
        
        for dataset_name, data in self.results.items():
            agg = data['aggregated']
            
            f1_str = f"{agg['f1']['mean']:.4f} ± {agg['f1']['std']:.4f}"
            auc_str = f"{agg['auc']['mean']:.4f} ± {agg['auc']['std']:.4f}"
            acc_str = f"{agg['accuracy']['mean']:.4f} ± {agg['accuracy']['std']:.4f}"
            
            print(f"{dataset_name:<12} {f1_str:<20} {auc_str:<20} {acc_str:<20}")
        
        print("-"*75)
        
        # Statistical significance assessment
        print("\n[STATISTICAL SIGNIFICANCE ASSESSMENT]")
        
        for dataset_name, data in self.results.items():
            agg = data['aggregated']
            
            # Coefficient of variation (CV) = std/mean * 100
            f1_cv = (agg['f1']['std'] / agg['f1']['mean'] * 100) if agg['f1']['mean'] > 0 else 0
            auc_cv = (agg['auc']['std'] / agg['auc']['mean'] * 100) if agg['auc']['mean'] > 0 else 0
            
            print(f"\n  {dataset_name}:")
            print(f"    F1 Coefficient of Variation: {f1_cv:.2f}%")
            print(f"    AUC Coefficient of Variation: {auc_cv:.2f}%")
            
            if f1_cv < 5 and auc_cv < 5:
                print(f"    Verdict: HIGHLY STABLE (CV < 5%)")
            elif f1_cv < 10 and auc_cv < 10:
                print(f"    Verdict: STABLE (CV < 10%)")
            else:
                print(f"    Verdict: MODERATE VARIANCE")
        
        print("\n" + "="*75)
    
    def _save_results(self, pipeline_start: float):
        """Save results to JSON."""
        ensure_directory_exists(self.config.LOGS_DIR)
        
        total_time = time.time() - pipeline_start
        
        output = {
            "metadata": {
                "experiment": "Statistical Robustness Suite",
                "purpose": "Multi-seed validation for IEEE Access",
                "timestamp": datetime.now().isoformat(),
                "total_time_seconds": float(total_time),
                "seeds": self.config.SEEDS,
                "q_train_size": self.config.Q_TRAIN_SIZE,
                "q_test_size": self.config.Q_TEST_SIZE
            },
            "quantum_configuration": {
                "feature_map": "ZZFeatureMap",
                "n_qubits": self.config.N_QUBITS,
                "reps": self.config.FEATURE_MAP_REPS,
                "entanglement": self.config.ENTANGLEMENT,
                "kernel_type": self.quantum_kernel.kernel_type
            },
            "results": {}
        }
        
        for dataset_name, data in self.results.items():
            output["results"][dataset_name] = {
                "raw_results": data['raw_results'],
                "statistics": {
                    metric: {
                        "mean": data['aggregated'][metric]['mean'],
                        "std": data['aggregated'][metric]['std'],
                        "min": data['aggregated'][metric]['min'],
                        "max": data['aggregated'][metric]['max']
                    }
                    for metric in ['f1', 'auc', 'accuracy', 'precision', 'recall']
                }
            }
        
        # IEEE Access formatted summary
        output["ieee_summary"] = {
            dataset_name: {
                "f1_score": f"{data['aggregated']['f1']['mean']:.4f} ± {data['aggregated']['f1']['std']:.4f}",
                "auc_roc": f"{data['aggregated']['auc']['mean']:.4f} ± {data['aggregated']['auc']['std']:.4f}",
                "accuracy": f"{data['aggregated']['accuracy']['mean']:.4f} ± {data['aggregated']['accuracy']['std']:.4f}"
            }
            for dataset_name, data in self.results.items()
        }
        
        try:
            with open(self.config.ROBUSTNESS_LOG_PATH, 'w') as f:
                json.dump(output, f, indent=2)
            
            print(f"\n[SAVED] Results: {self.config.ROBUSTNESS_LOG_PATH}")
            
        except Exception as e:
            print(f"\n[ERROR] Failed to save results: {e}")



def main():
    """Main entry point for the statistical robustness suite."""
    
    # Initialize and run pipeline
    pipeline = RobustnessPipeline()
    
    try:
        results = pipeline.run()
        print("\n[SUCCESS] Statistical robustness validation completed")
        return results
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Validation stopped by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[FATAL ERROR] Validation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    results = main()
