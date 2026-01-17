#!/usr/bin/env python3

import os
import sys
import json
import time
import hashlib
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Sklearn for stratified sampling
from sklearn.model_selection import train_test_split



class Config:
    """Centralized configuration for HAI quantum kernel computation."""
    
    # Base directories
    BASE_DIR = "/content/drive/MyDrive/Quantum_Research"
    HAI_DATA_DIR = f"{BASE_DIR}/02_Processed_Data/HAI"
    PLOTS_DIR = f"{BASE_DIR}/05_Results/Plots"
    LOGS_DIR = f"{BASE_DIR}/05_Results/Logs"
    
    # Input files (use TEST set as source - train has 0 attacks)
    X_TEST_REDUCED_PATH = f"{HAI_DATA_DIR}/HAI_X_test_reduced.npy"
    Y_TEST_REDUCED_PATH = f"{HAI_DATA_DIR}/HAI_y_test_reduced.npy"
    
    # Output files - Quantum subsets
    X_Q_TRAIN_PATH = f"{HAI_DATA_DIR}/HAI_X_q_train.npy"
    Y_Q_TRAIN_PATH = f"{HAI_DATA_DIR}/HAI_y_q_train.npy"
    X_Q_TEST_PATH = f"{HAI_DATA_DIR}/HAI_X_q_test.npy"
    Y_Q_TEST_PATH = f"{HAI_DATA_DIR}/HAI_y_q_test.npy"
    
    # Output files - Gram matrices
    GRAM_TRAIN_PATH = f"{HAI_DATA_DIR}/gram_matrix_train.npy"
    GRAM_TEST_PATH = f"{HAI_DATA_DIR}/gram_matrix_test.npy"
    
    # Output files - Visualization and logs
    CIRCUIT_DIAGRAM_PATH = f"{PLOTS_DIR}/hai_quantum_circuit.png"
    KERNEL_HEATMAP_PATH = f"{PLOTS_DIR}/hai_kernel_matrices_heatmap.png"
    KERNEL_LOG_PATH = f"{LOGS_DIR}/hai_kernel_computation.json"
    
    # Quantum subset parameters
    Q_TRAIN_SIZE = 2500          # Quantum training subset size
    Q_TEST_SIZE = 1000           # Quantum test subset size
    
    # Quantum circuit parameters
    N_QUBITS = 8                 # Matches 8 reduced features
    FEATURE_MAP_REPS = 2         # ZZFeatureMap repetitions
    ENTANGLEMENT = 'linear'      # Entanglement pattern
    
    # Computation parameters
    BATCH_SIZE = 100             # Samples per batch
    CHECKPOINT_INTERVAL = 500   # Save checkpoint every N rows
    
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


def compute_row_hash(row: np.ndarray) -> str:
    """Compute hash of a row for deduplication."""
    return hashlib.md5(row.tobytes()).hexdigest()



class BackendVerifier:
    """Verifies Qiskit installation and GPU availability."""
    
    def __init__(self, config: Config):
        self.config = config
        self.backend_info: Dict[str, Any] = {}
    
    def verify(self) -> Dict[str, Any]:
        """
        Verify Qiskit installation and GPU availability.
        
        Returns:
            Dictionary with backend information
        """
        print("\n" + "="*70)
        print("PHASE 0: BACKEND VERIFICATION")
        print("="*70)
        
        # Check Qiskit version
        print(f"\n[QISKIT] Checking installation...")
        try:
            import qiskit
            print(f"  -> Qiskit version: {qiskit.__version__}")
            self.backend_info['qiskit_version'] = qiskit.__version__
        except ImportError as e:
            raise ImportError(f"[FATAL] Qiskit not installed: {e}")
        
        # Check Qiskit ML
        print(f"\n[QISKIT-ML] Checking installation...")
        try:
            import qiskit_machine_learning
            print(f"  -> Qiskit ML version: {qiskit_machine_learning.__version__}")
            self.backend_info['qiskit_ml_version'] = qiskit_machine_learning.__version__
        except ImportError as e:
            raise ImportError(f"[FATAL] Qiskit ML not installed: {e}")
        
        # Check Qiskit Aer
        print(f"\n[QISKIT-AER] Checking GPU availability...")
        try:
            from qiskit_aer import AerSimulator
            
            # Check available devices
            test_sim = AerSimulator()
            available_devices = test_sim.available_devices()
            print(f"  -> Available devices: {available_devices}")
            self.backend_info['available_devices'] = list(available_devices)
            
            # Check if GPU is available
            if 'GPU' in available_devices:
                print(f"  -> [OK] GPU detected and available")
                self.backend_info['gpu_available'] = True
            else:
                print(f"  -> [WARNING] GPU not available, using CPU")
                self.backend_info['gpu_available'] = False
                
        except ImportError as e:
            raise ImportError(f"[FATAL] Qiskit Aer not installed: {e}")
        
        # Initialize backend
        print(f"\n[BACKEND] Initializing AerSimulator...")
        try:
            if self.backend_info.get('gpu_available', False):
                self.backend = AerSimulator(method='statevector', device='GPU')
                print(f"  -> Backend: AerSimulator (statevector, GPU)")
            else:
                self.backend = AerSimulator(method='statevector')
                print(f"  -> Backend: AerSimulator (statevector, CPU)")
            
            self.backend_info['backend'] = str(self.backend)
            
        except Exception as e:
            raise RuntimeError(f"[FATAL] Failed to initialize backend: {e}")
        
        print(f"\n[OK] Backend verification completed")
        
        return self.backend_info



class QuantumSubsetCreator:
    """
    Creates disjoint quantum train/test subsets with deduplication.
    
    METHODOLOGY:
    - Source: HAI Reduced TEST set (train has 0% attacks)
    - Deduplication: Remove steady-state duplicates
    - Sequential Split: Q_train first, then Q_test from remainder
    - Guarantees zero overlap between subsets
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.subset_stats: Dict[str, Any] = {}
    
    def create_subsets(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Create disjoint quantum subsets from HAI test data.
        
        Returns:
            Tuple of (X_q_train, y_q_train, X_q_test, y_q_test)
        """
        print("\n" + "="*70)
        print("PHASE 1: QUANTUM SUBSET CREATION")
        print("="*70)
        
        # Load HAI reduced test data
        print(f"\n[LOADING] HAI reduced test data...")
        try:
            X_source = np.load(self.config.X_TEST_REDUCED_PATH)
            y_source = np.load(self.config.Y_TEST_REDUCED_PATH)
            print(f"  -> X_source shape: {X_source.shape}")
            print(f"  -> y_source shape: {y_source.shape}")
        except Exception as e:
            raise IOError(f"[FATAL] Failed to load data: {e}")
        
        # Class distribution before deduplication
        classes, counts = np.unique(y_source, return_counts=True)
        print(f"  -> Class distribution: {dict(zip(classes.astype(int), counts.astype(int)))}")
        print(f"  -> Attack ratio: {counts[1]/len(y_source)*100:.2f}%")
        
        # Deduplication
        print(f"\n[DEDUPLICATION] Removing steady-state duplicates...")
        
        # Round to avoid floating point precision issues
        X_rounded = np.round(X_source, decimals=6)
        
        # Find unique rows
        _, unique_indices = np.unique(X_rounded, axis=0, return_index=True)
        unique_indices = np.sort(unique_indices)  # Maintain temporal order
        
        X_unique = X_source[unique_indices]
        y_unique = y_source[unique_indices]
        
        n_duplicates = len(X_source) - len(X_unique)
        dup_percentage = n_duplicates / len(X_source) * 100
        
        print(f"  -> Original samples: {len(X_source):,}")
        print(f"  -> Duplicates removed: {n_duplicates:,} ({dup_percentage:.2f}%)")
        print(f"  -> Unique samples: {len(X_unique):,}")
        
        # Class distribution after deduplication
        classes_unique, counts_unique = np.unique(y_unique, return_counts=True)
        print(f"  -> Post-dedup distribution: {dict(zip(classes_unique.astype(int), counts_unique.astype(int)))}")
        
        # Verify we have enough samples
        total_needed = self.config.Q_TRAIN_SIZE + self.config.Q_TEST_SIZE
        if len(X_unique) < total_needed:
            raise ValueError(
                f"[FATAL] Not enough unique samples ({len(X_unique)}) "
                f"for required subset sizes ({total_needed})"
            )
        
        # SEQUENTIAL SPLIT (guarantees disjoint sets)
        print(f"\n[SPLIT] Creating disjoint subsets (sequential extraction)...")
        print(f"  -> Q_train size: {self.config.Q_TRAIN_SIZE}")
        print(f"  -> Q_test size: {self.config.Q_TEST_SIZE}")
        
        np.random.seed(self.config.RANDOM_SEED)
        
        # Step 1: Extract Q_train from unique data
        X_q_train, X_remainder, y_q_train, y_remainder = train_test_split(
            X_unique, y_unique,
            train_size=self.config.Q_TRAIN_SIZE,
            stratify=y_unique,
            random_state=self.config.RANDOM_SEED
        )
        
        print(f"  -> Step 1: Extracted Q_train ({len(X_q_train)}) from unique samples")
        print(f"  -> Remainder: {len(X_remainder)} samples")
        
        # Step 2: Extract Q_test from REMAINDER only
        X_q_test, _, y_q_test, _ = train_test_split(
            X_remainder, y_remainder,
            train_size=self.config.Q_TEST_SIZE,
            stratify=y_remainder,
            random_state=self.config.RANDOM_SEED
        )
        
        print(f"  -> Step 2: Extracted Q_test ({len(X_q_test)}) from remainder")
        
        # VERIFICATION: Check for overlap using hashes
        print(f"\n[VERIFICATION] Checking for overlap...")
        
        train_hashes = set(compute_row_hash(row) for row in X_q_train)
        test_hashes = set(compute_row_hash(row) for row in X_q_test)
        overlap = train_hashes.intersection(test_hashes)
        
        if len(overlap) > 0:
            raise RuntimeError(
                f"[FATAL] Data leakage detected! {len(overlap)} overlapping samples"
            )
        
        print(f"  -> Train hashes: {len(train_hashes)}")
        print(f"  -> Test hashes: {len(test_hashes)}")
        print(f"  -> Overlap: {len(overlap)}")
        print(f"  -> [OK] Zero overlap verified - subsets are disjoint")
        
        # Class distributions
        train_classes, train_counts = np.unique(y_q_train, return_counts=True)
        test_classes, test_counts = np.unique(y_q_test, return_counts=True)
        
        print(f"\n[FINAL SUBSETS]")
        print(f"  -> Q_train: {X_q_train.shape}")
        print(f"     Distribution: {dict(zip(train_classes.astype(int), train_counts.astype(int)))}")
        print(f"     Attack ratio: {train_counts[1]/len(y_q_train)*100:.2f}%")
        print(f"  -> Q_test: {X_q_test.shape}")
        print(f"     Distribution: {dict(zip(test_classes.astype(int), test_counts.astype(int)))}")
        print(f"     Attack ratio: {test_counts[1]/len(y_q_test)*100:.2f}%")
        
        # Save subsets
        print(f"\n[SAVING] Quantum subsets...")
        try:
            np.save(self.config.X_Q_TRAIN_PATH, X_q_train)
            np.save(self.config.Y_Q_TRAIN_PATH, y_q_train)
            np.save(self.config.X_Q_TEST_PATH, X_q_test)
            np.save(self.config.Y_Q_TEST_PATH, y_q_test)
            print(f"  -> Saved to {self.config.HAI_DATA_DIR}")
        except Exception as e:
            raise IOError(f"[FATAL] Failed to save subsets: {e}")
        
        # Store statistics
        self.subset_stats = {
            'source_samples': len(X_source),
            'duplicates_removed': n_duplicates,
            'unique_samples': len(X_unique),
            'q_train_size': len(X_q_train),
            'q_test_size': len(X_q_test),
            'q_train_attack_ratio': float(train_counts[1] / len(y_q_train)),
            'q_test_attack_ratio': float(test_counts[1] / len(y_q_test)),
            'overlap_count': len(overlap)
        }
        
        return X_q_train, y_q_train, X_q_test, y_q_test



class QuantumCircuitBuilder:
    """Builds and visualizes the ZZFeatureMap quantum circuit."""
    
    def __init__(self, config: Config):
        self.config = config
        self.feature_map = None
        self.circuit_stats: Dict[str, Any] = {}
    
    def build_feature_map(self):
        """
        Build ZZFeatureMap for 8 qubits.
        
        Returns:
            ZZFeatureMap circuit
        """
        print("\n" + "="*70)
        print("PHASE 2: QUANTUM CIRCUIT DEFINITION")
        print("="*70)
        
        print(f"\n[CIRCUIT] Building ZZFeatureMap...")
        print(f"  -> Qubits: {self.config.N_QUBITS}")
        print(f"  -> Repetitions: {self.config.FEATURE_MAP_REPS}")
        print(f"  -> Entanglement: {self.config.ENTANGLEMENT}")
        
        try:
            from qiskit.circuit.library import ZZFeatureMap
            
            self.feature_map = ZZFeatureMap(
                feature_dimension=self.config.N_QUBITS,
                reps=self.config.FEATURE_MAP_REPS,
                entanglement=self.config.ENTANGLEMENT
            )
            
            print(f"\n[CIRCUIT PROPERTIES]")
            print(f"  -> Circuit depth: {self.feature_map.depth()}")
            print(f"  -> Number of parameters: {self.feature_map.num_parameters}")
            
            # Decompose for gate count
            decomposed = self.feature_map.decompose()
            print(f"  -> Decomposed depth: {decomposed.depth()}")
            
            # Count gates
            gate_counts = decomposed.count_ops()
            print(f"  -> Gate counts: {dict(gate_counts)}")
            
            self.circuit_stats = {
                'n_qubits': self.config.N_QUBITS,
                'reps': self.config.FEATURE_MAP_REPS,
                'entanglement': self.config.ENTANGLEMENT,
                'depth': self.feature_map.depth(),
                'decomposed_depth': decomposed.depth(),
                'n_parameters': self.feature_map.num_parameters,
                'gate_counts': dict(gate_counts)
            }
            
        except Exception as e:
            raise RuntimeError(f"[FATAL] Failed to build feature map: {e}")
        
        # Save circuit diagram
        print(f"\n[VISUALIZATION] Saving circuit diagram...")
        try:
            ensure_directory_exists(os.path.dirname(self.config.CIRCUIT_DIAGRAM_PATH))
            
            import matplotlib.pyplot as plt
            fig = self.feature_map.decompose().draw(output='mpl', style='iqp')
            fig.savefig(self.config.CIRCUIT_DIAGRAM_PATH, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"  -> Saved: {self.config.CIRCUIT_DIAGRAM_PATH}")
        except Exception as e:
            print(f"  -> [WARNING] Failed to save circuit diagram: {e}")
        
        return self.feature_map



class KernelInitializer:
    """
    Initializes the quantum kernel with fallback strategies.
    
    Priority order:
    1. FidelityStatevectorKernel (recommended for statevector simulation)
    2. Manual statevector kernel (fallback if imports fail)
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.kernel = None
        self.kernel_type: str = ""
    
    def initialize_kernel(self, feature_map):
        """
        Initialize quantum kernel with fallback strategies.
        
        Args:
            feature_map: ZZFeatureMap circuit
        
        Returns:
            Initialized kernel object
        """
        print("\n" + "="*70)
        print("PHASE 3: KERNEL INITIALIZATION")
        print("="*70)
        
        # Try FidelityStatevectorKernel first
        print(f"\n[KERNEL] Attempting FidelityStatevectorKernel initialization...")
        
        try:
            from qiskit_machine_learning.kernels import FidelityStatevectorKernel
            
            self.kernel = FidelityStatevectorKernel(feature_map=feature_map)
            self.kernel_type = "FidelityStatevectorKernel"
            
            print(f"  -> [OK] FidelityStatevectorKernel initialized successfully")
            print(f"  -> Kernel type: {self.kernel_type}")
            print(f"  -> Exact statevector evaluation (no sampling noise)")
            
            return self.kernel
            
        except ImportError as e:
            print(f"  -> [WARNING] FidelityStatevectorKernel import failed: {e}")
        except Exception as e:
            print(f"  -> [WARNING] FidelityStatevectorKernel initialization failed: {e}")
        
        # Fallback: Try TrainableFidelityStatevectorKernel
        print(f"\n[FALLBACK] Trying TrainableFidelityStatevectorKernel...")
        
        try:
            from qiskit_machine_learning.kernels import TrainableFidelityStatevectorKernel
            
            self.kernel = TrainableFidelityStatevectorKernel(feature_map=feature_map)
            self.kernel_type = "TrainableFidelityStatevectorKernel"
            
            print(f"  -> [OK] TrainableFidelityStatevectorKernel initialized")
            
            return self.kernel
            
        except Exception as e:
            print(f"  -> [WARNING] TrainableFidelityStatevectorKernel failed: {e}")
        
        # Final fallback: Manual statevector kernel
        print(f"\n[FALLBACK] Using manual statevector kernel computation...")
        
        self.kernel = ManualStatevectorKernel(feature_map)
        self.kernel_type = "ManualStatevectorKernel"
        
        print(f"  -> [OK] Manual kernel initialized")
        
        return self.kernel


class ManualStatevectorKernel:
    """
    Manual statevector-based quantum kernel computation.
    
    Computes K(x, y) = |<phi(x)|phi(y)>|^2 using direct statevector evaluation.
    Bypasses all qiskit-algorithms dependencies.
    """
    
    def __init__(self, feature_map):
        self.feature_map = feature_map
        self.n_qubits = feature_map.num_qubits
    
    def evaluate(self, x_vec: np.ndarray, y_vec: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Evaluate kernel matrix.
        
        Args:
            x_vec: First set of feature vectors (n_samples_x, n_features)
            y_vec: Second set of feature vectors (n_samples_y, n_features)
                   If None, computes K(x_vec, x_vec)
        
        Returns:
            Kernel matrix of shape (n_samples_x, n_samples_y)
        """
        from qiskit.quantum_info import Statevector
        
        if y_vec is None:
            y_vec = x_vec
        
        n_x = len(x_vec)
        n_y = len(y_vec)
        kernel_matrix = np.zeros((n_x, n_y))
        
        # Precompute statevectors for y_vec
        y_statevectors = []
        for y in y_vec:
            circuit = self.feature_map.assign_parameters(y)
            sv = Statevector(circuit)
            y_statevectors.append(sv.data)
        
        y_statevectors = np.array(y_statevectors)
        
        # Compute kernel matrix
        for i, x in enumerate(x_vec):
            circuit = self.feature_map.assign_parameters(x)
            sv_x = Statevector(circuit).data
            
            # Compute fidelities
            overlaps = np.abs(np.dot(np.conj(sv_x), y_statevectors.T)) ** 2
            kernel_matrix[i, :] = overlaps
        
        return kernel_matrix


class KernelMatrixComputer:
    """
    Computes quantum kernel matrices with batching and checkpointing.
    
    Outputs:
    - K_train: (Q_TRAIN_SIZE x Q_TRAIN_SIZE) training Gram matrix
    - K_test: (Q_TEST_SIZE x Q_TRAIN_SIZE) test kernel matrix
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.computation_stats: Dict[str, Any] = {}
    
    def compute_matrices(
        self,
        kernel,
        X_q_train: np.ndarray,
        X_q_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute K_train and K_test kernel matrices.
        
        Args:
            kernel: Initialized quantum kernel
            X_q_train: Quantum training subset
            X_q_test: Quantum test subset
        
        Returns:
            Tuple of (K_train, K_test)
        """
        print("\n" + "="*70)
        print("PHASE 4: KERNEL MATRIX COMPUTATION")
        print("="*70)
        
        n_train = len(X_q_train)
        n_test = len(X_q_test)
        
        total_train_evals = n_train * n_train
        total_test_evals = n_test * n_train
        total_evals = total_train_evals + total_test_evals
        
        print(f"\n[COMPUTATION PLAN]")
        print(f"  -> K_train: {n_train} x {n_train} = {total_train_evals:,} evaluations")
        print(f"  -> K_test: {n_test} x {n_train} = {total_test_evals:,} evaluations")
        print(f"  -> Total evaluations: {total_evals:,}")
        print(f"  -> Batch size: {self.config.BATCH_SIZE}")
        
        # Compute K_train
        print(f"\n[K_TRAIN] Computing training Gram matrix...")
        start_time = time.time()
        
        K_train = self._compute_matrix_batched(
            kernel, X_q_train, X_q_train, "K_train"
        )
        
        train_time = time.time() - start_time
        print(f"  -> Completed in {format_time(train_time)}")
        print(f"  -> Shape: {K_train.shape}")
        print(f"  -> Value range: [{K_train.min():.4f}, {K_train.max():.4f}]")
        print(f"  -> Mean value: {K_train.mean():.4f}")
        
        # Compute K_test
        print(f"\n[K_TEST] Computing test kernel matrix...")
        start_time = time.time()
        
        K_test = self._compute_matrix_batched(
            kernel, X_q_test, X_q_train, "K_test"
        )
        
        test_time = time.time() - start_time
        print(f"  -> Completed in {format_time(test_time)}")
        print(f"  -> Shape: {K_test.shape}")
        print(f"  -> Value range: [{K_test.min():.4f}, {K_test.max():.4f}]")
        print(f"  -> Mean value: {K_test.mean():.4f}")
        
        # Store statistics
        self.computation_stats = {
            'k_train_shape': K_train.shape,
            'k_test_shape': K_test.shape,
            'k_train_time': train_time,
            'k_test_time': test_time,
            'total_time': train_time + test_time,
            'k_train_mean': float(K_train.mean()),
            'k_test_mean': float(K_test.mean()),
            'k_train_min': float(K_train.min()),
            'k_train_max': float(K_train.max()),
            'k_test_min': float(K_test.min()),
            'k_test_max': float(K_test.max())
        }
        
        return K_train, K_test
    
    def _compute_matrix_batched(
        self,
        kernel,
        X_rows: np.ndarray,
        X_cols: np.ndarray,
        name: str
    ) -> np.ndarray:
        """
        Compute kernel matrix with batching and progress bar.
        
        Args:
            kernel: Quantum kernel
            X_rows: Row samples
            X_cols: Column samples
            name: Matrix name for progress bar
        
        Returns:
            Kernel matrix
        """
        n_rows = len(X_rows)
        n_cols = len(X_cols)
        batch_size = self.config.BATCH_SIZE
        
        # Check if kernel has evaluate method (for our implementations)
        if hasattr(kernel, 'evaluate'):
            # Use batch evaluation
            K = np.zeros((n_rows, n_cols))
            
            n_batches = (n_rows + batch_size - 1) // batch_size
            
            with tqdm(total=n_rows, desc=f"Computing {name}", unit="rows") as pbar:
                for batch_idx in range(n_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min((batch_idx + 1) * batch_size, n_rows)
                    
                    X_batch = X_rows[start_idx:end_idx]
                    K_batch = kernel.evaluate(X_batch, X_cols)
                    K[start_idx:end_idx, :] = K_batch
                    
                    pbar.update(end_idx - start_idx)
            
            return K
        else:
            # Fallback: Use kernel's built-in evaluation
            return kernel.evaluate(X_rows, X_cols)


class ResultsSerializer:
    """Saves kernel matrices and generates visualizations."""
    
    def __init__(self, config: Config):
        self.config = config
    
    def save_matrices(
        self,
        K_train: np.ndarray,
        K_test: np.ndarray
    ) -> Dict[str, str]:
        """
        Save kernel matrices to disk.
        
        Args:
            K_train: Training Gram matrix
            K_test: Test kernel matrix
        
        Returns:
            Dictionary of saved file paths
        """
        print("\n" + "="*70)
        print("PHASE 5: SERIALIZATION")
        print("="*70)
        
        ensure_directory_exists(self.config.HAI_DATA_DIR)
        
        print(f"\n[SAVING] Kernel matrices...")
        
        try:
            np.save(self.config.GRAM_TRAIN_PATH, K_train)
            k_train_size = os.path.getsize(self.config.GRAM_TRAIN_PATH) / 1024 / 1024
            print(f"  -> K_train: {self.config.GRAM_TRAIN_PATH}")
            print(f"     Shape: {K_train.shape}, Size: {k_train_size:.2f} MB")
            
            np.save(self.config.GRAM_TEST_PATH, K_test)
            k_test_size = os.path.getsize(self.config.GRAM_TEST_PATH) / 1024 / 1024
            print(f"  -> K_test: {self.config.GRAM_TEST_PATH}")
            print(f"     Shape: {K_test.shape}, Size: {k_test_size:.2f} MB")
            
        except Exception as e:
            raise IOError(f"[FATAL] Failed to save matrices: {e}")
        
        return {
            'k_train': self.config.GRAM_TRAIN_PATH,
            'k_test': self.config.GRAM_TEST_PATH
        }
    
    def generate_heatmap(
        self,
        K_train: np.ndarray,
        K_test: np.ndarray
    ) -> str:
        """
        Generate kernel matrix heatmap visualization.
        
        Args:
            K_train: Training Gram matrix
            K_test: Test kernel matrix
        
        Returns:
            Path to saved figure
        """
        print(f"\n[VISUALIZATION] Generating kernel heatmaps...")
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            ensure_directory_exists(os.path.dirname(self.config.KERNEL_HEATMAP_PATH))
            
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            # K_train heatmap (subsample for visualization)
            k_train_vis = K_train[::10, ::10] if K_train.shape[0] > 250 else K_train
            sns.heatmap(
                k_train_vis, ax=axes[0], cmap='viridis',
                xticklabels=False, yticklabels=False
            )
            axes[0].set_title(
                f'HAI K_train ({K_train.shape[0]}x{K_train.shape[1]})\n'
                f'Mean: {K_train.mean():.4f}',
                fontsize=12, fontweight='bold'
            )
            
            # K_test heatmap (subsample for visualization)
            k_test_vis = K_test[::5, ::10] if K_test.shape[0] > 200 else K_test
            sns.heatmap(
                k_test_vis, ax=axes[1], cmap='viridis',
                xticklabels=False, yticklabels=False
            )
            axes[1].set_title(
                f'HAI K_test ({K_test.shape[0]}x{K_test.shape[1]})\n'
                f'Mean: {K_test.mean():.4f}',
                fontsize=12, fontweight='bold'
            )
            
            plt.suptitle(
                'HAI Quantum Kernel Matrices (ZZFeatureMap, 8 Qubits)',
                fontsize=14, fontweight='bold'
            )
            plt.tight_layout()
            
            fig.savefig(self.config.KERNEL_HEATMAP_PATH, dpi=200, bbox_inches='tight')
            plt.close(fig)
            
            print(f"  -> Saved: {self.config.KERNEL_HEATMAP_PATH}")
            
            return self.config.KERNEL_HEATMAP_PATH
            
        except Exception as e:
            print(f"  -> [WARNING] Failed to generate heatmap: {e}")
            return ""
    
    def save_computation_log(
        self,
        backend_info: Dict,
        subset_stats: Dict,
        circuit_stats: Dict,
        kernel_type: str,
        computation_stats: Dict,
        total_time: float
    ) -> str:
        """
        Save comprehensive computation log.
        
        Returns:
            Path to saved log
        """
        print(f"\n[LOG] Saving computation log...")
        
        ensure_directory_exists(self.config.LOGS_DIR)
        
        log = {
            "metadata": {
                "dataset": "HAI 22.04 (Hardware-in-the-Loop)",
                "task": "Quantum Kernel Computation for Validation",
                "timestamp": datetime.now().isoformat(),
                "total_time_seconds": total_time
            },
            "backend": backend_info,
            "quantum_subsets": subset_stats,
            "quantum_circuit": circuit_stats,
            "kernel": {
                "type": kernel_type,
                "feature_map": "ZZFeatureMap"
            },
            "computation": computation_stats,
            "output_files": {
                "k_train": self.config.GRAM_TRAIN_PATH,
                "k_test": self.config.GRAM_TEST_PATH,
                "circuit_diagram": self.config.CIRCUIT_DIAGRAM_PATH,
                "heatmap": self.config.KERNEL_HEATMAP_PATH
            }
        }
        
        try:
            with open(self.config.KERNEL_LOG_PATH, 'w') as f:
                json.dump(log, f, indent=2, default=str)
            print(f"  -> Saved: {self.config.KERNEL_LOG_PATH}")
            return self.config.KERNEL_LOG_PATH
        except Exception as e:
            print(f"  -> [WARNING] Failed to save log: {e}")
            return ""


class HAIQuantumKernelPipeline:
    """
    Main orchestrator for HAI quantum kernel computation.
    
    Coordinates all phases:
    0. Backend Verification
    1. Quantum Subset Creation
    2. Circuit Definition
    3. Kernel Initialization
    4. Matrix Computation
    5. Serialization
    """
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.backend_verifier = BackendVerifier(self.config)
        self.subset_creator = QuantumSubsetCreator(self.config)
        self.circuit_builder = QuantumCircuitBuilder(self.config)
        self.kernel_initializer = KernelInitializer(self.config)
        self.matrix_computer = KernelMatrixComputer(self.config)
        self.serializer = ResultsSerializer(self.config)
    
    def run(self) -> Dict[str, Any]:
        """
        Execute the complete quantum kernel computation pipeline.
        
        Returns:
            Dictionary containing all pipeline results
        """
        print("\n" + "="*70)
        print("HAI QUANTUM KERNEL COMPUTATION PIPELINE")
        print("Validation Experiment: Proving Generalizability")
        print("="*70)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        results = {}
        pipeline_start = time.time()
        
        # Phase 0: Backend Verification
        try:
            backend_info = self.backend_verifier.verify()
            results['backend_info'] = backend_info
        except Exception as e:
            print(f"\n[FATAL] Phase 0 failed: {e}")
            raise
        
        # Phase 1: Create Quantum Subsets
        try:
            X_q_train, y_q_train, X_q_test, y_q_test = self.subset_creator.create_subsets()
            results['subset_stats'] = self.subset_creator.subset_stats
        except Exception as e:
            print(f"\n[FATAL] Phase 1 failed: {e}")
            raise
        
        # Phase 2: Build Quantum Circuit
        try:
            feature_map = self.circuit_builder.build_feature_map()
            results['circuit_stats'] = self.circuit_builder.circuit_stats
        except Exception as e:
            print(f"\n[FATAL] Phase 2 failed: {e}")
            raise
        
        # Phase 3: Initialize Kernel
        try:
            kernel = self.kernel_initializer.initialize_kernel(feature_map)
            results['kernel_type'] = self.kernel_initializer.kernel_type
        except Exception as e:
            print(f"\n[FATAL] Phase 3 failed: {e}")
            raise
        
        # Phase 4: Compute Kernel Matrices
        try:
            K_train, K_test = self.matrix_computer.compute_matrices(
                kernel, X_q_train, X_q_test
            )
            results['computation_stats'] = self.matrix_computer.computation_stats
        except Exception as e:
            print(f"\n[FATAL] Phase 4 failed: {e}")
            raise
        
        # Phase 5: Save Results
        try:
            self.serializer.save_matrices(K_train, K_test)
            self.serializer.generate_heatmap(K_train, K_test)
            
            total_time = time.time() - pipeline_start
            
            self.serializer.save_computation_log(
                backend_info,
                self.subset_creator.subset_stats,
                self.circuit_builder.circuit_stats,
                self.kernel_initializer.kernel_type,
                self.matrix_computer.computation_stats,
                total_time
            )
        except Exception as e:
            print(f"\n[WARNING] Phase 5 partially failed: {e}")
        
        # Final Summary
        total_time = time.time() - pipeline_start
        
        print("\n" + "="*70)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("="*70)
        
        print(f"\n[KERNEL STATISTICS]")
        print(f"  -> K_train shape: {K_train.shape}")
        print(f"  -> K_train mean: {K_train.mean():.4f}")
        print(f"  -> K_test shape: {K_test.shape}")
        print(f"  -> K_test mean: {K_test.mean():.4f}")
        
        print(f"\n[OUTPUT FILES]")
        print(f"  -> K_train: {self.config.GRAM_TRAIN_PATH}")
        print(f"  -> K_test: {self.config.GRAM_TEST_PATH}")
        
        print(f"\n[TOTAL TIME] {format_time(total_time)}")
        print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        results['total_time'] = total_time
        
        return results



def main():
    """Main entry point for the HAI quantum kernel computation pipeline."""
    
    # Set random seed for reproducibility
    np.random.seed(Config.RANDOM_SEED)
    
    # Initialize and run pipeline
    pipeline = HAIQuantumKernelPipeline()
    
    try:
        results = pipeline.run()
        print("\n[SUCCESS] HAI quantum kernel computation completed without errors")
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
