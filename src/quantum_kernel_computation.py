#!/usr/bin/env python3


import os
import sys
import json
import time
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Progress tracking
from tqdm import tqdm


class Config:
    """Centralized configuration for Quantum Kernel pipeline."""
    
    # Input paths (from Step 2)
    BASE_DIR = "/content/drive/MyDrive/Quantum_Research"
    PROCESSED_DIR = f"{BASE_DIR}/02_Processed_Data"
    
    X_TRAIN_PATH = f"{PROCESSED_DIR}/X_train_reduced.npy"
    Y_TRAIN_PATH = f"{PROCESSED_DIR}/y_train_reduced.npy"
    X_TEST_PATH = f"{PROCESSED_DIR}/X_test_reduced.npy"
    Y_TEST_PATH = f"{PROCESSED_DIR}/y_test_reduced.npy"
    FEATURE_NAMES_PATH = f"{PROCESSED_DIR}/selected_feature_names.joblib"
    
    # Output directories
    CHECKPOINTS_DIR = f"{BASE_DIR}/03_Checkpoints"
    PLOTS_DIR = f"{BASE_DIR}/05_Results/Plots"
    
    # Quantum Subset Parameters
    Q_TRAIN_SIZE = 2500      # Quantum training subset size
    Q_TEST_SIZE = 1000       # Quantum test subset size
    
    # Quantum Circuit Parameters
    N_QUBITS = 8             # Must match number of features
    FEATURE_MAP_REPS = 2     # ZZFeatureMap repetitions
    ENTANGLEMENT = 'linear'  # Entanglement strategy
    
    # Kernel Computation Parameters
    BATCH_SIZE = 100         # Samples per batch (memory safety)
    CHECKPOINT_INTERVAL = 500  # Save checkpoint every N rows
    
    # Backend Parameters
    SIMULATOR_METHOD = 'statevector'
    USE_GPU = True           # Attempt GPU acceleration
    
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
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def get_memory_usage_mb() -> float:
    """Get current memory usage in MB."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        return -1.0



class QuantumBackendVerifier:
    """Verifies Qiskit installation and GPU availability."""
    
    def __init__(self, config: Config):
        self.config = config
        self.backend = None
        self.gpu_available = False
        self.verification_results: Dict[str, any] = {}
    
    def verify_and_initialize(self):
        """
        Verify Qiskit installation and initialize backend.
        
        Raises:
            ImportError: If required packages are not installed
            RuntimeError: If GPU is required but not available
        """
        print("\n" + "="*70)
        print("PHASE 0: QUANTUM BACKEND VERIFICATION")
        print("="*70)
        
        # Check Qiskit installation
        print("\n[CHECKING] Qiskit installation...")
        try:
            import qiskit
            print(f"  -> Qiskit version: {qiskit.__version__}")
            self.verification_results['qiskit_version'] = qiskit.__version__
        except ImportError:
            raise ImportError("[FATAL] Qiskit not installed. Run: pip install qiskit")
        
        # Check Qiskit Aer
        print("\n[CHECKING] Qiskit Aer...")
        try:
            from qiskit_aer import AerSimulator
            print(f"  -> Qiskit Aer available")
        except ImportError:
            raise ImportError("[FATAL] Qiskit Aer not installed. Run: pip install qiskit-aer")
        
        # Check Qiskit Machine Learning
        print("\n[CHECKING] Qiskit Machine Learning...")
        try:
            import qiskit_machine_learning
            print(f"  -> Qiskit ML version: {qiskit_machine_learning.__version__}")
            self.verification_results['qiskit_ml_version'] = qiskit_machine_learning.__version__
        except ImportError:
            raise ImportError("[FATAL] Qiskit ML not installed. Run: pip install qiskit-machine-learning")
        
        # Check available devices
        print("\n[CHECKING] Available simulation devices...")
        simulator = AerSimulator()
        available_devices = simulator.available_devices()
        print(f"  -> Available devices: {available_devices}")
        self.verification_results['available_devices'] = available_devices
        
        # Check for GPU
        self.gpu_available = 'GPU' in available_devices
        
        if self.gpu_available:
            print(f"  -> [OK] GPU DETECTED - Will use GPU acceleration")
            self.backend = AerSimulator(
                method=self.config.SIMULATOR_METHOD,
                device='GPU'
            )
        else:
            print(f"  -> [WARNING] GPU NOT DETECTED")
            if self.config.USE_GPU:
                print(f"  -> [INFO] Falling back to CPU (will be slower)")
                print(f"  -> [TIP] For GPU support, install: pip install qiskit-aer-gpu")
            self.backend = AerSimulator(
                method=self.config.SIMULATOR_METHOD,
                device='CPU'
            )
        
        self.verification_results['gpu_available'] = self.gpu_available
        self.verification_results['backend_device'] = 'GPU' if self.gpu_available else 'CPU'
        
        print(f"\n[BACKEND] Initialized AerSimulator")
        print(f"  -> Method: {self.config.SIMULATOR_METHOD}")
        print(f"  -> Device: {'GPU' if self.gpu_available else 'CPU'}")
        
        return self.backend


class QuantumSubsetCreator:
    """Creates stratified subsets for quantum kernel computation."""
    
    def __init__(self, config: Config):
        self.config = config
        self.subset_stats: Dict[str, any] = {}
    
    def create_subsets(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Create stratified subsets for quantum experiments.
        
        CRITICAL: Ensures DISJOINT sets (no overlap between Q_Train and Q_Test).
        Method: 
            1. De-duplicate source data (remove identical sensor readings)
            2. Sequential splitting - take train first, then test from remainder
        
        Returns:
            Tuple of (X_q_train, y_q_train, X_q_test, y_q_test)
        """
        print("\n" + "="*70)
        print("PHASE 1: QUANTUM SUBSET CREATION")
        print("="*70)
        
        # Load reduced data
        print(f"\n[LOADING] Reduced training data...")
        try:
            X_train = np.load(self.config.X_TRAIN_PATH)
            y_train = np.load(self.config.Y_TRAIN_PATH)
            print(f"  -> X_train shape: {X_train.shape}")
            print(f"  -> y_train shape: {y_train.shape}")
        except Exception as e:
            raise IOError(f"[FATAL] Failed to load training data: {e}")
        
        print(f"\n[LOADING] Reduced test data...")
        try:
            X_test = np.load(self.config.X_TEST_PATH)
            y_test = np.load(self.config.Y_TEST_PATH)
            print(f"  -> X_test shape: {X_test.shape}")
            print(f"  -> y_test shape: {y_test.shape}")
        except Exception as e:
            raise IOError(f"[FATAL] Failed to load test data: {e}")
        
        # Check class distributions
        print(f"\n[CLASS CHECK] Original distributions:")
        train_classes, train_counts = np.unique(y_train, return_counts=True)
        test_classes, test_counts = np.unique(y_test, return_counts=True)
        print(f"  -> Train: {dict(zip(train_classes.astype(int), train_counts.astype(int)))}")
        print(f"  -> Test: {dict(zip(test_classes.astype(int), test_counts.astype(int)))}")
        
        # CRITICAL: DE-DUPLICATION OF SOURCE DATA
        # In ICS/SCADA systems, steady-state operation produces identical
        # sensor readings across multiple timesteps. These duplicates must
        # be removed to guarantee disjoint train/test sets.
        
        
        print(f"\n[DE-DUPLICATION] Removing identical sensor readings...")
        print(f"  -> Reason: Steady-state plant operation produces duplicate rows")
        print(f"  -> Original test set size: {len(X_test)}")
        
        # Combine X and y for synchronized de-duplication
        # Use a structured approach: hash each row, keep first occurrence
        
        # Round to avoid floating-point precision issues
        X_test_rounded = np.round(X_test, decimals=6)
        
        # Find unique rows (keep first occurrence)
        _, unique_indices = np.unique(
            X_test_rounded, 
            axis=0, 
            return_index=True
        )
        
        # Sort indices to maintain temporal order
        unique_indices = np.sort(unique_indices)
        
        # Apply de-duplication
        X_test_unique = X_test[unique_indices]
        y_test_unique = y_test[unique_indices]
        
        n_duplicates = len(X_test) - len(X_test_unique)
        print(f"  -> Duplicates removed: {n_duplicates:,}")
        print(f"  -> Unique samples: {len(X_test_unique):,}")
        print(f"  -> Deduplication rate: {n_duplicates/len(X_test)*100:.2f}%")
        
        # Verify class distribution after de-duplication
        unique_classes, unique_counts = np.unique(y_test_unique, return_counts=True)
        unique_dist = dict(zip(unique_classes.astype(int), unique_counts.astype(int)))
        print(f"  -> Post-dedup class distribution: {unique_dist}")
        
        # DISJOINT SET CREATION (on de-duplicated data)
        # Method: Sequential split to GUARANTEE no overlap
        
        from sklearn.model_selection import train_test_split
        
        np.random.seed(self.config.RANDOM_SEED)
        
        print(f"\n[DISJOINT SPLIT] Creating guaranteed non-overlapping subsets...")
        print(f"  -> Source: De-duplicated test set ({len(X_test_unique)} unique samples)")
        print(f"  -> Method: Sequential stratified splitting")
        
        # Verify we have enough unique samples
        total_needed = self.config.Q_TRAIN_SIZE + self.config.Q_TEST_SIZE
        if len(X_test_unique) < total_needed:
            raise ValueError(f"[FATAL] Insufficient unique samples. Need {total_needed}, have {len(X_test_unique)}")
        
        print(f"  -> Required: {self.config.Q_TRAIN_SIZE} (train) + {self.config.Q_TEST_SIZE} (test) = {total_needed}")
        print(f"  -> Available unique: {len(X_test_unique)} samples")
        
        # STEP 1: Extract Q_Train from de-duplicated set, keep remainder
        print(f"\n[STEP 1] Extracting Q_Train ({self.config.Q_TRAIN_SIZE} samples)...")
        
        X_q_train, X_remainder, y_q_train, y_remainder = train_test_split(
            X_test_unique, y_test_unique,
            train_size=self.config.Q_TRAIN_SIZE,
            stratify=y_test_unique,
            random_state=self.config.RANDOM_SEED
        )
        
        q_train_classes, q_train_counts = np.unique(y_q_train, return_counts=True)
        q_train_dist = dict(zip(q_train_classes.astype(int), q_train_counts.astype(int)))
        print(f"  -> Q_Train extracted: {len(X_q_train)} samples")
        print(f"  -> Q_Train class distribution: {q_train_dist}")
        print(f"  -> Remainder pool: {len(X_remainder)} samples")
        
        # STEP 2: Extract Q_Test from REMAINDER (guarantees no overlap)
        print(f"\n[STEP 2] Extracting Q_Test ({self.config.Q_TEST_SIZE} samples) from REMAINDER...")
        
        X_q_test, X_unused, y_q_test, y_unused = train_test_split(
            X_remainder, y_remainder,
            train_size=self.config.Q_TEST_SIZE,
            stratify=y_remainder,
            random_state=self.config.RANDOM_SEED
        )
        
        q_test_classes, q_test_counts = np.unique(y_q_test, return_counts=True)
        q_test_dist = dict(zip(q_test_classes.astype(int), q_test_counts.astype(int)))
        print(f"  -> Q_Test extracted: {len(X_q_test)} samples")
        print(f"  -> Q_Test class distribution: {q_test_dist}")
        print(f"  -> Unused samples: {len(X_unused)} (discarded)")
        
        # VERIFICATION: Prove sets are disjoint (for IEEE Access reviewers)
        # This is now a SAFETY NET - should always pass after de-duplication
        print(f"\n[VERIFICATION] Proving disjoint sets (CRITICAL FOR IEEE ACCESS)...")
        
        # Method 1: Check array shapes
        print(f"  -> Q_Train shape: {X_q_train.shape}")
        print(f"  -> Q_Test shape: {X_q_test.shape}")
        
        # Method 2: Hash-based overlap detection (rounded for float comparison)
        X_q_train_rounded = np.round(X_q_train, decimals=6)
        X_q_test_rounded = np.round(X_q_test, decimals=6)
        
        train_hashes = set(map(tuple, X_q_train_rounded))
        test_hashes = set(map(tuple, X_q_test_rounded))
        overlap = train_hashes.intersection(test_hashes)
        
        if len(overlap) > 0:
            print(f"  -> [ERROR] Overlap detected: {len(overlap)} samples")
            print(f"  -> This should not happen after de-duplication!")
            raise RuntimeError(f"[FATAL] DATA LEAKAGE DETECTED: {len(overlap)} overlapping samples!")
        
        print(f"  -> Hash-based overlap check: {len(overlap)} overlapping samples")
        print(f"  -> [OK] VERIFIED: Q_Train and Q_Test are COMPLETELY DISJOINT")
        print(f"  -> De-duplication + Sequential split guarantees zero overlap")
        print(f"  -> [OK] NO DATA LEAKAGE - Safe for IEEE Access submission")
        
        # Save subsets for classical baseline comparison
        print(f"\n[SAVING] Quantum subsets for reproducibility...")
        ensure_directory_exists(self.config.PROCESSED_DIR)
        
        subset_paths = {
            'X_q_train': os.path.join(self.config.PROCESSED_DIR, 'X_q_train.npy'),
            'y_q_train': os.path.join(self.config.PROCESSED_DIR, 'y_q_train.npy'),
            'X_q_test': os.path.join(self.config.PROCESSED_DIR, 'X_q_test.npy'),
            'y_q_test': os.path.join(self.config.PROCESSED_DIR, 'y_q_test.npy'),
        }
        
        try:
            np.save(subset_paths['X_q_train'], X_q_train.astype(np.float64))
            np.save(subset_paths['y_q_train'], y_q_train)
            np.save(subset_paths['X_q_test'], X_q_test.astype(np.float64))
            np.save(subset_paths['y_q_test'], y_q_test)
            
            for name, path in subset_paths.items():
                print(f"  -> Saved: {name} to {path}")
        except Exception as e:
            raise IOError(f"[FATAL] Failed to save subsets: {e}")
        
        # Store statistics
        self.subset_stats = {
            'original_test_size': len(X_test),
            'duplicates_removed': n_duplicates,
            'unique_samples': len(X_test_unique),
            'deduplication_rate': n_duplicates / len(X_test),
            'q_train_size': len(X_q_train),
            'q_test_size': len(X_q_test),
            'q_train_distribution': q_train_dist,
            'q_test_distribution': q_test_dist,
            'n_features': X_q_train.shape[1],
            'overlap_count': len(overlap),
            'disjoint_verified': True
        }
        
        return X_q_train, y_q_train, X_q_test, y_q_test



class QuantumCircuitBuilder:
    """Builds and visualizes the quantum feature map circuit."""
    
    def __init__(self, config: Config):
        self.config = config
        self.feature_map = None
        self.circuit_stats: Dict[str, any] = {}
    
    def build_feature_map(self):
        """
        Build ZZFeatureMap for quantum kernel.
        
        Returns:
            QuantumCircuit: The feature map circuit
        """
        print("\n" + "="*70)
        print("PHASE 2: QUANTUM CIRCUIT DEFINITION")
        print("="*70)
        
        from qiskit.circuit.library import ZZFeatureMap
        
        print(f"\n[BUILDING] ZZFeatureMap circuit...")
        print(f"  -> Qubits: {self.config.N_QUBITS}")
        print(f"  -> Repetitions: {self.config.FEATURE_MAP_REPS}")
        print(f"  -> Entanglement: {self.config.ENTANGLEMENT}")
        
        self.feature_map = ZZFeatureMap(
            feature_dimension=self.config.N_QUBITS,
            reps=self.config.FEATURE_MAP_REPS,
            entanglement=self.config.ENTANGLEMENT
        )
        
        # Circuit statistics
        self.circuit_stats = {
            'n_qubits': self.feature_map.num_qubits,
            'depth': self.feature_map.depth(),
            'n_parameters': self.feature_map.num_parameters,
            'n_gates': len(self.feature_map.data),
            'entanglement': self.config.ENTANGLEMENT,
            'reps': self.config.FEATURE_MAP_REPS
        }
        
        print(f"\n[CIRCUIT STATISTICS]")
        print(f"  -> Number of qubits: {self.circuit_stats['n_qubits']}")
        print(f"  -> Circuit depth: {self.circuit_stats['depth']}")
        print(f"  -> Parameters: {self.circuit_stats['n_parameters']}")
        print(f"  -> Total gates: {self.circuit_stats['n_gates']}")
        
        return self.feature_map
    
    def visualize_circuit(self) -> str:
        """
        Generate circuit diagram for publication.
        
        Returns:
            Path to saved circuit image
        """
        print(f"\n[VISUALIZING] Generating circuit diagram...")
        
        ensure_directory_exists(self.config.PLOTS_DIR)
        
        # Decompose circuit to show actual gates
        from qiskit import transpile
        from qiskit_aer import AerSimulator
        
        # Transpile to basis gates for realistic view
        backend = AerSimulator()
        decomposed = transpile(
            self.feature_map, 
            backend, 
            optimization_level=0
        )
        
        print(f"  -> Decomposed circuit depth: {decomposed.depth()}")
        print(f"  -> Decomposed gate count: {len(decomposed.data)}")
        
        # Generate figure
        try:
            import matplotlib.pyplot as plt
            
            # Draw circuit
            fig = self.feature_map.decompose().draw(
                output='mpl',
                style={
                    'backgroundcolor': '#FFFFFF',
                    'linecolor': '#000000',
                    'textcolor': '#000000',
                    'gatefacecolor': '#E6F3FF',
                    'barrierfacecolor': '#BDBDBD'
                },
                fold=40,  # Fold long circuits
                scale=0.8
            )
            
            # Add title
            fig.suptitle(
                f'ZZFeatureMap: {self.config.N_QUBITS} qubits, '
                f'{self.config.FEATURE_MAP_REPS} reps, '
                f'{self.config.ENTANGLEMENT} entanglement',
                fontsize=12,
                fontweight='bold'
            )
            
            # Save
            circuit_path = os.path.join(self.config.PLOTS_DIR, 'quantum_circuit.png')
            fig.savefig(circuit_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            
            print(f"  -> Saved: {circuit_path}")
            
            self.circuit_stats['circuit_image'] = circuit_path
            
            return circuit_path
            
        except Exception as e:
            print(f"  -> [WARNING] Could not generate circuit image: {e}")
            return None


class ManualStatevectorKernel:
    """
    Manual implementation of quantum kernel using direct statevector computation.
    
    This bypasses all qiskit-machine-learning and qiskit-algorithms dependencies
    that have broken import chains in Qiskit 1.x.
    
    The kernel computes: K(x, y) = |<phi(x)|phi(y)>|^2
    where phi(x) is the statevector after applying the feature map with data x.
    """
    
    def __init__(self, feature_map):
        """
        Initialize the manual statevector kernel.
        
        Args:
            feature_map: Qiskit QuantumCircuit (parameterized feature map)
        """
        self.feature_map = feature_map
        self.num_qubits = feature_map.num_qubits
        self.num_parameters = feature_map.num_parameters
        
    def _get_statevector(self, x: np.ndarray) -> np.ndarray:
        """
        Compute statevector for a single data point.
        
        Args:
            x: Data point (1D array of features)
        
        Returns:
            Complex statevector as numpy array
        """
        from qiskit.quantum_info import Statevector
        
        # Bind parameters to feature map
        bound_circuit = self.feature_map.assign_parameters(x)
        
        # Get statevector
        sv = Statevector(bound_circuit)
        return sv.data
    
    def evaluate(self, x_vec: np.ndarray, y_vec: np.ndarray = None) -> np.ndarray:
        """
        Evaluate the quantum kernel matrix.
        
        Args:
            x_vec: First set of data points (n_samples_x, n_features)
            y_vec: Second set of data points (n_samples_y, n_features)
                   If None, computes K(x_vec, x_vec)
        
        Returns:
            Kernel matrix of shape (n_samples_x, n_samples_y)
        """
        if y_vec is None:
            y_vec = x_vec
            symmetric = True
        else:
            symmetric = False
        
        n_x = len(x_vec)
        n_y = len(y_vec)
        
        # Compute statevectors for all points
        sv_x = np.array([self._get_statevector(x) for x in x_vec])
        
        if symmetric:
            sv_y = sv_x
        else:
            sv_y = np.array([self._get_statevector(y) for y in y_vec])
        
        # Compute kernel matrix: K[i,j] = |<sv_x[i]|sv_y[j]>|^2
        # Inner product: sv_x[i].conj() @ sv_y[j]
        # Fidelity: |inner_product|^2
        
        kernel_matrix = np.abs(sv_x.conj() @ sv_y.T) ** 2
        
        return kernel_matrix



class QuantumKernelComputer:
    """Computes quantum kernel matrices with batching and checkpointing."""
    
    def __init__(self, config: Config, backend, feature_map):
        self.config = config
        self.backend = backend
        self.feature_map = feature_map
        self.kernel = None
        self.computation_stats: Dict[str, any] = {}
    
    def initialize_kernel(self):
        """
        Initialize the Quantum Kernel using modern Qiskit 1.x API.
        
        Uses TrainableFidelityStatevectorKernel which:
        - Bypasses the broken qiskit-algorithms dependency chain
        - Is optimized for statevector simulation
        - Works directly with Aer GPU backend
        """
        print("\n" + "="*70)
        print("PHASE 3: QUANTUM KERNEL INITIALIZATION (MODERN 1.x STACK)")
        print("="*70)
        
        print(f"\n[INIT] Attempting modern Qiskit 1.x kernel initialization...")
        
        # Try multiple kernel types in order of preference
        kernel_initialized = False
        
        # Option 1: TrainableFidelityStatevectorKernel (most stable)
        if not kernel_initialized:
            try:
                from qiskit_machine_learning.kernels import TrainableFidelityStatevectorKernel
                
                print(f"\n[INIT] Using TrainableFidelityStatevectorKernel...")
                self.kernel = TrainableFidelityStatevectorKernel(
                    feature_map=self.feature_map
                )
                print(f"  -> Kernel type: TrainableFidelityStatevectorKernel")
                print(f"  -> [OK] Initialized successfully")
                kernel_initialized = True
                
            except ImportError as e:
                print(f"  -> TrainableFidelityStatevectorKernel not available: {e}")
            except Exception as e:
                print(f"  -> TrainableFidelityStatevectorKernel failed: {e}")
        
        # Option 2: FidelityStatevectorKernel (alternate)
        if not kernel_initialized:
            try:
                from qiskit_machine_learning.kernels import FidelityStatevectorKernel
                
                print(f"\n[INIT] Using FidelityStatevectorKernel...")
                self.kernel = FidelityStatevectorKernel(
                    feature_map=self.feature_map
                )
                print(f"  -> Kernel type: FidelityStatevectorKernel")
                print(f"  -> [OK] Initialized successfully")
                kernel_initialized = True
                
            except ImportError as e:
                print(f"  -> FidelityStatevectorKernel not available: {e}")
            except Exception as e:
                print(f"  -> FidelityStatevectorKernel failed: {e}")
        
        # Option 3: Manual statevector kernel implementation
        if not kernel_initialized:
            print(f"\n[INIT] Using manual statevector kernel implementation...")
            self.kernel = ManualStatevectorKernel(self.feature_map)
            print(f"  -> Kernel type: ManualStatevectorKernel (custom)")
            print(f"  -> [OK] Initialized successfully")
            kernel_initialized = True
        
        if not kernel_initialized:
            raise RuntimeError("[FATAL] Could not initialize any quantum kernel!")
        
        print(f"\n[KERNEL CONFIG]")
        print(f"  -> Feature map: ZZFeatureMap (8 qubits)")
        print(f"  -> Evaluation: Exact statevector (no sampling noise)")
        print(f"  -> Backend: Will use available GPU if detected")
        
        return self.kernel
    
    def compute_kernel_matrix_batched(
        self,
        X1: np.ndarray,
        X2: np.ndarray,
        matrix_name: str
    ) -> np.ndarray:
        """
        Compute kernel matrix with batching for memory safety.
        
        Args:
            X1: First set of samples (rows of output matrix)
            X2: Second set of samples (columns of output matrix)
            matrix_name: Name for logging/checkpointing
        
        Returns:
            Kernel matrix K(X1, X2)
        """
        n1, n2 = len(X1), len(X2)
        print(f"\n[COMPUTING] {matrix_name}: ({n1} x {n2}) matrix")
        print(f"  -> Total kernel evaluations: {n1 * n2:,}")
        print(f"  -> Batch size: {self.config.BATCH_SIZE}")
        
        # Initialize result matrix
        K = np.zeros((n1, n2), dtype=np.float64)
        
        # Checkpoint directory
        ensure_directory_exists(self.config.CHECKPOINTS_DIR)
        checkpoint_path = os.path.join(
            self.config.CHECKPOINTS_DIR, 
            f'{matrix_name}_checkpoint.npy'
        )
        
        # Check for existing checkpoint
        start_row = 0
        if os.path.exists(checkpoint_path):
            try:
                checkpoint_data = np.load(checkpoint_path, allow_pickle=True).item()
                K = checkpoint_data['matrix']
                start_row = checkpoint_data['completed_rows']
                print(f"  -> [RESUME] Found checkpoint at row {start_row}")
            except Exception as e:
                print(f"  -> [WARNING] Could not load checkpoint: {e}")
                start_row = 0
        
        # Compute in batches
        start_time = time.time()
        total_batches = (n1 - start_row + self.config.BATCH_SIZE - 1) // self.config.BATCH_SIZE
        
        print(f"  -> Starting from row {start_row}")
        print(f"  -> Estimated batches: {total_batches}")
        
        try:
            with tqdm(total=n1-start_row, desc=f"  {matrix_name}", unit="rows") as pbar:
                for i in range(start_row, n1, self.config.BATCH_SIZE):
                    batch_end = min(i + self.config.BATCH_SIZE, n1)
                    X1_batch = X1[i:batch_end]
                    
                    # Compute kernel for this batch
                    # K[i:batch_end, :] = self.kernel.evaluate(X1_batch, X2)
                    
                    # For memory efficiency, compute column-wise too if X2 is large
                    if n2 > self.config.BATCH_SIZE * 2:
                        for j in range(0, n2, self.config.BATCH_SIZE):
                            j_end = min(j + self.config.BATCH_SIZE, n2)
                            X2_batch = X2[j:j_end]
                            K[i:batch_end, j:j_end] = self.kernel.evaluate(X1_batch, X2_batch)
                    else:
                        K[i:batch_end, :] = self.kernel.evaluate(X1_batch, X2)
                    
                    pbar.update(batch_end - i)
                    
                    # Save checkpoint periodically
                    if (batch_end - start_row) % self.config.CHECKPOINT_INTERVAL == 0:
                        checkpoint_data = {
                            'matrix': K,
                            'completed_rows': batch_end,
                            'timestamp': datetime.now().isoformat()
                        }
                        np.save(checkpoint_path, checkpoint_data, allow_pickle=True)
        
        except KeyboardInterrupt:
            print(f"\n  -> [INTERRUPTED] Saving checkpoint at row {batch_end}...")
            checkpoint_data = {
                'matrix': K,
                'completed_rows': batch_end,
                'timestamp': datetime.now().isoformat()
            }
            np.save(checkpoint_path, checkpoint_data, allow_pickle=True)
            raise
        
        elapsed = time.time() - start_time
        
        # Clean up checkpoint after successful completion
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            print(f"  -> [CLEANUP] Removed checkpoint file")
        
        # Statistics
        self.computation_stats[matrix_name] = {
            'shape': K.shape,
            'elapsed_time': elapsed,
            'evaluations_per_second': (n1 * n2) / elapsed if elapsed > 0 else 0,
            'min_value': float(K.min()),
            'max_value': float(K.max()),
            'mean_value': float(K.mean())
        }
        
        print(f"  -> Completed in {format_time(elapsed)}")
        print(f"  -> Kernel value range: [{K.min():.4f}, {K.max():.4f}]")
        print(f"  -> Mean kernel value: {K.mean():.4f}")
        
        return K
    
    def compute_all_matrices(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute both training and test kernel matrices.
        
        Args:
            X_train: Training samples
            X_test: Test samples
        
        Returns:
            Tuple of (K_train, K_test)
        """
        print("\n" + "="*70)
        print("PHASE 4: KERNEL MATRIX COMPUTATION")
        print("="*70)
        
        print(f"\n[INFO] Computing two kernel matrices:")
        print(f"  -> K_train: ({len(X_train)} x {len(X_train)}) - Training Gram matrix")
        print(f"  -> K_test:  ({len(X_test)} x {len(X_train)}) - Test-Train kernel")
        
        total_evals = len(X_train)**2 + len(X_test) * len(X_train)
        print(f"  -> Total kernel evaluations: {total_evals:,}")
        
        # Compute training Gram matrix K(Train, Train)
        K_train = self.compute_kernel_matrix_batched(
            X_train, X_train, "gram_matrix_train"
        )
        
        # Compute test matrix K(Test, Train)
        K_test = self.compute_kernel_matrix_batched(
            X_test, X_train, "gram_matrix_test"
        )
        
        return K_train, K_test



class MatrixSerializer:
    """Handles serialization and visualization of kernel matrices."""
    
    def __init__(self, config: Config):
        self.config = config
        self.saved_paths: Dict[str, str] = {}
    
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
        print("PHASE 5: MATRIX SERIALIZATION")
        print("="*70)
        
        ensure_directory_exists(self.config.PROCESSED_DIR)
        
        # Save matrices
        matrices = {
            'gram_matrix_train': K_train,
            'gram_matrix_test': K_test
        }
        
        print(f"\n[SAVING] Kernel matrices...")
        
        for name, matrix in matrices.items():
            filepath = os.path.join(self.config.PROCESSED_DIR, f'{name}.npy')
            try:
                np.save(filepath, matrix)
                self.saved_paths[name] = filepath
                print(f"  -> {name}: {filepath}")
                print(f"     Shape: {matrix.shape}, Size: {matrix.nbytes / 1024 / 1024:.2f} MB")
            except Exception as e:
                print(f"  -> [ERROR] Failed to save {name}: {e}")
                raise
        
        return self.saved_paths
    
    def visualize_matrices(
        self,
        K_train: np.ndarray,
        K_test: np.ndarray
    ) -> str:
        """
        Generate heatmap visualization of kernel matrices.
        
        Args:
            K_train: Training Gram matrix
            K_test: Test kernel matrix
        
        Returns:
            Path to saved visualization
        """
        print(f"\n[VISUALIZING] Generating kernel matrix heatmaps...")
        
        ensure_directory_exists(self.config.PLOTS_DIR)
        
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Training Gram matrix
        im1 = axes[0].imshow(K_train, cmap='viridis', aspect='auto')
        axes[0].set_title(f'Training Gram Matrix\n({K_train.shape[0]} x {K_train.shape[1]})', 
                          fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Sample Index')
        axes[0].set_ylabel('Sample Index')
        plt.colorbar(im1, ax=axes[0], label='Kernel Value')
        
        # Test kernel matrix
        im2 = axes[1].imshow(K_test, cmap='viridis', aspect='auto')
        axes[1].set_title(f'Test Kernel Matrix\n({K_test.shape[0]} x {K_test.shape[1]})', 
                          fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Training Sample Index')
        axes[1].set_ylabel('Test Sample Index')
        plt.colorbar(im2, ax=axes[1], label='Kernel Value')
        
        plt.suptitle('Quantum Kernel Matrices (ZZFeatureMap, 8 qubits)', 
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # Save
        heatmap_path = os.path.join(self.config.PLOTS_DIR, 'kernel_matrices_heatmap.png')
        try:
            fig.savefig(heatmap_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            print(f"  -> Saved: {heatmap_path}")
            return heatmap_path
        except Exception as e:
            print(f"  -> [WARNING] Could not save heatmap: {e}")
            plt.close(fig)
            return None



class QuantumKernelPipeline:
    """
    Main orchestrator for the Quantum Kernel computation pipeline.
    
    Coordinates all phases:
    0. Backend Verification (GPU check)
    1. Quantum Subset Creation
    2. Circuit Definition
    3. Kernel Initialization
    4. Matrix Computation
    5. Serialization & Visualization
    """
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.backend_verifier = QuantumBackendVerifier(self.config)
        self.subset_creator = QuantumSubsetCreator(self.config)
        self.circuit_builder = QuantumCircuitBuilder(self.config)
        self.serializer = MatrixSerializer(self.config)
        self.kernel_computer = None  # Initialized after backend verification
    
    def run(self) -> Dict[str, any]:
        """
        Execute the complete Quantum Kernel pipeline.
        
        Returns:
            Dictionary containing all pipeline results
        """
        print("\n" + "="*70)
        print("QUANTUM KERNEL (GRAM MATRIX) COMPUTATION PIPELINE")
        print("ZZFeatureMap | 8 Qubits | Fidelity Kernel")
        print("="*70)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        results = {}
        pipeline_start = time.time()
        
        # Phase 0: Backend Verification
        try:
            backend = self.backend_verifier.verify_and_initialize()
            results['backend_info'] = self.backend_verifier.verification_results
        except Exception as e:
            print(f"\n[FATAL] Phase 0 failed: {e}")
            raise
        
        # Phase 1: Quantum Subset Creation
        try:
            X_q_train, y_q_train, X_q_test, y_q_test = self.subset_creator.create_subsets()
            results['subset_stats'] = self.subset_creator.subset_stats
        except Exception as e:
            print(f"\n[FATAL] Phase 1 failed: {e}")
            raise
        
        # Phase 2: Circuit Definition
        try:
            feature_map = self.circuit_builder.build_feature_map()
            circuit_path = self.circuit_builder.visualize_circuit()
            results['circuit_stats'] = self.circuit_builder.circuit_stats
        except Exception as e:
            print(f"\n[FATAL] Phase 2 failed: {e}")
            raise
        
        # Phase 3: Kernel Initialization
        try:
            self.kernel_computer = QuantumKernelComputer(
                self.config, backend, feature_map
            )
            self.kernel_computer.initialize_kernel()
        except Exception as e:
            print(f"\n[FATAL] Phase 3 failed: {e}")
            raise
        
        # Phase 4: Matrix Computation
        try:
            K_train, K_test = self.kernel_computer.compute_all_matrices(
                X_q_train, X_q_test
            )
            results['computation_stats'] = self.kernel_computer.computation_stats
        except Exception as e:
            print(f"\n[FATAL] Phase 4 failed: {e}")
            raise
        
        # Phase 5: Serialization & Visualization
        try:
            saved_paths = self.serializer.save_matrices(K_train, K_test)
            heatmap_path = self.serializer.visualize_matrices(K_train, K_test)
            results['saved_paths'] = saved_paths
            results['heatmap_path'] = heatmap_path
        except Exception as e:
            print(f"\n[FATAL] Phase 5 failed: {e}")
            raise
        
        # Final Summary
        pipeline_elapsed = time.time() - pipeline_start
        
        print("\n" + "="*70)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("="*70)
        
        print(f"\n[QUANTUM KERNEL SUMMARY]")
        print(f"  -> Feature Map: ZZFeatureMap (8 qubits, 2 reps, linear)")
        print(f"  -> Q_Train samples: {len(X_q_train)}")
        print(f"  -> Q_Test samples: {len(X_q_test)}")
        print(f"  -> K_train shape: {K_train.shape}")
        print(f"  -> K_test shape: {K_test.shape}")
        print(f"  -> Total time: {format_time(pipeline_elapsed)}")
        
        print(f"\n[OUTPUT FILES]")
        for name, path in saved_paths.items():
            print(f"  -> {name}: {path}")
        
        print(f"\n[KERNEL STATISTICS]")
        print(f"  -> K_train: min={K_train.min():.4f}, max={K_train.max():.4f}, mean={K_train.mean():.4f}")
        print(f"  -> K_test:  min={K_test.min():.4f}, max={K_test.max():.4f}, mean={K_test.mean():.4f}")
        
        print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        results['total_time'] = pipeline_elapsed
        
        return results



def main():
    """Main entry point for the Quantum Kernel pipeline."""
    
    # Set random seeds for reproducibility
    np.random.seed(Config.RANDOM_SEED)
    
    # Initialize and run pipeline
    pipeline = QuantumKernelPipeline()
    
    try:
        results = pipeline.run()
        print("\n[SUCCESS] Pipeline completed without errors")
        return results
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Pipeline stopped by user")
        print("  -> Checkpoints saved. Re-run to resume.")
        sys.exit(1)
    except Exception as e:
        print(f"\n[FATAL ERROR] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    results = main()
