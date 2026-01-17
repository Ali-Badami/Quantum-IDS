
import os
from pathlib import Path


def get_project_root():
    """
    Determine the project root directory.
    
    Returns the parent directory of wherever this config file is located.
    Works whether running from src/ or from the project root.
    """
    # Start from this file's location
    current_file = Path(__file__).resolve()
    
    # If in src/, go up one level
    if current_file.parent.name == "src":
        return current_file.parent.parent
    
    # Otherwise assume we're at project root
    return current_file.parent


def is_colab():
    """Check if running in Google Colab environment."""
    try:
        import google.colab
        return True
    except ImportError:
        return False


class LocalConfig:
    """Configuration for local development."""
    
    def __init__(self, project_root=None):
        if project_root is None:
            project_root = get_project_root()
        
        self.PROJECT_ROOT = project_root
        
        # Data directories
        self.DATA_DIR = os.path.join(project_root, "data")
        self.SWAT_RAW_DIR = os.path.join(self.DATA_DIR, "swat")
        self.HAI_RAW_DIR = os.path.join(self.DATA_DIR, "hai")
        self.PROCESSED_DIR = os.path.join(self.DATA_DIR, "processed")
        self.HAI_PROCESSED_DIR = os.path.join(self.PROCESSED_DIR, "HAI")
        
        # Raw data paths
        self.NORMAL_DATA_PATH = os.path.join(self.SWAT_RAW_DIR, "normal.csv")
        self.ATTACK_DATA_PATH = os.path.join(self.SWAT_RAW_DIR, "attack.csv")
        
        # Processed data paths (SWaT)
        self.X_TRAIN_PATH = os.path.join(self.PROCESSED_DIR, "X_train.npy")
        self.Y_TRAIN_PATH = os.path.join(self.PROCESSED_DIR, "y_train.npy")
        self.X_TEST_PATH = os.path.join(self.PROCESSED_DIR, "X_test.npy")
        self.Y_TEST_PATH = os.path.join(self.PROCESSED_DIR, "y_test.npy")
        self.X_TRAIN_REDUCED_PATH = os.path.join(self.PROCESSED_DIR, "X_train_reduced.npy")
        self.X_TEST_REDUCED_PATH = os.path.join(self.PROCESSED_DIR, "X_test_reduced.npy")
        self.FEATURE_NAMES_PATH = os.path.join(self.PROCESSED_DIR, "feature_names.joblib")
        self.SELECTED_FEATURES_PATH = os.path.join(self.PROCESSED_DIR, "selected_feature_names.joblib")
        self.SCALER_PATH = os.path.join(self.PROCESSED_DIR, "scaler.joblib")
        
        # Results directories
        self.RESULTS_DIR = os.path.join(project_root, "results")
        self.PLOTS_DIR = os.path.join(self.RESULTS_DIR, "plots")
        self.LOGS_DIR = os.path.join(self.RESULTS_DIR, "logs")
        
        # Checkpoints
        self.CHECKPOINTS_DIR = os.path.join(project_root, "checkpoints")
        
        # Backups
        self.BACKUPS_DIR = os.path.join(project_root, "backups")
        
        # Manifest
        self.MANIFEST_PATH = os.path.join(project_root, "manifest.txt")
        
        # Quantum parameters
        self.Q_TRAIN_SIZE = 2500
        self.Q_TEST_SIZE = 1000
        self.N_QUBITS = 8
        self.FEATURE_MAP_REPS = 2
        self.ENTANGLEMENT = "linear"
        
        # Data processing parameters
        self.WARMUP_ROWS_TO_DROP = 21600  # 6 hours at 1 Hz
        self.TRAIN_SPLIT_RATIO = 0.80
        self.LABEL_COLUMN = "Normal/Attack"
        self.LABEL_MAPPING = {"Normal": 0, "Attack": 1}
        
        # Feature selection parameters
        self.SUBSAMPLE_SIZE = 50000
        self.N_FEATURES_TO_SELECT = 8
        self.N_FEATURES_TO_PLOT = 15
        self.RF_N_ESTIMATORS = 100
        self.RF_N_JOBS = -1
        self.RF_RANDOM_STATE = 42
        self.RF_MAX_DEPTH = 20
        self.RF_MIN_SAMPLES_LEAF = 50
        
        # Kernel computation parameters
        self.BATCH_SIZE = 100
        self.CHECKPOINT_INTERVAL = 500
        self.SIMULATOR_METHOD = "statevector"
        self.USE_GPU = True
        
        # Random seed
        self.RANDOM_SEED = 42
        
        # Visualization
        self.PLOT_DPI = 150
        self.PLOT_FIGSIZE = (12, 8)


class ColabConfig(LocalConfig):
    """Configuration for Google Colab environment."""
    
    def __init__(self):
        # Don't call parent init yet
        self.PROJECT_ROOT = "/content/drive/MyDrive/Quantum_Research"
        
        # Data directories
        self.DATA_DIR = self.PROJECT_ROOT
        self.SWAT_RAW_DIR = "/content/drive/MyDrive/Datasets/swat"
        self.HAI_RAW_DIR = "/content/drive/MyDrive/Datasets/hai"
        self.PROCESSED_DIR = os.path.join(self.PROJECT_ROOT, "02_Processed_Data")
        self.HAI_PROCESSED_DIR = os.path.join(self.PROCESSED_DIR, "HAI")
        
        # Raw data paths
        self.NORMAL_DATA_PATH = os.path.join(self.SWAT_RAW_DIR, "normal.csv")
        self.ATTACK_DATA_PATH = os.path.join(self.SWAT_RAW_DIR, "attack.csv")
        
        # Processed data paths (SWaT)
        self.X_TRAIN_PATH = os.path.join(self.PROCESSED_DIR, "X_train.npy")
        self.Y_TRAIN_PATH = os.path.join(self.PROCESSED_DIR, "y_train.npy")
        self.X_TEST_PATH = os.path.join(self.PROCESSED_DIR, "X_test.npy")
        self.Y_TEST_PATH = os.path.join(self.PROCESSED_DIR, "y_test.npy")
        self.X_TRAIN_REDUCED_PATH = os.path.join(self.PROCESSED_DIR, "X_train_reduced.npy")
        self.X_TEST_REDUCED_PATH = os.path.join(self.PROCESSED_DIR, "X_test_reduced.npy")
        self.FEATURE_NAMES_PATH = os.path.join(self.PROCESSED_DIR, "feature_names.joblib")
        self.SELECTED_FEATURES_PATH = os.path.join(self.PROCESSED_DIR, "selected_feature_names.joblib")
        self.SCALER_PATH = os.path.join(self.PROJECT_ROOT, "04_Models", "scaler.joblib")
        
        # Results directories
        self.RESULTS_DIR = os.path.join(self.PROJECT_ROOT, "05_Results")
        self.PLOTS_DIR = os.path.join(self.RESULTS_DIR, "Plots")
        self.LOGS_DIR = os.path.join(self.RESULTS_DIR, "Logs")
        
        # Checkpoints
        self.CHECKPOINTS_DIR = os.path.join(self.PROJECT_ROOT, "03_Checkpoints")
        
        # Backups
        self.BACKUPS_DIR = os.path.join(self.PROJECT_ROOT, "BACKUPS")
        
        # Manifest
        self.MANIFEST_PATH = os.path.join(self.PROJECT_ROOT, "project_manifest.txt")
        
        # Copy all the numeric parameters from LocalConfig
        self.Q_TRAIN_SIZE = 2500
        self.Q_TEST_SIZE = 1000
        self.N_QUBITS = 8
        self.FEATURE_MAP_REPS = 2
        self.ENTANGLEMENT = "linear"
        self.WARMUP_ROWS_TO_DROP = 21600
        self.TRAIN_SPLIT_RATIO = 0.80
        self.LABEL_COLUMN = "Normal/Attack"
        self.LABEL_MAPPING = {"Normal": 0, "Attack": 1}
        self.SUBSAMPLE_SIZE = 50000
        self.N_FEATURES_TO_SELECT = 8
        self.N_FEATURES_TO_PLOT = 15
        self.RF_N_ESTIMATORS = 100
        self.RF_N_JOBS = -1
        self.RF_RANDOM_STATE = 42
        self.RF_MAX_DEPTH = 20
        self.RF_MIN_SAMPLES_LEAF = 50
        self.BATCH_SIZE = 100
        self.CHECKPOINT_INTERVAL = 500
        self.SIMULATOR_METHOD = "statevector"
        self.USE_GPU = True
        self.RANDOM_SEED = 42
        self.PLOT_DPI = 150
        self.PLOT_FIGSIZE = (12, 8)


def get_config(project_root=None):
  
    if is_colab():
        return ColabConfig()
    else:
        return LocalConfig(project_root)


# For backwards compatibility, expose a default config instance
Config = LocalConfig
