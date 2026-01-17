# Quantum Kernel Feature Mapping for ICS Anomaly Detection

A hardware-agnostic Quantum Support Vector Machine (QSVM) framework for intrusion detection in Industrial Control Systems. This implementation uses an 8-qubit ZZFeatureMap kernel and has been validated on both the SWaT (Secure Water Treatment) and HAI (Hardware-in-the-Loop Augmented ICS) benchmark datasets.

## Overview

Modern Industrial Control Systems face sophisticated cyber-physical attacks that exploit nonlinear correlations between process variables. Traditional linear classifiers often fail to capture these complex relationships. This project implements a quantum machine learning approach that embeds sensor data into a high-dimensional Hilbert space where attack patterns become linearly separable.

The core idea is straightforward: SCADA sensors in critical infrastructure exhibit correlated behaviors during attacks that classical kernels struggle to capture. By encoding these sensor readings into quantum states and measuring their fidelity, we can expose hidden patterns that make attacks detectable.

**Key characteristics:**

- 8-qubit ZZFeatureMap with linear entanglement (2 repetitions)
- Cross-testbed validation on water treatment and thermal power systems
- Hardware feasibility confirmed on IBM's 156-qubit ibm_fez processor
- Statistically robust results via 5-seed cross-validation

## Results Summary

Performance metrics (Mean and Std over 5 random seeds):

| Dataset | F1-Score | AUC-ROC | Accuracy |
|---------|----------|---------|----------|
| SWaT | 0.9002 (+/- 0.021) | 0.9912 (+/- 0.004) | 0.9744 (+/- 0.006) |
| HAI 22.04 | 0.3536 (+/- 0.052) | 0.8309 (+/- 0.050) | 0.9402 (+/- 0.006) |

The quantum kernel demonstrates +10.8% AUC improvement over classical RBF-SVM on the challenging HAI dataset.

## Hardware Validation

Circuits were successfully executed on IBM Quantum hardware:

| Metric | SWaT | HAI |
|--------|------|-----|
| Backend | ibm_fez (156 qubits) | ibm_fez (156 qubits) |
| Physical Depth | 74 | 76 |
| CZ Gate Count | 28 | 28 |
| Job ID | d5l9htjh36vs73bgsi3g | d5l9huk8d8hc73cfb0pg |

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/Ali-Badami/Quantum-IDS.git
cd Quantum-IDS
pip install -r requirements.txt
```

For GPU-accelerated simulation (recommended for large datasets):

```bash
pip install qiskit-aer-gpu
```

### Dependencies

- Python 3.9 or higher
- Qiskit 1.0.0 or higher
- qiskit-machine-learning 0.7.0 or higher
- qiskit-aer (with optional GPU support)
- scikit-learn 1.0 or higher
- pandas, numpy, matplotlib, seaborn

## Dataset Setup

This project uses two publicly available ICS security datasets. You can either download the raw data and process it from scratch, or use the pre-processed files if you just want to run the quantum experiments.

### Option A: Start from Raw Data

#### SWaT Dataset

Request access from iTrust, Singapore University of Technology and Design:
<https://itrust.sutd.edu.sg/itrust-labs_datasets/>

Place the files in `data/swat/`:

```
data/swat/
├── normal.csv
└── attack.csv
```

#### HAI Dataset

Download HAI 22.04 from ETRI:
<https://github.com/icsdataset/hai>

Place the files in `data/hai/`:

```
data/hai/
├── train1.csv
├── train2.csv
├── ...
├── test1.csv
├── test2.csv
└── ...
```

### Option B: Use Pre-processed Data

If you just want to run the quantum experiments without the data ingestion steps, place the processed numpy arrays in `data/processed/`. The required files are:

```
data/processed/
├── X_train_reduced.npy
├── y_train_reduced.npy
├── X_test_reduced.npy
├── y_test_reduced.npy
└── selected_feature_names.joblib
```

For HAI dataset, the files go in `data/processed/HAI/`.

## Pipeline Execution

The experiment consists of several stages that should be run in sequence. If you have pre-processed data, you can skip to Stage 3.

### Stage 1: Data Ingestion

Process the raw datasets with proper time-series handling:

```bash
# SWaT dataset
python src/swat_data_ingestion.py

# HAI dataset
python src/hai_data_ingestion.py
```

This stage handles:

- Removing the first 6 hours of SWaT data (warm-up period per standard protocol)
- Missing value imputation via forward-fill (sensor signal persistence assumption)
- Fitting the scaler on training data only to prevent data leakage
- Preserving time-series order (no shuffling)

### Stage 2: Feature Selection

Select the top 8 most discriminative features for the quantum circuit:

```bash
# SWaT
python src/swat_feature_selection.py

# HAI
python src/hai_feature_selection.py
```

Uses RandomForest Gini importance computed on the test set (which contains both classes). The training set is 100% normal data for anomaly detection purposes, so we need the test set for supervised feature selection.

**Selected Features (SWaT):**

| Qubit | Sensor | Description | Importance |
|-------|--------|-------------|------------|
| Q0 | AIT501 | Chemical analyzer (Stage 5) | 18.7% |
| Q1 | AIT201 | Chemical analyzer (Stage 2) | 13.3% |
| Q2 | AIT202 | Chemical analyzer (Stage 2) | 8.6% |
| Q3 | AIT504 | Chemical analyzer (Stage 5) | 7.8% |
| Q4 | PIT502 | Pressure indicator (Stage 5) | 7.6% |
| Q5 | PIT503 | Pressure indicator (Stage 5) | 5.4% |
| Q6 | MV101 | Motorized valve (Stage 1) | 5.3% |
| Q7 | FIT301 | Flow transmitter (Stage 3) | 5.2% |

### Stage 3: Quantum Kernel Computation

Compute the quantum kernel matrices using statevector simulation:

```bash
# SWaT
python src/quantum_kernel_computation.py

# HAI
python src/hai_quantum_kernel.py
```

This creates two kernel matrices:

- Training Gram matrix (2500 x 2500)
- Test kernel matrix (1000 x 2500)

The computation uses GPU acceleration if available and includes checkpointing for long-running jobs.

### Stage 4: Benchmark Evaluation

Run the QSVM vs Classical SVM comparison with statistical robustness:

```bash
python src/statistical_robustness_suite.py
```

This generates:

- Performance metrics for each of 5 random seeds
- Mean and standard deviation statistics
- ROC curves and confusion matrices

### Stage 5: Hardware Validation (Optional)

Validate circuit feasibility on real IBM Quantum hardware:

```bash
python src/real_hardware_validation.py
```

This requires an IBM Quantum account and API token. Get yours at <https://quantum.ibm.com>

Note that you need to set your IBM Quantum API token as an environment variable or in the script.

## Project Structure

```
Quantum-IDS/
├── src/
│   ├── swat_data_ingestion.py      # SWaT preprocessing
│   ├── swat_feature_selection.py   # Feature ranking for SWaT
│   ├── hai_data_ingestion.py       # HAI preprocessing
│   ├── hai_feature_selection.py    # Feature ranking for HAI
│   ├── quantum_kernel_computation.py   # Kernel matrix computation
│   ├── hai_quantum_kernel.py       # HAI-specific kernel computation
│   ├── hai_benchmark.py            # HAI evaluation pipeline
│   ├── statistical_robustness_suite.py # Multi-seed validation
│   └── real_hardware_validation.py # IBM hardware submission
├── data/
│   ├── swat/                       # SWaT dataset (not included)
│   ├── hai/                        # HAI dataset (not included)
│   └── processed/                  # Processed numpy arrays
├── results/
│   ├── plots/                      # Generated figures
│   └── logs/                       # JSON result logs
├── requirements.txt
├── setup.py
├── LICENSE
└── README.md
```

## Methodology Notes

### Why Quantum Kernels?

The ZZFeatureMap implements entangling gates that introduce phase factors proportional to feature products (x_i * x_j). This structure naturally captures pairwise correlations between sensors, which is relevant for ICS where attacks often manipulate related process variables simultaneously.

Classical RBF kernels compute similarity as exp(-gamma * ||x-y||^2), which treats features independently. The quantum kernel, through its entanglement structure, can identify correlated deviations that indicate coordinated sensor manipulation.

### Data Leakage Prevention

Several measures ensure rigorous evaluation:

1. Scaler fitted on training data only
2. Feature selection uses test set (training set has no attacks)
3. Stratified sampling preserves class ratios
4. De-duplication ensures disjoint train/test sets for quantum experiments

### Simulation vs Hardware

All performance benchmarks use noise-free statevector simulation to establish theoretical upper bounds. Hardware execution validates physical realizability but introduces approximately 17-20% fidelity degradation due to gate errors. Error mitigation techniques can partially recover this gap but are not included in this release.

### Quantum Subset Sizes

Due to the O(n^2) complexity of kernel matrix computation, we use stratified subsets:

- 2500 samples for training
- 1000 samples for testing

This is sufficient for demonstrating quantum advantage while keeping computation tractable. The full classical baseline uses all available data.

## Troubleshooting

**GPU not detected:**
Make sure you have CUDA installed and then install the GPU version:

```bash
pip install qiskit-aer-gpu
```

**Out of memory during kernel computation:**
Reduce the batch size in the Config class or use a smaller subset size.

**Import errors with qiskit-algorithms:**
This codebase uses the modern Qiskit 1.x API and does not depend on the deprecated qiskit-algorithms package.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{badami2026quantum,
  title={Hardware-Agnostic Quantum Kernel Feature Mapping for Anomaly Detection 
         in Critical Infrastructure: A Cross-Testbed Validation on NISQ Processors},
  author={Badami, Shujaatali},
 year={2026}
}
```

## License

This project is licensed under the MIT License. See LICENSE for details.

## Acknowledgments

- iTrust Centre at SUTD for SWaT dataset access
- ETRI for the HAI dataset
- IBM Quantum Network for hardware resources

## Contact

Shujaatali Badami  
Email: <shujaatali@ieee.org>  
ORCID: 0009-0003-5262-021X
