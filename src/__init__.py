"""
Quantum-IDS: Quantum Kernel Feature Mapping for ICS Anomaly Detection

This package provides tools for detecting anomalies in Industrial Control Systems
using quantum machine learning techniques. The implementation uses an 8-qubit
ZZFeatureMap kernel validated on SWaT and HAI benchmark datasets.

Modules:
    swat_data_ingestion: Process raw SWaT dataset
    swat_feature_selection: Select discriminative features for SWaT
    hai_data_ingestion: Process raw HAI dataset
    hai_feature_selection: Select discriminative features for HAI
    quantum_kernel_computation: Compute quantum kernel matrices
    hai_quantum_kernel: HAI-specific kernel computation
    statistical_robustness_suite: Multi-seed validation
    real_hardware_validation: IBM Quantum hardware validation

Author: Shujaatali Badami
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Shujaatali Badami"
__email__ = "shujaatali@ieee.org"
