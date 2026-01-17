# Processed Data

This directory contains the preprocessed numpy arrays ready for quantum kernel computation.

## File Descriptions

### SWaT Dataset (root level)

| File | Description | Shape |
|------|-------------|-------|
| X_train.npy | Full training features (all 51 features) | (N, 51) |
| y_train.npy | Training labels | (N,) |
| X_test.npy | Full test features (all 51 features) | (M, 51) |
| y_test.npy | Test labels | (M,) |
| X_train_reduced.npy | Training with selected 8 features | (N, 8) |
| X_test_reduced.npy | Test with selected 8 features | (M, 8) |
| y_train_reduced.npy | Training labels (same as y_train) | (N,) |
| y_test_reduced.npy | Test labels (same as y_test) | (M,) |
| X_q_train.npy | Quantum training subset | (2500, 8) |
| y_q_train.npy | Quantum training labels | (2500,) |
| X_q_test.npy | Quantum test subset | (1000, 8) |
| y_q_test.npy | Quantum test labels | (1000,) |
| gram_matrix_train.npy | Training kernel matrix | (2500, 2500) |
| gram_matrix_test.npy | Test kernel matrix | (1000, 2500) |
| feature_names.joblib | List of all feature names | 51 strings |
| selected_feature_names.joblib | List of selected features | 8 strings |

### HAI Dataset (HAI/ subdirectory)

Same structure as above but with HAI-specific data.

## Generating These Files

Run the pipeline stages in order:

```bash
# Stage 1: Data ingestion (creates X_train, X_test, etc.)
python src/swat_data_ingestion.py

# Stage 2: Feature selection (creates *_reduced files)
python src/swat_feature_selection.py

# Stage 3: Quantum kernel (creates X_q_*, gram_matrix_*)
python src/quantum_kernel_computation.py
```

## Notes on Data Leakage Prevention

- The scaler is fitted only on training data
- Feature selection uses test data (training has no attacks)
- Quantum subsets are created from deduplicated test data
- Train and test subsets are verified to be disjoint

## File Sizes

Approximate sizes for reference:

- Full arrays: 10-50 MB each depending on sample count
- Reduced arrays: 1-5 MB each
- Kernel matrices: 25-50 MB each (float64)
