# Makefile for Quantum-IDS project
# Common operations for running the pipeline

.PHONY: all install clean data-swat data-hai kernel benchmark hardware

# Default target
all: install

# Install dependencies
install:
	pip install -r requirements.txt

# Install with GPU support
install-gpu:
	pip install -r requirements.txt
	pip install qiskit-aer-gpu

# Install in development mode
install-dev:
	pip install -e ".[dev]"

# Run SWaT data pipeline
data-swat:
	python src/swat_data_ingestion.py
	python src/swat_feature_selection.py

# Run HAI data pipeline
data-hai:
	python src/hai_data_ingestion.py
	python src/hai_feature_selection.py

# Run all data pipelines
data-all: data-swat data-hai

# Compute quantum kernels (SWaT)
kernel-swat:
	python src/quantum_kernel_computation.py

# Compute quantum kernels (HAI)
kernel-hai:
	python src/hai_quantum_kernel.py

# Compute all kernels
kernel-all: kernel-swat kernel-hai

# Run benchmark evaluation
benchmark:
	python src/statistical_robustness_suite.py

# Run hardware validation (requires IBM Quantum account)
hardware:
	python src/real_hardware_validation.py

# Run full pipeline for SWaT
full-swat: data-swat kernel-swat

# Run full pipeline for HAI
full-hai: data-hai kernel-hai

# Clean generated files (keeps raw data)
clean:
	rm -rf data/processed/*.npy
	rm -rf data/processed/*.joblib
	rm -rf data/processed/HAI/*.npy
	rm -rf data/processed/HAI/*.joblib
	rm -rf results/plots/*.png
	rm -rf results/logs/*.json
	rm -rf checkpoints/
	rm -rf __pycache__/
	rm -rf src/__pycache__/

# Clean everything including build artifacts
clean-all: clean
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .eggs/

# Run linting
lint:
	flake8 src/
	black src/ --check

# Format code
format:
	black src/

# Help
help:
	@echo "Quantum-IDS Makefile"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  install      Install dependencies"
	@echo "  install-gpu  Install with GPU support"
	@echo "  install-dev  Install in development mode"
	@echo "  data-swat    Run SWaT data pipeline"
	@echo "  data-hai     Run HAI data pipeline"
	@echo "  data-all     Run all data pipelines"
	@echo "  kernel-swat  Compute SWaT quantum kernels"
	@echo "  kernel-hai   Compute HAI quantum kernels"
	@echo "  kernel-all   Compute all quantum kernels"
	@echo "  benchmark    Run benchmark evaluation"
	@echo "  hardware     Run hardware validation"
	@echo "  full-swat    Run full SWaT pipeline"
	@echo "  full-hai     Run full HAI pipeline"
	@echo "  clean        Clean generated files"
	@echo "  clean-all    Clean everything"
	@echo "  lint         Run linting"
	@echo "  format       Format code"
	@echo "  help         Show this help"
