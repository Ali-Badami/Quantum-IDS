# Contributing to Quantum-IDS

Thanks for your interest in contributing to this project. This document outlines how you can help improve the quantum intrusion detection system.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Create a new branch for your changes
4. Make your changes and test them
5. Submit a pull request

## Setting Up Your Development Environment

```bash
git clone https://github.com/YOUR_USERNAME/Quantum-IDS.git
cd Quantum-IDS
pip install -e ".[dev]"
```

This installs the package in editable mode along with development dependencies.

## Code Style

We follow PEP 8 guidelines with a few adjustments:

- Line length limit is 100 characters
- Use descriptive variable names, especially for quantum-related concepts
- Add docstrings to all public functions and classes
- Include type hints where practical

Run the linter before submitting:

```bash
flake8 src/
black src/ --check
```

## Testing

Before submitting a pull request, make sure your changes work:

1. Run the data ingestion pipeline on a small subset
2. Verify feature selection produces sensible results
3. Check that kernel computation completes without errors

Since the full pipeline requires large datasets, we appreciate if you can test on subsampled data and describe your testing approach in the PR.

## Types of Contributions

### Bug Reports

If you find a bug, please open an issue with:

- A clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Your environment (Python version, Qiskit version, OS)

### Feature Requests

Have an idea for improvement? Open an issue describing:

- What problem it solves
- Proposed approach
- Any relevant references or papers

### Code Contributions

Some areas where help would be appreciated:

- Error mitigation techniques for hardware execution
- Support for additional ICS datasets
- Performance optimizations for kernel computation
- Documentation improvements
- Unit tests

## Pull Request Process

1. Update the README if you're adding new features
2. Add yourself to the contributors list if you want
3. Make sure the code runs without errors
4. Describe your changes clearly in the PR description

## Questions?

Feel free to open an issue for any questions about the codebase or methodology.
