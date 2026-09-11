# Automated Test Suite (tests/)

## Purpose
Comprehensive automated testing suite verifying mathematical correctness,
data invariants, model architectures, and end-to-end pipeline integrity.

## Testing Tiers
- `unit/`: Fast, isolated tests verifying individual functions and classes.
- `invariants/`: Property-based checks verifying zero data leakage, coordinate bounds, and quantization.
- `integration/`: Multi-component tests verifying end-to-end execution flow.
- `conftest.py`: Shared pytest fixtures and test environment setup.
