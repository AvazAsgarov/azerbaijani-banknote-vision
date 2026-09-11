# Invariant & Property Tests (tests/invariants/)

## Purpose
Rigorous invariant checks verifying strict safety guarantees across dataset and models.

## Key Invariants
- `test_zero_leakage.py`: Confirms pairwise disjointness between train, val, and test splits.
- `test_spatial_bounds.py`: Validates bounding box coordinates reside strictly within [0, 1].
- `test_distribution_balance.py`: Verifies representation of all seven banknote classes across splits.
- `test_tensor_integrity.py`: Confirms HDF5 embeddings and numpy array integrity.
- `test_tinyml_invariants.py`: Guarantees model parameters and SRAM usage stay below MCU hardware limits.
