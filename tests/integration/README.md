# Integration Tests (tests/integration/)

## Purpose
End-to-end pipeline tests verifying interaction across multiple modules from data ingestion
to export and reporting.

## Test Areas
- `test_data_pipeline.py`: Full pass from raw files to stratified splits and manifest.
- `test_tinyml_pipeline.py`: Model initialization, forward pass, quantization, and C++ header generation.
- `test_invalidation.py`: Cache invalidation when input data or configuration changes.
