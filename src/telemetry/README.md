# Telemetry & Monitoring (src/telemetry/)

## Purpose
Hardware resource profiling and real-time training metrics telemetry for compute nodes.

## Key Files
- `hardware_monitor.py`: Queries NVML for GPU utilization, VRAM usage, temperature, CPU, and RAM.
- `metrics_logger.py`: Thread-safe JSON and CSV experiment logger recording epoch dynamics and ETA.
