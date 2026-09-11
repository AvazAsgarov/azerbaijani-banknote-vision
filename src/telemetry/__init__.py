"""Telemetry and Lifecycle Tracking Package.

Standardized callbacks and stream emitters record hardware metrics,
durations, and validation milestones during training iterations.
"""

from .telemetry_hook import (
    DashboardTelemetryCallback,
    attach_telemetry_callbacks,
)

__all__ = [
    "DashboardTelemetryCallback",
    "attach_telemetry_callbacks",
]
