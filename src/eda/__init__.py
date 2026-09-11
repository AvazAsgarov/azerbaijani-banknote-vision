"""Exploratory Data Analysis and visual manifold diagnostic package.

This module exports quantitative evaluation tools for latent feature projections,
spatial bounding box distributions, photometric signal profiles, cross-partition
leakage diagnostics, and fine-grained classification confusion patterns.
"""

from .chromatic import ChromaticMetrics, ForegroundChromaticAnalyzer
from .deep_dives import DeepDiveAnalyzer, DeepDiveMetrics
from .integrity import IntegrityAnalyzer, IntegrityMetrics
from .manifold import ManifoldAnalyzer, ManifoldMetrics
from .photometric import PhotometricAnalyzer, PhotometricMetrics
from .spatial import SpatialAnalyzer, SpatialMetrics
from .style import EDADesignSystem

__all__ = [
    "EDADesignSystem",
    "ManifoldAnalyzer",
    "ManifoldMetrics",
    "PhotometricAnalyzer",
    "PhotometricMetrics",
    "SpatialAnalyzer",
    "SpatialMetrics",
    "IntegrityAnalyzer",
    "IntegrityMetrics",
    "DeepDiveAnalyzer",
    "DeepDiveMetrics",
    "ForegroundChromaticAnalyzer",
    "ChromaticMetrics",
]
