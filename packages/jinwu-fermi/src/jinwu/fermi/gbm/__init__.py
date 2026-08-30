"""Fermi/GBM (Gamma-ray Burst Monitor) support.

This module provides utilities for working with GBM observations and data.

Classes:
    - GBMObservation: High-level GBM observation interface

Example:
    >>> from jinwu.fermi.gbm import GBMObservation
    >>> obs = GBMObservation('trigger_name')
"""

from __future__ import annotations

from .gbm_observation import GBMObservation
from .pipeline import (
    GBMBackgroundSummary,
    GBMCoverageResult,
    GBMDataManifest,
    GBMDetectorSelection,
    GBMFlareInterval,
    GBMSpectralProducts,
    check_gbm_coverage,
    extract_gbm_spectral_products,
    fetch_gbm_continuous_products,
    select_gbm_detectors,
)

__all__ = [
    'GBMObservation',
    'GBMBackgroundSummary',
    'GBMCoverageResult',
    'GBMDataManifest',
    'GBMDetectorSelection',
    'GBMFlareInterval',
    'GBMSpectralProducts',
    'check_gbm_coverage',
    'extract_gbm_spectral_products',
    'fetch_gbm_continuous_products',
    'select_gbm_detectors',
]
