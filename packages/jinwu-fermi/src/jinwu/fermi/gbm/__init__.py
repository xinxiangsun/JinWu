"""Fermi/GBM (Gamma-ray Burst Monitor) support.

This module provides utilities for working with GBM observations and data.

Classes:
    - GBMObservation: High-level GBM observation interface
    - GBMPipeline: Resumable continuous-data pipeline (``"fermi.gbm"``)

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
    GBMPipeline,
    GBMPipelineInput,
    GBMPipelineResult,
    GBMSpectralProducts,
    background_windows,
    check_gbm_coverage,
    extract_gbm_spectral_products,
    integrate_background_interval,
    validate_background_residuals,
    gti_intervals_for_paths,
    fetch_gbm_continuous_products,
    main,
    powerlaw_energy_flux_per_norm,
    read_ogip_products,
    select_gbm_detectors,
    single_response,
    tte_coverage,
)

__all__ = [
    'GBMObservation',
    'GBMBackgroundSummary',
    'GBMCoverageResult',
    'GBMDataManifest',
    'GBMDetectorSelection',
    'GBMFlareInterval',
    'GBMPipeline',
    'GBMPipelineInput',
    'GBMPipelineResult',
    'GBMSpectralProducts',
    'background_windows',
    'check_gbm_coverage',
    'extract_gbm_spectral_products',
    'integrate_background_interval',
    'validate_background_residuals',
    'gti_intervals_for_paths',
    'fetch_gbm_continuous_products',
    'main',
    'powerlaw_energy_flux_per_norm',
    'read_ogip_products',
    'select_gbm_detectors',
    'single_response',
    'tte_coverage',
]
