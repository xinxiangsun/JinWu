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

from .urls import generate_download_url
from .pipeline import (
    GBMBackgroundSummary,
    GBMCoverageResult,
    GBMPosHistSelection,
    GBMDataManifest,
    GBMDetectorSelection,
    GBMFlareInterval,
    GBMPipeline,
    GBMPipelineInput,
    GBMPipelineResult,
    GBMSpectralProducts,
    background_windows,
    check_gbm_coverage,
    estimate_gbm_orbit_period,
    extract_gbm_spectral_products,
    integrate_background_interval,
    validate_background_residuals,
    gti_intervals_for_paths,
    fetch_gbm_continuous_products,
    fetch_gbm_products_for_interval,
    find_gbm_poshist,
    main,
    powerlaw_energy_flux_per_norm,
    read_ogip_products,
    select_gbm_detectors,
    single_response,
    tte_coverage,
)
from .poshist import GBMGeometryState, fetch_poshist_for_time, read_gbm_geometry


def __getattr__(name):
    # The observation interface imports optional GDT localization machinery.
    # Keep headless pipeline configuration usable without that stack.
    if name == "GBMObservation":
        from .gbm_observation import GBMObservation
        globals()[name] = GBMObservation
        return GBMObservation
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    'GBMObservation',
    'generate_download_url',
    'GBMBackgroundSummary',
    'GBMCoverageResult',
    'GBMPosHistSelection',
    'GBMDataManifest',
    'GBMDetectorSelection',
    'GBMFlareInterval',
    'GBMPipeline',
    'GBMPipelineInput',
    'GBMPipelineResult',
    'GBMSpectralProducts',
    'GBMGeometryState',
    'background_windows',
    'check_gbm_coverage',
    'estimate_gbm_orbit_period',
    'extract_gbm_spectral_products',
    'integrate_background_interval',
    'validate_background_residuals',
    'gti_intervals_for_paths',
    'fetch_gbm_continuous_products',
    'fetch_gbm_products_for_interval',
    'fetch_poshist_for_time',
    'find_gbm_poshist',
    'main',
    'powerlaw_energy_flux_per_norm',
    'read_gbm_geometry',
    'read_ogip_products',
    'select_gbm_detectors',
    'single_response',
    'tte_coverage',
]
