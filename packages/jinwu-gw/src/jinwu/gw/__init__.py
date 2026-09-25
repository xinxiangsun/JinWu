"""Gravitational-wave localization and high-energy coverage tools."""

from .models import (
    CoverageResult,
    GWEvent,
    GWRunResult,
    Localization,
    SkyFootprint,
    SkyMap,
)
from .skymap import (
    ProbabilityIntegral,
    credible_region,
    credible_region_stats,
    load_skymap,
    probability_in_footprint,
    refined_probability,
    sky_map_footprint,
)
from .layers import load_layers, footprint_from_circle, footprint_from_polygon
from .alert import fetch_skymap_url, fetch_superevent, read_notice, skymap_from_notice
from .coverage import gbm_coverage_for_skymap
from .gracedb import GraceDBClient, GraceDBNotice, normalize_superevent_id
from .plot import plot_allsky, plot_gbm_diagnostic
from .pipeline import GWConfig, GWPipeline, GWPipelineInput, run_gw_pipeline

__all__ = [
    "CoverageResult",
    "GWEvent",
    "GWRunResult",
    "Localization",
    "SkyFootprint",
    "SkyMap",
    "credible_region",
    "credible_region_stats",
    "ProbabilityIntegral",
    "load_skymap",
    "probability_in_footprint",
    "refined_probability",
    "sky_map_footprint",
    "read_notice",
    "fetch_superevent",
    "fetch_skymap_url",
    "skymap_from_notice",
    "load_layers",
    "footprint_from_circle",
    "footprint_from_polygon",
    "gbm_coverage_for_skymap",
    "GraceDBClient",
    "GraceDBNotice",
    "normalize_superevent_id",
    "plot_allsky",
    "plot_gbm_diagnostic",
    "GWConfig",
    "GWPipeline",
    "GWPipelineInput",
    "run_gw_pipeline",
]

__version__ = "0.2.0"
