"""
Luminosity-function and high-redshift detectability utilities.

This package groups LF-facing calculations separately from low-level core I/O
and light-curve helpers.  New code should import redshift extrapolation and
high-z detectability tools from ``jinwu.lf``.
"""

from __future__ import annotations

from .detectability import DetectabilityResult, HighZDetectabilityEstimator
from .redshift import RedshiftTriggerExtrapolator
from .legacy_redshift import RedshiftExtrapolator
from .lcfake import (
    LCSimPairResult,
    LCSimResult,
    XspecConfig,
    build_fake_from_npz,
    build_fake_on_off_from_npz,
    generate_redshift_lightcurves,
    load_counts_npz,
    save_on_off_lightcurve,
    load_on_off_lightcurve,
)
from .specfake import KConfig, XspecKFactory, XspecSession, prepare_background_for_fakeit

__all__ = [
    "DetectabilityResult",
    "HighZDetectabilityEstimator",
    "RedshiftTriggerExtrapolator",
    "RedshiftExtrapolator",
    "XspecConfig",
    "LCSimResult",
    "LCSimPairResult",
    "build_fake_from_npz",
    "build_fake_on_off_from_npz",
    "generate_redshift_lightcurves",
    "load_counts_npz",
    "save_on_off_lightcurve",
    "load_on_off_lightcurve",
    "KConfig",
    "XspecSession",
    "XspecKFactory",
    "prepare_background_for_fakeit",
]
