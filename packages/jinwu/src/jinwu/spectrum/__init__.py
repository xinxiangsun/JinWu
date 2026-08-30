"""Spectrum generation and synthesis utilities.

This module provides:
- XspecKFactory: XSPEC-based spectrum synthesis factory
- XspecSession: XSPEC wrapper for fakeit and model evaluation
- KConfig: Configuration for spectral K (rate/flux) calculations

Example:
    >>> from jinwu.spectrum import XspecKFactory
    >>> factory = XspecKFactory()
"""

from __future__ import annotations

try:
    from .specfake import (
        KConfig,
        XspecSession,
        XspecKFactory,
        prepare_background_for_fakeit,
    )
    __all__ = [
        'KConfig',
        'XspecSession',
        'XspecKFactory',
        'prepare_background_for_fakeit',
    ]
except ImportError:
    __all__ = []
