"""Fermi mission utilities.

This module provides mission-specific helpers for Fermi observations,
including GBM (Gamma-ray Burst Monitor) support.

Submodules:
    - gbm: GBM-specific observation handling
"""

from __future__ import annotations

try:
    from . import gbm
    __all__ = ['gbm']
except ImportError:
    __all__ = []
