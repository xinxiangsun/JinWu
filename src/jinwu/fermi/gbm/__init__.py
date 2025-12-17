"""Fermi/GBM (Gamma-ray Burst Monitor) support.

This module provides utilities for working with GBM observations and data.

Classes:
    - GBMObservation: High-level GBM observation interface

Example:
    >>> from jinwu.fermi.gbm import GBMObservation
    >>> obs = GBMObservation('trigger_name')
"""

from __future__ import annotations

from .GBMObservation import GBMObservation

__all__ = ['GBMObservation']
