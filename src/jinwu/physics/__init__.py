"""Physics models and utilities for astrophysics.

This module provides:
- GeneralRelativity: GR effects (redshift, Doppler, etc.)
- Radiation: Radiation transfer and related utilities

Example:
    >>> from jinwu.physics import GeneralRelativity
    >>> gr = GeneralRelativity()
"""

from __future__ import annotations

from .gr import GeneralRelativity

try:
    from . import radiation
except ImportError:
    pass

__all__ = [
    'GeneralRelativity',
    'radiation',
]
