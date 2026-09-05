"""Physics models and utilities for astrophysics.

This module provides:
- GeneralRelativity: special-relativistic Doppler/boost kinematics
  (experimental; cosmological redshift and GR metric effects are
  NOT implemented)
- Radiation: Radiation transfer and related utilities

Example:
    >>> from jinwu.physics import GeneralRelativity
    >>> gr = GeneralRelativity()
"""

from __future__ import annotations

from .gr import GeneralRelativity

_physics_all = [
    'GeneralRelativity',
]

try:
    from . import radiation
    _physics_all.append('radiation')
except ImportError:
    pass

__all__ = _physics_all
