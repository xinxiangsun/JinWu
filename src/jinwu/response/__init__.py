"""Response matrix and effective area utilities.

This module provides:
- contgbmrsp: Continuous (unbinned) GBM response function

Submodules:
    - gbm: GBM-specific response handling
    - basic: Basic response utilities

Example:
    >>> from jinwu.response import contgbmrsp
"""

from __future__ import annotations

from .gbm import contgbmrsp

try:
    from . import basic
except ImportError:
    pass

__all__ = [
    'contgbmrsp',
    'basic',
]

