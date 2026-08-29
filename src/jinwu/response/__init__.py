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

from .gbm import (
    GBMResponseCommand,
    GBMResponseRun,
    build_gbm_response_command,
    contgbmrsp,
    generate_gbm_response,
)

try:
    from . import basic
except ImportError:
    pass

__all__ = [
    'GBMResponseCommand',
    'GBMResponseRun',
    'build_gbm_response_command',
    'contgbmrsp',
    'generate_gbm_response',
    'basic',
]
