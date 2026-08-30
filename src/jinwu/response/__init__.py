"""Response matrix and effective area utilities.

Submodules:
    - basic: Basic response utilities

Note:
    GBM-specific response handling moved to ``jinwu.fermi.gbm.response``.
"""

from __future__ import annotations

try:
    from . import basic
except ImportError:
    pass

__all__ = [
    'basic',
]
