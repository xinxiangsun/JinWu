"""Compatibility wrapper for redshift extrapolation utilities.

New code should import from ``jinwu.lf.redshift``.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "jinwu.core.redshift is deprecated and will be removed in 1.0; import from "
    "jinwu.lf.redshift instead",
    DeprecationWarning,
    stacklevel=2,
)

from jinwu.lf.redshift import *  # noqa: F401,F403
