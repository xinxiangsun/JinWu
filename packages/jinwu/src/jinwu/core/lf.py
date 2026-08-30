"""Compatibility wrapper for luminosity-function utilities.

New code should import from ``jinwu.lf`` or ``jinwu.lf.detectability``.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "jinwu.core.lf is deprecated and will be removed in 1.0; import from "
    "jinwu.lf.detectability instead",
    DeprecationWarning,
    stacklevel=2,
)

from jinwu.lf.detectability import *  # noqa: F401,F403
