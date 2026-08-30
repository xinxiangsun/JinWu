"""Compatibility shim: module renamed to :mod:`jinwu.swift.bat.bat_observation`.

New code should use ``from jinwu.swift.bat import BATObservation``.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "The jinwu.swift.bat.BATObservation module is deprecated; import "
    "BATObservation from jinwu.swift.bat or jinwu.swift.bat.bat_observation",
    DeprecationWarning,
    stacklevel=2,
)

from .bat_observation import BATObservation  # noqa: F401

__all__ = ["BATObservation"]
