"""Compatibility shim: module renamed to :mod:`jinwu.fermi.gbm.gbm_observation`.

New code should use ``from jinwu.fermi.gbm import GBMObservation``.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "The jinwu.fermi.gbm.GBMObservation module is deprecated; import "
    "GBMObservation from jinwu.fermi.gbm or jinwu.fermi.gbm.gbm_observation",
    DeprecationWarning,
    stacklevel=2,
)

from .gbm_observation import GBMObservation  # noqa: F401

__all__ = ["GBMObservation"]
