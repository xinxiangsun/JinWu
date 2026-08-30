"""Compatibility wrapper for SNR and lightcurve evaluation utilities.

New code should import from ``jinwu.core.utils``:

    >>> from jinwu.core.utils import li_ma_snr, LightcurveSNREvaluator
"""

from __future__ import annotations

from jinwu.core.utils import li_ma_snr, LightcurveSNREvaluator  # noqa: F401

__all__ = ["li_ma_snr", "LightcurveSNREvaluator"]
