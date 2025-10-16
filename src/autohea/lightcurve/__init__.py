"""
autohea.lightcurve
===================

Lightcurve evaluation and trigger decision utilities.

This package exposes two small, focused modules:
- duration: SNR evaluation against a time window using Li & Ma, with
	Bayesian-blocks-based T0 detection and optional MC.
- trigger: Minimal, deterministic trigger checks (sliding/head/cumulative),
	ready for future MC extension.

Typical usage
-------------
		from autohea.lightcurve import LightcurveSNREvaluator, li_ma_snr
		from autohea.lightcurve import TriggerDecider, BackgroundSimple
"""

from __future__ import annotations

# Public API re-exports
from .duration import LightcurveSNREvaluator, li_ma_snr
from .trigger import TriggerDecider, BackgroundSimple

__all__ = [
		'LightcurveSNREvaluator', 'li_ma_snr',
		'TriggerDecider', 'BackgroundSimple',
]

