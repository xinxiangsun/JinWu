"""Light-curve evaluation and trigger decision utilities.

This module provides tools for light-curve analysis including:
- SNR evaluation using Li & Ma statistics
- Bayesian blocks and peak detection
- Trigger decision logic (sliding, head, cumulative)
- Light-curve synthesis and fake data generation

Main Classes:
    - LightcurveSNREvaluator: Compute SNR against time windows
    - TriggerDecider: Determine trigger decisions
    - LightcurveFaker: Generate synthetic light curves

Functions:
    - li_ma_snr: Compute Li & Ma SNR statistic

Example:
    >>> from jinwu.lightcurve import LightcurveSNREvaluator, li_ma_snr
    >>> evaluator = LightcurveSNREvaluator(lightcurve, background_prior)
    >>> snr = li_ma_snr(n_on=100, n_off=50, alpha=1.0)
"""

from __future__ import annotations

from .duration import (
    li_ma_snr,
    LightcurveSNREvaluator,
)

from .trigger import (
    BackgroundSimple,
    TriggerDecider,
)

try:
    from .lcfake import LightcurveFaker
except ImportError:
    LightcurveFaker = None

__all__ = [
    'li_ma_snr',
    'LightcurveSNREvaluator',
    'BackgroundSimple',
    'TriggerDecider',
    'LightcurveFaker',
]


# Public API re-exports
from .duration import LightcurveSNREvaluator, li_ma_snr
from .trigger import TriggerDecider, BackgroundSimple

__all__ = [
		'LightcurveSNREvaluator', 'li_ma_snr',
		'TriggerDecider', 'BackgroundSimple',
]

