"""Background modeling and priors for X-ray spectral fitting.

This module provides:
- BackgroundPrior: Prior distributions for background parameters
- BackgroundCountsPosterior: Posterior computation from observed counts
- BackgroundSpectralPrior: Spectral priors for background modeling

Example:
    >>> from jinwu.background import BackgroundPrior
    >>> prior = BackgroundPrior(mean=10.0, std=2.0)
"""

from __future__ import annotations

from .backprior import (
    BackgroundPrior,
    BackgroundCountsPosterior,
    BackgroundSpectralPrior,
)

__all__ = [
    'BackgroundPrior',
    'BackgroundCountsPosterior',
    'BackgroundSpectralPrior',
]
