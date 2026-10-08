"""Experimental spectral component interfaces with XSPEC-inspired semantics.

The existing public classes are retained for user-defined ``evaluate`` methods.
A native JinWu likelihood or fitting backend is a future feature; these
interfaces alone do not provide spectral fitting.

This module provides:
- ModelBase: Abstract base class for all model components
- AdditiveModel: Models that produce spectra to be summed
- MultiplicativeModel: Models that modify spectra (absorption, etc.)
- ConvolutionModel: Models that convolve input spectra

Example:
    >>> from jinwu.model import AdditiveModel, MultiplicativeModel
    >>> class PowerLaw(AdditiveModel):
    ...     def evaluate(self, energy, **kwargs):
    ...         return energy ** (-self.params['gamma'])
"""

from __future__ import annotations

from .modelbase import (
    ModelBase,
    AdditiveModel,
    MultiplicativeModel,
    ConvolutionModel,
)

__all__ = [
    'ModelBase',
    'AdditiveModel',
    'MultiplicativeModel',
    'ConvolutionModel',
]
