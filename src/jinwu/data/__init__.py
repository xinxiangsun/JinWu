"""High-level, standardized data containers for JinWu.

This subpackage re-exports dataset abstractions from `jinwu.core.datasets`
(light curves, spectra, joint datasets) built on top of the low-level OGIP
dataclasses in `jinwu.core.file`. They are designed to make common operations
like background subtraction, grouping, slicing, and merging easier and
more uniform across instruments.
"""

from jinwu.core.datasets import (
    LightcurveDataset,
    SpectrumDataset,
    JointDataset,
    netdata,
)

__all__ = [
    "LightcurveDataset",
    "SpectrumDataset",
    "JointDataset",
    "netdata",
]
