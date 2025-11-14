"""
autohea.core
============

Core utilities for OGIP FITS IO and helpers.

This package layer exposes:
- Submodules such as `file`, `heasoft`, `plot`, `time`.
- Numpy-first OGIP readers in `core.file` for ARF/RMF/PHA/LC/EVT, returning
  concrete dataclasses with `kind` and `path` fields.

Typical usage
-------------
	from autohea.core import read_fits, guess_ogip_kind
	from autohea.core import read_arf, read_pha, OgipPhaReader
	from autohea.core import band_from_arf_bins, ChannelBand
"""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

# Submodules re-export
from . import file as file
from . import heasoft as heasoft
from . import plot as plot
from . import time as time

# Selected public API re-export from core.file
from .file import (
	# Data containers
	EnergyBand, ChannelBand,
	ArfData, RmfData, PhaData, LightcurveData, EventData,
	# Readers
	OgipArfReader, OgipRmfReader, OgipPhaReader, OgipLightcurveReader, OgipEventReader,
	# Aliases
	ArfReader, RmfReader, RspReader, LightcurveReader,
	# Utilities
	band_from_arf_bins, channel_mask_from_ebounds,
	# Unified helpers (direct dataclass returning)
	OgipData, guess_ogip_kind, read_fits, read_arf, read_rmf, read_pha, read_lc, read_evt,
)


# Package version
try:
	__version__ = version("autohea")
except PackageNotFoundError:  # pragma: no cover - during editable installs
	__version__ = "0.0.3"

__all__ = [
	# Submodules
	'file', 'heasoft', 'plot', 'time',
	# Data containers
	'EnergyBand', 'ChannelBand', 'ArfData', 'RmfData', 'PhaData', 'LightcurveData', 'EventData',
	# Readers
	'OgipArfReader', 'OgipRmfReader', 'OgipPhaReader', 'OgipLightcurveReader', 'OgipEventReader',
	# Aliases
	'ArfReader', 'RmfReader', 'RspReader', 'LightcurveReader',
	# Utilities
	'band_from_arf_bins', 'channel_mask_from_ebounds',
	# Unified helpers
	'OgipData', 'guess_ogip_kind', 'read_fits', 'read_arf', 'read_rmf', 'read_pha', 'read_lc', 'read_evt',
	# Package meta
	'__version__',
]