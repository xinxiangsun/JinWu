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
	from autohea.core import readfits, guess_ogip_kind
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
from . import ops as ops
from .datasets import *
from . import redshift
from .redshift import *
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
	OgipData, guess_ogip_kind, readfits, read_arf, read_rmf, read_pha, read_lc, read_evt,
)
# Operations from core.ops
from .ops import (
	slice_lightcurve, rebin_lightcurve,
	slice_pha, rebin_pha,
	slice_events, rebin_events_to_lightcurve,
)




# Package version
try:
	__version__ = version("autohea")
except PackageNotFoundError:  # pragma: no cover - during editable installs
	__version__ = "0.0.13"

__all__ = [
	# Submodules
	'file', 'heasoft', 'plot', 'time', 'ops',
	# Data containers
	'EnergyBand', 'ChannelBand', 'ArfData', 'RmfData', 'PhaData', 'LightcurveData', 'EventData',
	# Dataset containers
	'LightcurveDataset', 'SpectrumDataset', 'JointDataset',
	# Readers
	'OgipArfReader', 'OgipRmfReader', 'OgipPhaReader', 'OgipLightcurveReader', 'OgipEventReader',
	# Aliases
	'ArfReader', 'RmfReader', 'RspReader', 'LightcurveReader',
	# Utilities
	'band_from_arf_bins', 'channel_mask_from_ebounds',
	# Unified helpers
	'OgipData', 'guess_ogip_kind', 'readfits', 'read_arf', 'read_rmf', 'read_pha', 'read_lc', 'read_evt',
	# Operations
	'slice_lightcurve', 'rebin_lightcurve', 'slice_pha', 'rebin_pha', 'slice_events', 'rebin_events_to_lightcurve',
	# Dataset helper
	'netdata',
	# Package meta
	'__version__',
]