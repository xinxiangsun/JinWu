'''
Date: 2025-10-10 11:53:55
LastEditors: Xinxiang Sun sunxx@nao.cas.cn
LastEditTime: 2025-10-17 00:23:51
FilePath: /research/autohea/src/autohea/core/__init__.py
'''
"""
autohea.core
============

Core utilities for OGIP FITS IO and helpers.

This package layer exposes:
- Submodules such as `file`, `heasoft`, `rsp`, `arf`, `rmf`, `plot`, `time`.
- A unified, numpy-first OGIP readers API from `core.file` for ARF/RMF/PHA/LC/EVT.

Users typically import directly from here for convenience:
	from autohea.core import read_ogip, OgipUnifiedReader
	from autohea.core import OgipPhaReader, band_from_arf_bins, ChannelBand
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
	# Unified entry
	OgipUnifiedReader, OgipAnyData, OgipData, guess_ogip_kind, read_ogip,
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
	# Unified entry
	'OgipUnifiedReader', 'OgipAnyData', 'OgipData', 'guess_ogip_kind', 'read_ogip',
	# Package meta
	'__version__',
]