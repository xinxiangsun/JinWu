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

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from . import heasoft as heasoft
	from . import plot as plot
	from . import time as time
	from . import ops as ops
	from . import io as io
	from .base import (
		EnergyBand, ChannelBand, RegionArea, RegionAreaSet,
		HduHeader, FitsHeaderDump, OgipMeta,
		ArfBase, RmfBase, PhaBase, LightcurveDataBase, EventDataBase,
	)
	from .data import ArfData, RmfData, PhaData, LightcurveData, EventData, timescale
	from .datasets import LightcurveDataset, SpectrumDataset, JointDataset, netdata
	from .io import (
		OgipArfReader, OgipRmfReader, OgipPhaReader, OgipLightcurveReader, OgipEventReader,
		ArfReader, RmfReader, RspReader, LightcurveReader,
		PhaWriter, ArfWriter, RmfWriter, LightcurveWriter, EventWriter,
		OgipData, guess_ogip_kind, readfits, read_arf, read_rmf, read_pha, read_lc, read_evt,
		write_arf, write_rmf, write_pha, write_lc, write_evt, writefits,
		band_from_arf_bins, channel_mask_from_ebounds,
	)
	from .ops import (
		slice_lightcurve, rebin_lightcurve, slice_pha, rebin_pha, slice_events, rebin_events_to_lightcurve,
	)

# Package version
try:
	__version__ = version("autohea")
except PackageNotFoundError:  # pragma: no cover - during editable installs
	__version__ = "0.0.13"

_MODULE_EXPORTS = {
	'heasoft', 'plot', 'time', 'ops', 'io', 'lf', 'redshift',
}

_BASE_EXPORTS = {
	'EnergyBand', 'ChannelBand', 'RegionArea', 'RegionAreaSet',
	'HduHeader', 'FitsHeaderDump', 'OgipMeta',
	'ArfBase', 'RmfBase', 'PhaBase', 'LightcurveDataBase', 'EventDataBase',
}

_DATA_EXPORTS = {
	'ArfData', 'RmfData', 'PhaData', 'LightcurveData', 'EventData', 'timescale',
}

_IO_EXPORTS = {
	'OgipArfReader', 'OgipRmfReader', 'OgipPhaReader', 'OgipLightcurveReader', 'OgipEventReader',
	'ArfReader', 'RmfReader', 'RspReader', 'LightcurveReader',
	'PhaWriter', 'ArfWriter', 'RmfWriter', 'LightcurveWriter', 'EventWriter',
	'OgipData', 'guess_ogip_kind', 'readfits', 'read_arf', 'read_rmf', 'read_pha', 'read_lc', 'read_evt',
	'write_arf', 'write_rmf', 'write_pha', 'write_lc', 'write_evt', 'writefits',
	'band_from_arf_bins', 'channel_mask_from_ebounds',
}

_OPS_EXPORTS = {
	'slice_lightcurve', 'rebin_lightcurve', 'slice_pha', 'rebin_pha', 'slice_events', 'rebin_events_to_lightcurve',
}

_DATASET_EXPORTS = {
	'LightcurveDataset', 'SpectrumDataset', 'JointDataset', 'netdata',
}

__all__ = [
	# Submodules
	'heasoft', 'plot', 'time', 'ops', 'io', 'lf', 'redshift',
	# Data containers
	'EnergyBand', 'ChannelBand', 'RegionArea', 'RegionAreaSet', 'HduHeader', 'FitsHeaderDump', 'OgipMeta', 'ArfBase', 'RmfBase', 'PhaBase', 'ArfData', 'RmfData', 'PhaData', 'LightcurveDataBase', 'LightcurveData', 'EventDataBase', 'EventData', 'timescale',
	# Dataset containers
	'LightcurveDataset', 'SpectrumDataset', 'JointDataset',
	# Readers
	'OgipArfReader', 'OgipRmfReader', 'OgipPhaReader', 'OgipLightcurveReader', 'OgipEventReader',
	# Aliases
	'ArfReader', 'RmfReader', 'RspReader', 'LightcurveReader',
	# Utilities
	'band_from_arf_bins', 'channel_mask_from_ebounds',
	# Unified helpers
	'PhaWriter', 'ArfWriter', 'RmfWriter', 'LightcurveWriter', 'EventWriter',
	'OgipData', 'guess_ogip_kind', 'readfits', 'read_arf', 'read_rmf', 'read_pha', 'read_lc', 'read_evt',
	'write_arf', 'write_rmf', 'write_pha', 'write_lc', 'write_evt', 'writefits',
	# Operations
	'slice_lightcurve', 'rebin_lightcurve', 'slice_pha', 'rebin_pha', 'slice_events', 'rebin_events_to_lightcurve',
	# Dataset helper
	'netdata',
	# Package meta
	'__version__',
]


def __getattr__(name: str):
	if name in _MODULE_EXPORTS:
		mod = import_module(f'.{name}', __name__)
		globals()[name] = mod
		return mod

	if name in _BASE_EXPORTS:
		mod = import_module('.base', __name__)
		value = getattr(mod, name)
		globals()[name] = value
		return value

	if name in _DATA_EXPORTS:
		mod = import_module('.data', __name__)
		value = getattr(mod, name)
		globals()[name] = value
		return value

	if name in _IO_EXPORTS:
		mod = import_module('.io', __name__)
		value = getattr(mod, name)
		globals()[name] = value
		return value

	if name in _OPS_EXPORTS:
		mod = import_module('.ops', __name__)
		value = getattr(mod, name)
		globals()[name] = value
		return value

	if name in _DATASET_EXPORTS:
		mod = import_module('.datasets', __name__)
		value = getattr(mod, name)
		globals()[name] = value
		return value

	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
	return sorted(set(globals().keys()) | set(__all__))
