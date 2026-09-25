"""
jinwu.core
==========

Core utilities for OGIP FITS IO, fitting, plotting and pipeline helpers.

This package layer exposes:
- Submodules such as `fit`, `plot`, `time`, `ops`, `io`, `products`.
- Numpy-first OGIP readers in `core.io` for ARF/RMF/PHA/LC/EVT, returning
  concrete dataclasses with `kind` and `path` fields.

Typical usage
-------------
	from jinwu.core import readfits, guess_ogip_kind
	from jinwu.core import read_arf, read_pha, OgipPhaReader
	from jinwu.core import band_from_arf_bins, ChannelBand
	from jinwu.core import fit   # fit.fit_prepared / fit.fit_xray_models
"""
from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from . import heasoft as heasoft
	from . import plot as plot
	from . import plotpanel as plotpanel
	from . import time as time
	from . import ops as ops
	from . import io as io
	from . import timescale as timescale
	from .base import (
		EnergyBand, ChannelBand, RegionArea, RegionAreaSet,
		HduHeader, FitsHeaderDump, OgipMeta,
		ArfBase, RmfBase, PhaBase, LightcurveDataBase, EventDataBase,
	)
	from .ogip import (
		ValidationReport, ValidationMessage, OgipFitsBase, check_response_compatibility,
	)
	from .data import ArfData, RmfData, PhaData, LightcurveData, EventData
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
	from .upperlimit import (
		GaussianNetRateObservation, GaussianNetRateProfileResult,
		profile_gaussian_upper_bound, EmpiricalCalibrationAdapter,
	)

# Package version
try:
	__version__ = version("jinwu")
except PackageNotFoundError:  # pragma: no cover - during editable installs
	__version__ = "0.0.0"

# ``timescale`` 只按"模块"导出（_MODULE_EXPORTS 优先于 _DATA_EXPORTS 检查）。
# 注意：master 上 ``from jinwu.core import timescale`` 返回 data.py 里的
# 时标分析器类 ``timescale``；beta 新增 core/timescale.py 模块后同名冲突，
# 该类被静默遮蔽（0.2.0 破坏性变更，见 docs/changelog.rst）。现约定：
# - ``jinwu.core.timescale`` = 模块（data.py 内部 `from . import timescale
#   as timescale_mod` 依赖此绑定，故不可改回类）；
# - 分析器类改从 ``jinwu.core.data.timescale`` 获取，或继续用实例方法
#   ``lc.timescale(...)``（推荐，公开 API 不变）。
# - ``_DATA_EXPORTS`` 中的 'timescale' 已移除：它在模块分支之后、永远不可达，
#   留着会误导读者以为该名字仍解析到类。
_MODULE_EXPORTS = {
	'heasoft', 'plot', 'plotpanel', 'time', 'ops', 'io', 'lf', 'redshift', 'timescale',
	'model_comparison', 'bxa_fit',
}

_OGIP_EXPORTS = {
	'ValidationReport', 'ValidationMessage', 'OgipFitsBase', 'check_response_compatibility',
}

_BASE_EXPORTS = {
	'EnergyBand', 'ChannelBand', 'RegionArea', 'RegionAreaSet',
	'HduHeader', 'FitsHeaderDump', 'OgipMeta',
	'ArfBase', 'RmfBase', 'PhaBase', 'LightcurveDataBase', 'EventDataBase',
}

_DATA_EXPORTS = {
	'ArfData', 'RmfData', 'PhaData', 'LightcurveData', 'EventData',
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

_PLOTPANEL_EXPORTS = {
	'LightcurveLike', 'SpectrumLike', 'PanelSpec', 'multi_panel', 'overlay',
}

_UPPER_LIMIT_EXPORTS = {
	'GaussianNetRateObservation', 'GaussianNetRateProfileResult',
	'profile_gaussian_upper_bound', 'EmpiricalCalibrationAdapter',
}

_TIME_EXPORTS = {
	'Time', 'TimeDelta',
}

# Process-wide fit settings API (from .config); never imports bxa/xspec.
_FIT_SETTINGS_EXPORTS = {
	'get_fit_settings', 'set_fit_settings', 'reset_fit_settings',
	'set_fit_method', 'get_fit_method', 'fit_settings',
}

# Unified dispatch entry point (from .fit).
_FIT_EXPORTS = {
	'fit_spectral',
}

# BXA nested-sampling surface (from .bxa_fit); resolved lazily so that
# importing jinwu.core never requires bxa/xspec to be installed.
_BXA_EXPORTS = {
	'BXAFitResult', 'BXAPriorSpec', 'fit_prepared_bxa', 'fit_xray_models_bxa',
	'resolve_priors', 'run_bxa_pipeline',
}

__all__ = [
	# Submodules
	'heasoft', 'plot', 'time', 'ops', 'io', 'lf', 'redshift', 'timescale', 'model_comparison',
	# OGIP validation
	'ValidationReport', 'ValidationMessage', 'OgipFitsBase', 'check_response_compatibility',
	# Time primitives
	'Time', 'TimeDelta',
	# Data containers（'timescale' 已移除：该名字固定解析为模块，见上）
	'EnergyBand', 'ChannelBand', 'RegionArea', 'RegionAreaSet', 'HduHeader', 'FitsHeaderDump', 'OgipMeta', 'ArfBase', 'RmfBase', 'PhaBase', 'ArfData', 'RmfData', 'PhaData', 'LightcurveDataBase', 'LightcurveData', 'EventDataBase', 'EventData',
	# Dataset containers
	'LightcurveDataset', 'SpectrumDataset', 'JointDataset',
	# Multi-object plotting framework
	'plotpanel', 'LightcurveLike', 'SpectrumLike', 'PanelSpec', 'multi_panel', 'overlay',
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
	# Unit-aware Gaussian upper-limit primitives
	'GaussianNetRateObservation', 'GaussianNetRateProfileResult',
	'profile_gaussian_upper_bound', 'EmpiricalCalibrationAdapter',
	# Process-wide fit settings API
	'get_fit_settings', 'set_fit_settings', 'reset_fit_settings',
	'set_fit_method', 'get_fit_method', 'fit_settings',
	# Unified fit dispatch + BXA Bayesian nested sampling
	'fit_spectral', 'bxa_fit',
	'BXAFitResult', 'BXAPriorSpec', 'fit_prepared_bxa', 'fit_xray_models_bxa',
	'resolve_priors', 'run_bxa_pipeline',
	# Package meta
	'__version__',
]


def __getattr__(name: str):
	if name in _MODULE_EXPORTS:
		mod = import_module(f'.{name}', __name__)
		globals()[name] = mod
		return mod

	if name in _OGIP_EXPORTS:
		mod = import_module('.ogip', __name__)
		value = getattr(mod, name)
		globals()[name] = value
		return value

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

	if name in _PLOTPANEL_EXPORTS:
		mod = import_module('.plotpanel', __name__)
		value = getattr(mod, name)
		globals()[name] = value
		return value

	if name in _TIME_EXPORTS:
		mod = import_module('.time', __name__)
		value = getattr(mod, name)
		globals()[name] = value
		return value

	if name in _UPPER_LIMIT_EXPORTS:
		mod = import_module('.upperlimit', __name__)
		value = getattr(mod, name)
		globals()[name] = value
		return value

	if name in _FIT_SETTINGS_EXPORTS:
		mod = import_module('.config', __name__)
		value = getattr(mod, name)
		globals()[name] = value
		return value

	if name in _FIT_EXPORTS:
		mod = import_module('.fit', __name__)
		value = getattr(mod, name)
		globals()[name] = value
		return value

	if name in _BXA_EXPORTS:
		mod = import_module('.bxa_fit', __name__)
		value = getattr(mod, name)
		globals()[name] = value
		return value

	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
	return sorted(set(globals().keys()) | set(__all__))
