"""
JinWu: Joint Inference for high-energy transient light-curve & spectral analysis with Unifying physical modeling
=================================================================================================

JinWu is a comprehensive Python package for X-ray and gamma-ray astrophysics, combining
spectral and temporal analysis with unified physical modeling.

Key Features:
    - OGIP FITS I/O (ARF, RMF, PHA, lightcurves, events)
    - Light-curve and spectral analysis tools
    - Background modeling and inference
    - XSPEC-inspired component models
    - Mission-specific support (Einstein Probe, Fermi/GBM, Swift, etc.)
    - Timing and transient analysis

Main Subpackages:
    - core: OGIP FITS readers, unified data structures, plotting
    - lightcurve: SNR evaluation, trigger decision
    - background: Background priors and posteriors
    - model: Spectral model components (additive, multiplicative, convolution)
    - spectrum: Spectrum synthesis and fakeit integration
    - response: Response matrices and ARF/RMF utilities
    - physics: General relativity and radiation effects
    - ftools: Pure-Python HEASOFT equivalents
    - fermi: Fermi/GBM mission utilities
    - ep: Einstein Probe utilities
    - timing: Timing analysis tools

Quick Start:
    >>> import jinwu as jw
    >>> pha = jw.core.read_pha('source.pha')
    >>> lc = jw.core.read_lc('lightcurve.fits')
    >>> from jinwu.background import BackgroundPrior
    >>> prior = BackgroundPrior()

Documentation: https://jinwu.readthedocs.io
"""

from __future__ import annotations

import sys
import types
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING
import tomllib

if TYPE_CHECKING:
    from . import (
        core,
        lightcurve,
        background,
        spectrum,
        response,
        timing,
        fermi,
        ep,
        swift,
        ftools,
        model,
        physics,
    )
    from .core import time as time
    from .core import (
        netdata,
        readfits,
        read_arf,
        read_rmf,
        read_pha,
        read_lc,
        read_evt,
        ArfData,
        RmfData,
        PhaData,
        LightcurveDataBase,
        LightcurveData,
        EventDataBase,
        EventData,
    )
    if find_spec('.data', __name__) is not None:
        from . import data


def _read_version_from_pyproject() -> str:
    """Read version from pyproject.toml in editable installs."""
    root = Path(__file__).resolve().parents[2]
    project_file = root / 'pyproject.toml'
    if not project_file.exists():
        return '0.0.0+unknown'
    try:
        with project_file.open('rb') as fh:
            data = tomllib.load(fh)
        return data.get('project', {}).get('version', '0.0.0+unknown')
    except Exception:
        return '0.0.0+unknown'


# Version resolution priority:
# 1) setuptools_scm generated _version module (after build)
# 2) Installed distribution metadata (pip install)
# 3) pyproject.toml (editable/source installation)
try:
    from ._version import version as __version__  # type: ignore
except Exception:
    try:
        __version__ = _pkg_version('jinwu')
    except PackageNotFoundError:
        __version__ = _read_version_from_pyproject()

__author__ = "Xinxiang Sun"
__email__ = "sunxx@nao.cas.cn"
__license__ = "GPL-3.0-or-later"

_SUBPACKAGES = {
    'core',
    'lightcurve',
    'background',
    'spectrum',
    'response',
    'timing',
    'fermi',
    'ep',
    'swift',
    'ftools',
    'model',
    'physics',
}

_CORE_EXPORTS = {
    'netdata',
    'readfits',
    'read_arf',
    'read_rmf',
    'read_pha',
    'read_lc',
    'read_evt',
    'ArfData',
    'RmfData',
    'PhaData',
    'LightcurveDataBase',
    'LightcurveData',
    'EventDataBase',
    'EventData',
}

_HAS_DATA = find_spec('.data', __name__) is not None


class _LazyTimeModule(types.ModuleType):
    _loaded: types.ModuleType | None = None

    def _load(self) -> types.ModuleType:
        if self._loaded is None:
            self._loaded = import_module('.core.time', __name__)
            sys.modules[f'{__name__}.time'] = self._loaded
        return self._loaded

    def __getattr__(self, item: str):
        return getattr(self._load(), item)

    def __dir__(self) -> list[str]:
        return sorted(set(super().__dir__()) | set(dir(self._load())))


if f'{__name__}.time' not in sys.modules:
    sys.modules[f'{__name__}.time'] = _LazyTimeModule(f'{__name__}.time')

__all__ = [
    # Subpackages
    'core',
    'time',
    'lightcurve',
    'background',
    'spectrum',
    'response',
    'timing',
    'fermi',
    'ep',
    'swift',
    'ftools',
    'model',
    'physics',
    # Common utilities
    'netdata',
    'readfits',
    'read_arf',
    'read_rmf',
    'read_pha',
    'read_lc',
    'read_evt',
    # Core data classes
    'ArfData',
    'RmfData',
    'PhaData',
    'LightcurveDataBase',
    'LightcurveData',
    'EventDataBase',
    'EventData',
    # Package metadata
    '__version__',
    '__author__',
    '__email__',
    '__license__',
]

if _HAS_DATA:
    __all__.append('data')


def __getattr__(name: str):
    if name == 'time':
        mod = sys.modules[f'{__name__}.time']
        globals()[name] = mod
        return mod

    if name in _SUBPACKAGES:
        mod = import_module(f'.{name}', __name__)
        globals()[name] = mod
        return mod

    if name == 'data' and _HAS_DATA:
        mod = import_module('.data', __name__)
        globals()[name] = mod
        return mod

    if name in _CORE_EXPORTS:
        core_mod = import_module('.core', __name__)
        value = getattr(core_mod, name)
        globals()[name] = value
        return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))

