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

from importlib.metadata import PackageNotFoundError, version as _pkg_version
from pathlib import Path
import tomllib


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
__license__ = "BSD-3-Clause"

# Re-export subpackages for ergonomic imports
from . import (
    core,
    lightcurve,
    background,
    spectrum,
    response,
    timing,
    fermi,
    ep,
    ftools,
    model,
    physics,
)

try:
    from . import data
    _has_data = True
except ImportError:
    _has_data = False

# Re-export common utilities from core
from .core import netdata, readfits

__all__ = [
    # Subpackages
    'core',
    'lightcurve',
    'background',
    'spectrum',
    'response',
    'timing',
    'fermi',
    'ep',
    'ftools',
    'model',
    'physics',
    # Common utilities
    'netdata',
    'readfits',
    # Package metadata
    '__version__',
    '__author__',
    '__email__',
    '__license__',
]

if _has_data:
    __all__.append('data')

