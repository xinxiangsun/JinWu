"""
AutoHEA
=======

A Python package for automatic high-energy astrophysics analysis.

Top-level package re-exports commonly used subpackages for convenience:
- core: OGIP FITS IO, unified readers, and utilities
- lightcurve: SNR evaluation and trigger decision helpers
- background: Priors/posteriors for background modeling
- spectrum, missions, response, timing, data: domain subpackages
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

# Package metadata
try:
	__version__ = version("autohea")
except PackageNotFoundError:  # pragma: no cover - editable installs or source tree
	__version__ = "0.0.3"

__author__ = "Xinxiang Sun"
__email__ = "sunxinxiang24@mails.ucas.ac.cn"
__description__ = "AutoHEA: A Python package for automatic high-energy astrophysics analysis."

# Re-export subpackages for ergonomic imports
from . import core
from . import lightcurve
from . import background
from . import spectrum
from . import missions
from . import response
from . import timing
from . import data

__all__ = [
	# Subpackages
	'core', 'lightcurve', 'background', 'spectrum', 'missions', 'response', 'timing', 'data',
	# Package meta
	'__version__',
]
