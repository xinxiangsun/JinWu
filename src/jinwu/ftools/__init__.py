"""Lightweight Pure-Python HEASOFT ftools equivalents.

This package provides minimal Pure-Python implementations of common HEASOFT ftools
functionality, designed to work within JinWu without external Fortran/C dependencies.

Modules:
    - fextract: Extract events into PHA spectra (fextract equivalent)
    - ftgrouppha: Group PHA by minimum counts (ftgrouppha equivalent)
    - ftrbnpha: Rebin PHA to desired channel count (ftrbnpha equivalent)
    - ftrbnrmf: Rebin ARF/RMF to new energy bins (ftrbnrmf equivalent)
    - ftselect: Simple expression filtering for event tables
    - region: DS9 region parsing and point-in-region filtering
    - grppha: Minimal grppha-like grouping
    - teldef: Teldef parsing and coordinate transformations
    - rmf_mapping: RMF to energy mapping

Example:
    >>> from jinwu.ftools import fextract, ftgrouppha
    >>> pha = fextract.extract(events_file, pha_file)
    >>> grouped_pha = ftgrouppha.group_min_counts(pha, min_counts=10)
"""

from __future__ import annotations

from . import fextract
from . import ftgrouppha
from . import ftrbnpha
from . import ftrbnrmf
from . import ftselect
from . import region
from . import grppha
from . import teldef
from . import rmf_mapping
from . import teldef_helpers
from . import xselect_mdb

__all__ = [
    'fextract',
    'ftgrouppha',
    'ftrbnpha',
    'ftrbnrmf',
    'ftselect',
    'region',
    'grppha',
    'teldef',
    'rmf_mapping',
    'teldef_helpers',
    'xselect_mdb',
]

