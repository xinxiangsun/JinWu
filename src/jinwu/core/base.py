from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, Literal, Optional, Tuple, TYPE_CHECKING

import numpy as np

from .ogip import OgipResponseBase, OgipSpectrumBase, OgipTimeSeriesBase
from .time import Time

if TYPE_CHECKING:
    from .xselect import XSelectSession


@dataclass(slots=True)
class EnergyBand:
    emin: float
    emin_unit: str
    emax: float
    emax_unit: str


@dataclass(slots=True)
class ChannelBand:
    ch_lo: int
    ch_hi: int


@dataclass(slots=True)
class RegionArea:
    role: Literal['src', 'bkg', 'unk']
    shape: Optional[str]
    area: Optional[float]
    component: Optional[int]


@dataclass(slots=True)
class RegionAreaSet:
    src: list[RegionArea] = field(default_factory=list)
    bkg: list[RegionArea] = field(default_factory=list)
    unk: list[RegionArea] = field(default_factory=list)

    @property
    def src_area(self) -> Optional[float]:
        vals = [d.area for d in self.src if d.area is not None]
        return float(sum(vals)) if vals else None

    @property
    def bkg_area(self) -> Optional[float]:
        vals = [d.area for d in self.bkg if d.area is not None]
        return float(sum(vals)) if vals else None

    @classmethod
    def from_regions(cls, regions: list[RegionArea] | None) -> 'RegionAreaSet':
        inst = cls()
        if not regions:
            return inst
        for r in regions:
            if r.role == 'src':
                inst.src.append(r)
            elif r.role == 'bkg':
                inst.bkg.append(r)
            else:
                inst.unk.append(r)
        return inst


@dataclass(slots=True)
class HduHeader:
    name: str
    ver: Optional[int]
    header: Dict[str, Any]


@dataclass(slots=True)
class FitsHeaderDump:
    primary: Dict[str, Any]
    extensions: list[HduHeader]


@dataclass(slots=True)
class OgipMeta:
    telescop: Optional[str]
    instrume: Optional[str]
    detnam: Optional[str]
    timesys: Optional[str]
    timeunit: Optional[str]
    mjdref: Optional[float]
    tstart: Optional[float]
    tstop: Optional[float]
    object: Optional[str]
    obs_id: Optional[str]
    binsize: Optional[float]
    timezero: Optional[float]
    trefpos: Optional[str]
    dateobs: Optional[str]


@dataclass(slots=True)
class ArfBase(OgipResponseBase):
    """Pure field-only base dataclass for ARF response data."""

    kind: ClassVar[Literal['arf']] = 'arf'

    energ_lo: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=float))
    energ_hi: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=float))
    specresp: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=float))
    columns: Tuple[str, ...] = ()


@dataclass(slots=True)
class RmfBase(OgipResponseBase):
    """Pure field-only base dataclass for RMF response data."""

    kind: ClassVar[Literal['rmf']] = 'rmf'

    energ_lo: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=float))
    energ_hi: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=float))
    n_grp: Optional[np.ndarray] = None
    f_chan: Optional[np.ndarray] = None
    n_chan: Optional[np.ndarray] = None
    matrix: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=object))
    channel: Optional[np.ndarray] = None
    e_min: Optional[np.ndarray] = None
    e_max: Optional[np.ndarray] = None
    columns: Tuple[str, ...] = ()


@dataclass(slots=True)
class PhaBase(OgipSpectrumBase):
    """Pure field-only base dataclass for PHA spectrum data."""

    kind: ClassVar[Literal['pha']] = 'pha'

    channels: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=int))
    counts: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=float))
    stat_err: Optional[np.ndarray] = None
    exposure: float = 0.0
    backscal: Optional[float] = None
    areascal: Optional[float] = None
    respfile: Optional[str] = None
    ancrfile: Optional[str] = None
    quality: Optional[np.ndarray] = None
    grouping: Optional[np.ndarray] = None
    ebounds: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
    columns: Tuple[str, ...] = ()


@dataclass(slots=True)
class LightcurveDataBase(OgipTimeSeriesBase):
    """Pure field-only base dataclass for lightcurve data."""

    kind: ClassVar[Literal['lc']] = 'lc'

    time: Optional[np.ndarray] = None
    value: Optional[np.ndarray] = None

    timezero: float = 0.0
    timezero_obj: Optional[Time] = None
    dt: Optional[np.ndarray | float] = None

    bin_lo: Optional[np.ndarray] = None
    bin_hi: Optional[np.ndarray] = None
    bin_width: Optional[np.ndarray] = None
    binning: Literal['uniform', 'variable', 'unknown'] = 'unknown'
    tstart: Optional[float] = None
    tseg: Optional[float] = None

    error: Optional[np.ndarray] = None
    is_rate: bool = False

    counts: Optional[np.ndarray] = None
    rate: Optional[np.ndarray] = None
    counts_err: Optional[np.ndarray] = None
    rate_err: Optional[np.ndarray] = None
    err_dist: Optional[Literal['poisson', 'gauss']] = None

    gti_start: Optional[np.ndarray] = None
    gti_stop: Optional[np.ndarray] = None
    quality: Optional[np.ndarray] = None
    fracexp: Optional[np.ndarray] = None
    backscal: Optional[np.ndarray | float] = None
    areascal: Optional[np.ndarray | float] = None

    telescop: Optional[str] = None
    timesys: Optional[str] = None
    mjdref: Optional[float] = None

    exposure: Optional[float] = None
    bin_exposure: Optional[np.ndarray] = None
    region: Optional[RegionArea] = None
    columns: Tuple[str, ...] = ()
    ratio: Optional[float] = None


@dataclass(slots=True)
class EventDataBase(OgipTimeSeriesBase):
    """Pure field-only base dataclass for event data."""

    kind: ClassVar[Literal['evt']] = 'evt'

    time: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=float))
    timezero: float = 0.0
    timezero_obj: Optional[Any] = None
    telescop: Optional[str] = None

    pi: Optional[np.ndarray] = None
    channel: Optional[np.ndarray] = None
    x: Optional[np.ndarray] = None
    y: Optional[np.ndarray] = None

    gti_start: Optional[np.ndarray] = None
    gti_stop: Optional[np.ndarray] = None
    gti_start_obj: Optional[Any] = None
    gti_stop_obj: Optional[Any] = None
    gti: Optional[list] = None

    raw_columns: Optional[Dict[str, np.ndarray]] = None
    colmap: Optional[Dict[str, Optional[str]]] = None
    energy: Optional[np.ndarray] = None
    ebounds: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
    columns: Tuple[str, ...] = ()

    _xselect_session: Optional['XSelectSession'] = field(default=None, init=False, repr=False, compare=False)


__all__ = [
    "EnergyBand",
    "ChannelBand",
    "RegionArea",
    "RegionAreaSet",
    "HduHeader",
    "FitsHeaderDump",
    "OgipMeta",
    "ArfBase",
    "RmfBase",
    "PhaBase",
    "LightcurveDataBase",
    "EventDataBase",
]
