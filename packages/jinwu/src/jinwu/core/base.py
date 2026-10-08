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
        """求已知源区域面积之和 / Sum the known source-region areas.

        忽略 area=None；无可用面积时返回 None。单位沿用区域定义，
        本属性不执行几何合并或单位换算，重叠区域会直接重复累加。
        Ignore None areas and return None when none are known. Units follow
        the region definitions; no union or conversion is applied to overlaps.
        """
        vals = [d.area for d in self.src if d.area is not None]
        return float(sum(vals)) if vals else None

    @property
    def bkg_area(self) -> Optional[float]:
        """求已知背景区域面积之和 / Sum the known background-region areas.

        忽略未知面积；无已知值返回 None。须由调用者统一面积单位并处理重叠。
        Ignore missing areas; return None if all are unknown. The caller ensures
        consistent area units and handles overlapping regions.
        """
        vals = [d.area for d in self.bkg if d.area is not None]
        return float(sum(vals)) if vals else None

    @classmethod
    def from_regions(cls, regions: list[RegionArea] | None) -> 'RegionAreaSet':
        """按区域角色分类 / Partition region records by source/background role.

        返回新容器，保留原 RegionArea 对象引用；未知角色归入 unk。
        None 或空列表返回空容器，不计算面积。
        Return a new container referencing the original records; unrecognized
        roles go to unk. Missing/empty input gives empty lists; no area calculation.
        """
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


@dataclass(slots=True, eq=False)
class ArfBase(OgipResponseBase):
    """Pure field-only base dataclass for ARF response data."""

    kind: ClassVar[Literal['arf']] = 'arf'

    energ_lo: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=float))
    energ_hi: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=float))
    specresp: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=float))
    columns: Tuple[str, ...] = ()
    # OGIP 版本声明（HDUVERS，如 '1.1.0'）
    hduvers: Optional[str] = None


@dataclass(slots=True, eq=False)
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
    # 通道约定键（对齐 HEASoft 6.37 heasp 读入行为）：
    # tlmin = F_CHAN 的 TLMINn（0 基/1 基），det_chans = DETCHANS。
    tlmin: Optional[int] = None
    det_chans: Optional[int] = None
    hduvers: Optional[str] = None

@dataclass(slots=True, eq=False)
class PhaBase(OgipSpectrumBase):
    """Pure field-only base dataclass for PHA spectrum data."""

    kind: ClassVar[Literal['pha']] = 'pha'

    channels: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=int))
    counts: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=float))
    rate: Optional[np.ndarray] = None
    stat_err: Optional[np.ndarray] = None
    exposure: float = 0.0
    backscal: Optional[float] = None
    areascal: Optional[float] = None
    respfile: Optional[str] = None
    ancrfile: Optional[str] = None
    quality: Optional[np.ndarray] = None
    grouping: Optional[np.ndarray] = None
    ebounds: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
    raw_spectrum_columns: Optional[Dict[str, np.ndarray]] = None
    columns: Tuple[str, ...] = ()
    # 通道编号三件套（TLMIN1/TLMAX1/DETCHANS），缺失时由通道数组回退推断。
    tlmin: Optional[int] = None
    tlmax: Optional[int] = None
    det_chans: Optional[int] = None


@dataclass(slots=True, eq=False)
class LightcurveDataBase(OgipTimeSeriesBase):
    """Pure field-only base dataclass for lightcurve data."""

    kind: ClassVar[Literal['lc']] = 'lc'

    time: Optional[np.ndarray] = None
    time_raw: Optional[np.ndarray] = None
    time_rel: Optional[np.ndarray] = None
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


@dataclass(slots=True, eq=False)
class EventDataBase(OgipTimeSeriesBase):
    """Pure field-only base dataclass for event data."""

    kind: ClassVar[Literal['evt']] = 'evt'

    time: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=float))
    time_raw: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=float))
    time_rel: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=float))
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
