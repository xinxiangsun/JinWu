"""
通用 OGIP FITS 读取器（Numpy 优先的输出，含中文优先双语注释）

本模块面向高能天体物理常见 FITS 产品（ARF 响应面积、RMF 响应矩阵、PHA 能谱、光变曲线、事件表），
提供健壮、可复用的读取器，统一返回标准化的 numpy 数组与轻量级数据类，便于后续科学计算与可视化。

设计目标
--------
- 自包含：仅依赖 astropy.io.fits 与 numpy；可与外部库（如 gdt）互操作，但不强依赖。
- 贴合 OGIP：理解常见扩展/列名，并对缺失的可选字段做出合理处理。
- Numpy 优先：输出以 ndarray 或小型数据类为主，便于向下游数值流程衔接。
- 实用工具：支持能段/道（channel）筛选、RMF 稀疏到稠密重建等常用操作。

约定
----
- 对 EP 数据，常用能段为 0.5–4.0 keV。ARF 常以 bin 81~780 定义该能段边界；
    PHA/RMF 的道选择通常依据 EBOUNDS 的能段重叠与 1024 道的 51~399 范围。

English summary
---------------
General-purpose OGIP FITS readers with numpy-first outputs.
This module consolidates robust, reusable readers for common HEP FITS products
(ARF, RMF, PHA, lightcurve, events), returning standardized numpy arrays and
lightweight dataclasses. Utilities include band/channel filtering and RMF
sparse-to-dense rebuild. Conventions follow EP practice: ARF bins 81–780 map
to ~0.5–4 keV; PHA/RMF channels often use EBOUNDS overlap and 51–399 range.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, cast, Union, Literal, overload

import numpy as np
from astropy.io import fits
from .ogip import (
    OgipFitsBase, OgipTimeSeriesBase, OgipSpectrumBase, OgipResponseBase,
    ValidationReport, ValidationMessage,
)

__all__ = [
    # Data containers
    "EnergyBand", "ChannelBand",
    "RegionArea", "RegionAreaSet",
    "HduHeader", "FitsHeaderDump", "OgipMeta", "OgipFitsBase", "OgipTimeSeriesBase", "OgipSpectrumBase", "OgipResponseBase",
    "ArfData", "RmfData", "PhaData", "LightcurveData", "EventData",
    # Readers
    "OgipArfReader", "OgipRmfReader", "OgipPhaReader", "OgipLightcurveReader", "OgipEventReader",
    # Backward-compat aliases
    "ArfReader", "RmfReader", "RspReader",
    # Utilities
    "band_from_arf_bins", "channel_mask_from_ebounds",
    # Kind inference and direct dataclass helpers
    "OgipData", "guess_ogip_kind",
    # Direct dataclass-returning helpers
    "read_fits", "read_arf", "read_rmf", "read_pha", "read_lc", "read_evt",
]


# ---------- Data containers ----------

@dataclass(slots=True)
class EnergyBand:
    """能段描述数据类（中文优先 / CN first）

    字段
    - emin/emax: 能段下/上边界（数值）
    - emin_unit/emax_unit: 单位字符串，通常为 "keV"

    English
    - emin/emax: energy band lower/upper bounds (numeric)
    - emin_unit/emax_unit: unit strings, typically "keV"
    """
    emin: float
    emin_unit: str
    emax: float
    emax_unit: str

@dataclass(slots=True)
class ChannelBand:
    """道（Channel）范围数据类

    字段
    - ch_lo/ch_hi: 起止道号（闭区间）

    English
    - ch_lo/ch_hi: inclusive channel range
    """
    ch_lo: int
    ch_hi: int


@dataclass(slots=True)
class RegionArea:
    """单个区域面积信息。

    role:
        - 'src': 源区(source)
        - 'bkg': 背景区（background）
        - 'unk': 尚未能可靠判断角色（unknown）
    """
    role: Literal['src', 'bkg', 'unk']
    shape: Optional[str]
    area: Optional[float]
    component: Optional[int]


@dataclass(slots=True)
class RegionAreaSet:
    """区域聚合：按 src/bkg/unk 分类存放 RegionArea 列表。

    当前 LightcurveData 仍返回扁平列表；后续需要聚合时可用本类包装。
    """
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
    def from_regions(cls, regions: list[RegionArea] | None) -> RegionAreaSet:
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
    """单个扩展（HDU）头信息

    字段
    - name: 扩展名（EXTNAME）
    - ver: 扩展版本（EXTVER）
    - header: 该扩展所有关键字的字典

    English
    - name: EXTNAME
    - ver: EXTVER
    - header: full header key-value map for this HDU
    """
    name: str
    ver: Optional[int]
    header: Dict[str, Any]


@dataclass(slots=True)
class FitsHeaderDump:
    """整文件级别的头部集合（包含主HDU与全部扩展的关键字）"""
    primary: Dict[str, Any]
    extensions: list[HduHeader]


@dataclass(slots=True)
class OgipMeta:
    """常见 OGIP 元数据（便于快速访问望远镜/仪器/探测器与时间系统）

    - telescop/instrume/detnam: 望远镜/仪器/探测器名
    - timesys/timeunit: 时间系统/单位
    - mjdref: 参考 MJD（优先合并 MJDREFI+MJDREFF）
    - tstart/tstop: 数据时间范围（若存在）
    - object/obs_id: 目标名/观测号（若存在）
    """
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
    # 新增：时间相关元数据
    binsize: Optional[float]
    timezero: Optional[float]
@dataclass(slots=True)
class ArfData(OgipResponseBase):
    """ARF 响应面积数据

    字段
    - kind: 常量字面量 'arf'
    - path: 文件路径
    - energ_lo/energ_hi: 每个能 bin 的下/上边界（keV）
    - specresp: 对应每个能 bin 的有效面积（cm^2）
    - header: 头关键字字典

    English
    - OGIP ARF: energy bin edges (ENERG_LO/HI) and effective area (SPECRESP).
    """
    kind: Literal['arf']
    energ_lo: np.ndarray  # shape (N,)
    energ_hi: np.ndarray  # shape (N,)
    specresp: np.ndarray  # shape (N,), cm^2
    columns: Tuple[str, ...] = ()

    def validate(self) -> ValidationReport:  # type: ignore[override]
        rpt = super().validate()
        colset = {c.upper() for c in self.columns}
        for c in ["ENERG_LO", "ENERG_HI", "SPECRESP"]:
            if c not in colset:
                rpt.add('ERROR', 'MISSING_COLUMN', f"ARF missing column {c}")
        rpt.ok = len(rpt.errors()) == 0
        return rpt

@dataclass(slots=True)
class RmfData(OgipResponseBase):
    """RMF 响应矩阵数据（支持稀疏字段与 EBOUNDS）

    字段
    - energ_lo/energ_hi: 入射能 bin 边界
    - n_grp/f_chan/n_chan: OGIP 稀疏分组定义（可选）
    - matrix: 若为稀疏存储则为逐行对象数组；若已重建则为 2D 稠密矩阵
    - channel/e_min/e_max: EBOUNDS（道到能的映射）
    - header: 头关键字字典

    English
    - Supports OGIP sparse representation and EBOUNDS mapping.
    - Use rebuild_dense() to obtain (N_E, N_C) dense redistribution matrix.
    """
    kind: Literal['rmf']
    energ_lo: np.ndarray  # shape (N_E,)
    energ_hi: np.ndarray  # shape (N_E,)
    # OGIP sparse representation
    n_grp: Optional[np.ndarray]  # shape (N_E,) or None
    f_chan: Optional[np.ndarray]  # variable-length sequences concatenated
    n_chan: Optional[np.ndarray]
    matrix: np.ndarray  # object array of per-row matrices, or 2D dense if rebuilt
    # EBOUNDS
    channel: Optional[np.ndarray]  # shape (N_C,) CHANNEL indices
    e_min: Optional[np.ndarray]    # shape (N_C,)
    e_max: Optional[np.ndarray]    # shape (N_C,)
    columns: Tuple[str, ...] = ()

    def rebuild_dense(self) -> np.ndarray:
        """从稀疏列重建稠密响应矩阵（形状为 (N_E, N_C)）

        返回
        - np.ndarray: 每个能 bin 对应各道的概率（稠密矩阵）

        English
        Rebuild a dense redistribution matrix of shape (N_E, N_C) if sparse
        columns are present. Returns probability per channel for each energy
        bin.
        """
        if self.channel is None or self.e_min is None or self.e_max is None:
            # No EBOUNDS; best effort: try to stack MATRIX if already dense-like
            if self.matrix.ndim == 2:
                return self.matrix
            raise ValueError("Cannot rebuild dense RMF without EBOUNDS and sparse definition")

        n_e = self.energ_lo.size
        n_c = int(self.channel.size)
        out = np.zeros((n_e, n_c), dtype=float)

        # If OGIP sparse columns exist, use them
        if (self.n_grp is not None) and (self.f_chan is not None) and (self.n_chan is not None):
            # MATRIX may be a variable-length vector per energy row; flatten sequentially per groups
            # We'll iterate per row, then within groups allocate consecutive channels
            idx_mat = 0
            for i in range(n_e):
                ng = int(self.n_grp[i]) if np.ndim(self.n_grp) > 0 else int(self.n_grp)
                # Each row's MATRIX may be stored in self.matrix[i]
                row_vals = self.matrix[i]
                # Some files store per-row vector as ndarray already
                if isinstance(row_vals, np.ndarray):
                    # Use per-row cursor
                    cursor = 0
                    # Flatten per-row f_chan/n_chan sequences: sometimes these are ragged; read via FITS tables
                    # Here, we assume f_chan/n_chan are stored per-row like MATRIX
                    fcs = np.atleast_1d(self.f_chan[i]) if np.ndim(self.f_chan) > 1 else np.atleast_1d(self.f_chan)
                    ncs = np.atleast_1d(self.n_chan[i]) if np.ndim(self.n_chan) > 1 else np.atleast_1d(self.n_chan)
                    for g in range(ng):
                        start = int(fcs[g])
                        width = int(ncs[g])
                        out[i, start:start+width] = row_vals[cursor:cursor+width]
                        cursor += width
                else:
                    # Fallback: if entire matrix is rectangular 2D already
                    if np.ndim(self.matrix) == 2 and self.matrix.shape[1] == n_c:
                        out[i, :] = self.matrix[i, :]
                    else:
                        raise ValueError("Unsupported RMF MATRIX layout")
            return out
        # Else, MATRIX might already be a (N_E, N_C) dense matrix
        if self.matrix.ndim == 2 and self.matrix.shape[1] == n_c:
            return self.matrix
        raise ValueError("RMF does not contain recognizable sparse columns and MATRIX is not dense")

    def validate(self) -> ValidationReport:  # type: ignore[override]
        rpt = super().validate()
        colset = {c.upper() for c in self.columns}
        for c in ["ENERG_LO", "ENERG_HI", "MATRIX"]:
            if c not in colset:
                rpt.add('ERROR', 'MISSING_COLUMN', f"RMF missing column {c}")
        # If EBOUNDS present, ensure CHANNEL/E_MIN/E_MAX columns are recorded
        if self.channel is not None:
            for c in ["CHANNEL", "E_MIN", "E_MAX"]:
                if c not in colset:
                    rpt.add('WARN', 'MISSING_COLUMN', f"EBOUNDS expected column {c}")
        rpt.ok = len(rpt.errors()) == 0
        return rpt

    @property
    def dense_matrix(self) -> np.ndarray:
        """Convenience: 返回重建的稠密矩阵 (缓存可后续扩展)"""
        return self.rebuild_dense()

@dataclass(slots=True)
class PhaData(OgipSpectrumBase):
    """PHA 光子谱数据

    字段
    - channels: 道号
    - counts: 计数（COUNTS）
    - stat_err: 统计误差（可选）
    - exposure/backscal/areascal/quality/grouping: 常见 OGIP 字段（可选）
    - ebounds: 若存在 EBOUNDS，则为 (CHANNEL, E_MIN, E_MAX)
    - header: 头关键字字典

    English
    - OGIP PHA SPECTRUM with optional EBOUNDS and quality/grouping vectors.
    """
    kind: Literal['pha']
    channels: np.ndarray  # shape (N,)
    counts: np.ndarray    # shape (N,)
    stat_err: Optional[np.ndarray]
    exposure: float
    backscal: Optional[float]
    areascal: Optional[float]
    quality: Optional[np.ndarray]
    grouping: Optional[np.ndarray]
    ebounds: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]  # (CHANNEL, E_MIN, E_MAX)
    columns: Tuple[str, ...] = ()

    def validate(self) -> ValidationReport:  # type: ignore[override]
        rpt = super().validate()
        colset = {c.upper() for c in self.columns}
        for c in ["CHANNEL", "COUNTS"]:
            if c not in colset:
                rpt.add('ERROR', 'MISSING_COLUMN', f"PHA missing column {c}")
        rpt.ok = len(rpt.errors()) == 0
        return rpt

    @property
    def count_rate(self) -> Optional[np.ndarray]:
        if self.exposure and self.exposure > 0:
            return self.counts / self.exposure
        return None

@dataclass(slots=True)
class LightcurveData(OgipTimeSeriesBase):
    """光变曲线数据

    字段
    - time: 时间轴（bin 左缘或中心）
    - value: RATE 或 COUNTS
    - error: 不确定度（可选）
    - dt: 典型时间分辨率（若可推断）
    - exposure: 曝光时间（头关键字推断，可选）
    - is_rate: 是否为速率（True 为 RATE，False 为 COUNTS）
    - header: 头关键字字典

    English
    - Light curve table with TIME and RATE/COUNTS.
    """
    kind: Literal['lc']
    time: np.ndarray   # bin left edges or centers
    value: np.ndarray  # RATE or COUNTS
    error: Optional[np.ndarray]
    dt: Optional[float]
    exposure: Optional[float]
    is_rate: bool
    columns: Tuple[str, ...] = ()

    def validate(self) -> ValidationReport:  # type: ignore[override]
        rpt = super().validate()
        colset = {c.upper() for c in self.columns}
        if "TIME" not in colset:
            rpt.add('ERROR', 'MISSING_COLUMN', "Lightcurve missing TIME column")
        if not any(x in colset for x in ["RATE", "COUNTS"]):
            rpt.add('ERROR', 'MISSING_COLUMN', "Lightcurve missing RATE/COUNTS column")
        rpt.ok = len(rpt.errors()) == 0
        return rpt

    @property
    def rate(self) -> Optional[np.ndarray]:
        if self.is_rate:
            return self.value
        if (not self.is_rate) and self.dt and self.dt > 0:
            return self.value / self.dt
        return None

    @property
    def counts(self) -> Optional[np.ndarray]:
        if not self.is_rate:
            return self.value
        if self.is_rate and self.dt and self.dt > 0:
            return self.value * self.dt
        return None
    # 若可从区域扩展推断（如 WXT 的 REG00101），记录单个区域描述
    region: Optional[RegionArea] = None

@dataclass(slots=True)
class EventData(OgipTimeSeriesBase):
    """事件列表数据

    字段
    - time: 事件到达时间
    - pi/channel: 事件能道（可选）
    - gti_start/gti_stop: GTI（可选）
    - header: 头关键字字典

    English
    - Event list with optional PI/CHANNEL and GTI.
    """
    kind: Literal['evt']
    time: np.ndarray   # events times (s)
    pi: Optional[np.ndarray]       # PI or CHANNEL if present
    channel: Optional[np.ndarray]
    gti_start: Optional[np.ndarray]
    gti_stop: Optional[np.ndarray]
    columns: Tuple[str, ...] = ()

    def validate(self) -> ValidationReport:  # type: ignore[override]
        rpt = super().validate()
        colset = {c.upper() for c in self.columns}
        if "TIME" not in colset:
            rpt.add('ERROR', 'MISSING_COLUMN', "EVENTS missing TIME column")
        rpt.ok = len(rpt.errors()) == 0
        return rpt

    @property
    def duration(self) -> Optional[float]:
        if self.time.size:
            return float(self.time.max() - self.time.min())
        return None

    @property
    def gti_exposure(self) -> Optional[float]:
        if self.gti_start is None or self.gti_stop is None:
            return None
        return float(np.sum(self.gti_stop - self.gti_start))


# Unified data union
OgipData = Union[ArfData, RmfData, PhaData, LightcurveData, EventData]


# ---------- Utilities ----------

def _combine_mjdref(header: Dict[str, Any]) -> Optional[float]:
    """优先合并 MJDREFI + MJDREFF，若无则尝试 MJDREF。"""
    if header is None:
        return None
    if ("MJDREFI" in header) or ("MJDREFF" in header):
        mjdi = float(header.get("MJDREFI", 0.0))
        mjdf = float(header.get("MJDREFF", 0.0))
        return mjdi + mjdf
    if "MJDREF" in header:
        try:
            return float(header["MJDREF"])
        except Exception:
            return None
    return None


def _first_non_empty(keys: list[str], *headers: Dict[str, Any]) -> Optional[Any]:
    """在多个 header 中按顺序寻找第一个存在且非空的键。"""
    for key in keys:
        for hdr in headers:
            if hdr is None:
                continue
            if key in hdr and hdr[key] not in (None, "", " "):
                return hdr[key]
    return None


def _collect_headers_dump(hdul: fits.HDUList) -> FitsHeaderDump:
    hdr0 = cast(Any, getattr(hdul[0], 'header', {})) if len(hdul) > 0 else {}
    primary = dict(hdr0) if hdr0 else {}
    exts: list[HduHeader] = []
    for i, hdu in enumerate(hdul[1:], start=1):
        name = str(getattr(hdu, 'name', '') or '')
        ver_val = cast(Any, hdu.header).get('EXTVER', None)
        try:
            ver = int(ver_val) if ver_val is not None else None
        except Exception:
            ver = None
        exts.append(HduHeader(name=name, ver=ver, header=dict(cast(Any, hdu.header))))
    return FitsHeaderDump(primary=primary, extensions=exts)


def _build_meta(hdul: fits.HDUList, prefer_header: Optional[Dict[str, Any]]) -> OgipMeta:
    """根据优先头与主头/其他扩展构建常用 OGIP 元数据。"""
    dump = _collect_headers_dump(hdul)
    primary = dump.primary
    # 为了容错，也从其他扩展寻找关键字
    other_ext_headers = [x.header for x in dump.extensions]
    # telescope/instrument/detector
    telescop = _first_non_empty(["TELESCOP"], *(hdr for hdr in [prefer_header, primary] + other_ext_headers if hdr))
    instrume = _first_non_empty(["INSTRUME"], *(hdr for hdr in [prefer_header, primary] + other_ext_headers if hdr))
    detnam = _first_non_empty(["DETNAM", "DETNAME"], *(hdr for hdr in [prefer_header, primary] + other_ext_headers if hdr))
    # time system
    timesys = _first_non_empty(["TIMESYS"], *(hdr for hdr in [prefer_header, primary] + other_ext_headers if hdr))
    timeunit = _first_non_empty(["TIMEUNIT"], *(hdr for hdr in [prefer_header, primary] + other_ext_headers if hdr))
    # mjdref 可在任何层出现（优先 prefer_header -> primary -> others）
    mjdref = None
    for hdr in [prefer_header, primary] + other_ext_headers:
        if hdr is None:
            continue
        mjdref = _combine_mjdref(hdr)
        if mjdref is not None:
            break
    # time range
    tstart = None
    tstop = None
    for hdr in [prefer_header, primary] + other_ext_headers:
        if hdr is None:
            continue
        if (tstart is None) and ("TSTART" in hdr):
            try:
                tstart = float(hdr["TSTART"])  # type: ignore[arg-type]
            except Exception:
                pass
        if (tstop is None) and ("TSTOP" in hdr):
            try:
                tstop = float(hdr["TSTOP"])  # type: ignore[arg-type]
            except Exception:
                pass
        if (tstart is not None) and (tstop is not None):
            break
    # object & obs id
    obj = _first_non_empty(["OBJECT"], *(hdr for hdr in [prefer_header, primary] + other_ext_headers if hdr))
    obs_id = _first_non_empty(["OBS_ID", "OBS_ID"], *(hdr for hdr in [prefer_header, primary] + other_ext_headers if hdr))

    # binsize 优先尝试常见关键字：BIN_SIZE/BINSIZE/TIMEDEL/DELTAT/TBIN/TIMEBIN
    binsize_val = _first_non_empty(
        ["BIN_SIZE", "BINSIZE", "TIMEDEL", "DELTAT", "TBIN", "TIMEBIN"],
        *(hdr for hdr in [prefer_header, primary] + other_ext_headers if hdr)
    )
    try:
        binsize = float(binsize_val) if binsize_val is not None else None
    except Exception:
        binsize = None

    # timezero（时间零点偏移）
    timezero_val = _first_non_empty(["TIMEZERO"], *(hdr for hdr in [prefer_header, primary] + other_ext_headers if hdr))
    try:
        timezero = float(timezero_val) if timezero_val is not None else None
    except Exception:
        timezero = None
    return OgipMeta(
        telescop=str(telescop) if telescop is not None else None,
        instrume=str(instrume) if instrume is not None else None,
        detnam=str(detnam) if detnam is not None else None,
        timesys=str(timesys) if timesys is not None else None,
        timeunit=str(timeunit) if timeunit is not None else None,
        mjdref=float(mjdref) if mjdref is not None else None,
        tstart=float(tstart) if tstart is not None else None,
        tstop=float(tstop) if tstop is not None else None,
        object=str(obj) if obj is not None else None,
        obs_id=str(obs_id) if obs_id is not None else None,
        binsize=binsize,
        timezero=timezero,
    )

def band_from_arf_bins(arf_path: str | Path, bin_lo: int = 81, bin_hi: int = 780) -> EnergyBand:
    with fits.open(arf_path) as h:
        hd = cast(Any, h["SPECRESP"])  # BinTableHDU
        d = hd.data
        elo = np.asarray(d["ENERG_LO"], float)
        ehi = np.asarray(d["ENERG_HI"], float)
    i0 = max(0, int(bin_lo) - 1)
    i1 = min(ehi.size - 1, int(bin_hi) - 1)
    emin = float(elo[i0])
    emax = float(ehi[i1])
    return EnergyBand(emin=emin, emin_unit="keV", emax=emax, emax_unit="keV")


def channel_mask_from_ebounds(
    ebounds: Tuple[np.ndarray, np.ndarray, np.ndarray],
    band: EnergyBand,
    ch_band: Optional[ChannelBand] = None,
) -> np.ndarray:
    ch, e_lo, e_hi = ebounds
    mask = (e_hi > float(band.emin)) & (e_lo < float(band.emax))
    if ch_band is not None:
        mask &= (ch >= int(ch_band.ch_lo)) & (ch <= int(ch_band.ch_hi))
    return mask


# ---------- Readers ----------

class OgipArfReader:
    """ARF 读取器

    用法：reader = OgipArfReader(path); data = reader.read()

    English
    Reader for OGIP ARF files.
    """
    def __init__(self, path: str | Path):
        """参数
        - path: ARF 文件路径

        English
        - path: path to ARF file
        """
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(str(self.path))
        self._data: Optional[ArfData] = None

    def read(self) -> ArfData:
        """读取 ARF 并返回 `ArfData`。

        English: Read ARF and return `ArfData`.
        """
        with fits.open(self.path) as h:
            hd = cast(Any, h["SPECRESP"])  # BinTableHDU
            d = hd.data
            energ_lo = np.asarray(d["ENERG_LO"], float)
            energ_hi = np.asarray(d["ENERG_HI"], float)
            specresp = np.asarray(d["SPECRESP"], float)
            header = dict(cast(Any, hd.header))
            headers_dump = _collect_headers_dump(h)
            meta = _build_meta(h, header)
        self._data = ArfData(
            kind='arf', path=self.path,
            energ_lo=energ_lo, energ_hi=energ_hi, specresp=specresp,
            header=header, meta=meta, headers_dump=headers_dump,
        )
        return self._data

    def validate(self) -> ValidationReport:
        """运行数据类自身的 validate() 并返回报告。"""
        if self._data is None:
            self.read()
        assert self._data is not None
        return self._data.validate()


class OgipRmfReader:
    """RMF 读取器

    English
    Reader for OGIP RMF (or RSP) files with MATRIX and optional EBOUNDS.
    """
    def __init__(self, path: str | Path):
        """参数
        - path: RMF/RSP 文件路径

        English
        - path: path to RMF/RSP file
        """
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(str(self.path))
        self._data: Optional[RmfData] = None

    def read(self) -> RmfData:
        """读取 RMF 并返回 `RmfData`。

        English: Read RMF and return `RmfData`.
        """
        with fits.open(self.path) as h:
            hm = cast(Any, h["MATRIX"])  # BinTableHDU
            dm = hm.data
            energ_lo = np.asarray(dm["ENERG_LO"], float)
            energ_hi = np.asarray(dm["ENERG_HI"], float)
            header = dict(cast(Any, hm.header))
            headers_dump = _collect_headers_dump(h)
            meta = _build_meta(h, header)

            matrix = np.asarray(dm["MATRIX"], dtype=object)
            n_grp = dm["N_GRP"] if "N_GRP" in dm.columns.names else None
            f_chan = dm["F_CHAN"] if "F_CHAN" in dm.columns.names else None
            n_chan = dm["N_CHAN"] if "N_CHAN" in dm.columns.names else None

            channel = e_min = e_max = None
            if "EBOUNDS" in h:
                de = cast(Any, h["EBOUNDS"]).data
                channel = np.asarray(de["CHANNEL"], int)
                e_min = np.asarray(de["E_MIN"], float)
                e_max = np.asarray(de["E_MAX"], float)

        self._data = RmfData(
            kind='rmf',
            path=self.path,
            energ_lo=energ_lo,
            energ_hi=energ_hi,
            n_grp=(np.asarray(n_grp) if n_grp is not None else None),
            f_chan=(np.asarray(f_chan) if f_chan is not None else None),
            n_chan=(np.asarray(n_chan) if n_chan is not None else None),
            matrix=matrix,
            channel=channel,
            e_min=e_min,
            e_max=e_max,
            header=header,
            meta=meta,
            headers_dump=headers_dump,
        )
        return self._data

    def validate(self) -> ValidationReport:
        if self._data is None:
            self.read()
        assert self._data is not None
        return self._data.validate()


class OgipPhaReader:
    """PHA 光子谱读取器

    English
    Reader for OGIP PHA SPECTRUM with optional EBOUNDS.
    """
    def __init__(self, path: str | Path):
        """参数
        - path: PHA 文件路径

        English
        - path: path to PHA file
        """
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(str(self.path))
        self._data: Optional[PhaData] = None

    def read(self) -> PhaData:
        """读取 PHA 并返回 `PhaData`。

        English: Read PHA and return `PhaData`.
        """
        with fits.open(self.path) as h:
            hs = cast(Any, h["SPECTRUM"])  # BinTableHDU
            ds = hs.data
            channels = np.asarray(ds["CHANNEL"], int)
            counts = np.asarray(ds["COUNTS"], float)
            stat_err = np.asarray(ds["STAT_ERR"], float) if "STAT_ERR" in ds.columns.names else None
            header_map = cast(Any, hs.header)
            exposure = float(header_map.get("EXPOSURE", header_map.get("EXPTIME", np.nan)))
            backscal = float(header_map.get("BACKSCAL")) if "BACKSCAL" in header_map else None
            areascal = float(header_map.get("AREASCAL")) if "AREASCAL" in header_map else None
            quality = np.asarray(ds["QUALITY"], int) if "QUALITY" in ds.columns.names else None
            grouping = np.asarray(ds["GROUPING"], int) if "GROUPING" in ds.columns.names else None
            ebounds = None
            if "EBOUNDS" in h:
                de = cast(Any, h["EBOUNDS"]).data
                ebounds = (
                    np.asarray(de["CHANNEL"], int),
                    np.asarray(de["E_MIN"], float),
                    np.asarray(de["E_MAX"], float),
                )
            header = dict(header_map)
            headers_dump = _collect_headers_dump(h)
            meta = _build_meta(h, header)
        self._data = PhaData(
            kind='pha', path=self.path,
            channels=channels, counts=counts, stat_err=stat_err,
            exposure=exposure, backscal=backscal, areascal=areascal,
            quality=quality, grouping=grouping, ebounds=ebounds,
            header=header, meta=meta, headers_dump=headers_dump,
        )
        return self._data

    def validate(self) -> ValidationReport:
        if self._data is None:
            self.read()
        assert self._data is not None
        return self._data.validate()

    def select_by_band(
        self,
        band: EnergyBand,
        rmf_chan_band: Optional[ChannelBand] = ChannelBand(51, 399),
    ) -> Tuple[np.ndarray, np.ndarray]:
        """依据能段筛选并返回 (channels, counts)

        行为
        - 若存在 EBOUNDS：使用 EBOUNDS 与给定能段重叠进行筛选；
        - 若不存在：退回到道范围（默认 51–399）。

        English
        Return (channels, counts) filtered by energy band via EBOUNDS; if
        EBOUNDS is missing, fall back to a channel range (e.g., 51–399).
        """
        if self._data is None:
            _ = self.read()
        d = self._data
        assert d is not None
        if d.ebounds is not None:
            mask = channel_mask_from_ebounds(d.ebounds, band, rmf_chan_band)
            # EBOUNDS and SPECTRUM channels should align; ensure alignment via channel indices
            ch_all = d.ebounds[0]
            # Build lookup from channel to spectrum index
            idx_map = {int(c): i for i, c in enumerate(d.channels)}
            sel_channels = ch_all[mask]
            sel_idx = np.array([idx_map[c] for c in sel_channels if c in idx_map], dtype=int)
            return d.channels[sel_idx], d.counts[sel_idx]
        # Fallback to channel-band only
        if rmf_chan_band is None:
            return d.channels, d.counts
        m = (d.channels >= rmf_chan_band.ch_lo) & (d.channels <= rmf_chan_band.ch_hi)
        return d.channels[m], d.counts[m]


class OgipLightcurveReader:
    """光变曲线读取器

    English
    Reader for OGIP-like light curve tables containing TIME and RATE/COUNTS.
    """
    def __init__(self, path: str | Path):
        """参数
        - path: 光变曲线 FITS 文件路径

        English
        - path: path to light curve FITS file
        """
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(str(self.path))
        self._data: Optional[LightcurveData] = None

    def read(self) -> LightcurveData:
        """读取光变曲线并返回 `LightcurveData`。

        English: Read light curve and return `LightcurveData`.
        """
        with fits.open(self.path) as h:
            # Try common names: 'RATE' ext, or first table with TIME
            hdu = None
            for ext in h:
                ext_any = cast(Any, ext)
                if getattr(ext_any, "data", None) is None:
                    continue
                names = getattr(ext_any.data, "columns", None)
                if names is not None and ("TIME" in names.names):
                    hdu = ext_any
                    break
            if hdu is None:
                raise ValueError("No suitable lightcurve HDU with TIME column found")
            d = cast(Any, hdu.data)
            time = np.asarray(d["TIME"], float)
            val = err = None
            is_rate = False
            if "RATE" in d.columns.names:
                val = np.asarray(d["RATE"], float)
                err = np.asarray(d["ERROR"], float) if "ERROR" in d.columns.names else None
                is_rate = True
            elif "COUNTS" in d.columns.names:
                val = np.asarray(d["COUNTS"], float)
                err = np.asarray(d["ERROR"], float) if "ERROR" in d.columns.names else None
                is_rate = False
            else:
                raise ValueError("Lightcurve HDU lacks RATE/COUNTS column")
            header = dict(cast(Any, hdu.header))
            headers_dump = _collect_headers_dump(h)
            meta = _build_meta(h, header)
            # Infer dt if possible
            dt = float(np.median(np.diff(time))) if time.size >= 2 else None
            exposure_val = header.get("EXPOSURE", header.get("EXPTIME", np.nan))
            exposure = float(exposure_val) if np.isfinite(exposure_val) else None
            # WXT 专用：若存在 REG00101 扩展，解析区域面积信息
            region = None
            try:
                if (meta.instrume or "").upper() == "WXT":
                    region = _load_wxt_regions(h)
            except Exception:
                # 保守失败，不中断读取
                region = None

        self._data = LightcurveData(
            kind='lc', path=self.path,
            time=time, value=val, error=err, dt=dt, exposure=exposure,
            is_rate=is_rate, header=header, meta=meta, headers_dump=headers_dump,
            region=region,
        )
        return self._data

    def validate(self) -> ValidationReport:
        if self._data is None:
            self.read()
        assert self._data is not None
        return self._data.validate()


def _load_wxt_regions(hdul: fits.HDUList) -> Optional[RegionArea]:
    """解析 WXT 光变曲线 REG00101 中的区域面积信息。

    只返回单个 `RegionArea`（优先 src，其次 bkg，再次首个条目），
    方便 lightcurve 直接记录代表性区域；若缺少 REGION 扩展则返回 None。
    """
    # 查找名为 REG00101 的扩展（大小写不敏感），若缺失则返回 None
    reg_hdu = None
    for ext in hdul:
        name = (getattr(ext, 'name', '') or '').upper()
        if name == 'REG00101':
            reg_hdu = ext
            break
    if reg_hdu is None or getattr(reg_hdu, 'data', None) is None:
        return None

    reg_any = cast(Any, reg_hdu)
    if getattr(reg_any, 'data', None) is None:
        return None
    data = reg_any.data
    cols = getattr(data, 'columns', None)
    colnames = [str(n).upper() for n in (cols.names if cols is not None else [])]
    # 允许多行；遍历所有区域定义
    def _get_col(name_variants: list[str]) -> Optional[str]:
        for nn in name_variants:
            if nn in colnames:
                return nn
        return None

    # 可能存在字符串列 SHAPE
    shape_col = _get_col(['SHAPE'])
    component_col = _get_col(['COMPONENT'])
    # 半径列的多种命名
    r_col = _get_col(['R', 'RADIUS', 'R0'])
    rin_col = _get_col(['R_IN', 'RIN', 'R1'])
    rout_col = _get_col(['R_OUT', 'ROUT', 'R2'])

    nrows = len(data)
    rows_info: list[Dict[str, Any]] = []  # {'shape': str, 'area': float|None, 'component': int|None, 'role': str}

    def _as_float(v: Any) -> Optional[float]:
        try:
            return float(v)
        except Exception:
            try:
                # v may be a 1-element array
                return float(v[0])
            except Exception:
                return None

    for i in range(nrows):
        # shape
        shape_val = ''
        if shape_col and shape_col in colnames:
            try:
                shape_val = str(data[shape_col][i]).upper().strip()
            except Exception:
                shape_val = ''
        # component if any
        comp_val = None
        if component_col and component_col in colnames:
            try:
                comp_val = int(data[component_col][i])
            except Exception:
                tmp = _as_float(data[component_col][i])
                if tmp is not None:
                    try:
                        comp_val = int(tmp)
                    except Exception:
                        comp_val = None
                else:
                    comp_val = None

        area = None
        # Circle
        if (shape_val == 'CIRCLE') or (shape_col is None and r_col is not None and (rin_col is None or rout_col is None)):
            rv = data[r_col][i] if r_col else None
            r = _as_float(rv) if rv is not None else None
            if r is not None and r > 0:
                area = np.pi * (r ** 2)
        # Annulus
        elif shape_val == 'ANNULUS' or (rin_col is not None and rout_col is not None):
            rin = _as_float(data[rin_col][i]) if rin_col else None
            rout = _as_float(data[rout_col][i]) if rout_col else None
            if rin is None and rout is None and r_col:
                # ANNULUS 可能用 R 列存储 [R_in, R_out]
                rv = data[r_col][i]
                try:
                    rin = float(rv[0])
                    rout = float(rv[1])
                except Exception:
                    rin = None
                    rout = None
            if (rin is not None) and (rout is not None) and rout > rin:
                area = np.pi * (rout ** 2 - rin ** 2)

        rows_info.append({'shape': shape_val, 'area': area, 'component': comp_val, 'role': 'unknown'})

    if not rows_info:
        return None

    annuli = [r for r in rows_info if r['shape'] == 'ANNULUS' and r['area']]
    circles = [r for r in rows_info if r['shape'] == 'CIRCLE' and r['area']]

    # 1) COMPONENT 指定了角色
    if component_col and any(r['component'] is not None for r in rows_info):
        for r in rows_info:
            comp = r['component']
            if comp == 1:
                r['role'] = 'source'
            elif comp is not None:
                r['role'] = 'background'

    # 2) 若没有 component 信息，使用几何启发：ANNULUS -> 背景，CIRCLE -> 源
    if not component_col:
        if annuli:
            for r in annuli:
                if r['role'] == 'unknown':
                    r['role'] = 'background'
            if circles:
                for r in circles:
                    if r['role'] == 'unknown':
                        r['role'] = 'source'
        elif len(circles) >= 1:
            sorted_c = sorted(circles, key=lambda x: x['area'] or 0)
            if sorted_c:
                if sorted_c[0]['role'] == 'unknown':
                    sorted_c[0]['role'] = 'source'
                for r in sorted_c[1:]:
                    if r['role'] == 'unknown':
                        r['role'] = 'background'

    # 3) 若仍未标记，但只有一个区域，则视作源区
    if len(rows_info) == 1 and rows_info[0]['role'] == 'unknown':
        rows_info[0]['role'] = 'source'

    def _normalize_role(role: str) -> Literal['src', 'bkg', 'unk']:
        if role == 'source':
            return 'src'
        if role == 'background':
            return 'bkg'
        return 'unk'

    regions: list[RegionArea] = []
    for r in rows_info:
        regions.append(
            RegionArea(
                role=_normalize_role(cast(Any, r['role'])),
                shape=r['shape'] or None,
                area=r['area'],
                component=r['component'],
            )
        )

    if not regions:
        return None

    src_region = next((r for r in regions if r.role == 'src'), None)
    if src_region is not None:
        return src_region
    bkg_region = next((r for r in regions if r.role == 'bkg'), None)
    if bkg_region is not None:
        return bkg_region
    return regions[0]


class OgipEventReader:
    """事件表读取器

    English
    Reader for OGIP-like event lists (EVENTS) with optional GTI.
    """
    def __init__(self, path: str | Path):
        """参数
        - path: 事件表 FITS 文件路径

        English
        - path: path to event FITS file
        """
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(str(self.path))
        self._data: Optional[EventData] = None

    def read(self) -> EventData:
        """读取事件表并返回 `EventData`。

        English: Read EVENTS table and return `EventData`.
        """
        with fits.open(self.path) as h:
            # EVENTS
            # Find first binary table HDU with TIME column
            hevt = None
            for ext in h:
                ext_any = cast(Any, ext)
                if getattr(ext_any, "data", None) is None:
                    continue
                names = getattr(ext_any.data, "columns", None)
                if names is not None and ("TIME" in names.names):
                    hevt = ext_any
                    break
            if hevt is None:
                raise ValueError("No EVENTS-like HDU with TIME column found")
            de = cast(Any, hevt.data)
            time = np.asarray(de["TIME"], float)
            pi = np.asarray(de["PI"], int) if "PI" in de.columns.names else None
            channel = np.asarray(de["CHANNEL"], int) if "CHANNEL" in de.columns.names else None
            header = dict(cast(Any, hevt.header))
            headers_dump = _collect_headers_dump(h)
            meta = _build_meta(h, header)

            # GTI
            gti_start = gti_stop = None
            if "GTI" in h:
                hg = cast(Any, h["GTI"]).data
                gti_start = np.asarray(hg["START"], float)
                gti_stop = np.asarray(hg["STOP"], float)

        self._data = EventData(
            kind='evt', path=self.path,
            time=time, pi=pi, channel=channel,
            gti_start=gti_start, gti_stop=gti_stop,
            header=header, meta=meta, headers_dump=headers_dump,
        )
        return self._data

    def validate(self) -> ValidationReport:
        if self._data is None:
            self.read()
        assert self._data is not None
        return self._data.validate()


# ---------- Backward-compat aliases ----------

class ArfReader(OgipArfReader):
    """Backward-compatible alias for ARF reader."""
    pass


class RmfReader(OgipRmfReader):
    """Backward-compatible alias for RMF reader."""
    pass


class RspReader(OgipRmfReader):
    """Alias for response reader; RSP often stores MATRIX+EBOUNDS akin to RMF."""
    pass


class LightcurveReader(OgipLightcurveReader):
    """Backward-compatible alias for lightcurve reader."""
    pass


# ---------- Kind inference ----------

def guess_ogip_kind(path: str | Path) -> Literal['arf', 'rmf', 'pha', 'lc', 'evt']:
    """推断给定 FITS 文件的 OGIP 类型

    规则
    - 先基于文件名后缀（.arf/.rmf/.rsp/.pha/.pi）快速判断；
    - 再基于扩展名（SPECRESP/MATRIX/SPECTRUM/EVENTS 等）与列名（TIME/RATE/COUNTS）判定；
    - 在 LC/EVT 模糊时，若存在 RATE/COUNTS 列则判为 LC，否则 EVT。

    English
    Guess the OGIP kind of a FITS file using filename hints and HDU contents.
    """
    p = Path(path)
    name = p.name.lower()
    # quick filename-based hints
    if name.endswith('.arf'):
        return 'arf'
    if name.endswith('.rmf') or name.endswith('.rsp'):
        return 'rmf'
    if name.endswith('.pha') or name.endswith('.pi'):
        return 'pha'
    # Content-based fallback
    with fits.open(p) as h:
        # Prefer extension name checks
        extnames = {getattr(x, 'name', '').upper() for x in h}
        if 'SPECRESP' in extnames and 'MATRIX' not in extnames:
            return 'arf'
        if 'MATRIX' in extnames:
            return 'rmf'
        if 'SPECTRUM' in extnames:
            return 'pha'
        # Look for any table that has TIME column (events or lc)
        has_time = False
        for x in h:
            d = getattr(x, 'data', None)
            cols = getattr(d, 'columns', None)
            names = getattr(cols, 'names', ()) if cols is not None else ()
            if 'TIME' in names:
                has_time = True
                break
        if 'EVENTS' in extnames or has_time:
            # Ambiguity between LC and EVT: try RATE/COUNTS columns
            for x in h:
                d = getattr(x, 'data', None)
                cols = getattr(d, 'columns', None)
                names = getattr(cols, 'names', ()) if cols is not None else ()
                if 'RATE' in names or 'COUNTS' in names:
                    return 'lc'
            return 'evt'
    # default to PHA if uncertain
    return 'pha'


# ---------- Direct dataclass-returning helpers ----------

@overload
def read_fits(path: str | Path, kind: Literal['arf']) -> ArfData: ...

@overload
def read_fits(path: str | Path, kind: Literal['rmf']) -> RmfData: ...

@overload
def read_fits(path: str | Path, kind: Literal['pha']) -> PhaData: ...

@overload
def read_fits(path: str | Path, kind: Literal['lc']) -> LightcurveData: ...

@overload
def read_fits(path: str | Path, kind: Literal['evt']) -> EventData: ...

@overload
def read_fits(path: str | Path, kind: None = ...) -> OgipData: ...

def read_fits(path: str | Path, kind: Optional[Literal['arf','rmf','pha','lc','evt']] = None) -> OgipData:
    """统一入口：直接返回具体 OGIP 数据类（Chinese first）

    行为
    - 若提供 kind，则按 kind 调用相应读取器；
    - 若未提供，则通过 `guess_ogip_kind` 自动推断。

    返回
    - 对应的具体数据类：`ArfData`/`RmfData`/`PhaData`/`LightcurveData`/`EventData`

    English
    Unified reader returning the concrete dataclass directly. When `kind`
    is None, the function attempts to infer it using `guess_ogip_kind`.
    """
    k = kind or guess_ogip_kind(path)
    if k == 'arf':
        return OgipArfReader(path).read()
    if k == 'rmf':
        return OgipRmfReader(path).read()
    if k == 'pha':
        return OgipPhaReader(path).read()
    if k == 'lc':
        return OgipLightcurveReader(path).read()
    if k == 'evt':
        return OgipEventReader(path).read()
    raise ValueError(f"Unknown OGIP kind: {k}")


def read_arf(path: str | Path) -> ArfData:
    """简便函数：读取 ARF 并返回 `ArfData`（English: convenience ARF reader）。"""
    return OgipArfReader(path).read()


def read_rmf(path: str | Path) -> RmfData:
    """简便函数：读取 RMF 并返回 `RmfData`（English: convenience RMF reader）。"""
    return OgipRmfReader(path).read()


def read_pha(path: str | Path) -> PhaData:
    """简便函数：读取 PHA 并返回 `PhaData`（English: convenience PHA reader）。"""
    return OgipPhaReader(path).read()


def read_lc(path: str | Path) -> LightcurveData:
    """简便函数：读取光变曲线并返回 `LightcurveData`（English: convenience LC reader）。"""
    return OgipLightcurveReader(path).read()


def read_evt(path: str | Path) -> EventData:
    """简便函数：读取事件表并返回 `EventData`（English: convenience EVT reader）。"""
    return OgipEventReader(path).read()



