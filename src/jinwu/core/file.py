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

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, cast, Union, Literal, overload

import numpy as np
from astropy.io import fits

__all__ = [
    # Data containers
    "EnergyBand", "ChannelBand",
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
    # Direct dataclass-returning helpers
    "read_ogip", "read_arf", "read_rmf", "read_pha", "read_lc", "read_evt",
]


# ---------- Data containers ----------

@dataclass
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

@dataclass
class ChannelBand:
    """道（Channel）范围数据类

    字段
    - ch_lo/ch_hi: 起止道号（闭区间）

    English
    - ch_lo/ch_hi: inclusive channel range
    """
    ch_lo: int
    ch_hi: int

@dataclass
class ArfData:
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
    path: Path
    energ_lo: np.ndarray  # shape (N,)
    energ_hi: np.ndarray  # shape (N,)
    specresp: np.ndarray  # shape (N,), cm^2
    header: Dict[str, Any]

@dataclass
class RmfData:
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
    path: Path
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
    header: Dict[str, Any]

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

@dataclass
class PhaData:
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
    path: Path
    channels: np.ndarray  # shape (N,)
    counts: np.ndarray    # shape (N,)
    stat_err: Optional[np.ndarray]
    exposure: float
    backscal: Optional[float]
    areascal: Optional[float]
    quality: Optional[np.ndarray]
    grouping: Optional[np.ndarray]
    ebounds: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]  # (CHANNEL, E_MIN, E_MAX)
    header: Dict[str, Any]

@dataclass
class LightcurveData:
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
    path: Path
    time: np.ndarray   # bin left edges or centers
    value: np.ndarray  # RATE or COUNTS
    error: Optional[np.ndarray]
    dt: Optional[float]
    exposure: Optional[float]
    is_rate: bool
    header: Dict[str, Any]

@dataclass
class EventData:
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
    path: Path
    time: np.ndarray   # events times (s)
    pi: Optional[np.ndarray]       # PI or CHANNEL if present
    channel: Optional[np.ndarray]
    gti_start: Optional[np.ndarray]
    gti_stop: Optional[np.ndarray]
    header: Dict[str, Any]


# Unified data union
OgipData = Union[ArfData, RmfData, PhaData, LightcurveData, EventData]


# ---------- Utilities ----------

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
        self._data = ArfData(kind='arf', path=self.path, energ_lo=energ_lo, energ_hi=energ_hi, specresp=specresp, header=header)
        return self._data


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
        )
        return self._data


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
        self._data = PhaData(kind='pha', path=self.path, channels=channels, counts=counts, stat_err=stat_err, exposure=exposure, backscal=backscal, areascal=areascal, quality=quality, grouping=grouping, ebounds=ebounds, header=header)
        return self._data

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
            # Infer dt if possible
            dt = float(np.median(np.diff(time))) if time.size >= 2 else None
            exposure_val = header.get("EXPOSURE", header.get("EXPTIME", np.nan))
            exposure = float(exposure_val) if np.isfinite(exposure_val) else None
        self._data = LightcurveData(kind='lc', path=self.path, time=time, value=val, error=err, dt=dt, exposure=exposure, is_rate=is_rate, header=header)
        return self._data


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

            # GTI
            gti_start = gti_stop = None
            if "GTI" in h:
                hg = cast(Any, h["GTI"]).data
                gti_start = np.asarray(hg["START"], float)
                gti_stop = np.asarray(hg["STOP"], float)

        self._data = EventData(kind='evt', path=self.path, time=time, pi=pi, channel=channel, gti_start=gti_start, gti_stop=gti_stop, header=header)
        return self._data


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
def read_ogip(path: str | Path, kind: Literal['arf']) -> ArfData: ...

@overload
def read_ogip(path: str | Path, kind: Literal['rmf']) -> RmfData: ...

@overload
def read_ogip(path: str | Path, kind: Literal['pha']) -> PhaData: ...

@overload
def read_ogip(path: str | Path, kind: Literal['lc']) -> LightcurveData: ...

@overload
def read_ogip(path: str | Path, kind: Literal['evt']) -> EventData: ...

@overload
def read_ogip(path: str | Path, kind: None = ...) -> OgipData: ...

def read_ogip(path: str | Path, kind: Optional[Literal['arf','rmf','pha','lc','evt']] = None) -> OgipData:
    """统一入口：直接返回具体数据类（Chinese first）

    行为
    - 若提供 kind，则按 kind 调用相应读取器；
    - 若未提供，则通过 `guess_ogip_kind` 自动推断。

    返回
    - 对应的具体数据类：`ArfData`/`RmfData`/`PhaData`/`LightcurveData`/`EventData`

    English
    Unified reader that returns the concrete dataclass directly. When `kind`
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



