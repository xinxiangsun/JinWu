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
from typing import Optional, Tuple, Dict, Any, cast, Union, Literal, overload, ClassVar, TYPE_CHECKING
from astropy.time import TimeDelta
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
    "readfits", "read_arf", "read_rmf", "read_pha", "read_lc", "read_evt",
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
    trefpos: Optional[str]
    dateobs: Optional[str]
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
    kind: ClassVar[Literal['arf']] = 'arf'
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

    # --- Plot helper ---
    def plot(self, ax: Optional[Any] = None, *, energy_unit: str = 'keV', yscale: str = 'linear', title: Optional[str] = None, **kwargs):
        """绘制 ARF 有效面积曲线。

        参数
        - ax: 复用的 matplotlib Axes；若为 None 自动创建
        - energy_unit: 仅标签展示（本身数据默认 keV）
        - yscale: 'linear' 或 'log'
        - title: 自定义标题；默认使用文件名
        - kwargs: 传递给 matplotlib.plot
        """
        import matplotlib.pyplot as _plt  # lazy import
        mid_e = 0.5 * (np.asarray(self.energ_lo) + np.asarray(self.energ_hi))
        ax = ax or _plt.gca()
        assert ax is not None
        kwargs.setdefault('lw', 1.0)
        ax.plot(mid_e, self.specresp, **kwargs)
        ax.set_xlabel(f"Energy ({energy_unit})")
        ax.set_ylabel("Effective Area (cm$^2$)")
        ax.set_yscale(yscale)
        if title is None:
            title = Path(str(getattr(self, 'path', ''))).name or 'ARF'
        ax.set_title(title)
        ax.grid(alpha=0.3, ls='--')
        return ax

    def rebin(self, factor: int) -> 'ArfData':
        """Rebin ARF energy bins by integer factor. Delegates to :func:`jinwu.core.ops.rebin_arf`.

        This merges consecutive energy bins in groups of `factor` and returns
        a new `ArfData` instance.
        """
        # Merge consecutive energy bins by integer factor.
        if factor <= 0:
            raise ValueError('factor must be > 0')
        from ..ftools.ftrbnrmf import rebin_arf
        elo = np.asarray(self.energ_lo, dtype=float)
        ehi = np.asarray(self.energ_hi, dtype=float)
        area = np.asarray(self.specresp, dtype=float)
        n = elo.size
        # compute new bin edges by grouping consecutive bins
        groups = [(i, min(i+factor, n)) for i in range(0, n, factor)]
        new_elo = np.array([elo[s] for s, e in groups], dtype=float)
        new_ehi = np.array([ehi[e-1] for s, e in groups], dtype=float)
        new_area = rebin_arf(elo, ehi, area, new_elo, new_ehi)
        return ArfData(path=self.path, energ_lo=new_elo, energ_hi=new_ehi, specresp=new_area,
                   header=self.header, meta=self.meta, headers_dump=self.headers_dump)

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
    kind: ClassVar[Literal['rmf']] = 'rmf'
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

    def plot(self, ax: Optional[Any] = None, *, kind: str = 'matrix', row: int = 0, yscale: str = 'linear', cmap: str = 'viridis', title: Optional[str] = None, **kwargs):
        """绘制 RMF 响应：

        kind='matrix' 显示二维响应矩阵；kind='row' 显示单行（某入射能对应的分布）。
        - ax: 复用 Axes
        - row: kind='row' 时选择的能 bin 索引
        - yscale: 行模式下的 y 轴尺度
        - cmap: 矩阵模式色图
        - kwargs: 传递给 imshow 或 plot
        """
        import matplotlib.pyplot as _plt
        ax = ax or _plt.gca()
        assert ax is not None
        if kind == 'matrix':
            dm = self.dense_matrix
            im = ax.imshow(dm, aspect='auto', origin='lower', cmap=cmap, **kwargs)
            ax.set_xlabel('Channel')
            ax.set_ylabel('Energy bin')
            if title is None:
                title = Path(str(getattr(self, 'path', ''))).name or 'RMF'
            ax.set_title(title)
            ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Prob.')
        else:
            dm = self.dense_matrix
            row = max(0, min(dm.shape[0]-1, int(row)))
            kwargs.setdefault('lw', 1.0)
            ax.plot(dm[row], **kwargs)
            ax.set_xlabel('Channel')
            ax.set_ylabel('Probability')
            ax.set_yscale(yscale)
            if title is None:
                title = f"RMF row {row}"
            ax.set_title(title)
        ax.grid(alpha=0.3, ls='--')
        return ax

    def rebin(self, factor: int) -> 'RmfData':
        """Rebin RMF input energy bins by integer factor. Delegates to :func:`jinwu.core.ops.rebin_rmf`.

        This collapses consecutive energy bins (rows) by `factor` and returns
        a new `RmfData` instance with aggregated matrix/energy bounds.
        """
        if factor <= 0:
            raise ValueError('factor must be > 0')
        # Rebin energy rows by integer factor: aggregate consecutive rows and sum
        # their probability distributions to form new rows.
        dense = self.rebuild_dense()
        n_e, n_c = dense.shape
        groups = [(i, min(i+factor, n_e)) for i in range(0, n_e, factor)]
        new_rows = []
        new_elo = []
        new_ehi = []
        for s, e in groups:
            summed = np.sum(dense[s:e, :], axis=0)
            new_rows.append(summed)
            new_elo.append(float(self.energ_lo[s]))
            new_ehi.append(float(self.energ_hi[e-1]))
        new_matrix = np.vstack(new_rows)
        return RmfData(path=self.path, energ_lo=np.asarray(new_elo, dtype=float), energ_hi=np.asarray(new_ehi, dtype=float),
                   n_grp=None, f_chan=None, n_chan=None, matrix=new_matrix, channel=self.channel, e_min=self.e_min, e_max=self.e_max,
                   header=self.header, meta=self.meta, headers_dump=self.headers_dump)

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
    kind: ClassVar[Literal['pha']] = 'pha'
    channels: np.ndarray  # shape (N,)
    counts: np.ndarray    # shape (N,)
    stat_err: Optional[np.ndarray] = None
    exposure: float = 0.0
    backscal: Optional[float] = None
    areascal: Optional[float] = None
    respfile: Optional[str] = None
    ancrfile: Optional[str] = None
    quality: Optional[np.ndarray] = None
    grouping: Optional[np.ndarray] = None
    ebounds: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None  # (CHANNEL, E_MIN, E_MAX)
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

    def plot(self, **kwargs):
        """委托到 plot.plot_spectrum 进行标准 PHA 绘图。

        kwargs 透传给 plot_spectrum，例如 ykind / show_errorbar / ax。
        """
        from .plot import plot_spectrum  # lazy to avoid circular import at module import time
        return plot_spectrum(self, **kwargs)

    def slice(self, *, emin: Optional[float] = None, emax: Optional[float] = None, ch_lo: Optional[int] = None, ch_hi: Optional[int] = None) -> 'PhaData':
        """按能量或道范围筛选 PHA，返回新实例。委托到 ops.slice_pha。"""
        from .ops import slice_pha
        return slice_pha(self, emin=emin, emax=emax, ch_lo=ch_lo, ch_hi=ch_hi)

    def rebin(self, *, factor: Optional[int] = None, min_counts: Optional[float] = None) -> 'PhaData':
        """道聚合：按固定因子或最小计数阈值合并道。委托到 ops.rebin_pha。
        
        参数
        - factor: 固定聚合因子（如 2 表示两两合并）；与 min_counts 互斥
        - min_counts: 基于最小计数阈值聚合；若既未提供 factor 也未提供 min_counts，
                      将使用已有的 pha.grouping（若存在）
        
        English
        Rebin PHA by factor or min_counts; delegates to ops.rebin_pha.
        """
        from .ops import rebin_pha
        return rebin_pha(self, factor=factor, min_counts=min_counts)

    def grppha(self, *, min_counts: Optional[float] = None, groupfile: Optional[str] = None, rebin: bool = False, outfile: Optional[str] = None, overwrite: bool = False) -> 'PhaData':
        """Convenience wrapper to run grppha-like grouping on this PhaData.

        Parameters
        - min_counts: minimum counts per group (greedy)
        - groupfile: explicit group ranges file
        - rebin: if True, return a rebinned PhaData (one row per group)
        - outfile: optional path to write grouped/rebinned PHA
        - overwrite: whether to overwrite outfile

        Returns
        - PhaData: grouped or rebinned spectrum
        """
        from ..ftools.grppha import grppha as _grppha
        return _grppha(self, min_counts=min_counts, groupfile=groupfile, rebin=rebin, outfile=outfile, overwrite=overwrite)

    def group_by_min_counts(self, min_counts: float) -> np.ndarray:
        """Compute grouping array (int per-channel group id) using min_counts."""
        from ..ftools.grppha import compute_grouping_by_min_counts
        return compute_grouping_by_min_counts(self.counts, min_counts)

@dataclass(slots=True)
class LightcurveData(OgipTimeSeriesBase):
    """光变曲线数据 (OGIP-93-003 兼容)

    **核心字段设计（遵循 OGIP TIMEZERO 标准）**
    
    时间处理遵循 OGIP-93-003 规范：
    - time: FITS TIME 列的原始值（可能是相对时间 0,1,2... 或绝对 MET）
    - timezero: TIMEZERO 关键字值（时间偏移，单位秒）
    - 绝对时间 = time + timezero
    
    两种常见模式：
    1. **绝对时间模式**: TIMEZERO=0, TIME=MET（如123456.78）
    2. **相对时间模式**: TIMEZERO=MET_START, TIME=相对秒数（0,1,2...）
    
    时间坐标系统：
    - time: TIME 列原始值（核心字段）
    - timezero: TIMEZERO 偏移量（默认0）
    - timezero_obj: 根据 TELESCOP 自动生成的 Time 对象（用于时间转换）
    - dt: bin 宽度（秒，标量或数组）
    - bin_lo, bin_hi: 时间箱边界（便利字段，= time ± dt/2）
    
    计数/速率数据：
    - value: 主要数据列（RATE 或 COUNTS，核心字段）
    - error: 误差列（核心字段）
    - is_rate: 标记 value 是速率还是计数
    - counts, rate: 分离存储（可选，用于同时保存两者）
    
    GTI 与质量：
    - gti_start, gti_stop: GTI 时间数组
    - quality: 质量标志
    - fracexp: 分数曝光
    - backscal, areascal: 刻度因子
    
    时间系统元数据：
    - telescop: 望远镜名称（用于自动生成 timezero_obj）
    - timesys: 时间系统（'TT', 'UTC'等）
    - mjdref: MJD 参考（可选，不是核心）
    
    其他：
    - exposure: 总曝光时间
    - region: 区域信息
    
    English: OGIP-93-003 compliant lightcurve with proper TIMEZERO handling.
    Two modes supported: absolute (TIMEZERO=0, TIME=MET) and relative 
    (TIMEZERO=MET_START, TIME=0,1,2...). Automatically generates Time object
    based on TELESCOP for time conversions.
    """
    
    kind: ClassVar[Literal['lc']] = 'lc'
    
    # ========== 核心必需字段（读取时必须填充，但给默认值以兼容 dataclass 继承）==========
    time: Optional[np.ndarray] = None       # TIME 列原始值（相对或绝对）
    value: Optional[np.ndarray] = None      # 主要数据（RATE 或 COUNTS）
    
    # ========== 核心时间字段（有默认值）==========
    timezero: float = 0.0                   # TIMEZERO 偏移量（秒）
    timezero_obj: Optional[Any] = None      # Time 对象（根据 TELESCOP 生成）
    dt: Optional[np.ndarray | float] = None # bin 宽度（秒）
    
    # ========== 便利时间字段 ==========
    bin_lo: Optional[np.ndarray] = None     # bin 左缘（time - dt/2）
    bin_hi: Optional[np.ndarray] = None     # bin 右缘（time + dt/2）
    tstart: Optional[float] = None          # 观测起始（绝对时间）
    tseg: Optional[float] = None            # 观测总时长
    
    # ========== 核心数据字段 ==========
    error: Optional[np.ndarray] = None      # 误差
    is_rate: bool = False                   # value 是速率还是计数
    
    # ========== 分离存储（可选，用于同时保存计数和速率）==========
    counts: Optional[np.ndarray] = None     # 原始计数
    rate: Optional[np.ndarray] = None       # 计数速率
    counts_err: Optional[np.ndarray] = None # 计数误差
    rate_err: Optional[np.ndarray] = None   # 速率误差
    err_dist: Optional[Literal['poisson', 'gauss']] = None
    
    # ========== GTI 与质量 ==========
    gti_start: Optional[np.ndarray] = None  # GTI 起始（绝对时间）
    gti_stop: Optional[np.ndarray] = None   # GTI 结束（绝对时间）
    quality: Optional[np.ndarray] = None    # 质量标志
    fracexp: Optional[np.ndarray] = None    # 分数曝光
    backscal: Optional[np.ndarray | float] = None
    areascal: Optional[np.ndarray | float] = None
    
    # ========== 时间系统元数据 ==========
    telescop: Optional[str] = None          # 望远镜名称（用于时间转换）
    timesys: Optional[str] = None           # 时间系统
    mjdref: Optional[float] = None          # MJD 参考（可选）
    
    # ========== 其他 ==========
    exposure: Optional[float] = None
    bin_exposure: Optional[np.ndarray] = None
    region: Optional[RegionArea] = None
    columns: Tuple[str, ...] = ()
    ratio:Optional[float]=None

    def validate(self) -> ValidationReport:  # type: ignore[override]
        rpt = super().validate()
        colset = {c.upper() for c in self.columns}
        if "TIME" not in colset:
            rpt.add('ERROR', 'MISSING_COLUMN', "Lightcurve missing TIME column")
        if not any(x in colset for x in ["RATE", "COUNTS"]):
            rpt.add('ERROR', 'MISSING_COLUMN', "Lightcurve missing RATE/COUNTS column")
        
        # 验证核心时间字段
        if self.time is None or len(self.time) == 0:
            rpt.add('ERROR', 'MISSING_TIME', "time array is empty or None")
        
        # 验证核心数据字段
        if self.value is None or len(self.value) == 0:
            rpt.add('ERROR', 'MISSING_VALUE', "value array is empty or None")
        
        # 验证 time 和 value 长度一致
        if self.time is not None and self.value is not None:
            if len(self.time) != len(self.value):
                rpt.add('ERROR', 'LENGTH_MISMATCH', f"time ({len(self.time)}) and value ({len(self.value)}) length mismatch")
        
        # 验证 GTI（若存在）
        if self.gti_start is not None and self.gti_stop is not None:
            if len(self.gti_start) != len(self.gti_stop):
                rpt.add('ERROR', 'GTI_MISMATCH', "gti_start and gti_stop must have same length")
            if len(self.gti_start) > 0 and not np.all(self.gti_start < self.gti_stop):
                rpt.add('ERROR', 'GTI_ORDER', "gti_start must be < gti_stop")
        
        # 验证时间单调性（警告级别）
        if self.time is not None and len(self.time) > 1:
            if not np.all(np.diff(self.time) > 0):
                rpt.add('WARN', 'TIME_NOT_SORTED', "time array is not strictly increasing")
        
        # 验证曝光时间
        if self.exposure is not None and self.exposure <= 0:
            rpt.add('WARN', 'BAD_EXPOSURE', f"exposure must be > 0, got {self.exposure}")
        
        rpt.ok = len(rpt.errors()) == 0
        return rpt

    # ========== 便利属性 ==========
    
    @property
    def n(self) -> int:
        """数据点数"""
        return len(self.time) if self.time is not None else 0
    
    @property
    def absolute_time(self) -> np.ndarray:
        """绝对时间 = time + timezero（OGIP 标准）"""
        return self.time + self.timezero
    
    @property
    def gti(self) -> Optional[list[tuple[float, float]]]:
        """GTI 元组列表 [(start0, stop0), ...]"""
        if self.gti_start is None or self.gti_stop is None:
            return None
        return [(float(s), float(e)) for s, e in zip(self.gti_start, self.gti_stop)]
    
    def get_time_object(self, index: Optional[int] = None) -> Optional[Any]:
        """获取 Time 对象（用于时间转换）
        
        参数：
        - index: 若指定，返回该索引的时间对象；否则返回所有时间的数组
        
        返回：
        - astropy.time.Time 对象或 None
        """
        if self.timezero_obj is None:
            return None
        
        
        absolute_times = self.absolute_time
        
        if index is not None:
            dt = TimeDelta(absolute_times[index], format='sec')
        else:
            dt = TimeDelta(absolute_times, format='sec')
        
        return self.timezero_obj + dt

    # ========== 便利属性与转换 ==========
    
    @property
    def bin_centers(self) -> Optional[np.ndarray]:
        """Bin 中心时刻 = (bin_lo + bin_hi) / 2"""
        if self.bin_lo is None or self.bin_hi is None:
            return None
        return 0.5 * (self.bin_lo + self.bin_hi)
    
    @property
    def mean_rate(self) -> Optional[float]:
        """平均计数速率 (counts/sec)"""
        if self.rate is None:
            return None
        if len(self.rate) == 0:
            return None
        return float(np.mean(self.rate))
    
    @property
    def mean_counts(self) -> Optional[float]:
        """平均计数"""
        if self.counts is None:
            return None
        if len(self.counts) == 0:
            return None
        return float(np.mean(self.counts))
    
    @property
    def total_counts(self) -> Optional[float]:
        """总计数"""
        if self.counts is None:
            return None
        return float(np.sum(self.counts))
    
    # ========== 废弃属性（向后兼容）==========
    
    @property
    def _legacy_value(self) -> Optional[np.ndarray]:
        """Deprecated: access counts or rate directly instead."""
        import warnings
        warnings.warn(
            "LightcurveData.value is deprecated; use counts or rate instead.",
            DeprecationWarning, stacklevel=2
        )
        if self.counts is not None:
            return self.counts
        elif self.rate is not None:
            if self.dt is not None and np.any(self.dt > 0):
                return self.rate * (self.dt if isinstance(self.dt, np.ndarray) else self.dt)
        return None
    
    @property
    def _legacy_error(self) -> Optional[np.ndarray]:
        """Deprecated: use counts_err or rate_err instead."""
        import warnings
        warnings.warn(
            "LightcurveData.error is deprecated; use counts_err or rate_err instead.",
            DeprecationWarning, stacklevel=2
        )
        if self.counts_err is not None:
            return self.counts_err
        elif self.rate_err is not None and self.dt is not None:
            return self.rate_err * self.dt
        return None
    
    @property
    def _legacy_is_rate(self) -> bool:
        """Deprecated: check counts/rate fields directly."""
        import warnings
        warnings.warn(
            "LightcurveData.is_rate is deprecated; check counts/rate fields instead.",
            DeprecationWarning, stacklevel=2
        )
        return self.rate is not None and (self.counts is None or self.counts.sum() == 0)
    # ========== 核心操作方法（GTI 感知）==========
    
    def apply_gti(self, inplace: bool = False) -> 'LightcurveData':
        """按 GTI 过滤 bin，返回新的 LightcurveData。
        
        参数
        - inplace: 若 True 则原地修改；否则返回新对象
        
        返回
        - LightcurveData: 过滤后的光变曲线
        """
        if self.gti is None:
            return self if inplace else self._copy()
        
        from stingray.gti import create_gti_mask
        gti_arr = np.array(self.gti)
        mask = create_gti_mask(self.bin_lo, gti_arr, dt=self.dt if isinstance(self.dt, float) else None)
        return self.apply_mask(mask, inplace=inplace)
    
    def split_by_gti(self, min_points: int = 1) -> list['LightcurveData']:
        """按 GTI 分割光变曲线为多个独立对象。
        
        参数
        - min_points: 最小数据点数，少于此值的片段会被忽略
        
        返回
        - list[LightcurveData]: GTI 段对应的光变曲线列表
        """
        if self.gti is None or len(self.gti) == 0:
            return [self]
        
        from stingray.gti import create_gti_mask
        result = []
        
        for start, stop in self.gti:
            gti_arr = np.array([[start, stop]])
            mask = create_gti_mask(self.bin_lo, gti_arr, dt=self.dt if isinstance(self.dt, float) else None)
            
            if np.sum(mask) < min_points:
                continue
            
            lc_segment = self.apply_mask(mask, inplace=False)
            # 更新该段的 GTI
            lc_segment.gti_start = np.array([start])
            lc_segment.gti_stop = np.array([stop])
            result.append(lc_segment)
        
        return result if result else [self]
    
    def apply_mask(self, mask: np.ndarray, inplace: bool = False, 
                   filtered_attrs: Optional[list[str]] = None) -> 'LightcurveData':
        """按布尔掩码过滤 bin。
        
        参数
        - mask: 布尔数组，长度必须等于 n
        - inplace: 若 True 则原地修改；否则返回新对象
        - filtered_attrs: 要保留的数组属性列表；若 None 则保留全部
        
        返回
        - LightcurveData: 过滤后的光变曲线
        """
        import copy
        
        # 定义所有可过滤的数组属性
        all_array_attrs = [
            'bin_lo', 'bin_hi', 'counts', 'rate', 'counts_err', 'rate_err',
            'quality', 'fracexp', 'bin_exposure'
        ]
        if filtered_attrs is None:
            filtered_attrs = all_array_attrs
        
        if inplace:
            lc_new = self
        else:
            lc_new = self._copy()
        
        # 过滤数组属性
        for attr in all_array_attrs:
            val = getattr(lc_new, attr, None)
            if val is None:
                continue
            if attr not in filtered_attrs:
                setattr(lc_new, attr, None)
            else:
                try:
                    filtered_val = np.asanyarray(val)[mask]
                    setattr(lc_new, attr, filtered_val)
                except (IndexError, TypeError):
                    pass
        
        # 重新计算 tstart, tseg
        if lc_new.bin_lo is not None and len(lc_new.bin_lo) > 0:
            lc_new.tstart = float(lc_new.bin_lo[0])
            if lc_new.bin_hi is not None:
                lc_new.tseg = float(lc_new.bin_hi[-1] - lc_new.bin_lo[0])
        
        return lc_new
    
    def join(self, other: 'LightcurveData', skip_checks: bool = False) -> 'LightcurveData':
        """合并两条光变曲线为一个对象。
        
        参数
        - other: 另一条 LightcurveData
        - skip_checks: 若 True 则跳过数据检查
        
        返回
        - LightcurveData: 合并后的光变曲线
        
        说明
        - 若两条 LC 的 mjdref 不同，会发出警告并转换 other 到 self 的 mjdref
        - 若时间范围重叠，会发出信息提示
        """
        import warnings
        import copy
        
        if self.mjdref is not None and other.mjdref is not None and self.mjdref != other.mjdref:
            warnings.warn(
                f"MJDref mismatch: self={self.mjdref}, other={other.mjdref}. "
                f"Converting other to self's mjdref.",
                UserWarning
            )
            other = copy.deepcopy(other)
            # 简单方案：调整 time 偏移（实际应转换到绝对时间）
            # 已经在上面的 if 中检查过非 None，这里直接使用
            assert other.mjdref is not None and self.mjdref is not None
            time_offset = (other.mjdref - self.mjdref) * 86400.0
            other.bin_lo = other.bin_lo + time_offset
            other.bin_hi = other.bin_hi + time_offset
            if other.gti_start is not None:
                other.gti_start = other.gti_start + time_offset
            if other.gti_stop is not None:
                other.gti_stop = other.gti_stop + time_offset
            other.mjdref = self.mjdref
        
        # 确定顺序
        if self.tstart is not None and other.tstart is not None and self.tstart < other.tstart:
            first_lc = self
            second_lc = other
        else:
            first_lc = other
            second_lc = self
        
        # 检查重叠
        if len(np.intersect1d(self.bin_lo, other.bin_lo)) > 0:
            warnings.warn(
                "The two light curves have overlapping time ranges. "
                "In overlapping regions, counts will be summed.",
                UserWarning
            )
        
        # 合并数据
        new_bin_lo = np.concatenate([first_lc.bin_lo, second_lc.bin_lo])
        new_bin_hi = np.concatenate([first_lc.bin_hi, second_lc.bin_hi])
        new_counts = np.concatenate([first_lc.counts, second_lc.counts]) if (first_lc.counts is not None and second_lc.counts is not None) else None
        new_rate = np.concatenate([first_lc.rate, second_lc.rate]) if (first_lc.rate is not None and second_lc.rate is not None) else None
        new_counts_err = np.concatenate([first_lc.counts_err, second_lc.counts_err]) if (first_lc.counts_err is not None and second_lc.counts_err is not None) else None
        new_rate_err = np.concatenate([first_lc.rate_err, second_lc.rate_err]) if (first_lc.rate_err is not None and second_lc.rate_err is not None) else None
        
        # 合并 GTI
        if first_lc.gti is not None and second_lc.gti is not None:
            from stingray.gti import join_gtis
            new_gti = join_gtis(np.array(first_lc.gti), np.array(second_lc.gti))
            new_gti_start = new_gti[:, 0]
            new_gti_stop = new_gti[:, 1]
        else:
            new_gti_start = None
            new_gti_stop = None
        
        lc_new = LightcurveData(
            path=self.path,
            bin_lo=new_bin_lo, bin_hi=new_bin_hi,
            counts=new_counts, rate=new_rate,
            counts_err=new_counts_err, rate_err=new_rate_err,
            dt=self.dt, tstart=float(new_bin_lo[0]),
            gti_start=new_gti_start, gti_stop=new_gti_stop,
            mjdref=self.mjdref, timesys=self.timesys,
            exposure=self.exposure, header=self.header,
            meta=self.meta, headers_dump=self.headers_dump, columns=self.columns
        )
        lc_new.tseg = float(new_bin_hi[-1] - new_bin_lo[0]) if len(new_bin_hi) > 0 else None
        return lc_new
    
    def truncate(self, tmin: Optional[float] = None, tmax: Optional[float] = None) -> 'LightcurveData':
        """时间范围裁剪。
        
        参数
        - tmin: 最小时刻；若 None 则不限
        - tmax: 最大时刻；若 None 则不限
        
        返回
        - LightcurveData: 裁剪后的光变曲线
        """
        # 确保 tmin/tmax 不为 None，便于后续比较
        tmin_val: float = tmin if tmin is not None else (self.bin_lo[0] if len(self.bin_lo) > 0 else 0.0)
        tmax_val: float = tmax if tmax is not None else (self.bin_hi[-1] if len(self.bin_hi) > 0 else float(np.inf))
        
        # 现在 tmin_val 和 tmax_val 都是 float 了
        mask = (self.bin_lo >= tmin_val) & (self.bin_hi <= tmax_val)
        lc_truncated = self.apply_mask(mask, inplace=False)
        
        # 更新 GTI：仅保留 [tmin_val, tmax_val] 内的 GTI
        if lc_truncated.gti is not None:
            gti_filtered = [
                (max(float(s), tmin_val), min(float(e), tmax_val)) 
                for s, e in lc_truncated.gti 
                if min(float(e), tmax_val) > max(float(s), tmin_val)
            ]
            if gti_filtered:
                lc_truncated.gti_start = np.array([s for s, e in gti_filtered])
                lc_truncated.gti_stop = np.array([e for s, e in gti_filtered])
            else:
                lc_truncated.gti_start = None
                lc_truncated.gti_stop = None
        
        return lc_truncated
    
    def sort(self, inplace: bool = False) -> 'LightcurveData':
        """按 bin_lo 排序光变曲线。
        
        参数
        - inplace: 若 True 则原地修改；否则返回新对象
        
        返回
        - LightcurveData: 排序后的光变曲线
        """
        sort_idx = np.argsort(self.bin_lo)
        mask = np.zeros(len(self.bin_lo), dtype=bool)
        mask[sort_idx] = True
        return self.apply_mask(mask, inplace=inplace)
    
    def _copy(self) -> 'LightcurveData':
        """返回浅复制（数组仍共享内存）"""
        import copy
        return copy.copy(self)
    
    def slice(self, tmin: Optional[float] = None, tmax: Optional[float] = None) -> 'LightcurveData':
        """按时间范围筛选光变曲线，返回新实例。委托到 ops.slice_lightcurve。"""
        from .ops import slice_lightcurve
        return slice_lightcurve(self, tmin=tmin, tmax=tmax)
    

    def rebin(self, binsize: float, method: Literal['sum', 'mean'] = 'sum', 
              *, align_ref: Optional[float] = None, empty_bin: Literal['zero', 'nan'] = 'zero') -> 'LightcurveData':
        """时间重采样(GTI 感知，参考 Stingray 实现）。
        
        参数
        - binsize: 新的时间分辨率（秒）
        - method: 聚合方法 ('sum' 或 'mean')
        - align_ref: 可选的对齐参考时间点
        - empty_bin: 空 bin 处理方式 ('zero' 或 'nan')
        
        返回
        - LightcurveData: 新采样的光变曲线
        
        说明
        - 若定义了 GTI，则对每个 GTI 段分别重采样
        - 自动更新 counts_err/rate_err 的误差传播
        """
        from .ops import rebin_lightcurve
        return rebin_lightcurve(self, binsize=binsize, method=method, 
                               align_ref=align_ref, empty_bin=empty_bin)
    
    def __sub__(self, other: 'LightcurveData', *, 
                ratio: Optional[float] = None,
                use_exposure_weighted_ratio: bool = True) -> 'LightcurveData':
        """光变曲线减法：self - ratio * other
        
        参数
        ----
        other : LightcurveData
            要减去的光变曲线（通常是背景）
        ratio : float, optional
            缩放比例。若为 None，自动计算：
            - use_exposure_weighted_ratio=True: (源面积×源曝光)/(背景面积×背景曝光)
            - 否则: 源面积/背景面积
        use_exposure_weighted_ratio : bool, default=True
            是否使用曝光加权比值
        
        返回
        ----
        LightcurveData
            净光变曲线
        
        示例
        ----
        >>> net = source - background  # 自动计算 ratio
        >>> net = source.__sub__(background, ratio=1.5)
        
        说明
        ----
        该方法是 netdata 的语法糖，实际调用 netdata(self, other, ...)。
        核心逻辑在 netdata 中实现，以便复用于 PHA 等其他数据类型。
        """
        from jinwu.core.datasets import netdata
        # 直接委托给 netdata，它会自动计算 ratio（当 ratio=None）
        return netdata(self, other, ratio=ratio, use_exposure_weighted_ratio=use_exposure_weighted_ratio)
    
    def __add__(self, other: 'LightcurveData') -> 'LightcurveDataset':
        """两个光变曲线相加，返回 LightcurveDataset 容器
        
        示例
        ----
        >>> ds = lc1 + lc2  # 返回 LightcurveDataset([lc1, lc2])
        >>> ds = lc1 + lc2 + lc3  # 链式添加
        """
        from jinwu.core.datasets import LightcurveDataset
        return LightcurveDataset(data=[self, other])
    
    def plot(self, **kwargs):
        """绘制光变曲线，支持 GTI 阴影（如可用）。
        
        kwargs 透传给 plot_lightcurve，支持：
        - ax: matplotlib.axes.Axes 对象
        - show_gti: 是否绘制 GTI 阴影（默认 True）
        - ...其他参数（见 plot.plot_lightcurve 文档）
        """
        from .plot import plot_lightcurve
        return plot_lightcurve(self, **kwargs)


@dataclass(slots=True)
class EventData(OgipTimeSeriesBase):
    """事件列表数据（支持 TIMEZERO 标准）

    **时间处理设计（与 LightcurveData 一致）**
    
    时间处理遵循 OGIP-93-003 规范：
    - time: 相对时间（相对于第一个事件，单位秒）
    - timezero: 原始 TIMEZERO + 第一个事件时间（MET 秒）
    - timezero_obj: Time 对象（根据 TELESCOP 生成，用于时间转换）
    - 绝对时间 = time + timezero
    
    其他时间数据（如 GTI）也转换为相对时间存储。

    字段
    - time: 事件到达时间（相对时间，秒）
    - timezero: 时间偏移量（MET 秒）
    - timezero_obj: Time 对象（用于时间转换）
    - pi/channel: 事件能道（可选）
    - gti_start/gti_stop: GTI（相对时间，可选）
    - header: 头关键字字典

    English
    - Event list with TIMEZERO support (relative time storage).
    - time: relative time (seconds from first event)
    - timezero_obj: Time object for absolute time conversion
    """
    kind: ClassVar[Literal['evt']] = 'evt'
    
    # ========== 核心时间字段 ==========
    time: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=float))   # 相对时间 (s)
    timezero: float = 0.0                   # TIMEZERO + 第一个事件时间（MET 秒）
    timezero_obj: Optional[Any] = None      # Time 对象（根据 TELESCOP 生成）
    telescop: Optional[str] = None          # 望远镜名称（用于时间转换）
    
    # ========== 事件属性 ==========
    pi: Optional[np.ndarray] = None       # PI or CHANNEL if present
    channel: Optional[np.ndarray] = None
    x: Optional[np.ndarray] = None
    y: Optional[np.ndarray] = None
    
    # ========== GTI ==========
    gti_start: Optional[np.ndarray] = None  # GTI 起始（相对时间，秒）
    gti_stop: Optional[np.ndarray] = None   # GTI 结束（相对时间，秒）
    gti_start_obj: Optional[Any] = None     # GTI 起始 Time 对象数组
    gti_stop_obj: Optional[Any] = None      # GTI 结束 Time 对象数组
    gti: Optional[list] = None              # GTI 元组列表 [(start, stop), ...] 相对时间
    
    # ========== 其他字段 ==========
    # 原始列字典（列名 -> ndarray），便于上层直接访问原始表格数据
    raw_columns: Optional[Dict[str, np.ndarray]] = None
    # 别名映射：alias -> 原始列名
    colmap: Optional[Dict[str, Optional[str]]] = None
    # 若事件表中包含能量列（ENERGY 等），复制到此字段以方便使用
    energy: Optional[np.ndarray] = None
    # 若文件/同目录存在 EBOUNDS 表，记录 (CHANNEL, E_MIN, E_MAX)
    ebounds: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
    # 列名元组（继承自基类时已有 path/header/meta/headers_dump，无需重复声明）
    columns: Tuple[str, ...] = ()

    # 内部 XSelectSession，用于跨多次 filter 调用累积筛选条件；
    # 默认不创建，只有在首次调用 xselect()/filter_* 时才懒加载。
    _xselect_session: Optional['XSelectSession'] = field(default=None, init=False, repr=False, compare=False)

    if TYPE_CHECKING:  # only for type checkers to avoid circular import at runtime
        from .xselect import XSelectSession  # noqa: F401

    def validate(self) -> ValidationReport:  # type: ignore[override]
        rpt = super().validate()
        colset = {c.upper() for c in self.columns}
        if "TIME" not in colset:
            rpt.add('ERROR', 'MISSING_COLUMN', "EVENTS missing TIME column")
        rpt.ok = len(rpt.errors()) == 0
        return rpt

    @property
    def n(self) -> int:
        """事件数量"""
        return len(self.time) if self.time is not None else 0
    
    @property
    def duration(self) -> Optional[float]:
        if self.time.size:
            return float(self.time.max() - self.time.min())
        return None

    @property
    def absolute_time(self) -> np.ndarray:
        """绝对时间 = time + timezero（OGIP 标准）"""
        return self.time + self.timezero
    
    @property
    def gti_exposure(self) -> Optional[float]:
        if self.gti_start is None or self.gti_stop is None:
            return None
        return float(np.sum(self.gti_stop - self.gti_start))
    
    def get_time_object(self, index: Optional[int] = None) -> Optional[Any]:
        """获取 Time 对象（用于时间转换）
        
        参数：
        - index: 若指定，返回该索引的时间对象；否则返回所有时间的数组
        
        返回：
        - astropy.time.Time 对象或 None
        """
        if self.timezero_obj is None:
            return None
        
        absolute_times = self.absolute_time
        
        if index is not None:
            dt = TimeDelta(absolute_times[index], format='sec')
        else:
            dt = TimeDelta(absolute_times, format='sec')
        
        return self.timezero_obj + dt

    def get_energy(self, rmf: Optional[RmfData] = None) -> Optional[np.ndarray]:
        """返回事件能量 (keV)。

        - 若事件表中已包含 `energy` 字段则直接返回；
        - 若仅有 PI/CHANNEL 且提供 RMF/EBOUNDS，则可在外部实现转换；
        - 当前实现为占位：若 `energy` 缺失且 no rmf 则返回 None。
        """
        if getattr(self, 'energy', None) is not None:
            return self.energy
        # Try using provided RMF/RSP (preferred) — attempt RMF-based posterior mapping first
        if rmf is not None:
            try:
                # Obtain dense matrix and energy bin centers from RmfData-like object
                if hasattr(rmf, 'dense_matrix'):
                    dm = np.asarray(rmf.dense_matrix)
                else:
                    dm = np.asarray(rmf.rebuild_dense())
                # energy bin centers: prefer energ_lo/energ_hi
                if getattr(rmf, 'energ_lo', None) is not None and getattr(rmf, 'energ_hi', None) is not None:
                    e_centers = 0.5 * (np.asarray(rmf.energ_lo) + np.asarray(rmf.energ_hi))
                elif getattr(rmf, 'e_min', None) is not None and getattr(rmf, 'e_max', None) is not None:
                    e_centers = 0.5 * (np.asarray(rmf.e_min) + np.asarray(rmf.e_max))
                else:
                    e_centers = None

                # normalize orientation: dm may be (N_E, N_C) or (N_C, N_E)
                if dm.ndim == 2:
                    if e_centers is not None and dm.shape[0] == e_centers.size and dm.shape[1] != e_centers.size:
                        # (N_E, N_C) -> transpose to (N_C, N_E)
                        dm_t = dm.T
                    else:
                        # assume rows are channels
                        dm_t = dm
                else:
                    dm_t = np.asarray(dm)

                # choose event channel/PI
                if getattr(self, 'channel', None) is not None:
                    ch_ev = np.asarray(self.channel).astype(int)
                elif getattr(self, 'pi', None) is not None:
                    ch_ev = np.asarray(self.pi).astype(int)
                else:
                    ch_ev = None

                if (ch_ev is not None) and (dm_t is not None) and (e_centers is not None):
                    # Lazy import of mapping implementation
                    from ..ftools.rmf_mapping import map_channels_to_energy

                    mapped = map_channels_to_energy(dm_t, e_centers, ch_ev, method='expected')
                    self.energy = np.asarray(mapped, dtype=float)
                    return self.energy
            except Exception:
                # Fall back to simpler EBOUNDS mapping below on any failure
                pass
        # Next, try EBOUNDS found in the same file (stored on EventData.ebounds by reader)
        if getattr(self, 'ebounds', None) is not None:
            ebounds_tuple = self.ebounds
            if ebounds_tuple is not None and len(ebounds_tuple) == 3:
                ch, e_lo, e_hi = ebounds_tuple
                emid = 0.5 * (np.asarray(e_lo) + np.asarray(e_hi))
            if getattr(self, 'channel', None) is not None:
                ch_ev = np.asarray(self.channel)
            elif getattr(self, 'pi', None) is not None:
                ch_ev = np.asarray(self.pi)
            else:
                return None
            cmap = {int(c): float(e) for c, e in zip(np.asarray(ch, int), emid)}
            mapped = np.array([cmap.get(int(cc), np.nan) for cc in ch_ev], dtype=float)
            self.energy = mapped
            return self.energy
        # No conversion available
        return None

    def plot(self, ax: Optional[Any] = None, *, bin_size: Optional[float] = None, max_bins: int = 200, yscale: str = 'linear', title: Optional[str] = None, **kwargs):
        """简单事件时间分布绘图。

        - bin_size: 指定时间分辨率；若为 None 自动使用 (tmax-tmin)/min(max_bins, N//5)
        - max_bins: 自动模式时的最大 bin 数
        - yscale: y 轴尺度（linear/log）
        - kwargs: 传给 matplotlib.bar
        """
        import matplotlib.pyplot as _plt
        ax = ax or _plt.gca()
        assert ax is not None
        t = np.asarray(self.time, float)
        if t.size == 0:
            ax.text(0.5, 0.5, 'No events', transform=ax.transAxes, ha='center')
            return ax
        tmin, tmax = float(t.min()), float(t.max())
        if bin_size is None:
            span = tmax - tmin
            est_bins = min(max_bins, max(1, int(t.size // 5)))
            bin_size = span / est_bins if span > 0 else 1.0
        nbins = max(1, int(np.ceil((tmax - tmin) / bin_size)))
        edges = tmin + np.arange(nbins + 1) * bin_size
        hist, _ = np.histogram(t, bins=edges)
        centers = 0.5 * (edges[:-1] + edges[1:])
        kwargs.setdefault('alpha', 0.7)
        ax.bar(centers, hist, width=bin_size * 0.9, align='center', **kwargs)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Counts per bin')
        ax.set_yscale(yscale)
        if title is None:
            title = Path(str(getattr(self, 'path', ''))).name or 'EVENTS'
        ax.set_title(title)
        ax.grid(alpha=0.3, ls='--')
        # 可选叠加 GTI 区域阴影
        if self.gti_start is not None and self.gti_stop is not None:
            for s, e in zip(self.gti_start, self.gti_stop):
                ax.axvspan(float(s), float(e), color='orange', alpha=0.1)
        return ax

    def slice(self, tmin: Optional[float] = None, tmax: Optional[float] = None, *, pi_min: Optional[int] = None, pi_max: Optional[int] = None, ch_min: Optional[int] = None, ch_max: Optional[int] = None) -> 'EventData':
        """按时间和/或能量筛选事件，返回新实例。委托到 ops.slice_events。"""
        from .ops import slice_events
        return slice_events(self, tmin=tmin, tmax=tmax, pi_min=pi_min, pi_max=pi_max, ch_min=ch_min, ch_max=ch_max)

    def rebin(self, binsize: float, *, tmin: Optional[float] = None, tmax: Optional[float] = None) -> 'LightcurveData':
        """从事件生成分 bin 光变曲线。委托到 ops.rebin_events_to_lightcurve。"""
        from .ops import rebin_events_to_lightcurve
        return rebin_events_to_lightcurve(self, binsize=binsize, tmin=tmin, tmax=tmax)

    # Convenience wrappers delegating to xselect helpers
    def xselect(self) -> 'XSelectSession':
        """创建一个（或复用已有的）XSelectSession 来管理过滤操作。

        返回的 session 对象可以用来应用多个过滤条件，这些条件会被记录并统一应用。

        示例：
            session = ev.xselect()
            session.apply_time(tmin=100, tmax=200)
            session.apply_energy(pi_min=50, pi_max=500)
            filtered_ev = session.current
        """
        from .xselect import XSelectSession
        if self._xselect_session is None:
            self._xselect_session = XSelectSession(self)
        return self._xselect_session
    
    def filter_time(self, tmin: Optional[float] = None, tmax: Optional[float] = None) -> 'XSelectSession':
        """应用时间过滤并返回 XSelectSession 以便继续过滤。

        tmin/tmax 可以是:
        - float（以秒为单位，直接与 TIME 列比较）；
        - astropy.time.Time / TimeDelta 或 jinwu.core.time.Time，
            会自动转换为对应任务 MET 的秒数。

        即使忽略返回值，后续在本 EventData 上调用 extract_*/save()
        也会自动基于累积后的过滤结果进行操作。
        """
        session = self.xselect()
        session.apply_time(tmin=tmin, tmax=tmax)
        return session

    def filter_region(self, region) -> 'XSelectSession':
        """应用区域过滤并返回 XSelectSession 以便继续过滤。"""
        session = self.xselect()
        session.apply_region(region)
        return session
    
    def filter_energy(self, pi_min: Optional[int] = None, pi_max: Optional[int] = None) -> 'XSelectSession':
        """应用能量过滤并返回 XSelectSession 以便继续过滤。"""
        session = self.xselect()
        session.apply_energy(pi_min=pi_min, pi_max=pi_max)
        return session
    
    
    def _current_for_products(self) -> 'EventData':
        """返回用于提取/保存派生产品的 EventData。

        若已通过 filter_*/xselect() 创建了会话，则返回会话的 current，
        否则返回自身。这样可以支持：

            evt.filter_time(...)
            lc = evt.extract_curve(...)
            evt.save('xx.fits', kind='evt')

        这类写法自动使用所有累积的过滤条件。
        """
        sess = getattr(self, '_xselect_session', None)
        if sess is not None:
            cur = getattr(sess, 'current', None)
            if cur is not None:
                return cur
        return self

    def extract_spectrum(self, **kwargs) -> PhaData:
        """提取能谱；若已存在会话，则使用累积过滤后的事件。"""
        from . import xselect
        src = self._current_for_products()
        if src is not self:
            # 避免在同一对象上再次检查 session，直接在过滤结果上调用
            return src.extract_spectrum(**kwargs)
        return xselect.extract_spectrum(self, **kwargs)

    def extract_curve(self, binsize: float, **kwargs) -> 'LightcurveData':
        """提取光变曲线；若已存在会话，则使用累积过滤后的事件。"""
        from . import xselect
        src = self._current_for_products()
        if src is not self:
            return src.extract_curve(binsize=binsize, **kwargs)
        return xselect.extract_curve(self, binsize=binsize, **kwargs)

    def extract_image(self, **kwargs):
        """提取图像；若已存在会话，则使用累积过滤后的事件。"""
        from . import xselect
        src = self._current_for_products()
        if src is not self:
            return src.extract_image(**kwargs)
        return xselect.extract_image(self, **kwargs)

    def save(self, outpath: str | Path, kind: str = 'evt', overwrite: bool = False, **kwargs) -> Path:
        """Save EventData or derived products.

        kind: 'evt' (save events as a table), 'lc' (extract+save lightcurve), 'pha' (extract+save pha), 'img' (extract+save image)
        The kwargs are forwarded to the corresponding xselect writer/extractor.
        """
        # 若存在会话，则优先在过滤后的 current 上执行保存逻辑，
        # 这样 evt.filter_xxx(...); evt.save(...) 会使用所有筛选条件。
        src = self._current_for_products()
        if src is not self:
            return src.save(outpath, kind=kind, overwrite=overwrite, **kwargs)

        outp = Path(outpath)
        if kind == 'evt':
            # Write events to FITS table while preserving original columns and HDU headers when possible.
            from astropy.table import Table
            # Prefer writing original raw_columns if available to preserve all original columns
            if getattr(self, 'raw_columns', None):
                try:
                    t = Table(self.raw_columns)
                except Exception:
                    # fallback to building minimal table
                    cols = {'TIME': self.time}
                    if getattr(self, 'x', None) is not None:
                        cols['X'] = getattr(self, 'x')
                    if getattr(self, 'y', None) is not None:
                        cols['Y'] = getattr(self, 'y')
                    if getattr(self, 'energy', None) is not None:
                        cols['ENERGY'] = getattr(self, 'energy')
                    t = Table(cols)
            else:
                cols = {'TIME': self.time}
                if getattr(self, 'x', None) is not None:
                    cols['X'] = getattr(self, 'x')
                if getattr(self, 'y', None) is not None:
                    cols['Y'] = getattr(self, 'y')
                if getattr(self, 'energy', None) is not None:
                    cols['ENERGY'] = getattr(self, 'energy')
                t = Table(cols)

            # Write table to file first, then update headers to preserve original header blocks
            t.write(outp, format='fits', overwrite=overwrite)
            try:
                # Re-open and patch primary + event HDU headers from stored dumps if present
                with fits.open(outp, mode='update') as hdul:
                    # primary header
                    if getattr(self, 'headers_dump', None) is not None and isinstance(self.headers_dump, FitsHeaderDump):
                        if getattr(self.headers_dump, 'primary', None) is not None:
                            for k, v in (self.headers_dump.primary or {}).items():
                                try:
                                    prih = cast(Any, hdul[0])
                                    prih.header[k] = v
                                except Exception:
                                    continue
                    # event/table header: use self.header if present
                    if getattr(self, 'header', None) is not None and len(hdul) > 1:
                        tbl = cast(Any, hdul[1])
                        for k, v in (self.header or {}).items():
                            # avoid overwriting structural keywords created by astropy that are required
                            if k in ('TFIELDS', 'TTYPE1', 'TFORM1', 'XTENSION', 'BITPIX', 'NAXIS'):
                                continue
                            try:
                                tbl.header[k] = v
                            except Exception:
                                continue
                    hdul.flush()
            except Exception:
                # best-effort: if header patching fails, still return the written path
                pass
            return outp
        elif kind == 'lc':
            lc = self.extract_curve(**kwargs)
            from . import xselect
            return xselect.write_curve(lc, outp, overwrite=overwrite)
        elif kind == 'pha':
            pha = self.extract_spectrum(**kwargs)
            from . import xselect
            return xselect.write_pha(pha, outp, overwrite=overwrite)
        elif kind == 'img':
            img, xedges, yedges = self.extract_image(**kwargs)
            from . import xselect
            return xselect.write_image(img, xedges, yedges, outp, overwrite=overwrite)
        else:
            raise ValueError('unknown kind: ' + str(kind))

        

    def clear_region(self, *, use_original: bool = True) -> 'EventData':
        """清除 REGION 过滤。

        优先使用内部 XSelectSession（若已存在），以便：

            evt.filter_time(...)
            evt.filter_region(...)
            evt.clear_region()
            lc = evt.extract_curve(...)

        这类写法能在同一个 EventData 上生效。

        若当前没有会话，则退回到旧逻辑：从原始文件重新读取并根据
        当前对象的 time/energy 推断窗口，返回一个新的 EventData。
        """
        sess = getattr(self, '_xselect_session', None)
        if sess is not None:
            sess.clear_region()
            # 会话存在时，后续 extract/save 会自动使用更新后的 current
            return self._current_for_products()

        # 兼容：无会话时使用旧实现
        src = None
        if use_original and getattr(self, '_original_events', None) is not None:
            src = getattr(self, '_original_events')
        elif getattr(self, 'path', None) is not None:
            src = read_evt(self.path)
        else:
            raise ValueError('No original events available to clear region')

        tmin = None
        tmax = None
        time_arr = getattr(self, 'time', None)
        if time_arr is not None and isinstance(time_arr, np.ndarray) and time_arr.size:
            t = np.asarray(time_arr, dtype=float)
            tmin = float(t.min())
            tmax = float(t.max())

        pi_min = None
        pi_max = None
        if getattr(self, 'pi', None) is not None:
            arr = np.asarray(self.pi, dtype=int)
            if arr.size:
                pi_min = int(arr.min())
                pi_max = int(arr.max())
        elif getattr(self, 'channel', None) is not None:
            arr = np.asarray(self.channel, dtype=int)
            if arr.size:
                pi_min = int(arr.min())
                pi_max = int(arr.max())

        from . import xselect as _xsel
        return _xsel.select_events(src, tmin=tmin, tmax=tmax, pi_min=pi_min, pi_max=pi_max)

    def clear_time(self, *, use_original: bool = True) -> 'EventData':
        """清除 TIME 过滤。

        若已存在内部会话，则直接调用 session.clear_time() 并返回当前用于
        派生产品的 EventData；否则退回到旧逻辑：从原始文件重建，仅保留
        能量窗口。
        """
        sess = getattr(self, '_xselect_session', None)
        if sess is not None:
            sess.clear_time()
            return self._current_for_products()

        src = None
        if use_original and getattr(self, '_original_events', None) is not None:
            src = getattr(self, '_original_events')
        elif getattr(self, 'path', None) is not None:
            src = read_evt(self.path)
        else:
            raise ValueError('No original events available to clear time')

        pi_min = None
        pi_max = None
        if getattr(self, 'pi', None) is not None:
            arr = np.asarray(self.pi, dtype=int)
            if arr.size:
                pi_min = int(arr.min())
                pi_max = int(arr.max())
        elif getattr(self, 'channel', None) is not None:
            arr = np.asarray(self.channel, dtype=int)
            if arr.size:
                pi_min = int(arr.min())
                pi_max = int(arr.max())

        from . import xselect as _xsel
        return _xsel.select_events(src, region=None, pi_min=pi_min, pi_max=pi_max)

    def clear_energy(self, *, use_original: bool = True) -> 'EventData':
        """清除 ENERGY/PI 过滤。

        若已存在内部会话，则调用 session.clear_energy() 并返回当前 EventData；
        否则退回到旧实现，仅保留时间窗口。
        """
        sess = getattr(self, '_xselect_session', None)
        if sess is not None:
            sess.clear_energy()
            return self._current_for_products()

        src = None
        if use_original and getattr(self, '_original_events', None) is not None:
            src = getattr(self, '_original_events')
        elif getattr(self, 'path', None) is not None:
            src = read_evt(self.path)
        else:
            raise ValueError('No original events available to clear energy')

        tmin = None
        tmax = None
        time_arr = getattr(self, 'time', None)
        if time_arr is not None and isinstance(time_arr, np.ndarray) and time_arr.size:
            t = np.asarray(time_arr, dtype=float)
            tmin = float(t.min())
            tmax = float(t.max())

        from . import xselect as _xsel
        return _xsel.select_events(src, tmin=tmin, tmax=tmax, region=None)

    def clear_all(self, *, use_original: bool = True) -> 'EventData':
        """清除所有过滤，返回未筛选的完整 EventData。

        若已存在内部会话，则调用 session.clear_all()，使 current 回到 original，
        并返回当前用于派生产品的 EventData；否则退回到旧逻辑直接重新读取。
        """
        sess = getattr(self, '_xselect_session', None)
        if sess is not None:
            sess.clear_all()
            return self._current_for_products()

        if use_original and getattr(self, '_original_events', None) is not None:
            return getattr(self, '_original_events')
        elif getattr(self, 'path', None) is not None:
            return read_evt(self.path)
        else:
            raise ValueError('No original events available to clear all')


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
    # 时间参考位置（TREFPOS）与观测起始日期（DATE-OBS）——FITS 4.0 推荐项
    trefpos_val = _first_non_empty(["TREFPOS", "TREFDIR"], *(hdr for hdr in [prefer_header, primary] + other_ext_headers if hdr))
    try:
        trefpos = str(trefpos_val) if trefpos_val is not None else None
    except Exception:
        trefpos = None
    dateobs_val = _first_non_empty(["DATE-OBS", "DATE_OBS"], *(hdr for hdr in [prefer_header, primary] + other_ext_headers if hdr))
    try:
        dateobs = str(dateobs_val) if dateobs_val is not None else None
    except Exception:
        dateobs = None
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
        trefpos=trefpos,
        dateobs=dateobs,
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

            # If EBOUNDS extension present in file, extract and attach to event
            ebounds = None
            if 'EBOUNDS' in h:
                try:
                    de = cast(Any, h['EBOUNDS']).data
                    ebounds = (
                        np.asarray(de['CHANNEL'], int),
                        np.asarray(de['E_MIN'], float),
                        np.asarray(de['E_MAX'], float),
                    )
                except Exception:
                    ebounds = None
        self._data = ArfData(
            path=self.path,
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
        # RESPFILE / ANCRFILE may be present in the SPECTRUM header; normalize to str or None
        respfile = None
        ancrfile = None
        try:
            if 'RESPFILE' in header and header.get('RESPFILE') not in (None, '', ' '):
                respfile = str(header.get('RESPFILE'))
        except Exception:
            respfile = None
        try:
            if 'ANCRFILE' in header and header.get('ANCRFILE') not in (None, '', ' '):
                ancrfile = str(header.get('ANCRFILE'))
        except Exception:
            ancrfile = None

        self._data = PhaData(
            path=self.path,
            channels=channels, counts=counts, stat_err=stat_err,
            exposure=exposure, backscal=backscal, areascal=areascal,
            respfile=respfile,
            ancrfile=ancrfile,
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
        """读取光变曲线并返回 `LightcurveData`（完整支持 TIMEZERO）。
        
        TIMEZERO 处理规则（OGIP-93-003 标准）：
        - TIMEZERO=0, TIME=MET: 绝对时间模式（直接使用 TIME）
        - TIMEZERO=MET_START, TIME=0,1,2...: 相对时间模式（绝对时间 = TIME + TIMEZERO）
        
        自动根据 TELESCOP 生成 Time 对象用于时间转换。
        
        English: Read light curve with full TIMEZERO support per OGIP-93-003.
        """
        with fits.open(self.path) as h:
            # ===== 第 1 步：找到包含 TIME 列的 HDU =====
            hdu = None
            for ext in h:
                ext_any = cast(Any, ext)
                if getattr(ext_any, "data", None) is None:
                    continue
                names = getattr(ext_any.data, "columns", None)
                if names is not None and ("TIME" in [n.upper() for n in names.names]):
                    hdu = ext_any
                    break
            
            if hdu is None:
                raise ValueError("No suitable lightcurve HDU with TIME column found")
            
            d = cast(Any, hdu.data)
            col_names_upper = [n.upper() for n in d.columns.names]
            header = dict(cast(Any, hdu.header))
            
            # ===== 第 2 步：读取 TIME 列（原始值，可能是相对或绝对）=====
            time_raw = np.asarray(d["TIME"], dtype=float)
            
            # ===== 第 3 步：读取 TIMEZERO（默认 0）=====
            timezero_raw = 0.0
            if "TIMEZERO" in header:
                try:
                    timezero_raw = float(header["TIMEZERO"])
                except (ValueError, TypeError):
                    timezero_raw = 0.0
            
            # ===== 第 3.5 步：转换时间为相对时间 =====
            # time 改为相对于第一个值的相对时间
            # timezero 改为 timezero_raw + time_raw[0]（包含第一个原始值）
            time_offset = float(time_raw[0]) if len(time_raw) > 0 else 0.0
            time = time_raw - time_offset
            timezero = timezero_raw + time_offset
            
            # ===== 第 4 步：根据 TELESCOP 生成 Time 对象 =====
            telescop = None
            timezero_obj = None
            
            # 尝试从多个位置读取 TELESCOP
            for hdr in [header, dict(h[0].header) if len(h) > 0 else {}]:
                if "TELESCOP" in hdr:
                    telescop = str(hdr["TELESCOP"]).strip().upper()
                    break
            
            # 根据 TELESCOP 创建 Time 对象
            # 使用更新后的 timezero（= timezero_raw + time_raw[0]）
            if telescop is not None and timezero != 0.0:
                try:
                    from .time import Time  # 使用 jinwu.core.time
                    
                    # 根据不同任务选择对应的 MET 格式
                    # 使用 astropy 已注册的 MET 名称（去掉 _met 后缀）
                    format_map = {
                        'FERMI': 'fermi',
                        'EP': 'ep',
                        'LEIA': 'leia',
                        'GECAM': 'gecam',
                        'HXMT': 'hxmt',
                        'SWIFT': 'swift',
                        'GRID': 'grid',
                        'MAXI': 'maxi',
                        'SUZAKU': 'suzaku',
                        'XMM': 'newton',
                        'NEWTON': 'newton',
                        'XRISM': 'xrism',
                    }
                    
                    met_format = None
                    for key, fmt in format_map.items():
                        if key in telescop:
                            met_format = fmt
                            break
                    
                    # 创建 TIMEZERO 对应的 Time 对象
                    if met_format is not None:
                        timezero_obj = Time(timezero, format=met_format)
                    else:
                        # 未知任务，尝试使用 UTC
                        try:
                            timezero_obj = Time(timezero, format='unix', scale='utc')
                        except Exception:
                            pass
                except ImportError:
                    # 若 jinwu.core.time 不可用，忽略
                    pass
            
            # ===== 第 5 步：推断 dt (bin 宽度) =====
            dt = None
            if "TIMEDEL" in header:
                try:
                    dt = float(header["TIMEDEL"])
                except (ValueError, TypeError):
                    pass
            if dt is None and time.size >= 2:
                dt = float(np.median(np.diff(time)))
            if dt is None:
                dt = 1.0  # 默认回退
            
            # ===== 第 6 步：计算 bin_lo/bin_hi（便利字段）=====
            # 根据 HEASoft 约定：TIME 是 bin 左缘
            bin_lo = time.copy()
            bin_hi = time + dt
            
            # ===== 第 7 步：读取 RATE/COUNTS =====
            rate = None
            counts = None
            is_rate = False
            value = None  # 主要数据列
            
            if "RATE" in col_names_upper:
                rate = np.asarray(d["RATE"], dtype=float)
                value = rate
                is_rate = True
                # 若无 COUNTS，从 RATE 推算
                if "COUNTS" not in col_names_upper and dt > 0:
                    counts = rate * dt
            
            if "COUNTS" in col_names_upper:
                counts = np.asarray(d["COUNTS"], dtype=float)
                # 若 value 未设置（无 RATE），使用 COUNTS
                if value is None:
                    value = counts
                    is_rate = False
                # 若无 RATE，从 COUNTS 推算
                if rate is None and dt > 0:
                    rate = counts / dt
            
            if value is None:
                raise ValueError("Lightcurve HDU lacks RATE/COUNTS column")
            
            # ===== 第 8 步：读取误差列 =====
            error = None
            rate_err = None
            counts_err = None
            
            if "ERROR" in col_names_upper:
                error = np.asarray(d["ERROR"], dtype=float)
                # 根据 is_rate 分配误差
                if is_rate:
                    rate_err = error
                    if counts is not None and dt > 0:
                        counts_err = error * dt
                else:
                    counts_err = error
                    if rate is not None and dt > 0:
                        rate_err = error / dt
            
            # ===== 第 9 步：读取可选列 =====
            fracexp = None
            quality = None
            backscal_col = None
            areascal_col = None
            
            if "FRACEXP" in col_names_upper:
                fracexp = np.asarray(d["FRACEXP"], dtype=float)
            
            if "QUALITY" in col_names_upper:
                quality = np.asarray(d["QUALITY"], dtype=int)
            
            if "BACKSCAL" in col_names_upper:
                backscal_col = d["BACKSCAL"]
            elif "BACK_SCAL" in col_names_upper:
                backscal_col = d["BACK_SCAL"]
            
            if "AREASCAL" in col_names_upper:
                areascal_col = d["AREASCAL"]
            elif "AREA_SCAL" in col_names_upper:
                areascal_col = d["AREA_SCAL"]
            
            # ===== 第 10 步：提取 GTI（注意：GTI 时间应该是绝对时间）=====
            gti_start = None
            gti_stop = None
            try:
                gti_list = self._extract_gti(h)
                if gti_list is not None:
                    gti_start = np.array([s for s, e in gti_list], dtype=float)
                    gti_stop = np.array([e for s, e in gti_list], dtype=float)
            except Exception:
                pass
            
            # ===== 第 11 步：计算时间范围 =====
            # time 现在是相对时间，timezero 是第一个点的绝对时间
            # 绝对时间 = time + timezero（都是从零开始的相对时间）
            tstart = timezero  # 第一个点的绝对时间
            if len(time) > 0:
                tstop = timezero + time[-1] + dt
                tseg = float(tstop - tstart)
            else:
                tstop = tstart
                tseg = None
            
            # ===== 第 12 步：读取元数据 =====
            headers_dump = _collect_headers_dump(h)
            meta = _build_meta(h, header)
            
            # ===== 第 13 步：计算曝光时间 =====
            exposure = None
            if "EXPOSURE" in header:
                try:
                    exposure = float(header["EXPOSURE"])
                except (ValueError, TypeError):
                    pass
            if exposure is None and "EXPTIME" in header:
                try:
                    exposure = float(header["EXPTIME"])
                except (ValueError, TypeError):
                    pass
            
            bin_exposure = None
            if fracexp is not None and exposure is not None:
                bin_exposure = fracexp * exposure
            elif exposure is not None and len(time) > 0:
                bin_exposure = np.full(len(time), exposure / len(time))
            
            # ===== 第 14 步：推断误差分布 =====
            err_dist = None
            if counts is not None:
                err_dist = "poisson"
            elif rate is not None:
                err_dist = "gauss"
            
            # ===== 第 15 步：读取区域信息 =====
            region = None
            try:
                region = _load_regions(h)
            except Exception:
                pass
            
            # ===== 第 16 步：读取 TIMESYS =====
            timesys = None
            if "TIMESYS" in header:
                timesys = str(header["TIMESYS"])
            elif meta and meta.timesys:
                timesys = meta.timesys
            
            # ===== 第 17 步：读取 MJDREF（可选）=====
            mjdref = None
            if meta and meta.mjdref:
                mjdref = meta.mjdref
            
            # ===== 第 18 步：构造 LightcurveData 对象 =====
            self._data = LightcurveData(
                path=self.path,
                # 核心时间字段（OGIP 标准）
                time=time,
                timezero=timezero,
                timezero_obj=timezero_obj,
                dt=dt,
                # 便利时间字段
                bin_lo=bin_lo,
                bin_hi=bin_hi,
                tstart=tstart,
                tseg=tseg,
                # 核心数据字段
                value=value,
                error=error,
                is_rate=is_rate,
                # 分离存储（可选）
                counts=counts,
                rate=rate,
                counts_err=counts_err,
                rate_err=rate_err,
                err_dist=err_dist,
                # GTI
                gti_start=gti_start,
                gti_stop=gti_stop,
                # 质量与曝光
                quality=quality,
                fracexp=fracexp,
                exposure=exposure,
                bin_exposure=bin_exposure,
                backscal=backscal_col,
                areascal=areascal_col,
                # 时间系统元数据
                telescop=telescop,
                timesys=timesys,
                mjdref=mjdref,
                # 其他
                region=region,
                header=header,
                meta=meta,
                headers_dump=headers_dump,
                columns=tuple(d.columns.names),
            )
        
        return self._data
    
    def _extract_gti(self, hdul: fits.HDUList) -> Optional[list[tuple[float, float]]]:
        """从 FITS HDU 列表中提取 GTI，返回 [(start0, stop0), ...] 列表。
        
        此方法复用 OgipTimeSeriesBase.extract_gti 的逻辑。
        """
        if hdul is None:
            return None
        
        # 查找 GTI 扩展
        for hdu in hdul:
            hdr = getattr(hdu, 'header', {})
            name = (hdr.get('EXTNAME') or '').upper() if hdr else ''
            if name == 'GTI' or getattr(hdu, 'name', '').upper() == 'GTI':
                data = getattr(hdu, 'data', None)
                if data is None:
                    continue
                
                cols = getattr(data, 'columns', None)
                colnames = [n.upper() for n in (cols.names if cols is not None else [])]
                
                # 寻找 START/STOP 列
                start_col = None
                stop_col = None
                for n in ['START', 'TSTART']:
                    if n in colnames:
                        start_col = n
                        break
                for n in ['STOP', 'TSTOP']:
                    if n in colnames:
                        stop_col = n
                        break
                
                if start_col and stop_col:
                    arr_start = data[start_col]
                    arr_stop = data[stop_col]
                    try:
                        return [(float(s), float(e)) for s, e in zip(arr_start, arr_stop)]
                    except (TypeError, ValueError):
                        return None
        
        return None

    def validate(self) -> ValidationReport:
        if self._data is None:
            self.read()
        assert self._data is not None
        return self._data.validate()


def _load_regions(hdul: fits.HDUList) -> Optional[RegionArea]:
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
    Converts TIME to relative time (from first event) and stores
    timezero_obj as a Time object for absolute time conversion.
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
        """读取事件表并返回 `EventData`（完整支持 TIMEZERO）。
        
        TIMEZERO 处理规则（与 LightcurveData 一致）：
        - time: 转换为相对时间（相对于第一个事件，单位秒）
        - timezero: 原始 TIMEZERO + 第一个事件时间（MET 秒）
        - timezero_obj: 根据 TELESCOP 生成的 Time 对象
        - GTI: 也转换为相对时间

        English: Read EVENTS table with TIMEZERO support (relative time storage).
        """
        with fits.open(self.path) as h:
            # ===== 第 1 步：找到包含 TIME 列的 HDU =====
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
            # collect all column names and raw columns as numpy arrays
            colnames = list(getattr(de, 'columns').names) if getattr(de, 'columns', None) is not None else []
            raw_columns: Dict[str, np.ndarray] = {}
            for cn in colnames:
                try:
                    raw_columns[cn] = np.asarray(de[cn])
                except Exception:
                    try:
                        raw_columns[cn] = np.asarray([r[cn] for r in de])
                    except Exception:
                        raw_columns[cn] = np.asarray([])

            # ===== 第 2 步：读取 TIME 列（原始值）=====
            time_raw = np.asarray(raw_columns.get('TIME') if 'TIME' in raw_columns else de['TIME'], float)
            
            # ===== 第 3 步：读取 TIMEZERO（默认 0）=====
            header = dict(cast(Any, hevt.header))
            timezero_raw = 0.0
            if "TIMEZERO" in header:
                try:
                    timezero_raw = float(header["TIMEZERO"])
                except (ValueError, TypeError):
                    timezero_raw = 0.0
            
            # ===== 第 4 步：转换为相对时间 =====
            # time 改为相对于第一个值的相对时间
            # timezero 改为 timezero_raw + time_raw[0]
            time_offset = float(time_raw[0]) if len(time_raw) > 0 else 0.0
            time = time_raw - time_offset
            timezero = timezero_raw + time_offset
            
            # ===== 第 5 步：根据 TELESCOP 生成 Time 对象 =====
            telescop = None
            timezero_obj = None
            
            # 尝试从多个位置读取 TELESCOP
            for hdr in [header, dict(h[0].header) if len(h) > 0 else {}]:
                if "TELESCOP" in hdr:
                    telescop = str(hdr["TELESCOP"]).strip().upper()
                    break
            
            # 根据 TELESCOP 创建 Time 对象
            if telescop is not None and timezero != 0.0:
                try:
                    from .time import Time
                    
                    format_map = {
                        'FERMI': 'fermi',
                        'EP': 'ep',
                        'LEIA': 'leia',
                        'GECAM': 'gecam',
                        'HXMT': 'hxmt',
                        'SWIFT': 'swift',
                        'GRID': 'grid',
                        'MAXI': 'maxi',
                        'SUZAKU': 'suzaku',
                        'XMM': 'newton',
                        'NEWTON': 'newton',
                        'XRISM': 'xrism',
                    }
                    
                    met_format = None
                    for key, fmt in format_map.items():
                        if key in telescop:
                            met_format = fmt
                            break
                    
                    if met_format is not None:
                        timezero_obj = Time(timezero, format=met_format)
                    else:
                        try:
                            timezero_obj = Time(timezero, format='unix', scale='utc')
                        except Exception:
                            pass
                except ImportError:
                    pass
            
            # ===== 第 6 步：读取其他列 =====
            pi = np.asarray(raw_columns['PI'], int) if 'PI' in raw_columns else None
            channel = np.asarray(raw_columns['CHANNEL'], int) if 'CHANNEL' in raw_columns else None
            headers_dump = _collect_headers_dump(h)
            meta = _build_meta(h, header)

            # ===== 第 7 步：读取 EBOUNDS =====
            ebounds = None
            try:
                if 'EBOUNDS' in h:
                    de_eb = cast(Any, h['EBOUNDS']).data
                    ebounds = (
                        np.asarray(de_eb['CHANNEL'], int),
                        np.asarray(de_eb['E_MIN'], float),
                        np.asarray(de_eb['E_MAX'], float),
                    )
            except Exception:
                ebounds = None

            # ===== 第 8 步：读取 GTI 并转换为相对时间 + Time 对象 =====
            gti_start = gti_stop = None
            gti_start_obj = gti_stop_obj = None
            gti_list = None
            try:
                for hdu in h:
                    if getattr(hdu, 'name', '').upper() == 'GTI':
                        gti_data = getattr(hdu, 'data', None)
                        if gti_data is not None and 'START' in gti_data.columns.names and 'STOP' in gti_data.columns.names:
                            # 原始 GTI（MET 时间）
                            gti_start_raw = np.asarray(gti_data['START'], float)
                            gti_stop_raw = np.asarray(gti_data['STOP'], float)
                            # 转换为相对时间（减去 time_offset）
                            gti_start = gti_start_raw - time_offset
                            gti_stop = gti_stop_raw - time_offset
                            gti_list = [(float(s), float(e)) for s, e in zip(gti_start, gti_stop)]
                            
                            # 生成 GTI 的 Time 对象（如果 timezero_obj 可用）
                            if timezero_obj is not None:
                                try:
                                    # GTI 绝对时间 = gti_start + timezero（相对时间 + timezero = MET）
                                    # gti_start_obj = timezero_obj + TimeDelta(gti_start + timezero - timezero) 
                                    #               = timezero_obj + TimeDelta(gti_start)
                                    gti_start_obj = timezero_obj + TimeDelta(gti_start, format='sec')
                                    gti_stop_obj = timezero_obj + TimeDelta(gti_stop, format='sec')
                                except Exception:
                                    gti_start_obj = gti_stop_obj = None
                            break
            except Exception:
                gti_start = gti_stop = None
                gti_start_obj = gti_stop_obj = None
                gti_list = None
            
            # ===== 第 9 步：构建列别名映射 =====
            u2orig = {cn.upper(): cn for cn in colnames}

            def _find(*cands: str) -> Optional[str]:
                for c in cands:
                    if c is None:
                        continue
                    uc = c.upper()
                    if uc in u2orig:
                        return u2orig[uc]
                return None

            colmap: Dict[str, Optional[str]] = {}
            colmap['x'] = _find('X', 'XRAW', 'RAWX', 'DETX', 'DET_X', 'SKX', 'XDET')
            colmap['y'] = _find('Y', 'YRAW', 'RAWY', 'DETY', 'DET_Y', 'SKY', 'YDET')
            colmap['ra'] = _find('RA', 'RA_OBJ', 'RAX', 'RA_DEG')
            colmap['dec'] = _find('DEC', 'DEC_OBJ', 'DECX', 'DEC_DEG')
            colmap['energy'] = _find('ENERGY', 'E', 'ENERG', 'PHOTON_ENERGY')
            colmap['pha'] = _find('PHA', 'PI')

            key_x = colmap.get('x')
            key_y = colmap.get('y')
            key_energy = colmap.get('energy')
            xarr = np.asarray(raw_columns[key_x]) if (key_x is not None and key_x in raw_columns) else None
            yarr = np.asarray(raw_columns[key_y]) if (key_y is not None and key_y in raw_columns) else None
            energy = np.asarray(raw_columns[key_energy]) if (key_energy is not None and key_energy in raw_columns) else None

        # ===== 第 10 步：构建 EventData =====
        self._data = EventData(
            path=self.path,
            time=time,
            timezero=timezero,
            timezero_obj=timezero_obj,
            telescop=telescop,
            pi=pi, channel=channel,
            x=xarr, y=yarr,
            gti_start=gti_start, gti_stop=gti_stop,
            gti_start_obj=gti_start_obj, gti_stop_obj=gti_stop_obj,
            gti=gti_list,
            raw_columns=raw_columns, colmap=colmap, energy=energy,
            ebounds=ebounds,
            header=header, meta=meta, headers_dump=headers_dump,
            columns=tuple(colnames),
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
def readfits(path: str | Path, kind: Literal['arf']) -> ArfData: ...

@overload
def readfits(path: str | Path, kind: Literal['rmf']) -> RmfData: ...

@overload
def readfits(path: str | Path, kind: Literal['pha']) -> PhaData: ...

@overload
def readfits(path: str | Path, kind: Literal['lc']) -> LightcurveData: ...

@overload
def readfits(path: str | Path, kind: Literal['evt']) -> EventData: ...

@overload
def readfits(path: str | Path, kind: None = ...) -> OgipData: ...

def readfits(path: str | Path, kind: Optional[Literal['arf','rmf','pha','lc','evt']] = None) -> OgipData:
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



