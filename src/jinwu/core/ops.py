# -*- coding: utf-8 -*-
"""
jinwu.core.ops

数据操作函数（Operations）：对 OGIP 数据类进行切片、重采样等操作。

本模块提供纯函数式接口，所有操作返回新实例，保持输入不可变
- slice_*: 按时间/能量/道筛选
- rebin_*: 重采样/聚合
- 其他实用转换函数

设计原则：
- 纯函数，无副作用
- 返回新数据类实例
- 支持链式调用
- 可被数据类方法委托调用

English summary
---------------
Operations module for OGIP data manipulation (slicing, rebinning, etc.).
Pure functional interface; all operations return new instances.
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from .file import LightcurveData, PhaData, EventData

__all__ = [
    # Lightcurve operations
    "slice_lightcurve",
    "rebin_lightcurve",
    # PHA operations
    "slice_pha",
    "rebin_pha",
    # Event operations
    "slice_events",
    "rebin_events_to_lightcurve",
]


# ==================== Lightcurve Operations ====================

def slice_lightcurve(
    lc: 'LightcurveData',
    tmin: Optional[float] = None,
    tmax: Optional[float] = None,
) -> 'LightcurveData':
    """按时间范围筛选光变曲线，返回新实例。

    参数
    - lc: 输入光变曲线数据
    - tmin/tmax: 时间下/上界（闭区间）；None 表示不限

    返回
    - 新的 LightcurveData 实例

    English
    Filter lightcurve by time range [tmin, tmax]; returns new instance.
    """
    from .file import LightcurveData  # lazy import to avoid circular dependency
    
    mask = np.ones(lc.time.size, dtype=bool)
    if tmin is not None:
        mask &= (lc.time >= float(tmin))
    if tmax is not None:
        mask &= (lc.time <= float(tmax))
    
    return LightcurveData(
        kind='lc',
        path=lc.path,
        time=lc.time[mask],
        value=lc.value[mask] if lc.value.ndim == 1 else lc.value[mask, :],
        error=(
            lc.error[mask] if (lc.error is not None and lc.error.ndim == 1)
            else (lc.error[mask, :] if lc.error is not None else None)
        ),
        dt=lc.dt,
        exposure=lc.exposure,
        is_rate=lc.is_rate,
        header=lc.header,
        meta=lc.meta,
        headers_dump=lc.headers_dump,
        region=lc.region,
    )


def rebin_lightcurve(
    lc: 'LightcurveData',
    binsize: float,
    method: Literal['sum', 'mean'] = 'sum'
) -> 'LightcurveData':
    """光变曲线时间重采样（rebinning）。
    
    将原始光变曲线按新的时间分辨率重新分组聚合。
    
    参数 (Parameters)
    ----------------
    lc : LightcurveData
        输入光变曲线数据
    binsize : float
        新的时间分辨率（秒），即每个 bin 的宽度
    method : 'sum' | 'mean', default='sum'
        聚合方法：
        - 'sum': 对计数求和（适用于 counts）
        - 'mean': 对速率求平均（适用于 rate）
    
    返回 (Returns)
    -------------
    LightcurveData
        重采样后的新光变曲线实例
    
    原理 (Principle)
    ---------------
    1. 根据 binsize 将时间轴分成等宽区间
    2. 将原始数据点归入对应 bin
    3. 对每个 bin 内的数据进行聚合（求和或平均）
    4. 误差传播：
       - sum: σ_new = √(Σ σ_i²)
       - mean: σ_new = √(Σ σ_i²) / N
    
    示例 (Example)
    -------------
    >>> # 重采样到 10 秒 bin
    >>> lc_rebinned = lc.rebin(binsize=10.0, method='sum')
    >>> lc_rebinned.plot()
    
    English
    -------
    Rebin lightcurve to new time resolution by grouping and aggregating data points.
    """
    from . import file as f
    if lc.value.ndim > 1:
        raise NotImplementedError("Rebin for multi-band LC not yet supported; slice bands first.")
    
    tmin, tmax = float(lc.time.min()), float(lc.time.max())
    nbins = max(1, int(np.ceil((tmax - tmin) / binsize)))
    edges = tmin + np.arange(nbins + 1) * binsize
    centers = 0.5 * (edges[:-1] + edges[1:])
    
    # 分配每个原始点到 bin
    bin_idx = np.digitize(lc.time, edges) - 1
    bin_idx = np.clip(bin_idx, 0, nbins - 1)
    
    new_val = np.zeros(nbins, dtype=float)
    new_err = np.zeros(nbins, dtype=float) if lc.error is not None else None
    
    for i in range(nbins):
        sel = (bin_idx == i)
        if not np.any(sel):
            continue
        if method == 'sum':
            new_val[i] = np.sum(lc.value[sel])
            if lc.error is not None:
                new_err[i] = np.sqrt(np.sum(lc.error[sel] ** 2))
        else:  # mean
            new_val[i] = np.mean(lc.value[sel])
            if lc.error is not None:
                new_err[i] = np.sqrt(np.sum(lc.error[sel] ** 2)) / np.sum(sel)
    
    return f.LightcurveData(
        kind='lc', path=lc.path,
        time=centers, value=new_val, error=new_err, dt=binsize,
        exposure=lc.exposure, is_rate=(method == 'mean'),
        header=lc.header, meta=lc.meta, headers_dump=lc.headers_dump,
        region=lc.region,
    )


# ==================== PHA Operations ====================

def slice_pha(
    pha: 'PhaData',
    *,
    emin: Optional[float] = None,
    emax: Optional[float] = None,
    ch_lo: Optional[int] = None,
    ch_hi: Optional[int] = None,
) -> 'PhaData':
    """按能量或道范围筛选 PHA，返回新实例。

    参数
    - pha: 输入 PHA 数据
    - emin/emax: 能量范围(keV); 需 ebounds 存在
    - ch_lo/ch_hi: 道范围（优先级低于能量）

    返回
    - 新的 PhaData 实例

    English
    Filter PHA by energy (needs ebounds) or channel range; returns new instance.
    """
    from .file import PhaData
    
    mask = np.ones(pha.channels.size, dtype=bool)
    
    if emin is not None or emax is not None:
        if pha.ebounds is None:
            raise ValueError(
                "Energy-based slicing requires EBOUNDS; use ch_lo/ch_hi instead."
            )
        ch, e_lo, e_hi = pha.ebounds
        if emin is not None:
            mask &= (e_hi > float(emin))
        if emax is not None:
            mask &= (e_lo < float(emax))
        
        # ebounds 与 channels 对齐
        idx_map = {int(c): i for i, c in enumerate(pha.channels)}
        sel_ch = ch[mask]
        sel_idx = np.array(
            [idx_map[int(c)] for c in sel_ch if int(c) in idx_map],
            dtype=int
        )
    else:
        if ch_lo is not None:
            mask &= (pha.channels >= int(ch_lo))
        if ch_hi is not None:
            mask &= (pha.channels <= int(ch_hi))
        sel_idx = np.where(mask)[0]
    
    return PhaData(
        kind='pha',
        path=pha.path,
        channels=pha.channels[sel_idx],
        counts=pha.counts[sel_idx],
        stat_err=pha.stat_err[sel_idx] if pha.stat_err is not None else None,
        exposure=pha.exposure,
        backscal=pha.backscal,
        areascal=pha.areascal,
        quality=pha.quality[sel_idx] if pha.quality is not None else None,
        grouping=pha.grouping[sel_idx] if pha.grouping is not None else None,
        ebounds=pha.ebounds,
        header=pha.header,
        meta=pha.meta,
        headers_dump=pha.headers_dump,
    )


def rebin_pha(pha: 'PhaData', factor: int) -> 'PhaData':
    """道聚合（rebinning）：每 factor 个道合并为一个。

    参数
    - pha: 输入 PHA 数据
    - factor: 聚合因子（如 2 表示两两合并）

    返回
    - 新实例，channels/counts 长度约为原来的 1/factor

    English
    Rebin PHA by grouping channels; returns new instance.
    """
    from .file import PhaData
    
    n = pha.channels.size
    nb = n // int(factor)
    
    new_ch = np.zeros(nb, dtype=int)
    new_counts = np.zeros(nb, dtype=float)
    new_err = np.zeros(nb, dtype=float) if pha.stat_err is not None else None
    
    for i in range(nb):
        start = i * factor
        end = min(start + factor, n)
        new_ch[i] = int(pha.channels[start])  # 取第一个道号
        new_counts[i] = np.sum(pha.counts[start:end])
        if pha.stat_err is not None:
            new_err[i] = np.sqrt(np.sum(pha.stat_err[start:end] ** 2))
    
    return PhaData(
        kind='pha',
        path=pha.path,
        channels=new_ch,
        counts=new_counts,
        stat_err=new_err,
        exposure=pha.exposure,
        backscal=pha.backscal,
        areascal=pha.areascal,
        quality=None,
        grouping=None,
        ebounds=pha.ebounds,
        header=pha.header,
        meta=pha.meta,
        headers_dump=pha.headers_dump,
    )


# ==================== Event Operations ====================

def slice_events(
    evt: 'EventData',
    tmin: Optional[float] = None,
    tmax: Optional[float] = None,
    *,
    pi_min: Optional[int] = None,
    pi_max: Optional[int] = None,
    ch_min: Optional[int] = None,
    ch_max: Optional[int] = None,
) -> 'EventData':
    """按时间和/或能量范围筛选事件，返回新实例。

    参数 (Parameters)
    ----------------
    evt : EventData
        输入事件数据
    tmin, tmax : float, optional
        时间下/上界（闭区间）；None 表示不限
    pi_min, pi_max : int, optional
        PI 通道范围（闭区间）；需 evt.pi 存在
    ch_min, ch_max : int, optional
        CHANNEL 范围（闭区间）；需 evt.channel 存在；优先级低于 PI

    返回 (Returns)
    -------------
    EventData
        筛选后的新事件实例

    示例 (Example)
    -------------
    >>> # 仅时间筛选
    >>> evt_t = evt.slice(tmin=100, tmax=500)
    >>> 
    >>> # 时间 + PI 能段筛选（例如 0.5-4 keV 对应 PI 50-400）
    >>> evt_filtered = evt.slice(tmin=100, tmax=500, pi_min=50, pi_max=400)
    >>> 
    >>> # 仅能段筛选
    >>> evt_energy = evt.slice(pi_min=50, pi_max=400)

    English
    -------
    Filter events by time and/or energy (PI/CHANNEL) range; returns new instance.
    """
    from .file import EventData
    
    mask = np.ones(evt.time.size, dtype=bool)
    
    # 时间筛选
    if tmin is not None:
        mask &= (evt.time >= float(tmin))
    if tmax is not None:
        mask &= (evt.time <= float(tmax))
    
    # 能量筛选：优先 PI，其次 CHANNEL
    if pi_min is not None or pi_max is not None:
        if evt.pi is None:
            raise ValueError("PI-based slicing requires evt.pi; use ch_min/ch_max instead.")
        if pi_min is not None:
            mask &= (evt.pi >= int(pi_min))
        if pi_max is not None:
            mask &= (evt.pi <= int(pi_max))
    elif ch_min is not None or ch_max is not None:
        if evt.channel is None:
            raise ValueError("CHANNEL-based slicing requires evt.channel.")
        if ch_min is not None:
            mask &= (evt.channel >= int(ch_min))
        if ch_max is not None:
            mask &= (evt.channel <= int(ch_max))
    
    return EventData(
        kind='evt',
        path=evt.path,
        time=evt.time[mask],
        pi=evt.pi[mask] if evt.pi is not None else None,
        channel=evt.channel[mask] if evt.channel is not None else None,
        gti_start=evt.gti_start,
        gti_stop=evt.gti_stop,
        header=evt.header,
        meta=evt.meta,
        headers_dump=evt.headers_dump,
    )


def rebin_events_to_lightcurve(
    evt: 'EventData',
    binsize: float,
    *,
    tmin: Optional[float] = None,
    tmax: Optional[float] = None
) -> 'LightcurveData':
    """从事件数据生成分 bin 光变曲线。
    
    将事件列表按时间分组统计，生成光变曲线。
    
    参数 (Parameters)
    ----------------
    evt : EventData
        输入事件数据
    binsize : float
        时间分辨率（秒），即每个 bin 的宽度
    tmin, tmax : float, optional
        可选的时间范围；默认使用全部事件的时间范围
    
    返回 (Returns)
    -------------
    LightcurveData
        生成的光变曲线（COUNTS 模式）
    
    原理 (Principle)
    ---------------
    1. 根据 binsize 定义时间网格
    2. 用直方图统计每个 bin 内的事件数
    3. 误差假设为泊松分布：σ = √N
    
    示例 (Example)
    -------------
    >>> # 从事件生成 1 秒 bin 的光变曲线
    >>> lc = evt.rebin(binsize=1.0)
    >>> lc.plot()
    
    English
    -------
    Bin events into lightcurve with given time resolution; returns LightcurveData.
    """
    from . import file as f
    t = evt.time
    if tmin is None:
        tmin = float(t.min()) if t.size > 0 else 0.0
    if tmax is None:
        tmax = float(t.max()) if t.size > 0 else tmin + binsize
    
    nbins = max(1, int(np.ceil((tmax - tmin) / binsize)))
    edges = tmin + np.arange(nbins + 1) * binsize
    hist, _ = np.histogram(t, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    err = np.sqrt(hist)
    
    return f.LightcurveData(
        kind='lc', path=evt.path,
        time=centers, value=hist.astype(float), error=err, dt=binsize,
        exposure=None, is_rate=False,
        header=evt.header, meta=evt.meta, headers_dump=evt.headers_dump,
        region=None,
    )
