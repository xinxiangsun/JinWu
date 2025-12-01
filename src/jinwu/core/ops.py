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
from . import gti as gtimod
from ..ftools import xselect_mdb
from pathlib import Path as _Path
import warnings

if TYPE_CHECKING:
    from .file import LightcurveData, PhaData, EventData

__all__ = [
    # Lightcurve operations
    "slice_lightcurve",
    "rebin_lightcurve",
    # PHA operations
    "slice_pha",
    "rebin_pha",
    # ARF/RMF operations (none exported)
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
        path=lc.path,
        time=lc.time[mask],
        value=lc.value[mask] if lc.value.ndim == 1 else lc.value[mask, :],
        error=(
            lc.error[mask] if (lc.error is not None and lc.error.ndim == 1)
            else (lc.error[mask, :] if lc.error is not None else None)
        ),
        dt=lc.dt,
        exposure=lc.exposure,
        bin_exposure=(lc.bin_exposure[mask] if (hasattr(lc, 'bin_exposure') and lc.bin_exposure is not None) else None),
        is_rate=lc.is_rate,
        header=lc.header,
        meta=lc.meta,
        headers_dump=lc.headers_dump,
        region=lc.region,
    )


def rebin_lightcurve(
    lc: 'LightcurveData',
    binsize: float,
    method: Literal['sum', 'mean'] = 'sum',
    *,
    align_ref: Optional[float] = None,
    empty_bin: Literal['zero', 'nan'] = 'zero',
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

    # Determine original bin edges. If lc.dt is provided, assume `lc.time` are
    # bin centers and construct edges accordingly. Otherwise treat times as
    # instantaneous and use small epsilon half-width equal to median spacing.
    t = np.asarray(lc.time, dtype=float)
    if t.size == 0:
        return f.LightcurveData(path=lc.path, time=np.array([], dtype=float), value=np.array([], dtype=float), error=None, dt=binsize, exposure=lc.exposure, bin_exposure=None, is_rate=lc.is_rate, header=lc.header, meta=lc.meta, headers_dump=lc.headers_dump, region=lc.region)

    # infer original dt per bin
    if lc.dt is not None and lc.dt > 0:
        orig_dt = float(lc.dt)
        orig_left = t - 0.5 * orig_dt
        orig_right = t + 0.5 * orig_dt
    else:
        # fallback: estimate spacing and assume these are centers
        est_dt = float(np.median(np.diff(t))) if t.size >= 2 else float(binsize)
        orig_dt = est_dt
        orig_left = t - 0.5 * est_dt
        orig_right = t + 0.5 * est_dt

    # Choose reference alignment for new bins. Preference order:
    # 1) explicit `align_ref` argument
    # 2) lc.meta.timezero if present
    # 3) left edge of the first original bin
    if align_ref is not None:
        ref = float(align_ref)
    else:
        meta = getattr(lc, 'meta', None)
        if meta is not None and getattr(meta, 'timezero', None) is not None:
            ref = float(meta.timezero)
        else:
            ref = float(orig_left.min())

    # Compute number of bins so that range [ref, last_edge] covers original data
    tmax = float(orig_right.max())
    nbins = max(1, int(np.ceil((tmax - ref) / binsize)))
    edges = ref + np.arange(nbins + 1, dtype=float) * binsize
    centers = 0.5 * (edges[:-1] + edges[1:])

    # Convert original values to counts for safe aggregation
    vals = np.asarray(lc.value, dtype=float)
    errs = np.asarray(lc.error, dtype=float) if lc.error is not None else None
    if lc.is_rate:
        orig_counts = vals * orig_dt
        orig_err_counts = errs * orig_dt if errs is not None else None
    else:
        orig_counts = vals.copy()
        orig_err_counts = errs.copy() if errs is not None else None

    # If no explicit errors, assume Poisson on counts
    if orig_err_counts is None:
        # small safeguard: counts may be float; ensure non-negative
        orig_err_counts = np.sqrt(np.maximum(orig_counts, 0.0))

    new_counts = np.zeros(nbins, dtype=float)
    new_var = np.zeros(nbins, dtype=float)
    # accumulate new-bin effective exposure (seconds)
    new_exposure = np.zeros(nbins, dtype=float)

    # original per-bin exposure if provided
    orig_bin_expos = getattr(lc, 'bin_exposure', None)
    if orig_bin_expos is not None:
        orig_bin_expos = np.asarray(orig_bin_expos, dtype=float)
        # If MDB requests adjustgti, attempt to rebuild original per-bin exposures
        try:
            # load MDB tree (package-local cache)
            base = _Path(__file__).resolve().parents[1]
            mdb_path = base / 'data' / 'xselect.mdb'
            cache_path = str(_Path(__file__).resolve().parents[1] / 'xselect_mdb.pkl')
            if mdb_path.exists():
                tree = xselect_mdb.load_mdb(str(mdb_path), use_cache=True, cache_path=cache_path)
            else:
                tree = xselect_mdb.load_mdb('jinwu/src/jinwu/data/xselect.mdb') if _Path('jinwu/src/jinwu/data/xselect.mdb').exists() else {}

            adj_flag, tp_val, frame_dt = xselect_mdb.infer_adjustgti_timepixr_and_frame(tree, lc.header if hasattr(lc,'header') else None, lc.meta if hasattr(lc,'meta') else None)
            if adj_flag and (frame_dt is not None):
                # build GTI intervals from original per-bin exposures: contiguous bins with exposure>0
                pos = orig_bin_expos > 0.0
                if np.any(pos):
                    # orig_left/right computed above
                    starts = []
                    stops = []
                    i = 0
                    n = pos.size
                    while i < n:
                        if not pos[i]:
                            i += 1
                            continue
                        j = i
                        while j + 1 < n and pos[j + 1]:
                            j += 1
                        starts.append(float(orig_left[i]))
                        stops.append(float(orig_right[j]))
                        i = j + 1
                    if starts:
                        ms = np.asarray(starts, dtype=float)
                        me = np.asarray(stops, dtype=float)
                        ms_adj, me_adj = gtimod.adjust_gti_to_frame(ms, me, float(frame_dt), timepixr=float(tp_val or 0.0))
                        if ms_adj is not None and me_adj is not None:
                            # recompute per-original-bin exposures based on adjusted GTIs
                            new_orig_expos = np.zeros_like(orig_bin_expos, dtype=float)
                            for idx in range(orig_bin_expos.size):
                                a = float(orig_left[idx])
                                b = float(orig_right[idx])
                                ov = 0.0
                                for s, e in zip(ms_adj, me_adj):
                                    ov += max(0.0, min(b, float(e)) - max(a, float(s)))
                                new_orig_expos[idx] = ov
                            orig_bin_expos = new_orig_expos
        except Exception:
            # conservative: if anything fails, keep original exposures
            pass

    # For each original bin, distribute its counts into overlapping new bins
    for i in range(orig_counts.size):
        a = orig_left[i]
        b = orig_right[i]
        if b <= edges[0] or a >= edges[-1]:
            continue
        # find overlapping new bin indices
        j0 = int(np.searchsorted(edges, a, side='right') - 1)
        j1 = int(np.searchsorted(edges, b, side='left') - 1)
        j0 = max(0, min(nbins - 1, j0))
        j1 = max(0, min(nbins - 1, j1))
        for j in range(j0, j1 + 1):
            new_l = edges[j]
            new_r = edges[j + 1]
            overlap = max(0.0, min(b, new_r) - max(a, new_l))
            if overlap <= 0.0:
                continue
            frac = overlap / orig_dt
            contrib = orig_counts[i] * frac
            new_counts[j] += contrib
            new_var[j] += (orig_err_counts[i] * frac) ** 2
            # exposure contribution: if original per-bin exposure present,
            # scale it by the overlap fraction; else treat exposure as overlap length
            if orig_bin_expos is not None:
                new_exposure[j] += float(orig_bin_expos[i]) * frac
            else:
                new_exposure[j] += overlap

    # Build outputs depending on requested aggregation
    if method == 'sum':
        out_counts = new_counts
        out_err_counts = np.sqrt(new_var)
        out_is_rate = False
        out_dt = binsize
    else:  # mean -> return rates
        out_counts = new_counts
        out_err_counts = np.sqrt(new_var)
        out_is_rate = True
        out_dt = binsize

    # Convert back to desired value space (counts or rate) and handle empty bins
    if method == 'sum':
        out_value = out_counts.copy()
        out_err = out_err_counts.copy()
    else:
        out_value = out_counts / binsize
        out_err = out_err_counts / binsize
    # Decide gap/empty-bin based on exposure when possible (GTI-aware). If
    # original LC provided per-bin exposures, use accumulated new_exposure;
    # otherwise fall back to zero counts.
    if orig_bin_expos is not None:
        zero_mask = (new_exposure == 0.0)
    else:
        zero_mask = (out_counts == 0.0)

    if empty_bin == 'nan':
        out_value = out_value.astype(float)
        out_value[zero_mask] = np.nan
        out_err = out_err.astype(float)
        out_err[zero_mask] = np.nan

    # choose returned per-bin exposure array: if any non-zero exposures were
    # accumulated, return the array; else None
    ret_bin_exposure = new_exposure if np.any(new_exposure != 0.0) or np.any(new_exposure == 0.0) else None

    return f.LightcurveData(
        path=lc.path,
        time=centers, value=out_value, error=out_err, dt=out_dt,
        exposure=(float(np.sum(ret_bin_exposure)) if ret_bin_exposure is not None else lc.exposure),
        bin_exposure=(ret_bin_exposure if ret_bin_exposure is not None else None),
        is_rate=out_is_rate,
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
    
    from ..ftools.grppha import compute_grouping_by_min_counts

    # Support two modes: numeric factor (old behavior) or grouping array present in pha
    # If pha.grouping is provided and non-empty, use it to collapse channels.
    ch = pha.channels
    cnt = pha.counts
    err = pha.stat_err

    if getattr(pha, 'grouping', None) is not None:
        grouping = np.asarray(pha.grouping, dtype=int)
        gids = np.unique(grouping[grouping > 0])
        new_ch = []
        new_counts = []
        new_err = []
        for gid in gids:
            mask = grouping == int(gid)
            if not np.any(mask):
                continue
            new_ch.append(int(ch[mask][0]))
            s = float(np.sum(cnt[mask]))
            new_counts.append(s)
            if err is not None:
                new_err.append(float(np.sqrt(np.sum(err[mask] ** 2))))
            else:
                new_err.append(float(np.sqrt(s)))
        new_ch = np.asarray(new_ch, dtype=int)
        new_counts = np.asarray(new_counts, dtype=float)
        new_err = np.asarray(new_err, dtype=float)
    else:
        # Factor-based rebin as before, but handle tail channels by including them in last bin
        n = ch.size
        if n == 0:
            return PhaData(path=pha.path, channels=np.array([], dtype=int), counts=np.array([], dtype=float), stat_err=None, exposure=pha.exposure, backscal=pha.backscal, areascal=pha.areascal, quality=None, grouping=None, ebounds=pha.ebounds, header=pha.header, meta=pha.meta, headers_dump=pha.headers_dump)
        nb = int(np.ceil(n / float(factor)))
        new_ch = np.zeros(nb, dtype=int)
        new_counts = np.zeros(nb, dtype=float)
        new_err = np.zeros(nb, dtype=float)
        for i in range(nb):
            start = i * factor
            end = min(start + factor, n)
            new_ch[i] = int(ch[start])
            s = float(np.sum(cnt[start:end]))
            new_counts[i] = s
            if err is not None:
                new_err[i] = float(np.sqrt(np.sum(err[start:end] ** 2)))
            else:
                new_err[i] = float(np.sqrt(s))

    # Aggregate EBOUNDS if present: for each new channel, set e_lo=min(e_lo), e_hi=max(e_hi)
    new_ebounds = None
    if pha.ebounds is not None:
        ch_all, e_lo, e_hi = pha.ebounds
        if getattr(pha, 'grouping', None) is not None:
            # map per-group
            eb_ch = []
            eb_lo = []
            eb_hi = []
            for gid in np.unique(grouping[grouping > 0]):
                mask = grouping == int(gid)
                idxs = np.where(mask)[0]
                eb_ch.append(int(ch[idxs][0]))
                eb_lo.append(float(np.min(e_lo[idxs])))
                eb_hi.append(float(np.max(e_hi[idxs])))
            new_ebounds = (np.asarray(eb_ch, dtype=int), np.asarray(eb_lo, dtype=float), np.asarray(eb_hi, dtype=float))
        else:
            # factor-based grouping
            eb_ch = []
            eb_lo = []
            eb_hi = []
            n = ch.size
            nb = new_ch.size
            for i in range(nb):
                start = i * factor
                end = min(start + factor, n)
                idxs = np.arange(start, end, dtype=int)
                eb_ch.append(int(ch[idxs][0]))
                eb_lo.append(float(np.min(e_lo[idxs])))
                eb_hi.append(float(np.max(e_hi[idxs])))
            new_ebounds = (np.asarray(eb_ch, dtype=int), np.asarray(eb_lo, dtype=float), np.asarray(eb_hi, dtype=float))

    return PhaData(
        path=pha.path,
        channels=new_ch,
        counts=new_counts,
        stat_err=new_err if new_err.size > 0 else None,
        exposure=pha.exposure,
        backscal=pha.backscal,
        areascal=pha.areascal,
        quality=None,
        grouping=None,
        ebounds=new_ebounds if new_ebounds is not None else pha.ebounds,
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
    edges = tmin + np.arange(nbins + 1) * binsize
    centers = 0.5 * (edges[:-1] + edges[1:])

    # If GTI present, only keep events inside GTIs and compute per-bin exposure
    bin_exposure = np.zeros(nbins, dtype=float)
    if (evt.gti_start is not None) and (evt.gti_stop is not None):
        gti_s = np.asarray(evt.gti_start, dtype=float)
        gti_e = np.asarray(evt.gti_stop, dtype=float)
        # Attempt to apply adjustgti/frame alignment based on xselect.mdb
        try:
            # lazy load and reuse MDB tree
            try:
                base = _Path(__file__).resolve().parents[1]
                mdb_path = base / 'data' / 'xselect.mdb'
                cache_path = str(mdb_path) + '.pkl'
                if mdb_path.exists():
                    mdb_path_tree = xselect_mdb.load_mdb(str(mdb_path), use_cache=True, cache_path=cache_path)
                else:
                    mdb_path_tree = xselect_mdb.load_mdb('jinwu/src/jinwu/data/xselect.mdb') if _Path('jinwu/src/jinwu/data/xselect.mdb').exists() else {}
            except Exception:
                mdb_path_tree = {}

            adj_flag, tp_val, fd_val = xselect_mdb.infer_adjustgti_timepixr_and_frame(mdb_path_tree, evt.header if hasattr(evt,'header') else None, evt.meta if hasattr(evt,'meta') else None)
            if adj_flag and fd_val is not None:
                ms_adj, me_adj = gtimod.adjust_gti_to_frame(gti_s, gti_e, fd_val, timepixr=float(tp_val or 0.0))
                if ms_adj is not None and me_adj is not None:
                    gti_s, gti_e = ms_adj, me_adj
            elif adj_flag and fd_val is None:
                warnings.warn('adjustgti requested by xselect.mdb but frame_dt not found in event header/meta; skipping adjust')
        except Exception:
            # fallback: proceed without adjustgti
            pass
        # Filter events to those inside any GTI
        if t.size > 0:
            evt_mask = np.zeros(t.size, dtype=bool)
            for s, e in zip(gti_s, gti_e):
                evt_mask |= (t >= s) & (t < e)
            t_filt = t[evt_mask]
        else:
            t_filt = t
        # Compute exposure for each new bin as sum of overlaps with GTIs
        for i in range(nbins):
            a = edges[i]
            b = edges[i + 1]
            # sum overlap of [a,b) with each GTI
            ov = 0.0
            for s, e in zip(gti_s, gti_e):
                ov += max(0.0, min(b, e) - max(a, s))
            bin_exposure[i] = ov
    else:
        # No GTI: exposure equals full bin width
        t_filt = t
        bin_exposure[:] = binsize

    hist, _ = np.histogram(t_filt, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    err = np.sqrt(hist)

    return f.LightcurveData(
        path=evt.path,
        time=centers, value=hist.astype(float), error=err, dt=binsize,
        exposure=float(np.sum(bin_exposure)), is_rate=False,
        header=evt.header, meta=evt.meta, headers_dump=evt.headers_dump,
        region=None, bin_exposure=bin_exposure,
    )
