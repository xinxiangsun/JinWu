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
from .utils import snr_li_ma
from astropy.stats import bayesian_blocks
from .file import LightcurveData, PhaData, EventData


if TYPE_CHECKING:
    pass

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
    # Bayesian Blocks
    "BayesianBlocksBinner",
    "bin_bblocks",
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
    if lc.value.ndim > 1:
        raise NotImplementedError("Rebin for multi-band LC not yet supported; slice bands first.")

    # Determine original bin edges. If lc.dt is provided, assume `lc.time` are
    # bin centers and construct edges accordingly. Otherwise treat times as
    # instantaneous and use small epsilon half-width equal to median spacing.
    t = np.asarray(lc.time, dtype=float)
    if t.size == 0:
        return LightcurveData(path=lc.path, time=np.array([], dtype=float), value=np.array([], dtype=float), error=None, dt=binsize, exposure=lc.exposure, bin_exposure=None, is_rate=lc.is_rate, header=lc.header, meta=lc.meta, headers_dump=lc.headers_dump, region=lc.region)

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

    # Per-bin宽度（允许轻微非均匀）
    orig_width = orig_right - orig_left

    # XRONOS 要求 newbin 不得短于最长原始 bin
    max_bin = float(np.max(orig_width)) if orig_width.size else float(binsize)
    if binsize < max_bin:
        binsize = max_bin

    # 对齐参考点：优先 align_ref，否则按数据本身的最左边缘。
    if align_ref is not None:
        ref = float(align_ref)
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
        # 若有逐 bin 暴露时间，优先使用它从速率恢复计数
        bin_expo = getattr(lc, 'bin_exposure', None)
        if bin_expo is not None:
            bin_expo_arr = np.asarray(bin_expo, dtype=float)
            orig_counts = vals * bin_expo_arr
            orig_err_counts = errs * bin_expo_arr if errs is not None else None
        else:
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
            frac = overlap / float(orig_width[i])
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

    # 始终返回累积的新 bin 曝光，用于下游转换/筛选
    ret_bin_exposure = new_exposure

    return LightcurveData(
        path=lc.path,
        time=centers,
        value=out_value,
        error=out_err,
        dt=out_dt,
        # 时间与参考点
        timezero=getattr(lc, 'timezero', 0.0),
        timezero_obj=getattr(lc, 'timezero_obj', None),
        bin_lo=edges[:-1],
        bin_hi=edges[1:],
        # 曝光
        exposure=(float(np.sum(ret_bin_exposure)) if ret_bin_exposure is not None else lc.exposure),
        bin_exposure=(ret_bin_exposure if ret_bin_exposure is not None else None),
        is_rate=out_is_rate,
        # 误差分布与分离存储占位
        err_dist=getattr(lc, 'err_dist', None),
        counts=None if out_is_rate else out_value,
        rate=out_value if out_is_rate else None,
        counts_err=None if out_is_rate else out_err,
        rate_err=out_err if out_is_rate else None,
        # GTI 及元数据
        gti_start=getattr(lc, 'gti_start', None),
        gti_stop=getattr(lc, 'gti_stop', None),
        quality=getattr(lc, 'quality', None),
        fracexp=getattr(lc, 'fracexp', None),
        backscal=getattr(lc, 'backscal', None),
        areascal=getattr(lc, 'areascal', None),
        telescop=getattr(lc, 'telescop', None),
        timesys=getattr(lc, 'timesys', None),
        mjdref=getattr(lc, 'mjdref', None),
        header=lc.header,
        meta=lc.meta,
        headers_dump=lc.headers_dump,
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


def rebin_pha(pha: 'PhaData', *, factor: Optional[int] = None, min_counts: Optional[float] = None) -> 'PhaData':
    """道聚合（rebinning）：按固定因子或最小计数阈值合并道。

    参数
    - pha: 输入 PHA 数据
    - factor: 固定聚合因子（如 2 表示两两合并）；与 min_counts 互斥
    - min_counts: 基于最小计数阈值聚合（调用 grppha 的方法）；若既未提供 factor 也未提供 min_counts，
                  将使用 pha.grouping 若存在，否则默认 factor=1（不聚合）

    返回
    - 新实例，channels/counts 长度取决于聚合方式

    English
    Rebin PHA by grouping channels (fixed factor, min counts, or existing grouping); returns new instance.
    """
    from ..ftools.grppha import compute_grouping_by_min_counts

    ch = pha.channels
    cnt = pha.counts
    err = pha.stat_err

    # 确定分组数组
    grouping = None
    if min_counts is not None:
        # 基于最小计数阈值的贪心聚合
        grouping = compute_grouping_by_min_counts(cnt, min_counts)
    elif factor is not None and factor > 1:
        # 固定因子聚合
        n = ch.size
        grouping = np.zeros(n, dtype=int)
        gid = 1
        for i in range(n):
            grouping[i] = gid
            if (i + 1) % int(factor) == 0 and i < n - 1:
                gid += 1
    elif getattr(pha, 'grouping', None) is not None:
        # 使用已有的 grouping 数组
        grouping = np.asarray(pha.grouping, dtype=int)
    else:
        # 默认：不聚合（factor=1）
        return pha

    # 按 grouping 数组聚合
    gids = np.unique(grouping[grouping > 0])
    if gids.size == 0:
        return pha

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
    new_err = np.asarray(new_err, dtype=float) if new_err else None

    # 聚合 EBOUNDS（若存在）
    new_ebounds = None
    if pha.ebounds is not None:
        ch_all, e_lo, e_hi = pha.ebounds
        eb_ch = []
        eb_lo = []
        eb_hi = []
        for gid in gids:
            mask = grouping == int(gid)
            idxs = np.where(mask)[0]
            eb_ch.append(int(ch[idxs][0]))
            eb_lo.append(float(np.min(e_lo[idxs])))
            eb_hi.append(float(np.max(e_hi[idxs])))
        new_ebounds = (np.asarray(eb_ch, dtype=int), np.asarray(eb_lo, dtype=float), np.asarray(eb_hi, dtype=float))

    return PhaData(
        path=pha.path,
        channels=new_ch,
        counts=new_counts,
        stat_err=new_err if new_err is not None and new_err.size > 0 else None,
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

    return LightcurveData(
        path=evt.path,
        time=centers, value=hist.astype(float), error=err, dt=binsize,
        exposure=float(np.sum(bin_exposure)), is_rate=False,
        header=evt.header, meta=evt.meta, headers_dump=evt.headers_dump,
        region=None, bin_exposure=bin_exposure,
    )


# ==================== Bayesian Blocks Binning ====================

class BayesianBlocksBinner:
    """基于贝叶斯块的自适应分 bin，并确保每个 bin 的 SNR≥阈值。

    用法
    ----
    - 传入 `LightcurveData`（counts 或 rate），采用 `astropy.stats.bayesian_blocks`
      计算时间边界；随后按 SNR 阈值（默认 3）合并相邻块以满足要求。

    参数
    ----
    - p0: False positive rate (Scargle 2013)，控制块数量敏感度
    - min_snr: 每个输出块的最小 SNR 阈值（默认 3.0）
    - fitness: Bayesian Blocks 统计模型，可选：
      * 'events': 泊松事件（光子计数等）
      * 'regular_events': 规则采样的事件数据
      * 'measures': 带误差的测量值（高斯统计，适用于已分bin的光变）

    返回
    ----
    - `LightcurveData` 新实例（counts 或 rate 与输入一致），时间为块中心，
      值为每块的聚合值，误差按平方和开方传播；每块的 `bin_exposure` 为实际覆盖曝光。
    """

    def __init__(self, p0: float = 0.05, min_snr: float = 3.0, fitness: str = 'regular') -> None:
        self.p0 = float(p0)
        self.min_snr = float(min_snr)
        self.fitness = str(fitness)
        # 暴露接口：便于后续 Txx 计算使用原始或合并后的边界
        self.last_edges: Optional[np.ndarray] = None
        self.last_merged_indices: Optional[list[np.ndarray]] = None

    def _compute_snr(self, counts: np.ndarray, var: np.ndarray) -> float:
        # SNR = sum(counts) / sqrt(sum(var))；若输入为 rate，外层已换算为 counts 再计算
        s = float(np.sum(counts))
        v = float(np.sum(var))
        if v <= 0.0:
            return 0.0
        return s / float(np.sqrt(v))

    def fit(self, lc: 'LightcurveData') -> 'LightcurveData':
        try:
            from astropy.stats import bayesian_blocks
        except Exception:
            raise RuntimeError("需要 astropy.stats.bayesian_blocks 支持，请安装 astropy>=4.0")

        if lc.value.ndim > 1:
            raise NotImplementedError("暂不支持多能段 LC 的贝叶斯块分 bin；请先按能段切片")

        t = np.asarray(lc.time, dtype=float)
        if t.size == 0:
            return LightcurveData(path=lc.path, time=np.array([], dtype=float), value=np.array([], dtype=float), error=None, dt=lc.dt, exposure=lc.exposure, bin_exposure=None, is_rate=lc.is_rate, header=lc.header, meta=lc.meta, headers_dump=lc.headers_dump, region=lc.region)

        # 将输入统一到 counts 及其方差，便于块内聚合与 SNR 计算
        vals = np.asarray(lc.value, dtype=float)
        errs = np.asarray(lc.error, dtype=float) if lc.error is not None else None

        # 推断原始 bin 边界与 dt
        if lc.dt is not None and lc.dt > 0:
            orig_dt = float(lc.dt)
            left = t - 0.5 * orig_dt
            right = t + 0.5 * orig_dt
        else:
            est_dt = float(np.median(np.diff(t))) if t.size >= 2 else 1.0
            orig_dt = est_dt
            left = t - 0.5 * est_dt
            right = t + 0.5 * est_dt

        if lc.is_rate:
            counts = vals * orig_dt
            err_counts = (errs * orig_dt) if errs is not None else None
        else:
            counts = vals.copy()
            err_counts = errs.copy() if errs is not None else None

        if err_counts is None:
            err_counts = np.sqrt(np.maximum(counts, 0.0))

        var_counts = err_counts ** 2

        # 调用 bayesian_blocks 构建初始块边界
        # fitness 类型：'events'(泊松事件), 'regular_events'(规则采样事件), 'measures'(带误差测量)
        edges = bayesian_blocks(t, counts, fitness=self.fitness, p0=self.p0)
        # heapy/ppsignal 的做法会扩展首末边界到原始范围；保持与之兼容
        full_left = float(left.min())
        full_right = float(right.max())
        if edges.size > 0:
            edges = np.concatenate(([full_left], edges, [full_right]))
        else:
            edges = np.asarray([full_left, full_right], dtype=float)
        self.last_edges = edges.copy()

        # 根据块边界，将原始 bins 分配到各块并聚合，随后按 SNR 阈值合并相邻块
        nb = edges.size - 1
        block_slices = []
        for i in range(nb):
            a = edges[i]
            b = edges[i + 1]
            mask = (left < b) & (right > a)
            block_slices.append(np.where(mask)[0])

        # 初步块的合并以满足 SNR 阈值
        merged_indices = []  # 列表项为索引数组
        i = 0
        while i < nb:
            curr = block_slices[i]
            # 聚合当前块的 counts/var
            csum = np.sum(counts[curr])
            vsum = np.sum(var_counts[curr])
            snr = (csum / np.sqrt(vsum)) if vsum > 0 else 0.0
            j = i
            while snr < self.min_snr and (j + 1) < nb:
                j += 1
                curr = np.concatenate((curr, block_slices[j]))
                csum = np.sum(counts[curr])
                vsum = np.sum(var_counts[curr])
                snr = (csum / np.sqrt(vsum)) if vsum > 0 else 0.0
            merged_indices.append(np.unique(curr))
            i = j + 1

        # 保存以便 Txx 使用
        self.last_merged_indices = [np.asarray(ix, dtype=int) for ix in merged_indices]
        # 生成输出 LC：每个合并后的块 -> 一个点
        out_time = []
        out_val = []
        out_err = []
        out_expo = []
        out_dt = []

        # 计算每块的实际曝光：使用原始 bin_exposure 如有，否则用时间覆盖长度
        orig_expo = getattr(lc, 'bin_exposure', None)
        for idxs in merged_indices:
            a = float(np.min(left[idxs]))
            b = float(np.max(right[idxs]))
            # 聚合 counts/var
            csum = float(np.sum(counts[idxs]))
            vsum = float(np.sum(var_counts[idxs]))
            # 值/误差空间：保留与输入一致（counts 或 rate）
            if lc.is_rate:
                val = csum / (b - a)
                err = (np.sqrt(vsum) / (b - a)) if vsum > 0 else 0.0
            else:
                val = csum
                err = (np.sqrt(vsum)) if vsum > 0 else 0.0
            out_time.append(0.5 * (a + b))
            out_val.append(val)
            out_err.append(err)
            out_dt.append(b - a)
            if orig_expo is not None:
                out_expo.append(float(np.sum(orig_expo[idxs])))
            else:
                out_expo.append(b - a)

        out_time = np.asarray(out_time, dtype=float)
        out_val = np.asarray(out_val, dtype=float)
        out_err = np.asarray(out_err, dtype=float)
        out_dt = float(np.median(out_dt)) if len(out_dt) > 0 else (lc.dt or 0.0)
        bin_exposure = np.asarray(out_expo, dtype=float)

        return LightcurveData(
            path=lc.path,
            time=out_time,
            value=out_val,
            error=out_err,
            dt=out_dt,
            exposure=float(np.sum(bin_exposure)),
            bin_exposure=bin_exposure,
            is_rate=lc.is_rate,
            header=lc.header,
            meta=lc.meta,
            headers_dump=lc.headers_dump,
            region=lc.region,
        )

    def fit_src_bkg(self, lc_src: 'LightcurveData', lc_bkg: 'LightcurveData', alpha: Optional[float] = None) -> 'LightcurveData':
        """对源与背景光变同时进行贝叶斯块分 bin，并按 Li&Ma 近似的 SNR 过滤。

        - `alpha`: 源/背景缩放因子（如面积或BACKSCAL比值）。若未提供，
          尝试以每块曝光之比近似：alpha = expo_src / expo_bkg。
        - SNR 计算：net = S - alpha*B；var = S + alpha^2 * B；SNR = net/sqrt(var)。
        - 初始块边界基于源 LC 的 `bayesian_blocks`。
        """
        

        # 统一到 counts 域
        def lc_to_counts(lc: 'LightcurveData'):
            t = np.asarray(lc.time, dtype=float)
            vals = np.asarray(lc.value, dtype=float)
            errs = np.asarray(lc.error, dtype=float) if lc.error is not None else None
            if lc.dt is not None and lc.dt > 0:
                dt = float(lc.dt)
                left = t - 0.5 * dt
                right = t + 0.5 * dt
            else:
                est_dt = float(np.median(np.diff(t))) if t.size >= 2 else 1.0
                dt = est_dt
                left = t - 0.5 * est_dt
                right = t + 0.5 * est_dt
            if lc.is_rate:
                c = vals * dt
                e = (errs * dt) if errs is not None else None
            else:
                c = vals.copy()
                e = errs.copy() if errs is not None else None
            if e is None:
                e = np.sqrt(np.maximum(c, 0.0))
            return t, left, right, c, e

        t_s, l_s, r_s, c_s, e_s = lc_to_counts(lc_src)
        t_b, l_b, r_b, c_b, e_b = lc_to_counts(lc_bkg)
        if t_s.size == 0:
            return LightcurveData(path=lc_src.path, time=np.array([], dtype=float), value=np.array([], dtype=float), error=None, dt=lc_src.dt, exposure=lc_src.exposure, bin_exposure=None, is_rate=lc_src.is_rate, header=lc_src.header, meta=lc_src.meta, headers_dump=lc_src.headers_dump, region=lc_src.region)

        # 初始块边界使用源计数
        edges = bayesian_blocks(t_s, c_s, fitness=self.fitness, p0=self.p0)
        # 对齐 heapy/ppsignal 的边界扩展策略
        full_left = float(l_s.min())
        full_right = float(r_s.max())
        if edges.size > 0:
            edges = np.concatenate(([full_left], edges, [full_right]))
        else:
            edges = np.asarray([full_left, full_right], dtype=float)
        self.last_edges = edges.copy()

        # 将源/背景原始 bins 分配到各块
        nb = edges.size - 1
        src_slices = []
        bkg_slices = []
        for i in range(nb):
            a = edges[i]
            b = edges[i + 1]
            src_slices.append(np.where((l_s < b) & (r_s > a))[0])
            bkg_slices.append(np.where((l_b < b) & (r_b > a))[0])

        # 合并相邻块直到满足 SNR 阈值（Li&Ma近似）
        merged = []
        i = 0
        while i < nb:
            src_idx = src_slices[i]
            bkg_idx = bkg_slices[i]
            S = float(np.sum(c_s[src_idx]))
            B = float(np.sum(c_b[bkg_idx]))
            # 估计每块曝光（若存在 bin_exposure）
            expo_src_blk = None
            expo_bkg_blk = None
            if getattr(lc_src, 'bin_exposure', None) is not None:
                expo_src_blk = float(np.sum(np.asarray(lc_src.bin_exposure)[src_idx]))
            if getattr(lc_bkg, 'bin_exposure', None) is not None:
                expo_bkg_blk = float(np.sum(np.asarray(lc_bkg.bin_exposure)[bkg_idx]))
            alpha_blk = alpha
            if alpha_blk is None:
                if (expo_src_blk is not None) and (expo_bkg_blk is not None) and (expo_bkg_blk > 0):
                    alpha_blk = expo_src_blk / expo_bkg_blk
                else:
                    alpha_blk = 1.0
            # 使用统一的 Li&Ma 近似 SNR（保持与 heapy 使用思路一致）
            # 这里用近似的 net/sqrt(var) 版本与 utils.snr_li_ma 一致性：当计数较大时接近
            net = S - alpha_blk * B
            var = S + (alpha_blk ** 2) * B
            snr = (net / np.sqrt(var)) if var > 0 else 0.0
            j = i
            while snr < self.min_snr and (j + 1) < nb:
                j += 1
                src_idx = np.concatenate((src_idx, src_slices[j]))
                bkg_idx = np.concatenate((bkg_idx, bkg_slices[j]))
                S = float(np.sum(c_s[src_idx]))
                B = float(np.sum(c_b[bkg_idx]))
                expo_src_blk = None
                expo_bkg_blk = None
                if getattr(lc_src, 'bin_exposure', None) is not None:
                    expo_src_blk = float(np.sum(np.asarray(lc_src.bin_exposure)[src_idx]))
                if getattr(lc_bkg, 'bin_exposure', None) is not None:
                    expo_bkg_blk = float(np.sum(np.asarray(lc_bkg.bin_exposure)[bkg_idx]))
                alpha_blk = alpha
                if alpha_blk is None:
                    if (expo_src_blk is not None) and (expo_bkg_blk is not None) and (expo_bkg_blk > 0):
                        alpha_blk = expo_src_blk / expo_bkg_blk
                    else:
                        alpha_blk = 1.0
                net = S - alpha_blk * B
                var = S + (alpha_blk ** 2) * B
                snr = (net / np.sqrt(var)) if var > 0 else 0.0
            merged.append((np.unique(src_idx), np.unique(bkg_idx)))
            i = j + 1

        # 输出 LC（以净计数或净率表示；误差按 var 的 sqrt）
        self.last_merged_indices = []
        out_time = []
        out_val = []
        out_err = []
        out_expo = []
        out_dt_list = []
        for src_idx, bkg_idx in merged:
            a = float(np.min(l_s[src_idx]))
            b = float(np.max(r_s[src_idx]))
            S = float(np.sum(c_s[src_idx]))
            B = float(np.sum(c_b[bkg_idx]))
            expo_src_blk = None
            expo_bkg_blk = None
            if getattr(lc_src, 'bin_exposure', None) is not None:
                expo_src_blk = float(np.sum(np.asarray(lc_src.bin_exposure)[src_idx]))
            if getattr(lc_bkg, 'bin_exposure', None) is not None:
                expo_bkg_blk = float(np.sum(np.asarray(lc_bkg.bin_exposure)[bkg_idx]))
            alpha_blk = alpha
            if alpha_blk is None:
                if (expo_src_blk is not None) and (expo_bkg_blk is not None) and (expo_bkg_blk > 0):
                    alpha_blk = expo_src_blk / expo_bkg_blk
                else:
                    alpha_blk = 1.0
            net = S - alpha_blk * B
            var = S + (alpha_blk ** 2) * B
            # 输出空间：保持输入源 LC 的 is_rate 习惯
            if lc_src.is_rate:
                duration = (b - a)
                val = net / duration
                err = (np.sqrt(var) / duration) if var > 0 else 0.0
            else:
                val = net
                err = (np.sqrt(var)) if var > 0 else 0.0
            out_time.append(0.5 * (a + b))
            out_val.append(val)
            out_err.append(err)
            out_dt_list.append(b - a)
            out_expo.append(expo_src_blk if expo_src_blk is not None else (b - a))
            # 保存索引用于 Txx 接口
            self.last_merged_indices.append(np.asarray(src_idx, dtype=int))

        out_time = np.asarray(out_time, dtype=float)
        out_val = np.asarray(out_val, dtype=float)
        out_err = np.asarray(out_err, dtype=float)
        out_dt = float(np.median(out_dt_list)) if len(out_dt_list) > 0 else (lc_src.dt or 0.0)
        bin_exposure = np.asarray(out_expo, dtype=float)

        return LightcurveData(
            path=lc_src.path,
            time=out_time,
            value=out_val,
            error=out_err,
            dt=out_dt,
            exposure=float(np.sum(bin_exposure)),
            bin_exposure=bin_exposure,
            is_rate=lc_src.is_rate,
            header=lc_src.header,
            meta=lc_src.meta,
            headers_dump=lc_src.headers_dump,
            region=lc_src.region,
        )


def bin_bblocks(
    lc,
    background: Optional['LightcurveData'] = None,
    *,
    alpha: Optional[float] = None,
    p0: float = 0.05,
    min_snr: float = 3.0
) -> 'LightcurveData':
    """贝叶斯块自适应分 bin，支持单独光变或源+背景联合处理。

    参数
    ----
    lc : LightcurveData | LightcurveDataset
        - 若为 `LightcurveData`：视为源光变；需配合 `background` 参数传入背景（可选）。
        - 若为 `LightcurveDataset`：自动提取 `.data` 作为源，`.background.data` 作为背景，
          `.area_ratio` 作为默认 alpha（若未显式传入）。
    background : LightcurveData, optional
        背景光变数据（仅当 `lc` 为 `LightcurveData` 时需要）。
        若 `lc` 为 `LightcurveDataset` 且已有 `.background`，此参数被忽略。
    alpha : float, optional
        源/背景缩放因子（面积或 BACKSCAL 比值）。优先级：
        1. 显式传入的 `alpha` 参数（最高）
        2. `LightcurveDataset.area_ratio`（若输入为 dataset）
        3. `lc_src.region.area / lc_bkg.region.area`（若 region 存在）
        4. 回退为 1.0
    p0 : float, default=0.05
        Bayesian Blocks 的假阳性率（控制分块敏感度）。
    min_snr : float, default=3.0
        每个输出块的最小 SNR 阈值；低于此值的相邻块会被合并。

    返回
    ----
    LightcurveData
        分 bin 后的光变（若有背景则为净光变）。

    示例
    ----
    >>> # 1. 单独源光变（无背景）
    >>> lc_binned = bin_lightcurve_bblocks(lc_src, p0=0.05, min_snr=3.0)
    >>>
    >>> # 2. 源+背景（直接传 LightcurveData）
    >>> lc_net = bin_lightcurve_bblocks(lc_src, background=lc_bkg, alpha=1.2, p0=0.05)
    >>>
    >>> # 3. 传入 LightcurveDataset（自动提取 background 和 area_ratio）
    >>> ds = netdata(lc_src, lc_bkg, area_ratio=1.2)
    >>> lc_net = bin_lightcurve_bblocks(ds, p0=0.05, min_snr=3.0)
    """
    # 判断输入类型并提取源/背景/alpha
    try:
        # 尝试作为 LightcurveDataset（检查是否有 .data 属性和 LightcurveData 类型）
        if hasattr(lc, 'data') and isinstance(getattr(lc, 'data', None), LightcurveData):
            # LightcurveDataset 输入
            lc_src = lc.data
            lc_bkg = getattr(lc.background, 'data', None) if getattr(lc, 'background', None) is not None else None
            # alpha 优先级：显式传入 > dataset.area_ratio > region 推断 > 1.0
            if alpha is None:
                alpha = getattr(lc, 'area_ratio', None)
        else:
            # LightcurveData 输入
            lc_src = lc
            lc_bkg = background
    except Exception:
        # 回退：当作 LightcurveData
        lc_src = lc
        lc_bkg = background

    # 无背景：单独源光变分 bin
    if lc_bkg is None:
        return BayesianBlocksBinner(p0=p0, min_snr=min_snr, fitness='regular_events').fit(lc_src)

    # 有背景：源+背景联合分 bin
    # alpha 最终回退逻辑（若前面未设置）
    if alpha is None:
        try:
            a_src = float(getattr(getattr(lc_src, 'region', None), 'area', None) or 0.0)
            a_bkg = float(getattr(getattr(lc_bkg, 'region', None), 'area', None) or 0.0)
            alpha = (a_src / a_bkg) if (a_src > 0 and a_bkg > 0) else 1.0
        except Exception:
            alpha = 1.0
    return BayesianBlocksBinner(p0=p0, min_snr=min_snr, fitness='regular_events').fit_src_bkg(lc_src, lc_bkg, alpha=alpha)
