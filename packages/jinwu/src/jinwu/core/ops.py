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

from dataclasses import dataclass as _dataclass
import os as _os
import shlex as _shlex
import shutil as _shutil
import subprocess as _subprocess
import tempfile as _tempfile
from typing import Iterable as _Iterable, Mapping as _Mapping, Optional, Sequence, Literal, cast, TYPE_CHECKING

import numpy as np
from . import gti as gtimod
from .base import LightcurveDataBase, EventDataBase
from ..ftools import xselect_mdb
from pathlib import Path as _Path
import warnings
from .utils import li_ma_snr
from astropy.stats import bayesian_blocks


if TYPE_CHECKING:
    from .data import LightcurveData, PhaData, EventData


def _infer_bin_geometry(lc: 'LightcurveData') -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resolve (bin_lo, bin_hi, bin_width) with variable-bin support."""
    t = np.asarray(lc.time, dtype=float)
    if t.size == 0:
        return np.asarray([], float), np.asarray([], float), np.asarray([], float)

    if getattr(lc, 'bin_lo', None) is not None and getattr(lc, 'bin_hi', None) is not None:
        lo = np.asarray(lc.bin_lo, dtype=float)
        hi = np.asarray(lc.bin_hi, dtype=float)
        if lo.shape == hi.shape == t.shape:
            return lo, hi, hi - lo

    if getattr(lc, 'bin_width', None) is not None:
        bw = np.asarray(lc.bin_width, dtype=float)
        if bw.shape == t.shape:
            return t - 0.5 * bw, t + 0.5 * bw, bw

    dt_raw = getattr(lc, 'dt', None)
    if dt_raw is not None:
        dt_arr = np.asarray(dt_raw, dtype=float)
        if dt_arr.ndim == 0:
            dt_val = float(dt_arr)
            if np.isfinite(dt_val) and dt_val > 0:
                bw = np.full_like(t, dt_val, dtype=float)
                return t - 0.5 * bw, t + 0.5 * bw, bw
        elif dt_arr.shape == t.shape:
            bw = dt_arr
            return t - 0.5 * bw, t + 0.5 * bw, bw

    est = float(np.median(np.diff(t))) if t.size >= 2 else 1.0
    bw = np.full_like(t, est, dtype=float)
    return t - 0.5 * bw, t + 0.5 * bw, bw


def _effective_exposure_from_lc(lc: 'LightcurveData', width: np.ndarray) -> np.ndarray:
    expo = getattr(lc, 'bin_exposure', None)
    if expo is None:
        return width
    expo_arr = np.asarray(expo, dtype=float)
    if expo_arr.size == 0 or expo_arr.shape != width.shape:
        return width
    return np.where(np.isfinite(expo_arr) & (expo_arr > 0), expo_arr, width)


def _infer_binning_kind(width: np.ndarray) -> Literal['uniform', 'variable', 'unknown']:
    if width.size == 0:
        return 'unknown'
    med = float(np.median(width))
    return 'uniform' if np.allclose(width, med, rtol=1e-8, atol=1e-12) else 'variable'


def _ensure_lc_columns(columns: Optional[tuple[str, ...]], *, is_rate: bool) -> tuple[str, ...]:
    """Ensure LightcurveData columns include mandatory TIME and RATE/COUNTS tags."""
    base = tuple(columns or ())
    upper = {str(c).upper() for c in base}
    out = list(base)
    if "TIME" not in upper:
        out.append("TIME")
    if not any(k in upper for k in ("RATE", "COUNTS")):
        out.append("RATE" if is_rate else "COUNTS")
    return tuple(out)

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
    "bayesian_blocks_exposure",
    "bin_bblocks",
    "autobin",
    "txx",
    "txx_iterbkg",
]


# ==================== Lightcurve Operations ====================
from .time import Time, TimeDelta
def slice_lightcurve(
    lc: 'LightcurveData',
    tmin: Optional[float | Time | TimeDelta] = None,
    tmax: Optional[float | Time | TimeDelta] = None,
) -> 'LightcurveData':
    """按时间范围筛选光变曲线，返回新实例。

    参数
    - lc: 输入光变曲线数据
    - tmin/tmax: 时间下/上界（闭区间）；可为相对秒 (float) 或 astropy/jinwu 的 Time/TimeDelta；None 表示不限

    返回
    - 新的 LightcurveData 实例

    English
    Filter lightcurve by time range [tmin, tmax]; returns new instance.
    """
    if lc.time is None:
        raise ValueError("Lightcurve time array is None; cannot slice.")

    # 统一将时间轴转换为相对秒，便于混合类型比较。
    anchor_timezero_obj = getattr(lc, 'timezero_obj', None)
    time_data = lc.time
    if isinstance(time_data, Time):
        if anchor_timezero_obj is None:
            if time_data.size == 0:
                raise ValueError("Cannot infer anchor for Time axis because it is empty and timezero_obj is missing.")
            anchor_timezero_obj = time_data[0]
        time_seconds = (time_data - anchor_timezero_obj).to_value('sec')
    elif isinstance(time_data, TimeDelta):
        time_seconds = time_data.to_value('sec')
    else:
        time_seconds = np.asarray(time_data, dtype=float)

    def _bound_to_seconds(bound: Optional[float | Time | TimeDelta]) -> Optional[float]:
        if bound is None:
            return None
        if isinstance(bound, Time):
            if anchor_timezero_obj is None:
                raise ValueError("tmin/tmax given as Time requires lc.timezero_obj or a Time axis anchor.")
            return float((bound - anchor_timezero_obj).to_value('sec'))
        if isinstance(bound, TimeDelta):
            return float(bound.to_value('sec'))
        return float(bound)

    tmin_sec = _bound_to_seconds(tmin)
    tmax_sec = _bound_to_seconds(tmax)

    mask = np.ones(time_seconds.size, dtype=bool)
    if tmin_sec is not None:
        mask &= (time_seconds >= tmin_sec)
    if tmax_sec is not None:
        mask &= (time_seconds <= tmax_sec)

    sliced_seconds = time_seconds[mask]

    # 重新计算 timezero 和 timezero_obj（让切片后的时间从 0 开始）
    if sliced_seconds.size > 0:
        t0 = float(sliced_seconds[0])
        new_time_sec = sliced_seconds - t0
        if isinstance(time_data, (Time, TimeDelta)):
            new_time = TimeDelta(new_time_sec, format='sec')
        else:
            new_time = new_time_sec
        new_timezero = getattr(lc, 'timezero', 0.0) + t0
        if anchor_timezero_obj is not None:
            new_timezero_obj = anchor_timezero_obj + TimeDelta(t0, format='sec')
        else:
            new_timezero_obj = getattr(lc, 'timezero_obj', None)
    else:
        if isinstance(time_data, (Time, TimeDelta)):
            new_time = time_data[mask]
        else:
            new_time = time_seconds[mask]
        new_timezero = getattr(lc, 'timezero', 0.0)
        new_timezero_obj = getattr(lc, 'timezero_obj', None)
    
    lc_cls = type(lc)
    return lc_cls(
        path=lc.path,
        time=new_time,
        value=lc.value[mask] if lc.value.ndim == 1 else lc.value[mask, :],
        error=(
            lc.error[mask] if (lc.error is not None and lc.error.ndim == 1)
            else (lc.error[mask, :] if lc.error is not None else None)
        ),
        dt=(lc.dt[mask] if isinstance(lc.dt, np.ndarray) and lc.dt.shape[0] == mask.shape[0] else lc.dt),
        # 时间字段
        timezero=new_timezero,
        timezero_obj=new_timezero_obj,
        bin_lo=(lc.bin_lo[mask] if getattr(lc, 'bin_lo', None) is not None else None),
        bin_hi=(lc.bin_hi[mask] if getattr(lc, 'bin_hi', None) is not None else None),
        bin_width=(lc.bin_width[mask] if getattr(lc, 'bin_width', None) is not None else None),
        binning=getattr(lc, 'binning', 'unknown'),
        tstart=getattr(lc, 'tstart', None),
        tseg=getattr(lc, 'tseg', None),
        # 数据字段
        is_rate=lc.is_rate,
        counts=(lc.counts[mask] if getattr(lc, 'counts', None) is not None else None),
        rate=(lc.rate[mask] if getattr(lc, 'rate', None) is not None else None),
        counts_err=(lc.counts_err[mask] if getattr(lc, 'counts_err', None) is not None else None),
        rate_err=(lc.rate_err[mask] if getattr(lc, 'rate_err', None) is not None else None),
        err_dist=getattr(lc, 'err_dist', None),
        # GTI 与质量
        gti_start=getattr(lc, 'gti_start', None),
        gti_stop=getattr(lc, 'gti_stop', None),
        quality=(lc.quality[mask] if getattr(lc, 'quality', None) is not None else None),
        fracexp=(lc.fracexp[mask] if getattr(lc, 'fracexp', None) is not None else None),
        backscal=getattr(lc, 'backscal', None),
        areascal=getattr(lc, 'areascal', None),
        # 曝光
        exposure=lc.exposure,
        bin_exposure=(lc.bin_exposure[mask] if getattr(lc, 'bin_exposure', None) is not None else None),
        # 时间系统元数据
        telescop=getattr(lc, 'telescop', None),
        timesys=getattr(lc, 'timesys', None),
        mjdref=getattr(lc, 'mjdref', None),
        # 其他
        header=lc.header,
        meta=lc.meta,
        headers_dump=lc.headers_dump,
        region=lc.region,
        columns=getattr(lc, 'columns', ()),
        ratio=getattr(lc, 'ratio', None),
    )


def rebin_lightcurve(
    lc: 'LightcurveData',
    binsize: float,
    method: Literal['auto', 'sum', 'mean'] = 'auto',
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
    method : 'auto' | 'sum' | 'mean', default='auto'
        聚合方法：
        - 'auto': 保持原始纵轴形式（rate->mean, counts->sum）
        - 'sum': 对计数求和（输出为 counts）
        - 'mean': 对速率求平均（输出为 rate；GTI 缺口时按有效曝光归一化）
    
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
    lc_cls = type(lc)

    if method == 'auto':
        method = 'mean' if lc.is_rate else 'sum'

    if lc.value.ndim > 1:
        raise NotImplementedError("Rebin for multi-band LC not yet supported; slice bands first.")

    # Determine original bin edges. If lc.dt is provided, assume `lc.time` are
    # bin centers and construct edges accordingly. Otherwise treat times as
    # instantaneous and use small epsilon half-width equal to median spacing.
    t = np.asarray(lc.time, dtype=float)
    if t.size == 0:
        return lc_cls(path=lc.path, time=np.array([], dtype=float), value=np.array([], dtype=float), error=None, dt=binsize, exposure=lc.exposure, bin_exposure=None, is_rate=lc.is_rate, header=lc.header, meta=lc.meta, headers_dump=lc.headers_dump, region=lc.region, bin_width=np.array([], dtype=float), binning='unknown')

    orig_left, orig_right, orig_width = _infer_bin_geometry(lc)

    # Per-bin宽度（允许非均匀）

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
    orig_eff_expo = _effective_exposure_from_lc(lc, orig_width)
    if lc.is_rate:
        orig_counts = vals * orig_eff_expo
        orig_err_counts = errs * orig_eff_expo if errs is not None else None
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
            cache_path = xselect_mdb.default_cache_path()
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
                            new_orig_expos = np.zeros_like(orig_eff_expo, dtype=float)
                            for idx in range(orig_bin_expos.size):
                                a = float(orig_left[idx])
                                b = float(orig_right[idx])
                                ov = 0.0
                                for s, e in zip(ms_adj, me_adj):
                                    ov += max(0.0, min(b, float(e)) - max(a, float(s)))
                                new_orig_expos[idx] = ov
                            orig_eff_expo = new_orig_expos
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
                new_exposure[j] += float(orig_eff_expo[i]) * frac
            else:
                new_exposure[j] += overlap

    # Build outputs
    out_counts = new_counts
    out_err_counts = np.sqrt(new_var)
    out_dt = binsize

    # Convert back to desired value space (counts or rate) and handle empty bins
    if method == 'sum':
        out_is_rate = False
        out_value = out_counts.copy()
        out_err = out_err_counts.copy()
    else:
        out_is_rate = True
        # XRONOS/lcurve-style: rate = counts / effective exposure.
        # If we have per-bin exposures, use accumulated effective exposure;
        # otherwise fall back to binsize.
        denom = new_exposure if orig_bin_expos is not None else np.full_like(out_counts, binsize, dtype=float)
        denom_safe = np.where(denom > 0.0, denom, np.nan)
        out_value = out_counts / denom_safe
        out_err = out_err_counts / denom_safe
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

    return lc_cls(
        path=lc.path,
        time=centers,
        value=out_value,
        error=out_err,
        dt=out_dt,
        # 时间与参考点
        timezero=getattr(lc, 'timezero', -1),
        timezero_obj=getattr(lc, 'timezero_obj', None),
        bin_lo=edges[:-1],
        bin_hi=edges[1:],
        tstart=getattr(lc, 'tstart', None),
        tseg=getattr(lc, 'tseg', None),
        bin_width=np.diff(edges),
        binning='uniform',
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
        columns=getattr(lc, 'columns', ()),
        ratio=getattr(lc, 'ratio', None),
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
    
    sliced_ebounds = None
    if pha.ebounds is not None:
        eb_ch, eb_lo, eb_hi = (np.asarray(part) for part in pha.ebounds)
        lookup = {int(channel): index for index, channel in enumerate(eb_ch)}
        selected = np.asarray(pha.channels)[sel_idx]
        if any(int(channel) not in lookup for channel in selected):
            raise ValueError("EBOUNDS does not cover every selected PHA channel")
        eb_idx = np.asarray([lookup[int(channel)] for channel in selected], dtype=int)
        sliced_ebounds = (eb_ch[eb_idx], eb_lo[eb_idx], eb_hi[eb_idx])

    pha_cls = type(pha)
    return pha_cls(
        path=pha.path,
        channels=pha.channels[sel_idx],
        counts=pha.counts[sel_idx],
        stat_err=pha.stat_err[sel_idx] if pha.stat_err is not None else None,
        exposure=pha.exposure,
        backscal=pha.backscal,
        areascal=pha.areascal,
        quality=pha.quality[sel_idx] if pha.quality is not None else None,
        grouping=pha.grouping[sel_idx] if pha.grouping is not None else None,
        ebounds=sliced_ebounds,
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
    from ..ftools.grppha import compute_grouping_by_min_counts, fold_group_quality, min_counts_group_quality

    ch = pha.channels
    cnt = pha.counts
    err = pha.stat_err

    # 确定分组数组
    grouping = None
    generated_group_ids = False
    # 逐通道有效质量：min_counts 路径叠加"尾组 QUALITY=2"（HEASoft loadMin），
    # 其余路径沿用输入 quality；随后按组折叠（rebinChannels 规则）。
    per_ch_quality = None
    if min_counts is not None:
        # 基于最小计数阈值的贪心聚合
        grouping = compute_grouping_by_min_counts(cnt, min_counts)
        tail_q = min_counts_group_quality(cnt, min_counts)
        if pha.quality is not None:
            per_ch_quality = np.maximum(np.asarray(pha.quality, dtype=int).ravel(), tail_q)
        else:
            per_ch_quality = tail_q
    elif factor is not None and factor > 1:
        # 固定因子聚合
        n = ch.size
        grouping = np.zeros(n, dtype=int)
        gid = 1
        for i in range(n):
            grouping[i] = gid
            if (i + 1) % int(factor) == 0 and i < n - 1:
                gid += 1
        generated_group_ids = True
    elif getattr(pha, 'grouping', None) is not None:
        # 使用已有的 grouping 数组
        grouping = np.asarray(pha.grouping, dtype=int)
    else:
        # 默认：不聚合（factor=1）
        return pha

    if per_ch_quality is None and pha.quality is not None:
        per_ch_quality = np.asarray(pha.quality, dtype=int).ravel()

    # grouping 兼容：支持 OGIP 标志位(1/-1/0) 与历史组号编码(1,2,3...)
    g_arr = np.asarray(grouping, dtype=int)
    nz = g_arr[g_arr != 0]
    if not generated_group_ids and nz.size > 0 and np.all(np.isin(nz, [-1, 1])):
        gid_arr = np.zeros_like(g_arr)
        gid = 0
        for i, val in enumerate(g_arr):
            if val == 0:
                gid_arr[i] = 0
            elif val == 1:
                gid += 1
                gid_arr[i] = gid
            else:  # -1
                gid_arr[i] = gid if gid > 0 else 0
    else:
        gid_arr = np.where(g_arr > 0, g_arr, 0)

    # 按 grouping 数组聚合
    gids = np.unique(gid_arr[gid_arr > 0])
    if gids.size == 0:
        return pha

    new_ch = []
    new_counts = []
    new_err = []
    for gid in gids:
        mask = gid_arr == int(gid)
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
    # 组质量折叠：组首质量起步、其后最后一个非零成员覆盖（HEASoft
    # pha::rebinChannels 的 "any bad quality in a bin makes the bin bad" 规则）
    new_quality = fold_group_quality(per_ch_quality, gid_arr, gids)

    # 聚合 EBOUNDS（若存在）
    new_ebounds = None
    if pha.ebounds is not None:
        ch_all, e_lo, e_hi = pha.ebounds
        eb_ch = []
        eb_lo = []
        eb_hi = []
        for gid in gids:
            mask = gid_arr == int(gid)
            idxs = np.where(mask)[0]
            eb_ch.append(int(ch[idxs][0]))
            eb_lo.append(float(np.min(e_lo[idxs])))
            eb_hi.append(float(np.max(e_hi[idxs])))
        new_ebounds = (np.asarray(eb_ch, dtype=int), np.asarray(eb_lo, dtype=float), np.asarray(eb_hi, dtype=float))

    pha_cls = type(pha)
    return pha_cls(
        path=pha.path,
        channels=new_ch,
        counts=new_counts,
        stat_err=new_err if new_err is not None and new_err.size > 0 else None,
        exposure=pha.exposure,
        backscal=pha.backscal,
        areascal=pha.areascal,
        quality=new_quality,
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
    
    evt_cls = type(evt)
    return evt_cls(
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
    from .data import LightcurveData as _LightcurveData

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
                cache_path = xselect_mdb.default_cache_path()
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

    return _LightcurveData(
        path=evt.path,
        time=centers, value=hist.astype(float), error=err, dt=binsize,
        exposure=float(np.sum(bin_exposure)), is_rate=False,
        header=evt.header, meta=evt.meta, headers_dump=evt.headers_dump,
        region=None, bin_exposure=bin_exposure,
        bin_lo=edges[:-1], bin_hi=edges[1:],
        bin_width=np.diff(edges),
        binning='uniform',
        # 透传绝对时间零点：evt.time 已重定基，绝对时刻保存在 timezero
        # （absolute_time = time + timezero），下游据此还原绝对 MET
        timezero=getattr(evt, 'timezero', 0.0) or 0.0,
        timezero_obj=getattr(evt, 'timezero_obj', None),
    )


# ==================== Bayesian Blocks Binning ====================

def bayesian_blocks_exposure(
    counts: np.ndarray | Sequence[float],
    exposure: np.ndarray | Sequence[float],
    *,
    p0: float = 0.05,
    gamma: Optional[float] = None,
    ncp_prior: Optional[float] = None,
) -> np.ndarray:
    """曝光加权的分箱 Bayesian Blocks（Scargle 2013）。

    移植自 HEASoft 6.37 ``burstcube/lib/bayesian_blocks.py``（其基于 astropy 版本修改）。
    相对 ``astropy.stats.bayesian_blocks`` 的差别（原实现文件头注明）：

    1. 修复了 astropy#14017；
    2. 支持 0 计数箱（Scargle 约定至少 1 计数，此处放宽）；
    3. 适应函数用逐箱曝光量 ``T_k`` 而非等宽时间——对 GTI 间隙/帧效应导致的
       逐箱曝光变化（如 EP/WXT）是正确的处理；
    4. 返回变点处的 **箱索引**：第 ``i`` 个块覆盖 ``bins[idx[i]:idx[i+1]]``。

    参数
    ----
    counts : 逐箱计数（允许 0）。
    exposure : 逐箱有效曝光（秒），与 ``counts`` 同形，应 > 0。
    p0 : 假阳性率；与 ``gamma``/``ncp_prior`` 三选一（后者优先）。

    返回：变点箱索引数组（int），首尾分别为 0 与 ``n``。
    """
    counts = np.asarray(counts, dtype=float)
    exposure = np.asarray(exposure, dtype=float)
    if counts.shape != exposure.shape:
        raise ValueError("counts 与 exposure 必须同形")
    n = counts.size
    if n == 0:
        return np.asarray([0], dtype=int)

    # 先验：块数的惩罚项。Scargle (2013) Eq. 21（注意原文缺 log）
    if ncp_prior is None:
        if gamma is not None:
            ncp_prior = -float(np.log(gamma))
        elif p0 is not None:
            ncp_prior = 4.0 - float(np.log(73.53 * p0 * (n ** -0.478)))
        else:
            raise ValueError("p0/gamma/ncp_prior 至少需提供一个")
    ncp_prior = float(ncp_prior)

    # 反向累加便于 O(1) 取任意后缀和；尾部补 0 使 R 取到末尾时后缀和为全段。
    exposure_cumsum = np.append(np.cumsum(exposure[::-1])[::-1], 0.0)
    counts_cumsum = np.append(np.cumsum(counts[::-1])[::-1], 0.0)

    best = np.zeros(n, dtype=float)   # best[R]: 前 R+1 箱的最优总适应度
    last = np.zeros(n, dtype=int)     # last[R]: 最优划分下最后一块的起始箱

    for R in range(n):
        # 适应函数：Scargle 2013 Eq. 19，用曝光 T_k 代替等宽时间。
        T_k = exposure_cumsum[: R + 1] - exposure_cumsum[R + 1]
        N_k = counts_cumsum[: R + 1] - counts_cumsum[R + 1]

        # N_k=0 时 N·ln(N/T) 的极限为 0（避免 nan）；T_k=0 亦跳过。
        fit_vec_log = np.zeros(N_k.size)
        with np.errstate(divide='ignore', invalid='ignore'):
            np.log(N_k / T_k, out=fit_vec_log, where=(N_k != 0) & (T_k > 0))
        fit_vec = N_k * fit_vec_log

        A_R = fit_vec - ncp_prior
        A_R[1:] += best[:R]

        i_max = int(np.argmax(A_R))
        last[R] = i_max
        best[R] = A_R[i_max]

    # 从末尾逐块剥离恢复变点序列（同原实现）。
    change_points = np.zeros(n, dtype=int)
    i_cp = n
    ind = n
    while i_cp > 0:
        i_cp -= 1
        change_points[i_cp] = ind
        if ind == 0:
            break
        ind = last[ind - 1]
    if i_cp == 0:
        change_points[i_cp] = 0
    return change_points[i_cp:]


class BayesianBlocksBinner:
    """基于贝叶斯块的自适应分 bin。

    用法
    ----
        - 传入 `LightcurveData`（counts 或 rate），采用 `astropy.stats.bayesian_blocks`
            计算时间边界；输出块边界由 Bayesian Blocks 直接决定，不再做 SNR 阈值后合并。
        - `use_exposure=True` 时改用曝光加权的分箱贝叶斯块（`bayesian_blocks_exposure`，
            移植自 HEASoft burstcube），适应函数用逐箱曝光而非等宽时间，适合逐箱曝光变化显著的仪器（如 EP/WXT）。

    参数
    ----
    - p0: False positive rate (Scargle 2013)，控制块数量敏感度
    - fitness: Bayesian Blocks 统计模型，可选：
      * 'events': 泊松事件（光子计数等）
      * 'regular_events': 规则采样的事件数据
      * 'measures': 带误差的测量值（高斯统计，适用于已分bin的光变）
    - use_exposure: 启用逐箱曝光加权变点检测（此时忽略 fitness，仅作用于 ``fit``）

    返回
    ----
    - `LightcurveData` 新实例（counts 或 rate 与输入一致），时间为块中心，
      值为每块的聚合值，误差按平方和开方传播；每块的 `bin_exposure` 为实际覆盖曝光。
    """

    def __init__(
        self,
        p0: float = 0.05,
        fitness: Literal['events', 'regular_events', 'measures'] = 'measures',
        use_exposure: bool = False,
        **kwargs,
    ) -> None:
        if 'min_snr' in kwargs:
            kwargs.pop('min_snr')
            warnings.warn(
                "BayesianBlocksBinner: 参数 min_snr 已移除并忽略。",
                RuntimeWarning,
                stacklevel=2,
            )
        if len(kwargs) > 0:
            unknown = ", ".join(sorted(str(k) for k in kwargs.keys()))
            raise TypeError(f"BayesianBlocksBinner() got unexpected keyword argument(s): {unknown}")
        self.p0 = float(p0)
        self.fitness: Literal['events', 'regular_events', 'measures'] = fitness
        self.use_exposure = bool(use_exposure)
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
        lc_cls = type(lc)

        try:
            from astropy.stats import bayesian_blocks
        except Exception:
            raise RuntimeError("需要 astropy.stats.bayesian_blocks 支持，请安装 astropy>=4.0")

        if lc.value.ndim > 1:
            raise NotImplementedError("暂不支持多能段 LC 的贝叶斯块分 bin；请先按能段切片")

        t = np.asarray(lc.time, dtype=float)
        if t.size == 0:
            empty_cols = _ensure_lc_columns(getattr(lc, 'columns', ()), is_rate=lc.is_rate)
            return lc_cls(
                path=lc.path,
                time=np.array([], dtype=float),
                value=np.array([], dtype=float),
                error=None,
                dt=lc.dt,
                timezero=getattr(lc, 'timezero', 0.0),
                timezero_obj=getattr(lc, 'timezero_obj', None),
                tstart=getattr(lc, 'tstart', None),
                tseg=getattr(lc, 'tseg', None),
                exposure=lc.exposure,
                bin_exposure=None,
                is_rate=lc.is_rate,
                counts=(np.array([], dtype=float) if not lc.is_rate else None),
                rate=(np.array([], dtype=float) if lc.is_rate else None),
                counts_err=None,
                rate_err=None,
                err_dist=getattr(lc, 'err_dist', None),
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
                columns=empty_cols,
                ratio=getattr(lc, 'ratio', None),
            )

        # 将输入统一到 counts 及其方差，便于块内聚合与 SNR 计算
        vals = np.asarray(lc.value, dtype=float)
        errs = np.asarray(lc.error, dtype=float) if lc.error is not None else None

        # 推断原始 bin 边界与每 bin 宽度（支持不等宽）
        left, right, width = _infer_bin_geometry(lc)
        orig_dt = float(np.median(width)) if width.size > 0 else 1.0
        eff_width = _effective_exposure_from_lc(lc, width)

        if lc.is_rate:
            counts = vals * eff_width
            err_counts = (errs * eff_width) if errs is not None else None
        else:
            counts = vals.copy()
            err_counts = errs.copy() if errs is not None else None

        if err_counts is None:
            err_counts = np.sqrt(np.maximum(counts, 0.0))

        var_counts = err_counts ** 2

        # 调用 bayesian_blocks 构建初始块边界
        # fitness 类型：'events'(泊松事件), 'regular_events'(规则采样事件), 'measures'(带误差测量)
        if self.use_exposure:
            # 曝光加权变点检测（burstcube 移植版）：逐箱曝光优先用 bin_exposure，
            # 否则退化为有效箱宽。变点索引→时间边界：块 i 覆盖 bins[cp[i]:cp[i+1]]。
            _orig_expo = getattr(lc, 'bin_exposure', None)
            expo_bins = np.asarray(_orig_expo if _orig_expo is not None else eff_width, dtype=float)
            cp = bayesian_blocks_exposure(counts, expo_bins, p0=self.p0)
            edges = np.concatenate(([left[int(cp[0])]], right[cp[1:] - 1]))
        elif self.fitness == 'regular_events':
            edges = bayesian_blocks(t, counts, fitness=self.fitness, p0=self.p0, dt=orig_dt)
        elif self.fitness == 'measures':
            edges = bayesian_blocks(t, counts, fitness=self.fitness, p0=self.p0, sigma=err_counts)
        else:
            edges = bayesian_blocks(t, counts, fitness=self.fitness, p0=self.p0)
        # heapy/ppsignal 的做法会扩展首末边界到原始范围；保持与之兼容。
        # use_exposure 路径的边界已覆盖全段，无需再拼接。
        full_left = float(left.min())
        full_right = float(right.max())
        if not self.use_exposure:
            if edges.size > 0:
                edges = np.concatenate(([full_left], edges, [full_right]))
            else:
                edges = np.asarray([full_left, full_right], dtype=float)
        self.last_edges = edges.copy()

        # 根据块边界，将原始 bins 分配到各块并聚合（不做 SNR 阈值后合并）
        nb = edges.size - 1
        block_slices = []
        for i in range(nb):
            a = edges[i]
            b = edges[i + 1]
            centers = (left + right) / 2
            mask = (centers >= a) & (centers < b)
            block_slices.append(np.where(mask)[0])
        merged_indices = [np.unique(ix) for ix in block_slices if ix.size > 0]

        # 保存以便 Txx 使用
        self.last_merged_indices = [np.asarray(ix, dtype=int) for ix in merged_indices]
        # 生成输出 LC：每个合并后的块 -> 一个点
        out_time = []
        out_val = []
        out_err = []
        out_expo = []
        out_dt = []
        out_left = []
        out_right = []

        # 计算每块的实际曝光：使用原始 bin_exposure 如有，否则用时间覆盖长度
        orig_expo = getattr(lc, 'bin_exposure', None)
        for idxs in merged_indices:
            a = float(np.min(left[idxs]))
            b = float(np.max(right[idxs]))
            # 聚合 counts/var
            csum = float(np.sum(counts[idxs]))
            vsum = float(np.sum(var_counts[idxs]))
            expo_blk = float(np.sum(orig_expo[idxs])) if orig_expo is not None else (b - a)
            denom = expo_blk if expo_blk > 0.0 else (b - a)
            # 值/误差空间：保留与输入一致（counts 或 rate）
            if lc.is_rate:
                val = csum / denom
                err = (np.sqrt(vsum) / denom) if vsum > 0 else 0.0
            else:
                val = csum
                err = (np.sqrt(vsum)) if vsum > 0 else 0.0
            out_time.append(0.5 * (a + b))
            out_val.append(val)
            out_err.append(err)
            out_dt.append(b - a)
            out_left.append(a)
            out_right.append(b)
            out_expo.append(expo_blk)

        out_time = np.asarray(out_time, dtype=float)
        out_val = np.asarray(out_val, dtype=float)
        out_err = np.asarray(out_err, dtype=float)
        if len(out_dt) > 0:
            out_dt = float(np.median(out_dt))
        else:
            _, _, w0 = _infer_bin_geometry(lc)
            out_dt = float(np.median(w0)) if w0.size > 0 else 0.0
        out_left_arr = np.asarray(out_left, dtype=float)
        out_right_arr = np.asarray(out_right, dtype=float)
        bin_exposure = np.asarray(out_expo, dtype=float)
        out_cols = _ensure_lc_columns(getattr(lc, 'columns', ()), is_rate=lc.is_rate)

        return lc_cls(
            path=lc.path,
            time=out_time,
            value=out_val,
            error=out_err,
            dt=out_dt,
            timezero=getattr(lc, 'timezero', 0.0),
            timezero_obj=getattr(lc, 'timezero_obj', None),
            bin_lo=out_left_arr,
            bin_hi=out_right_arr,
            bin_width=(out_right_arr - out_left_arr),
            binning=_infer_binning_kind(out_right_arr - out_left_arr),
            tstart=getattr(lc, 'tstart', None),
            tseg=getattr(lc, 'tseg', None),
            exposure=float(np.sum(bin_exposure)),
            bin_exposure=bin_exposure,
            is_rate=lc.is_rate,
            counts=None if lc.is_rate else out_val,
            rate=out_val if lc.is_rate else None,
            counts_err=None if lc.is_rate else out_err,
            rate_err=out_err if lc.is_rate else None,
            err_dist=getattr(lc, 'err_dist', None),
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
            columns=out_cols,
            ratio=getattr(lc, 'ratio', None),
        )

    def fit_src_bkg(self, lc_src: 'LightcurveData', lc_bkg: 'LightcurveData', alpha: Optional[float] = None) -> 'LightcurveData':
        """对源与背景光变同时进行贝叶斯块分 bin，并按 Li&Ma 显著性过滤。

        - `alpha`: 源/背景缩放因子（如面积或BACKSCAL比值）。
          若未显式传入，会尝试从输入数据元信息（ratio/backscal/region.area）推断。
          若仍无法确定，则报错。
        - 初始块边界（fitness='measures'）基于净计数 N = S - alpha*B。
        - 净计数误差传播：sigma_N^2 = sigma_S^2 + alpha^2 * sigma_B^2。
        - 块合并阈值使用带符号 Li&Ma 显著性。
        """
        
        alpha_val = _resolve_alpha_for_src_bkg(
            alpha,
            lc_src,
            lc_bkg,
            context="fit_src_bkg",
        )

        lc_cls = type(lc_src)

        # 统一到 counts 域
        def lc_to_counts(lc: 'LightcurveData'):
            t = np.asarray(lc.time, dtype=float)
            vals = np.asarray(lc.value, dtype=float)
            errs = np.asarray(lc.error, dtype=float) if lc.error is not None else None
            left, right, width = _infer_bin_geometry(lc)
            eff_width = _effective_exposure_from_lc(lc, width)
            if lc.is_rate:
                c = vals * eff_width
                e = (errs * eff_width) if errs is not None else None
            else:
                c = vals.copy()
                e = errs.copy() if errs is not None else None
            if e is None:
                e = np.sqrt(np.maximum(c, 0.0))
            return t, left, right, c, e, np.asarray(eff_width, dtype=float)

        def _project_by_overlap(
            src_left: np.ndarray,
            src_right: np.ndarray,
            src_values: np.ndarray,
            tgt_left: np.ndarray,
            tgt_right: np.ndarray,
            *,
            power: float = 1.0,
        ) -> np.ndarray:
            """Project source-bin quantities onto target bins by overlap fractions.

            power=1 uses linear scaling (counts/exposure), power=2 for variance split.
            """
            out = np.zeros(tgt_left.size, dtype=float)
            if tgt_left.size == 0 or src_left.size == 0:
                return out
            for i in range(tgt_left.size):
                a = float(tgt_left[i])
                b = float(tgt_right[i])
                acc = 0.0
                for j in range(src_left.size):
                    l = float(src_left[j])
                    r = float(src_right[j])
                    if r <= a or l >= b:
                        continue
                    width = max(r - l, 1e-12)
                    ov = max(0.0, min(b, r) - max(a, l))
                    if ov <= 0.0:
                        continue
                    frac = ov / width
                    acc += float(src_values[j]) * (frac ** power)
                out[i] = acc
            return out

        t_s, l_s, r_s, c_s, e_s, expo_s = lc_to_counts(lc_src)
        t_b, l_b, r_b, c_b, e_b, expo_b = lc_to_counts(lc_bkg)
        if t_s.size == 0:
            empty_cols = _ensure_lc_columns(getattr(lc_src, 'columns', ()), is_rate=lc_src.is_rate)
            return lc_cls(
                path=lc_src.path,
                time=np.array([], dtype=float),
                value=np.array([], dtype=float),
                error=None,
                dt=lc_src.dt,
                timezero=getattr(lc_src, 'timezero', 0.0),
                timezero_obj=getattr(lc_src, 'timezero_obj', None),
                tstart=getattr(lc_src, 'tstart', None),
                tseg=getattr(lc_src, 'tseg', None),
                exposure=lc_src.exposure,
                bin_exposure=None,
                is_rate=lc_src.is_rate,
                counts=(np.array([], dtype=float) if not lc_src.is_rate else None),
                rate=(np.array([], dtype=float) if lc_src.is_rate else None),
                counts_err=None,
                rate_err=None,
                err_dist=getattr(lc_src, 'err_dist', None),
                gti_start=getattr(lc_src, 'gti_start', None),
                gti_stop=getattr(lc_src, 'gti_stop', None),
                quality=getattr(lc_src, 'quality', None),
                fracexp=getattr(lc_src, 'fracexp', None),
                backscal=getattr(lc_src, 'backscal', None),
                areascal=getattr(lc_src, 'areascal', None),
                telescop=getattr(lc_src, 'telescop', None),
                timesys=getattr(lc_src, 'timesys', None),
                mjdref=getattr(lc_src, 'mjdref', None),
                header=lc_src.header,
                meta=lc_src.meta,
                headers_dump=lc_src.headers_dump,
                region=lc_src.region,
                columns=empty_cols,
                ratio=getattr(lc_src, 'ratio', None),
            )

        # 将背景投影到源时间网格：用于净计数初始边界与 alpha_i 估计
        b_on_s = _project_by_overlap(l_b, r_b, c_b, l_s, r_s, power=1.0)
        var_b_on_s = _project_by_overlap(l_b, r_b, e_b ** 2, l_s, r_s, power=2.0)
        expo_b_on_s = _project_by_overlap(l_b, r_b, expo_b, l_s, r_s, power=1.0)

        base_alpha = alpha_val
        with np.errstate(divide='ignore', invalid='ignore'):
            expo_ratio_i = np.where(expo_b_on_s > 0.0, expo_s / expo_b_on_s, 1.0)
        alpha_i = base_alpha * expo_ratio_i
        alpha_i = np.where(np.isfinite(alpha_i) & (alpha_i > 0.0), alpha_i, 1.0)

        net_for_edges = c_s - alpha_i * b_on_s
        sigma_net = np.sqrt(np.maximum((e_s ** 2) + (alpha_i ** 2) * var_b_on_s, 0.0))
        sigma_net = np.where(np.isfinite(sigma_net) & (sigma_net > 0.0), sigma_net, 1e-12)

        # 初始块边界：measures 用净计数，其余 fitness 保持源计数语义
        est_dt = float(np.median(r_s - l_s)) if r_s.size else 1.0
        if self.fitness == 'regular_events':
            edges = bayesian_blocks(t_s, c_s, fitness=self.fitness, p0=self.p0, dt=est_dt)
        elif self.fitness == 'measures':
            edges = bayesian_blocks(t_s, net_for_edges, fitness=self.fitness, p0=self.p0, sigma=sigma_net)
        else:
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
            src_slices.append(np.where(((l_s + r_s) / 2 >= a) & ((l_s + r_s) / 2 < b))[0])
            bkg_slices.append(np.where(((l_b + r_b) / 2 >= a) & ((l_b + r_b) / 2 < b))[0])

        merged = []
        for i in range(nb):
            src_unique = np.unique(src_slices[i])
            if src_unique.size == 0:
                continue
            bkg_unique = np.unique(bkg_slices[i])
            merged.append((src_unique, bkg_unique))

        # 输出 LC（以净计数或净率表示；误差按 var 的 sqrt）
        self.last_merged_indices = []
        out_time = []
        out_val = []
        out_err = []
        out_expo = []
        out_dt_list = []
        out_left = []
        out_right = []
        for src_idx, bkg_idx in merged:
            if src_idx.size == 0:
                continue
            a = float(np.min(l_s[src_idx]))
            b = float(np.max(r_s[src_idx]))
            S = float(np.sum(c_s[src_idx]))
            B = float(np.sum(c_b[bkg_idx]))
            expo_src_blk = float(np.sum(expo_s[src_idx])) if src_idx.size > 0 else 0.0
            expo_bkg_blk = float(np.sum(expo_b[bkg_idx])) if bkg_idx.size > 0 else 0.0
            if expo_bkg_blk > 0:
                expo_ratio_blk = expo_src_blk / expo_bkg_blk
            else:
                expo_ratio_blk = 1.0
            alpha_blk = alpha_val * expo_ratio_blk
            net = S - alpha_blk * B
            var_s = float(np.sum((e_s[src_idx]) ** 2)) if src_idx.size > 0 else 0.0
            var_b = float(np.sum((e_b[bkg_idx]) ** 2)) if bkg_idx.size > 0 else 0.0
            var = var_s + (alpha_blk ** 2) * var_b
            expo_blk = expo_src_blk if expo_src_blk > 0.0 else (b - a)
            denom = expo_blk if expo_blk > 0.0 else (b - a)
            # 输出空间：保持输入源 LC 的 is_rate 习惯
            if lc_src.is_rate:
                val = net / denom
                err = (np.sqrt(var) / denom) if var > 0 else 0.0
            else:
                val = net
                err = (np.sqrt(var)) if var > 0 else 0.0
            out_time.append(0.5 * (a + b))
            out_val.append(val)
            out_err.append(err)
            out_dt_list.append(b - a)
            out_left.append(a)
            out_right.append(b)
            out_expo.append(expo_blk)
            # 保存索引用于 Txx 接口
            self.last_merged_indices.append(np.asarray(src_idx, dtype=int))

        if len(out_time) == 0:
            empty_cols = _ensure_lc_columns(getattr(lc_src, 'columns', ()), is_rate=lc_src.is_rate)
            return lc_cls(
                path=lc_src.path,
                time=np.array([], dtype=float),
                value=np.array([], dtype=float),
                error=None,
                dt=lc_src.dt,
                timezero=getattr(lc_src, 'timezero', 0.0),
                timezero_obj=getattr(lc_src, 'timezero_obj', None),
                tstart=getattr(lc_src, 'tstart', None),
                tseg=getattr(lc_src, 'tseg', None),
                exposure=lc_src.exposure,
                bin_exposure=None,
                is_rate=lc_src.is_rate,
                counts=(np.array([], dtype=float) if not lc_src.is_rate else None),
                rate=(np.array([], dtype=float) if lc_src.is_rate else None),
                counts_err=None,
                rate_err=None,
                err_dist=getattr(lc_src, 'err_dist', None),
                gti_start=getattr(lc_src, 'gti_start', None),
                gti_stop=getattr(lc_src, 'gti_stop', None),
                quality=getattr(lc_src, 'quality', None),
                fracexp=getattr(lc_src, 'fracexp', None),
                backscal=getattr(lc_src, 'backscal', None),
                areascal=getattr(lc_src, 'areascal', None),
                telescop=getattr(lc_src, 'telescop', None),
                timesys=getattr(lc_src, 'timesys', None),
                mjdref=getattr(lc_src, 'mjdref', None),
                header=lc_src.header,
                meta=lc_src.meta,
                headers_dump=lc_src.headers_dump,
                region=lc_src.region,
                columns=empty_cols,
                ratio=getattr(lc_src, 'ratio', None),
            )

        out_time = np.asarray(out_time, dtype=float)
        out_val = np.asarray(out_val, dtype=float)
        out_err = np.asarray(out_err, dtype=float)
        if len(out_dt_list) > 0:
            out_dt = float(np.median(out_dt_list))
        else:
            _, _, w0 = _infer_bin_geometry(lc_src)
            out_dt = float(np.median(w0)) if w0.size > 0 else 0.0
        out_left_arr = np.asarray(out_left, dtype=float)
        out_right_arr = np.asarray(out_right, dtype=float)
        bin_exposure = np.asarray(out_expo, dtype=float)
        out_cols = _ensure_lc_columns(getattr(lc_src, 'columns', ()), is_rate=lc_src.is_rate)

        return lc_cls(
            path=lc_src.path,
            time=out_time,
            value=out_val,
            error=out_err,
            dt=out_dt,
            timezero=getattr(lc_src, 'timezero', 0.0),
            timezero_obj=getattr(lc_src, 'timezero_obj', None),
            bin_lo=out_left_arr,
            bin_hi=out_right_arr,
            bin_width=(out_right_arr - out_left_arr),
            binning=_infer_binning_kind(out_right_arr - out_left_arr),
            tstart=getattr(lc_src, 'tstart', None),
            tseg=getattr(lc_src, 'tseg', None),
            exposure=float(np.sum(bin_exposure)),
            bin_exposure=bin_exposure,
            is_rate=lc_src.is_rate,
            counts=None if lc_src.is_rate else out_val,
            rate=out_val if lc_src.is_rate else None,
            counts_err=None if lc_src.is_rate else out_err,
            rate_err=out_err if lc_src.is_rate else None,
            err_dist=getattr(lc_src, 'err_dist', None),
            gti_start=getattr(lc_src, 'gti_start', None),
            gti_stop=getattr(lc_src, 'gti_stop', None),
            quality=getattr(lc_src, 'quality', None),
            fracexp=getattr(lc_src, 'fracexp', None),
            backscal=getattr(lc_src, 'backscal', None),
            areascal=getattr(lc_src, 'areascal', None),
            telescop=getattr(lc_src, 'telescop', None),
            timesys=getattr(lc_src, 'timesys', None),
            mjdref=getattr(lc_src, 'mjdref', None),
            header=lc_src.header,
            meta=lc_src.meta,
            headers_dump=lc_src.headers_dump,
            region=lc_src.region,
            columns=out_cols,
            ratio=getattr(lc_src, 'ratio', None),
        )


def _alpha_scalar_or_none(v: object) -> Optional[float]:
    """将任意标量/数组输入规整为正有限标量；不可用则返回 None。"""
    if v is None:
        return None
    try:
        arr = np.asarray(v, dtype=float).reshape(-1)
    except Exception:
        return None
    if arr.size == 0:
        return None
    val = float(np.nanmedian(arr))
    if (not np.isfinite(val)) or val <= 0.0:
        return None
    return val


def _resolve_alpha_for_src_bkg(
    alpha: Optional[float],
    lc_src: 'LightcurveData',
    lc_bkg: 'LightcurveData',
    *,
    context: str,
    dataset_area_ratio: Optional[float] = None,
) -> float:
    """解析 src/bkg 的 alpha。优先显式参数，其次数据元信息，最后报错。"""
    if alpha is not None:
        alpha_val = _alpha_scalar_or_none(alpha)
        if alpha_val is None:
            raise ValueError(f"{context}: alpha 必须为正且有限，当前={alpha}")
        return alpha_val

    area_ratio = _alpha_scalar_or_none(dataset_area_ratio)
    if area_ratio is not None:
        return area_ratio

    ratio_attr = _alpha_scalar_or_none(getattr(lc_src, 'ratio', None))
    if ratio_attr is not None:
        return ratio_attr

    bs_src = _alpha_scalar_or_none(getattr(lc_src, 'backscal', None))
    bs_bkg = _alpha_scalar_or_none(getattr(lc_bkg, 'backscal', None))
    if bs_src is None:
        bs_src = _alpha_scalar_or_none(
            getattr(lc_src, 'get_keyword_ci', lambda *_args, **_kwargs: None)(
                'BACKSCAL', None
            )
        )
    if bs_bkg is None:
        bs_bkg = _alpha_scalar_or_none(
            getattr(lc_bkg, 'get_keyword_ci', lambda *_args, **_kwargs: None)(
                'BACKSCAL', None
            )
        )
    if bs_src is not None and bs_bkg is not None and bs_bkg > 0.0:
        return float(bs_src / bs_bkg)

    a_src = _alpha_scalar_or_none(getattr(getattr(lc_src, 'region', None), 'area', None))
    a_bkg = _alpha_scalar_or_none(getattr(getattr(lc_bkg, 'region', None), 'area', None))
    if a_src is not None and a_bkg is not None and a_bkg > 0.0:
        return float(a_src / a_bkg)

    raise ValueError(
        f"{context}: 无法从数据元信息推断 alpha。"
        "请显式传入 alpha，或提供 ratio/backscal/region.area。"
    )


def bin_bblocks(
    lc,
    background: Optional['LightcurveData'] = None,
    *,
    alpha: Optional[float] = None,
    p0: float = 0.05,
    **kwargs,
) -> 'LightcurveData':
    """贝叶斯块自适应分 bin，支持单独光变或源+背景联合处理。

    参数
    ----
    lc : LightcurveData | LightcurveDataset
        - 若为 `LightcurveData`：视为源光变；需配合 `background` 参数传入背景（可选）。
        - 若为 `LightcurveDataset`：自动提取 `.data` 作为源，`.background.data` 作为背景。
    background : LightcurveData, optional
        背景光变数据（仅当 `lc` 为 `LightcurveData` 时需要）。
        若 `lc` 为 `LightcurveDataset` 且已有 `.background`，此参数被忽略。
    alpha : float, optional
        源/背景缩放因子（面积或 BACKSCAL 比值）。
        若未显式传入，会依次尝试从 `dataset.area_ratio`、`ratio`、`BACKSCAL`
        与 `region.area` 推断；若仍失败则报错。
    p0 : float, default=0.05
        Bayesian Blocks 的假阳性率（控制分块敏感度）。

    返回
    ----
    LightcurveData
        分 bin 后的光变（若有背景则为净光变）。

    示例
    ----
    >>> # 1. 单独源光变（无背景）
    >>> lc_binned = bin_lightcurve_bblocks(lc_src, p0=0.05)
    >>>
    >>> # 2. 源+背景（直接传 LightcurveData）
    >>> lc_net = bin_lightcurve_bblocks(lc_src, background=lc_bkg, alpha=1.2, p0=0.05)
    >>>
    >>> # 3. 传入 LightcurveDataset（可从 area_ratio 自动获取）
    >>> ds = netdata(lc_src, lc_bkg, area_ratio=1.2)
    >>> lc_net = bin_lightcurve_bblocks(ds, p0=0.05)
    """
    if 'min_snr' in kwargs:
        kwargs.pop('min_snr')
        warnings.warn(
            "bin_bblocks: 参数 min_snr 已移除并忽略。",
            RuntimeWarning,
            stacklevel=2,
        )
    if len(kwargs) > 0:
        unknown = ", ".join(sorted(str(k) for k in kwargs.keys()))
        raise TypeError(f"bin_bblocks() got unexpected keyword argument(s): {unknown}")

    # 判断输入类型并提取源/背景/alpha
    try:
        # 尝试作为 LightcurveDataset（检查是否有 .data 属性和 LightcurveData 类型）
        if hasattr(lc, 'data') and isinstance(getattr(lc, 'data', None), LightcurveData):
            # LightcurveDataset 输入
            lc_src = lc.data
            lc_bkg = getattr(lc.background, 'data', None) if getattr(lc, 'background', None) is not None else None
            dataset_area_ratio = getattr(lc, 'area_ratio', None)
        else:
            # LightcurveData 输入
            lc_src = lc
            lc_bkg = background
            dataset_area_ratio = None
    except Exception:
        # 回退：当作 LightcurveData
        lc_src = lc
        lc_bkg = background
        dataset_area_ratio = None

    def _select_fitness(lc_in: 'LightcurveData') -> Literal['events', 'regular_events', 'measures']:
        """根据输入光变选择 bayesian_blocks 的 fitness。

        - `regular_events` 仅适用于每个时间 tick 取值为 0/1 的事件序列。
        - 常规已分 bin 光变（counts>1、或 rate）使用 `measures` 更稳健。
        """
        try:
            vals = np.asarray(lc_in.value, dtype=float)
        except Exception:
            return 'measures'

        if vals.size == 0:
            return 'measures'

        if lc_in.is_rate:
            _, _, width = _infer_bin_geometry(lc_in)
            width = _effective_exposure_from_lc(lc_in, width)
            counts_like = vals * width
        else:
            counts_like = vals

        finite = counts_like[np.isfinite(counts_like)]
        if finite.size == 0:
            return 'measures'

        int_like = np.all(np.isclose(finite, np.round(finite)))
        binary_like = np.all((np.round(finite) == 0) | (np.round(finite) == 1))
        non_negative = np.all(finite >= 0)

        if int_like and binary_like and non_negative:
            return 'regular_events'
        return 'measures'

    fitness: Literal['events', 'regular_events', 'measures'] = _select_fitness(lc_src)

    # 无背景：单独源光变分 bin
    if lc_bkg is None:
        return BayesianBlocksBinner(p0=p0, fitness=fitness).fit(lc_src)

    # 有背景：源+背景联合分 bin（支持从数据元信息推断 alpha）
    alpha_val = _resolve_alpha_for_src_bkg(
        alpha,
        lc_src,
        lc_bkg,
        context="bin_bblocks",
        dataset_area_ratio=dataset_area_ratio,
    )
    return BayesianBlocksBinner(p0=p0, fitness=fitness).fit_src_bkg(lc_src, lc_bkg, alpha=alpha_val)



def autobin(
    lc_src: 'LightcurveData',
    background: Optional['LightcurveData'] = None,
    *,
    alpha: Optional[float] = None,
    min_sigma: float = 3.0,
    burst_tstart: Optional[float] = None,
    burst_tstop: Optional[float] = None,
    p0: float = 0.05,
) -> 'LightcurveData':
    """按 Li&Ma 阈值做渐进累积分 bin（面向爆发窗）。

    规则
    ----
    - 在爆发窗内逐个累积原始 bin，直到满足 Li&Ma>=min_sigma 且净计数>0。
    - 满足后输出一个新 bin 并重置累积器。
    - 末尾若未达阈值但净计数>0，则保留并给出 warning。
    """

    lc_cls = type(lc_src)

    def _lc_to_counts(lc: 'LightcurveData'):
        t = np.asarray(lc.time, dtype=float)
        vals = np.asarray(lc.value, dtype=float)
        errs = np.asarray(lc.error, dtype=float) if lc.error is not None else None
        left, right, width = _infer_bin_geometry(lc)
        eff_width = _effective_exposure_from_lc(lc, width)
        if lc.is_rate:
            counts = vals * eff_width
            err_counts = (errs * eff_width) if errs is not None else None
        else:
            counts = vals.copy()
            err_counts = errs.copy() if errs is not None else None
        if err_counts is None:
            err_counts = np.sqrt(np.maximum(counts, 0.0))
        return t, left, right, counts, err_counts, eff_width

    def _project_by_overlap(
        src_left: np.ndarray,
        src_right: np.ndarray,
        src_values: np.ndarray,
        tgt_left: np.ndarray,
        tgt_right: np.ndarray,
        *,
        power: float = 1.0,
    ) -> np.ndarray:
        out = np.zeros(tgt_left.shape[0], dtype=float)
        for i in range(tgt_left.shape[0]):
            a = float(tgt_left[i])
            b = float(tgt_right[i])
            overlap = np.minimum(src_right, b) - np.maximum(src_left, a)
            mask = overlap > 0.0
            if not np.any(mask):
                continue
            width = np.maximum(src_right[mask] - src_left[mask], 1e-12)
            frac = (overlap[mask] / width) ** power
            out[i] = float(np.sum(src_values[mask] * frac))
        return out

    t_s, l_s, r_s, c_s, e_s, expo_s = _lc_to_counts(lc_src)
    if t_s.size == 0:
        raise ValueError("lc_src 为空，无法 autobin")

    if background is None:
        c_b = np.zeros_like(c_s)
        e_b = np.zeros_like(c_s)
        alpha_eff = 0.0
    else:
        if alpha is None:
            raise ValueError("提供 background 时必须显式传入 alpha")
        _, l_b, r_b, c_b_raw, e_b_raw, _ = _lc_to_counts(background)
        c_b = _project_by_overlap(l_b, r_b, c_b_raw, l_s, r_s, power=1.0)
        v_b = _project_by_overlap(l_b, r_b, e_b_raw ** 2, l_s, r_s, power=2.0)
        e_b = np.sqrt(np.maximum(v_b, 0.0))
        alpha_eff = float(alpha)

    if burst_tstart is None or burst_tstop is None:
        binner = BayesianBlocksBinner(p0=p0, fitness='measures')
        if background is None:
            _ = binner.fit(lc_src)
        else:
            _ = binner.fit_src_bkg(lc_src, background, alpha=alpha_eff)
        edges = binner.last_edges
        if edges is not None and edges.size >= 3:
            if burst_tstart is None:
                burst_tstart = float(edges[1])
            if burst_tstop is None:
                burst_tstop = float(edges[-2])

    if burst_tstart is None:
        burst_tstart = float(l_s.min())
    if burst_tstop is None:
        burst_tstop = float(r_s.max())

    in_burst = (l_s < float(burst_tstop)) & (r_s > float(burst_tstart))
    if not np.any(in_burst):
        raise ValueError("爆发区间内无数据，无法 autobin")

    idxs = np.where(in_burst)[0]
    groups: list[np.ndarray] = []
    run: list[int] = []

    for idx in idxs:
        run.append(int(idx))
        run_arr = np.asarray(run, dtype=int)
        src_sum = float(np.sum(c_s[run_arr]))
        bkg_sum = float(np.sum(c_b[run_arr]))
        net_sum = src_sum - alpha_eff * bkg_sum
        if alpha_eff > 0.0:
            sig = li_ma_snr(src_sum, bkg_sum, alpha_eff)
        else:
            sig = float(net_sum / np.sqrt(max(src_sum, 1e-12)))
        if (sig >= float(min_sigma)) and (net_sum > 0.0):
            groups.append(run_arr)
            run = []

    if len(run) > 0:
        run_arr = np.asarray(run, dtype=int)
        src_sum = float(np.sum(c_s[run_arr]))
        bkg_sum = float(np.sum(c_b[run_arr]))
        net_sum = src_sum - alpha_eff * bkg_sum
        if net_sum > 0.0:
            groups.append(run_arr)
            warnings.warn(
                "autobin: 保留了末尾低显著性但净计数为正的 tail bin。",
                RuntimeWarning,
                stacklevel=2,
            )

    if len(groups) == 0:
        raise ValueError("autobin 未生成有效 bin（无满足阈值或正净计数尾段）")

    out_time = []
    out_val = []
    out_err = []
    out_expo = []
    out_left = []
    out_right = []

    for g in groups:
        src_sum = float(np.sum(c_s[g]))
        bkg_sum = float(np.sum(c_b[g]))
        net_sum = src_sum - alpha_eff * bkg_sum
        var_sum = float(np.sum(e_s[g] ** 2 + (alpha_eff ** 2) * (e_b[g] ** 2)))
        left_g = float(np.min(l_s[g]))
        right_g = float(np.max(r_s[g]))
        expo_g = float(np.sum(expo_s[g]))

        out_left.append(left_g)
        out_right.append(right_g)
        out_expo.append(expo_g)
        out_time.append(0.5 * (left_g + right_g))

        if lc_src.is_rate:
            denom = expo_g if expo_g > 0.0 else (right_g - left_g)
            denom = max(denom, 1e-12)
            out_val.append(net_sum / denom)
            out_err.append(np.sqrt(max(var_sum, 0.0)) / denom)
        else:
            out_val.append(net_sum)
            out_err.append(np.sqrt(max(var_sum, 0.0)))

    out_time_arr = np.asarray(out_time, dtype=float)
    out_val_arr = np.asarray(out_val, dtype=float)
    out_err_arr = np.asarray(out_err, dtype=float)
    out_left_arr = np.asarray(out_left, dtype=float)
    out_right_arr = np.asarray(out_right, dtype=float)
    out_expo_arr = np.asarray(out_expo, dtype=float)
    out_width_arr = out_right_arr - out_left_arr

    return lc_cls(
        path=lc_src.path,
        time=out_time_arr,
        value=out_val_arr,
        error=out_err_arr,
        dt=(out_width_arr if out_width_arr.size > 1 else (float(out_width_arr[0]) if out_width_arr.size == 1 else 0.0)),
        timezero=getattr(lc_src, 'timezero', 0.0),
        timezero_obj=getattr(lc_src, 'timezero_obj', None),
        bin_lo=out_left_arr,
        bin_hi=out_right_arr,
        bin_width=out_width_arr,
        binning=_infer_binning_kind(out_width_arr),
        tstart=getattr(lc_src, 'tstart', None),
        tseg=getattr(lc_src, 'tseg', None),
        exposure=float(np.sum(out_expo_arr)),
        bin_exposure=out_expo_arr,
        is_rate=lc_src.is_rate,
        counts=None if lc_src.is_rate else out_val_arr,
        rate=out_val_arr if lc_src.is_rate else None,
        counts_err=None if lc_src.is_rate else out_err_arr,
        rate_err=out_err_arr if lc_src.is_rate else None,
        err_dist=getattr(lc_src, 'err_dist', None),
        gti_start=getattr(lc_src, 'gti_start', None),
        gti_stop=getattr(lc_src, 'gti_stop', None),
        quality=getattr(lc_src, 'quality', None),
        fracexp=getattr(lc_src, 'fracexp', None),
        backscal=getattr(lc_src, 'backscal', None),
        areascal=getattr(lc_src, 'areascal', None),
        telescop=getattr(lc_src, 'telescop', None),
        timesys=getattr(lc_src, 'timesys', None),
        mjdref=getattr(lc_src, 'mjdref', None),
        header=lc_src.header,
        meta=lc_src.meta,
        headers_dump=lc_src.headers_dump,
        region=lc_src.region,
        columns=_ensure_lc_columns(getattr(lc_src, 'columns', ()), is_rate=lc_src.is_rate),
        ratio=getattr(lc_src, 'ratio', None),
    )



# 时标计算已拆分至 .timescale（txx 与 txx_iterbkg 两种方法学）；
# 此处重导出保持 `jinwu.core.ops.txx` 及其内部助手的既有导入路径不变。
from .timescale import (  # noqa: E402,F401
    txx,
    txx_iterbkg,
    _txx54_is_lightcurve_like,
    _txx54_array_to_lc,
    _txx54_to_counts,
    _txx54_project_counts_to_src_grid,
    _txx54_overlap_sum,
    _txx54_contiguous_groups,
    _txx54_cross_target,
    _txx54_asymm_err_from_samples,
    _txx54_robust_sigma,
    _txx54_is_event_file_input,
    _txx54_is_event_data_input,
    _txx54_read_evt_file,
    _txx54_read_event_object,
    _txx54_event_input_to_dict,
    _txx54_positive_scalar,
    _txx54_bin_evt_to_array,
    _txx54_convert_event_inputs,
)

# ==================== HEASoft/FTOOLS process operations ====================
# These helpers are intentionally mission-agnostic. They prepare a safe
# non-interactive HEASoft environment and execute external HEASoft commands.

@_dataclass(frozen=True)
class CommandResult:
    """Result returned by a HEASoft/FTOOLS command invocation."""

    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    cwd: str | None = None
    log_path: str | None = None
    script_path: str | None = None

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def _candidate_headas_dirs() -> list[_Path]:
    candidates: list[_Path] = []
    env_headas = _os.environ.get("HEADAS")
    if env_headas:
        candidates.append(_Path(env_headas))
    conda_prefix = _os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.append(_Path(conda_prefix) / "heasoft")
    # Fallback: check common conda environments for a 'hea' HEASoft install
    for _home_base in (_Path.home(), _Path("/home/xinxiang")):
        _hea_candidate = _home_base / "miniconda3" / "envs" / "hea" / "heasoft"
        if (_hea_candidate / "bin").exists():
            candidates.append(_hea_candidate)
            break
    return candidates


def _resolve_headas(headas: str | _Path | None = None) -> _Path:
    if headas is not None:
        path = _Path(headas).expanduser().resolve()
        if not (path / "bin").exists():
            raise FileNotFoundError(f"HEADAS bin directory not found: {path / 'bin'}")
        return path
    for candidate in _candidate_headas_dirs():
        candidate = candidate.expanduser()
        if (candidate / "bin").exists():
            return candidate.resolve()
    raise FileNotFoundError("Cannot locate HEASoft HEADAS. Set HEADAS or pass headas=...")


def _prepend_env_path(value: str | None, entry: _Path) -> str:
    parts = [] if not value else value.split(_os.pathsep)
    entry_s = str(entry)
    return _os.pathsep.join([entry_s] + [p for p in parts if p != entry_s])


def _safe_headas_dir(path: str | _Path | None, prefix: str) -> _Path:
    base = _Path(_tempfile.gettempdir()) / prefix if path is None else _Path(path).expanduser()
    base.mkdir(parents=True, exist_ok=True)
    return base.resolve()


def ensure_headas_env(
    *,
    headas: str | _Path | None = None,
    env: _Mapping[str, str] | None = None,
    home: str | _Path | None = None,
    pfiles: str | _Path | None = None,
) -> dict[str, str]:
    """Return an environment dictionary suitable for HEASoft commands.

    The function does not source ``headas-init.sh``. Instead it sets the
    stable runtime variables exported by the Conda ``heainit.sh`` wrapper,
    including the Perl task runtime used by ``batsurvey``.  This keeps
    non-interactive/sandboxed runs from failing because the user's real HOME
    or PFILES directory is not writable.
    """

    out = dict(_os.environ if env is None else env)
    headas_path = _resolve_headas(headas)
    headas_bin = headas_path / "bin"
    headas_python = headas_path / "lib" / "python"
    syspfiles = headas_path / "syspfiles"

    out["HEADAS"] = str(headas_path)
    # ``headas-init.sh`` supplies these paths before any of the Perl-driver
    # tasks can run.  In particular, ``batsurvey`` exits immediately when
    # LHEAPERL is missing even though the executable itself is on PATH.  Fill
    # them from the selected HEADAS tree only when the caller has not already
    # supplied an explicit value.
    headas_lib = headas_path / "lib"
    conda_prefix = headas_path.parent
    perl_executable = conda_prefix / "bin" / "perl"
    environment_defaults = {
        "LHEASOFT": str(headas_path),
        "FTOOLS": str(headas_path),
        "XANADU": str(headas_path),
        "XANBIN": str(headas_path),
        "LHEA_DATA": str(headas_path / "refdata"),
        "LHEA_HELP": str(headas_path / "help"),
        "XRDEFAULTS": str(headas_path / "xrdefaults"),
        "PERL5LIB": str(headas_lib / "perl"),
        "PERLLIB": str(headas_lib / "perl"),
        "PGPLOT_DIR": str(headas_lib),
        "PGPLOT_FONT": str(headas_lib / "grfont.dat"),
        "PGPLOT_RGB": str(headas_lib / "rgb.txt"),
        "POW_LIBRARY": str(headas_lib / "pow"),
        "TCLRL_LIBDIR": str(headas_lib),
    }
    if perl_executable.is_file():
        environment_defaults["LHEAPERL"] = str(perl_executable)
    for name, value in environment_defaults.items():
        # Empty variables are what ``headas-init.sh`` leaves for some
        # installations (notably LHEAPERL before the wrapper is sourced),
        # but they are not usable runtime values.  Preserve a non-empty
        # caller override while filling only missing/empty entries.
        if not out.get(name):
            out[name] = value
    out["HEADASNOQUERY"] = "1"
    out["PATH"] = _prepend_env_path(out.get("PATH"), headas_bin)
    if headas_python.exists():
        out["PYTHONPATH"] = _prepend_env_path(out.get("PYTHONPATH"), headas_python)

    pfiles_dir = _safe_headas_dir(pfiles, "headas_pfiles")
    out["PFILES"] = f"{pfiles_dir}{_os.pathsep}{syspfiles}" if syspfiles.exists() else str(pfiles_dir)
    out["HOME"] = str(_safe_headas_dir(home, "headas_home"))
    return out


def find_headas_task(task: str, *, headas: str | _Path | None = None, env: _Mapping[str, str] | None = None) -> str:
    """Resolve a HEASoft task executable path using a prepared environment."""

    task_env = ensure_headas_env(headas=headas, env=env)
    found = _shutil.which(task, path=task_env.get("PATH"))
    if found is None:
        raise FileNotFoundError(f"HEASoft task not found in PATH: {task}")
    return found


def _normalize_headas_command(command: str | Sequence[str]) -> list[str]:
    if isinstance(command, str):
        return _shlex.split(command)
    return [str(part) for part in command]


def _write_command_log(path: str | _Path, *, command: Sequence[str], stdout: str, stderr: str, returncode: int) -> str:
    log_path = _Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    content = [
        "$ " + " ".join(_shlex.quote(x) for x in command),
        f"returncode={returncode}",
        "\n[stdout]",
        stdout or "",
        "\n[stderr]",
        stderr or "",
    ]
    log_path.write_text("\n".join(content), encoding="utf-8")
    return str(log_path)


def run_command(
    command: str | Sequence[str],
    *,
    cwd: str | _Path | None = None,
    env: _Mapping[str, str] | None = None,
    headas: str | _Path | None = None,
    input_text: str | None = None,
    log_path: str | _Path | None = None,
    check: bool = False,
) -> CommandResult:
    """Run a non-interactive HEASoft/FTOOLS command."""

    cmd = _normalize_headas_command(command)
    cmd_env = ensure_headas_env(headas=headas, env=env)
    proc = _subprocess.run(
        cmd,
        input=input_text,
        text=True,
        capture_output=True,
        cwd=None if cwd is None else str(cwd),
        env=cmd_env,
    )
    written_log = None
    if log_path is not None:
        written_log = _write_command_log(log_path, command=cmd, stdout=proc.stdout, stderr=proc.stderr, returncode=proc.returncode)
    result = CommandResult(
        command=tuple(cmd),
        returncode=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
        cwd=None if cwd is None else str(cwd),
        log_path=written_log,
    )
    if check and not result.ok:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(cmd)}\n{result.stderr or result.stdout}")
    return result


def _xselect_script_text(commands: str | _Iterable[str], *, append_exit: bool = True) -> str:
    if isinstance(commands, str):
        lines = [line.rstrip() for line in commands.splitlines()]
    else:
        lines = [str(line).rstrip() for line in commands]
    if append_exit:
        stripped = [line.strip().lower() for line in lines if line.strip()]
        if not stripped or stripped[-1] not in {"exit", "quit"}:
            lines.extend(["exit", "no"])
    return "\n".join(lines) + "\n"


def run_xselect_script(
    commands: str | _Iterable[str],
    *,
    cwd: str | _Path | None = None,
    env: _Mapping[str, str] | None = None,
    headas: str | _Path | None = None,
    script_path: str | _Path | None = None,
    log_path: str | _Path | None = None,
    append_exit: bool = True,
    check: bool = False,
) -> CommandResult:
    """Run official HEASoft ``xselect`` with commands supplied via stdin."""

    cmd_env = ensure_headas_env(headas=headas, env=env)
    xselect_exe = find_headas_task("xselect", env=cmd_env)
    script = _xselect_script_text(commands, append_exit=append_exit)

    written_script = None
    if script_path is not None:
        spath = _Path(script_path)
        spath.parent.mkdir(parents=True, exist_ok=True)
        spath.write_text(script, encoding="utf-8")
        written_script = str(spath)

    proc = _subprocess.run(
        [xselect_exe],
        input=script,
        text=True,
        capture_output=True,
        cwd=None if cwd is None else str(cwd),
        env=cmd_env,
    )
    written_log = None
    if log_path is not None:
        written_log = _write_command_log(log_path, command=[xselect_exe], stdout=proc.stdout, stderr=proc.stderr, returncode=proc.returncode)
        with _Path(written_log).open("a", encoding="utf-8") as f:
            f.write("\n\n[xselect script]\n")
            f.write(script)
    result = CommandResult(
        command=(xselect_exe,),
        returncode=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
        cwd=None if cwd is None else str(cwd),
        log_path=written_log,
        script_path=written_script,
    )
    if check and not result.ok:
        raise RuntimeError(f"xselect failed ({result.returncode})\n{result.stderr or result.stdout}")
    return result


__all__.extend([
    "CommandResult",
    "ensure_headas_env",
    "find_headas_task",
    "run_command",
    "run_xselect_script",
])
