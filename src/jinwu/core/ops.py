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

from typing import Optional, TYPE_CHECKING, Literal, Sequence

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
    if expo_arr.shape != width.shape:
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
    "bin_bblocks",
    "autobin",
    "txx",
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
    
    return LightcurveData(
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
    if method == 'auto':
        method = 'mean' if lc.is_rate else 'sum'

    if lc.value.ndim > 1:
        raise NotImplementedError("Rebin for multi-band LC not yet supported; slice bands first.")

    # Determine original bin edges. If lc.dt is provided, assume `lc.time` are
    # bin centers and construct edges accordingly. Otherwise treat times as
    # instantaneous and use small epsilon half-width equal to median spacing.
    t = np.asarray(lc.time, dtype=float)
    if t.size == 0:
        return LightcurveData(path=lc.path, time=np.array([], dtype=float), value=np.array([], dtype=float), error=None, dt=binsize, exposure=lc.exposure, bin_exposure=None, is_rate=lc.is_rate, header=lc.header, meta=lc.meta, headers_dump=lc.headers_dump, region=lc.region, bin_width=np.array([], dtype=float), binning='unknown')

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

    return LightcurveData(
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
        bin_lo=edges[:-1], bin_hi=edges[1:],
        bin_width=np.diff(edges),
        binning='uniform',
    )


# ==================== Bayesian Blocks Binning ====================

class BayesianBlocksBinner:
    """基于贝叶斯块的自适应分 bin。

    用法
    ----
        - 传入 `LightcurveData`（counts 或 rate），采用 `astropy.stats.bayesian_blocks`
            计算时间边界；输出块边界由 Bayesian Blocks 直接决定，不再做 SNR 阈值后合并。

    参数
    ----
    - p0: False positive rate (Scargle 2013)，控制块数量敏感度
    - fitness: Bayesian Blocks 统计模型，可选：
      * 'events': 泊松事件（光子计数等）
      * 'regular_events': 规则采样的事件数据
      * 'measures': 带误差的测量值（高斯统计，适用于已分bin的光变）

    返回
    ----
    - `LightcurveData` 新实例（counts 或 rate 与输入一致），时间为块中心，
      值为每块的聚合值，误差按平方和开方传播；每块的 `bin_exposure` 为实际覆盖曝光。
    """

    def __init__(
        self,
        p0: float = 0.05,
        fitness: Literal['events', 'regular_events', 'measures'] = 'measures',
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
            empty_cols = _ensure_lc_columns(getattr(lc, 'columns', ()), is_rate=lc.is_rate)
            return LightcurveData(
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
        if self.fitness == 'regular_events':
            edges = bayesian_blocks(t, counts, fitness=self.fitness, p0=self.p0, dt=orig_dt)
        elif self.fitness == 'measures':
            edges = bayesian_blocks(t, counts, fitness=self.fitness, p0=self.p0, sigma=err_counts)
        else:
            edges = bayesian_blocks(t, counts, fitness=self.fitness, p0=self.p0)
        # heapy/ppsignal 的做法会扩展首末边界到原始范围；保持与之兼容
        full_left = float(left.min())
        full_right = float(right.max())
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
            mask = (left < b) & (right > a)
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

        return LightcurveData(
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

        - `alpha`: 源/背景缩放因子（如面积或BACKSCAL比值）。若未提供，
          尝试以每块曝光之比近似：alpha = expo_src / expo_bkg。
        - 初始块边界（fitness='measures'）基于净计数 N = S - alpha*B。
        - 净计数误差传播：sigma_N^2 = sigma_S^2 + alpha^2 * sigma_B^2。
        - 块合并阈值使用带符号 Li&Ma 显著性。
        """
        

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
            return LightcurveData(
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

        base_alpha = 1.0 if alpha is None else float(alpha)
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
            src_slices.append(np.where((l_s < b) & (r_s > a))[0])
            bkg_slices.append(np.where((l_b < b) & (r_b > a))[0])

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
            alpha_blk = (1.0 if alpha is None else float(alpha)) * expo_ratio_blk
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
            return LightcurveData(
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

        return LightcurveData(
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
    >>> # 3. 传入 LightcurveDataset（自动提取 background 和 area_ratio）
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

    # 有背景：源+背景联合分 bin
    # alpha 最终回退逻辑（若前面未设置）
    if alpha is None:
        try:
            a_src = float(getattr(getattr(lc_src, 'region', None), 'area', None) or 0.0)
            a_bkg = float(getattr(getattr(lc_bkg, 'region', None), 'area', None) or 0.0)
            alpha = (a_src / a_bkg) if (a_src > 0 and a_bkg > 0) else 1.0
        except Exception:
            alpha = 1.0
    return BayesianBlocksBinner(p0=p0, fitness=fitness).fit_src_bkg(lc_src, lc_bkg, alpha=alpha)



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
            sig_raw = snr_li_ma(np.asarray([src_sum]), np.asarray([bkg_sum]), alpha_eff)
            sig = float(np.asarray(sig_raw, dtype=float).reshape(-1)[0])
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

    return LightcurveData(
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


def txx(
    lc_src: 'LightcurveData | np.ndarray',
    background: Optional['LightcurveData | np.ndarray'] = None,
    *,
    alpha: Optional[float] = None,
    percent: float | Sequence[float] = (0.5, 0.9),
    nmc: int = 1000,
    p0: float = 0.05,
    use_edge_bkg: bool = False,
    lbkg: Optional[float] = None,
    rbkg: Optional[float] = None,
    burst_tstart: Optional[float] = None,
    burst_tstop: Optional[float] = None,
    tpeak: Optional[float] = None,
    src_dist: Literal['poisson', 'gaussian'] = 'poisson',
    bkg_dist: Literal['poisson', 'gaussian'] = 'poisson',
    seed: Optional[int] = None,
    timebins: Optional[Sequence[float]] = None,
    small_bin_threshold: float = 4.0,
    weak_peak_bins: Sequence[float] = (8.0, 16.0, 32.0),
    weak_peak_weight: float = 0.2,
    window_mode: Literal['auto', 'density', 'weak_peak', 'peak'] = 'auto',
    density_quantile: float = 60.0,
    **kwargs,
) -> dict:
    """基于 Bayesian Blocks + MC 的 Txx 计算。

    约定
    ----
    - percent 为任意 (0, 1) 浮点或列表，默认 0.9 (T90)
    - 默认双泊松；不做自动模式判定
    - 背景实时扣除（按每个 bin 处理）
    - 爆发区间按 battblocks：第一块与最后一块作为背景，爆发区为它们之间
    - use_edge_bkg=True 时，仅使用首末块估计背景基线（适用于非成像）

        输入
        ----
        - lc_src/background 可为 LightcurveData 或 ndarray
            * 1D ndarray: 视为 counts，time=0..N-1, dt=1
            * 2D ndarray (N,2): [time, counts]
            * 2D ndarray (N,3): [time, counts, err]

        返回
    ----
    dict: 包含 txx/txx_err, txx1/txx1_err, txx2/txx2_err, peak/peak_err 等字段
    """
    if 'min_snr' in kwargs:
        kwargs.pop('min_snr')
        warnings.warn(
            "txx: 参数 min_snr 已移除并忽略。",
            RuntimeWarning,
            stacklevel=2,
        )
    if len(kwargs) > 0:
        unknown = ", ".join(sorted(str(k) for k in kwargs.keys()))
        raise TypeError(f"txx() got unexpected keyword argument(s): {unknown}")

    rng = np.random.default_rng(seed)

    def _is_lightcurve_like(obj: object) -> bool:
        # Normal case
        if isinstance(obj, LightcurveData):
            return True

        # Notebook reload compatibility: jinwu.core.file may have a new class identity
        try:
            from jinwu.core import file as _jwfile
            if isinstance(obj, _jwfile.LightcurveData):
                return True
        except Exception:
            pass

        # Duck-typing fallback for lc-like objects
        kind = getattr(obj, 'kind', None)
        if kind == 'lc':
            required_attrs = ('time', 'value', 'error', 'dt', 'is_rate')
            return all(hasattr(obj, attr) for attr in required_attrs)
        return False

    # ndarray 输入转换
    def _array_to_lc(arr: np.ndarray, name: str) -> 'LightcurveData':
        arr = np.asarray(arr)
        if arr.ndim == 1:
            counts = arr.astype(float)
            time = np.arange(counts.size, dtype=float)
            err = None
        elif arr.ndim == 2 and arr.shape[1] in (2, 3):
            time = arr[:, 0].astype(float)
            counts = arr[:, 1].astype(float)
            err = arr[:, 2].astype(float) if arr.shape[1] == 3 else None
        else:
            raise ValueError(f"{name} ndarray 仅支持 1D 或 (N,2)/(N,3) 形状")
        if time.size >= 2:
            dt = float(np.median(np.diff(time)))
        else:
            dt = 1.0
        bin_expo = np.full_like(time, dt, dtype=float)
        return LightcurveData(
            path=_Path("<array_input>"),
            time=time,
            value=counts,
            error=err,
            dt=dt,
            exposure=float(np.sum(bin_expo)),
            bin_exposure=bin_expo,
            is_rate=False,
            header={},
            meta={},
            headers_dump=None,
            region=None,
            bin_lo=(time - 0.5 * dt),
            bin_hi=(time + 0.5 * dt),
            bin_width=np.full_like(time, dt, dtype=float),
            binning='uniform',
        )

    if not _is_lightcurve_like(lc_src):
        lc_src = _array_to_lc(lc_src, "lc_src")
    if background is not None and not _is_lightcurve_like(background):
        background = _array_to_lc(background, "background")

    # 多 binsize 模式：小 binsize 走 density，大 binsize 走 weak_peak（低权重先验）
    if timebins is not None:
        bins = np.asarray(list(timebins), dtype=float)
        bins = bins[np.isfinite(bins) & (bins > 0.0)]
        if bins.size == 0:
            raise ValueError("timebins 为空或无有效正值")
        bins = np.unique(bins)

        weak_bins_arr = np.asarray(list(weak_peak_bins), dtype=float)

        per_binsize: list[dict] = []
        for k, bs in enumerate(bins):
            lc_src_i = rebin_lightcurve(lc_src, float(bs), method='auto')
            lc_bkg_i = rebin_lightcurve(background, float(bs), method='auto') if background is not None else None

            if float(bs) <= float(small_bin_threshold):
                mode_i: Literal['auto', 'density', 'weak_peak', 'peak'] = 'density'
                weak_w_i = 0.0
            elif np.any(np.isclose(float(bs), weak_bins_arr, rtol=0.0, atol=1e-8)):
                mode_i = 'weak_peak'
                weak_w_i = float(max(weak_peak_weight, 0.0))
            else:
                mode_i = 'peak'
                weak_w_i = 1.0

            res_i = txx(
                lc_src_i,
                background=lc_bkg_i,
                alpha=alpha,
                percent=percent,
                nmc=nmc,
                p0=p0,
                use_edge_bkg=use_edge_bkg,
                lbkg=lbkg,
                rbkg=rbkg,
                burst_tstart=burst_tstart,
                burst_tstop=burst_tstop,
                tpeak=tpeak,
                src_dist=src_dist,
                bkg_dist=bkg_dist,
                seed=(None if seed is None else int(seed + k)),
                timebins=None,
                small_bin_threshold=small_bin_threshold,
                weak_peak_bins=weak_peak_bins,
                weak_peak_weight=weak_w_i,
                window_mode=mode_i,
                density_quantile=density_quantile,
            )

            t90_i = float(np.asarray(res_i.get('t90', np.nan)).reshape(-1)[0])
            t90e_i = np.asarray(res_i.get('t90_err', [np.nan, np.nan]), dtype=float).reshape(-1)
            relerr_i = np.nan
            if np.isfinite(t90_i) and t90_i > 0 and t90e_i.size >= 2 and np.all(np.isfinite(t90e_i[:2])):
                relerr_i = float(np.mean(t90e_i[:2]) / t90_i)

            duty_i = float(res_i.get('density_positive_duty', np.nan))
            run_i = float(res_i.get('density_positive_run_ratio', np.nan))
            lowconf_i = bool(res_i.get('burst_window_low_confidence', False))
            valid_i = np.isfinite(t90_i) and (not lowconf_i)

            score_i = -1e9
            if valid_i:
                score_i = float((0.6 * duty_i + 0.4 * run_i) - 0.3 * (relerr_i if np.isfinite(relerr_i) else 1.0))
                if mode_i == 'weak_peak':
                    score_i += 0.05 * float(max(weak_w_i, 0.0))

            out_i = dict(res_i)
            out_i['binsize'] = float(bs)
            out_i['window_mode_used'] = mode_i
            out_i['selection_score'] = float(score_i)
            out_i['selection_relerr_t90'] = float(relerr_i) if np.isfinite(relerr_i) else np.nan
            out_i['selection_valid'] = bool(valid_i)
            per_binsize.append(out_i)

        scores = np.asarray([float(d.get('selection_score', -1e9)) for d in per_binsize], dtype=float)
        best_idx = int(np.nanargmax(scores)) if scores.size > 0 else 0
        best = dict(per_binsize[best_idx])
        best['multi_bin_used'] = True
        best['best_binsize'] = float(best.get('binsize', np.nan))
        best['per_binsize'] = per_binsize
        best['binsize_rank'] = [float(d.get('binsize', np.nan)) for d in sorted(per_binsize, key=lambda x: float(x.get('selection_score', -1e9)), reverse=True)]
        best['selection_reason'] = "max_density_stability_score"
        return best

    # percent 处理：对外保持用户输入，对内总是补齐 T50/T90
    if isinstance(percent, (float, int)):
        user_percent_arr = np.asarray([float(percent)], dtype=float)
    else:
        user_percent_arr = np.asarray(list(percent), dtype=float)
    if np.any((user_percent_arr <= 0.0) | (user_percent_arr >= 1.0)):
        raise ValueError("percent 必须在 (0, 1) 内")
    core_percent_arr = np.unique(np.concatenate([user_percent_arr, np.asarray([0.5, 0.9], dtype=float)]))

    # 将 LC 转为 counts 并推断边界
    def _lc_to_counts(lc: 'LightcurveData'):
        t = np.asarray(lc.time, dtype=float)
        vals = np.asarray(lc.value, dtype=float)
        errs = np.asarray(lc.error, dtype=float) if lc.error is not None else None
        left, right, width = _infer_bin_geometry(lc)
        width = right - left
        eff_width = _effective_exposure_from_lc(lc, width)
        if lc.is_rate:
            counts = vals * eff_width
            err_counts = (errs * eff_width) if errs is not None else None
        else:
            counts = vals.copy()
            err_counts = errs.copy() if errs is not None else None
        if err_counts is None:
            err_counts = np.sqrt(np.maximum(counts, 0.0))
        return t, left, right, width, counts, err_counts

    t_s, l_s, r_s, w_s, c_s, e_s = _lc_to_counts(lc_src)
    if t_s.size == 0:
        raise ValueError("lc_src 为空，无法计算 Txx")

    # Bayesian Blocks 分段，用于爆发区间定位
    binner = BayesianBlocksBinner(p0=p0, fitness='measures')
    _ = binner.fit(lc_src)
    edges = binner.last_edges
    if edges is None or edges.size < 2:
        edges = np.asarray([float(l_s.min()), float(r_s.max())], dtype=float)

    # 背景处理
    def _rebin_to_edges(src_left, src_right, src_counts, src_var, tgt_left, tgt_right):
        tgt_counts = np.zeros_like(tgt_left, dtype=float)
        tgt_var = np.zeros_like(tgt_left, dtype=float)
        for i, (a, b) in enumerate(zip(tgt_left, tgt_right)):
            overlap = np.minimum(src_right, b) - np.maximum(src_left, a)
            mask = overlap > 0
            if not np.any(mask):
                continue
            width = src_right[mask] - src_left[mask]
            frac = overlap[mask] / width
            tgt_counts[i] = float(np.sum(src_counts[mask] * frac))
            tgt_var[i] = float(np.sum(src_var[mask] * frac))
        return tgt_counts, tgt_var

    if use_edge_bkg:
        # 仅用首末块估计背景（适用于非成像仪器）
        bkg_counts = np.zeros_like(c_s, dtype=float)
        bkg_err = np.zeros_like(c_s, dtype=float)
        alpha = 1.0
        # 首末块区间
        t_left_bg = float(edges[0])
        t_right_bg = float(edges[1]) if edges.size >= 3 else float(l_s.min())
        t_left_bg2 = float(edges[-2]) if edges.size >= 3 else float(r_s.max())
        t_right_bg2 = float(edges[-1])

        mask1 = (l_s < t_right_bg) & (r_s > t_left_bg)
        mask2 = (l_s < t_right_bg2) & (r_s > t_left_bg2)
        rate1 = (np.sum(c_s[mask1]) / np.sum(w_s[mask1])) if np.any(mask1) else 0.0
        rate2 = (np.sum(c_s[mask2]) / np.sum(w_s[mask2])) if np.any(mask2) else rate1
        t1 = 0.5 * (t_left_bg + t_right_bg)
        t2 = 0.5 * (t_left_bg2 + t_right_bg2)
        if t2 == t1:
            t2 = t1 + 1.0
        # 线性背景
        center = 0.5 * (l_s + r_s)
        slope = (rate2 - rate1) / (t2 - t1)
        bkg_rate = rate1 + slope * (center - t1)
        bkg_counts = bkg_rate * w_s
        bkg_err = np.sqrt(np.maximum(bkg_counts, 0.0))
    else:
        if background is None:
            bkg_counts = np.zeros_like(c_s, dtype=float)
            bkg_err = np.zeros_like(c_s, dtype=float)
            alpha = 0.0
        else:
            if alpha is None:
                raise ValueError("提供 background 时必须显式传入 alpha")
            t_b, l_b, r_b, w_b, c_b, e_b = _lc_to_counts(background)
            var_b = e_b ** 2
            bkg_counts, bkg_var = _rebin_to_edges(l_b, r_b, c_b, var_b, l_s, r_s)
            bkg_err = np.sqrt(np.maximum(bkg_var, 0.0))

    # 净计数
    net_counts = c_s - (alpha or 0.0) * bkg_counts
    net_err = np.sqrt(np.maximum(e_s ** 2 + (alpha or 0.0) ** 2 * bkg_err ** 2, 0.0))

    # 基于贝叶斯块净计数识别主爆发窗(T100)与前驱窗
    block_left = edges[:-1]
    block_right = edges[1:]
    block_width = np.maximum(block_right - block_left, 1e-12)

    def _block_aggregate_from_series(series: np.ndarray, *, power: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        out = np.zeros(block_left.size, dtype=float)
        nraw = np.zeros(block_left.size, dtype=int)
        for i, (a, b) in enumerate(zip(block_left, block_right)):
            overlap = np.minimum(r_s, b) - np.maximum(l_s, a)
            mask = overlap > 0.0
            if not np.any(mask):
                continue
            frac = overlap[mask] / np.maximum((r_s[mask] - l_s[mask]), 1e-12)
            out[i] = float(np.sum(series[mask] * (frac ** power)))
            nraw[i] = int(np.count_nonzero(mask))
        return out, nraw

    def _block_net_from_series(series: np.ndarray) -> np.ndarray:
        out, _ = _block_aggregate_from_series(series, power=1.0)
        return out

    def _segments(mask: np.ndarray) -> list[tuple[int, int]]:
        segs: list[tuple[int, int]] = []
        i = 0
        n = mask.size
        while i < n:
            if not mask[i]:
                i += 1
                continue
            j = i + 1
            while j < n and mask[j]:
                j += 1
            segs.append((i, j))
            i = j
        return segs

    def _infer_windows_from_series(series: np.ndarray, series_err: Optional[np.ndarray] = None) -> tuple[float, float, float, float, bool]:
        # 参数：双阈值扩张 + 短桥接 + 边际停止（左侧更严格，抑制过早起始）
        z_seed = 2.5
        z_tail_left = 1.2
        z_tail_right = 0.8
        frac_tail_left = 0.03
        frac_tail_right = 0.015
        frac_rate_left = 0.35
        frac_rate_right = 0.02
        marginal_frac = 0.005
        marginal_n_left = 1
        marginal_n_right = 2
        g_max_left = 0
        g_max_right = 1
        min_raw_bins = 2

        mode_now = window_mode
        if mode_now == 'auto':
            median_width = float(np.nanmedian(np.maximum(w_s, 1e-12))) if w_s.size > 0 else 1.0
            weak_bins_arr_local = np.asarray(list(weak_peak_bins), dtype=float)
            if median_width <= float(small_bin_threshold):
                mode_now = 'density'
            elif np.any(np.isclose(median_width, weak_bins_arr_local, rtol=0.0, atol=1e-8)):
                mode_now = 'weak_peak'
            else:
                mode_now = 'peak'

        block_net_raw, block_nraw = _block_aggregate_from_series(series, power=1.0)
        if series_err is not None:
            block_var, _ = _block_aggregate_from_series(np.maximum(series_err, 0.0) ** 2, power=2.0)
        else:
            block_var = np.maximum(block_net_raw, 0.0)
        block_err = np.sqrt(np.maximum(block_var, 1e-12))

        # 残余基线扣除：块率稳健中位数
        block_rate = block_net_raw / block_width
        z_raw = block_net_raw / block_err
        non_sig = np.isfinite(block_rate) & np.isfinite(z_raw) & (np.abs(z_raw) < 1.0)
        if np.any(non_sig):
            baseline_rate = float(np.nanmedian(block_rate[non_sig]))
        elif np.any(np.isfinite(block_rate)):
            baseline_rate = float(np.nanmedian(block_rate[np.isfinite(block_rate)]))
        else:
            baseline_rate = 0.0
        block_net = block_net_raw - baseline_rate * block_width

        block_z = block_net / block_err
        finite = np.isfinite(block_net) & np.isfinite(block_z)
        if not np.any(finite):
            mid = int(np.argmax(np.maximum(series, -np.inf))) if series.size > 0 else 0
            mid = int(np.clip(mid, 0, l_s.size - 1))
            i0 = max(mid - 1, 0)
            i1 = min(mid + 1, l_s.size - 1)
            return float(l_s[i0]), float(r_s[i1]), np.nan, np.nan, True

        peak_block_net = float(np.nanmax(np.where(finite, block_net, np.nan)))
        if not np.isfinite(peak_block_net):
            peak_block_net = 0.0
        tail_thr_left = frac_tail_left * max(peak_block_net, 0.0)
        tail_thr_right = frac_tail_right * max(peak_block_net, 0.0)
        peak_block_rate = float(np.nanmax(np.where(finite, block_net / block_width, np.nan)))
        if not np.isfinite(peak_block_rate):
            peak_block_rate = 0.0
        rate_thr_left = frac_rate_left * max(peak_block_rate, 0.0)
        rate_thr_right = frac_rate_right * max(peak_block_rate, 0.0)
        total_pos = float(np.sum(np.maximum(block_net[finite], 0.0)))
        total_pos = max(total_pos, 1e-12)

        wide_enough = block_nraw >= int(min_raw_bins)
        block_rate_corr = block_net / block_width

        dens_base = float(np.nanpercentile(np.maximum(block_rate_corr[finite], 0.0), float(np.clip(density_quantile, 1.0, 99.0)))) if np.any(finite) else 0.0
        dens_floor = max(dens_base, 0.0)

        if mode_now == 'density':
            seed_mask = finite & wide_enough & ((block_rate_corr >= dens_floor) | (block_z >= 1.0))
            tail_mask_left = finite & wide_enough & (block_rate_corr >= 0.5 * dens_floor)
            tail_mask_right = tail_mask_left.copy()
            g_max_left_local = 1
            g_max_right_local = 1
            z_tail_left_local = 0.2
            z_tail_right_local = 0.2
        elif mode_now == 'weak_peak':
            peak_gate_left = ((block_net >= tail_thr_left) & (block_rate_corr >= rate_thr_left))
            peak_gate_right = ((block_net >= tail_thr_right) & (block_rate_corr >= rate_thr_right))
            density_gate_left = (block_rate_corr >= 0.4 * dens_floor)
            density_gate_right = (block_rate_corr >= 0.35 * dens_floor)
            seed_mask = finite & wide_enough & ((block_z >= z_seed) | (block_rate_corr >= dens_floor)) & (block_net > 0.0)
            tail_mask_left = finite & wide_enough & ((block_z >= z_tail_left) | density_gate_left | (peak_gate_left & (weak_peak_weight > 0.0)))
            tail_mask_right = finite & wide_enough & ((block_z >= z_tail_right) | density_gate_right | (peak_gate_right & (weak_peak_weight > 0.0)))
            g_max_left_local = g_max_left
            g_max_right_local = g_max_right
            z_tail_left_local = z_tail_left
            z_tail_right_local = z_tail_right
        else:
            seed_mask = finite & wide_enough & (block_z >= z_seed) & (block_net > 0.0)
            tail_mask_left = finite & wide_enough & ((block_z >= z_tail_left) | (block_net >= tail_thr_left)) & (block_rate_corr >= rate_thr_left)
            tail_mask_right = finite & wide_enough & ((block_z >= z_tail_right) | (block_net >= tail_thr_right)) & (block_rate_corr >= rate_thr_right)
            g_max_left_local = g_max_left
            g_max_right_local = g_max_right
            z_tail_left_local = z_tail_left
            z_tail_right_local = z_tail_right

        def _expand_one_side(
            seed_idx: int,
            direction: int,
            tail_mask: np.ndarray,
            z_tail_local: float,
            marginal_n_local: int,
            g_max_local: int,
        ) -> int:
            i = seed_idx + direction
            last = seed_idx
            gaps_used = 0
            marginal_run = 0
            nblk = block_net.size
            while 0 <= i < nblk:
                if tail_mask[i]:
                    last = i
                    frac_i = max(block_net[i], 0.0) / total_pos
                    if (block_z[i] < z_tail_local) and (frac_i < marginal_frac):
                        marginal_run += 1
                    else:
                        marginal_run = 0
                    if marginal_run >= marginal_n_local:
                        break
                    i += direction
                    continue

                if gaps_used < g_max_local:
                    j = i + direction
                    if 0 <= j < nblk and tail_mask[j]:
                        # 保留短桥接块
                        last = j
                        gaps_used += 1
                        frac_j = max(block_net[j], 0.0) / total_pos
                        if (block_z[j] < z_tail_local) and (frac_j < marginal_frac):
                            marginal_run += 1
                        else:
                            marginal_run = 0
                        if marginal_run >= marginal_n_local:
                            break
                        i = j + direction
                        continue
                break
            return int(last)

        def _window_score(i0: int, i1: int) -> float:
            if i1 < i0:
                return -np.inf
            return float(np.sum(np.maximum(block_net[i0:i1 + 1], 0.0)))

        seeds = np.where(seed_mask)[0]
        pre_start = np.nan
        pre_stop = np.nan
        low_confidence = False

        if len(seeds) == 0:
            # 失败保护：不回退全时段，回退到主峰附近受限窗口
            peak_idx = int(np.nanargmax(np.where(finite, block_net, np.nan)))
            i0 = max(peak_idx - 1, 0)
            i1 = min(peak_idx + 1, block_net.size - 1)
            return float(block_left[i0]), float(block_right[i1]), np.nan, np.nan, True

        # 锚定主峰：在种子中选校正后净计数最大的块，避免被早期弱正种子牵引
        svals = block_net[seeds]
        seed_idx = int(seeds[int(np.nanargmax(svals))])

        right_idx = _expand_one_side(seed_idx, +1, tail_mask_right, z_tail_right_local, marginal_n_right, g_max_right_local)
        left_idx = _expand_one_side(seed_idx, -1, tail_mask_left, z_tail_left_local, marginal_n_left, g_max_left_local)
        mi0 = min(left_idx, seed_idx)
        mi1 = max(right_idx, seed_idx)

        # 左端贡献裁剪：去掉主窗内贡献极小的前导段，抑制起点过早
        trim_left_frac = 0.01
        if mi1 >= mi0:
            seg_net = np.maximum(block_net[mi0:mi1 + 1], 0.0)
            seg_sum = float(np.sum(seg_net))
            if np.isfinite(seg_sum) and seg_sum > 0.0:
                seg_cum = np.cumsum(seg_net)
                left_target = trim_left_frac * seg_sum
                rel = int(np.searchsorted(seg_cum, left_target, side='left'))
                rel = int(np.clip(rel, 0, seg_net.size - 1))
                candidate = mi0 + rel
                # 仅当候选块满足基本尾部条件时才裁剪
                cond = ((block_z[candidate] >= z_tail_left_local) or (block_net[candidate] >= tail_thr_left) or (block_rate_corr[candidate] >= 0.5 * dens_floor))
                if cond and candidate <= mi1:
                    mi0 = int(candidate)

        # 前驱窗：仅统计主窗左侧连续正净计数段中得分最高者
        pre_mask = np.isfinite(block_net) & (block_net > 0.0)
        pre_mask[mi0:] = False
        pre_segs = _segments(pre_mask)
        if len(pre_segs) > 0:
            pre_scores = np.asarray([_window_score(i0, j0 - 1) for (i0, j0) in pre_segs], dtype=float)
            pk = int(np.nanargmax(pre_scores))
            pi0, pj0 = pre_segs[pk]
            pre_start = float(block_left[pi0])
            pre_stop = float(block_right[pj0 - 1])

        return float(block_left[mi0]), float(block_right[mi1]), pre_start, pre_stop, low_confidence

    precursor_tstart = np.nan
    precursor_tstop = np.nan
    burst_low_confidence = False
    if (burst_tstart is None) or (burst_tstop is None):
        burst_tstart_auto, burst_tstop_auto, precursor_tstart, precursor_tstop, burst_low_confidence = _infer_windows_from_series(net_counts, net_err)
        if burst_tstart is None:
            burst_tstart = burst_tstart_auto
        if burst_tstop is None:
            burst_tstop = burst_tstop_auto

    if burst_tstop <= burst_tstart:
        raise ValueError("爆发区间无效：burst_tstop <= burst_tstart")

    # 爆发区间内的 bin
    in_burst = (l_s < burst_tstop) & (r_s > burst_tstart)
    if not np.any(in_burst):
        raise ValueError("爆发区间内无数据 bin")
    l_burst = l_s[in_burst]
    r_burst = r_s[in_burst]
    w_burst = w_s[in_burst]
    net_burst = net_counts[in_burst]
    err_burst = net_err[in_burst]

    pos_mask_burst = np.isfinite(net_burst) & (net_burst > 0.0)
    density_positive_duty = float(np.mean(pos_mask_burst)) if net_burst.size > 0 else np.nan
    density_positive_rate = float(np.sum(np.maximum(net_burst, 0.0)) / np.sum(np.maximum(w_burst, 1e-12))) if net_burst.size > 0 else np.nan
    if net_burst.size > 0:
        pos_i = pos_mask_burst.astype(int)
        max_run = 0
        cur = 0
        for v in pos_i:
            if v == 1:
                cur += 1
                if cur > max_run:
                    max_run = cur
            else:
                cur = 0
        density_positive_run_ratio = float(max_run / max(int(net_burst.size), 1))
    else:
        density_positive_run_ratio = np.nan

    # 使用观测值计算一次 Txx（作为名义值）
    def _crossing_time(right_edges, csum, target):
        if target <= 0:
            return float(right_edges[0])
        if target >= csum[-1]:
            return float(right_edges[-1])
        idx = int(np.searchsorted(csum, target, side='left'))
        if idx == 0:
            return float(right_edges[0])
        c0 = csum[idx - 1]
        c1 = csum[idx]
        t0 = right_edges[idx - 1]
        t1 = right_edges[idx]
        if c1 <= c0:
            return float(t1)
        return float(t0 + (target - c0) * (t1 - t0) / (c1 - c0))

    def _compute_txx_from_counts(counts, right_edges, perc_arr):
        csum = np.cumsum(counts)
        if csum.size == 0:
            return np.full(perc_arr.shape, np.nan), np.full(perc_arr.shape, np.nan), np.full(perc_arr.shape, np.nan)
        csum = np.maximum.accumulate(csum)
        total = float(csum[-1])
        if total <= 0:
            return np.full(perc_arr.shape, np.nan), np.full(perc_arr.shape, np.nan), np.full(perc_arr.shape, np.nan)
        t1_list = []
        t2_list = []
        for p in perc_arr:
            low = 0.5 * (1.0 - p) * total
            high = 0.5 * (1.0 + p) * total
            t1 = _crossing_time(right_edges, csum, low)
            t2 = _crossing_time(right_edges, csum, high)
            t1_list.append(t1)
            t2_list.append(t2)
        t1_arr = np.asarray(t1_list, dtype=float)
        t2_arr = np.asarray(t2_list, dtype=float)
        return t2_arr - t1_arr, t1_arr, t2_arr

    dur_nom_all, t1_nom_all, t2_nom_all = _compute_txx_from_counts(net_burst, r_burst, core_percent_arr)

    def _indices_for(perc_all: np.ndarray, perc_pick: np.ndarray) -> np.ndarray:
        idx = []
        for p in perc_pick:
            hits = np.where(np.isclose(perc_all, p, rtol=0.0, atol=1e-12))[0]
            if hits.size == 0:
                raise RuntimeError(f"内部百分位映射失败: p={p}")
            idx.append(int(hits[0]))
        return np.asarray(idx, dtype=int)

    idx_user = _indices_for(core_percent_arr, user_percent_arr)
    idx50 = int(_indices_for(core_percent_arr, np.asarray([0.5], dtype=float))[0])
    idx90 = int(_indices_for(core_percent_arr, np.asarray([0.9], dtype=float))[0])

    txx_nom = dur_nom_all[idx_user]
    txx1_nom = t1_nom_all[idx_user]
    txx2_nom = t2_nom_all[idx_user]

    def _window_counts_and_exposure(counts, left_arr, right_arr, t0, t1):
        overlap = np.minimum(right_arr, t1) - np.maximum(left_arr, t0)
        mask = overlap > 0.0
        if not np.any(mask):
            return 0.0, 0.0
        width = np.maximum(right_arr[mask] - left_arr[mask], 1e-12)
        frac = overlap[mask] / width
        csum = float(np.sum(counts[mask] * frac))
        expo = float(np.sum((right_arr[mask] - left_arr[mask]) * frac))
        return csum, expo

    def _mean_rate_in_interval(counts, left_arr, right_arr, t0, t1):
        csum, expo = _window_counts_and_exposure(counts, left_arr, right_arr, t0, t1)
        if expo <= 0.0:
            return np.nan
        return csum / expo

    def _peak_time_from_counts(counts, left_arr, right_arr):
        if tpeak is None:
            width = np.maximum(right_arr - left_arr, 1e-12)
            idx = int(np.argmax(counts / width))
            return float(0.5 * (left_arr[idx] + right_arr[idx]))
        window = float(tpeak)
        if window <= 0.0:
            return np.nan
        starts = np.asarray(left_arr, dtype=float)
        best = -np.inf
        best_t = 0.5 * (burst_tstart + burst_tstop)
        for start in starts:
            end = start + window
            if end > float(np.max(right_arr)):
                continue
            csum, _ = _window_counts_and_exposure(counts, left_arr, right_arr, start, end)
            if csum > best:
                best = csum
                best_t = start + 0.5 * window
        return float(best_t)

    def _peak_rate_from_counts(counts, left_arr, right_arr, window: float = 1.0):
        window = float(window)
        if window <= 0.0:
            return np.nan, np.nan
        starts = np.asarray(left_arr, dtype=float)
        best_rate = -np.inf
        best_t = np.nan
        rmax = float(np.max(right_arr))
        for start in starts:
            end = start + window
            if end > rmax:
                continue
            csum, expo = _window_counts_and_exposure(counts, left_arr, right_arr, start, end)
            if expo <= 0.0:
                continue
            rate = csum / expo
            if rate > best_rate:
                best_rate = rate
                best_t = start + 0.5 * window
        if not np.isfinite(best_rate):
            csum, expo = _window_counts_and_exposure(counts, left_arr, right_arr, float(np.min(left_arr)), rmax)
            if expo <= 0.0:
                return np.nan, np.nan
            return csum / expo, 0.5 * (float(np.min(left_arr)) + rmax)
        return float(best_rate), float(best_t)

    peak_nom = _peak_time_from_counts(net_burst, l_burst, r_burst)
    peak_rate_nom, _ = _peak_rate_from_counts(net_burst, l_burst, r_burst, window=1.0)

    t50_nom = float(dur_nom_all[idx50])
    t50_start_nom = float(t1_nom_all[idx50])
    t50_stop_nom = float(t2_nom_all[idx50])
    t90_nom = float(dur_nom_all[idx90])
    t90_start_nom = float(t1_nom_all[idx90])
    t90_stop_nom = float(t2_nom_all[idx90])
    t100_nom = float(burst_tstop - burst_tstart)

    mean_rate_t50_nom = _mean_rate_in_interval(net_burst, l_burst, r_burst, t50_start_nom, t50_stop_nom)
    mean_rate_t90_nom = _mean_rate_in_interval(net_burst, l_burst, r_burst, t90_start_nom, t90_stop_nom)
    mean_rate_t100_nom = _mean_rate_in_interval(net_burst, l_burst, r_burst, float(burst_tstart), float(burst_tstop))

    # MC 误差估计
    nmc = int(nmc)
    txx_samples_all = np.zeros((nmc, core_percent_arr.size), dtype=float)
    txx1_samples_all = np.zeros((nmc, core_percent_arr.size), dtype=float)
    txx2_samples_all = np.zeros((nmc, core_percent_arr.size), dtype=float)
    peak_samples = np.zeros(nmc, dtype=float)
    peak_rate_samples = np.zeros(nmc, dtype=float)
    t100_samples = np.zeros(nmc, dtype=float)
    mean_rate_t50_samples = np.zeros(nmc, dtype=float)
    mean_rate_t90_samples = np.zeros(nmc, dtype=float)
    mean_rate_t100_samples = np.zeros(nmc, dtype=float)

    def _draw_counts(mean, err, dist):
        if dist == 'poisson':
            return rng.poisson(np.maximum(mean, 0.0))
        return rng.normal(mean, err)

    for i in range(nmc):
        src_draw = _draw_counts(c_s, e_s, src_dist)
        bkg_draw = _draw_counts(bkg_counts, bkg_err, bkg_dist) if (not use_edge_bkg) else bkg_counts
        net_draw = src_draw - (alpha or 0.0) * bkg_draw
        if (burst_tstart is None) or (burst_tstop is None):
            draw_err = np.sqrt(np.maximum(src_draw + ((alpha or 0.0) ** 2) * np.maximum(bkg_draw, 0.0), 0.0))
            bt0, bt1, _, _, _ = _infer_windows_from_series(net_draw, draw_err)
        else:
            bt0, bt1 = float(burst_tstart), float(burst_tstop)
        if bt1 <= bt0:
            pk = int(np.argmax(net_draw))
            i0 = max(pk - 1, 0)
            i1 = min(pk + 1, l_s.size - 1)
            bt0, bt1 = float(l_s[i0]), float(r_s[i1])

        draw_mask = (l_s < bt1) & (r_s > bt0)
        if not np.any(draw_mask):
            draw_mask = in_burst
            bt0, bt1 = float(burst_tstart), float(burst_tstop)
        l_draw = l_s[draw_mask]
        r_draw = r_s[draw_mask]
        net_draw_burst = net_draw[draw_mask]

        txx_i_all, t1_i_all, t2_i_all = _compute_txx_from_counts(net_draw_burst, r_draw, core_percent_arr)
        txx_samples_all[i, :] = txx_i_all
        txx1_samples_all[i, :] = t1_i_all
        txx2_samples_all[i, :] = t2_i_all
        peak_samples[i] = _peak_time_from_counts(net_draw_burst, l_draw, r_draw)
        peak_rate_i, _ = _peak_rate_from_counts(net_draw_burst, l_draw, r_draw, window=1.0)
        peak_rate_samples[i] = peak_rate_i
        t100_samples[i] = bt1 - bt0

        t50_s = float(t1_i_all[idx50])
        t50_e = float(t2_i_all[idx50])
        t90_s = float(t1_i_all[idx90])
        t90_e = float(t2_i_all[idx90])
        mean_rate_t50_samples[i] = _mean_rate_in_interval(net_draw_burst, l_draw, r_draw, t50_s, t50_e)
        mean_rate_t90_samples[i] = _mean_rate_in_interval(net_draw_burst, l_draw, r_draw, t90_s, t90_e)
        mean_rate_t100_samples[i] = _mean_rate_in_interval(net_draw_burst, l_draw, r_draw, bt0, bt1)

    def _asym_err(samples, central):
        p16 = np.nanpercentile(samples, 16, axis=0)
        p84 = np.nanpercentile(samples, 84, axis=0)
        low = np.maximum(central - p16, 0.0)
        high = np.maximum(p84 - central, 0.0)
        return np.vstack([low, high]).T

    txx_samples = txx_samples_all[:, idx_user]
    txx1_samples = txx1_samples_all[:, idx_user]
    txx2_samples = txx2_samples_all[:, idx_user]

    txx_err = _asym_err(txx_samples, txx_nom)
    txx1_err = _asym_err(txx1_samples, txx1_nom)
    txx2_err = _asym_err(txx2_samples, txx2_nom)
    peak_err = _asym_err(peak_samples, peak_nom).reshape(2,)
    peak_rate_err = _asym_err(peak_rate_samples, peak_rate_nom).reshape(2,)
    t100_err = _asym_err(t100_samples, t100_nom).reshape(2,)

    t50_err = _asym_err(txx_samples_all[:, idx50], t50_nom).reshape(2,)
    t50_start_err = _asym_err(txx1_samples_all[:, idx50], t50_start_nom).reshape(2,)
    t50_stop_err = _asym_err(txx2_samples_all[:, idx50], t50_stop_nom).reshape(2,)
    t90_err = _asym_err(txx_samples_all[:, idx90], t90_nom).reshape(2,)
    t90_start_err = _asym_err(txx1_samples_all[:, idx90], t90_start_nom).reshape(2,)
    t90_stop_err = _asym_err(txx2_samples_all[:, idx90], t90_stop_nom).reshape(2,)

    mean_rate_t50_err = _asym_err(mean_rate_t50_samples, mean_rate_t50_nom).reshape(2,)
    mean_rate_t90_err = _asym_err(mean_rate_t90_samples, mean_rate_t90_nom).reshape(2,)
    mean_rate_t100_err = _asym_err(mean_rate_t100_samples, mean_rate_t100_nom).reshape(2,)

    return {
        "percent": user_percent_arr,
        "txx": txx_nom,
        "txx_err": txx_err,
        "txx1": txx1_nom,
        "txx1_err": txx1_err,
        "txx2": txx2_nom,
        "txx2_err": txx2_err,

        "t50": t50_nom,
        "t50_err": t50_err,
        "t50_tstart": t50_start_nom,
        "t50_tstart_err": t50_start_err,
        "t50_tstop": t50_stop_nom,
        "t50_tstop_err": t50_stop_err,

        "t90": t90_nom,
        "t90_err": t90_err,
        "t90_tstart": t90_start_nom,
        "t90_tstart_err": t90_start_err,
        "t90_tstop": t90_stop_nom,
        "t90_tstop_err": t90_stop_err,

        "peak": peak_nom,
        "peak_err": peak_err,
        "peak_time": peak_nom,
        "peak_time_err": peak_err,
        "peak_rate": peak_rate_nom,
        "peak_rate_err": peak_rate_err,

        "t100": float(burst_tstop - burst_tstart),
        "t100_err": t100_err,
        "t100_tstart": float(burst_tstart),
        "t100_tstop": float(burst_tstop),

        "mean_rate_t50": mean_rate_t50_nom,
        "mean_rate_t50_err": mean_rate_t50_err,
        "mean_rate_t90": mean_rate_t90_nom,
        "mean_rate_t90_err": mean_rate_t90_err,
        "mean_rate_t100": mean_rate_t100_nom,
        "mean_rate_t100_err": mean_rate_t100_err,

        "precursor_tstart": float(precursor_tstart) if np.isfinite(precursor_tstart) else np.nan,
        "precursor_tstop": float(precursor_tstop) if np.isfinite(precursor_tstop) else np.nan,
        "precursor_t100": (
            float(precursor_tstop - precursor_tstart)
            if (np.isfinite(precursor_tstart) and np.isfinite(precursor_tstop))
            else np.nan
        ),
        "burst_tstart": burst_tstart,
        "burst_tstop": burst_tstop,
        "burst_window_low_confidence": bool(burst_low_confidence),
        "window_mode_used": window_mode,
        "density_positive_duty": density_positive_duty,
        "density_positive_rate": density_positive_rate,
        "density_positive_run_ratio": density_positive_run_ratio,
    }