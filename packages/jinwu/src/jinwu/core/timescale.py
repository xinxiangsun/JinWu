"""时标（T100/T90/T50 等）计算方法。

从 ``jinwu.core.ops`` 拆分出来的时标专题模块，汇集两种独立方法学：

1. :func:`txx` —— EXTraS 背景时间变换定窗 + 有符号累计 + Koshut 误差；
2. :func:`txx_iterbkg` —— 迭代背景自洽的分箱贝叶斯块法（移植自
   HEASoft 6.37 ``burstcube/lib/bayesian_lc.py``），含 Giacomo 技巧、
   循环检测收敛与整条流水线的 Poisson 重采样误差。

推荐入口为 ``jinwu.core.data.timescale`` 分析器（``method`` 参数可选方法）；
``jinwu.core.ops.txx`` / ``ops.txx_iterbkg`` 作为兼容别名保留。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path as _Path
from typing import Literal, Optional, Sequence, cast, TYPE_CHECKING
import warnings

import numpy as np
from astropy.stats import bayesian_blocks

from .base import LightcurveDataBase, EventDataBase
from .utils import li_ma_snr

if TYPE_CHECKING:
    from .data import LightcurveData, EventData

__all__ = [
    "txx",
    "IterativeBBResult",
    "iterative_bayesian_blocks",
    "txx_iterbkg",
]


# ==================== Txx：事件级贝叶斯块 + A&A 5.4 分位累计 ====================
def _txx54_is_lightcurve_like(obj: object) -> bool:
    if isinstance(obj, LightcurveDataBase):
        return True
    kind = getattr(obj, 'kind', None)
    if kind == 'lc':
        required_attrs = ('time', 'value', 'error', 'dt', 'is_rate')
        return all(hasattr(obj, attr) for attr in required_attrs)
    return False


def _txx54_array_to_lc(arr: np.ndarray, name: str) -> 'LightcurveData':
    from .data import LightcurveData as _LightcurveData

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

    dt = float(np.median(np.diff(time))) if time.size >= 2 else 1.0
    bin_expo = np.full_like(time, dt, dtype=float)
    return _LightcurveData(
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


def _txx54_to_counts(lc: 'LightcurveData') -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # 延迟导入 .ops 的几何/曝光助手，避免模块循环（.ops 会重导出本模块的 txx）。
    from .ops import _effective_exposure_from_lc, _infer_bin_geometry

    left, right, width = _infer_bin_geometry(lc)
    width = np.maximum(np.asarray(width, dtype=float), 1e-12)
    vals = np.asarray(lc.value, dtype=float)
    if vals.ndim != 1:
        raise NotImplementedError("txx 新实现暂不支持多能段光变，请先切片到单能段。")

    if lc.is_rate:
        eff = _effective_exposure_from_lc(lc, width)
        counts = vals * np.asarray(eff, dtype=float)
    else:
        counts = vals.copy()

    order = np.argsort(left)
    return (
        np.asarray(left, dtype=float)[order],
        np.asarray(right, dtype=float)[order],
        np.asarray(width, dtype=float)[order],
        np.asarray(counts, dtype=float)[order],
    )


def _txx54_project_counts_to_src_grid(
    src_left: np.ndarray,
    src_right: np.ndarray,
    bkg_left: np.ndarray,
    bkg_right: np.ndarray,
    bkg_counts: np.ndarray,
) -> np.ndarray:
    out = np.zeros(src_left.size, dtype=float)
    bkg_width = np.maximum(bkg_right - bkg_left, 1e-12)
    for i in range(src_left.size):
        a = float(src_left[i])
        b = float(src_right[i])
        overlap = np.minimum(bkg_right, b) - np.maximum(bkg_left, a)
        mask = overlap > 0.0
        if not np.any(mask):
            continue
        frac = overlap[mask] / bkg_width[mask]
        out[i] = float(np.sum(bkg_counts[mask] * frac))
    return out


def _txx54_overlap_sum(
    values: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    a: float,
    b: float,
) -> float:
    overlap = np.minimum(right, b) - np.maximum(left, a)
    mask = overlap > 0.0
    if not np.any(mask):
        return 0.0
    width = np.maximum(right[mask] - left[mask], 1e-12)
    frac = overlap[mask] / width
    return float(np.sum(values[mask] * frac))


def _txx54_contiguous_groups(mask: np.ndarray) -> list[tuple[int, int]]:
    groups: list[tuple[int, int]] = []
    i = 0
    n = mask.size
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and mask[j + 1]:
            j += 1
        groups.append((i, j))
        i = j + 1
    return groups


def _txx54_cross_target(
    target: float,
    seg_left: np.ndarray,
    seg_right: np.ndarray,
    seg_counts: np.ndarray,
) -> float:
    if seg_left.size == 0:
        return np.nan
    if target <= 0.0:
        return float(seg_left[0])

    csum = 0.0
    for l, r, c in zip(seg_left, seg_right, seg_counts):
        if csum + c >= target:
            if c <= 0.0:
                return float(l)
            frac = (target - csum) / c
            return float(l + frac * (r - l))
        csum += c
    return float(seg_right[-1])


def _txx54_asymm_err_from_samples(samples: np.ndarray, nominal: float) -> tuple[float, float]:
    vals = np.asarray(samples, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size < 10 or (not np.isfinite(nominal)):
        return np.nan, np.nan
    q16, q84 = np.percentile(vals, [16.0, 84.0])
    # Signed offsets preserve intervals that do not contain the nominal value.
    err_m = float(nominal - q16)
    err_p = float(q84 - nominal)
    return err_m, err_p


def _txx54_robust_sigma(samples: np.ndarray) -> float:
    vals = np.asarray(samples, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size < 2:
        return np.nan
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    sig = 1.4826 * mad
    if np.isfinite(sig) and sig > 0.0:
        return float(sig)
    std = float(np.std(vals, ddof=1))
    if np.isfinite(std) and std >= 0.0:
        return std
    return np.nan


def _txx54_is_event_file_input(obj: object) -> bool:
    if isinstance(obj, (str, _Path)):
        p = _Path(obj).expanduser()
        return p.exists() and p.suffix.lower() in {'.evt', '.fits', '.fit'}
    return False


def _txx54_is_event_data_input(obj: object) -> bool:
    return isinstance(obj, EventDataBase)


def _txx54_read_evt_file(path: _Path) -> dict:
    from astropy.io import fits
    from .io import read_evt

    # Use the same TIMEZERO/relative-GTI contract as object inputs.
    event = _txx54_read_event_object(read_evt(path), arg_name=str(path))

    with fits.open(path, memmap=True) as hdul:
        if 'EVENTS' not in hdul:
            raise ValueError(f"事件文件缺少 EVENTS 扩展: {path}")

        evt_hdu = hdul['EVENTS']
        if evt_hdu.data is None or 'TIME' not in evt_hdu.columns.names:
            raise ValueError(f"EVENTS 扩展缺少 TIME 列: {path}")

        backscal = None
        backscal_raw = evt_hdu.header.get('BACKSCAL', None)
        if backscal_raw is not None:
            try:
                b = float(backscal_raw)
                if np.isfinite(b) and b > 0.0:
                    backscal = b
            except Exception:
                backscal = None

        area = None
        if 'REG00101' in hdul and hdul['REG00101'].data is not None and len(hdul['REG00101'].data) > 0:
            reg = hdul['REG00101'].data[0]
            try:
                shape = reg['SHAPE']
                if isinstance(shape, (bytes, bytearray)):
                    shape = shape.decode(errors='ignore')
                shape_u = str(shape).strip().upper()
                r = np.asarray(reg['R'], dtype=float).reshape(-1)
                if shape_u.startswith('CIRCLE') and r.size >= 1:
                    area = float(np.pi * r[0] ** 2)
                elif shape_u.startswith('ANNULUS') and r.size >= 2:
                    area = float(np.pi * (r[1] ** 2 - r[0] ** 2))
            except Exception:
                area = None

        # 无 region 时，退化为把 BACKSCAL 作为面积代理
        if area is None and backscal is not None:
            area = backscal

    event.update(area=area, backscal=backscal)
    return event


def _txx54_read_event_object(ev: EventDataBase, *, arg_name: str) -> dict:
    """将 EventDataBase/EventData 统一抽取为 txx 内部事件字典。"""
    meta = getattr(ev, 'meta', None)
    header = getattr(ev, 'header', {}) or {}
    from .io import _combine_mjdref
    column_unit = None
    for key,value in header.items():
        if str(key).startswith('TTYPE') and str(value).upper() == 'TIME':
            column_unit = header.get('TUNIT'+str(key)[5:])
    unit = getattr(meta, 'timeunit', None) or header.get('TIMEUNIT') or column_unit or 's'
    if str(unit).strip().lower() not in {'s', 'sec', 'second', 'seconds'}:
        raise ValueError(f"{arg_name}: TIMEUNIT={unit!r}; txx requires seconds")
    try:
        if hasattr(ev, 'absolute_time'):
            times = np.asarray(getattr(ev, 'absolute_time'), dtype=float)
        else:
            time_raw = np.asarray(getattr(ev, 'time', None), dtype=float)
            tz = float(getattr(ev, 'timezero', 0.0) or 0.0)
            times = time_raw + tz
    except Exception as exc:
        raise ValueError(f"{arg_name}: 读取事件时间失败") from exc

    times = np.asarray(times, dtype=float)
    times = times[np.isfinite(times)]
    if times.size == 0:
        raise ValueError(f"{arg_name}: 事件数据为空，无法计算 Txx")

    gti_start = None
    gti_stop = None
    gti_s = getattr(ev, 'gti_start', None)
    gti_e = getattr(ev, 'gti_stop', None)
    if gti_s is not None and gti_e is not None:
        gs = np.asarray(gti_s, dtype=float).reshape(-1)
        ge = np.asarray(gti_e, dtype=float).reshape(-1)
        if gs.size != ge.size or np.any(~np.isfinite(gs)) or np.any(~np.isfinite(ge)) or np.any(ge <= gs):
            raise ValueError(f'{arg_name}: invalid GTI')
        if gs.size:
            tz = float(getattr(ev, 'timezero', 0.0) or 0.0)
            gti_start, gti_stop = gs+tz, ge+tz

    if gti_start is None or gti_stop is None:
        raw_zero = float(getattr(meta, 'timezero', None) or header.get('TIMEZERO', 0.0))
        start = getattr(meta, 'tstart', None)
        stop = getattr(meta, 'tstop', None)
        start = header.get('TSTART') if start is None else start
        stop = header.get('TSTOP') if stop is None else stop
        gti_start = np.asarray([float(start)+raw_zero if start is not None else float(np.min(times))])
        gti_stop = np.asarray([float(stop)+raw_zero if stop is not None else float(np.max(times))])

    backscal = _txx54_positive_scalar(getattr(ev, 'backscal', None))
    if backscal is None:
        try:
            backscal = _txx54_positive_scalar(getattr(ev, 'get_keyword_ci')('BACKSCAL', None))
        except Exception:
            backscal = None
    if backscal is None:
        hdr = getattr(ev, 'header', None)
        if isinstance(hdr, dict):
            backscal = _txx54_positive_scalar(hdr.get('BACKSCAL', hdr.get('backscal', None)))

    ev_path = getattr(ev, 'path', None)
    return {
        'path': str(ev_path) if ev_path is not None else f"<{type(ev).__name__}>",
        'time': np.asarray(times, dtype=float),
        'gti_start': np.asarray(gti_start, dtype=float),
        'gti_stop': np.asarray(gti_stop, dtype=float),
        'area': None,
        'backscal': backscal,
        'time_reference': {
            'timesys': getattr(meta, 'timesys', None) or header.get('TIMESYS'),
            'mjdref': getattr(meta, 'mjdref', None) if getattr(meta, 'mjdref', None) is not None else _combine_mjdref(header),
            'trefpos': getattr(meta, 'trefpos', None) or header.get('TREFPOS'),
        },
    }


def _txx54_event_input_to_dict(obj: object, *, arg_name: str) -> dict:
    if isinstance(obj, (str, _Path)):
        p = _Path(obj).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"{arg_name}: 文件不存在: {p}")
        if p.suffix.lower() not in {'.evt', '.fits', '.fit', '.fts'}:
            raise TypeError(f"{arg_name}: 仅支持事件文件 (.evt/.fits/.fit/.fts)，当前={p}")
        return _txx54_read_evt_file(p)

    if _txx54_is_event_data_input(obj):
        return _txx54_read_event_object(cast(EventDataBase, obj), arg_name=arg_name)

    raise TypeError(
        f"{arg_name}: txx 仅支持事件文件路径或 EventDataBase/EventData 输入，当前={type(obj).__name__}"
    )


def _txx54_positive_scalar(v: object) -> Optional[float]:
    if v is None:
        return None
    try:
        arr = np.asarray(v, dtype=float).reshape(-1)
    except Exception:
        return None
    if arr.size == 0:
        return None
    val = float(np.nanmedian(arr))
    if not np.isfinite(val) or val <= 0.0:
        return None
    return val


def _txx54_bin_evt_to_array(evt: dict, binsize: float, t0: float, t1: float) -> np.ndarray:
    if not np.isfinite(t0) or not np.isfinite(t1) or t1 <= t0:
        raise ValueError(f"无效时间范围: t0={t0}, t1={t1}")
    if not np.isfinite(binsize) or binsize <= 0.0:
        raise ValueError(f"evt_binsize 必须为正数，当前={binsize}")

    edges = _duration_fixed_edges(float(t0), float(t1), float(binsize))

    hist, _ = np.histogram(np.asarray(evt['time'], dtype=float), bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return np.column_stack([centers.astype(float), hist.astype(float)])


def _txx54_convert_event_inputs(
    lc_src,
    background,
    alpha: Optional[float],
    *,
    evt_binsize: float,
) -> tuple[dict, Optional[dict], Optional[float]]:
    # evt_binsize 在事件直输模式下不参与输入转换；保留参数仅为兼容签名
    _ = float(evt_binsize)

    src_evt = _txx54_event_input_to_dict(lc_src, arg_name='lc_src')
    bkg_evt = None if background is None else _txx54_event_input_to_dict(background, arg_name='background')

    # 事件输入下，若未显式传 alpha，则优先用事件元信息估计
    if bkg_evt is not None and alpha is None:
        bs_src = _txx54_positive_scalar(src_evt.get('backscal', None))
        bs_bkg = _txx54_positive_scalar(bkg_evt.get('backscal', None))
        if bs_src is not None and bs_bkg is not None:
            alpha = float(bs_src / bs_bkg)
        else:
            a_src = src_evt.get('area', None)
            a_bkg = bkg_evt.get('area', None)
            if a_src is not None and a_bkg is not None and np.isfinite(a_src) and np.isfinite(a_bkg) and a_bkg > 0:
                alpha = float(a_src / a_bkg)

    return src_evt, bkg_evt, alpha


def _duration_fixed_edges(start: float, stop: float, width: float) -> np.ndarray:
    """Uniform edges in seconds, with a possibly short final bin; never overshoot."""
    if not (np.isfinite(start) and np.isfinite(stop) and stop > start
            and np.isfinite(width) and width > 0):
        raise ValueError('Invalid duration interval or evt_binsize')
    return np.r_[start, np.arange(start + width, stop, width), stop]


def _duration_event_blocks(times: np.ndarray, start: float, stop: float, p0: float) -> np.ndarray:
    """Event BB edges on a known observing interval (same units as times).

    Astropy uses the first/last event as its outside edges. Replace those two
    edges with the observation boundaries, rather than adding artificial end
    blocks containing only the first or last photon.
    """
    if times.size < 2 or np.unique(times).size < 2:
        return np.asarray([start, stop])
    edges = np.asarray(bayesian_blocks(times, fitness='events', p0=p0), float)
    return np.r_[start, edges[1:-1], stop]


def _duration_reference_boundary_options(overrides: Optional[dict] = None) -> dict:
    """Reference WXT edge-refinement settings; every value is in seconds.

    Defaults reproduce config.json and the two fixed snapping distances in
    wuqinyu/EFXT_WXT_data_processing at dd5a3f55c7c60d56901a2de29860bd8eeceb2d0c.
    The returned fresh dictionary can be recorded with each result.
    """
    options = dict(dt0_threshold=1., next_event_gap=100., short_gap_1=30.,
                   short_diff_1=15., short_gap_2=100., short_diff_2=25.,
                   all_gap=200., src_gap=400., added_edge_distance=25.)
    if overrides is not None:
        unknown = set(overrides) - options.keys()
        if unknown:
            raise ValueError('Unknown reference boundary option(s): ' + ', '.join(sorted(unknown)))
        options.update(overrides)
    for key, value in options.items():
        value = float(value)
        if not np.isfinite(value) or value < 0:
            raise ValueError(f'Reference boundary option {key} must be finite and nonnegative seconds')
        options[key] = value
    return options


def _duration_reference_edges(on_times: np.ndarray, off_times: np.ndarray,
                              raw_edges: np.ndarray, *, options: dict,
                              additional_edges: Optional[np.ndarray] = None) -> np.ndarray:
    """Refine physical BB edges using the reference WXT event rules.

    All arrays and options are seconds in one common coordinate. Event arrays
    must be sorted. Extra edges are explicit inputs, never read from a file.
    This preserves the reference nearest-event tie policy, two isolated-source
    snapping passes, combined-event endpoint/gap additions and final snap pass.
    It does not add GTI edges or re-optimize the BB fitness after moving edges.

    Reference: wxt_pipeline/lc_analysis.py:34--122, commit
    dd5a3f55c7c60d56901a2de29860bd8eeceb2d0c, wuqinyu/EFXT_WXT_data_processing.
    """
    on = np.asarray(on_times, float)
    off = np.asarray(off_times, float)
    if on.ndim != 1 or off.ndim != 1 or not on.size:
        raise ValueError('Reference edge refinement requires a nonempty ON event array')
    if np.any(~np.isfinite(on)) or np.any(~np.isfinite(off)) or np.any(np.diff(on) < 0) or np.any(np.diff(off) < 0):
        raise ValueError('Reference event times must be finite and sorted')
    edges = np.asarray(raw_edges, float)
    if additional_edges is not None:
        edges = np.r_[edges, np.asarray(additional_edges, float)]
    if edges.ndim != 1 or np.any(~np.isfinite(edges)):
        raise ValueError('Reference block edges must be finite and one-dimensional')
    snapped = []
    for edge in np.sort(edges):
        nearest = on[np.abs(on-edge).argmin()]
        after = np.searchsorted(on, edge)
        next_gap = on[after]-edge if after < on.size else np.inf
        snapped.append(nearest if abs(edge-nearest) <= options['dt0_threshold']
                       or next_gap > options['next_event_gap'] else on[after])
    snapped = np.asarray(snapped, float)

    def isolated(events, gap, *, include_ends=False):
        middle = events[1:-1][(np.diff(events)[:-1] >= gap) | (np.diff(events)[1:] >= gap)]
        return np.r_[events[:1], middle, events[-1:]] if include_ends else middle

    def snap_to_candidates(candidates, distance):
        if not candidates.size:
            return
        for i, edge in enumerate(snapped):
            diffs = np.abs(candidates-edge)
            eligible = np.flatnonzero(diffs <= distance)
            if eligible.size:
                snapped[i] = candidates[eligible[np.argmin(diffs[eligible])]]

    for suffix in ('1', '2'):
        snap_to_candidates(isolated(on, options['short_gap_'+suffix]),
                           options['short_diff_'+suffix])
    combined = np.sort(np.r_[on, off])
    lonely_all = isolated(combined, options['all_gap'], include_ends=True)
    added = np.asarray([on[np.abs(on-edge).argmin()] for edge in lonely_all], float)
    added = np.r_[added, isolated(on, options['src_gap'])]
    snap_to_candidates(added, options['added_edge_distance'])
    return np.unique(np.r_[snapped, added])


def _duration_plateau(curve_time: np.ndarray, cumulative: np.ndarray,
                      interval: tuple[float, float]) -> tuple[float, float, int]:
    """Mean and sample *scatter variance* of cumulative samples, not mean error."""
    a, b = interval
    use = (curve_time >= a) & (curve_time <= b)
    vals = cumulative[use]
    if vals.size < 2:
        return np.nan, np.nan, int(vals.size)
    return float(np.mean(vals)), float(np.var(vals, ddof=1)), int(vals.size)


def _duration_koshut(curve_time: np.ndarray, cumulative: np.ndarray,
                     on: np.ndarray, off: np.ndarray, alpha: float,
                     levels: tuple[float, float], variances: tuple[float, float],
                     fractions: np.ndarray, nominal: np.ndarray,
                     signal_range: Optional[tuple[float, float]] = None) -> dict:
    """Koshut (1996) error prescription adapted to ON/OFF and a fixed window.

    Time inputs/outputs are seconds, levels/counts are counts. The extension
    to measured OFF data uses N_on + alpha**2 N_off in Eq. 9. The paper assumes
    a known background model. Linear interpolation and the battblocks midpoint
    convention replace BATSE's bin-edge rounding. Delta-t is the FULL width
    between S_f +/- sigma, not half that width. Uncrossed thresholds stay NaN.
    With signal_range, levels are the cumulative values at its boundaries,
    and Eq. 9 count variance is limited to that same window. Plateau scatter
    remains an error-model assumption; this is not the paper's plateau-mean
    definition of the nominal fluence. Sigma crossings use all observed data.
    """
    lz, lt = levels
    vz, vt = variances
    nodes = np.ones(curve_time.size, dtype=bool)
    bins = np.ones(on.size, dtype=bool)
    if signal_range is not None:
        a, z = signal_range
        nodes = (curve_time >= a) & (curve_time <= z)
        bins = (curve_time[1:] > a) & (curve_time[:-1] < z)
    above = np.flatnonzero(nodes & (cumulative > lz))
    tau0 = float(curve_time[above[0]]) if above.size else np.nan
    sigma = np.full(fractions.size, np.nan)
    crossings = np.full((fractions.size, 2), np.nan)
    widths = np.full(fractions.size, np.nan)
    count_variances = np.full(fractions.size, np.nan)
    status = []
    for i, (q, tau) in enumerate(zip(fractions, nominal)):
        if not np.all(np.isfinite([lz, lt, vz, vt, tau0, tau])):
            status.append('missing_plateau_or_nominal_crossing')
            continue
        # Eq. 9 sums discrete observed-bin variances, including the crossing
        # bins. Do not fractionally scale Poisson variances as counts**2.
        use = bins & (curve_time[1:] >= tau0) & (curve_time[:-1] <= tau)
        count_var = float(np.sum((on + alpha**2 * off)[use]))
        count_variances[i] = count_var
        sigma[i] = np.sqrt(count_var + (1-q)**2 * vz + q**2 * vt)
        target = (1-q)*lz + q*lt
        for j, level in enumerate((target-sigma[i], target+sigma[i])):
            try:
                crossings[i, j] = _crossing_midpoint(curve_time, cumulative, level)
            except ValueError:
                pass
        if np.all(np.isfinite(crossings[i])) and crossings[i, 1] >= crossings[i, 0]:
            widths[i] = crossings[i, 1] - crossings[i, 0]
            status.append('ok')
        else:
            status.append('uncrossed_or_reversed_sigma_threshold')
    return dict(fractions=fractions, sigma_counts=sigma, crossing_times=crossings,
                crossing_full_width=widths, status=status, tau0=tau0,
                count_variances=count_variances, fluence_levels=levels,
                threshold_counts=(1-fractions)*lz+fractions*lt,
                signal_range=signal_range,
                crossing_search_range=np.asarray([curve_time[0], curve_time[-1]]))


def txx(
    lc_src: 'EventDataBase | str | _Path',
    background: Optional['EventDataBase | str | _Path'] = None,
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
    evt_binsize: float = 1.0,
    cumulative_mode: Literal['adaptive', 'fixed'] = 'adaptive',
    block_snr_threshold: float = 3.0,
    plateau_intervals: Optional[Sequence[Sequence[float]]] = None,
    reference_boundary_options: Optional[dict[str, float]] = None,
    additional_block_edges: Optional[Sequence[float]] = None,
    **kwargs,
) -> dict:
    """Event window with reference WXT edge rules and signed T50/T90, in seconds.

    OFF-event Bayesian blocks estimate a strictly positive piecewise background
    rate. Source event times are transformed by its integral (De Luca et al.
    2021, Sect. 5.4), segmented, and mapped back. The physical edges are refined
    by the event snapping/isolation rules of EFXT_WXT_data_processing, commit
    dd5a3f55c7c60d56901a2de29860bd8eeceb2d0c. Signed Li & Ma block significance
    greater than or equal to the threshold selects the activity window.
    The reference empirical-CDF transform is not enabled by this version.
    Identical,
    continuous ON/OFF GTIs and constant relative exposure/area alpha are required.
    Fractional exposure or instrument-response changes require a different model.

    T100 is the selected first-to-last significant BB interval (or explicit
    burst_tstart/stop). Signed ON-alpha*OFF cumulative counts are normalized
    from zero at its start to the total net counts at its stop. All nominal
    percentile crossings are searched only inside that same T100, so the
    T50 interval is contained in T90, which is contained in T100. BB segments
    are used in ``adaptive`` mode, or bounded evt_binsize bins in ``fixed`` mode.
    Pre/post cumulative plateaus use uniform evt_binsize samples. Supply two
    emission-free absolute-second intervals via ``plateau_intervals``; otherwise
    the intervals outside the selected BB window are automatic candidates and
    need scientific inspection. Plateau means are diagnostics, not nominal
    fluence levels. The Koshut (1996) Eqs. 8--16 error prescription is adapted
    to these window boundary levels, plateau scatter, and the independent
    measured-OFF variance. It is a conditional error-model approximation,
    with linear interpolation rather than BATSE bin rounding; it is not the
    paper's plateau-mean nominal estimator, a calibrated 68% interval, or a
    total systematic error. Sigma crossings can extend outside T100.

    ``nmc`` controls an additional parametric Poisson bootstrap: fit ON and OFF
    piecewise rates over the full observation, redraw photon arrivals, then
    rerun background fitting, time transform, BB selection, plateaus and the SAME
    cumulative estimator. Quantiles, failures and boundary hits are diagnostics
    under that fitted model, never folded into Koshut errors. nmc=0 skips only
    this bootstrap. Missing sigma crossings/errors remain NaN with status.

    The historical method identifier and array shapes are retained for callers;
    inspect implementation_version/error_method for the changed semantics.

    ``reference_boundary_options`` overrides the reference snapping distances
    and isolation gaps (all in seconds); resolved defaults are saved in output.
    ``additional_block_edges`` provides explicit absolute-second edges, which
    undergo the same refinement as reference user-added edges. No implicit
    user-edge file or reference checkout is read by this function.
    """
    if kwargs:
        raise TypeError('txx() got unexpected keyword argument(s): ' + ', '.join(sorted(kwargs)))
    if use_edge_bkg or any(v is not None for v in (lbkg, rbkg, tpeak, timebins)):
        raise ValueError('Legacy edge/peak/timebins options are unsupported; use plateau_intervals or burst_tstart/stop')
    if src_dist != 'poisson' or bkg_dist != 'poisson' or window_mode != 'auto':
        raise ValueError('Event durations require poisson distributions and window_mode="auto"')
    if (small_bin_threshold != 4.0 or tuple(weak_peak_bins) != (8.0,16.0,32.0)
            or weak_peak_weight != .2 or density_quantile != 60.0):
        raise ValueError('Legacy weak-peak/density tuning is unsupported by this event estimator')
    if not (np.isfinite(p0) and 0 < p0 < 1):
        raise ValueError('p0 must be in (0,1)')
    if not (np.isfinite(block_snr_threshold) and block_snr_threshold > 0):
        raise ValueError('block_snr_threshold must be positive and finite')
    if not (np.isfinite(evt_binsize) and evt_binsize > 0):
        raise ValueError('evt_binsize must be positive and finite')
    if cumulative_mode not in {'adaptive', 'fixed'}:
        raise ValueError('cumulative_mode must be adaptive or fixed')
    boundary_options = _duration_reference_boundary_options(reference_boundary_options)
    if int(nmc) != nmc or nmc < 0:
        raise ValueError('nmc must be a nonnegative integer')
    requested = np.atleast_1d(np.asarray(percent, float))
    if requested.ndim != 1 or requested.size == 0 or np.any(~np.isfinite(requested)) or np.any((requested <= 0) | (requested >= 1)):
        raise ValueError('percent must contain finite fractions in (0,1)')
    perc = np.unique(np.r_[requested, .5, .9])
    fractions = np.column_stack(((1-perc)/2, (1+perc)/2)).ravel()
    src, bkg, alpha = _txx54_convert_event_inputs(lc_src, background, alpha, evt_binsize=evt_binsize)
    from .gti import merge_gti

    def bounds(evt):
        gs, ge = np.asarray(evt['gti_start'], float), np.asarray(evt['gti_stop'], float)
        if gs.shape != ge.shape or not gs.size or np.any(~np.isfinite(gs)) or np.any(~np.isfinite(ge)) or np.any(ge <= gs):
            raise ValueError('Invalid GTI')
        gs, ge = merge_gti(gs, ge)
        if gs.size != 1:
            raise ValueError('GTI 存在空洞; restrict inputs to one fully observed interval before measuring duration')
        return np.asarray([gs[0], ge[0]])

    limits = bounds(src)
    if bkg is not None:
        if not np.allclose(limits, bounds(bkg), rtol=0, atol=1e-7):
            raise ValueError('ON/OFF GTI coverage differs; explicitly select identical continuous GTIs')
        for key in ('timesys', 'mjdref', 'trefpos'):
            a, b = src['time_reference'].get(key), bkg['time_reference'].get(key)
            if a is not None and b is not None:
                equal = np.isclose(a, b, rtol=0, atol=1e-10) if key == 'mjdref' else str(a).upper() == str(b).upper()
                if not equal:
                    raise ValueError(f'ON/OFF time reference differs: {key}')
        alpha = _txx54_positive_scalar(alpha)
        if alpha is None:
            raise ValueError('background requires alpha or BACKSCAL/region area metadata')
    else:
        alpha = 0.0
    t0, t1 = map(float, limits)
    # Relative internal time improves precision for mission seconds.
    span = t1-t0
    extra_bb_edges = np.array([]) if additional_block_edges is None else np.asarray(additional_block_edges, float)-t0
    if extra_bb_edges.ndim != 1 or np.any(~np.isfinite(extra_bb_edges)) or np.any((extra_bb_edges < 0)|(extra_bb_edges > span)):
        raise ValueError('additional_block_edges must be finite absolute seconds inside the GTI')
    on_times = np.sort(np.asarray(src['time'], float) - t0)
    off_times = np.sort(np.asarray(bkg['time'], float) - t0) if bkg is not None else np.array([])
    on_times = on_times[(on_times >= 0) & (on_times <= span)]
    off_times = off_times[(off_times >= 0) & (off_times <= span)]
    if on_times.size == 0:
        raise ValueError('No source events inside GTI')
    if bkg is not None and off_times.size == 0:
        raise ValueError('No OFF photons; cannot infer a positive background time transform')
    manual_window = None
    if burst_tstart is not None or burst_tstop is not None:
        if burst_tstart is None or burst_tstop is None:
            raise ValueError('Provide both burst_tstart and burst_tstop')
        manual_window = (float(burst_tstart)-t0, float(burst_tstop)-t0)
        if not (0 <= manual_window[0] < manual_window[1] <= span):
            raise ValueError('burst_tstart/stop must lie inside the GTI')
    plateaus = None
    if plateau_intervals is not None:
        plateaus = np.asarray(plateau_intervals, float) - t0
        if plateaus.shape != (2,2) or np.any(~np.isfinite(plateaus)) or np.any(plateaus[:,1] <= plateaus[:,0]) or np.any(plateaus < 0) or np.any(plateaus > span):
            raise ValueError('plateau_intervals requires two valid intervals inside the GTI')

    def estimate(on_t, off_t, *, errors):
        # ML piecewise OFF rate; no zero-rate prior or extrapolated coverage.
        bg_edges = _duration_event_blocks(off_t, 0., span, p0) if bkg is not None else np.array([0., span])
        bg_counts = np.histogram(off_t, bg_edges)[0].astype(float)
        bg_rate = bg_counts / np.diff(bg_edges) if bkg is not None else np.ones(1)
        if np.any(bg_rate <= 0):
            raise ValueError('Background model has a zero-rate block; time transform is not invertible')
        transform_rate = alpha*bg_rate if bkg is not None else bg_rate
        prime_edges = np.r_[0., np.cumsum(transform_rate*np.diff(bg_edges))]
        prime = np.interp(on_t, bg_edges, prime_edges)
        bb_prime = _duration_event_blocks(prime, 0., prime_edges[-1], p0)
        raw_bb = np.interp(bb_prime, prime_edges, bg_edges)
        bb = _duration_reference_edges(on_t, off_t, raw_bb, options=boundary_options,
                                       additional_edges=extra_bb_edges)
        if bb.size < 2:
            if manual_window is None:
                raise RuntimeError('Reference edge refinement has no positive-width block; provide an explicit signal interval')
            bb = np.asarray(manual_window)
        # Reference T100 selection counts every candidate block as [left,right),
        # including the final candidate block. Nominal cumulative policy stays
        # unchanged: only the full observation's final endpoint is right-closed.
        on_bb = np.diff(np.searchsorted(on_t, bb, side='left')).astype(float)
        off_bb = np.diff(np.searchsorted(off_t, bb, side='left')).astype(float)
        if bkg is not None:
            snr = np.array([li_ma_snr(float(n),float(m),alpha) for n,m in zip(on_bb,off_bb)])
            snr = np.where(on_bb < alpha*off_bb, -np.abs(snr), snr)
        else:
            snr = np.sqrt(on_bb)
        sig = np.flatnonzero(snr >= block_snr_threshold)
        if manual_window is None:
            if not sig.size:
                raise RuntimeError('没有任何贝叶斯块的 SNR >= threshold')
            a, z = float(bb[sig[0]]), float(bb[sig[-1]+1])
        else:
            a, z = manual_window
        adaptive_edges = np.unique(np.r_[bb,bg_edges])
        seg = np.r_[a, adaptive_edges[(adaptive_edges>a)&(adaptive_edges<z)], z] if cumulative_mode == 'adaptive' else _duration_fixed_edges(a,z,evt_binsize)
        pi = np.array([[0., a], [z, span]]) if plateaus is None else plateaus
        if pi[0,1] > a or pi[1,0] < z:
            raise ValueError('Plateau intervals must precede/follow the selected activity window')
        # Uniform background samples preserve plateau fluctuations even when
        # adaptive signal segments contain only a single BB.
        pre = _duration_fixed_edges(0., a, evt_binsize) if a > 0 else np.array([0.])
        post = _duration_fixed_edges(z, span, evt_binsize) if z < span else np.array([span])
        extra = pi.ravel() if plateaus is not None else np.array([])
        edges = np.unique(np.r_[pre, seg, post, extra])
        on_c = np.histogram(on_t, edges)[0].astype(float)
        off_c = np.histogram(off_t, edges)[0].astype(float)
        curve_t, cum = _signed_cumulative_curve(edges[:-1], edges[1:], on_c, alpha*off_c, (0.,span))
        lz,vz,nz = _duration_plateau(curve_t,cum,tuple(pi[0]))
        lt,vt,nt = _duration_plateau(curve_t,cum,tuple(pi[1]))
        missing = nz < 2 or nt < 2
        # Reuse the same signed-curve helper as the other duration estimators.
        # Both normalization and crossings must use the returned T100 support.
        signal_t, signal_cum = _signed_cumulative_curve(
            edges[:-1], edges[1:], on_c, alpha*off_c, (a,z))
        total = float(signal_cum[-1])
        if not np.isfinite(total) or total <= 0:
            raise ValueError('Nonpositive signed total fluence inside T100')
        levels = (float(np.interp(a,curve_t,cum)), float(np.interp(z,curve_t,cum)))
        times = np.full(fractions.size,np.nan)
        for i,q in enumerate(fractions):
            try:
                times[i] = _crossing_midpoint(signal_t,signal_cum,q*total)
            except ValueError:
                pass
        pairs = times.reshape(-1,2)
        durations = pairs[:,1]-pairs[:,0]
        durations[durations <= 0] = np.nan
        if not np.any(np.isfinite(durations)):
            raise ValueError('No ordered percentile crossings in the observed cumulative curve')
        ordered = times[np.argsort(fractions)]
        finite = ordered[np.isfinite(ordered)]
        tol = 32*np.finfo(float).eps*max(span,1.)
        if (np.any(finite < a-tol) or np.any(finite > z+tol)
                or np.any(np.diff(finite) < -tol)):
            raise RuntimeError('Percentile intervals are not nested inside T100')
        kh = _duration_koshut(curve_t,cum,on_c,off_c,alpha,levels,(vz,vt),
                              fractions,times,signal_range=(a,z)) if errors else None
        return dict(bb=bb,bb_prime=np.interp(bb,bg_edges,prime_edges),raw_bb=raw_bb,
                    raw_bb_prime=bb_prime,snr=snr,off_bb=off_bb,window=(a,z),seg=seg,
                    bg_edges=bg_edges,bg_rate=bg_rate,edges=edges,on=on_c,off=off_c,
                    curve=cum,levels=levels,signal_curve=signal_cum,
                    plateau_levels=(lz,lt),variances=(vz,vt),plateaus=pi,
                    plateau_missing=missing,durations=durations,pairs=pairs,koshut=kh)

    fit = estimate(on_times,off_times,errors=True)
    kh = fit['koshut']
    widths = kh['crossing_full_width'].reshape(-1,2)
    stat = np.repeat(np.sqrt(np.sum(widths**2,axis=1))[:,None],2,axis=1)
    # Start/stop legacy error pairs each contain Delta-t, i.e. full crossing
    # width, not +/- half-width confidence limits. Raw crossings are in koshut.
    endpoint_stat = np.repeat(widths[:,:,None],2,axis=2)
    nan_err = np.full_like(stat,np.nan)
    mc = dict(requested=int(nmc),valid=0,failed=0,seed=seed,
              method='full_observation_parametric_poisson_bootstrap',
              confidence_status='model_sensitivity_not_calibrated_confidence',failure_reasons={})
    if nmc:
        rng = np.random.default_rng(seed)
        # BB is fitted in transformed time. Preserve that fitted ON intensity
        # when the physical background rate changes within an ON block.
        on_model_edges = np.unique(np.r_[fit['bb'],fit['bg_edges']])
        raw_bg_integral = np.r_[0.,np.cumsum(fit['bg_rate']*np.diff(fit['bg_edges']))]
        on_prime = np.interp(on_model_edges,fit['bg_edges'],raw_bg_integral)
        bb_prime = np.interp(fit['bb'],fit['bg_edges'],raw_bg_integral)
        indices = np.searchsorted(fit['bb'],.5*(on_model_edges[:-1]+on_model_edges[1:]),side='right')-1
        bb_counts = np.histogram(on_times,fit['bb'])[0]
        # Refined reference edges lie at source events, rather than spanning
        # the GTI. Preserve the existing fitted-rate distribution inside them;
        # the photon-free outside intervals have zero ON expectation.
        inside = (indices >= 0)&(indices < bb_counts.size)
        on_model_counts = np.zeros(on_model_edges.size-1)
        on_model_counts[inside] = bb_counts[indices[inside]]*np.diff(on_prime)[inside]/np.diff(bb_prime)[indices[inside]]
        off_model_edges = fit['bg_edges']
        off_model_counts = np.histogram(off_times,off_model_edges)[0]
        mc['generating_model'] = dict(on_edges_time=on_model_edges+t0,on_expected_counts=on_model_counts,
                                     off_edges_time=off_model_edges+t0,off_expected_counts=off_model_counts,
                                     on_model='BB_constant_rate_in_transformed_time')
        def draw(edges,counts):
            ns = rng.poisson(counts)
            return np.sort(np.concatenate([rng.uniform(l,r,int(n)) for l,r,n in zip(edges[:-1],edges[1:],ns)]))
        ds,ws,ps = [],[],[]
        for _ in range(int(nmc)):
            try:
                sample = estimate(draw(on_model_edges,on_model_counts),draw(off_model_edges,off_model_counts) if bkg is not None else np.array([]),errors=False)
                ds.append(sample['durations']); ws.append(sample['window']); ps.append(sample['pairs'])
            except (ValueError,RuntimeError) as exc:
                reason = str(exc)
                mc['failure_reasons'][reason] = mc['failure_reasons'].get(reason,0)+1
        mc['valid'] = len(ds); mc['failed'] = int(nmc)-len(ds)
        if ds:
            ds,ws,ps = np.asarray(ds),np.asarray(ws),np.asarray(ps)
            mc.update(percent=perc,duration_samples=ds,window_samples=ws+t0,
                      endpoint_samples=ps+t0,valid_per_percent=np.sum(np.isfinite(ds),axis=0),
                      boundary_hits=int(np.sum((ws[:,0] == 0)|(ws[:,1] == span))))
            quantiles = np.full((3,perc.size),np.nan)
            for j in range(perc.size):
                good = ds[:,j][np.isfinite(ds[:,j])]
                if good.size:
                    quantiles[:,j] = np.percentile(good,[16,50,84])
            mc['duration_quantiles_16_50_84'] = quantiles
            mc['nominal_inside_16_84'] = (fit['durations'] >= quantiles[0])&(fit['durations'] <= quantiles[2])
    # Keep chosen-bin comparisons separate from statistical errors.
    sensitivity = []
    a,z = fit['window']
    for scale in (.5,1.,2.):
        e = _duration_fixed_edges(a,z,evt_binsize*scale)
        e = np.unique(np.r_[fit['edges'][fit['edges'] < a],e,fit['edges'][fit['edges'] > z]])
        on_c = np.histogram(on_times,e)[0]
        off_c = alpha*np.histogram(off_times,e)[0]
        vals = []
        for p in perc:
            try:
                x = _quantile_interval(e[:-1],e[1:],on_c,off_c,(a,z),p)
                vals.append(x[1]-x[0] if x[1] >= x[0] else np.nan)
            except ValueError:
                vals.append(np.nan)
        sensitivity.append(vals)
    pick = np.array([np.flatnonzero(np.isclose(perc,p,rtol=0,atol=1e-12))[0] for p in requested])
    out = dict(method='aanda_2021_sec5_4',implementation_version='extras_timewarp_reference_edges_signed_koshut_onoff_v4',
               method_description='Fitted OFF integral transform + reference WXT refined T100 edges + signed percentiles + Koshut adaptation',
               error_method='koshut_1996_eq8_16_window_onoff_adaptation',error_scope='conditional_on_window_and_plateau_scatter_model',
               confidence_status='not_calibrated_68_percent',systematic_error_status='not_estimated',
               negative_net_policy='signed',crossing_policy='first_last_midpoint_linear',
               alpha=alpha,evt_binsize=evt_binsize,percent=requested,
               t100=z-a,t100_tstart=a+t0,t100_tstop=z+t0,t100_err=np.full(2,np.nan),
               t100_definition='explicit_burst_interval' if manual_window is not None else 'first_last_significant_BB_edges',
               cumulative_search_range=np.asarray([a+t0,z+t0]),
               duration_within_activity_window=((fit['pairs'][:,0] >= a)&(fit['pairs'][:,1] <= z))[pick],
               cumulative_normalization='signed_net_counts_within_t100',
               cumulative_levels=fit['levels'],cumulative_signal_signed_counts=fit['signal_curve'],
               signal_total_net_counts=float(fit['signal_curve'][-1]),
               plateau_means_used_for_nominal=False,
               histogram_endpoint_policy='left_closed_right_open_except_GTI_stop',
               burst_tstart=a+t0,burst_tstop=z+t0,
               bb_edges_time=fit['bb']+t0,bb_edges_tprime=fit['bb_prime'],
               raw_bb_edges_time=fit['raw_bb']+t0,raw_bb_edges_tprime=fit['raw_bb_prime'],
               block_boundary_policy='reference_wxt_event_snapping_and_lonely_edges',
               block_boundary_reference='wuqinyu/EFXT_WXT_data_processing@dd5a3f55c7c60d56901a2de29860bd8eeceb2d0c',
               reference_boundary_options=boundary_options,
               additional_block_edges=extra_bb_edges+t0,
               block_snr_comparison='greater_equal',
               block_snr_endpoint_policy='left_closed_right_open_including_last_candidate',
               bb_block_snr=fit['snr'],bb_snr_threshold=block_snr_threshold,
               bb_block_bkg_model_counts=alpha*fit['off_bb'],
               background_time_model_edges=fit['bg_edges']+t0,
               background_time_model_rate=alpha*fit['bg_rate'] if bkg is not None else np.zeros(1),
               time_transform='integral_scaled_off_rate' if bkg is not None else 'identity_no_background',
               time_transform_units='expected_background_counts_in_on_region' if bkg is not None else 's',
               exposure_status='continuous_common_gti_constant_relative_exposure_assumed',
               input_time_reference=dict(source=src['time_reference'],off=bkg['time_reference'] if bkg is not None else None),
               cumulative_mode=cumulative_mode,cumulative_edges_time=fit['seg']+t0,
               cumulative_curve_time=fit['edges']+t0,cumulative_signed_counts=fit['curve'],
               cumulative_on_counts=fit['on'],cumulative_off_counts=fit['off'],
               plateau_intervals=fit['plateaus']+t0,plateau_levels=fit['plateau_levels'],
               plateau_variances=fit['variances'],plateau_source='explicit' if plateaus is not None else 'automatic_outside_BB_window',
               plateau_status='missing_no_koshut_error' if fit['plateau_missing'] else 'requires_emission_free_inspection',
               bootstrap=mc,binning_sensitivity=dict(percent=perc,widths=evt_binsize*np.array([.5,1,2]),
                                                    durations=np.asarray(sensitivity),confidence_status='diagnostic_only'))
    # Select full-observation histogram bins so an event at the internal T100
    # stop is not counted again as if it were the final observation endpoint.
    use=(fit['edges'][:-1] >= a)&(fit['edges'][1:] <= z)
    seg_on=fit['on'][use]
    seg_off=alpha*fit['off'][use]
    out.update(signal_on_counts=seg_on,signal_off_counts=fit['off'][use],
               background_model_counts=seg_off,background_rate=seg_off/np.diff(fit['seg']))
    kh['crossing_times'] += t0
    kh['tau0'] += t0
    kh['signal_range'] = np.asarray(kh['signal_range'])+t0
    kh['crossing_search_range'] += t0
    out['koshut'] = kh
    for key,values,err in [('txx',fit['durations'],stat),('txx1',fit['pairs'][:,0]+t0,endpoint_stat[:,0]),('txx2',fit['pairs'][:,1]+t0,endpoint_stat[:,1])]:
        out[key]=values[pick];out[key+'_err']=err[pick];out[key+'_err_stat']=err[pick];out[key+'_err_sys']=nan_err[pick]
    for name,p in [('t50',.5),('t90',.9)]:
        i=int(np.flatnonzero(np.isclose(perc,p,rtol=0,atol=1e-12))[0])
        out[name]=float(fit['durations'][i])
        out[name+'_err']=stat[i];out[name+'_err_stat']=stat[i];out[name+'_err_sys']=nan_err[i]
        for j,suffix in enumerate(('tstart','tstop')):
            key=name+'_'+suffix
            out[key]=float(fit['pairs'][i,j]+t0)
            out[key+'_err']=endpoint_stat[i,j];out[key+'_err_stat']=endpoint_stat[i,j];out[key+'_err_sys']=nan_err[i]
        out[name+'_error_status']='ok' if np.all(np.isfinite(stat[i])) else 'unresolved_koshut_crossings'
    return out


# ==================== 迭代背景自洽贝叶斯块法（burstcube 移植）====================
# 以下实现自 .iterblocks 合并而来；对 .ops 的导入均为函数内延迟导入以避免模块循环。
@dataclass
class IterativeBBResult:
    """迭代背景自洽贝叶斯块分析的完整产物。

    属性
    ----
    bb_index : 变点箱索引（块 ``i`` 覆盖原始箱 ``bb_index[i]:bb_index[i+1]``）
    block_left/right/rates : 贝叶斯块的时间边界与块率
    bkg_counts : 每原始箱的背景模型计数
    signal_range : 显著信号区间 ``(tstart, tstop)``（prominence 定界 + 缓冲区外推）
    peak : 信号区间内的峰时刻
    n_iterations / converged : 迭代收敛信息
    bkg_times : 最终用于拟合背景的时间区间
    """

    bb_index: np.ndarray
    block_left: np.ndarray
    block_right: np.ndarray
    block_rates: np.ndarray
    bkg_counts: np.ndarray
    signal_range: tuple[float, float]
    peak: float
    n_iterations: int
    converged: bool
    bkg_times: list[tuple[float, float]] = field(default_factory=list)


def _fit_poly_background_counts(
    times: np.ndarray,
    counts: np.ndarray,
    exposure: np.ndarray,
    intervals: Sequence[tuple[float, float]],
    order: int,
) -> np.ndarray:
    """在背景区间上按曝光加权拟合多项式背景率，外推到全部箱并换算为计数。

    权重取 ``sqrt(exposure)``：计数率的 Poisson 方差近似 ``rate/exposure``，
    加权最小二乘下等价于 ``w ∝ sqrt(exposure)``。区间内样本不足时自动降阶。
    """
    mask = np.zeros(times.size, dtype=bool)
    for a, b in intervals:
        mask |= (times >= a) & (times <= b)
    order = int(max(0, min(order, max(mask.sum() - 1, 0))))
    if mask.sum() == 0:
        # 无背景区间：退化为全段常数背景
        total = float(counts.sum())
        expo_tot = float(exposure.sum())
        rate0 = total / expo_tot if expo_tot > 0 else 0.0
        return np.full(times.size, rate0) * exposure
    rate = counts[mask] / exposure[mask]
    w = np.sqrt(np.maximum(exposure[mask], 1e-12))
    coef = np.polynomial.polynomial.polyfit(times[mask], rate, order, w=w)
    bkg_rate = np.polynomial.polynomial.polyval(times, coef)
    return np.maximum(bkg_rate, 0.0) * exposure


# 方法：迭代背景自洽贝叶斯块：每轮在信号排除区做曝光加权多项式背景拟合 → 以
#       "曝光:=背景计数"的 Giacomo 技巧运行贝叶斯块（把含背景的 Poisson 过程化为
#       近似齐次 Poisson，等效率恒定）→ 块率峰的 prominence 定信号区，向外排除
#       buffer_blocks 个边缘块宽（不超过到光变端距离之半）作背景区 → 收敛判据为
#       背景区间重复（含 2-3 周期近似振荡的循环检测）。
# 参考：移植自 HEASoft 6.37 BurstCube GDT burstcube/lib/bayesian_lc.py
#       （BayesianBlocksLightcurve.compute_bayesian_blocks；本仓库
#       external_sources/heasoft-6.37/ 同路径可查）；Giacomo 技巧见 threeML
#       utils/bayesian_blocks.py (G. Vianello)，github.com/threeML/threeML
#       e31db70 .../threeML/utils/bayesian_blocks.py#L171；贝叶斯块本身见
#       Scargle, Norris, Jackson & Chiang 2013, ApJ 764, 167 (arXiv:1207.5578)。
def iterative_bayesian_blocks(
    left: np.ndarray,
    right: np.ndarray,
    counts: np.ndarray,
    *,
    exposure: Optional[np.ndarray] = None,
    bkg_counts_fixed: Optional[np.ndarray] = None,
    p0: float = 0.05,
    poly: int = 2,
    buffer_blocks: int = 5,
    max_iter: int = 100,
    signal_range: Optional[tuple[float, float]] = None,
) -> IterativeBBResult:
    """迭代背景自洽的贝叶斯块分析（核心算法，对应原 ``compute_bayesian_blocks``）。

    参数
    ----
    left / right / counts : 原始箱边界与计数。
    exposure : 逐箱有效曝光；默认取箱宽。
    bkg_counts_fixed : 若提供（如已有背景模型×alpha），背景不再拟合，
        仅迭代信号区间；否则在信号排除区间上拟合 ``poly`` 阶多项式背景。
    p0 / poly / buffer_blocks / max_iter : 同原实现语义。
    signal_range : 可选的信号区间初值。

    返回 :class:`IterativeBBResult`。
    """
    from scipy.signal import argrelmax, peak_prominences

    from .ops import bayesian_blocks_exposure

    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    counts = np.asarray(counts, dtype=float)
    n = counts.size
    if n == 0:
        raise ValueError("输入光变为空")
    times = 0.5 * (left + right)
    expo = np.asarray(exposure, dtype=float) if exposure is not None else (right - left)
    t_lo, t_hi = float(left[0]), float(right[-1])

    # 初始背景区间：全域，或由信号初值切成左右两段
    if signal_range is None:
        bkg_times = [(t_lo, t_hi)]
    else:
        s0, s1 = sorted(float(x) for x in signal_range)
        bkg_times = [(t_lo, s0), (s1, t_hi)]

    previous_bkg_times: list[list[tuple[float, float]]] = []
    converged = False
    iteration = 0
    bb_index = np.asarray([0, n], dtype=int)
    bkg_counts = np.zeros(n)
    block_left = np.asarray([t_lo])
    block_right = np.asarray([t_hi])
    block_rates = np.asarray([float(counts.sum()) / max(float(expo.sum()), 1e-12)])
    signal_tstart, signal_tstop = t_lo, t_hi

    for iteration in range(int(max_iter)):
        # ---- 1. 背景模型 ----
        if bkg_counts_fixed is not None:
            bkg_counts = np.asarray(bkg_counts_fixed, dtype=float)
        else:
            # 背景阶数随迭代逐步放开（前几轮只拟常数/线性，避免过拟合信号翼）
            bkg_counts = _fit_poly_background_counts(
                times, counts, expo, bkg_times, order=min(int(poly), iteration)
            )

        # ---- 2. 贝叶斯块（Giacomo 技巧：曝光 := 背景计数）----
        bb_index = bayesian_blocks_exposure(counts, bkg_counts, p0=p0)
        block_left = left[bb_index[:-1]]
        block_right = right[bb_index[1:] - 1]
        block_expo = np.asarray(
            [float(expo[bb_index[i]:bb_index[i + 1]].sum()) for i in range(bb_index.size - 1)]
        )
        block_counts = np.asarray(
            [float(counts[bb_index[i]:bb_index[i + 1]].sum()) for i in range(bb_index.size - 1)]
        )
        with np.errstate(divide='ignore', invalid='ignore'):
            block_rates = np.where(block_expo > 0, block_counts / block_expo, 0.0)

        # ---- 3. 峰与 prominence 定信号边界 ----
        peaks = argrelmax(block_rates)[0]
        if peaks.size == 0:
            # 原实现此处直接报错返回；这里退化为"无显著信号"，由上层决定处理。
            break
        prominence, left_base, right_base = peak_prominences(block_rates, peaks)
        leftmost_base = int(np.min(left_base))
        rightmost_base = int(np.max(right_base))

        new_start_signal = int(bb_index[leftmost_base + 1])
        new_stop_signal = int(bb_index[rightmost_base])
        signal_tstart = float(left[new_start_signal])
        signal_tstop = float(right[new_stop_signal - 1])

        # 缓冲区：向外再排除若干"边缘块宽度"，避免信号翼污染背景拟合；
        # 且不外推超过到光变两端距离的一半（同原实现）。
        # 注意：缓冲区 **仅用于背景排除**，不并入 signal_range；
        # Txx 分位计算基于 prominence 区间（与原实现 signal_range 口径一致）。
        block_widths = block_right - block_left
        left_buffer = min(
            float(buffer_blocks) * float(block_widths[left_base[0] + 1]),
            (signal_tstart - t_lo) / 2.0,
        )
        right_buffer = min(
            float(buffer_blocks) * float(block_widths[right_base[-1] - 1]),
            (t_hi - signal_tstop) / 2.0,
        )
        cand = np.digitize(
            [signal_tstart - left_buffer, signal_tstop + right_buffer], left
        ) - 1
        new_start, new_stop = int(cand[0]), int(cand[1])

        # ---- 4. 更新背景区间 ----
        bkgex_tstart = max(float(left[new_start]), float(left[bb_index[1]]))
        bkgex_tstop = min(float(right[new_stop - 1]), float(right[bb_index[-2] - 1]))
        new_bkg_times = [(t_lo, bkgex_tstart), (bkgex_tstop, t_hi)]

        # ---- 5. 收敛/循环检测 ----
        # 收敛时相邻两轮背景区间相同；但有时会出现 2-3 周期的近似振荡，
        # 若当前区间在历史中出现过（且已迭代至少 2 轮）即判定收敛（同原实现）。
        if new_bkg_times in previous_bkg_times and iteration >= 2:
            bkg_times = new_bkg_times
            converged = True
            break
        # 完全不变的相邻两轮也视为收敛
        if bkg_times == new_bkg_times:
            bkg_times = new_bkg_times
            converged = True
            break
        previous_bkg_times.append(bkg_times)
        bkg_times = new_bkg_times
        if bkg_counts_fixed is not None:
            # 背景固定时，一轮即自洽（信号区间不再反馈背景）
            converged = True
            break
    else:
        warnings.warn(
            f"iterative_bayesian_blocks: 达到最大迭代次数 {max_iter} 仍未收敛",
            RuntimeWarning,
            stacklevel=2,
        )

    # 峰时刻：信号区间内的最大计数率箱中心
    i0 = int(np.clip(np.digitize(signal_tstart, left) - 1, 0, n - 1))
    i1 = int(np.clip(np.digitize(signal_tstop, left), 1, n))
    seg_rates = counts[i0:i1] / np.maximum(expo[i0:i1], 1e-12)
    peak = float(times[i0 + int(np.argmax(seg_rates))]) if seg_rates.size else t_lo

    return IterativeBBResult(
        bb_index=bb_index,
        block_left=block_left,
        block_right=block_right,
        block_rates=block_rates,
        bkg_counts=bkg_counts,
        signal_range=(signal_tstart, signal_tstop),
        peak=peak,
        n_iterations=iteration + 1,
        converged=converged,
        bkg_times=bkg_times,
    )


def _signed_cumulative_curve(
    left: np.ndarray,
    right: np.ndarray,
    counts: np.ndarray,
    bkg_counts: np.ndarray,
    signal_range: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Build the piecewise-linear signed net-count cumulative curve.

    Only the positive overlap of each bin with ``signal_range`` contributes.
    A bin cut by either boundary contributes in proportion to its overlap,
    equivalent to a uniform rate within that bin.  Gaps are represented by
    horizontal segments rather than silently accumulating unavailable time.
    """
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    counts = np.asarray(counts, dtype=float)
    bkg_counts = np.asarray(bkg_counts, dtype=float)
    if not (left.shape == right.shape == counts.shape == bkg_counts.shape):
        raise ValueError("left/right/counts/bkg_counts 必须同形")
    if np.any(~np.isfinite(left)) or np.any(~np.isfinite(right)):
        raise ValueError("分箱边界必须有限")
    if np.any(right <= left) or np.any(np.diff(left) < 0.0):
        raise ValueError("分箱必须按时间排序且具有正宽度")

    s0, s1 = sorted(float(x) for x in signal_range)
    if not (np.isfinite(s0) and np.isfinite(s1) and s1 > s0):
        raise ValueError("signal_range 必须是有限的正宽区间")

    overlap_left = np.maximum(left, s0)
    overlap_right = np.minimum(right, s1)
    use = overlap_right > overlap_left
    if not np.any(use):
        raise ValueError("signal_range 与光变分箱没有正重叠")

    curve_time = [s0]
    cumulative = [0.0]
    current_time = s0
    current_counts = 0.0
    net = counts - bkg_counts
    for i in np.flatnonzero(use):
        a = float(overlap_left[i])
        b = float(overlap_right[i])
        if a > current_time:
            curve_time.append(a)
            cumulative.append(current_counts)
        width = float(right[i] - left[i])
        current_counts += float(net[i]) * (b - a) / width
        curve_time.append(b)
        cumulative.append(current_counts)
        current_time = b

    if current_time < s1:
        curve_time.append(s1)
        cumulative.append(current_counts)
    return np.asarray(curve_time, dtype=float), np.asarray(cumulative, dtype=float)


# 方法：同一阈值多次穿越时，取最早与最晚交点的中点作为分位时刻（稳健化处理，同
#       Swift/BAT battblocks 的 T(X%) 时刻约定）。
# 参考：HEASoft/HEADAS battblocks 文档, https://heasarc.gsfc.nasa.gov/docs/software/lheasoft/help/battblocks.html
def _crossing_midpoint(
    curve_time: np.ndarray,
    cumulative: np.ndarray,
    target: float,
) -> float:
    """Return the midpoint of the earliest and latest target crossings."""
    scale = max(1.0, float(np.max(np.abs(cumulative))), abs(float(target)))
    atol = 32.0 * np.finfo(float).eps * scale
    crossings: list[float] = []
    for t0, t1, c0, c1 in zip(
        curve_time[:-1], curve_time[1:], cumulative[:-1], cumulative[1:]
    ):
        d0 = float(c0 - target)
        d1 = float(c1 - target)
        if abs(d0) <= atol and abs(d1) <= atol:
            crossings.extend((float(t0), float(t1)))
            continue
        if d0 * d1 > 0.0 or abs(float(c1 - c0)) <= atol:
            continue
        frac = float((target - c0) / (c1 - c0))
        if -atol <= frac <= 1.0 + atol:
            crossings.append(float(t0 + np.clip(frac, 0.0, 1.0) * (t1 - t0)))
    if not crossings:
        raise ValueError(f"有符号累计曲线未穿越目标计数 {target:g}")
    return 0.5 * (min(crossings) + max(crossings))


# 方法：双侧净计数分位区间：累计净计数曲线达到 (1-q)/2 与 1-(1-q)/2 倍总量的交点即为
#       Txx 的起止时刻（负净计数箱保持有符号贡献；区间按时间有序返回）。
# 参考：Koshut, Paciesas, Kouveliotou, et al. 1996, ApJ 463, 570 (doi:10.1086/177272)；
#       q=0.9 即标准 T90 定义 t(95%)-t(5%)：Kouveliotou et al. 1993, ApJ 413, L101
#       (doi:10.1086/186969)；battblocks (HEASoft) 交点约定。
def _quantile_interval(
    left: np.ndarray,
    right: np.ndarray,
    counts: np.ndarray,
    bkg_counts: np.ndarray,
    signal_range: tuple[float, float],
    quantile: float,
) -> tuple[float, float]:
    """Return a symmetric signed-net-count quantile interval.

    Negative background-subtracted bins remain negative.  If a threshold is
    crossed more than once, the reported crossing is the midpoint between the
    earliest and latest solutions, matching the ``battblocks`` convention.
    """
    q = float(quantile)
    if not 0.0 < q < 1.0:
        raise ValueError("quantile 必须在 (0, 1) 内")
    curve_time, cumulative = _signed_cumulative_curve(
        left, right, counts, bkg_counts, signal_range
    )
    total = float(cumulative[-1])
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("signal_range 内有符号净计数总量必须为正")
    half = 0.5 * (1.0 - q)
    t1 = _crossing_midpoint(curve_time, cumulative, half * total)
    t2 = _crossing_midpoint(curve_time, cumulative, (1.0 - half) * total)
    if t2 < t1:
        raise ValueError("有符号累计分位交点反序")
    return (t1, t2)


def _project_counts_by_overlap(
    src_left: np.ndarray,
    src_right: np.ndarray,
    src_counts: np.ndarray,
    tgt_left: np.ndarray,
    tgt_right: np.ndarray,
) -> np.ndarray:
    """按重叠比例把源网格箱的计数投影到目标网格（线性缩放）。"""
    out = np.zeros(tgt_left.size, dtype=float)
    if tgt_left.size == 0 or src_left.size == 0:
        return out
    for i in range(tgt_left.size):
        a, b = float(tgt_left[i]), float(tgt_right[i])
        overlap = np.minimum(src_right, b) - np.maximum(src_left, a)
        frac = np.where(src_right > src_left, np.maximum(overlap, 0.0) / np.maximum(src_right - src_left, 1e-12), 0.0)
        out[i] = float(np.sum(src_counts * frac))
    return out


def txx_iterbkg(
    lc_src,
    background=None,
    *,
    alpha: Optional[float] = None,
    percent: float | Sequence[float] = (0.5, 0.9),
    p0: float = 0.05,
    poly: int = 2,
    buffer_blocks: int = 5,
    max_iter: int = 100,
    evt_binsize: float = 1.0,
    nsamples: int = 0,
    containment: float | Sequence[float] = 0.68,
    seed: Optional[int] = None,
) -> dict:
    """迭代背景自洽贝叶斯块法计算 T100/T90/T50（burstcube 方法，分箱光变输入）。

    与 ``jinwu.core.ops.txx``（事件级、A&A 5.4 方法）互为独立方法学：

    - 本方法先在分箱光变上迭代拟合背景、做曝光加权贝叶斯块与显著性定界，
      再用背景扣除累积计数插值出各分位时长；
    - ``nsamples > 0`` 时，用 Poisson 重采样把 **整条流水线** 重跑得到
      时长统计误差（含块选择效应），并叠加时间分箱系统误差。
    - 通过 ``jinwu.core.data.timescale.compute(method='iterbkg')`` 使用，可与默认方法对比。

    参数
    ----
    lc_src : LightcurveData | EventData | str | Path
        源数据。事件输入会先按 ``evt_binsize`` 重分箱为光变。
    background : 同上，可选。提供时按 ``alpha`` 缩放并投影到源网格，
        作为 **固定** 背景模型（不再多项式拟合）；不提供则迭代拟合多项式背景。
    alpha : 源/背景缩放因子；未提供时尝试从元信息推断（同 ``bin_bblocks``）。
    percent : 分位（如 0.9 → T90）。
    nsamples : 重采样次数（0=不计算统计误差）。
    containment : 误差区间置信度。
    seed : 重采样随机种子。

    返回
    ----
    dict：键与 ``txx`` 兼容（``method/percent/txx/t50/t90/t100`` 及 ``*_err*``、
    ``*_tstart/_tstop`` 族），另含 ``peak``、``n_iterations``、``converged``、
    ``bb_edges_time``、``background_rate``、``background_model_counts``；
    以及 ``alpha``（源/背景缩放因子，无法推断或无背景时为 ``None``）和
    ``time_reference``（``"absolute"`` 表示各 ``*_tstart/_tstop`` 与
    ``peak`` 为绝对 MET——输入 timezero 非零；``"relative"`` 表示相对秒）。
    """
    from .ops import _effective_exposure_from_lc, _infer_bin_geometry, _resolve_alpha_for_src_bkg

    # ---- 输入归一：统一到分箱光变的 (left, right, counts, exposure) ----
    def _to_binned(obj):
        from .data import EventDataBase, LightcurveDataBase

        if isinstance(obj, (str,)) or hasattr(obj, '__fspath__'):
            # 按内容判型分发（guess_ogip_kind 区分事件/光变），
            # 与对象输入汇合到同一处理路径（AUD-02）
            from .io import guess_ogip_kind, read_evt, read_lc
            kind = guess_ogip_kind(str(obj))
            if kind == 'evt':
                obj = read_evt(str(obj))
            elif kind == 'lc':
                obj = read_lc(str(obj))
            else:
                raise TypeError(
                    f"txx_iterbkg: 输入文件 {obj} 被识别为 {kind!r}，"
                    "需要事件文件或光变文件"
                )
        if isinstance(obj, EventDataBase):
            from .ops import rebin_events_to_lightcurve
            obj = rebin_events_to_lightcurve(obj, binsize=float(evt_binsize))
        if not isinstance(obj, LightcurveDataBase):
            raise TypeError(f"txx_iterbkg: 不支持的输入类型 {type(obj)!r}")
        left, right, width = _infer_bin_geometry(obj)
        # 绝对时间框架：读取器把 time 重定基到 0，绝对量保存在 timezero
        # （absolute_time = time + timezero）。把 bin 几何平移回绝对框架，
        # 使 src/bkg 共同网格按绝对时刻对齐（各自 timezero 可能不同），
        # 且返回的 tstart/tstop 与 txx 一样是绝对 MET（AUD-02）。
        tz = float(getattr(obj, 'timezero', 0.0) or 0.0)
        if tz != 0.0:
            left = left + tz
            right = right + tz
        expo = _effective_exposure_from_lc(obj, width)
        vals = np.asarray(obj.value, dtype=float)
        if obj.is_rate:
            counts = vals * expo
        else:
            counts = vals.copy()
        return left, right, np.maximum(counts, 0.0), np.asarray(expo, dtype=float), obj

    left, right, counts, expo, lc_obj = _to_binned(lc_src)

    # ---- 背景：投影到源网格并按 alpha 缩放 ----
    bkg_counts_fixed = None
    alpha_val = None
    if background is not None:
        b_left, b_right, b_counts, _b_expo, bkg_obj = _to_binned(background)
        alpha_val = _resolve_alpha_for_src_bkg(alpha, lc_obj, bkg_obj, context="txx_iterbkg")
        bkg_counts_fixed = float(alpha_val) * _project_counts_by_overlap(
            b_left, b_right, b_counts, left, right
        )

    # ---- 用户分位 → 内部统一补 0.5/0.9（与 txx 口径一致）----
    if np.ndim(percent) == 0:
        user_percent = np.asarray([float(percent)])
    else:
        user_percent = np.asarray(list(percent), dtype=float)
    if np.any((user_percent <= 0.0) | (user_percent >= 1.0)):
        raise ValueError("percent 必须在 (0, 1) 内")
    core_percent = np.unique(np.concatenate([user_percent, [0.5, 0.9]]))

    def _run(src_counts: np.ndarray) -> IterativeBBResult:
        return iterative_bayesian_blocks(
            left, right, src_counts,
            exposure=expo,
            bkg_counts_fixed=bkg_counts_fixed,
            p0=p0, poly=poly, buffer_blocks=buffer_blocks, max_iter=max_iter,
        )

    result = _run(counts)

    # ---- 名义时长：T100=prominence 信号区间，分位区间由累积净计数插值 ----
    t100_start, t100_stop = result.signal_range
    intervals = {float(q): _quantile_interval(left, right, counts, result.bkg_counts, result.signal_range, float(q))
                 for q in core_percent}
    intervals[1.0] = (t100_start, t100_stop)  # 对应原实现 signal_range(1)

    def _vectors(quants: np.ndarray):
        dur = np.array([intervals[float(q)][1] - intervals[float(q)][0] for q in quants])
        t1 = np.array([intervals[float(q)][0] for q in quants])
        t2 = np.array([intervals[float(q)][1] for q in quants])
        return dur, t1, t2

    dur_nom, t1_nom, t2_nom = _vectors(core_percent)

    # ---- 分箱系统误差（名义口径：区间首尾箱半宽和）----
    def _sys_row(q: float) -> np.ndarray:
        t1q, t2q = intervals[float(q)]
        i0 = int(np.clip(np.digitize(t1q, left) - 1, 0, left.size - 1))
        i1 = int(np.clip(np.digitize(t2q, left), 0, left.size - 1))
        half_bins = 0.5 * (right[i0] - left[i0]) + 0.5 * (right[i1] - left[i1])
        return np.array([half_bins, half_bins])

    dur_err_sys_all = np.array([_sys_row(float(q)) for q in core_percent])

    # ---- 统计误差：整条流水线在 Poisson 重采样上重跑（捕获选择效应）----
    err_shape = (core_percent.size, 2)
    dur_err_stat_all = np.full(err_shape, np.nan)
    t1_err_stat_all = np.full(err_shape, np.nan)
    t2_err_stat_all = np.full(err_shape, np.nan)
    if nsamples and int(nsamples) > 0:
        # 均值模型：背景模型 + 信号区间内的块率模型（原实现 duration_error 的做法）
        mean_counts = result.bkg_counts.copy()
        block_edges_all = np.concatenate((result.block_left, [float(result.block_right[-1])]))
        idx_in = np.clip(np.digitize(0.5 * (left + right), block_edges_all) - 1, 0, result.block_rates.size - 1)
        block_model = result.block_rates[idx_in] * expo
        in_sig = (left >= t100_start) & (right <= t100_stop)
        mean_counts[in_sig] = block_model[in_sig]
        mean_counts = np.maximum(mean_counts, 0.0)

        rng = np.random.default_rng(None if seed is None else int(seed))
        containment_arr = np.atleast_1d(np.asarray(containment, dtype=float))
        samples = {k: np.empty((int(nsamples), core_percent.size)) for k in ("dur", "t1", "t2")}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # 重采样内的未收敛告警不逐条上报
            for s in range(int(nsamples)):
                fluct = rng.poisson(mean_counts).astype(float)
                r_s = _run(fluct)
                for i_q, q in enumerate(core_percent):
                    seg = _quantile_interval(left, right, fluct, r_s.bkg_counts, r_s.signal_range, float(q))
                    samples["dur"][s, i_q] = seg[1] - seg[0]
                    samples["t1"][s, i_q] = seg[0]
                    samples["t2"][s, i_q] = seg[1]

        # 误差 = 分位边界 - 中位数（原实现口径），取第一个置信度的半宽为 ± 误差。
        q_lo = float((1.0 - containment_arr[0]) / 2.0)
        q_hi = float(1.0 - (1.0 - containment_arr[0]) / 2.0)
        for i_q in range(core_percent.size):
            for key, out in (("dur", dur_err_stat_all), ("t1", t1_err_stat_all), ("t2", t2_err_stat_all)):
                med = float(np.quantile(samples[key][:, i_q], 0.5))
                lo = float(np.quantile(samples[key][:, i_q], q_lo)) - med
                hi = float(np.quantile(samples[key][:, i_q], q_hi)) - med
                out[i_q] = (lo, hi)

    # 总误差：统计与分箱系统误差正交相加（无统计误差时仅系统项）
    def _tot(stat_row: np.ndarray, sys_row: np.ndarray) -> np.ndarray:
        if np.all(np.isfinite(stat_row)):
            return np.sqrt(stat_row ** 2 + sys_row ** 2)
        return sys_row

    dur_err_tot_all = np.array([
        _tot(dur_err_stat_all[i], dur_err_sys_all[i]) for i in range(core_percent.size)
    ])

    def _pick(arr: np.ndarray) -> np.ndarray:
        idx = [int(np.where(np.isclose(core_percent, p))[0][0]) for p in user_percent]
        return arr[idx]

    def _scalar(q: float) -> int:
        return int(np.where(np.isclose(core_percent, q))[0][0])

    i50, i90 = _scalar(0.5), _scalar(0.9)
    nan_pair = np.array([np.nan, np.nan])
    nan_mat = np.full((user_percent.size, 2), np.nan)

    def _row(arr: np.ndarray, i: int) -> np.ndarray:
        return arr[i] if np.all(np.isfinite(arr[i])) else nan_pair

    seg_width = np.maximum(right - left, 1e-12)
    time_reference = (
        "absolute"
        if float(getattr(lc_obj, 'timezero', 0.0) or 0.0) != 0.0
        else "relative"
    )
    return {
        "method": "iterbkg_bblocks_burstcube",
        "alpha": (float(alpha_val) if alpha_val is not None else None),
        "time_reference": time_reference,
        "percent": user_percent,
        "txx": _pick(dur_nom),
        "txx_err": _pick(dur_err_tot_all),
        "txx_err_stat": _pick(dur_err_stat_all) if nsamples else nan_mat,
        "txx_err_sys": _pick(dur_err_sys_all),
        "txx1": _pick(t1_nom),
        "txx2": _pick(t2_nom),
        "txx1_err_stat": _pick(t1_err_stat_all) if nsamples else nan_mat,
        "txx2_err_stat": _pick(t2_err_stat_all) if nsamples else nan_mat,
        "t50": float(dur_nom[i50]),
        "t50_err": _row(dur_err_tot_all, i50),
        "t50_err_stat": _row(dur_err_stat_all, i50) if nsamples else nan_pair,
        "t50_err_sys": _row(dur_err_sys_all, i50),
        "t50_tstart": float(t1_nom[i50]),
        "t50_tstop": float(t2_nom[i50]),
        "t90": float(dur_nom[i90]),
        "t90_err": _row(dur_err_tot_all, i90),
        "t90_err_stat": _row(dur_err_stat_all, i90) if nsamples else nan_pair,
        "t90_err_sys": _row(dur_err_sys_all, i90),
        "t90_tstart": float(t1_nom[i90]),
        "t90_tstop": float(t2_nom[i90]),
        "t100": float(t100_stop - t100_start),
        "t100_err": nan_pair,
        "t100_tstart": float(t100_start),
        "t100_tstop": float(t100_stop),
        "burst_tstart": float(t100_start),
        "burst_tstop": float(t100_stop),
        "peak": float(result.peak),
        "n_iterations": int(result.n_iterations),
        "converged": bool(result.converged),
        "bb_edges_time": np.concatenate((result.block_left, [float(result.block_right[-1])])),
        "background_rate": result.bkg_counts / seg_width,
        "background_model_counts": result.bkg_counts,
    }
