"""时标（T100/T90/T50 等）计算方法。

从 ``jinwu.core.ops`` 拆分出来的时标专题模块，汇集两种独立方法学：

1. :func:`txx` —— 事件级贝叶斯块 + A&A 5.4 双侧分位累计（原有默认方法）；
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
    err_m = max(float(nominal - q16), 0.0)
    err_p = max(float(q84 - nominal), 0.0)
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

    with fits.open(path, memmap=True) as hdul:
        if 'EVENTS' not in hdul:
            raise ValueError(f"事件文件缺少 EVENTS 扩展: {path}")

        evt_hdu = hdul['EVENTS']
        if evt_hdu.data is None or 'TIME' not in evt_hdu.columns.names:
            raise ValueError(f"EVENTS 扩展缺少 TIME 列: {path}")

        times = np.asarray(evt_hdu.data['TIME'], dtype=float)
        if times.size == 0:
            raise ValueError(f"事件文件 EVENTS 为空: {path}")

        if 'GTI' in hdul and hdul['GTI'].data is not None:
            gti_start = np.asarray(hdul['GTI'].data['START'], dtype=float)
            gti_stop = np.asarray(hdul['GTI'].data['STOP'], dtype=float)
        else:
            hdr = evt_hdu.header
            tstart = float(hdr.get('TSTART', np.min(times)))
            tstop = float(hdr.get('TSTOP', np.max(times)))
            gti_start = np.asarray([tstart], dtype=float)
            gti_stop = np.asarray([tstop], dtype=float)

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

    return {
        'path': str(path),
        'time': times,
        'gti_start': gti_start,
        'gti_stop': gti_stop,
        'area': area,
        'backscal': backscal,
    }


def _txx54_read_event_object(ev: EventDataBase, *, arg_name: str) -> dict:
    """将 EventDataBase/EventData 统一抽取为 txx 内部事件字典。"""
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
        try:
            gs = np.asarray(gti_s, dtype=float).reshape(-1)
            ge = np.asarray(gti_e, dtype=float).reshape(-1)
            if gs.size > 0 and ge.size > 0 and gs.size == ge.size:
                tz = float(getattr(ev, 'timezero', 0.0) or 0.0)
                gs = gs + tz
                ge = ge + tz
                good = np.isfinite(gs) & np.isfinite(ge) & (ge > gs)
                if np.any(good):
                    gti_start = gs[good]
                    gti_stop = ge[good]
        except Exception:
            gti_start = None
            gti_stop = None

    if gti_start is None or gti_stop is None:
        gti_start = np.asarray([float(np.min(times))], dtype=float)
        gti_stop = np.asarray([float(np.max(times))], dtype=float)

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

    edges = np.arange(float(t0), float(t1) + float(binsize), float(binsize), dtype=float)
    if edges.size < 2:
        edges = np.asarray([float(t0), float(t1)], dtype=float)
    elif edges[-1] < float(t1):
        edges = np.append(edges, float(t1))

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
    **kwargs,
) -> dict:
    """基于事件时间（photon events）直接运行 Bayesian Blocks 计算 T100/T90/T50。

    输入限制
    --------
    - `lc_src` 仅支持：事件文件路径（.evt/.fits/.fit/.fts）或 EventDataBase/EventData。
    - `background` 若提供，也必须是同类事件输入。

    说明
    ----
    - Bayesian Blocks 直接使用源事件到达时刻（fitness='events'）。
    - `cumulative_mode` 默认值是 `'adaptive'`（即不传该参数时走自适应累计）。
    - `cumulative_mode='adaptive'`：在 T100 内使用 Bayesian block 边界做累计（推荐）。
    - `cumulative_mode='fixed'`：使用 `evt_binsize` 等宽分段累计（兼容旧行为）。
    - 统计误差的 Poisson MC 在默认 adaptive 路径下以 0.5 s 细时间 bin 进行采样。
    - `block_snr_threshold` 默认 3.0，可按需求修改。
    - `alpha` 优先使用显式参数；若未提供，尝试从事件元信息（BACKSCAL/区域面积）推断。
    - 为兼容旧接口保留参数签名，其中 nmc/use_edge_bkg/timebins 等参数不使用。
    """
    # 兼容旧接口：保留参数但提示未使用。
    _unused = {
        'use_edge_bkg': use_edge_bkg,
        'lbkg': lbkg,
        'rbkg': rbkg,
        'burst_tstart': burst_tstart,
        'burst_tstop': burst_tstop,
        'tpeak': tpeak,
        'src_dist': src_dist,
        'bkg_dist': bkg_dist,
        'timebins': timebins,
        'small_bin_threshold': small_bin_threshold,
        'weak_peak_bins': weak_peak_bins,
        'weak_peak_weight': weak_peak_weight,
        'window_mode': window_mode,
        'density_quantile': density_quantile,
    }
    if kwargs:
        unknown = ", ".join(sorted(str(k) for k in kwargs.keys()))
        raise TypeError(f"txx() got unexpected keyword argument(s): {unknown}")
    if any(v is not None for k, v in _unused.items() if k in ('lbkg', 'rbkg', 'burst_tstart', 'burst_tstop', 'tpeak', 'timebins')) or use_edge_bkg:
        warnings.warn(
            "txx 新实现使用 A&A 5.4 方法，部分旧参数当前被忽略。",
            RuntimeWarning,
            stacklevel=2,
        )

    if isinstance(percent, (float, int)):
        user_percent_arr = np.asarray([float(percent)], dtype=float)
    else:
        user_percent_arr = np.asarray(list(percent), dtype=float)
    if np.any((user_percent_arr <= 0.0) | (user_percent_arr >= 1.0)):
        raise ValueError("percent 必须在 (0, 1) 内")
    core_percent_arr = np.unique(np.concatenate([user_percent_arr, np.asarray([0.5, 0.9], dtype=float)]))

    src_evt, bkg_evt, alpha = _txx54_convert_event_inputs(
        lc_src,
        background,
        alpha,
        evt_binsize=float(evt_binsize),
    )

    def _evt_bounds(evt: dict) -> tuple[float, float]:
        t = np.asarray(evt['time'], dtype=float)
        t = t[np.isfinite(t)]
        if t.size == 0:
            raise ValueError(f"事件输入为空: {evt.get('path', '<unknown>')}")

        gs = np.asarray(evt.get('gti_start', np.asarray([], dtype=float)), dtype=float).reshape(-1)
        ge = np.asarray(evt.get('gti_stop', np.asarray([], dtype=float)), dtype=float).reshape(-1)
        good = np.isfinite(gs) & np.isfinite(ge) & (ge > gs)
        if np.any(good):
            return float(np.min(gs[good])), float(np.max(ge[good]))
        return float(np.min(t)), float(np.max(t))

    src_bounds = _evt_bounds(src_evt)
    if bkg_evt is not None:
        bkg_bounds = _evt_bounds(bkg_evt)
        # 源/背景分析统一约束到公共时间范围，避免窗口不一致带来的统计偏差。
        t0 = max(src_bounds[0], bkg_bounds[0])
        t1 = min(src_bounds[1], bkg_bounds[1])
        if t1 <= t0:
            raise ValueError(
                f"源/背景事件 GTI 无交集: src={src_evt.get('path')}, bkg={bkg_evt.get('path')}"
            )
        if (
            (not np.isclose(src_bounds[0], bkg_bounds[0], rtol=0.0, atol=1e-9))
            or (not np.isclose(src_bounds[1], bkg_bounds[1], rtol=0.0, atol=1e-9))
        ):
            warnings.warn(
                (
                    "txx: 源/背景时间窗不完全一致，"
                    f"将使用公共时间范围 [{t0:.6f}, {t1:.6f}] 进行约束。"
                ),
                RuntimeWarning,
                stacklevel=2,
            )
    else:
        t0, t1 = src_bounds

    src_times_abs = np.asarray(src_evt['time'], dtype=float)
    src_times_abs = src_times_abs[np.isfinite(src_times_abs)]
    src_use = src_times_abs[(src_times_abs >= t0) & (src_times_abs <= t1)]
    if src_use.size == 0:
        raise ValueError("源事件在分析时间窗内为空，无法计算 Txx")

    if bkg_evt is not None:
        bkg_times_abs = np.asarray(bkg_evt['time'], dtype=float)
        bkg_times_abs = bkg_times_abs[np.isfinite(bkg_times_abs)]
        bkg_use = bkg_times_abs[(bkg_times_abs >= t0) & (bkg_times_abs <= t1)]
        alpha_val = _txx54_positive_scalar(alpha)
        if alpha_val is None:
            raise ValueError(
                "txx: 提供 background 时必须显式传入 alpha，"
                "或保证源/背景事件包含可推断的 BACKSCAL/区域面积信息。"
            )
        alpha = float(alpha_val)
    else:
        bkg_use = np.asarray([], dtype=float)
        alpha = 0.0

    duration = float(t1 - t0)
    if duration <= 0.0:
        raise ValueError(f"无效分析时长: t0={t0}, t1={t1}")

    src_rel = np.sort(src_use - t0)
    try:
        bb_edges_rel = np.asarray(
            bayesian_blocks(src_rel, fitness='events', p0=float(p0)),
            dtype=float,
        )
    except Exception as exc:
        raise RuntimeError(
            "Bayesian Blocks 在事件时间轴运行失败，请检查事件输入或 p0 参数。"
        ) from exc

    bb_edges_rel = bb_edges_rel[np.isfinite(bb_edges_rel)]
    if bb_edges_rel.size == 0:
        bb_edges_rel = np.asarray([0.0, duration], dtype=float)
    bb_edges_rel = np.clip(np.unique(bb_edges_rel), 0.0, duration)
    if bb_edges_rel.size < 2:
        bb_edges_rel = np.asarray([0.0, duration], dtype=float)
    else:
        # 确保边界覆盖完整分析窗，后续 T100 与累计分段不丢端点。
        if bb_edges_rel[0] > 0.0:
            bb_edges_rel = np.concatenate(([0.0], bb_edges_rel))
        if bb_edges_rel[-1] < duration:
            bb_edges_rel = np.concatenate((bb_edges_rel, [duration]))

    bb_edges_time = bb_edges_rel + t0
    bb_edges_tprime = bb_edges_rel.copy()

    src_sorted = np.sort(src_use)
    bkg_sorted = np.sort(bkg_use)

    def _count_in(sorted_arr: np.ndarray, a: float, b: float) -> float:
        if sorted_arr.size == 0 or b <= a:
            return 0.0
        i0 = int(np.searchsorted(sorted_arr, a, side='left'))
        i1 = int(np.searchsorted(sorted_arr, b, side='left'))
        return float(max(i1 - i0, 0))

    block_net = []
    block_snr = []
    block_bkg_model = []
    for a, b in zip(bb_edges_time[:-1], bb_edges_time[1:]):
        # 每个 BB 块独立计算源计数、背景模型与显著性。
        s_blk = _count_in(src_sorted, float(a), float(b))
        if bkg_evt is not None:
            b_raw_blk = _count_in(bkg_sorted, float(a), float(b))
            b_blk = float(alpha) * b_raw_blk
        else:
            b_raw_blk = 0.0
            b_blk = 0.0

        net_blk = s_blk - b_blk
        var_blk = max(s_blk + b_blk, 1e-12)
        block_net.append(net_blk)
        block_bkg_model.append(b_blk)

        if bkg_evt is not None and alpha > 0.0:
            s_pos = max(float(s_blk), 0.0)
            b_pos = max(float(b_raw_blk), 0.0)
            try:
                # Keep the exact Li & Ma boundary branches, including N_off=0.
                snr_blk = li_ma_snr(float(s_pos), float(b_pos), float(alpha))
            except Exception:
                snr_blk = float(net_blk / np.sqrt(var_blk))
            if not np.isfinite(snr_blk):
                snr_blk = float(net_blk / np.sqrt(var_blk))
            if net_blk < 0.0 and np.isfinite(snr_blk):
                snr_blk = -abs(snr_blk)
        else:
            snr_blk = float(net_blk / np.sqrt(var_blk))
        block_snr.append(snr_blk)

    block_snr_arr = np.asarray(block_snr, dtype=float)
    block_bkg_model_arr = np.asarray(block_bkg_model, dtype=float)

    snr_thr = float(block_snr_threshold)
    if (not np.isfinite(snr_thr)) or snr_thr <= 0.0:
        raise ValueError(
            f"block_snr_threshold 必须为正且有限，当前={block_snr_threshold}"
        )
    snr_mask = block_snr_arr > snr_thr
    if not np.any(snr_mask):
        raise RuntimeError(
            f"没有任何贝叶斯块的 SNR > {snr_thr:.3g}，无法按阈值定义 T0/T100。"
        )

    # T100 由第一个/最后一个高显著块外边界定义。
    i_first = int(np.where(snr_mask)[0][0])
    i_last = int(np.where(snr_mask)[0][-1])
    t100_start = float(bb_edges_time[i_first])
    t100_stop = float(bb_edges_time[i_last + 1])

    mode = str(cumulative_mode).strip().lower()
    if mode not in {'adaptive', 'fixed'}:
        raise ValueError(f"cumulative_mode 必须是 'adaptive' 或 'fixed'，当前={cumulative_mode!r}")

    binsize = float(evt_binsize)
    if (not np.isfinite(binsize)) or binsize <= 0.0:
        raise ValueError(f"evt_binsize 必须为正且有限，当前={evt_binsize}")

    if mode == 'adaptive':
        # 自适应模式：直接使用 T100 内 BB 边界累计。
        seg_edges = np.asarray(bb_edges_time[i_first:i_last + 2], dtype=float)
        seg_edges = seg_edges[np.isfinite(seg_edges)]
        if seg_edges.size < 2:
            seg_edges = np.asarray([t100_start, t100_stop], dtype=float)
        else:
            seg_edges = np.clip(seg_edges, t100_start, t100_stop)
            seg_edges = np.unique(seg_edges)
            if seg_edges.size < 2:
                seg_edges = np.asarray([t100_start, t100_stop], dtype=float)
            else:
                if seg_edges[0] > t100_start:
                    seg_edges = np.concatenate(([t100_start], seg_edges))
                if seg_edges[-1] < t100_stop:
                    seg_edges = np.concatenate((seg_edges, [t100_stop]))
    else:
        # 固定模式：按等宽时间步长累计。
        seg_edges = np.arange(t100_start, t100_stop + binsize, binsize, dtype=float)
        if seg_edges.size < 2:
            seg_edges = np.asarray([t100_start, t100_stop], dtype=float)
        elif seg_edges[-1] < t100_stop:
            seg_edges = np.append(seg_edges, t100_stop)

    if seg_edges.size < 2:
        seg_edges = np.asarray([t100_start, t100_stop], dtype=float)

    src_hist, _ = np.histogram(src_use, bins=seg_edges)
    if bkg_evt is not None and alpha > 0.0:
        bkg_hist, _ = np.histogram(bkg_use, bins=seg_edges)
        bkg_model_counts = float(alpha) * bkg_hist.astype(float)
    else:
        bkg_model_counts = np.zeros_like(src_hist, dtype=float)

    seg_left_arr = np.asarray(seg_edges[:-1], dtype=float)
    seg_right_arr = np.asarray(seg_edges[1:], dtype=float)
    seg_net_pos_arr = np.maximum(src_hist.astype(float) - bkg_model_counts, 0.0)

    def _duration_vectors(
        seg_left_local: np.ndarray,
        seg_right_local: np.ndarray,
        seg_net_local: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_local = core_percent_arr.size
        dur_local = np.full(n_local, np.nan, dtype=float)
        t1_local = np.full(n_local, np.nan, dtype=float)
        t2_local = np.full(n_local, np.nan, dtype=float)

        total_local = float(np.sum(seg_net_local)) if seg_net_local.size > 0 else 0.0
        if total_local <= 0.0 or seg_net_local.size == 0:
            return dur_local, t1_local, t2_local

        for i_p, p in enumerate(core_percent_arr):
            # A&A 5.4: 以累计净计数的双侧分位定义 Txx 及其左右边界。
            low = 0.5 * (1.0 - float(p)) * total_local
            high = 0.5 * (1.0 + float(p)) * total_local
            t1v = _txx54_cross_target(low, seg_left_local, seg_right_local, seg_net_local)
            t2v = _txx54_cross_target(high, seg_left_local, seg_right_local, seg_net_local)
            t1_local[i_p] = float(t1v)
            t2_local[i_p] = float(t2v)
            dur_local[i_p] = float(t2v - t1v)

        return dur_local, t1_local, t2_local

    def _duration_vectors_from_edges(seg_edges_local: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        src_local, _ = np.histogram(src_use, bins=seg_edges_local)
        if bkg_evt is not None and alpha > 0.0:
            bkg_local_raw, _ = np.histogram(bkg_use, bins=seg_edges_local)
            bkg_local = float(alpha) * bkg_local_raw.astype(float)
        else:
            bkg_local = np.zeros_like(src_local, dtype=float)

        seg_left_local = np.asarray(seg_edges_local[:-1], dtype=float)
        seg_right_local = np.asarray(seg_edges_local[1:], dtype=float)
        seg_net_local = np.maximum(src_local.astype(float) - bkg_local, 0.0)
        return _duration_vectors(seg_left_local, seg_right_local, seg_net_local)

    dur_all_arr, t1_all_arr, t2_all_arr = _duration_vectors(seg_left_arr, seg_right_arr, seg_net_pos_arr)

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

    txx_nom = dur_all_arr[idx_user]
    txx1_nom = t1_all_arr[idx_user]
    txx2_nom = t2_all_arr[idx_user]

    t50_nom = float(dur_all_arr[idx50])
    t50_start_nom = float(t1_all_arr[idx50])
    t50_stop_nom = float(t2_all_arr[idx50])
    t90_nom = float(dur_all_arr[idx90])
    t90_start_nom = float(t1_all_arr[idx90])
    t90_stop_nom = float(t2_all_arr[idx90])
    t100_nom = float(t100_stop - t100_start)

    seg_width = np.maximum(seg_right_arr - seg_left_arr, 1e-12)
    background_rate = bkg_model_counts / seg_width if seg_width.size > 0 else np.asarray([], dtype=float)

    n_core = int(core_percent_arr.size)
    err_shape = (n_core, 2)
    dur_err_stat_all = np.full(err_shape, np.nan, dtype=float)
    t1_err_stat_all = np.full(err_shape, np.nan, dtype=float)
    t2_err_stat_all = np.full(err_shape, np.nan, dtype=float)
    dur_err_sys_all = np.full(err_shape, np.nan, dtype=float)
    t1_err_sys_all = np.full(err_shape, np.nan, dtype=float)
    t2_err_sys_all = np.full(err_shape, np.nan, dtype=float)

    nmc_eff = int(max(float(nmc), 0.0))
    if nmc_eff >= 20:
        # 统计误差：在 T100 内细分网格做 Poisson MC（adaptive 默认 0.5 s 细 bin），并重新计算分位时标。
        span = max(float(t100_stop - t100_start), 1e-6)
        dt_floor = max(span / 512.0, 0.1)
        dt_cap = max(float(evt_binsize), 0.25)
        if mode == 'adaptive':
            dt_ref = min(max(dt_floor, min(float(evt_binsize), 1.0) / 2.0), dt_cap)
        else:
            dt_ref = min(max(dt_floor, float(evt_binsize) / 2.0), dt_cap)
        if (not np.isfinite(dt_ref)) or dt_ref <= 0.0:
            dt_ref = max(span / 256.0, 0.25)

        n_ref = int(np.clip(np.ceil(span / dt_ref), 20, 1200))
        mc_edges = np.linspace(float(t100_start), float(t100_stop), n_ref + 1, dtype=float)
        src_ref_hist, _ = np.histogram(src_use, bins=mc_edges)
        if bkg_evt is not None and alpha > 0.0:
            bkg_ref_raw, _ = np.histogram(bkg_use, bins=mc_edges)
            bkg_ref_raw = bkg_ref_raw.astype(float)
        else:
            bkg_ref_raw = np.zeros_like(src_ref_hist, dtype=float)

        rng_seed = None if seed is None else int(seed)
        rng = np.random.default_rng(rng_seed)

        dur_samples: list[np.ndarray] = []
        t1_samples: list[np.ndarray] = []
        t2_samples: list[np.ndarray] = []
        src_ref_nonneg = np.maximum(src_ref_hist.astype(float), 0.0)
        bkg_ref_nonneg = np.maximum(bkg_ref_raw, 0.0)

        for _ in range(nmc_eff):
            src_draw = rng.poisson(src_ref_nonneg).astype(float)
            if bkg_evt is not None and alpha > 0.0:
                # 背景按“先对原始背景抽样，再乘 alpha”处理。
                bkg_draw_raw = rng.poisson(bkg_ref_nonneg).astype(float)
                bkg_draw = float(alpha) * bkg_draw_raw
            else:
                bkg_draw = np.zeros_like(src_draw, dtype=float)

            seg_left_mc = mc_edges[:-1]
            seg_right_mc = mc_edges[1:]
            seg_net_mc = np.maximum(src_draw - bkg_draw, 0.0)
            d_mc, t1_mc, t2_mc = _duration_vectors(seg_left_mc, seg_right_mc, seg_net_mc)
            if np.any(np.isfinite(d_mc)):
                dur_samples.append(d_mc)
                t1_samples.append(t1_mc)
                t2_samples.append(t2_mc)

        if len(dur_samples) >= 20:
            dur_samples_arr = np.asarray(dur_samples, dtype=float)
            t1_samples_arr = np.asarray(t1_samples, dtype=float)
            t2_samples_arr = np.asarray(t2_samples, dtype=float)

            for j in range(n_core):
                em, ep = _txx54_asymm_err_from_samples(dur_samples_arr[:, j], dur_all_arr[j])
                dur_err_stat_all[j, 0] = em
                dur_err_stat_all[j, 1] = ep

                em, ep = _txx54_asymm_err_from_samples(t1_samples_arr[:, j], t1_all_arr[j])
                t1_err_stat_all[j, 0] = em
                t1_err_stat_all[j, 1] = ep

                em, ep = _txx54_asymm_err_from_samples(t2_samples_arr[:, j], t2_all_arr[j])
                t2_err_stat_all[j, 0] = em
                t2_err_stat_all[j, 1] = ep

    variant_edges: list[np.ndarray] = []
    _seen_edge_keys: set[tuple[float, ...]] = set()

    def _add_variant_edges(edges_in: np.ndarray) -> None:
        e = np.asarray(edges_in, dtype=float)
        e = e[np.isfinite(e)]
        if e.size < 2:
            return
        e = np.clip(np.unique(e), t100_start, t100_stop)
        if e.size < 2:
            return
        if e[0] > t100_start:
            e = np.concatenate(([t100_start], e))
        if e[-1] < t100_stop:
            e = np.concatenate((e, [t100_stop]))
        e = np.clip(np.unique(e), t100_start, t100_stop)
        if e.size < 2:
            return
        key = tuple(float(v) for v in np.round(np.asarray(e, dtype=float), 6))
        if key in _seen_edge_keys:
            return
        _seen_edge_keys.add(key)
        variant_edges.append(e)

    _add_variant_edges(np.asarray(seg_edges, dtype=float))
    _add_variant_edges(np.asarray(bb_edges_time[i_first:i_last + 2], dtype=float))
    for scale in (0.5, 1.0, 2.0):
        # 系统误差：使用多种边界方案评估分段敏感性。
        bs_i = max(0.25, float(binsize) * float(scale))
        e_i = np.arange(t100_start, t100_stop + bs_i, bs_i, dtype=float)
        if e_i.size < 2:
            e_i = np.asarray([t100_start, t100_stop], dtype=float)
        elif e_i[-1] < t100_stop:
            e_i = np.append(e_i, t100_stop)
        _add_variant_edges(e_i)

    var_dur: list[np.ndarray] = []
    var_t1: list[np.ndarray] = []
    var_t2: list[np.ndarray] = []
    for e_var in variant_edges:
        d_var, t1_var, t2_var = _duration_vectors_from_edges(e_var)
        if np.any(np.isfinite(d_var)):
            var_dur.append(d_var)
            var_t1.append(t1_var)
            var_t2.append(t2_var)

    if len(var_dur) >= 2:
        var_dur_arr = np.asarray(var_dur, dtype=float)
        var_t1_arr = np.asarray(var_t1, dtype=float)
        var_t2_arr = np.asarray(var_t2, dtype=float)

        for j in range(n_core):
            sig = _txx54_robust_sigma(var_dur_arr[:, j])
            if np.isfinite(sig):
                dur_err_sys_all[j, :] = float(sig)

            sig = _txx54_robust_sigma(var_t1_arr[:, j])
            if np.isfinite(sig):
                t1_err_sys_all[j, :] = float(sig)

            sig = _txx54_robust_sigma(var_t2_arr[:, j])
            if np.isfinite(sig):
                t2_err_sys_all[j, :] = float(sig)

    def _combine_err(stat_arr: np.ndarray, sys_arr: np.ndarray) -> np.ndarray:
        # 总误差按上下误差分量分别做二范数合成。
        out = np.full_like(stat_arr, np.nan, dtype=float)
        for col in range(stat_arr.shape[1]):
            s = stat_arr[:, col]
            y = sys_arr[:, col]

            both = np.isfinite(s) & np.isfinite(y)
            out[both, col] = np.sqrt(np.maximum(s[both], 0.0) ** 2 + np.maximum(y[both], 0.0) ** 2)

            only_s = np.isfinite(s) & (~np.isfinite(y))
            out[only_s, col] = np.maximum(s[only_s], 0.0)

            only_y = (~np.isfinite(s)) & np.isfinite(y)
            out[only_y, col] = np.maximum(y[only_y], 0.0)
        return out

    dur_err_tot_all = _combine_err(dur_err_stat_all, dur_err_sys_all)
    t1_err_tot_all = _combine_err(t1_err_stat_all, t1_err_sys_all)
    t2_err_tot_all = _combine_err(t2_err_stat_all, t2_err_sys_all)

    txx_err_stat_user = dur_err_stat_all[idx_user]
    txx1_err_stat_user = t1_err_stat_all[idx_user]
    txx2_err_stat_user = t2_err_stat_all[idx_user]
    txx_err_sys_user = dur_err_sys_all[idx_user]
    txx1_err_sys_user = t1_err_sys_all[idx_user]
    txx2_err_sys_user = t2_err_sys_all[idx_user]
    txx_err_tot_user = dur_err_tot_all[idx_user]
    txx1_err_tot_user = t1_err_tot_all[idx_user]
    txx2_err_tot_user = t2_err_tot_all[idx_user]

    t50_err_stat = dur_err_stat_all[idx50]
    t50_start_err_stat = t1_err_stat_all[idx50]
    t50_stop_err_stat = t2_err_stat_all[idx50]
    t50_err_sys = dur_err_sys_all[idx50]
    t50_start_err_sys = t1_err_sys_all[idx50]
    t50_stop_err_sys = t2_err_sys_all[idx50]
    t50_err_tot = dur_err_tot_all[idx50]
    t50_start_err_tot = t1_err_tot_all[idx50]
    t50_stop_err_tot = t2_err_tot_all[idx50]

    t90_err_stat = dur_err_stat_all[idx90]
    t90_start_err_stat = t1_err_stat_all[idx90]
    t90_stop_err_stat = t2_err_stat_all[idx90]
    t90_err_sys = dur_err_sys_all[idx90]
    t90_start_err_sys = t1_err_sys_all[idx90]
    t90_stop_err_sys = t2_err_sys_all[idx90]
    t90_err_tot = dur_err_tot_all[idx90]
    t90_start_err_tot = t1_err_tot_all[idx90]
    t90_stop_err_tot = t2_err_tot_all[idx90]

    nan_pair = np.asarray([np.nan, np.nan], dtype=float)

    return {
        "method": "aanda_2021_sec5_4",
        "percent": user_percent_arr,
        "txx": txx_nom,
        "txx_err": txx_err_tot_user,
        "txx_err_stat": txx_err_stat_user,
        "txx_err_sys": txx_err_sys_user,
        "txx1": txx1_nom,
        "txx1_err": txx1_err_tot_user,
        "txx1_err_stat": txx1_err_stat_user,
        "txx1_err_sys": txx1_err_sys_user,
        "txx2": txx2_nom,
        "txx2_err": txx2_err_tot_user,
        "txx2_err_stat": txx2_err_stat_user,
        "txx2_err_sys": txx2_err_sys_user,

        "t50": t50_nom,
        "t50_err": t50_err_tot,
        "t50_err_stat": t50_err_stat,
        "t50_err_sys": t50_err_sys,
        "t50_tstart": t50_start_nom,
        "t50_tstart_err": t50_start_err_tot,
        "t50_tstart_err_stat": t50_start_err_stat,
        "t50_tstart_err_sys": t50_start_err_sys,
        "t50_tstop": t50_stop_nom,
        "t50_tstop_err": t50_stop_err_tot,
        "t50_tstop_err_stat": t50_stop_err_stat,
        "t50_tstop_err_sys": t50_stop_err_sys,

        "t90": t90_nom,
        "t90_err": t90_err_tot,
        "t90_err_stat": t90_err_stat,
        "t90_err_sys": t90_err_sys,
        "t90_tstart": t90_start_nom,
        "t90_tstart_err": t90_start_err_tot,
        "t90_tstart_err_stat": t90_start_err_stat,
        "t90_tstart_err_sys": t90_start_err_sys,
        "t90_tstop": t90_stop_nom,
        "t90_tstop_err": t90_stop_err_tot,
        "t90_tstop_err_stat": t90_stop_err_stat,
        "t90_tstop_err_sys": t90_stop_err_sys,

        "t100": t100_nom,
        "t100_err": nan_pair,
        "t100_tstart": t100_start,
        "t100_tstop": t100_stop,
        "burst_tstart": t100_start,
        "burst_tstop": t100_stop,

        "bb_edges_tprime": bb_edges_tprime,
        "bb_edges_time": bb_edges_time,
        "bb_block_snr": block_snr_arr,
        "bb_snr_threshold": snr_thr,
        "cumulative_mode": mode,
        "cumulative_edges_time": seg_edges,
        "background_rate": background_rate,
        "background_model_counts": bkg_model_counts,
        "bb_block_bkg_model_counts": block_bkg_model_arr,
    }


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


def _quantile_interval(
    left: np.ndarray,
    right: np.ndarray,
    counts: np.ndarray,
    bkg_counts: np.ndarray,
    signal_range: tuple[float, float],
    quantile: float,
) -> tuple[float, float]:
    """背景扣除累积计数的对称分位区间（如 0.9 → 含 90% 净计数的区间）。

    与原实现一致用线性插值；逐箱净计数先截断为非负再累计（弱信号翼部可能出现
    负净计数，避免累计曲线回落，也保证插值 ``xp`` 单调）。
    """
    s0, s1 = signal_range
    # 窗口箱选择：i0 为左边界 ≤ s0 的最后一箱，i1 为右边界 ≥ s1 的第一箱之后。
    # 不能用 digitize(s1, left)：当箱边界相对 s0 有小偏移时（如事件重分箱的 0.37s 相位），
    # s1 恰好等于某个 left 值会被归入左侧，丢掉末尾一箱。
    i0 = int(np.clip(np.searchsorted(right, s0, side='left'), 0, counts.size - 1))
    i1 = int(np.clip(np.searchsorted(left, s1, side='right'), 1, counts.size))
    net = counts[i0:i1] - bkg_counts[i0:i1]
    if net.size == 0 or float(net.sum()) <= 0.0:
        return (s0, s1)
    # 同原实现：累积序列头部补 0；时间轴取窗口内各箱的右边界（相对窗口起点），
    # 与累积序列（同样头部补 0）等长。注意不能对 i1 之后的箱做减法（会越界失真）。
    cum = np.concatenate(([0.0], np.cumsum(np.maximum(net, 0.0))))
    total = float(cum[-1])
    if total <= 0.0:
        return (s0, s1)
    cumtime = np.concatenate(([0.0], right[i0:i1] - float(left[i0])))
    half = 0.5 * (1.0 - float(quantile))
    frac = cum / total
    # cum 是非负净计数的累计（头部补 0），frac 必然非递减，满足 np.interp 对 xp 的要求。
    t1 = float(np.interp(half, frac, cumtime)) + float(left[i0])
    t2 = float(np.interp(1.0 - half, frac, cumtime)) + float(left[i0])
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
    ``bb_edges_time``、``background_rate``、``background_model_counts``。
    """
    from .ops import _effective_exposure_from_lc, _infer_bin_geometry, _resolve_alpha_for_src_bkg

    # ---- 输入归一：统一到分箱光变的 (left, right, counts, exposure) ----
    def _to_binned(obj):
        from .data import EventDataBase, LightcurveDataBase

        if isinstance(obj, (str,)) or hasattr(obj, '__fspath__'):
            from .io import read_lc
            obj = read_lc(str(obj))
        if isinstance(obj, EventDataBase):
            from .ops import rebin_events_to_lightcurve
            obj = rebin_events_to_lightcurve(obj, binsize=float(evt_binsize))
        if not isinstance(obj, LightcurveDataBase):
            raise TypeError(f"txx_iterbkg: 不支持的输入类型 {type(obj)!r}")
        left, right, width = _infer_bin_geometry(obj)
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
    return {
        "method": "iterbkg_bblocks_burstcube",
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
