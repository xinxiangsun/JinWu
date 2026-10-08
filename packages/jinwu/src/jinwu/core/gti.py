"""GTI（Good Time Interval）纯 Python 实现：合并、生成、对齐与曝光计算。

目标：重现 maketime/mgtime 的核心语义用于 Jinwu 的纯 Python 流程：
- merge_gti / union_gti（合并重叠/相邻区间，mgtime 行为）
- intervals_from_mask（从时间序列与布尔掩码生成 GTI，类似 maketime）
- adjust_gti_to_frame（将 GTI 边界对齐到帧边界，类似 adjustgti）
- exposure_per_bins（计算每个时间 bin 与 GTI 的重叠曝光）

实现尽量简单且数值稳定；细节（例如 TIMEPIXR 的不同解释）按常见用法处理：
  - 当对齐到帧时，start 向上取整到最近帧起点，stop 向下取整到最近帧末端。
"""

from __future__ import annotations

from typing import Optional, Tuple, List
import numpy as np

try:
    from numba import njit
    _HAVE_NUMBA = True
except ImportError:
    _HAVE_NUMBA = False
    def njit(*args, **kwargs):
        """缺少 Numba 时保留原函数 / Keep functions unchanged without Numba.

        支持 ``@njit`` 与 ``@njit(...)`` 两种形式；忽略编译选项。
        Accept both decorator forms and ignore compilation options.
        """
        def decorator(func):
            """原样返回被装饰函数 / Return the decorated function unchanged."""
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator


def merge_gti(starts: Optional[np.ndarray], stops: Optional[np.ndarray], *, tol: float = 1e-9) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """合并 GTI 并按起点排序 / Merge and sort good-time intervals.

    Parameters
    ----------
    starts, stops : array-like or None
        对应的区间起止时间，须使用同一参考零点和单位；通常为秒。
        Paired interval boundaries in the same time reference and unit,
        usually seconds. The caller supplies valid, equally sized arrays.
    tol : float
        允许合并的最大间隙，单位与时间相同；默认 1e-9。
        Maximum gap to merge, in the input time unit; defaults to 1e-9.

    Returns
    -------
    tuple of numpy.ndarray or tuple of None
        非重叠、升序的 ``(starts, stops)``；缺失或空输入返回
        ``(None, None)``。相邻或重叠区间也会合并。
        Sorted, non-overlapping boundaries, or ``(None, None)`` for missing
        or empty input. Intervals merge when next_start <= current_stop + tol.
    """
    if starts is None or stops is None:
        return None, None
    s = np.asarray(starts, dtype=float)
    e = np.asarray(stops, dtype=float)
    if s.size == 0 or e.size == 0:
        return None, None
    order = np.argsort(s)
    s = s[order]
    e = e[order]
    merged_s: List[float] = []
    merged_e: List[float] = []
    cur_s = float(s[0])
    cur_e = float(e[0])
    for i in range(1, s.size):
        ns = float(s[i])
        ne = float(e[i])
        if ns <= cur_e + float(tol):
            cur_e = max(cur_e, ne)
        else:
            merged_s.append(cur_s)
            merged_e.append(cur_e)
            cur_s = ns
            cur_e = ne
    merged_s.append(cur_s)
    merged_e.append(cur_e)
    return np.asarray(merged_s, dtype=float), np.asarray(merged_e, dtype=float)


def union_gti(list_of_starts: List[np.ndarray], list_of_stops: List[np.ndarray], *, tol: float = 1e-9) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """合并多组 GTI 的并集 / Return the merged union of several GTI tables.

    ``list_of_starts`` 与 ``list_of_stops`` 按位置成对处理；跳过 None
    或空区间组，然后调用 :func:`merge_gti`。所有时间与 ``tol`` 须使用
    同一单位和参考零点；无有效区间时返回 ``(None, None)``。
    Pair the two lists positionally, skip missing/empty groups, and delegate
    to :func:`merge_gti`. Times and tolerance share a unit and reference.
    Return ``(None, None)`` when no intervals remain. Lists must have matching
    lengths; this implementation uses zip and does not validate that condition.
    """
    starts_flat = []
    stops_flat = []
    for s, e in zip(list_of_starts, list_of_stops):
        if s is None or e is None:
            continue
        s = np.asarray(s, dtype=float)
        e = np.asarray(e, dtype=float)
        if s.size == 0 or e.size == 0:
            continue
        starts_flat.extend(s.tolist())
        stops_flat.extend(e.tolist())
    if len(starts_flat) == 0:
        return None, None
    return merge_gti(np.asarray(starts_flat), np.asarray(stops_flat), tol=tol)


def intervals_from_mask(times: np.ndarray, mask: np.ndarray, *, min_gap: float = 0.0) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """从连续 True 样本生成区间 / Extract intervals from True sample runs.

    ``times`` 是升序的一维时间数组，``mask`` 是等长布尔数组。
    每个连续 True 段的首末样本时间直接成为起止边界；不会推断采样帧宽，
    也不会因相邻 True 样本时间跨度大而拆段。孤立样本产生零长度区间。
    ``min_gap > 0`` 时进一步合并间隙不大于该阈值的区间，单位同时间。
    ``times`` is a sorted 1-D time array and ``mask`` has the same length.
    Boundaries are the first/last selected sample times, without inferred
    frame widths. A large gap within a True run does not split it; an isolated
    sample gives a zero-duration interval. Positive ``min_gap`` merges short
    gaps in the same time unit.

    返回 ``(starts, stops)`` 数组；空输入或无 True 段返回 ``(None, None)``，
    非空数组长度不匹配时抛出 ValueError。
    Return boundary arrays, or ``(None, None)`` for empty input/no selected
    runs. Unequal nonempty array lengths raise ValueError.
    """
    times = np.asarray(times, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if times.size == 0 or mask.size == 0:
        return None, None
    if times.size != mask.size:
        raise ValueError('times and mask must have same length')
    # find rising edges and falling edges
    diff = np.diff(mask.astype(int))
    starts_idx = np.where(diff == 1)[0] + 1
    stops_idx = np.where(diff == -1)[0] + 1
    if mask[0]:
        starts_idx = np.concatenate(([0], starts_idx))
    if mask[-1]:
        stops_idx = np.concatenate((stops_idx, [mask.size]))
    if starts_idx.size == 0 or stops_idx.size == 0:
        return None, None
    starts = times[starts_idx]
    # stops_idx points to index after last True; use times[stops_idx - 1] as last event
    stops = times[np.maximum(0, stops_idx - 1)]
    # 保留最后一个 True 样本的时间作为 stop，不额外延长区间。
    # Keep the last selected sample time as stop, without extending the interval.
    if min_gap > 0.0 and starts.size > 1:
        merged_s = [float(starts[0])]
        merged_e = []
        cur_e = float(stops[0])
        for ns, ne in zip(starts[1:], stops[1:]):
            gap = float(ns) - float(cur_e)
            if gap <= float(min_gap):
                cur_e = max(cur_e, float(ne))
            else:
                merged_e.append(cur_e)
                merged_s.append(float(ns))
                cur_e = float(ne)
        merged_e.append(cur_e)
        return np.asarray(merged_s, dtype=float), np.asarray(merged_e, dtype=float)
    return np.asarray(starts, dtype=float), np.asarray(stops, dtype=float)


def adjust_gti_to_frame(starts: np.ndarray, stops: np.ndarray, frame_dt: float, timepixr: float = 0.0) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """向内对齐 GTI 到时间网格 / Align GTI inward to a frame grid.

    ``starts``、``stops``、正的 ``frame_dt`` 和 ``timepixr`` 均使用同一
    时间单位，通常为秒。本网格为 ``timepixr + k * frame_dt``；当前实现
    将 ``timepixr`` 直接作为时间偏移，不会乘以帧长，调用者须据此换算
    无量纲 FITS TIMEPIXR。起点向上、终点向下取整，舍弃零或负长度区间。
    All boundaries, positive ``frame_dt``, and ``timepixr`` share a time unit,
    usually seconds. The grid is ``timepixr + k * frame_dt``. Here timepixr
    is used directly as a time offset, not multiplied by frame duration;
    callers must convert a dimensionless FITS TIMEPIXR accordingly. Round
    starts up and stops down, discarding nonpositive durations.

    返回新的边界数组；缺失、空输入或全部被舍弃时返回 ``(None, None)``。
    Return new boundary arrays, or ``(None, None)`` for missing/empty input
    or when no positive-duration interval survives.
    """
    if starts is None or stops is None:
        return None, None
    s = np.asarray(starts, dtype=float)
    e = np.asarray(stops, dtype=float)
    if s.size == 0 or e.size == 0:
        return None, None
    new_s = []
    new_e = []
    for si, ei in zip(s, e):
        # map original times to frame index k where frame reference time = timepixr + k*frame_dt
        # frame start times are timepixr + k*frame_dt, frame end times are timepixr + (k+1)*frame_dt
        k_start = np.ceil((si - timepixr) / frame_dt - 1e-12)
        aligned_start = timepixr + k_start * frame_dt
        k_stop = np.floor((ei - timepixr) / frame_dt + 1e-12)
        aligned_stop = timepixr + k_stop * frame_dt
        if aligned_stop > aligned_start:
            new_s.append(float(aligned_start))
            new_e.append(float(aligned_stop))
    if len(new_s) == 0:
        return None, None
    return np.asarray(new_s, dtype=float), np.asarray(new_e, dtype=float)


if _HAVE_NUMBA:
    @njit
    def _exposure_per_bins_core(ms: np.ndarray, me: np.ndarray, bins: np.ndarray) -> np.ndarray:
        """逐 bin 累加 GTI 交叠时长 / Sum GTI overlaps per bin using Numba.

        输入为规范化的非重叠 GTI 与 bin 边界，单位一致；返回各 bin 曝光。
        Inputs are non-overlapping GTI and bin edges in one time unit;
        return exposure durations in that unit. No dead-time correction.
        """
        nb = bins.size - 1
        expo = np.zeros(nb, dtype=np.float64)
        for i in range(nb):
            b0 = bins[i]
            b1 = bins[i + 1]
            for j in range(ms.size):
                s = ms[j]
                e = me[j]
                lo = max(b0, s)
                hi = min(b1, e)
                if hi > lo:
                    expo[i] += (hi - lo)
        return expo
else:
    def _exposure_per_bins_core(ms: np.ndarray, me: np.ndarray, bins: np.ndarray) -> np.ndarray:
        """逐 bin 累加 GTI 交叠时长 / Sum GTI overlaps without Numba.

        与加速版本相同，假定 GTI 非重叠；不做死时间校正。
        Same input/output contract as the accelerated core; assumes GTI do
        not overlap and applies no dead-time correction.
        """
        nb = bins.size - 1
        expo = np.zeros(nb, dtype=float)
        for i in range(nb):
            b0 = float(bins[i])
            b1 = float(bins[i + 1])
            for s, e in zip(ms, me):
                lo = max(b0, float(s))
                hi = min(b1, float(e))
                if hi > lo:
                    expo[i] += (hi - lo)
        return expo


def exposure_per_bins(ms: np.ndarray, me: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """计算各 bin 的 GTI 交叠曝光 / Compute GTI overlap exposure per bin.

    ``ms``、``me`` 为已合并的非重叠 GTI 起止时间；``bins`` 为升序边界，
    长度 nbins+1。返回长度 nbins 的浮点数组，单位同输入时间（通常秒）。
    本函数直接累加交叠长度，不合并 GTI、不校正死时间；重叠 GTI 会重复
    计入曝光，故调用者应先用 :func:`merge_gti` 规范化。
    ``ms``/``me`` are merged non-overlapping GTI boundaries; ``bins`` contains
    sorted edges of length nbins+1. Return nbins float durations in the input
    time unit, usually seconds. Overlaps are summed without GTI merging or
    dead-time correction; call :func:`merge_gti` first to avoid double counting.
    """
    ms = np.asarray(ms, dtype=np.float64)
    me = np.asarray(me, dtype=np.float64)
    bins = np.asarray(bins, dtype=np.float64)
    return _exposure_per_bins_core(ms, me, bins)
