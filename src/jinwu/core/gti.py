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
        """Dummy decorator when numba is not available."""
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator


def merge_gti(starts: Optional[np.ndarray], stops: Optional[np.ndarray], *, tol: float = 1e-9) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """合并并规范化 GTI 列表：返回非重叠、升序的 (starts, stops)。

    - 会按 start 排序并合并相邻或重叠区间（next_start <= cur_stop + tol 时合并）。
    - 若输入为空或长度为0，返回 (None, None)。
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
    """将多个 GTI 列表取并集，返回合并规范化后的 (starts, stops)。

    用于模拟 mgtime：把多个 GTI 表的区间合并为一个整体。
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
    """从时间序列和布尔掩码生成 GTI 区间（连续 True 段）。

    - `times` 应为单调升序的时间数组（事件或 MKF 时间中心）。
    - 连续的 True 段（相邻索引之间时间差可任意）会被抽取为 [t_start, t_stop]。
    - `min_gap` 可用于合并短间隙（若 gap <= min_gap 则视为连续）。
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
    # Convert to half-open intervals [start, stop] using small epsilon extension
    # but here we return inclusive stops as in xselect (use observed times)
    # Optionally merge short gaps
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
    """将 GTI 边界对齐到帧边界：start 向上取整到下一个帧起点，stop 向下取整到帧末端。

    - frame_dt: 帧时长（秒）
    - timepixr: 时间参考偏移（通常 0.0 或 0.5），表示每帧参考点与边界的偏移
    返回新的 (starts, stops)，并丢弃长度 <= 0 的区间。
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
        """Numba-accelerated core for exposure_per_bins calculation."""
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
        """Pure Python fallback for exposure_per_bins calculation."""
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
    """计算每个时间 bin 与合并 GTI 的重叠曝光时长。bins 为 edges（len = nbins+1）。"""
    ms = np.asarray(ms, dtype=np.float64)
    me = np.asarray(me, dtype=np.float64)
    bins = np.asarray(bins, dtype=np.float64)
    return _exposure_per_bins_core(ms, me, bins)
