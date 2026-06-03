from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union, Literal

import numpy as np
from astropy.stats import bayesian_blocks

from .utils import snr_li_ma


def _cross_target_time(
    target: float,
    left_arr: np.ndarray,
    right_arr: np.ndarray,
    net_arr: np.ndarray
) -> float:
    """
    通过阈值穿越求时间。

    参数:
        target: 目标累积值
        left_arr: 每段左边界
        right_arr: 每段右边界
        net_arr: 每段净计数

    返回:
        对应 target 的时间
    """
    cumsum = 0.0
    for i in range(len(net_arr)):
        seg_net = net_arr[i]
        seg_width = max(right_arr[i] - left_arr[i], 1e-12)
        seg_rate = seg_net / seg_width
        cumsum += seg_net
        if cumsum >= target:
            overshoot = cumsum - target
            t_cross = right_arr[i] - overshoot / seg_rate if seg_rate > 0 else left_arr[i]
            return float(t_cross)
    return float(right_arr[-1])


def _compute_txx_from_cumulative(
    t0: float,
    t100: float,
    times: np.ndarray,
    bkg_times: Optional[np.ndarray],
    alpha: Optional[float],
    seg_edges: np.ndarray,
) -> Tuple[float, float, float, float, float]:
    """
    从分段边界计算 T90/T50。

    返回:
        (t90, t90_start, t90_stop, t50, t50_start, t50_stop)
    """
    src_hist, _ = np.histogram(times, bins=seg_edges)
    if bkg_times is not None and alpha is not None and alpha > 0:
        bkg_hist, _ = np.histogram(bkg_times, bins=seg_edges)
        bkg_model = float(alpha) * bkg_hist.astype(float)
    else:
        bkg_model = np.zeros_like(src_hist, dtype=float)

    seg_left = seg_edges[:-1]
    seg_right = seg_edges[1:]
    seg_net = np.maximum(src_hist.astype(float) - bkg_model, 0.0)
    total_net = float(np.sum(seg_net))

    if total_net <= 0.0 or seg_net.size == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    t90 = _cross_target_time(0.95 * total_net, seg_left, seg_right, seg_net) - \
           _cross_target_time(0.05 * total_net, seg_left, seg_right, seg_net)
    t90_start = _cross_target_time(0.05 * total_net, seg_left, seg_right, seg_net)
    t90_stop = _cross_target_time(0.95 * total_net, seg_left, seg_right, seg_net)

    t50 = _cross_target_time(0.75 * total_net, seg_left, seg_right, seg_net) - \
           _cross_target_time(0.25 * total_net, seg_left, seg_right, seg_net)
    t50_start = _cross_target_time(0.25 * total_net, seg_left, seg_right, seg_net)
    t50_stop = _cross_target_time(0.75 * total_net, seg_left, seg_right, seg_net)

    return t90, t90_start, t90_stop, t50, t50_start, t50_stop


def _poisson_mc_stat_error(
    times: np.ndarray,
    bkg_times: Optional[np.ndarray],
    alpha: Optional[float],
    t0: float,
    t100: float,
    seg_edges: np.ndarray,
    n_mc: int = 100,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    统计误差：泊松重采样 MC。

    返回:
        (t90_err_stat, t50_err_stat)
    """
    rng = np.random.default_rng(seed)
    times = np.asarray(times, dtype=float)
    times = times[np.isfinite(times)]
    times_in_range = times[(times >= t0) & (times <= t100)]

    if bkg_times is not None:
        bkg_times_arr = np.asarray(bkg_times, dtype=float)
        bkg_times_arr = bkg_times_arr[np.isfinite(bkg_times_arr)]
        bkg_in_range = bkg_times_arr[(bkg_times_arr >= t0) & (bkg_times_arr <= t100)]
    else:
        bkg_in_range = np.asarray([], dtype=float)

    span = max(float(t100 - t0), 1e-9)
    widths = np.diff(np.asarray(seg_edges, dtype=float)) if np.asarray(seg_edges).size >= 2 else np.asarray([], dtype=float)
    widths = widths[np.isfinite(widths) & (widths > 0.0)]

    if widths.size > 0:
        dt_ref = float(np.median(widths) / 4.0)
    else:
        dt_ref = span / 200.0

    dt_ref = max(dt_ref, span / 512.0, 0.1)
    dt_ref = min(dt_ref, max(span / 20.0, 0.25), 2.0)

    n_ref = int(np.clip(np.ceil(span / dt_ref), 20, 1200))
    mc_edges = np.linspace(float(t0), float(t100), n_ref + 1, dtype=float)

    src_hist, _ = np.histogram(times_in_range, bins=mc_edges)
    if bkg_in_range.size > 0 and alpha is not None and alpha > 0:
        bkg_hist_ref, _ = np.histogram(bkg_in_range, bins=mc_edges)
        bkg_ref = float(alpha) * bkg_hist_ref.astype(float)
    else:
        bkg_ref = np.zeros_like(src_hist, dtype=float)

    t90_samples = []
    t50_samples = []

    for _ in range(n_mc):
        src_mc = rng.poisson(src_hist.astype(float))
        bkg_mc = rng.poisson(bkg_ref) if bkg_ref.size > 0 else bkg_ref
        seg_net_mc = np.maximum(src_mc - bkg_mc, 0.0)
        total_net_mc = float(np.sum(seg_net_mc))

        if total_net_mc <= 0.0:
            continue

        left = mc_edges[:-1]
        right = mc_edges[1:]
        t90_mc = _cross_target_time(0.95 * total_net_mc, left, right, seg_net_mc) - \
             _cross_target_time(0.05 * total_net_mc, left, right, seg_net_mc)
        t50_mc = _cross_target_time(0.75 * total_net_mc, left, right, seg_net_mc) - \
             _cross_target_time(0.25 * total_net_mc, left, right, seg_net_mc)
        t90_samples.append(t90_mc)
        t50_samples.append(t50_mc)

    if len(t90_samples) < 10:
        return np.nan, np.nan

    t90_samples = np.array(t90_samples)
    t50_samples = np.array(t50_samples)

    t90_p16, t90_p84 = np.percentile(t90_samples, [16, 84])
    t50_p16, t50_p84 = np.percentile(t50_samples, [16, 84])

    t90_stat = max((t90_p84 - t90_p16) / 2.0, 0.0)
    t50_stat = max((t50_p84 - t50_p16) / 2.0, 0.0)

    return float(t90_stat), float(t50_stat)


def compute_burst_txx(
    times: np.ndarray,
    bkg_times: Optional[np.ndarray] = None,
    alpha: Optional[float] = None,
    dt: float = 1.0,
    gamma: Optional[float] = None,
    p0: float = 0.05,
    ncp_prior: Optional[float] = None,
    fitness: Literal['events', 'measures', 'regular_events'] = 'events',
    threshold: float = 3.0,
    rates: Optional[np.ndarray] = None,
    errors: Optional[np.ndarray] = None,
    cumulative_mode: Literal['adaptive', 'fixed'] = 'adaptive',
    compute_errors: bool = False,
    n_mc: int = 100,
    mc_seed: int = 42,
) -> dict:
    """
    使用 Bayesian Blocks + LIMA 方法计算爆发时标 T0, T100, T90, T50。

    方法 (参考 A&A 2021 Sec 5.4 和 Systematic Effects on Duration Measurements of GRBs):
        1. 使用 Bayesian Blocks 对事件时间进行分块 (fitness='events')
        2. 计算每个块的 LIMA 信噪比
        3. T0 = 第一个 σ>threshold 块的左边
        4. T100 = 最后一个 σ>threshold 块的右边
        5. 在 T0-T100 范围内计算 T90/T50

    参数:
        times: 事件时间数组 (秒)
        bkg_times: 背景事件时间数组 (可选)
        alpha: 背景缩放因子 = BACKSCAL_src / BACKSCAL_bkg (可选)
        dt: 时间 bin 宽度 (秒)，用于 T90/T50 计算 (fixed 模式) 或自适应分段的最小宽度
        gamma: Bayesian blocks 先验参数 (可选)
        p0: 假警报概率，默认 0.05
        ncp_prior: 先验参数 (可选)
        fitness: 拟合函数类型，默认 'events'
        threshold: LIMA 信噪比阈值，默认 3.0
        rates: 通量/计数率数组 (可选，fitness='measures' 时用)
        errors: 误差数组 (可选)
        cumulative_mode: 'adaptive' (使用 BB 边界) 或 'fixed' (使用等宽分段)
        compute_errors: 是否计算误差估计
        n_mc: MC 重采样次数 (用于统计误差)
        mc_seed: MC 随机种子

    返回:
        dict: 包含 T0, T100, T90, T50 及其边界时间、分块信息等，
              以及误差字段 (当 compute_errors=True 时):
              - t90_err_stat, t90_err_sys, t90_err_tot
              - t50_err_stat, t50_err_sys, t50_err_tot
    """
    if len(times) < 2:
        return {
            'T0': None, 'T100': None, 'T90': None, 'T50': None,
            'edges': None, 'block_info': None,
            'cumulative_mode': cumulative_mode,
        }

    times = np.asarray(times, dtype=float)
    times = times[np.isfinite(times)]
    t_min, t_max = times.min(), times.max()
    duration = t_max - t_min

    if duration <= 0:
        return {
            'T0': None, 'T100': None, 'T90': None, 'T50': None,
            'edges': None, 'block_info': None,
            'cumulative_mode': cumulative_mode,
        }

    times_rel = times - t_min

    edges_rel = bayesian_blocks(
        times_rel,
        fitness=fitness,
        gamma=gamma,
        p0=p0,
        ncp_prior=ncp_prior
    )
    edges_rel = np.asarray(edges_rel, dtype=float)
    edges_rel = edges_rel[np.isfinite(edges_rel)]

    if edges_rel.size == 0:
        edges_rel = np.asarray([0.0, duration], dtype=float)
    edges_rel = np.clip(np.unique(edges_rel), 0.0, duration)

    if edges_rel.size < 2:
        edges_rel = np.asarray([0.0, duration], dtype=float)
    else:
        if edges_rel[0] > 0.0:
            edges_rel = np.concatenate(([0.0], edges_rel))
        if edges_rel[-1] < duration:
            edges_rel = np.concatenate((edges_rel, [duration]))

    edges_time = edges_rel + t_min
    n_blocks = len(edges_rel) - 1

    times_sorted = np.sort(times)
    if bkg_times is not None:
        bkg_sorted = np.sort(np.asarray(bkg_times, dtype=float))
    else:
        bkg_sorted = np.array([], dtype=float)

    def _count_in(sorted_arr: np.ndarray, a: float, b: float) -> float:
        if sorted_arr.size == 0 or b <= a:
            return 0.0
        i0 = int(np.searchsorted(sorted_arr, a, side='left'))
        i1 = int(np.searchsorted(sorted_arr, b, side='left'))
        return float(max(i1 - i0, 0))

    block_info = []
    for i in range(n_blocks):
        a, b = float(edges_time[i]), float(edges_time[i + 1])
        S = _count_in(times_sorted, a, b)
        B_raw = _count_in(bkg_sorted, a, b)
        B = float(alpha) * B_raw if alpha is not None and alpha > 0 else 0.0

        net = S - B
        var = max(S + B, 1e-12)

        if alpha is not None and alpha > 0 and B_raw > 0:
            s_pos = max(float(S), 0.0)
            b_pos = max(float(B_raw), 0.0)
            try:
                snr = float(snr_li_ma(s_pos, b_pos, float(alpha)))
                if net < 0 and np.isfinite(snr):
                    snr = -abs(snr)
            except Exception:
                snr = net / np.sqrt(var) if var > 0 else 0.0
        else:
            snr = net / np.sqrt(var) if var > 0 else 0.0

        block_info.append({
            'index': i,
            't_start': a,
            't_stop': b,
            'width': b - a,
            'S': S,
            'B_raw': B_raw,
            'B': B,
            'SNR': snr,
            'high_snr': snr > threshold
        })

    high_snr_blocks = [b for b in block_info if b['high_snr']]

    if not high_snr_blocks:
        result = {
            'T0': None,
            'T100': None,
            'T90': None,
            'T50': None,
            'T0_start': None,
            'T100_stop': None,
            'T90_start': None,
            'T90_stop': None,
            'T50_start': None,
            'T50_stop': None,
            'edges': edges_time,
            'block_info': block_info,
            'n_blocks': n_blocks,
            'threshold': threshold,
            'cumulative_mode': cumulative_mode,
            'message': f'No blocks with SNR > {threshold}'
        }
        return result

    T0 = high_snr_blocks[0]['t_start']
    T100 = high_snr_blocks[-1]['t_stop']

    if cumulative_mode == 'adaptive':
        bb_edges = np.array([b['t_start'] for b in block_info] + [block_info[-1]['t_stop']])
        seg_edges = np.unique(np.concatenate([bb_edges, np.array([T0, T100])]))
        seg_edges = seg_edges[(seg_edges >= T0) & (seg_edges <= T100)]
        if seg_edges.size < 2:
            seg_edges = np.asarray([T0, T100], dtype=float)
        else:
            if seg_edges[0] > T0:
                seg_edges = np.concatenate([[T0], seg_edges])
            if seg_edges[-1] < T100:
                seg_edges = np.concatenate([seg_edges, [T100]])
            seg_edges = np.unique(seg_edges)
    else:
        seg_edges = np.arange(T0, T100 + dt, dt, dtype=float)
        if seg_edges.size < 2:
            seg_edges = np.asarray([T0, T100], dtype=float)
        elif seg_edges[-1] < T100:
            seg_edges = np.append(seg_edges, T100)

    times_in_range = times[(times >= T0) & (times <= T100)]
    bkg_in_range = bkg_sorted[(bkg_sorted >= T0) & (bkg_sorted <= T100)] if bkg_sorted.size > 0 else np.array([])

    t90, t90_start, t90_stop, t50, t50_start, t50_stop = _compute_txx_from_cumulative(
        T0, T100, times, bkg_times, alpha, seg_edges
    )

    result = {
        'T0': T0,
        'T100': T100,
        'T90': t90,
        'T50': t50,
        'T0_start': T0,
        'T100_stop': T100,
        'T90_start': t90_start,
        'T90_stop': t90_stop,
        'T50_start': t50_start,
        'T50_stop': t50_stop,
        'edges': edges_time,
        'block_info': block_info,
        'n_blocks': n_blocks,
        'n_high_snr_blocks': len(high_snr_blocks),
        'threshold': threshold,
        'dt': dt,
        'p0': p0,
        'alpha': alpha,
        'gamma': gamma,
        'cumulative_mode': cumulative_mode,
        'seg_edges': seg_edges,
    }

    if compute_errors:
        t90_stat, t50_stat = _poisson_mc_stat_error(
            times, bkg_times, alpha, T0, T100, seg_edges,
            n_mc=n_mc, seed=mc_seed
        )

        sys_dts = [0.5, 1.0, 2.0, 5.0] if cumulative_mode == 'adaptive' else [0.5, 1.0, 5.0]
        t90_sys_samples = []
        t50_sys_samples = []

        for sys_dt in sys_dts:
            sys_seg = np.arange(T0, T100 + sys_dt, sys_dt, dtype=float)
            if sys_seg.size < 2:
                continue
            elif sys_seg[-1] < T100:
                sys_seg = np.append(sys_seg, T100)

            t90_s, _, _, t50_s, _, _ = _compute_txx_from_cumulative(
                T0, T100, times, bkg_times, alpha, sys_seg
            )
            if np.isfinite(t90_s):
                t90_sys_samples.append(t90_s)
            if np.isfinite(t50_s):
                t50_sys_samples.append(t50_s)

        if len(t90_sys_samples) >= 2:
            t90_sys = float(np.std(t90_sys_samples, ddof=1))
        else:
            t90_sys = np.nan

        if len(t50_sys_samples) >= 2:
            t50_sys = float(np.std(t50_sys_samples, ddof=1))
        else:
            t50_sys = np.nan

        t90_tot = np.sqrt(t90_stat**2 + t90_sys**2) if np.isfinite(t90_stat) and np.isfinite(t90_sys) else np.nan
        t50_tot = np.sqrt(t50_stat**2 + t50_sys**2) if np.isfinite(t50_stat) and np.isfinite(t50_sys) else np.nan

        result['t90_err_stat'] = t90_stat
        result['t50_err_stat'] = t50_stat
        result['t90_err_sys'] = t90_sys
        result['t50_err_sys'] = t50_sys
        result['t90_err_tot'] = t90_tot
        result['t50_err_tot'] = t50_tot

    return result


def compute_txx_direct(times: np.ndarray, dt: float = 1.0) -> dict:
    """
    直接从事件时间计算 T100, T90, T50，不使用 Bayesian Blocks。

    使用累积计数方法：
    - T100: 5% - 95% 累积计数对应的时间
    - T90:  5% - 95% 累积计数对应的时间
    - T50: 25% - 75% 累积计数对应的时间

    参数:
        times: 事件时间数组 (秒)
        dt: 时间 bin 宽度

    返回:
        dict: 包含 T100, T90, T50 及其边界时间
    """
    if len(times) < 10:
        return {'T100': None, 'T90': None, 'T50': None, 'edges': None}

    t_min, t_max = times.min(), times.max()
    duration = t_max - t_min

    if duration <= 0:
        return {'T100': None, 'T90': None, 'T50': None, 'edges': None}

    n_bins = max(int(duration / dt), 10)
    bin_edges = np.linspace(t_min - dt/2, t_max + dt/2, n_bins + 1)

    counts_per_bin, _ = np.histogram(times, bins=bin_edges)
    total_counts = np.sum(counts_per_bin)

    if total_counts == 0:
        return {'T100': None, 'T90': None, 'T50': None, 'edges': None}

    cumulative = np.cumsum(counts_per_bin)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    def find_time_at_fraction(frac: float) -> float:
        target = frac * total_counts
        idx = np.searchsorted(cumulative, target)
        if idx >= len(bin_centers):
            return bin_centers[-1]
        if idx == 0:
            return bin_centers[0]
        return bin_centers[idx]

    t5 = find_time_at_fraction(0.05)
    t95 = find_time_at_fraction(0.95)
    t25 = find_time_at_fraction(0.25)
    t75 = find_time_at_fraction(0.75)

    return {
        'T100': t95 - t5 if t5 is not None and t95 is not None else None,
        'T90': t95 - t5 if t5 is not None and t95 is not None else None,
        'T50': t75 - t25 if t25 is not None and t75 is not None else None,
        'T100_start': t5,
        'T100_stop': t95,
        'T90_start': t5,
        'T90_stop': t95,
        'T50_start': t25,
        'T50_stop': t75,
        'total_counts': total_counts,
    }


def compute_txx_from_lc(
    lc_file: str,
    tcol: str = 'TIME',
    rcol: str = 'RATE',
    ecol: str = 'ERROR',
    gamma: Optional[float] = None,
    p0: float = 0.05,
    method: Literal['direct', 'bayesian'] = 'direct',
    dt: float = 1.0,
) -> dict:
    """
    从光变曲线文件计算 T100, T90, T50。

    参数:
        lc_file: 光变曲线 FITS 文件路径
        tcol: 时间列名
        rcol: 通量/计数列名
        ecol: 误差列名
        gamma: Bayesian blocks 参数
        p0: 假警报概率
        method: 'direct' (直接累积) 或 'bayesian' (使用 Bayesian Blocks)
        dt: 时间 bin 宽度 (direct 模式用)

    返回:
        dict: 包含 T100, T90, T50 及其边界时间
    """
    from astropy.io import fits

    with fits.open(lc_file) as hdul:
        if tcol not in hdul[1].columns.names:
            raise ValueError(f'Column {tcol} not found')

        times = hdul[1].data[tcol]
        rate_col_exists = rcol in hdul[1].columns.names
        rates = hdul[1].data[rcol] if rate_col_exists else None
        errors = hdul[1].data[ecol] if ecol in hdul[1].columns.names else None

    if method == 'direct':
        if not rate_col_exists:
            return compute_txx_direct(times, dt=dt)

        rates_arr = np.asarray(rates)

        valid_mask = np.isfinite(times) & np.isfinite(rates_arr)
        if errors is not None:
            errors_arr = np.asarray(errors)
            valid_mask &= np.isfinite(errors_arr)

        times_valid = times[valid_mask]
        rates_valid = rates_arr[valid_mask]

        if len(times_valid) < 10:
            return {'T100': None, 'T90': None, 'T50': None, 'edges': None}

        return compute_txx_direct(times_valid, dt=dt)
    else:
        return compute_burst_txx(
            times=times,
            rates=rates,
            errors=errors,
            gamma=gamma,
            p0=p0,
            fitness='measures' if rates is not None else 'events',
        )


def compute_txx_from_eventdata(
    event_data,
    bkg_data=None,
    alpha: Optional[float] = None,
    dt: float = 1.0,
    gamma: Optional[float] = None,
    p0: float = 0.05,
    method: Literal['direct', 'bayesian'] = 'direct',
    threshold: float = 3.0,
    cumulative_mode: Literal['adaptive', 'fixed'] = 'adaptive',
    compute_errors: bool = False,
) -> dict:
    """
    从 EventData 对象计算 T0, T100, T90, T50。

    参数:
        event_data: EventData 对象
        bkg_data: 背景 EventData 对象 (可选)
        alpha: 背景缩放因子 (可选)
        dt: 时间 bin 宽度 (秒)
        gamma: Bayesian blocks 参数 (bayesian 模式用)
        p0: 假警报概率 (bayesian 模式用)
        method: 'direct' (直接累积) 或 'bayesian' (使用 Bayesian Blocks + LIMA)
        threshold: LIMA 信噪比阈值 (bayesian 模式用)
        cumulative_mode: 'adaptive' 或 'fixed'
        compute_errors: 是否计算误差

    返回:
        dict: 包含 T0, T100, T90, T50 及其边界时间
    """
    times = event_data.time
    if times is None:
        raise ValueError('No time data in EventData')

    if len(times) < 10:
        return {'T0': None, 'T100': None, 'T90': None, 'T50': None, 'edges': None}

    if method == 'bayesian':
        bkg_times = bkg_data.time if bkg_data is not None else None
        return compute_burst_txx(
            times=times,
            bkg_times=bkg_times,
            alpha=alpha,
            dt=dt,
            gamma=gamma,
            p0=p0,
            fitness='events',
            threshold=threshold,
            cumulative_mode=cumulative_mode,
            compute_errors=compute_errors,
        )
    else:
        return compute_txx_direct(times, dt=dt)


def compute_txx(
    input_data: Union[str, 'EventData'],
    tcol: str = 'TIME',
    rcol: str = 'RATE',
    ecol: str = 'ERROR',
    dt: float = 1.0,
    gamma: Optional[float] = None,
    p0: float = 0.05,
    method: Literal['direct', 'bayesian'] = 'direct',
) -> dict:
    """
    计算爆发时标 T100, T90, T50 的统一接口。

    参数:
        input_data: 光变曲线文件路径 或 EventData 对象
        tcol: 时间列名 (文件模式用)
        rcol: 通量列名 (文件模式用)
        ecol: 误差列名 (文件模式用)
        dt: 时间 bin 宽度 (direct 模式用)
        gamma: Bayesian blocks 参数 (bayesian 模式用)
        p0: 假警报概率 (bayesian 模式用)
        method: 'direct' (直接累积) 或 'bayesian' (使用 Bayesian Blocks)

    返回:
        dict: 包含 T100, T90, T50 及其边界时间
    """
    if isinstance(input_data, str):
        return compute_txx_from_lc(
            lc_file=input_data,
            tcol=tcol,
            rcol=rcol,
            ecol=ecol,
            gamma=gamma,
            p0=p0,
            method=method,
            dt=dt,
        )
    elif hasattr(input_data, 'time'):
        return compute_txx_from_eventdata(
            event_data=input_data,
            dt=dt,
            gamma=gamma,
            p0=p0,
            method=method,
        )
    else:
        raise ValueError('input_data must be a file path or EventData object')


def compute_cumulative_lightcurve(
    times: np.ndarray,
    rates: np.ndarray,
    gti_start: Optional[float] = None,
    gti_stop: Optional[float] = None,
    dt: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算累积光变曲线。

    参数:
        times: 时间数组
        rates: 通量/计数数组
        gti_start: GTI 开始时间
        gti_stop: GTI 结束时间
        dt: 时间 bin 宽度

    返回:
        (cum_times, cum_values): 累积时间和累积值
    """
    if gti_start is not None and gti_stop is not None:
        mask = (times >= gti_start) & (times <= gti_stop)
        t = times[mask]
        r = rates[mask]
    else:
        t = times
        r = rates

    sorted_idx = np.argsort(t)
    t_sorted = t[sorted_idx]
    r_sorted = r[sorted_idx]

    cum_values = np.cumsum(r_sorted)
    cum_times = t_sorted

    return cum_times, cum_values


__all__ = [
    'compute_burst_txx',
    'compute_txx_direct',
    'compute_txx_from_lc',
    'compute_txx_from_eventdata',
    'compute_txx',
    'compute_cumulative_lightcurve',
]
