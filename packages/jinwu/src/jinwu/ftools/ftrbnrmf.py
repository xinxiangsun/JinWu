"""ARF / RMF 重分箱（简化的纯 Python 实现）。

此模块提供两个主要函数：
- `rebin_arf(elo, ehi, area, new_elo, new_ehi) -> new_area`
    把给定的 ARF（每个能量区间恒定的有效面积）重分箱到新的能量区间。
    算法：把原始区间视为分段常数，按重叠能量段宽加权并归一化到新 bin 宽。

- `rebin_rmf(matrix, channel_map, row_map=None) -> new_matrix`
    把 RMF 矩阵（2D numpy 数组）按列（通道）进行重分箱。`channel_map` 是长度
    为 old_nchan 的整型数组，指定每个旧通道映射到的新通道索引。可选的 `row_map`
    用于对能量行进行重分箱（同 channel_map 语义）。

这些实现为简化实用版，适用于常见的重分箱场景。若需要完全还原 HEASOFT 行为，
建议使用原始工具或基于 `external_sources` 的源实现进行逐行对比与回归测试。
"""
from __future__ import annotations

import numpy as np
from typing import Optional


def _interval_overlap(a0, a1, b0, b1):
    """返回区间 [a0,a1) 与 [b0,b1) 的重叠宽度（>=0）。"""
    lo = max(a0, b0)
    hi = min(a1, b1)
    return max(0.0, hi - lo)


def rebin_arf(elo: np.ndarray, ehi: np.ndarray, area: np.ndarray, new_elo: np.ndarray, new_ehi: np.ndarray) -> np.ndarray:
    """把 ARF 从原始能量网格重分箱到新网格。

    参数:
    - `elo`, `ehi`: 原 ARF 的能量区间边界数组（同长度 N，表示 N 个区间 [elo[i], ehi[i])）
    - `area`: 每个原区间的有效面积（长度 N）
    - `new_elo`, `new_ehi`: 目标网格的区间边界数组（长度 M）

    返回:
    - `new_area`: 长度 M 的数组，表示目标每个区间的面积（按区间平均值，即面积/能量宽度）

    说明: 我们把原 ARF 视为在每个原区间上常数 area[i]，对新区间计算面积的加权平均：
        new_area[j] = (1 / width_new_j) * sum_i area[i] * overlap_width(i,j)
    这是将有效面积在能量上做平均的一种保守实现。
    """
    elo = np.asarray(elo, dtype=float)
    ehi = np.asarray(ehi, dtype=float)
    area = np.asarray(area, dtype=float)
    new_elo = np.asarray(new_elo, dtype=float)
    new_ehi = np.asarray(new_ehi, dtype=float)

    if not (elo.size == ehi.size == area.size):
        raise ValueError('elo/ehi/area must have same length')
    if not (new_elo.size == new_ehi.size):
        raise ValueError('new_elo/new_ehi must have same length')

    M = new_elo.size
    new_area = np.zeros(M, dtype=float)
    for j in range(M):
        wNew = float(new_ehi[j] - new_elo[j])
        if wNew <= 0:
            new_area[j] = 0.0
            continue
        acc = 0.0
        for i in range(elo.size):
            ol = _interval_overlap(elo[i], ehi[i], new_elo[j], new_ehi[j])
            if ol <= 0.0:
                continue
            acc += float(area[i]) * ol
        # average over new bin width
        new_area[j] = acc / wNew
    return new_area


def rebin_rmf(matrix: np.ndarray, channel_map: np.ndarray, row_map: Optional[np.ndarray] = None) -> np.ndarray:
    """对 RMF 矩阵进行重分箱。

    参数:
    - `matrix`: 2D numpy 数组，形状 (n_rows, n_old_channels)。按列表示旧通道。
    - `channel_map`: 长度 n_old_channels 的整型数组，channel_map[i] 指定旧通道 i 映射到的新通道索引（0..n_new-1）。
    - `row_map`: 可选，长度 n_rows 的整型数组，指定每行（能量行）映射到的新行索引（0..n_new_rows-1）。

    返回:
    - `new_matrix`: 2D numpy 数组，形状 (n_new_rows, n_new_channels) 或 (n_rows, n_new_channels)（当 row_map 为 None 时保留原行数）。

    该实现按映射索引直接对矩阵行/列求和，保持概率守恒（即把对应列累加到新列）。
    """
    mat = np.asarray(matrix, dtype=float)
    if mat.ndim != 2:
        raise ValueError('matrix must be 2D')
    nrows, ncols = mat.shape
    channel_map = np.asarray(channel_map, dtype=int)
    if channel_map.size != ncols:
        raise ValueError('channel_map length must equal number of columns in matrix')
    n_new_ch = int(channel_map.max()) + 1 if channel_map.size else 0

    if row_map is None:
        # only rebin columns
        new_mat = np.zeros((nrows, n_new_ch), dtype=float)
        for icol in range(ncols):
            tgt = int(channel_map[icol])
            new_mat[:, tgt] += mat[:, icol]
        return new_mat
    else:
        row_map = np.asarray(row_map, dtype=int)
        if row_map.size != nrows:
            raise ValueError('row_map length must equal number of rows in matrix')
        n_new_row = int(row_map.max()) + 1
        new_mat = np.zeros((n_new_row, n_new_ch), dtype=float)
        for irow in range(nrows):
            rtarget = int(row_map[irow])
            for icol in range(ncols):
                ctarget = int(channel_map[icol])
                new_mat[rtarget, ctarget] += mat[irow, icol]
        return new_mat


def map_bins_by_edges(old_lo: np.ndarray, old_hi: np.ndarray, new_lo: np.ndarray, new_hi: np.ndarray) -> np.ndarray:
    """辅助：根据能量区间把旧区间映射到目标区间索引（按中心归属）。

    对于每个旧区间，计算中心 = 0.5*(old_lo+old_hi)，并找出它落在哪个 new 区间内，返回所对应的新索引；
    若不落入任何区间，返回 -1（调用者可据此处理或忽略）。
    """
    old_lo = np.asarray(old_lo, dtype=float)
    old_hi = np.asarray(old_hi, dtype=float)
    new_lo = np.asarray(new_lo, dtype=float)
    new_hi = np.asarray(new_hi, dtype=float)
    centers = 0.5 * (old_lo + old_hi)
    idx = np.full(centers.size, -1, dtype=int)
    for i, c in enumerate(centers):
        for j in range(new_lo.size):
            if (c >= new_lo[j]) and (c < new_hi[j]):
                idx[i] = j
                break
    return idx
