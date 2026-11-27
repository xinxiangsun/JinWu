"""简化的 ftgrouppha 功能：按最小计数合并通道（grouping）。

提供 `group_min_counts(pha: PhaData, min_counts:int) -> PhaData`。

对齐 HEASoft 6.37 语义（`heasptools/ftgrouppha/ftgrouppha.cxx` →
`heacore/heasp/grouping.cxx grouping::loadMin` + `pha.cxx pha::setGrouping`）：

- 完整组：从左到右贪心累计，一旦组内计数和 >= min_counts 立即封组（与
  loadMin 的 "sum >= Minimum 即 -1 封组" 一致）。
- 尾组（未达阈值的剩余通道）：HEASoft 不把它们并成一组好数据，而是每个
  尾通道各自单列一组（GROUPING=1）并标 QUALITY=2（坏数据），拟合时被忽略。
  本实现因 PhaData 是物理合并模型，等价处理为：每个尾通道保持独立一箱，
  quality 强制 >= 2。
- 组质量：HEASoft `pha::rebinChannels` 的折叠规则是"组内任一通道质量非零，
  整组即坏"；这里先对尾通道强制 2，再按组首质量起步、其后最后一个非零成员
  覆盖的规则折叠，无 quality 输入时为 0。
"""
from __future__ import annotations

import numpy as np
from ..core.data import PhaData
from .grppha import _derived_pha, _load_min_groups

# 参考：HEASoft 6.37 heasptools/ftgrouppha/ftgrouppha.cxx（getMinCountsGrouping
#       → grouping::loadMin；setGrouping 应用分组并把坏通道强制自成组）；
#       heacore/heasp/grouping.cxx grouping::loadMin（尾组：istart..Last 全部
#       GROUPING=1 且 QUALITY=2）；heacore/heasp/pha.cxx pha::setGrouping /
#       pha::rebinChannels（"任一成员质量非零则整组坏"的折叠规则）。
def group_min_counts(pha: PhaData, min_counts: int) -> PhaData:
    if min_counts <= 0:
        return pha

    ch = np.asarray(pha.channels, dtype=int)
    cnt = np.asarray(pha.counts, dtype=float)
    if ch.size == 0:
        return pha

    qual = pha.quality
    qual_arr = (np.zeros(ch.size, dtype=int) if qual is None
                else np.asarray(qual, dtype=int).ravel())
    if qual_arr.size != ch.size:  # 输入 quality 长度异常时按无质量处理
        qual_arr = np.zeros(ch.size, dtype=int)

    # 贪心分组：组内和 >= min_counts 即封组；尾通道各自单列一组（loadMin 语义）
    groups, tail_start = _load_min_groups(cnt, min_counts)
    qual_arr = qual_arr.copy()
    qual_arr[tail_start:] = 2
    new_channels = np.empty(len(groups), dtype=int)
    new_counts = np.empty(len(groups), dtype=float)
    new_quality = np.empty(len(groups), dtype=int)
    for row, members in enumerate(groups):
        first, stop = members[0], members[-1] + 1
        new_channels[row] = ch[first]
        new_counts[row] = cnt[first:stop].sum()
        quality = qual_arr[first:stop]
        flagged = np.flatnonzero(quality)
        new_quality[row] = quality[flagged[-1]] if flagged.size else 0

    new_stat_err = np.sqrt(new_counts)  # Poisson: var ~ counts
    return _derived_pha(pha, channels=new_channels, counts=new_counts,
                        stat_err=new_stat_err, quality=new_quality, grouping=None,
                        ebounds=pha.ebounds)
