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
  整组即坏"；这里取组内成员 quality 的最大值（最坏者），无 quality 输入时为 0。
"""
from __future__ import annotations

import numpy as np
from ..core.data import PhaData

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

    # 贪心分组：组内和 >= min_counts 即封组（loadMin 语义）
    member_groups: list[list[int]] = []
    cur: list[int] = []
    acc = 0.0
    for i, v in enumerate(cnt):
        cur.append(i)
        acc += float(v)
        if acc >= float(min_counts):
            member_groups.append(cur)
            cur = []
            acc = 0.0
    # 尾组：未达阈值的剩余通道各自单列一组（HEASoft: GROUPING=1 + QUALITY=2）
    tail_members = cur
    if tail_members:
        member_groups.extend([i] for i in tail_members)
    tail_set = set(tail_members)

    new_channels = []
    new_counts = []
    new_stat_err = []
    new_quality = []
    for members in member_groups:
        idx = np.asarray(members, dtype=int)
        new_channels.append(int(ch[idx[0]]))
        s = float(cnt[idx].sum())
        new_counts.append(s)
        new_stat_err.append(float(np.sqrt(s)))  # Poisson: var ~ counts
        member_quality = [2 if int(i) in tail_set else int(qual_arr[i]) for i in idx]
        q = next((value for value in reversed(member_quality) if value != 0), member_quality[0])
        new_quality.append(q)

    return PhaData(path=pha.path, channels=np.array(new_channels, dtype=int),
                   counts=np.array(new_counts, dtype=float), stat_err=np.array(new_stat_err, dtype=float),
                   exposure=pha.exposure, backscal=pha.backscal, areascal=pha.areascal,
                   quality=np.array(new_quality, dtype=int), grouping=None, ebounds=pha.ebounds, header=pha.header,
                   meta=pha.meta, headers_dump=pha.headers_dump, columns=pha.columns)
