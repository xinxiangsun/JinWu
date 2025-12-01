"""简化的 ftgrouppha 功能：按最小计数合并通道（grouping）。

提供 `group_min_counts(pha: PhaData, min_counts:int) -> PhaData`。
这是一个保守的实现：从左到右合并相邻通道直到累计计数 >= min_counts。
"""
from __future__ import annotations

import numpy as np
from ..core.file import PhaData

def group_min_counts(pha: PhaData, min_counts: int) -> PhaData:
    if min_counts <= 0:
        return pha

    ch = np.asarray(pha.channels, dtype=int)
    cnt = np.asarray(pha.counts, dtype=float)
    if ch.size == 0:
        return pha

    new_channels = []
    new_counts = []
    new_stat_err = []

    acc_cnt = 0.0
    acc_chs = []
    acc_err2 = 0.0
    for c, v in zip(ch, cnt):
        acc_cnt += float(v)
        acc_chs.append(int(c))
        acc_err2 += float(v)  # poisson var ~ counts
        if acc_cnt >= min_counts:
            # produce a single output bin
            new_channels.append(int(acc_chs[0]))
            new_counts.append(acc_cnt)
            new_stat_err.append(np.sqrt(acc_err2))
            acc_cnt = 0.0
            acc_chs = []
            acc_err2 = 0.0

    # flush remainder
    if acc_chs:
        new_channels.append(int(acc_chs[0]))
        new_counts.append(acc_cnt)
        new_stat_err.append(np.sqrt(acc_err2))

    return PhaData(kind=pha.kind, path=pha.path, channels=np.array(new_channels, dtype=int),
                   counts=np.array(new_counts, dtype=float), stat_err=np.array(new_stat_err, dtype=float),
                   exposure=pha.exposure, backscal=pha.backscal, areascal=pha.areascal,
                   quality=pha.quality, grouping=None, ebounds=pha.ebounds, header=pha.header,
                   meta=pha.meta, headers_dump=pha.headers_dump, columns=pha.columns)
