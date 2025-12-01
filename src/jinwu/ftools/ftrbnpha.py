"""简单的 PHA 重分箱工具（纯 Python）。

提供 `rebin_pha(pha: PhaData, nbins: int) -> PhaData`。
此实现会把连续通道等分到 `nbins` 个新通道上，或按因子合并已有通道。
"""
from __future__ import annotations

import numpy as np
from ..core.file import PhaData

def rebin_pha(pha: PhaData, nbins: int) -> PhaData:
    ch = np.asarray(pha.channels, dtype=int)
    cnt = np.asarray(pha.counts, dtype=float)
    if ch.size == 0:
        return pha

    ch_min = int(ch.min())
    ch_max = int(ch.max())
    nch = ch_max - ch_min + 1
    if nbins <= 0:
        raise ValueError('nbins must be > 0')

    # If requested nbins equals current channel count, return copy
    if nbins >= nch:
        # pad to match nbins if needed
        channels = np.arange(ch_min, ch_min + nbins, dtype=int)
        counts = np.zeros(nbins, dtype=float)
        # map existing counts into target indices
        for c, v in zip(ch, cnt):
            idx = int(c - ch_min)
            if idx < nbins:
                counts[idx] += v
        stat_err = np.sqrt(counts)
        return PhaData(kind=pha.kind, path=pha.path, channels=channels, counts=counts, stat_err=stat_err,
                       exposure=pha.exposure, backscal=pha.backscal, areascal=pha.areascal,
                       quality=pha.quality, grouping=None, ebounds=pha.ebounds, header=pha.header,
                       meta=pha.meta, headers_dump=pha.headers_dump, columns=pha.columns)

    # Merge factor
    factor = float(nch) / float(nbins)
    # compute target bin indices for each original channel
    target = np.floor((np.arange(nch) / factor)).astype(int)
    counts = np.zeros(nbins, dtype=float)
    for i, c in enumerate(range(ch_min, ch_max + 1)):
        # find value for channel c
        mask = (ch == c)
        if not np.any(mask):
            continue
        val = float(cnt[mask][0])
        idx = int(target[i])
        if idx >= nbins:
            idx = nbins - 1
        counts[idx] += val

    stat_err = np.sqrt(counts)
    channels = np.arange(0, nbins, dtype=int)
    return PhaData(kind=pha.kind, path=pha.path, channels=channels, counts=counts, stat_err=stat_err,
                   exposure=pha.exposure, backscal=pha.backscal, areascal=pha.areascal,
                   quality=pha.quality, grouping=None, ebounds=pha.ebounds, header=pha.header,
                   meta=pha.meta, headers_dump=pha.headers_dump, columns=pha.columns)
