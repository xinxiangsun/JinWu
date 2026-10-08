"""简化的 fextract 等价实现（纯 Python）。

提供函数 `extract`，把事件文件或 EventData 转换为 `PhaData`。
实现要点：时间/能量/region 过滤、通道计数直方、曝光估算（基于 GTI 或事件跨度）。

此实现尽量与原 fextract 的行为保持一致（对常见用例）。
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from ..core.base import OgipMeta
from ..core.data import PhaData, EventData
from ..core.io import read_evt
from ..core import gti as gtimod
from . import region as regionmod

def _estimate_exposure_from_eventdata(ev: EventData, *, tmin=None, tmax=None) -> float:
    def clipped_width(starts, stops):
        starts = np.asarray(starts, dtype=float)
        stops = np.asarray(stops, dtype=float)
        if tmin is not None:
            starts = np.maximum(starts, float(tmin))
        if tmax is not None:
            stops = np.minimum(stops, float(tmax))
        return float(np.maximum(stops - starts, 0.0).sum())

    if ev.gti_start is not None and ev.gti_stop is not None:
        ms, me = gtimod.merge_gti(ev.gti_start, ev.gti_stop)
        if ms is not None and me is not None and ms.size > 0:
            return clipped_width(ms, me)
    if ev.meta is not None and isinstance(ev.meta, OgipMeta):
        tstart = getattr(ev.meta, 'tstart', None)
        tstop = getattr(ev.meta, 'tstop', None)
        if tstart is not None and tstop is not None:
            return clipped_width([tstart], [tstop])
    if hasattr(ev, 'time') and ev.time is not None and getattr(ev.time, 'size', 0) > 0:
        return clipped_width([np.min(ev.time)], [np.max(ev.time)])
    return 0.0

def extract(path_or_ev: str | Path | EventData, *, region: Optional[dict] = None,
            tmin: Optional[float] = None, tmax: Optional[float] = None,
            ch_min: Optional[int] = None, ch_max: Optional[int] = None,
            nbins: Optional[int] = None, channel_col: str = 'pi') -> PhaData:
    """从事件文件或 EventData 提取 PHA（返回 PhaData）。

    参数与行为尽量与 fextract 保持一致的简单子集。
    """
    if isinstance(path_or_ev, (str, Path)):
        ev = read_evt(path_or_ev)
    else:
        ev = path_or_ev

    # 时间过滤
    if tmin is not None or tmax is not None:
        if tmin is not None and tmax is not None and tmin > tmax:
            raise ValueError('tmin must not exceed tmax')
        ev = ev.slice(tmin=tmin, tmax=tmax)

    # region 过滤（简单 dict 区域）
    if region is not None:
        ev = regionmod.apply_region_mask_to_events(ev, [region], invert=False)

    # 选择通道数组
    arr = None
    if channel_col == 'pi' and ev.pi is not None:
        arr = np.asarray(ev.pi, dtype=int)
    elif channel_col == 'channel' and ev.channel is not None:
        arr = np.asarray(ev.channel, dtype=int)
    else:
        if ev.pi is not None:
            arr = np.asarray(ev.pi, dtype=int)
        elif ev.channel is not None:
            arr = np.asarray(ev.channel, dtype=int)
        else:
            raise ValueError('No PI/CHANNEL column available for extraction')

    if ch_min is not None:
        arr = arr[arr >= int(ch_min)]
    if ch_max is not None:
        arr = arr[arr <= int(ch_max)]

    if arr.size == 0 and nbins is None:
        exposure = _estimate_exposure_from_eventdata(ev, tmin=tmin, tmax=tmax)
        # PhaData.kind 是 ClassVar，不需要在构造函数中传入
        return PhaData(path=ev.path, channels=np.array([], dtype=int), counts=np.array([], dtype=float),
                       stat_err=None, exposure=exposure, backscal=None, areascal=None,
                       quality=None, grouping=None, ebounds=None, header=ev.header, meta=ev.meta,
                       headers_dump=ev.headers_dump, columns=())

    if nbins is None:
        ch_lo = int(arr.min())
        ch_hi = int(arr.max())
        bins = np.arange(ch_lo, ch_hi + 2, dtype=int)
        channels = np.arange(ch_lo, ch_hi + 1, dtype=int)
        counts, _ = np.histogram(arr, bins=bins)
    else:
        bins = np.arange(0, int(nbins) + 1, dtype=int)
        channels = np.arange(0, int(nbins), dtype=int)
        counts, _ = np.histogram(arr, bins=bins)

    stat_err = np.sqrt(counts.astype(float))
    exposure = _estimate_exposure_from_eventdata(ev, tmin=tmin, tmax=tmax)

    # 构造 PhaData 时同样不需要传入 kind
    pha = PhaData(path=ev.path, channels=channels, counts=counts.astype(float), stat_err=stat_err,
                  exposure=exposure, backscal=None, areascal=None, quality=None, grouping=None, ebounds=None,
                  header=ev.header, meta=ev.meta, headers_dump=ev.headers_dump, columns=('CHANNEL', 'COUNTS'))
    return pha
