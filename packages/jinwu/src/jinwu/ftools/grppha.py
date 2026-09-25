"""Pure-Python minimal implementation of grppha-like grouping.

Features implemented:
- grouping by minimum counts per group (greedy left-to-right)
- reading a simple group file (text lines: "start end" inclusive channel ranges)
- option to write GROUPING column into a new PHA file, or to produce a rebinned PHA

This aims to reproduce the common grppha use-case: ensure each output group
has at least `min_counts` counts by merging adjacent channels.

Limitations / intentionally simplified behaviour:
- Does not implement all historical grppha options (e.g. SNR-based grouping,
  complex mapping files with weights, interactive modes, or advanced header
  keyword edits). Focused on batch grouping behavior.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np

from ..core.data import PhaData
from ..core.io import read_pha, write_pha as write_pha_core

__all__ = [
    'compute_grouping_by_min_counts', 'min_counts_group_quality', 'fold_group_quality',
    'read_groupfile', 'grppha', 'write_grouped_pha',
]


# 方法：min-counts 分组——从左到右贪心累计，组内和一旦 >= min_counts 立即封组；
#       末尾累计不足 min_counts 的"尾组"通道各自单列一组（GROUPING=1）并标
#       QUALITY=2（坏数据），而不是并成一个正常组（HEASoft grouping::loadMin：
#       `for i in istart..Last: m_groupingFlag[i]=1; m_qualityFlag[i]=2`）。
# 参考：HEASoft 6.37 heacore/heasp/grouping.cxx grouping::loadMin；
#       heasptools/ftgrouppha/ftgrouppha.cxx（getMinCountsGrouping 调用路径）；
#       heacore/heasp/pha.cxx pha::setGrouping（坏通道强制自成组、质量取最坏）。
def compute_grouping_by_min_counts(counts: np.ndarray, min_counts: float) -> np.ndarray:
    """Compute grouping array given per-channel `counts` and `min_counts`.

    Returns an OGIP-style integer array `grouping` of same length as `counts`, where
    `1` marks the start of a group, `-1` marks continuation in the same group,
    and `0` means channel ignored (not produced by this function).

    Algorithm: greedy left-to-right accumulate counts until >= min_counts,
    then start a new group. Channels of the final incomplete group (sum <
    min_counts) each form their own single-channel group — HEASoft marks them
    QUALITY=2 via :func:`min_counts_group_quality`; use that companion to fold
    the tail flag into a QUALITY column.
    """
    counts = np.asarray(counts, dtype=float)
    n = counts.size
    grouping = np.zeros(n, dtype=int)
    if n == 0:
        return grouping
    acc = 0.0
    is_new_group = True
    istart = 0
    for i in range(n):
        acc += float(counts[i])
        grouping[i] = 1 if is_new_group else -1
        if acc >= float(min_counts):
            acc = 0.0
            is_new_group = True
            istart = i + 1
        else:
            is_new_group = False
    if istart < n:
        # 尾组：未达阈值的剩余通道各自单列一组（HEASoft loadMin 语义）
        grouping[istart:] = 1
    return grouping


def min_counts_group_quality(counts: np.ndarray, min_counts: float) -> np.ndarray:
    """Per-channel QUALITY flags accompanying :func:`compute_grouping_by_min_counts`.

    Channels of the final incomplete group (sum < min_counts) get QUALITY=2
    (bad); all others get 0 — the loadMin tail rule. Combine with any input
    per-channel quality via element-wise max before folding into output bins.
    """
    counts = np.asarray(counts, dtype=float)
    n = counts.size
    quality = np.zeros(n, dtype=int)
    if n == 0:
        return quality
    acc = 0.0
    istart = 0
    for i in range(n):
        acc += float(counts[i])
        if acc >= float(min_counts):
            acc = 0.0
            istart = i + 1
    if istart < n:
        quality[istart:] = 2
    return quality


def fold_group_quality(per_channel_quality: np.ndarray | None, gid_arr: np.ndarray,
                       gids: np.ndarray) -> np.ndarray | None:
    """Fold per-channel quality into per-group quality (HEASoft rebinChannels rule).

    Group quality starts as the start channel's quality; any later member with
    a non-zero quality overrides it (last non-zero wins — "any input element
    making up a bin has bad quality then the whole bin does"). Returns None if
    the input quality is None or its length mismatches the grouping array.
    """
    if per_channel_quality is None:
        return None
    q = np.asarray(per_channel_quality, dtype=int).ravel()
    if q.size != np.asarray(gid_arr).size:
        return None
    out = np.zeros(np.asarray(gids).size, dtype=int)
    for row, gid in enumerate(np.asarray(gids).ravel()):
        members = q[np.asarray(gid_arr) == int(gid)]
        if members.size == 0:
            continue
        qg = int(members[0])
        for m in members[1:]:
            if m != 0:
                qg = int(m)
        out[row] = qg
    return out


def read_groupfile(path: str | Path) -> List[Tuple[int, int]]:
    """Read a simple groupfile containing `start end` channel ranges (inclusive).

    Lines starting with '#' ignored. Returns a list of (start, end) tuples.
    """
    p = Path(path)
    out: List[Tuple[int, int]] = []
    with p.open('r') as fh:
        for ln in fh:
            s = ln.strip()
            if not s or s.startswith('#'):
                continue
            parts = s.split()
            if len(parts) < 2:
                continue
            try:
                a = int(parts[0])
                b = int(parts[1])
            except Exception:
                continue
            if b < a:
                a, b = b, a
            out.append((a, b))
    return out


def _grouping_from_ranges(channels: np.ndarray, ranges: List[Tuple[int, int]]) -> np.ndarray:
    """Create OGIP-style grouping flags from explicit ranges; channels outside ranges get 0."""
    ch = np.asarray(channels, dtype=int)
    g = np.zeros(ch.size, dtype=int)
    for a, b in ranges:
        mask = (ch >= int(a)) & (ch <= int(b))
        idx = np.where(mask)[0]
        if idx.size == 0:
            continue
        g[idx] = -1
        g[idx[0]] = 1
    return g


def _group_flags_to_ids(grouping: np.ndarray) -> np.ndarray:
    """Convert grouping (OGIP flags or legacy gid encoding) to 1-based group IDs."""
    g = np.asarray(grouping, dtype=int)
    if g.size == 0:
        return g
    nz = g[g != 0]
    if nz.size == 0:
        return np.zeros_like(g)
    is_flag = np.all(np.isin(nz, [-1, 1]))
    out = np.zeros_like(g)
    if is_flag:
        gid = 0
        for i, val in enumerate(g):
            if val == 0:
                out[i] = 0
            elif val == 1:
                gid += 1
                out[i] = gid
            elif val == -1:
                out[i] = gid if gid > 0 else 0
        return out
    # legacy group-id encoding already
    return np.where(g > 0, g, 0)


def write_grouped_pha(pha: PhaData, outpath: str | Path, grouping: np.ndarray, *, overwrite: bool = False) -> Path:
    """Write grouped PHA by delegating to unified `core.io.write_pha`."""
    grouped = PhaData(
        path=pha.path,
        channels=pha.channels,
        counts=pha.counts,
        rate=pha.rate,
        stat_err=pha.stat_err,
        exposure=pha.exposure,
        backscal=pha.backscal,
        areascal=pha.areascal,
        respfile=pha.respfile,
        ancrfile=pha.ancrfile,
        quality=pha.quality,
        grouping=np.asarray(grouping, dtype=int),
        ebounds=pha.ebounds,
        raw_spectrum_columns=pha.raw_spectrum_columns,
        header=pha.header,
        meta=pha.meta,
        headers_dump=pha.headers_dump,
        columns=pha.columns,
    )
    return write_pha_core(grouped, outpath, overwrite=overwrite)


def grppha(input_pha: str | PhaData, *, outfile: Optional[str] = None, min_counts: Optional[float] = None,
           groupfile: Optional[str] = None, rebin: bool = False, overwrite: bool = False) -> PhaData:
    """Main grppha-like entry.

    - `input_pha`: path to PHA file or `PhaData` instance
    - `min_counts`: if provided, compute grouping greedily by min counts
    - `groupfile`: optional path to explicit grouping ranges (overrides min_counts)
    - `rebin`: if True, collapse groups into rebinned PhaData (one row per group)
    - `outfile`: if provided, write a PHA file with GROUPING column (or rebinned spectrum)

    Returns `PhaData` (rebinned if requested, else original channels with `grouping` stored
    in `PhaData.grouping`).
    """
    # load
    if isinstance(input_pha, (str, Path)):
        pha = read_pha(str(input_pha))
    else:
        pha = input_pha

    ch = np.asarray(pha.channels, dtype=int)
    cnt = np.asarray(pha.counts, dtype=float)

    grouping = np.zeros(ch.size, dtype=int)

    if groupfile is not None:
        ranges = read_groupfile(groupfile)
        grouping = _grouping_from_ranges(ch, ranges)
    elif min_counts is not None:
        grouping = compute_grouping_by_min_counts(cnt, min_counts)
    else:
        raise ValueError('Either min_counts or groupfile must be provided')

    # attach grouping into result
    if not rebin:
        # 尾组通道标 QUALITY=2（HEASoft setGrouping：质量非零通道自成组、标记为坏）
        out_quality = pha.quality
        if min_counts is not None and groupfile is None:
            tail_q = min_counts_group_quality(cnt, min_counts)
            if pha.quality is None:
                out_quality = tail_q
            else:
                out_quality = np.maximum(np.asarray(pha.quality, dtype=int).ravel(), tail_q)
        # produce PhaData with same channels but grouping array filled
        newpha = PhaData( path=pha.path, channels=pha.channels, counts=pha.counts,
                         stat_err=pha.stat_err, exposure=pha.exposure, backscal=pha.backscal,
                         areascal=pha.areascal, quality=out_quality, grouping=grouping,
                         ebounds=pha.ebounds, header=pha.header, meta=pha.meta, headers_dump=pha.headers_dump, columns=pha.columns)
        if outfile is not None:
            write_grouped_pha(newpha, outfile, grouping, overwrite=overwrite)
        return newpha

    # rebin: collapse channels into groups
    gid_arr = _group_flags_to_ids(grouping)
    gids = np.unique(gid_arr[gid_arr > 0])
    new_channels = []
    new_counts = []
    new_stat = []
    for gid in gids:
        mask = gid_arr == int(gid)
        if not np.any(mask):
            continue
        new_channels.append(int(ch[mask][0]))
        s = float(np.sum(cnt[mask]))
        new_counts.append(s)
        new_stat.append(float(np.sqrt(s)))

    # 组质量折叠（rebinChannels 规则）；min_counts 路径先叠加尾组 QUALITY=2
    per_ch_quality = None
    if pha.quality is not None:
        per_ch_quality = np.asarray(pha.quality, dtype=int).ravel()
    if min_counts is not None and groupfile is None:
        tail_q = min_counts_group_quality(cnt, min_counts)
        per_ch_quality = (tail_q if per_ch_quality is None
                          else np.maximum(per_ch_quality, tail_q))
    new_quality = fold_group_quality(per_ch_quality, gid_arr, gids)

    newpha = PhaData(path=pha.path, channels=np.asarray(new_channels, dtype=int), counts=np.asarray(new_counts, dtype=float),
                     stat_err=np.asarray(new_stat, dtype=float), exposure=pha.exposure, backscal=pha.backscal,
                     areascal=pha.areascal, quality=new_quality, grouping=None,
                     ebounds=None, header=pha.header, meta=pha.meta, headers_dump=pha.headers_dump, columns=('CHANNEL','COUNTS'))

    if outfile is not None:
        write_pha_core(newpha, outfile, overwrite=overwrite)

    return newpha
