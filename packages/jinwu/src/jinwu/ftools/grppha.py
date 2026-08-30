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
    'compute_grouping_by_min_counts', 'read_groupfile', 'grppha', 'write_grouped_pha',
]


def compute_grouping_by_min_counts(counts: np.ndarray, min_counts: float) -> np.ndarray:
    """Compute grouping array given per-channel `counts` and `min_counts`.

    Returns an OGIP-style integer array `grouping` of same length as `counts`, where
    `1` marks the start of a group, `-1` marks continuation in the same group,
    and `0` means channel ignored (not produced by this function).

    Algorithm: greedy left-to-right accumulate counts until >= min_counts,
    then start a new group. The last group may have < min_counts.
    """
    counts = np.asarray(counts, dtype=float)
    n = counts.size
    grouping = np.zeros(n, dtype=int)
    if n == 0:
        return grouping
    acc = 0.0
    is_new_group = True
    for i in range(n):
        acc += float(counts[i])
        grouping[i] = 1 if is_new_group else -1
        if acc >= float(min_counts) and i < n - 1:
            acc = 0.0
            is_new_group = True
        else:
            is_new_group = False
    return grouping


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
        # produce PhaData with same channels but grouping array filled
        newpha = PhaData( path=pha.path, channels=pha.channels, counts=pha.counts,
                         stat_err=pha.stat_err, exposure=pha.exposure, backscal=pha.backscal,
                         areascal=pha.areascal, quality=pha.quality, grouping=grouping,
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

    newpha = PhaData(path=pha.path, channels=np.asarray(new_channels, dtype=int), counts=np.asarray(new_counts, dtype=float),
                     stat_err=np.asarray(new_stat, dtype=float), exposure=pha.exposure, backscal=pha.backscal,
                     areascal=pha.areascal, quality=None, grouping=None,
                     ebounds=None, header=pha.header, meta=pha.meta, headers_dump=pha.headers_dump, columns=('CHANNEL','COUNTS'))

    if outfile is not None:
        write_pha_core(newpha, outfile, overwrite=overwrite)

    return newpha
