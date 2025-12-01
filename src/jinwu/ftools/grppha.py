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
from astropy.io import fits

from ..core.file import PhaData, read_pha

__all__ = [
    'compute_grouping_by_min_counts', 'read_groupfile', 'grppha', 'write_grouped_pha',
]


def compute_grouping_by_min_counts(counts: np.ndarray, min_counts: float) -> np.ndarray:
    """Compute grouping array given per-channel `counts` and `min_counts`.

    Returns an integer array `grouping` of same length as `counts`, where
    grouping[i] is the group id (1-based) that channel i belongs to.

    Algorithm: greedy left-to-right accumulate counts until >= min_counts,
    then start a new group. The last group may have < min_counts.
    """
    counts = np.asarray(counts, dtype=float)
    n = counts.size
    grouping = np.zeros(n, dtype=int)
    if n == 0:
        return grouping
    gid = 1
    acc = 0.0
    for i in range(n):
        acc += float(counts[i])
        grouping[i] = gid
        if acc >= float(min_counts) and i < n - 1:
            gid += 1
            acc = 0.0
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
    """Create grouping array from explicit ranges; channels outside any range get 0."""
    ch = np.asarray(channels, dtype=int)
    g = np.zeros(ch.size, dtype=int)
    gid = 1
    for a, b in ranges:
        mask = (ch >= int(a)) & (ch <= int(b))
        g[mask] = gid
        gid += 1
    return g


def write_grouped_pha(pha: PhaData, outpath: str | Path, grouping: np.ndarray, *, overwrite: bool = False) -> Path:
    """Write a new PHA file containing GROUPING column (aligned with channels).

    Columns: CHANNEL, COUNTS, STAT_ERR (if present), GROUPING.
    Primary header copies some meta if available.
    """
    outp = Path(outpath)
    if outp.exists() and not overwrite:
        raise FileExistsError(str(outp))

    ch = np.asarray(pha.channels, dtype=int)
    cnt = np.asarray(pha.counts, dtype=float)
    cols = [fits.Column(name='CHANNEL', format='J', array=ch),
            fits.Column(name='COUNTS', format='E', array=cnt)]
    if pha.stat_err is not None:
        cols.append(fits.Column(name='STAT_ERR', format='E', array=np.asarray(pha.stat_err, dtype=float)))
    cols.append(fits.Column(name='GROUPING', format='J', array=np.asarray(grouping, dtype=int)))

    hdu_spec = fits.BinTableHDU.from_columns(cols, name='SPECTRUM')
    # attach some keys
    hdr = hdu_spec.header
    hdr['EXPOSURE'] = float(pha.exposure) if pha.exposure is not None else 0.0
    hdr['EXTNAME'] = 'SPECTRUM'

    prih = fits.PrimaryHDU()
    try:
        if pha.meta is not None:
            m = pha.meta
            if getattr(m, 'instrume', None):
                prih.header['INSTRUME'] = m.instrume
            if getattr(m, 'telescop', None):
                prih.header['TELESCOP'] = m.telescop
            if getattr(m, 'tstart', None) is not None:
                prih.header['TSTART'] = float(m.tstart)
            if getattr(m, 'tstop', None) is not None:
                prih.header['TSTOP'] = float(m.tstop)
    except Exception:
        pass

    hdul = fits.HDUList([prih, hdu_spec])
    hdul.writeto(outp, overwrite=overwrite)
    return outp


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
        newpha = PhaData(kind='pha', path=pha.path, channels=pha.channels, counts=pha.counts,
                         stat_err=pha.stat_err, exposure=pha.exposure, backscal=pha.backscal,
                         areascal=pha.areascal, quality=pha.quality, grouping=grouping,
                         ebounds=pha.ebounds, header=pha.header, meta=pha.meta, headers_dump=pha.headers_dump, columns=pha.columns)
        if outfile is not None:
            write_grouped_pha(newpha, outfile, grouping, overwrite=overwrite)
        return newpha

    # rebin: collapse channels into groups
    gids = np.unique(grouping[grouping > 0])
    new_channels = []
    new_counts = []
    new_stat = []
    for gid in gids:
        mask = grouping == int(gid)
        if not np.any(mask):
            continue
        new_channels.append(int(ch[mask][0]))
        s = float(np.sum(cnt[mask]))
        new_counts.append(s)
        new_stat.append(float(np.sqrt(s)))

    newpha = PhaData(kind='pha', path=pha.path, channels=np.asarray(new_channels, dtype=int), counts=np.asarray(new_counts, dtype=float),
                     stat_err=np.asarray(new_stat, dtype=float), exposure=pha.exposure, backscal=pha.backscal,
                     areascal=pha.areascal, quality=None, grouping=None,
                     ebounds=None, header=pha.header, meta=pha.meta, headers_dump=pha.headers_dump, columns=('CHANNEL','COUNTS'))

    if outfile is not None:
        # write rebinned spectrum: simple BinTable with CHANNEL/COUNTS/STAT_ERR
        outp = Path(outfile)
        if outp.exists() and not overwrite:
            raise FileExistsError(str(outp))
        cols = [fits.Column(name='CHANNEL', format='J', array=newpha.channels),
                fits.Column(name='COUNTS', format='E', array=newpha.counts)]
        if newpha.stat_err is not None:
            cols.append(fits.Column(name='STAT_ERR', format='E', array=newpha.stat_err))
        hdu_spec = fits.BinTableHDU.from_columns(cols, name='SPECTRUM')
        hdu_spec.header['EXPOSURE'] = float(newpha.exposure) if newpha.exposure is not None else 0.0
        prih = fits.PrimaryHDU()
        hdul = fits.HDUList([prih, hdu_spec])
        hdul.writeto(outp, overwrite=overwrite)

    return newpha
