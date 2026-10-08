"""Shared GBM TTE reading and overlap/exposure handling.

Moved from subthreshold.data without numerical changes; legacy imports remain
available there. Times are Fermi MET internally and relative seconds at output.
"""
from pathlib import Path
import re
import numpy as np
import astropy.units as u
from .pipeline import _merge_intervals

def latest_products(paths):
    """Select the latest version of each named GBM archive product."""
    chosen = {}
    for path in sorted({Path(p).expanduser().resolve() for p in paths}):
        match = re.match(r"(.+)_v(\d+)\.fit(?:s)?(?:\.gz)?$", path.name)
        key, version = (match[1], int(match[2])) if match else (path.name, 0)
        if key not in chosen or version > chosen[key][0]:
            chosen[key] = (version, path)
    return tuple(v[1] for k, v in sorted(chosen.items()))


def merge_tte_events(event_sets):
    """Union overlapping file events, preserving within-file multiplicities.

    Inputs are pairs of arrays (absolute MET seconds, native channel). Returns
    sorted seconds and channels. For equal (time, channel), retain the largest
    multiplicity in any one file, not the sum across duplicate files.
    """
    dtype = np.dtype([("time", "f8"), ("channel", "i4")])
    keys, counts = [], []
    for times, channels in event_sets:
        rows = np.empty(len(times), dtype=dtype)
        rows["time"], rows["channel"] = times, channels
        unique, n = np.unique(rows, return_counts=True)
        keys.append(unique)
        counts.append(n)
    if not keys:
        return np.array([], dtype=float), np.array([], dtype=int)
    unique, inverse = np.unique(np.concatenate(keys), return_inverse=True)
    multiplicity = np.zeros(len(unique), dtype=int)
    np.maximum.at(multiplicity, inverse, np.concatenate(counts))
    rows = np.repeat(unique, multiplicity)
    return rows["time"], rows["channel"]


def interval_exposure(edges, intervals):
    """Geometric GTI overlap in seconds per bin, with overlapping GTIs merged."""
    edges = np.asarray(edges, dtype=float)
    exposure = np.zeros(len(edges) - 1)
    for start, stop in _merge_intervals(intervals):
        exposure += np.maximum(0., np.minimum(edges[1:], stop) - np.maximum(edges[:-1], start))
    return exposure


def read_detector_events(paths, trigger_time, interval):
    """Read matching TTE files, retaining counts, EBOUNDS, GTIs and dead times.

    ``interval`` is a seconds Quantity relative to scalar ``trigger_time``.
    Native channel boundaries and dead-time settings must agree across files.
    Returns event arrays in relative seconds plus merged GTIs and metadata.
    """
    from gdt.missions.fermi.gbm.tte import GbmTte
    t0 = float(trigger_time.to_value("fermi"))
    lo, hi = np.asarray(interval.to_value(u.s), dtype=float) + t0
    events, intervals, reference = [], [], None
    for path in latest_products(paths):
        tte = GbmTte.open(str(path))
        offset = float(tte.trigtime or 0.)
        bounds = np.array(tte.ebounds.as_list(), dtype=float)
        metadata = (bounds, float(tte.event_deadtime), float(tte.overflow_deadtime))
        if reference is not None and (not np.array_equal(bounds, reference[0]) or metadata[1:] != reference[1:]):
            raise ValueError("TTE energy calibration or dead times change across files")
        reference = metadata
        times = np.asarray(tte.data.times, dtype=float) + offset
        channels = np.asarray(tte.data.channels, dtype=int)
        if len(bounds) != 128 or np.any((channels < 0) | (channels >= 128)):
            raise ValueError("GTS templates require the native 128-channel GBM TTE layout")
        gti = [(max(lo, a + offset), min(hi, b + offset)) for a, b in tte.gti.as_list()
               if min(hi, b + offset) > max(lo, a + offset)]
        mask = np.zeros(times.size, dtype=bool)
        for a, b in gti:
            mask |= (times >= a) & (times < b)
        events.append((times[mask], channels[mask]))
        intervals.extend(gti)
        tte.close()
    if reference is None:
        raise ValueError("no TTE files for detector")
    times, channels = merge_tte_events(events)
    return times - t0, channels, [(a - t0, b - t0) for a, b in _merge_intervals(intervals)], reference
