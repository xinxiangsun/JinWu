"""Continuous TTE preparation with explicit time, exposure and GTI contracts."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import warnings
import numpy as np
import astropy.units as u

from jinwu.core.time import Time
from ..pipeline import _intersect_intervals, _merge_intervals, validate_background_residuals
from .models import CHANNEL_EDGES, seconds


from ..tte import latest_products, merge_tte_events, interval_exposure, read_detector_events


@dataclass
class PreparedSearchData:
    """Uniformly binned counts, live seconds and midpoint background estimates."""
    edges: np.ndarray
    counts: np.ndarray
    exposure: np.ndarray
    rates: np.ndarray
    uncertainty: np.ndarray
    background_valid: np.ndarray
    gti: np.ndarray
    detectors: tuple[str, ...]

    def save(self, path):
        np.savez_compressed(path, **{name: getattr(self, name) for name in self.__dataclass_fields__})

    @classmethod
    def open(cls, path):
        with np.load(path, allow_pickle=False) as archive:
            values = {name: archive[name] for name in cls.__dataclass_fields__}
        values["detectors"] = tuple(values["detectors"].tolist())
        return cls(**values)


def prepare_search_data(paths_by_detector, trigger_time, config):
    """Bin TTE and fit per-channel sliding backgrounds without spanning GTI gaps.

    Returns (PreparedSearchData, diagnostics). Rates and uncertainties are
    counts/s; exposure includes native event and overflow dead times.
    Source-free control checks are diagnostics, not empirical FAR calibration.
    """
    from gdt.core.background.unbinned import NaivePoisson
    step = float(seconds(config.min_step))
    search_lo, search_hi = seconds(config.search_interval)
    width = float(seconds(config.background_window))
    context = max(float(seconds(config.background_context)), width + 30.)
    lo, hi = min(-context, search_lo - context), max(context, search_hi + context)
    edges = np.arange(np.floor(lo / step), np.ceil(hi / step) + 1) * step
    centers = (edges[:-1] + edges[1:]) / 2
    shape = (len(centers), len(config.detectors), 8)
    counts = np.zeros(shape, dtype=np.int64)
    exposure = np.zeros(shape[:2])
    rates, uncertainty = np.zeros(shape), np.zeros(shape)
    valid = np.zeros(shape, dtype=bool)
    common = [(edges[0], edges[-1])]
    diagnostics = {}
    for d, detector in enumerate(config.detectors):
        times, channels, gti, metadata = read_detector_events(
            paths_by_detector[detector], trigger_time, edges[[0, -1]] * u.s)
        common = _intersect_intervals(common, gti)
        native_edges = np.asarray(CHANNEL_EDGES["nai" if detector.startswith("n") else "bgo"])
        grouped = np.searchsorted(native_edges, channels, side="right") - 1
        for c in range(8):
            counts[:, d, c] = np.histogram(times[grouped == c], edges)[0]
        deadtime = (np.histogram(times[channels != 127], edges)[0] * metadata[1]
                    + np.histogram(times[channels == 127], edges)[0] * metadata[2])
        exposure[:, d] = np.maximum(0, interval_exposure(edges, gti) - deadtime)
        accumulated_live = np.r_[0., np.cumsum(exposure[:, d])]
        live_fraction = (np.interp(centers + width / 2, edges, accumulated_live)
                         - np.interp(centers - width / 2, edges, accumulated_live)) / width
        for a, b in gti:
            chosen = (times >= a) & (times < b)
            bins = (centers >= a + width / 2) & (centers <= b - width / 2)
            if not bins.any():
                continue
            for c in range(8):
                selected = times[chosen & (grouped == c)]
                if len(selected) < 3:
                    continue
                model = NaivePoisson([selected])
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        model.fit(window_width=width, fast=True)
                        r, err = model.interpolate(edges[:-1][bins], edges[1:][bins])
                    # NaivePoisson estimates counts per elapsed second; the
                    # likelihood multiplies a rate per live second by exposure.
                    rates[bins, d, c] = r[:, 0] / live_fraction[bins]
                    uncertainty[bins, d, c] = err[:, 0] / live_fraction[bins]
                    valid[bins, d, c] = np.isfinite(r[:, 0]) & np.isfinite(err[:, 0]) & (r[:, 0] > 0)
                except (ValueError, IndexError, FloatingPointError):
                    continue
        # Check source-free, non-overlapping ~4 s control blocks, summed over
        # the search channels. Low-count channel checks are not Gaussianized.
        search_channels = np.arange(1, 7) if detector.startswith("n") else np.arange(8)
        block = max(1, round(4.096 / step))
        n = len(centers) // block
        def blocked(values):
            return values[:n * block].reshape(n, block)
        obs = blocked(counts[:, d, search_channels].sum(axis=1)).sum(axis=1)
        mu = blocked((rates[:, d, search_channels].sum(axis=1) * exposure[:, d])).sum(axis=1)
        # Background model errors are correlated within a control block.
        var = blocked(np.sqrt((uncertainty[:, d, search_channels]**2).sum(axis=1)) * exposure[:, d]).sum(axis=1)**2
        t = blocked(centers).mean(axis=1)
        good = blocked(valid[:, d, search_channels].all(axis=1)).all(axis=1)
        block_start = edges[np.arange(n) * block]
        block_stop = edges[np.arange(1, n + 1) * block]
        # Exclude the nominal +/- width/2 support of every bin in each
        # control block. GDT NaivePoisson(fast=True) holds event count rather
        # than window width fixed, so its actual support can be wider. These
        # controls are diagnostics, not guaranteed source-free holdouts.
        nominal_source_free = ((block_stop + width / 2 <= search_lo)
                               | (block_start - width / 2 >= search_hi))
        use = good & nominal_source_free & (mu >= 20)
        diagnostic = (validate_background_residuals(obs[use], mu[use], np.sqrt(mu[use] + var[use]), times=t[use])
                      if use.any() else {"passed": False, "reason": "no_valid_background_control_blocks"})
        diagnostic.update(control_blocks=int(use.sum()), method="nominal_source_free_control_blocks",
                          independent_holdout=False)
        diagnostics[detector] = diagnostic
    return PreparedSearchData(edges, counts, exposure, rates, uncertainty, valid,
                              np.asarray(common, dtype=float).reshape(-1, 2), tuple(config.detectors)), diagnostics


class MeasuredHistory:
    """Merged measured POSHIST with explicit sample-state and gap validity."""
    def __init__(self, paths):
        from astropy.coordinates import CartesianRepresentation
        from gdt.core.coords import Quaternion
        from gdt.core.coords.spacecraft import SpacecraftFrame
        from gdt.missions.fermi.gbm.detectors import GbmDetectors
        from gdt.missions.fermi.gbm.poshist import GbmPosHist
        times, pos, vel, quat, good = [], [], [], [], []
        for path in latest_products(paths):
            with GbmPosHist.open(str(path)) as history:
                frame, states = history.get_spacecraft_frame(), history.get_spacecraft_states()
                times.append(frame.obstime.to_value("fermi"))
                pos.append(frame.obsgeoloc.xyz.to_value(u.m).T)
                vel.append(frame.obsgeovel.xyz.to_value(u.m / u.s).T)
                quat.append(np.asarray(frame.quaternion))
                good.append(np.asarray(states["good"]) & ~np.asarray(states["saa"]))
        if not times:
            raise ValueError("measured POSHIST required")
        self.times, indices = np.unique(np.concatenate(times), return_index=True)
        self.good = np.concatenate(good)[indices]
        self.frames = SpacecraftFrame(
            obsgeoloc=CartesianRepresentation(np.concatenate(pos)[indices].T * u.m),
            obsgeovel=CartesianRepresentation(np.concatenate(vel)[indices].T * u.m / u.s),
            quaternion=Quaternion(np.concatenate(quat)[indices]),
            obstime=Time(self.times, format="fermi"), detectors=GbmDetectors)

    def valid_interval(self, start, stop):
        if start < self.times[0] or stop > self.times[-1]:
            return False
        lo = max(0, np.searchsorted(self.times, start, side="right") - 1)
        hi = min(len(self.times), np.searchsorted(self.times, stop, side="left") + 1)
        return bool(self.good[lo:hi].all() and np.all(np.diff(self.times[lo:hi]) <= 5))

    def at(self, time):
        met = np.asarray(time.to_value("fermi"))
        if not self.valid_interval(float(met.min()), float(met.max())):
            raise ValueError("invalid measured attitude: SAA, missing samples or extrapolation")
        return self.frames.at(time)
