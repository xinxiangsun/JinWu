"""Explicit real-TTE adapter for the frozen paper method."""
from __future__ import annotations

import numpy as np
import astropy.units as u
from scipy.stats import chi2

from ..tte import read_detector_events, interval_exposure


def bin_events(events, interval, width, *, max_bins=10_000_000):
    """GDT bin_by_time edges, anchored at the explicit source start (seconds).

    Match the upstream np.arange(start, stop + dt, dt) procedure. A final
    partial source bin is retained and its exposure recorded by the caller.
    Events outside the requested half-open source interval are excluded.
    """
    from gdt.core.binning.unbinned import bin_by_time
    lo, hi = map(float, interval)
    if not np.isfinite(width) or width <= 0 or hi <= lo:
        raise ValueError("invalid bin width or interval")
    if int(np.ceil((hi - lo) / width)) > max_bins:
        raise ValueError("requested light curve exceeds max_bins")
    edges = bin_by_time(events, width, tstart=lo, tstop=hi)
    selected = events[(events >= lo) & (events < hi)]
    return np.histogram(selected, bins=edges)[0], edges


def prepare_detector(paths, trigger_time, source, backgrounds, config):
    """Return energy-selected events and a GDT polynomial background record.

    Source/background offsets and GTIs are seconds; energy bounds are keV.
    Fit the integrated selected band with GDT's original two-pass Polynomial.
    Dead time uses all native channels, including overflow, before selection.
    No source-background subtraction precedes the Haar estimator.
    """
    from gdt.core.data_primitives import Ebounds, EventList
    from gdt.core.background.binned import Polynomial
    windows = np.vstack([source, backgrounds])
    interval = [windows[:, 0].min(), windows[:, 1].max()] * u.s
    times, channels, gtis, (bounds, deadtime, overflow) = read_detector_events(paths, trigger_time, interval)
    for lo, hi in windows:
        covered = interval_exposure([lo, hi], gtis)[0]
        if not np.isclose(covered, hi - lo, rtol=0, atol=2e-6):
            raise ValueError(f"incomplete TTE GTI coverage in [{lo}, {hi}] s: {covered} s")
    # Use GDT's own edge-snapping semantics, including partially overlapping
    # energy channels, rather than inventing a different channel-center cut.
    ebounds = Ebounds.from_bounds(bounds[:, 0], bounds[:, 1])
    events = EventList(times=times, channels=channels, ebounds=ebounds)
    filtered = events.energy_slice(*config.energy_range.to_value(u.keV))
    selected_times = np.asarray(filtered.times)
    starts, stops, counts, exposures = [], [], [], []
    fractions = []
    dt = float(config.background_bin_width.to_value(u.s))
    for lo, hi in backgrounds:
        edges = np.arange(lo, hi, dt)
        edges = np.append(edges, hi)
        total = np.histogram(times, edges)[0]
        over = np.histogram(times[channels == len(bounds) - 1], edges)[0]
        dead = (total - over) * deadtime + over * overflow
        exposure = np.diff(edges) - dead
        if np.any(exposure <= 0):
            raise ValueError("nonpositive background livetime")
        counts.extend(np.histogram(selected_times, edges)[0])
        starts.extend(edges[:-1]); stops.extend(edges[1:]); exposures.extend(exposure)
        fractions.extend(dead / np.diff(edges))
    starts, stops = np.asarray(starts), np.asarray(stops)
    counts, exposures = np.asarray(counts), np.asarray(exposures)
    if not np.any(counts > 0):
        raise ValueError("no selected-band counts in the background intervals")
    fitter = Polynomial(counts[:, None], starts, stops, exposures)
    rates, uncertainties = fitter.fit(order=config.background_order)
    on_rate, on_error = fitter.interpolate(np.array([source[0]]), np.array([source[1]]))
    background_rate = float(on_rate[0, 0])
    if not np.isfinite(rates).all() or not np.isfinite(background_rate) or background_rate <= 0:
        raise ValueError("nonpositive/nonfinite background prediction")
    dof = int(np.asarray(fitter.dof).ravel()[0])
    chisq = float(np.asarray(fitter.statistic).ravel()[0])
    probability = float(chi2.sf(chisq, dof)) if dof > 0 else 0.
    src_total = np.count_nonzero((times >= source[0]) & (times < source[1]))
    src_over = np.count_nonzero((times >= source[0]) & (times < source[1]) & (channels == len(bounds) - 1))
    src_dead = (src_total - src_over) * deadtime + src_over * overflow
    src_fraction = src_dead / (source[1] - source[0])
    record = {"method": "GDT Polynomial on integrated selected energy band",
              "order": config.background_order, "source_average_rate_cps": background_rate,
              "source_average_rate_error_cps": float(on_error[0, 0]),
              "coefficients": fitter._coeff.tolist(), "covariance": fitter._covar.tolist(),
              "chi2": chisq, "dof": dof, "chi2_tail_probability": probability,
              "quality_passed": probability >= .001 and src_fraction < .1,
              "quality_rule": "diagnostic chi2 p>=0.001 and source deadtime fraction<0.1; not a significance calibration",
              "source_deadtime_fraction": src_fraction,
              "maximum_background_deadtime_fraction": float(np.max(fractions)),
              "actual_energy_range_keV": list(filtered.energy_range),
              "gti_s": gtis, "background_bin_start_s": starts.tolist(),
              "background_bin_stop_s": stops.tolist(), "background_counts": counts.tolist(),
              "background_exposure_s": exposures.tolist(), "fitted_rate_cps": rates[:, 0].tolist(),
              "fitted_rate_error_cps": uncertainties[:, 0].tolist()}
    return selected_times, record


def peak_snr(events, source, width, background_rate, *, max_bins=10_000_000):
    """Paper SNR_MVT = peak total counts / sqrt(mean background cps * dt)."""
    counts, _ = bin_events(events, source, width, max_bins=max_bins)
    return float(np.max(counts) / np.sqrt(background_rate * width))


def select_detectors(events, backgrounds, source, width, *, max_bins=10_000_000):
    """Rank individual SNRs and maximize the cumulative-prefix combined SNR.

    This is the frozen upstream find_optimal_detectors_for_run procedure,
    applied to observed counts and fitted real background instead of simulated
    source/background TTE. Return selected names and every ranking/prefix.
    """
    ranked = sorted(events, key=lambda d: peak_snr(events[d], source, width,
                    backgrounds[d]["source_average_rate_cps"], max_bins=max_bins), reverse=True)
    individual = [{"detector": d, "snr": peak_snr(events[d], source, width,
                  backgrounds[d]["source_average_rate_cps"], max_bins=max_bins)} for d in ranked]
    prefixes = []
    for k in range(1, len(ranked) + 1):
        dets = ranked[:k]
        times = np.concatenate([events[d] for d in dets])
        rate = sum(backgrounds[d]["source_average_rate_cps"] for d in dets)
        prefixes.append({"detectors": dets, "snr": peak_snr(times, source, width, rate, max_bins=max_bins)})
    best = max(prefixes, key=lambda p: p["snr"])
    return tuple(best["detectors"]), {"bin_width_s": width, "individual": individual, "prefixes": prefixes, "best": best}
