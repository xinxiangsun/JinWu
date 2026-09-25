"""Response-coherent search built on the attributed GTS likelihood core."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import QTable
from scipy.special import logsumexp
from scipy.spatial import cKDTree

from jinwu.core.time import Time
from .models import seconds, TEMPLATES
from .data import interval_exposure


def make_search_windows(config, gti):
    """Return aligned (start, duration) seconds inside both search range and GTI.

    The dyadic duration grid and phase stepping follow GTS. Bounds describe
    complete source windows, not bin centers. No window bridges a GTI gap.
    """
    lo, hi = seconds(config.search_interval)
    base = float(seconds(config.min_step))
    dmin, dmax = float(seconds(config.min_duration)), float(seconds(config.max_duration))
    windows = []
    for duration in dmin * 2.**np.arange(round(np.log2(dmax / dmin)) + 1):
        step_bins = max(1, round(duration / config.num_steps / base))
        for a, b in gti:
            start, stop = max(a, lo), min(b, hi)
            first = int(np.ceil((start - 1e-9) / base / step_bins)) * step_bins
            last = int(np.floor((stop - duration + 1e-9) / base))
            windows.extend((i * base, float(duration)) for i in range(first, last + 1, step_bins))
    return np.asarray(sorted(set(windows)), dtype=float).reshape(-1, 2)


def evaluate_search_likelihood(counts, background, background_variance, response, *, sky_size):
    """Evaluate GTS with explicit counts, full background variance and response.

    Response is counts per photon fluence (ph cm^-2) with shape
    (template, visible sky, detector-channel). The returned upstream object
    exposes dimensionless log ratios and fluence amplitudes. No sigma or FAR
    conversion is made. All bins are optimized, including weak uniform-sky
    bins which may become interesting under an external spatial prior.
    """
    from ._vendor.likelihood import Likelihood
    arrays = [np.asarray(x, dtype=float) for x in (counts, background, background_variance, response)]
    counts, background, background_variance, response = arrays
    if any(not np.all(np.isfinite(a)) for a in arrays):
        raise ValueError("nonfinite likelihood input")
    if counts.ndim != 1 or background.shape != counts.shape or background_variance.shape != counts.shape:
        raise ValueError("inconsistent likelihood vectors")
    if response.ndim != 3 or response.shape[-1] != counts.size or not response.shape[1]:
        raise ValueError("invalid response shape or no visible sky")
    if np.any(counts < 0) or np.any(background <= 0) or np.any(background_variance < 0) or np.any(response < 0):
        raise ValueError("invalid counts, background variance or negative response")
    like = Likelihood(response.shape[0], sky_size, prethresh=-np.inf)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        like.calculate(*arrays)
    if not np.all(np.isfinite(like.llr)) or not np.isfinite(like.marginal_llr):
        raise ValueError("nonfinite GTS likelihood output")
    return like


def spatial_prior_weights(grid_icrs, visible, *, position=None, map_vectors=None, map_probability=None):
    """Map an external prior onto native grid cells, retaining occulted mass.

    Returns normalized visible weights, visible probability and position-grid
    offset in degrees (NaN for maps). Pixel probabilities already include
    pixel area; depositing their mass avoids losing a narrow localization
    between response-grid points. Zero visible probability returns zeros.
    """
    visible = np.asarray(visible, bool)
    vectors = grid_icrs.cartesian.xyz.value.T
    if position is not None:
        separation = grid_icrs.separation(position).to_value(u.deg)
        index = int(np.argmin(separation))
        weights = np.zeros(len(visible))
        weights[index] = 1.
        offset = float(separation[index])
    elif map_vectors is not None:
        probability = np.asarray(map_probability, float)
        if np.any(probability < 0) or not np.all(np.isfinite(probability)) or probability.sum() <= 0:
            raise ValueError("invalid sky probability")
        indices = cKDTree(vectors).query(map_vectors)[1]
        weights = np.bincount(indices, weights=probability / probability.sum(), minlength=len(visible))
        offset = np.nan
    else:
        weights = np.full(len(visible), 1. / len(visible))
        offset = np.nan
    mass = float(weights[visible].sum())
    return weights[visible] / mass if mass > 0 else weights[visible], mass, offset


def read_spatial_map(path):
    """Return ICRS unit vectors and pixel masses from a local HEALPix sky map.

    Uses the shared light reader in ``jinwu.core.skymap`` (no ``jinwu-gw``
    dependency).  Coarse native cells are subdivided to order 6 with constant
    density; finer cells retain their original probability mass.
    """
    try:
        from jinwu.core.skymap import load_skymap, sky_map_pixel_vectors
    except ImportError as exc:
        raise ImportError("sky-map input requires jinwu-fermi[search-skymap]") from exc
    return sky_map_pixel_vectors(load_skymap(path))


def response_factory(detectors, template_root, history, t0):
    """Construct the GTS response with bounded read caching and periodic azimuth.

    Atmospheric interpolation uses neighboring circular azimuths; outside the
    upstream rocking-angle domain it retains the direct-only flag, which is
    propagated as needs_review by the search rather than silently accepted.
    """
    from ._vendor.response import GbmResponse
    from ._vendor.utils import SkyGrid

    class Response(GbmResponse):
        @lru_cache(maxsize=8)
        def _array(self, filename):
            # Official templates contain tiny interpolation roundoff below zero;
            # preflight rejects significant negatives before this projection.
            return np.maximum(np.load(filename, mmap_mode="r", allow_pickle=False), 0.)

        def load_direct_response(self, detector):
            kind = self.get_detector_type(detector)
            return self._array(str(Path(self.templates_directory) / "direct" / f"{kind}.npy"))[:, :, :, self.det_index[detector]]

        def load_atmospheric_response(self, detector, geo_az, geo_zen):
            if abs(geo_zen - self.rocking_zen) > self.zen_margin:
                return 0.
            angles = np.sort(np.unique(self.available_azimuths % (2 * np.pi)))
            if len(angles) < 2:
                raise ValueError("atmospheric templates require at least two azimuths")
            angle = float(geo_az % (2 * np.pi))
            right = np.searchsorted(angles, angle, side="right") % len(angles)
            left = (right - 1) % len(angles)
            width = (angles[right] - angles[left]) % (2 * np.pi)
            fraction = ((angle - angles[left]) % (2 * np.pi)) / width
            kind = self.get_detector_type(detector)
            def value(index):
                filename = Path(self.templates_directory) / f"atmo_{kind}" / f"atmrates_az{round(np.degrees(angles[index]))}_zen130.npy"
                return self._array(str(filename))[:, :, :, self.det_index[detector]]
            return (1 - fraction) * value(left) + fraction * value(right)

    grid = SkyGrid(5.)
    return Response(list(detectors), grid, str(template_root), history, t0, templates=[0, 1, 2]), grid


def select_search_candidates(table, *, score, threshold, overlap_factor):
    """Greedily cluster valid windows by score; never force a candidate.

    Reject overlap exceeding overlap_factor times the
    shorter window. Returns selected rows and per-row filtering reasons.
    """
    values = np.asarray(table[score], float)
    eligible = np.asarray(table["eligible"], bool)
    starts = u.Quantity(table["tstart"]).to_value(u.s)
    durations = u.Quantity(table["duration"]).to_value(u.s)
    reasons = np.where(~eligible, "quality", np.where(~np.isfinite(values), "no_prior_support",
                       np.where(values < threshold, "score_floor", "overlap"))).astype("U32")
    selected = []
    for index in np.argsort(-values, kind="stable"):
        if not eligible[index] or not np.isfinite(values[index]) or values[index] < threshold:
            continue
        if any(min(starts[index] + durations[index], starts[j] + durations[j]) - max(starts[index], starts[j])
               > overlap_factor * min(durations[index], durations[j]) for j in selected):
            continue
        selected.append(index)
        reasons[index] = "selected"
    return table[selected].copy(), reasons


def evaluate_prepared_window(prepared, response, grid, start, duration, *, sums=None):
    """Adapt one window to GTS with live-time corrections and full variance.

    Returns likelihood, visible mask, live seconds and unmasked low-channel
    diagnostics. ``start`` and ``duration`` are internal seconds on the
    prepared time grid; public entrypoints accept Quantity instead.
    """
    base = prepared.edges[1] - prepared.edges[0]
    i, j = np.rint((np.array([start, start + duration]) - prepared.edges[0]) / base).astype(int)
    if i < 0 or j >= len(prepared.edges) or j <= i:
        raise ValueError("window outside prepared data")
    mid = (i + j) // 2
    counts_all = prepared.counts[i:j].sum(axis=0) if sums is None else sums[0][j] - sums[0][i]
    exp = prepared.exposure[i:j].sum(axis=0) if sums is None else sums[1][j] - sums[1][i]
    bkg_all = prepared.rates[mid] * exp[:, None]
    var_all = (prepared.uncertainty[mid] * exp[:, None])**2
    mask = np.array([[False, True, True, True, True, True, True, False] if d.startswith("n") else [True]*8
                     for d in prepared.detectors]).ravel()
    if not prepared.background_valid[mid].ravel()[mask].all() or np.any(exp <= 0):
        raise ValueError("background_unavailable")
    matrix = response.load_response(start, start + duration)
    visible = response.sky_mask()
    scaled = matrix * np.repeat(exp / duration, 8)[None, None, :]
    like = evaluate_search_likelihood(counts_all.ravel()[mask], bkg_all.ravel()[mask], var_all.ravel()[mask],
                                     scaled[:, visible, :][:, :, mask], sky_size=grid.size)
    # Reuse the upstream phosphorescence veto variables with corrected variance.
    nai = np.array([d.startswith("n") for d in prepared.detectors])
    from types import SimpleNamespace
    from ._vendor.results import calculate_pe_variables
    pe = (np.nan, np.nan, np.nan)
    if nai.sum() >= 2 and np.all((bkg_all + var_all)[nai, :2] > 0):
        data = SimpleNamespace(counts=counts_all.ravel(), background_counts=bkg_all.ravel(), background_var=var_all.ravel())
        indices = np.arange(counts_all.size).reshape(-1, 8)[nai, :2]
        pe = calculate_pe_variables(SimpleNamespace(instrument_data={"gbm": data}), None, "gbm", indices)
    return like, visible, exp, pe


def run_search_grid(prepared, history, template_root, trigger_time, config, *, position=None, skymap=None):
    """Search all valid time windows; return a unit-bearing table and diagnostics."""
    t0 = float(trigger_time.to_value("fermi"))
    response, grid = response_factory(config.detectors, template_root, history, t0)
    requested = len(make_search_windows(config, [seconds(config.search_interval)]))
    windows = make_search_windows(config, prepared.gti)
    gti_count = len(windows)
    windows = np.array([(a, d) for a, d in windows if history.valid_interval(t0 + a, t0 + a + d)], dtype=float).reshape(-1, 2)
    if len(windows) and history.valid_interval(t0 + windows[:, 0].min(), t0 + (windows[:, 0] + windows[:, 1]).max()):
        response.preprocess(windows)
    map_vectors, map_probability = read_spatial_map(skymap) if skymap is not None else (None, None)
    countsum = np.concatenate((np.zeros_like(prepared.counts[:1]), np.cumsum(prepared.counts, axis=0)))
    expsum = np.concatenate((np.zeros_like(prepared.exposure[:1]), np.cumsum(prepared.exposure, axis=0)))
    rows, failures = [], []
    for start, duration in windows:
        try:
            like, visible, exp, pe = evaluate_prepared_window(prepared, response, grid, start, duration,
                                                            sums=(countsum, expsum))
            coords = SkyCoord(grid.radians[0], np.pi / 2 - grid.radians[1], frame=response.frame, unit="rad").icrs
            weights, mass, offset = spatial_prior_weights(coords, visible, position=position,
                                                           map_vectors=map_vectors, map_probability=map_probability)
            if position is not None and not bool(np.asarray(response.frame.location_visible(position)).all()):
                weights, mass = weights * 0, 0.
            prior_score = float(logsumexp(like.llr[:, weights > 0] + np.log(weights[weights > 0])[None, :])
                                - np.log(len(TEMPLATES))) if mass > 0 else np.nan
            point = coords[visible][like._max_idx[1]]
            ptemplate, pra, pdec, pflux, perror = "", np.nan, np.nan, np.nan, np.nan
            if mass > 0:
                logweights = np.full(len(weights), -np.inf)
                logweights[weights > 0] = np.log(weights[weights > 0])
                pidx = np.unravel_index(np.argmax(like.llr + logweights[None, :]), like.llr.shape)
                ppoint = coords[visible][pidx[1]]
                ptemplate, pra, pdec = TEMPLATES[pidx[0]], ppoint.ra.deg, ppoint.dec.deg
                pflux, perror = like._pflux[pidx] / duration, like._pflux_sig[pidx] / duration
            from ._vendor.filters import remove_pe
            pe_data = np.array([pe], dtype=[("pe0", float), ("pe1", float), ("pe2", float)])
            pe_ok = bool(remove_pe(pe_data)[0]) if np.isfinite(pe).all() else False
            # Coherent statistic and per-channel diagnostics are retained.
            rows.append((start, duration, like.marginal_llr, prior_score, mass, offset,
                         point.ra.deg, point.dec.deg, TEMPLATES[like._max_idx[0]],
                         like.photon_fluence / duration, like.photon_fluence_sigma / duration,
                         int(like.status), bool(response.in_rock), pe_ok,
                         float(like.optimal_snr), float(exp.min()), *pe, pe_ok,
                         ptemplate, pra, pdec, pflux, perror))
        except (ValueError, FloatingPointError) as exc:
            failures.append([float(start), float(duration), str(exc)])
    names = ("tstart", "duration", "loglr", "prior_loglr", "visible_prior_probability", "position_grid_offset",
             "ra", "dec", "template", "photon_flux", "photon_flux_error", "like_status", "atmospheric_response",
             "eligible", "optimal_snr", "min_exposure", "pe0", "pe1", "pe2", "pe_passed",
             "prior_template", "prior_ra", "prior_dec", "prior_photon_flux", "prior_photon_flux_error")
    dtype = (float,) * 8 + ("U8", float, float, int, bool, bool, float, float, float, float, float, bool,
                           "U8", float, float, float, float)
    table = QTable(rows=rows, names=names, dtype=dtype)
    for name in ("tstart", "duration", "min_exposure"):
        table[name].unit = u.s
    for name in ("ra", "dec", "position_grid_offset", "prior_ra", "prior_dec"):
        table[name].unit = u.deg
    for name in ("photon_flux", "photon_flux_error", "prior_photon_flux", "prior_photon_flux_error"):
        table[name].unit = u.ph / (u.cm**2 * u.s)
    table.meta.update(trigger_met=t0, trigger_utc=trigger_time.utc.isot,
                      score_definition="GTS amplitude-marginalized log ratio; not sigma",
                      photon_flux_band_keV=[50., 300.], localization="statistical_only")
    return table, {"requested_windows": requested, "gti_rejected_windows": requested - gti_count,
                   "attitude_rejected_windows": gti_count - len(windows),
                   "attempted_windows": len(windows), "failed_windows": failures,
                   "valid_windows": len(table), "calibrated": False}
