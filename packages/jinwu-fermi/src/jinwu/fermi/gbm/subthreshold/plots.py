"""Batch-safe search diagnostics and explicitly statistical localization maps."""
from pathlib import Path
import numpy as np
import astropy.units as u


def make_search_plots(full, candidates, prepared, directory, *, ranking):
    """Write waterfall and per-detector lightcurves; return artifact paths."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    directory = Path(directory)
    outputs = {}
    figure = Figure(figsize=(10, 4.5), constrained_layout=True)
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    if len(full):
        scatter = axis.scatter((full["tstart"] + full["duration"] / 2).to_value(u.s),
                               full["duration"].to_value(u.s), c=full[ranking], s=9, cmap="viridis")
        figure.colorbar(scatter, ax=axis, label=f"{ranking} (uncalibrated score)")
        axis.set_yscale("log", base=2)
    else:
        axis.text(.5, .5, "No valid search windows", ha="center", transform=axis.transAxes)
    axis.set(xlabel="Time since external trigger (s)", ylabel="Search duration (s)")
    path = directory / "waterfall.png"
    figure.savefig(path, dpi=150)
    outputs["waterfall"] = str(path)
    figure = Figure(figsize=(11, max(4, len(prepared.detectors) * 1.2)), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = np.atleast_1d(figure.subplots(len(prepared.detectors), 1, sharex=True))
    step = prepared.edges[1] - prepared.edges[0]
    block = max(1, round(.256 / step))
    n = len(prepared.counts) // block
    centers = ((prepared.edges[:-1] + prepared.edges[1:]) / 2)[:n * block].reshape(n, block).mean(axis=1)
    for d, (detector, axis) in enumerate(zip(prepared.detectors, axes)):
        channels = slice(1, 7) if detector.startswith("n") else slice(0, 8)
        counts = prepared.counts[:n * block, d, channels].sum(axis=1).reshape(n, block).sum(axis=1)
        exposure = prepared.exposure[:n * block, d].reshape(n, block).sum(axis=1)
        model = (prepared.rates[:n * block, d, channels].sum(axis=1) * prepared.exposure[:n * block, d]).reshape(n, block).sum(axis=1)
        good = exposure > 0
        axis.plot(centers[good], counts[good] / exposure[good], color="0.3", lw=.6)
        axis.plot(centers[good], model[good] / exposure[good], color="tab:orange", lw=.8)
        axis.set_ylabel(detector)
        if len(full):
            axis.set_xlim(full["tstart"].to_value(u.s).min(), (full["tstart"] + full["duration"]).to_value(u.s).max())
        for candidate in candidates:
            a, b = candidate["tstart"].to_value(u.s), (candidate["tstart"] + candidate["duration"]).to_value(u.s)
            axis.axvspan(a, b, color="tab:blue", alpha=.1)
    axes[-1].set_xlabel("Time since external trigger (s); detector rates in counts/s")
    path = directory / "lightcurves.png"
    figure.savefig(path, dpi=150)
    outputs["lightcurves"] = str(path)
    return outputs


def write_candidate_localizations(prepared, history, template_root, trigger_time, config, candidates, directory, *, ranking="loglr"):
    """Write GBM-only statistical HEALPix maps and response-stability diagnostics.

    Maps marginalize equally over spectral templates, include the measured
    Earth mask, and do not include a historical systematic localization model.
    Each selected candidate is also checked at the window endpoints against
    its midpoint response for the same ICRS peak direction.
    """
    import healpy as hp
    from astropy.coordinates import SkyCoord
    from scipy.special import logsumexp
    from .search import response_factory, evaluate_prepared_window
    from ._vendor.utils import grid_to_healpix
    response, grid = response_factory(config.detectors, template_root, history, float(trigger_time.to_value("fermi")))
    outputs, diagnostics = {}, []
    for index, row in enumerate(candidates):
        start, duration = row["tstart"].to_value(u.s), row["duration"].to_value(u.s)
        like, visible, _, _ = evaluate_prepared_window(prepared, response, grid, start, duration)
        frame = response.frame
        logmap = logsumexp(like.llr, axis=0) - np.log(like.llr.shape[0])
        probability = np.zeros(grid.size)
        probability[visible] = np.exp(logmap - logsumexp(logmap))
        projected, _ = grid_to_healpix(probability, grid.radians, frame, nside_out=64)
        theta, phi = hp.pix2ang(64, np.arange(len(projected)))
        coords = SkyCoord(phi * u.rad, (np.pi / 2 - theta) * u.rad, frame="icrs")
        projected[~np.asarray(frame.location_visible(coords), bool)] = 0
        projected = np.maximum(projected, 0.)
        if not np.isfinite(projected).all() or projected.sum() <= 0:
            raise ValueError("invalid statistical localization")
        projected /= projected.sum()
        path = Path(directory) / f"candidate_{index + 1:03d}_statistical_healpix.fits"
        hp.write_map(str(path), projected, coord="C", overwrite=True, dtype=np.float64,
                     column_names=["PROB"], extra_header=[("LOCTYPE", "STATISTICAL"), ("SYSERR", "NONE"),
                                                         ("TRIGMET", float(trigger_time.to_value("fermi")))])
        outputs[f"localization_{index + 1}"] = str(path)
        prefix = "prior_" if ranking == "prior_loglr" else ""
        point = SkyCoord(row[prefix + "ra"], row[prefix + "dec"], frame="icrs")
        matrix = response.response_matrix.copy()
        grid_coords = SkyCoord(grid.radians[0], np.pi / 2 - grid.radians[1], frame=frame, unit="rad").icrs
        selected = int(np.argmin(grid_coords.separation(point)))
        reference = matrix[:, selected, :].sum(axis=-1)
        differences = []
        for time in (start, start + duration):
            endpoint = response.load_response(time, time)
            coords = SkyCoord(grid.radians[0], np.pi / 2 - grid.radians[1], frame=response.frame, unit="rad").icrs
            nearest = int(np.argmin(coords.separation(point)))
            predicted = endpoint[:, nearest, :].sum(axis=-1)
            differences.append(float(np.max(np.abs(predicted - reference) / np.maximum(reference, np.finfo(float).tiny))))
        diagnostics.append({"candidate": index + 1, "response_fractional_change": max(differences),
                            "response_stable": max(differences) <= config.response_tolerance})
    return outputs, diagnostics
