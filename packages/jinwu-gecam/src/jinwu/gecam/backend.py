"""Thin interfaces to the official GECAMTools (Python import: ``gecam``).

No GECAMTools source or calibration data is bundled. Algorithms, channel
mapping, time bin alignment and background errors remain upstream-defined.
"""
from __future__ import annotations

from importlib import import_module
from pathlib import Path

import astropy.units as u
import numpy as np


def _backend():
    try:
        module = import_module("gecam")
    except ModuleNotFoundError as exc:
        if exc.name != "gecam":
            raise
        raise ImportError(
            "Install official GECAMTools >=20240514 from "
            "https://github.com/zhangpeng-sci/GECAMTools-Public "
            "into this Python environment (import name: gecam)."
        ) from exc
    release = str(module.get_software_version())
    if not release.isdigit() or int(release) < 20240514:
        raise ImportError(f"GECAMTools >=20240514 is required; found {release}")
    return module


def _values(value, unit, name):
    if not isinstance(value, u.Quantity):
        raise TypeError(f"{name} must be an astropy Quantity in units of {unit}")
    result = np.asarray(value.to_value(unit), dtype=float)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def _intervals(value, name):
    result = _values(value, u.s, name)
    if result.ndim != 2 or result.shape[1] != 2 or not len(result) or np.any(np.diff(result) <= 0):
        raise ValueError(f"{name} must contain increasing [start, stop] MET intervals")
    return result


def validate_time_bin(time_bin):
    """Validate a positive scalar GECAM time bin and return seconds."""
    step = _values(time_bin, u.s, "time_bin")
    if step.ndim != 0 or step <= 0:
        raise ValueError("time_bin must be a positive scalar")
    return step


def validate_selection(*, time_range, time_bin, energy_edges, gain=None,
                       gain_message="gain must be high or low"):
    """Validate GECAM time/energy selection and return normalized values.

    Order matches the public entry points: time range, time bin, energy edges,
    then gain when supplied. ``gain_message`` lets pipeline callers keep their
    more specific aggregation hint.
    """
    bounds = _values(time_range, u.s, "time_range")
    step = _values(time_bin, u.s, "time_bin")
    edges = _values(energy_edges, u.keV, "energy_edges")
    if bounds.shape != (2,) or bounds[1] <= bounds[0]:
        raise ValueError("time_range must be increasing [start, stop] MET")
    if step.ndim != 0 or step <= 0:
        raise ValueError("time_bin must be a positive scalar")
    if edges.ndim != 1 or len(edges) < 2 or np.any(edges <= 0) or np.any(np.diff(edges) <= 0):
        raise ValueError("energy_edges must be positive and increasing")
    if gain is not None and gain not in {"high", "low"}:
        raise ValueError(gain_message)
    return bounds, step, edges


def open_event(path: str | Path):
    """Return the native GECAMTools Evt for a GECAM A/B/C GRD event FITS.

    Parameters
    ----------
    path : str or Path
        Existing EVT file. Native metadata and all lower-level upstream APIs
        remain available on the returned object; no files are modified.
    """
    _backend()
    from gecam.data.evt import Evt

    event = Evt.open(str(Path(path).expanduser().resolve(strict=True)))
    if event.info.file_type != "EVT" or event.info.instrument != "GRD" or event.info.satellite not in {"a", "b", "c"}:
        raise ValueError("Expected a GECAM A/B/C GRD EVT file")
    return event


def extract_lightcurve(event, *, detector: int, gain: str, time_range: u.Quantity,
                       time_bin: u.Quantity, energy_edges: u.Quantity,
                       only_recommend: bool = True, correct_dead_time: bool = True):
    """Return a native GECAMTools LightCurve, preserving its counts/errors.

    ``event`` is returned by :func:`open_event`; ``detector`` is a GRD number
    and ``gain`` must be high or low (never mix gain calibrations).
    ``time_range`` is a two-element absolute GECAM MET Quantity in seconds;
    ``time_bin`` is a positive scalar duration; ``energy_edges`` are positive,
    increasing energies convertible to keV. Upstream rounds requested energies
    to calibrated channels; inspect returned ``energy_bins`` / ``channel_bins``.
    Both raw and corrected counts are available through native ``get_data``.
    """
    bounds, step, edges = validate_selection(
        time_range=time_range, time_bin=time_bin, energy_edges=energy_edges, gain=gain,
    )
    # Keep original events for dead-time calculation, as required by upstream.
    events = event.select_detector(detector).slice(
        gain_type=gain, time_range=bounds.tolist(), only_recommend=only_recommend,
    )
    return events.to_light_curve(
        time_range=bounds.tolist(), time_bin=float(step), energy_bin=edges.tolist(),
        correct_by_dead_time=correct_dead_time,
    )


def extract_spectra(event, *, detector: int, gain: str, source_intervals: u.Quantity,
                    background_intervals: u.Quantity, time_bin: u.Quantity,
                    outdir: str | Path, response: str | Path | None = None,
                    channel_bin: int = 1, fit_order: int = 1,
                    only_recommend: bool = True):
    """Generate native full/background/net type-II spectra; return upstream dict.

    Intervals are (N, 2) absolute GECAM MET Quantities in seconds, ``time_bin``
    is a positive scalar duration. Channel bin and polynomial order are passed
    unchanged to GECAMTools. ``response`` is an existing caller-selected RSP;
    None produces spectra without a response association, unsuitable for fitting.
    ``outdir`` must be new or empty to protect existing products. This delegates
    to upstream ``generate_spec_file_with_detecotrs`` (its original spelling).
    """
    _backend()
    from gecam.data.detector import GRD

    src = _intervals(source_intervals, "source_intervals")
    bg = _intervals(background_intervals, "background_intervals")
    step = validate_time_bin(time_bin)
    if gain not in {"high", "low"}:
        raise ValueError("gain must be high or low")
    if not isinstance(channel_bin, int) or isinstance(channel_bin, bool) or channel_bin < 1:
        raise ValueError("channel_bin must be a positive integer")
    if not isinstance(fit_order, int) or isinstance(fit_order, bool) or fit_order < 0:
        raise ValueError("fit_order must be a nonnegative integer")
    if np.any((src[:, None, 0] < bg[None, :, 1]) & (bg[None, :, 0] < src[:, None, 1])):
        raise ValueError("source and background intervals must not overlap")
    output = Path(outdir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Spectrum output directory must be empty: {output}")
    rsp = str(Path(response).expanduser().resolve(strict=True)) if response is not None else None
    rsp_pair = [rsp, None] if gain == "high" else [None, rsp]
    return event.generate_spec_file_with_detecotrs(
        [GRD(detector, gain_type=gain)],
        {"only_recommend": only_recommend},
        {"time_bin": float(step), "channel_bin": [channel_bin, channel_bin]},
        {"bg_time_range": bg.tolist(), "fit_order": fit_order},
        {"src_range_list": src.tolist(), "rsp_list": [rsp_pair], "out_dir": str(output)},
    )
