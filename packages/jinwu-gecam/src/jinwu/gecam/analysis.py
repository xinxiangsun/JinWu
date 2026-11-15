"""Translate native GECAM products into the shared one-source analysis inputs."""
from pathlib import Path

import astropy.units as u
from astropy.io import fits
from astropy.table import Table
import numpy as np

from jinwu.core.source_analysis import LightcurveProduct, SpectralObservation
from jinwu.core.spectrum_prep import prepare_ogip_spectrum
from jinwu.core.time import Time


def _met_reference(header, telescope, *, allow_mission_epoch=True):
    """Read a native clock or record the assumed epoch for a finished A/B product."""
    if "MJDREF" in header or {"MJDREFI", "MJDREFF"} <= set(header):
        if header.get("TIMESYS", "").strip().lower() != "tt" or header.get("TIMEUNIT", "").strip().lower() != "s":
            raise ValueError("Expected GECAM FITS time metadata in TT seconds")
        value = header.get("MJDREF", header.get("MJDREFI"))
        fraction = 0.0 if "MJDREF" in header else header["MJDREFF"]
        reference = Time(value, fraction, format="mjd", scale="tt").utc
        reference += float(header.get("TIMEZERO", 0.0)) * u.s
        return reference, "FITS MJDREF/TIMESYS"
    if not allow_mission_epoch or any(key in header for key in ("MJDREFI", "MJDREFF", "TIMEZERO")):
        raise ValueError("Incomplete GECAM FITS time reference")
    if telescope in {"GECAM-C", "HEBS"}:
        raise ValueError("Supply the GECAM-C/HEBS MET reference in UTC or FITS MJDREF/TIMESYS")
    if telescope not in {"GECAM-A", "GECAM-B"}:
        raise ValueError("Expected an explicit GECAM telescope and time reference")
    return Time(0, format="gecam").utc, "assumed GECAM-A/B mission epoch"


def prepare_gecam_spectra(source_files, background_files, responses, *, outdir,
                          energy_range: u.Quantity, target_id: str, gain: str, met_reference_utc: str | None = None):
    """Convert matching GECAMTools full/background type-II rows to observations.

    Input mappings use GRD detector numbers. ``energy_range`` sets the fitted
    observer-frame band. Full spectra carry raw Poisson counts and live exposure.
    Their background files must contain the modeled counts and Gaussian errors.
    Each row number identifies an epoch. The shared pipeline checks that its
    intervals match before tying spectra from separate detectors.
    ``met_reference_utc`` may supply the native clock reference explicitly.
    The native ``TIME`` is the left boundary and ``TIMEDEL`` is in seconds.
    """
    if not source_files or set(source_files) != set(background_files) or set(source_files) != set(responses):
        raise ValueError("Supply matching full spectra, background spectra and responses for every detector")
    if gain not in {"high", "low"}:
        raise ValueError("gain must be high or low")
    observations = []
    for detector in sorted(source_files):
        source, background, response = source_files[detector], background_files[detector], responses[detector]
        detector_label = f"GRD{detector:02d}_{gain}"
        with fits.open(source) as source_hdus, fits.open(background) as background_hdus, fits.open(response) as response_hdus:
            source_hdu = source_hdus["SPECTRUM"]
            telescope = source_hdu.header.get("TELESCOP", source_hdus[0].header.get("TELESCOP"))
            if telescope not in {"GECAM-A", "GECAM-B", "GECAM-C", "HEBS"}:
                raise ValueError("Expected GECAM TELESCOP metadata")
            headers = []
            for hdus, is_source in ((source_hdus, True), (background_hdus, False)):
                hdu = hdus["SPECTRUM"]
                required = {"SPEC_NUM", "TIME", "TIMEDEL", "CHANNEL", "COUNTS", "STAT_ERR", "GROUPING", "QUALITY", "EXPOSURE"}
                if not required <= set(hdu.columns.names) or np.asarray(hdu.data["CHANNEL"]).ndim != 2:
                    raise ValueError("Expected native GECAMTools type-II full/background spectrum")
                header = hdu.header
                if str(header.get("DETNUMS", hdus[0].header.get("DETNUMS"))).strip() != str(detector):
                    raise ValueError(f"Spectrum detector metadata does not match {detector_label}")
                if header.get("TELESCOP", hdus[0].header.get("TELESCOP")) != telescope:
                    raise ValueError("Source and background telescope metadata do not match")
                if "POISSERR" in header and header["POISSERR"] != is_source:
                    raise ValueError("Expected a Poisson full spectrum and Gaussian modeled background")
                combined = hdus[0].header.copy()
                combined.extend(header, update=True)
                headers.append(combined)
            bounds_header = response_hdus["EBOUNDS"].header
            detname = response_hdus[0].header.get("DETNAM", bounds_header.get("DETNAM", bounds_header.get("DETNAME")))
            if detname and not detname.upper().endswith(f"{detector:02d}{gain[0].upper()}"):
                raise ValueError(f"Response detector/gain metadata does not match {detector_label}: {detname}")
            if any(header.get("TELESCOP", telescope) != telescope for header in (response_hdus[0].header, bounds_header)):
                raise ValueError("Spectrum and response telescope metadata do not match")
            full, model = source_hdu.data, background_hdus["SPECTRUM"].data
            expected = source_hdus["EBOUNDS"].data
            background_bounds = background_hdus["EBOUNDS"].data
            response_bounds = response_hdus["EBOUNDS"].data
            if any(not np.array_equal(expected[name], background_bounds[name]) for name in ("CHANNEL", "E_MIN", "E_MAX")):
                raise ValueError(f"Full and background EBOUNDS must match for {detector_label}")
            # The native PHA masks the overflow channels, whose energies can be
            # different in the CALDB response. Compare usable channels only.
            good = np.any(full["QUALITY"] == 0, axis=0)
            if not np.array_equal(expected["CHANNEL"], response_bounds["CHANNEL"]) or any(
                not np.allclose(expected[name][good], response_bounds[name][good], rtol=1e-6, atol=1e-6)
                for name in ("E_MIN", "E_MAX")
            ):
                raise ValueError(f"Response EBOUNDS must match the usable spectrum channels for {detector_label}")
            for name in ("SPEC_NUM", "TIME", "TIMEDEL", "EXPOSURE", "CHANNEL", "GROUPING", "QUALITY"):
                if not np.array_equal(full[name], model[name]):
                    raise ValueError(f"Full and background {name} must match for {detector_label}")
            if np.any(full["COUNTS"] < 0) or not np.allclose(full["STAT_ERR"]**2, full["COUNTS"], rtol=1e-6, atol=1e-6):
                raise ValueError("Expected raw full spectrum counts with Poisson errors")
            source_header, background_header = headers
            clock_keys = {"MJDREF", "MJDREFI", "MJDREFF", "TIMEZERO"}
            if clock_keys.intersection(source_header) and clock_keys.intersection(background_header):
                source_zero, _ = _met_reference(source_header, telescope, allow_mission_epoch=False)
                background_zero, _ = _met_reference(background_header, telescope, allow_mission_epoch=False)
                if abs((source_zero - background_zero).to_value(u.s)) > 1e-6:
                    raise ValueError(f"Full and background MET references must match for {detector_label}")
            if met_reference_utc is None:
                met_reference, clock_source = _met_reference(source_header, telescope)
            else:
                met_reference, clock_source = Time(met_reference_utc, scale="utc"), "explicit MET reference"
            if not met_reference.isscalar:
                raise ValueError("met_reference_utc must be scalar")
            met_reference.precision = 9
            times = np.array(full["TIME"], dtype=float)
            widths = np.array(full["TIMEDEL"], dtype=float)
            if not len(times) or not np.all(np.isfinite(times)) or not np.all(np.isfinite(widths)) or np.any(widths <= 0):
                raise ValueError("Invalid GECAM spectrum interval")
        for index, (start, width) in enumerate(zip(times, widths), start=1):
            epoch = f"interval_{index}"
            prepared = prepare_ogip_spectrum(
                source, background=background, response=response, source_row=index, background_row=index,
                source_poisson=True, background_poisson=False,
                outdir=Path(outdir) / f"row_{index:03d}" / detector_label,
                energy_range=energy_range, instrument=telescope, obsid=target_id, detector=detector_label, source_id=epoch,
            )
            prepared.diagnostics.append(f"Native MET zero: {met_reference.isot} UTC ({clock_source})")
            if not clock_keys.intersection(background_header):
                prepared.diagnostics.append("Background FITS clock absent; assumed the full spectrum's native MET zero")
            bounds = met_reference + np.array([start, start + width]) * u.s
            bounds.precision = 9
            observations.append(SpectralObservation(prepared, bounds[0].isot, bounds[1].isot, "pgstat", epoch,
                                                    target_id=target_id))
    return tuple(observations)


def gecam_lightcurve_products(files, *, outdir, reference_utc: str, telescope: str | None = None,
                              target_id: str | None = None, met_reference_utc: str | None = None):
    """Convert native NPZ curves to per-detector/band total count-rate ECSV.

    Rates and errors are counts divided by actual bin width, retaining the
    native dead-time correction choice. Time bounds are in seconds relative to
    ``reference_utc``. Use the NPZ clock when available or explicitly provide
    ``met_reference_utc``. For older files, ``telescope`` selects the mission
    epoch fallback, with GECAM-B as the default.
    """
    output = Path(outdir)
    output.mkdir(parents=True, exist_ok=True)
    reference = Time(reference_utc, scale="utc")
    if not reference.isscalar:
        raise ValueError("reference_utc must be scalar")
    products = []
    for detector, path in files.items():
        with np.load(path) as data:
            native_telescope = str(data["telescope"].item()) if "telescope" in data else None
            if telescope and native_telescope and telescope != native_telescope:
                raise ValueError(f"Lightcurve telescope metadata does not match {telescope}")
            mission = native_telescope or telescope or "GECAM-B"
            if met_reference_utc is not None:
                met_reference, clock_source = Time(met_reference_utc, scale="utc"), "explicit MET reference"
            elif "met_reference_utc" in data:
                met_reference = Time(data["met_reference_utc"].item(), scale="utc")
                clock_source = str(data["met_clock_source"].item()) if "met_clock_source" in data else "native NPZ clock"
            else:
                met_reference, clock_source = _met_reference({}, mission)
            if not met_reference.isscalar:
                raise ValueError("met_reference_utc must be scalar")
            met_reference.precision = 9
            edges = data["time_edges_met_s"] + (met_reference - reference).to_value(u.s)
            width = np.diff(data["time_edges_met_s"])
            energy_edges = data["energy_edges_keV"]
            for index, (lo, hi) in enumerate(zip(energy_edges[:-1], energy_edges[1:])):
                table = Table({"time_start": edges[:-1] * u.s, "time_stop": edges[1:] * u.s,
                               "value": (data["counts"][index] / width) * u.ct / u.s,
                               "error": (data["counts_error"][index] / width) * u.ct / u.s})
                table.meta.update({"kind": "total_rate", "correct_dead_time": bool(data["correct_dead_time"]),
                                   "reference_utc": reference_utc, "met_reference_utc": met_reference.isot,
                                   "met_clock_source": clock_source,
                                   "actual_energy_range_keV": [float(lo), float(hi)]})
                if target_id is not None:
                    table.meta["target_id"] = target_id
                target = output / f"{detector}_band_{index}.ecsv"
                table.write(target, format="ascii.ecsv", overwrite=False)
                products.append(LightcurveProduct(target, f"{mission} {detector}", f"{lo:g}–{hi:g} keV",
                                                  reference_utc, target_id=target_id))
    return tuple(products)
