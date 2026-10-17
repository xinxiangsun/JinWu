"""FXT L2/L3 analysis, with independent spectral fits for each epoch.

Existing spectra/responses are reused.  Time subdivision requires extracted
source/background events (or full events and explicit regions), and an explicit
confirmation that those regions/time windows may use the supplied responses.
No calibration, pile-up correction, or spectral shape is invented.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from importlib.resources import files as resource_files
import os
from pathlib import Path
from typing import Any, Mapping, Sequence
import copy

import numpy as np
from astropy.io import fits
from astropy.table import Table

from ...core.base import EnergyBand
from ...core.config import FXT, InstrumentConfig
from ...core.fit import fit_xray_models
from ...core.galactic import resolve_galactic_absorption
from ...core.instruments import DataFile, Manifest, SpectrumBundle, scan
from ...core.io import channel_mask_from_ebounds, read_pha, read_rmf
from ...core.pipeline import PipelineInput, register_pipeline
from ...core.products import (
    FluxCurveResult, FluxPoint, NetLightcurve, build_net_lightcurve, collect_runtime_environment,
    plot_net_lightcurve, save_flux_curve, save_net_lightcurve, write_json,
)
from ...core.significance import li_ma_snr
from ...core.spectrum_prep import PreparedJointSpectrum, PreparedSpectrum, prepare_spectra
from ...core.time import Time
from ...core.upperlimit import (
    CountTemplateModel, UpperLimitObservation, XspecCountPredictor, estimate_upper_limit,
)
from ...core.xselect import extract_products_with_xselect

__all__ = ["FXTInput", "FXTPipeline", "FXTResult"]


@dataclass(frozen=True, slots=True, kw_only=True)
class FXTInput(PipelineInput):
    """One source's FXT products, optionally containing several observations.

    Coordinates are degrees; ``trigger_time_utc`` is ISO UTC. ``intervals`` maps
    obsid to (start, stop) seconds since T0, or EP MET when T0 is absent. Keys of
    explicit event/region maps are ``'<obsid>:FXTA'`` or ``'<obsid>:FXTB'``.
    ``spectra`` supplies already prepared inputs for nonstandard file names.
    ``response_compatible`` confirms that time subdivisions/regions may reuse
    those responses; it does not claim that pile-up has been corrected.
    """

    ra_deg: float
    dec_deg: float
    trigger_time_utc: str | None = None
    redshift: float = 0.0
    spectra: tuple[PreparedSpectrum, ...] = ()
    intervals: Mapping[str, Sequence[tuple[float, float]]] = field(default_factory=dict)
    event_files: Mapping[str, str | Path] = field(default_factory=dict)
    source_regions: Mapping[str, str | Path] = field(default_factory=dict)
    background_regions: Mapping[str, str | Path] = field(default_factory=dict)
    response_compatible: bool = False
    pileup_reviewed: bool = False
    nondetection_obsids: tuple[str, ...] = ()


@dataclass(slots=True)
class FXTResult:
    workspace: Path
    status: str
    epoch_results: list[dict[str, Any]] = field(default_factory=list)
    flux_rows: list[dict[str, Any]] = field(default_factory=list)
    count_rows: list[dict[str, Any]] = field(default_factory=list)
    products: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _header(path: Path) -> fits.Header:
    with fits.open(path, memmap=False) as hdul:
        header = hdul[0].header.copy()
        header.update(hdul[1].header)
    return header


def _provenance(spectrum: PreparedSpectrum):
    products = {}
    for role in ("source_pha", "grouped_pha", "background_pha", "arf", "rmf"):
        path = getattr(spectrum, role)
        if path is not None:
            header = _header(path)
            history = list(header.get("HISTORY", []))
            products[role] = {"path": str(path), "creator": header.get("CREATOR"),
                              "calibration_history": history}
    return {"module": spectrum.module, "products": products}


def _time_offset(header: fits.Header) -> float:
    """Verify the FITS clock against the registered EP epoch, including TT."""
    if str(header.get("TIMEUNIT", "s")).lower() not in {"s", "sec", "second"}:
        raise ValueError("FXT input TIMEUNIT must be seconds")
    if "MJDREFI" in header or "MJDREFF" in header:
        epoch = Time(float(header.get("MJDREFI", 0)), float(header.get("MJDREFF", 0)),
                     format="mjd", scale=str(header.get("TIMESYS", "tt")).lower())
    elif "MJDREF" in header:
        epoch = Time(float(header["MJDREF"]), format="mjd",
                     scale=str(header.get("TIMESYS", "tt")).lower())
    else:
        raise ValueError("Missing FITS MJDREF; cannot establish EP time")
    offset = float((epoch - Time(0, format="ep")).to_value("s"))
    if abs(offset) > 1e-3:
        raise ValueError("FXT FITS reference does not match the registered EP epoch")
    return float(header.get("TIMEZERO", 0.0))


def _count_observation(spectrum: PreparedSpectrum, band: tuple[float, float]):
    source = read_pha(spectrum.source_pha or spectrum.grouped_pha)
    background = read_pha(spectrum.background_pha)
    response = read_rmf(spectrum.rmf)
    if not np.array_equal(source.channels, background.channels):
        raise ValueError("Source/background PHA channels differ")
    if not np.array_equal(source.channels, response.channel):
        raise ValueError("PHA channels must match the RMF EBOUNDS for FXT counting")
    if any(value is None or float(value) <= 0 for value in
           (source.backscal, background.backscal, source.exposure, background.exposure)):
        raise ValueError("FXT counting requires positive EXPOSURE and BACKSCAL")
    if any(value is not None and float(value) != 1.0 for value in
           (source.areascal, background.areascal)):
        raise ValueError("Non-unit AREASCAL requires explicit FXT counting treatment")
    mask = channel_mask_from_ebounds(
        (response.channel, response.e_min, response.e_max),
        EnergyBand(emin=band[0], emax=band[1], emin_unit="keV", emax_unit="keV"), None,
    )
    if source.quality is not None:
        mask &= np.asarray(source.quality) == 0
    on, off = np.asarray(source.counts)[mask], np.asarray(background.counts)[mask]
    if any(np.any(values < 0) or not np.allclose(values, np.rint(values))
           for values in (on, off)):
        raise ValueError("FXT ON/OFF likelihood requires raw nonnegative integer counts")
    alpha_rate = float(source.backscal) / float(background.backscal)
    alpha = alpha_rate * float(source.exposure) / float(background.exposure)
    observation = UpperLimitObservation(
        name=f"{spectrum.obsid}:{spectrum.module}", source_counts=on,
        background_counts=off, alpha=alpha, exposure_s=float(source.exposure),
        response_path=spectrum.rmf, arf_path=spectrum.arf,
        metadata={"channel_mask": mask.tolist()},
    )
    row = {
        "instrument": "FXT", "obsid": spectrum.obsid, "module": spectrum.module,
        "n_on": float(on.sum()), "n_off": float(off.sum()), "alpha": alpha,
        "alpha_rate": alpha_rate, "exposure_s": float(source.exposure),
        "background_exposure_s": float(background.exposure),
        "net_counts": float(on.sum() - alpha * off.sum()),
        "net_rate": float((on.sum() - alpha * off.sum()) / source.exposure),
        "net_rate_error": float(np.sqrt(on.sum() + alpha**2 * off.sum()) / source.exposure),
        "significance": float(li_ma_snr(n_on=float(on.sum()), n_off=float(off.sum()), alpha=alpha)),
        "count_band_keV": band,
    }
    return observation, row


@register_pipeline("ep.fxt")
class FXTPipeline:
    """Fit local FXT epochs and return common-band measurements for TA.

    Default candidates are PL with zero/free additional absorption. Galactic NH
    is resolved through UKSSDC unless explicitly supplied (XSPEC 1e22 cm^-2).
    The conservative default fit band is 0.5--10 keV; a calibrated 0.3-keV lower
    bound can be explicitly supplied. Each epoch has its own NH/shape/flux.
    Non-detections require an explicit fixed ``upper_limit_shape`` (Gamma and
    additional NH); no spectrum from another epoch is silently inherited.
    """

    def __init__(
        self, input_data: FXTInput, *, config: InstrumentConfig | None = None,
        galactic_nh_1e22: float | None = None,
        candidate_keys: Sequence[str] | None = None,
        fit_energy_range: tuple[float, float] = (0.5, 10.0),
        flux_band_keV: tuple[float, float] = (0.5, 4.0),
        upper_limit_shape: Mapping[str, float] | None = None,
        upper_limit_sigma: float = 3.0, selection_metric: str = "aic",
        fit_options: Mapping[str, Any] | None = None,
        xselect_env: Mapping[str, str] | None = None,
    ):
        self.input = input_data
        self.config = config or FXT()
        self.galactic_nh = galactic_nh_1e22
        self.candidates = tuple(candidate_keys or ("powerlaw_nh0", "powerlaw_free_nh"))
        self.fit_band = tuple(map(float, fit_energy_range))
        self.flux_band = tuple(map(float, flux_band_keV))
        self.ul_shape = None if upper_limit_shape is None else dict(upper_limit_shape)
        self.ul_sigma = float(upper_limit_sigma)
        self.metric = selection_metric
        cfg = self.config.fitting
        if cfg.method != "mle":
            raise NotImplementedError("FXT model comparison currently requires method='mle'")
        self.fit_options = dict(stat_method=cfg.statistic, abundance=cfg.abundance,
            cross_section=cfg.cross_section, apec_root=cfg.apec_root,
            calculate_errors=cfg.calculate_errors, error_delta_stat=cfg.error_delta_stat,
            plot_formats=self.config.plotting.formats if self.config.plotting.enabled else (),
            plot_density=self.config.plotting.dpi,
            plot_required=self.config.plotting.enabled and self.config.plotting.required)
        self.fit_options.update(fit_options or {})
        self.xselect_env = dict(xselect_env or {})
        for band in (self.fit_band, self.flux_band):
            if not np.all(np.isfinite(band)) or len(band) != 2 or not 0 < band[0] < band[1]:
                raise ValueError("Energy bands must be increasing positive keV pairs")
        if not np.isfinite(input_data.redshift) or input_data.redshift < 0:
            raise ValueError("redshift must be finite and nonnegative")
        if self.galactic_nh is not None and (not np.isfinite(self.galactic_nh) or self.galactic_nh < 0):
            raise ValueError("galactic_nh_1e22 must be finite and nonnegative")
        if not np.isfinite(self.ul_sigma) or self.ul_sigma <= 0:
            raise ValueError("upper_limit_sigma must be positive")
        controlled = {"outdir", "candidate_keys", "galactic_nh_1e22", "redshift", "emin", "emax", "flux_band_keV", "selection_metric"}
        if controlled.intersection(self.fit_options):
            raise ValueError("Use FXTPipeline arguments rather than duplicate controlled fit_options")

    def _discover(self, outdir: Path, overwrite: bool):
        if self.input.spectra:
            return list(self.input.spectra), []
        catalog = scan(self.input.resolved_root(), instrument="FXT")
        # Reject duplicate versions/regions before core preparation's last-wins pairing.
        seen = set()
        for bundle in catalog.bundles:
            key = (bundle.obsid, bundle.module)
            if key in seen:
                raise ValueError(f"Ambiguous FXT spectra for {key}; select a version/region explicitly")
            seen.add(key)
        prepared = prepare_spectra(catalog, outdir=outdir / "prepared",
                                   group_min=self.config.spectrum.group_min_counts, overwrite=overwrite)
        source_ids = {bundle.source_pha.path: bundle.source_pha.source_id
                      for bundle in catalog.bundles if bundle.source_pha is not None}
        for spectrum in prepared.spectra:
            spectrum.source_id = source_ids.get(spectrum.source_pha, spectrum.source_id)
        return prepared.spectra, catalog.manifests

    def _subdivide(self, spectra, manifests, interval, outdir, overwrite):
        if not self.input.response_compatible:
            raise ValueError("Time re-extraction requires response_compatible=True after region/time calibration review")
        extracted = []
        for spectrum in spectra:
            key = f"{spectrum.obsid}:{spectrum.module}"
            files = [file for manifest in manifests for file in manifest.files
                     if file.obsid == spectrum.obsid and file.module == spectrum.module]
            event = self.input.event_files.get(key)
            roles = {}
            for role, region_map in (("src", self.input.source_regions), ("bkg", self.input.background_regions)):
                candidates = [file for file in files if file.role == "cleaned_event"
                              and file.meta.get("region") == role and file.source_id == spectrum.source_id]
                if event is not None:
                    if key not in region_map:
                        raise ValueError(f"Missing explicit {role} region for {key}")
                    event_path, region = Path(event), region_map[key]
                elif len(candidates) == 1:
                    event_path, region = candidates[0].path, None
                else:
                    raise ValueError(f"Missing unambiguous {role} events for {key}; cannot re-extract")
                offset = _time_offset(_header(event_path))
                pfiles = outdir / spectrum.module / ".pfiles" / role
                pfiles.mkdir(parents=True, exist_ok=True)
                headas = self.xselect_env.get("HEADAS", os.environ.get("HEADAS"))
                env = {"XSELECT_MDB": str(resource_files("jinwu.data").joinpath("xselect.mdb")),
                       "PFILES": f"{pfiles};{headas}/syspfiles" if headas else str(pfiles),
                       "HEADASNOQUERY": "1", "HEADASPROMPT": "/dev/null", **self.xselect_env}
                result = extract_products_with_xselect(
                    event_path, outdir / spectrum.module, products=("spectrum",),
                    prefix=key.replace(":", "_"), role=role, region=region,
                    time_range=(interval[0] - offset, interval[1] - offset), overwrite=overwrite,
                    env=env, timeout=self.config.execution.command_timeout_s,
                )
                path = result.spectrum
                original = spectrum.source_pha if role == "src" else spectrum.background_pha
                with fits.open(path, mode="update") as hdul:
                    hdul[1].header["BACKSCAL"] = _header(original)["BACKSCAL"]
                roles[role] = DataFile(path, "source_pha" if role == "src" else "background_pha",
                                      "time_extraction", "EP", "FXT", spectrum.obsid, spectrum.module)
            bundle = SpectrumBundle(roles["src"], roles["bkg"],
                DataFile(Path(spectrum.arf), "arf", "supplied", "EP", "FXT"),
                DataFile(Path(spectrum.rmf), "rmf", "supplied", "EP", "FXT"), spectrum.module, spectrum.obsid)
            manifest = Manifest(outdir, "EP", "FXT", "explicit", bundles=[bundle])
            extracted.extend(prepare_spectra(manifest, outdir=outdir / "prepared", overwrite=overwrite).spectra)
        return extracted

    def _upper_limit(self, observations, interval, outdir, nh):
        if self.ul_shape is None or "photon_index" not in self.ul_shape or "intrinsic_nh_1e22" not in self.ul_shape:
            raise ValueError("Non-detection requires explicit upper_limit_shape photon_index and intrinsic_nh_1e22")
        gamma, intrinsic = float(self.ul_shape["photon_index"]), float(self.ul_shape["intrinsic_nh_1e22"])
        if not np.isfinite(gamma) or not np.isfinite(intrinsic) or intrinsic < 0:
            raise ValueError("Upper-limit shape must be finite, with nonnegative additional NH")
        unit_flux = 1e-12
        model = CountTemplateModel(
            "fixed_absorbed_powerlaw", XspecCountPredictor(
                "tbabs*ztbabs*cflux*powerlaw",
                (nh, intrinsic, self.input.redshift, *self.flux_band, np.log10(unit_flux), gamma, 1.0),
                abundance=self.fit_options["abundance"], cross_section=self.fit_options["cross_section"],
            ), amplitude_unit="1e-12 erg cm^-2 s^-1 unabsorbed flux",
            flux_per_amplitude=unit_flux,
            metadata={"shape": self.ul_shape, "flux_band_keV": self.flux_band, "redshift": self.input.redshift},
        )
        config = copy.copy(self.config)
        config.upper_limit = replace(config.upper_limit, default_sigma=self.ul_sigma,
                                    upper_confidence_sigma=self.ul_sigma, result_modes=("observed_upper_bound",))
        return estimate_upper_limit(observations, model=model, instrument_config=config,
                                    interval=interval, energy_band=self.fit_band, output_dir=outdir)

    def run(self, *, overwrite: bool = False) -> FXTResult:
        """Write derived products outside the input tree; return per-epoch status.

        Source/background count curves remain separate by module. Flux rows are
        seconds since the supplied UTC T0, or EP MET with ``time_reference='ep'``.
        ``overwrite`` only applies to this explicitly selected output directory.
        """
        root, output = self.input.resolved_root(), self.input.resolved_output_root()
        if output == root or output.is_relative_to(root):
            raise ValueError("FXT output must be outside the input product directory")
        report = output / "fxt_result.json"
        if report.exists() and not overwrite:
            raise FileExistsError(f"Preserve existing results: {report}; select a new output directory")
        output.mkdir(parents=True, exist_ok=True)
        result = FXTResult(output, "missing_inputs")
        t0 = float(Time(self.input.trigger_time_utc, format="isot", scale="utc").ep) if self.input.trigger_time_utc else 0.0
        time_reference = "seconds_since_t0" if self.input.trigger_time_utc else "ep"
        nh = self.galactic_nh
        if nh is None:
            nh_result = resolve_galactic_absorption(self.input.ra_deg, self.input.dec_deg,
                                                  cache_path=output / "galactic_absorption.json")
            nh = nh_result.tbabs_nh_1e22
        spectra, manifests = self._discover(output, overwrite)
        by_obsid = {}
        for spectrum in spectra:
            if not spectrum.obsid or spectrum.module not in {"FXTA", "FXTB"}:
                raise ValueError("FXT spectra require an obsid and FXTA/FXTB module")
            group = by_obsid.setdefault(spectrum.obsid, [])
            if any(item.module == spectrum.module for item in group):
                raise ValueError(f"Duplicate FXT module for {spectrum.obsid}; select explicit inputs")
            group.append(spectrum)
        unknown = set(self.input.intervals).union(self.input.nondetection_obsids) - set(by_obsid)
        if unknown:
            raise ValueError(f"Requested FXT observations have no spectra: {sorted(unknown)}")
        flux_points = []
        for obsid, original in sorted(by_obsid.items()):
            intervals = self.input.intervals.get(obsid)
            segments = list(intervals) if intervals is not None else [None]
            if not segments:
                raise ValueError(f"Empty time intervals for {obsid}")
            for index, relative_interval in enumerate(segments):
                segment = f"seg{index:03d}" if relative_interval is not None else "epoch"
                epoch = {"obsid": obsid, "segment": segment, "status": "failed", "pileup_status":
                         "reviewed" if self.input.pileup_reviewed else "deferred", "warnings": []}
                destination = output / obsid / segment
                destination.mkdir(parents=True, exist_ok=True)
                result.epoch_results.append(epoch)
                try:
                    selected = original
                    if relative_interval is not None:
                        interval = tuple(float(value) + t0 for value in relative_interval)
                        if len(interval) != 2 or not np.all(np.isfinite(interval)) or interval[1] <= interval[0]:
                            raise ValueError("Time intervals must contain finite start < stop")
                        selected = self._subdivide(original, manifests, interval, destination / "extracted", overwrite)
                    if any(not spectrum.ready for spectrum in selected):
                        raise ValueError("Incomplete/failed prepared FXT spectrum: " + "; ".join(message for spectrum in selected for message in spectrum.diagnostics))
                    headers = [_header(spectrum.source_pha or spectrum.grouped_pha) for spectrum in selected]
                    bounds = [(float(h["TSTART"]) + _time_offset(h), float(h["TSTOP"]) + _time_offset(h)) for h in headers]
                    # Both modules must cover the same epoch, not just contain
                    # the shorter interval somewhere inside the longer one.
                    overlap = min(b for a, b in bounds) - max(a for a, b in bounds)
                    if (not np.all(np.isfinite(bounds)) or any(b <= a for a, b in bounds)
                            or overlap <= 0 or overlap < 0.99 * max(b - a for a, b in bounds)):
                        raise ValueError("FXTA/B spectra do not share the same epoch coverage")
                    start, stop = min(a for a, b in bounds), max(b for a, b in bounds)
                    if len(selected) == 1:
                        epoch["warnings"].append("Only one FXT module is available")
                    observations, counts = zip(*[_count_observation(item, self.fit_band) for item in selected])
                    for row in counts:
                        row.update(segment=segment, t_start=start - t0, t_stop=stop - t0,
                                   t_center=(start + stop) / 2 - t0, time_reference=time_reference,
                                   t0_utc=self.input.trigger_time_utc)
                        result.count_rows.append(row)
                        self._epoch_count_curve(row, destination, result, t0)
                    exposure = max(row["exposure_s"] for row in counts)
                    epoch.update(t_start=start - t0, t_stop=stop - t0, exposure_s=exposure,
                                 time_reference=time_reference, module_counts=list(counts),
                                 input_products=[_provenance(item) for item in selected],
                                 response_reuse_reviewed=self.input.response_compatible if relative_interval is not None else None)
                    row = {"instrument": "FXT", "obsid": obsid, "segment": segment,
                           "t_start": start - t0, "t_stop": stop - t0, "t_center": (start + stop) / 2 - t0,
                           "exposure_s": exposure, "time_reference": time_reference,
                           "t0_utc": self.input.trigger_time_utc, "flux_band_keV": self.flux_band,
                           "flux_frame": "observer", "flux_kind": "unabsorbed", "flux_unit": "erg cm^-2 s^-1",
                           "galactic_nh_1e22": nh, "redshift": self.input.redshift,
                           "pileup_status": epoch["pileup_status"]}
                    if obsid in self.input.nondetection_obsids:
                        upper = self._upper_limit(observations, (start, stop), destination / "upper_limit", nh)
                        limit = upper.observed_upper_bound
                        if limit is None or limit.flux_upper is None:
                            raise ValueError("Upper-limit engine returned no flux upper bound")
                        row.update(flux=limit.flux_upper, error_minus=None, error_plus=None,
                                   is_upper_limit=True, confidence=limit.level.confidence,
                                   model=upper.model_name, model_key="fixed_powerlaw", status="conditional_on_shape",
                                   shape=self.ul_shape, construction=limit.construction)
                        epoch["upper_limit"] = upper.artifacts
                    else:
                        joint = PreparedJointSpectrum(tuple(selected), obsid, tuple(item.module for item in selected))
                        comparison = fit_xray_models(
                            joint, outdir=destination / "fit", candidate_keys=self.candidates,
                            selection_metric=self.metric, galactic_nh_1e22=nh, redshift=self.input.redshift,
                            emin=self.fit_band[0], emax=self.fit_band[1], flux_band_keV=self.flux_band,
                            **self.fit_options,
                        )
                        fit = comparison.adopted_fit
                        flux = fit.get("flux_unabs")
                        if not flux or not np.isfinite(float(flux["erg_cm2_s"])):
                            raise ValueError("Adopted fit has no validated unabsorbed common-band flux")
                        row.update(flux=float(flux["erg_cm2_s"]), error_minus=flux.get("error_lo"),
                                   error_plus=flux.get("error_hi"), is_upper_limit=False,
                                   confidence=fit.get("xspec_settings", {}).get("profile_confidence"),
                                   error_kind=fit.get("xspec_settings", {}).get("profile_error_kind"),
                                   error_status=flux.get("error_status"),
                                   shape_status=fit.get("shape_status", {}).get("status"),
                                   model=fit.get("model"), model_key=comparison.adopted_key, status="spectral_fit")
                        if fit.get("model_family") in {"apec", "double_apec"}:
                            row["redshift"] = 0.0
                        epoch["comparison"] = comparison.to_dict()
                        epoch["warnings"].extend(comparison.warnings)
                        for model_key, candidate in epoch["comparison"].get("candidates", {}).items():
                            for i, path in enumerate(candidate.get("fit_products", {}).get("plots", [])):
                                result.products[f"{obsid}:{segment}:{model_key}:spectrum_plot_{i}"] = str(path)
                        if row["shape_status"] and row["shape_status"] != "constrained":
                            epoch["warnings"].append(f"Adopted spectral shape: {row['shape_status']}")
                    result.flux_rows.append(row)
                    flux_points.append(FluxPoint(
                        f"{obsid}_{segment}", row["t_start"], row["t_stop"], row["t_center"], (stop - start) / 2,
                        row["flux"], row["error_minus"], row["error_plus"], *self.flux_band,
                        "upper_limit" if row["is_upper_limit"] else "spectral_fit", not row["is_upper_limit"],
                        model=row["model"], model_key=row["model_key"], model_expression=row["model"],
                    ))
                    epoch["status"] = "completed"
                    result.warnings.extend(f"{obsid}/{segment}: {warning}" for warning in epoch["warnings"])
                    # Reuse only source/background LC bins with exactly matching identity.
                    if relative_interval is None:
                        for spectrum, count in zip(selected, counts):
                            self._count_curve(spectrum, count["alpha_rate"], manifests, destination, result, t0)
                except Exception as exc:
                    epoch["error"] = f"{type(exc).__name__}: {exc}"
                    result.warnings.append(f"{obsid}/{segment}: {epoch['error']}")
                write_json(destination / "epoch.json", epoch)
        if result.flux_rows:
            products = save_flux_curve(FluxCurveResult(tuple(flux_points)), output / "flux_curve", formats=())
            result.products.update({key: str(path) for key, path in products.items()})
            result.products.update(self._plot_flux(result.flux_rows, output))
            # Common TA fields are preserved without Flattening tuple/list metadata.
            write_json(output / "flux_rows.json", result.flux_rows)
            result.products["flux_rows"] = str(output / "flux_rows.json")
            Table(rows=[{key: value for key, value in row.items() if key != "shape"}
                        for row in result.flux_rows]).write(output / "flux_rows.ecsv", overwrite=True)
            result.products["flux_table"] = str(output / "flux_rows.ecsv")
        completed = sum(epoch["status"] == "completed" for epoch in result.epoch_results)
        result.status = ("completed" if completed == len(result.epoch_results) else "partial") if completed else "failed"
        if not result.epoch_results:
            result.status = "missing_inputs"
        if not self.input.pileup_reviewed:
            result.warnings.append("Pile-up processing deferred for supplied L2/L3 products")
        result.products["report"] = str(report)
        write_json(report, {"result": result, "input": self.input, "environment": collect_runtime_environment(),
                            "fit_band_keV": self.fit_band, "flux_band_keV": self.flux_band,
                            "candidate_keys": self.candidates, "galactic_nh_1e22": nh})
        return result

    def _count_curve(self, spectrum, alpha_rate, manifests, destination, result, t0):
        files = [file for manifest in manifests for file in manifest.files
                 if file.obsid == spectrum.obsid and file.module == spectrum.module
                 and file.source_id == spectrum.source_id]
        source = [file for file in files if file.role == "source_lightcurve"]
        background = [file for file in files if file.role == "background_lightcurve"]
        pairs = [(a, b) for a in source for b in background
                 if a.meta.get("binsize") == b.meta.get("binsize") and a.meta.get("version") == b.meta.get("version")]
        if not pairs:
            result.warnings.append(f"{spectrum.obsid}/{spectrum.module}: no matching count LC; per-epoch counts retained")
            return
        # Choose finest existing bins without mixing detector/version products.
        def width(pair):
            return float(_header(pair[0].path).get("TIMEDEL", np.inf))
        source, background = min(pairs, key=width)
        try:
            offset = _time_offset(_header(source.path))
            if abs(offset - _time_offset(_header(background.path))) > 1e-7:
                raise ValueError("Source/background LC TIMEZERO differs")
            curve = build_net_lightcurve(source.path, background.path, alpha=alpha_rate)
            base = destination / f"{spectrum.module}_count_curve"
            saved = save_net_lightcurve(curve, base)
            plots = plot_net_lightcurve(curve, base, title=f"{spectrum.obsid} {spectrum.module}",
                                       timezero=t0 if self.input.trigger_time_utc else None,
                                       timezero_obj=Time(self.input.trigger_time_utc, format="isot", scale="utc") if self.input.trigger_time_utc else None,
                                       formats=("png", "pdf"))
            for key, path in saved.items():
                result.products[f"{spectrum.obsid}:{spectrum.module}:count_{key}"] = str(path)
            for path in plots:
                result.products[f"{spectrum.obsid}:{spectrum.module}:count_{path.suffix[1:]}"] = str(path)
        except Exception as exc:
            result.warnings.append(f"{spectrum.obsid}/{spectrum.module}: count LC unavailable: {exc}")

    def _epoch_count_curve(self, row, destination, result, t0):
        """An observation-bin count LC is available even when no L3 LC exists."""
        width = row["t_stop"] - row["t_start"]
        source_exposure, background_exposure = row["exposure_s"], row["background_exposure_s"]
        curve = NetLightcurve(
            np.asarray([row["t_center"] + t0]), np.asarray([width]),
            np.asarray([source_exposure / width]), np.asarray([row["n_on"] / source_exposure]),
            np.asarray([np.sqrt(row["n_on"]) / source_exposure]),
            np.asarray([row["n_off"] / background_exposure]),
            np.asarray([np.sqrt(row["n_off"]) / background_exposure]),
            np.asarray([row["net_rate"]]), np.asarray([row["net_rate_error"]]), row["alpha_rate"], 0.0,
        )
        base = destination / f"{row['module']}_epoch_count_curve"
        products = save_net_lightcurve(curve, base)
        for path in plot_net_lightcurve(curve, base, title=f"{row['obsid']} {row['module']} observation bin",
                                       timezero=t0 if self.input.trigger_time_utc else None, formats=("png", "pdf")):
            products[f"plot_{path.suffix[1:]}"] = path
        for key, path in products.items():
            result.products[f"{row['obsid']}:{row['segment']}:{row['module']}:epoch_count_{key}"] = str(path)

    def _plot_flux(self, rows, output):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for row in rows:
            x, y = row["t_center"], row["flux"]
            xerr = (row["t_stop"] - row["t_start"]) / 2
            if row["is_upper_limit"]:
                ax.errorbar(x, y, xerr=xerr, yerr=0.25 * y, uplims=True, fmt="v", color="tab:blue")
            else:
                errors = None if row["error_minus"] is None or row["error_plus"] is None else np.asarray([[row["error_minus"]], [row["error_plus"]]])
                ax.errorbar(x, y, xerr=xerr, yerr=errors, fmt="o", color="tab:blue")
        if all(row["t_start"] > 0 for row in rows):
            ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Time since T0 [s]" if self.input.trigger_time_utc else "EP mission time [s]")
        ax.set_ylabel(f"Unabsorbed {self.flux_band[0]:g}–{self.flux_band[1]:g} keV flux\n[erg cm$^{{-2}}$ s$^{{-1}}$]")
        ax.set_title(f"{self.input.target_id} — FXT")
        fig.tight_layout()
        products = {}
        for extension in ("png", "pdf"):
            path = output / f"flux_curve.{extension}"
            fig.savefig(path, dpi=200)
            products[f"flux_plot_{extension}"] = str(path)
        plt.close(fig)
        return products
