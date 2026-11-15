"""One-source analysis of prepared spectra and calibrated time-series products.

Instrument reduction stays in its plugin. This workflow consumes explicit
products, fits each chosen epoch, and puts observations on a common clock.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from functools import cached_property
from importlib.metadata import version
import os
from pathlib import Path
import sys
import tempfile

import astropy.units as u
from astropy.table import Table
import numpy as np

from .config import ExecutionConfig, FitConfig
from .pipeline import InstrumentPipeline, PipelineInput, PipelineStage, StageResult, register_pipeline
from .products import jsonable, write_json
from .spectrum_prep import PreparedSpectrum
from .time import Time

__all__ = [
    "SpectralObservation", "LightcurveProduct", "SourceAnalysisInput",
    "SourceAnalysisConfig", "SourceAnalysisPipeline",
]

_SPECTRUM_PATHS = ("source_pha", "grouped_pha", "background_pha", "rmf", "arf")


@dataclass(frozen=True)
class SpectralObservation:
    """A prepared spectrum from one UTC interval of the input target.

    Equal, nonempty ``epoch`` labels request a joint fit. Their start/stop times
    must overlap and agree within the configured tolerance; unrelated epochs
    remain separate.
    ``statistic`` explicitly selects chi/cstat/pgstat/pstat for this spectrum.
    """
    spectrum: PreparedSpectrum
    start_utc: str
    stop_utc: str
    statistic: str
    epoch: str
    target_id: str | None = None


@dataclass(frozen=True)
class LightcurveProduct:
    """ECSV table of one instrument/band, with unit-bearing columns.

    Required columns: time_start/time_stop (seconds relative to reference_utc),
    value/error (same physical unit). Rates/fluxes are never added across
    instruments. ``band`` identifies the actual calibrated band or filter.
    This small file contract also accepts output from non-JinWu pipelines.
    """
    path: str | Path
    instrument: str
    band: str
    reference_utc: str
    target_id: str | None = None


@dataclass(frozen=True)
class SourceAnalysisInput(PipelineInput):
    """Explicit association of products with one source on an observer UTC clock.

    Product paths may be absolute or relative to root. The caller establishes
    source identity and supplies consistently corrected arrival times. Optional
    product target identifiers are checked against target_id.
    """
    spectra: tuple[SpectralObservation, ...] = ()
    lightcurves: tuple[LightcurveProduct, ...] = ()
    reference_utc: str = ""


@dataclass(frozen=True)
class SourceAnalysisConfig:
    """Common flux band and opt-in temporal model, using existing fit engines.

    Spectral fits run sequentially because PyXspec owns process-wide state.
    ``fit_options`` forwards scientific options (NH, redshift, frozen parameters)
    to fit_spectral. The workflow currently uses the MLE backend so mixed
    likelihoods and cross-calibration have explicit, tested semantics.
    Temporal fits use the supplied Gaussian errors and evaluate the model at
    bin centers. They run only when ``lightcurve_model`` is explicitly set.
    """
    fit: FitConfig = field(default_factory=lambda: FitConfig(model_name="cflux*powerlaw"))
    flux_band_keV: tuple[float, float] = (0.5, 4.0)
    joint_normalization: str = "tied"
    simultaneity_tolerance_s: float = 0.001
    fit_options: dict = field(default_factory=dict)
    lightcurve_model: str | None = None
    lightcurve_fit_options: dict = field(default_factory=dict)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    name: str = field(default="SourceAnalysis", init=False)
    pipeline: str = field(default="source", init=False)


@register_pipeline("source")
class SourceAnalysisPipeline(InstrumentPipeline[SourceAnalysisInput, dict]):
    """Validate → spectral fits → time-series analysis/plots → source report."""
    stages = (
        PipelineStage("validate"),
        PipelineStage("fit", ("validate",)),
        PipelineStage("lightcurves", ("validate", "fit")),
        PipelineStage("report", ("validate", "fit", "lightcurves")),
    )

    def __init__(self, input_data: SourceAnalysisInput, *, config: SourceAnalysisConfig | None = None):
        super().__init__(input_data, config=config or SourceAnalysisConfig())

    def _path(self, path):
        path = Path(path).expanduser()
        return (path if path.is_absolute() else self.input.resolved_root() / path).resolve()

    def stage_input_dependencies(self, stage):
        paths = [self._path(item.path) for item in self.input.lightcurves]
        for item in self.input.spectra:
            paths.extend(self._path(getattr(item.spectrum, name))
                         for name in _SPECTRUM_PATHS if getattr(item.spectrum, name) is not None)
        if self.input.spectra and self.config.fit.enabled and "apec" in self.config.fit.model_name.lower():
            apec = self.config.fit_options.get("apec_root", self.config.fit.apec_root)
            if apec is not None:
                paths.extend(self._path(f"{apec}_{name}.fits") for name in ("line", "coco"))
        return tuple(dict.fromkeys(paths))

    def stage_code_dependencies(self, stage):
        root = Path(__file__).parent
        return tuple(root / name for name in (
            "source_analysis.py", "fit.py", "plot.py", "plotstyle.py", "plotpanel.py",
            "spectrum_prep.py", "config.py", "products.py", "time.py", "pipeline.py",
        ))

    @staticmethod
    def runtime_versions(spectral: bool) -> dict:
        """Record dependencies used for fitting and cache validation."""
        runtime = {name: version(name) for name in ("numpy", "scipy", "astropy", "matplotlib", "pyerfa")}
        runtime.update({"python": sys.version.split()[0], "python_executable": sys.executable})
        if spectral:
            from .fit import _require_xspec
            try:
                xspec = _require_xspec()
            except ImportError:
                # Let the actual fit report a missing engine and record the
                # failed stage without a second import error from fingerprinting.
                runtime["xspec"] = None
            else:
                runtime.update({"xspec": xspec.Xset.version, "xspec_path": xspec.__file__})
            runtime["headas"] = os.environ.get("HEADAS")
        return runtime

    @cached_property
    def _runtime(self):
        return self.runtime_versions(bool(self.input.spectra and self.config.fit.enabled))

    def stage_config_dependencies(self, stage):
        return {"config": self.config, "runtime": self._runtime}

    @staticmethod
    def _utc(value, field_name):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} must be a scalar ISO UTC timestamp")
        return Time(value, scale="utc")

    def validate_input(self):
        from .fit import _prepared_spectrum_key
        if not self.input.target_id.strip():
            raise ValueError("target_id must identify one source")
        if not self.input.spectra and not self.input.lightcurves:
            raise ValueError("Supply spectra or lightcurves for this source")
        self._utc(self.input.reference_utc, "reference_utc")
        c = self.config
        if self.input.spectra and c.fit.enabled and (c.fit.method != "mle" or c.fit.backend != "xspec"):
            raise ValueError("SourceAnalysis currently supports the XSPEC MLE backend")
        if c.joint_normalization not in {"independent", "tied", "cross_calibration"}:
            raise ValueError("Unknown joint_normalization")
        if not np.isfinite(c.simultaneity_tolerance_s) or c.simultaneity_tolerance_s < 0:
            raise ValueError("simultaneity_tolerance_s must be finite and nonnegative")
        protected = {
            "prepared", "outdir", "method", "settings", "statistics", "stat_method",
            "joint_normalization", "flux_band_keV", "model_name", "calculate_errors",
            "error_delta_stat", "error_command", "srcname", "plot_required", "explicit_bindings",
            "abundance", "cross_section", "chain_path", "chain_kwargs",
        }
        if protected & c.fit_options.keys():
            raise ValueError(f"fit_options cannot override {sorted(protected & c.fit_options.keys())}")
        if c.fit_options.get("shared_absorption_only") and c.joint_normalization != "independent":
            raise ValueError("shared_absorption_only requires independent normalizations")
        if {"model", "absolute_sigma", "sigma"} & c.lightcurve_fit_options.keys():
            raise ValueError("lightcurve_fit_options cannot override the model or measurement errors")
        band = np.asarray(c.flux_band_keV, dtype=float)
        if band.shape != (2,) or not np.all(np.isfinite(band)) or not 0 < band[0] < band[1]:
            raise ValueError("flux_band_keV must be positive and increasing")
        epochs, keys, grouped_paths = {}, set(), set()
        for item in self.input.spectra:
            if item.target_id is not None and item.target_id != self.input.target_id:
                raise ValueError(f"Spectrum belongs to another target: {item.target_id!r}")
            key = (item.epoch, _prepared_spectrum_key(item.spectrum))
            if not item.epoch.strip() or key in keys:
                raise ValueError("Each epoch must have uniquely identified spectra")
            keys.add(key)
            if not item.spectrum.ready or item.spectrum.grouped_pha is None or item.spectrum.rmf is None:
                raise ValueError("Every spectrum requires prepared source data and an explicit response")
            grouped = self._path(item.spectrum.grouped_pha)
            if grouped in grouped_paths:
                raise ValueError(f"Grouped spectrum was supplied more than once: {grouped}")
            grouped_paths.add(grouped)
            if item.statistic not in {"chi", "cstat", "pgstat", "pstat"}:
                raise ValueError("Explicit supported statistic required for every spectrum")
            start = self._utc(item.start_utc, "start_utc")
            stop = self._utc(item.stop_utc, "stop_utc")
            if stop <= start:
                raise ValueError("Observation intervals must have increasing scalar UTC bounds")
            for other_start, other_stop in epochs.setdefault(item.epoch, []):
                mismatch = max(abs((start - other_start).sec), abs((stop - other_stop).sec))
                if mismatch > c.simultaneity_tolerance_s or max(start, other_start) >= min(stop, other_stop):
                    raise ValueError(f"Joint epoch {item.epoch!r} has different time intervals; fit separate epochs")
            epochs[item.epoch].append((start, stop))
        for path in self.stage_input_dependencies(self.stages[0]):
            if not path.is_file():
                raise FileNotFoundError(path)
            if self.workspace in path.parents:
                raise ValueError("Analysis workspace must not contain its input products")
        for item in self.input.lightcurves:
            self._lightcurve_arrays(item)

    def _lightcurve_arrays(self, product):
        if not product.instrument.strip() or not product.band.strip():
            raise ValueError("Lightcurve products require instrument and band labels")
        table = Table.read(self._path(product.path), format="ascii.ecsv")
        for target in (product.target_id, table.meta.get("target_id")):
            if target is not None and target != self.input.target_id:
                raise ValueError(f"Lightcurve belongs to another target: {target!r}")
        reference = self._utc(product.reference_utc, "lightcurve reference_utc")
        if "reference_utc" in table.meta and reference != self._utc(table.meta["reference_utc"], "ECSV reference_utc"):
            raise ValueError("LightcurveProduct reference_utc disagrees with the ECSV reference_utc")
        required = ("time_start", "time_stop", "value", "error")
        if any(name not in table.colnames or table[name].unit is None for name in required):
            raise ValueError("Lightcurve table needs unit-bearing time_start/time_stop/value/error")
        if any(table[name].ndim != 1 or np.any(np.ma.getmaskarray(table[name])) for name in required):
            raise ValueError("Lightcurve table requires scalar, unmasked measurements per row")
        offset = (reference - self._utc(self.input.reference_utc, "reference_utc")).sec
        start = table["time_start"].quantity.to_value(u.s) + offset
        stop = table["time_stop"].quantity.to_value(u.s) + offset
        unit = table["value"].unit
        values = table["value"].quantity.to_value(unit)
        errors = table["error"].quantity.to_value(unit)
        if not len(start) or any(not np.all(np.isfinite(a)) for a in (start, stop, values, errors)):
            raise ValueError("Lightcurve data must be nonempty and finite")
        if np.any(stop <= start) or np.any(errors < 0) or np.any(start[1:] < stop[:-1]):
            raise ValueError("Lightcurve bins must be ordered/nonoverlapping with nonnegative errors")
        metadata = {key: table.meta[key] for key in ("kind", "actual_energy_range_keV", "correct_dead_time")
                    if key in table.meta}
        return start, stop, values, errors, unit, offset, metadata

    def execute_stage(self, stage, context):
        if stage.name == "validate":
            return StageResult(data={"target_id": self.input.target_id, "reference_utc": self.input.reference_utc,
                                     "spectra": len(self.input.spectra), "lightcurves": len(self.input.lightcurves),
                                     "runtime_versions": self._runtime})
        if stage.name == "report":
            result = self.build_result(context)
            result["pipeline_status"] = "completed"
            path = write_json(self.workspace / "source_analysis.json", result)
            return StageResult(outputs={"report": str(path)})
        output = Path(tempfile.mkdtemp(prefix=f"{stage.name}-", dir=self.workspace))
        if stage.name == "fit":
            return self._fit(output)
        return self._lightcurves(output, context["fit"].data.get("epochs", []))

    def _fit(self, output):
        from .fit import fit_spectral, _prepared_spectrum_key
        if not self.config.fit.enabled or not self.input.spectra:
            reason = "disabled" if not self.config.fit.enabled else "no_spectra"
            return StageResult(data={"status": reason, "fit_performed": False, "epochs": []})
        epochs = {}
        for item in self.input.spectra:
            epochs.setdefault(item.epoch, []).append(item)
        rows, paths = [], {}
        for index, (epoch, observations) in enumerate(epochs.items()):
            spectra = []
            for item in observations:
                paths_resolved = {name: self._path(path) for name in _SPECTRUM_PATHS
                                  if (path := getattr(item.spectrum, name)) is not None}
                spectra.append(replace(item.spectrum, **paths_resolved))
            options = dict(self.config.fit_options)
            apec = options.get("apec_root", self.config.fit.apec_root)
            if apec is not None:
                options["apec_root"] = self._path(apec)
            result = fit_spectral(
                spectra, outdir=output / f"epoch_{index:03d}", method="mle", settings=self.config.fit,
                model_name=self.config.fit.model_name, calculate_errors=self.config.fit.calculate_errors,
                error_delta_stat=self.config.fit.error_delta_stat, flux_band_keV=self.config.flux_band_keV,
                joint_normalization=self.config.joint_normalization,
                statistics={_prepared_spectrum_key(spectrum): item.statistic
                            for spectrum, item in zip(spectra, observations)},
                srcname=self.input.target_id, plot_required=True, explicit_bindings=True, **options,
            )
            for key in ("result_json", "report_txt", "xspec_log", "xcm"):
                paths[f"{index}_{key}"] = result["fit_products"][key]
            for n, path in enumerate(result["fit_products"]["plots"]):
                paths[f"{index}_plot_{n}"] = path
            members = [
                {"key": _prepared_spectrum_key(spectrum), "instrument": spectrum.instrument,
                 "module": spectrum.module, "detector": spectrum.detector,
                 "start_utc": item.start_utc, "stop_utc": item.stop_utc}
                for spectrum, item in zip(spectra, observations)
            ]
            rows.append({"epoch": epoch, "start_utc": observations[0].start_utc,
                         "stop_utc": observations[0].stop_utc, "members": members, "fit": result})
        return StageResult(outputs=paths, data={"status": "completed", "fit_performed": True, "epochs": rows})

    def _lightcurves(self, output, epochs):
        import matplotlib.pyplot as plt
        from .fit import LightcurveFitter
        from .plotstyle import save_figure
        xlabel = f"Time since {self.input.reference_utc} UTC (s)"
        paths, summaries = {}, []
        for index, product in enumerate(self.input.lightcurves):
            start, stop, values, errors, unit, offset, metadata = self._lightcurve_arrays(product)
            center = (start + stop) / 2
            summary = {
                "instrument": product.instrument, "band": product.band, "unit": unit.to_string() or "1",
                "input_ecsv": str(self._path(product.path)), "input_reference_utc": product.reference_utc,
                "time_offset_s": offset, "bins": len(values), "time_range_s": [float(start[0]), float(stop[-1])],
                "min": float(values.min()), "max": float(values.max()), "metadata": metadata,
                "temporal_fit_status": "not_requested",
            }
            fit = None
            if self.config.lightcurve_model is not None:
                if np.any(errors <= 0):
                    summary["temporal_fit_status"] = "unavailable"
                    summary["temporal_fit_message"] = "Gaussian fitting requires strictly positive measurement errors"
                else:
                    fit = LightcurveFitter((center, values, errors)).fit(
                        self.config.lightcurve_model, absolute_sigma=True, **self.config.lightcurve_fit_options,
                    )
                    summary["temporal_fit"] = fit.to_dict()
                    summary["temporal_fit_status"] = (
                        "failed" if not fit.success else "requires_review"
                        if fit.dof <= 0 or fit.covariance is None else "completed"
                    )
            fig, ax = plt.subplots(figsize=(7, 3.8))
            try:
                ax.errorbar(center, values, xerr=(stop - start) / 2, yerr=errors, fmt=".", ms=3, lw=.7)
                if fit is not None and fit.success:
                    ax.plot(center, fit.fitted_curve, ".", ms=3, label=self.config.lightcurve_model)
                    ax.legend()
                kind = str(metadata.get("kind", "value")).replace("_", " ").capitalize()
                ax.set(xlabel=xlabel, ylabel=f'{kind} ({summary["unit"]})',
                       title=f"{self.input.target_id} · {product.instrument} · {product.band}")
                paths[f"lightcurve_{index}"] = str(save_figure(fig, output / f"lightcurve_{index:03d}")["png"])
            finally:
                plt.close(fig)
            summaries.append(summary)

        flux_rows = self._flux_rows(epochs)
        if flux_rows:
            paths["flux_table"] = str(write_json(output / "flux_lightcurve.json", flux_rows))
            fig, ax = plt.subplots(figsize=(7, 3.8))
            try:
                shown, colors = set(), {}
                for row in flux_rows:
                    instrument = row["instrument"]
                    if instrument not in colors:
                        colors[instrument] = f"C{len(colors) % 10}"
                    label = instrument
                    has_interval = row["error_status"] == "ok" and all(
                        value is not None and np.isfinite(value) and value > 0
                        for value in (row["error_lo"], row["error_hi"])
                    )
                    has_interval = has_interval and row["error_lo"] < row["flux"]
                    if not has_interval:
                        label += " (interval unavailable)"
                    error = [[row["error_lo"]], [row["error_hi"]]] if has_interval else None
                    ax.errorbar(
                        row["time_s"], row["flux"], xerr=row["time_error_s"], yerr=error,
                        fmt="o", color=colors[instrument], mfc=colors[instrument] if has_interval else "none", ms=4,
                        label=label if label not in shown else None,
                    )
                    shown.add(label)
                emin, emax = self.config.flux_band_keV
                ax.set(xlabel=xlabel, yscale="log", ylabel="Unabsorbed flux (erg cm$^{-2}$ s$^{-1}$)",
                       title=f"{self.input.target_id} · {emin:g}–{emax:g} keV (observer)")
                ax.legend(fontsize=7)
                paths["flux_plot"] = str(save_figure(fig, output / "flux_lightcurve")["png"])
            finally:
                plt.close(fig)
        return StageResult(outputs=paths, data={"lightcurves": summaries, "flux_rows": flux_rows})

    def _flux_rows(self, epochs):
        flux_rows = []
        reference = self._utc(self.input.reference_utc, "reference_utc")
        for epoch in epochs:
            result = epoch["fit"]
            groups = result["per_group"]
            joint = self.config.joint_normalization != "independent" and len(groups) > 1
            if joint:
                groups = [{"flux_unabs": result.get("flux_unabs"),
                           "member_spectra": [item["key"] for item in epoch["members"]]}]
            for group in groups:
                flux = group.get("flux_unabs") or {}
                value = flux.get("erg_cm2_s")
                if value is None or not np.isfinite(value) or value <= 0:
                    continue
                band = flux["energy_range_keV"]
                if not np.allclose(band, self.config.flux_band_keV, rtol=0, atol=1e-10) or flux["frame"] != "observer":
                    raise RuntimeError(f"Fit {epoch['epoch']!r} returned an unexpected flux band or frame")
                members = [item for item in epoch["members"] if item["key"] in group["member_spectra"]]
                labels = []
                for item in members:
                    instrument = item["instrument"]
                    scope = item["module"] or item["detector"] or instrument
                    labels.append(scope if scope.startswith(instrument) else f"{instrument} {scope}")
                start, stop = members[0]["start_utc"], members[0]["stop_utc"]
                bounds = (Time([start, stop], scale="utc") - reference).sec
                fit_bands = {item["key"]: result["energy_ranges"][item["key"]] for item in members}
                noticed_bands = {item["key"]: result["noticed_energy_ranges"].get(item["key"]) for item in members}
                flux_rows.append({
                    "epoch": epoch["epoch"], "start_utc": start, "stop_utc": stop,
                    "time_start_s": bounds[0], "time_stop_s": bounds[1], "time_s": bounds.mean(),
                    "time_error_s": (bounds[1] - bounds[0]) / 2, "reference_utc": self.input.reference_utc,
                    "instrument": "+".join(dict.fromkeys(labels)), "member_spectra": group["member_spectra"],
                    "flux": value, "unit": "erg cm-2 s-1", "error_lo": flux.get("error_lo"),
                    "error_hi": flux.get("error_hi"), "error_status": flux.get("error_status"),
                    "error_delta_stat": result["xspec_settings"]["error_delta_stat"],
                    "band_keV": band, "frame": flux["frame"], "definition": flux["definition"],
                    "fit_bands_keV": fit_bands, "noticed_energy_spans_keV": noticed_bands,
                    "cflux_extrapolated_spectra": [key for key in result["cflux_extrapolated_spectra"] if key in fit_bands],
                    "model_energy_grid": result.get("model_energy_grid"),
                    "joint_normalization": self.config.joint_normalization if joint else None,
                    "calibration_reference_spectrum": result.get("calibration_reference_spectrum"),
                })
        return sorted(flux_rows, key=lambda row: row["time_s"])

    def build_result(self, context):
        return jsonable({
            "target_id": self.input.target_id, "workspace": str(self.workspace),
            "pipeline_status": self.status().status.value, "science_status": "requires_review",
            "reference_utc": self.input.reference_utc, "config": asdict(self.config),
            "fit_performed": context.get("fit", StageResult()).data.get("fit_performed", False),
            "stages": {name: {"outputs": result.outputs, "data": result.data} for name, result in context.items()},
        })
