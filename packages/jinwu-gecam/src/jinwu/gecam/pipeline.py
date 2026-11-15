"""GECAMTools workflow using JinWu's shared stage and cache contracts."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from functools import cached_property
from importlib.metadata import version
from pathlib import Path
import tempfile
import warnings

import astropy.units as u
from astropy.io import fits
import numpy as np

from jinwu.core.config import ExecutionConfig, FitConfig
from jinwu.core.pipeline import InstrumentPipeline, PipelineInput, PipelineStage, PipelineStatus, StageResult, register_pipeline
from jinwu.core.products import write_json
from jinwu.core.source_analysis import SourceAnalysisConfig, SourceAnalysisInput, SourceAnalysisPipeline, LightcurveProduct
from jinwu.core.time import Time

from . import backend
from .analysis import _met_reference, gecam_lightcurve_products, prepare_gecam_spectra


@dataclass(frozen=True)
class GECAMInput(PipelineInput):
    """A GRD EVT file or finished full/background spectra and/or lightcurves.

    Input paths are relative to ``root`` unless absolute. For finished spectra
    ``met_reference_utc`` identifies the UTC instant of native MET zero.
    """

    event_file: str | Path = ""
    source_spectra: dict[int, str | Path] = field(default_factory=dict)
    background_spectra: dict[int, str | Path] = field(default_factory=dict)
    lightcurve_products: tuple[LightcurveProduct, ...] = ()
    met_reference_utc: str | None = None


@dataclass(frozen=True)
class GECAMConfig:
    """Explicit detector/gain and physical selections for the GECAM workflow.

    Time ranges are absolute mission MET (seconds), never offsets from trigger.
    Energy edges are Quantities convertible to keV. Source/background intervals
    are optional (N, 2) MET Quantities; supply both to generate native spectra.
    Spectrum binning uses ``spectrum_time_bin`` and integer ``channel_bin``,
    independently of the display lightcurve energy bins. ``responses`` maps GRD
    numbers to existing response paths for the selected gain; calibration and
    validity for the observation remain the caller's responsibility.
    ``analysis`` controls the shared fit and flux band. The default is
    cflux*powerlaw across the outer energy edges, with one PG-stat per detector.
    """

    detectors: tuple[int, ...]
    gain: str
    time_range: u.Quantity
    time_bin: u.Quantity
    energy_edges: u.Quantity
    only_recommend: bool = True
    correct_dead_time: bool = True
    source_intervals: u.Quantity | None = None
    background_intervals: u.Quantity | None = None
    spectrum_time_bin: u.Quantity = field(default_factory=lambda: 1 * u.s)
    channel_bin: int = 1
    fit_order: int = 1
    responses: dict[int, str | Path] = field(default_factory=dict)
    analysis: SourceAnalysisConfig | None = None
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    name: str = field(default="GECAM", init=False)
    pipeline: str = field(default="gecam", init=False)


@register_pipeline("gecam")
class GECAMPipeline(InstrumentPipeline[GECAMInput, dict]):
    """Inspect → lightcurve → spectra → analysis → report; resume via manifests.

    Outputs preserve upstream counts, errors, channel boundaries and type-II
    spectra. Completion is execution status, not scientific acceptance. Original
    inputs and earlier product generations are retained on forced reruns.
    """

    stages = (
        PipelineStage("inspect"),
        PipelineStage("lightcurve", ("inspect",)),
        PipelineStage("spectra", ("inspect",)),
        PipelineStage("analysis", ("inspect", "lightcurve", "spectra")),
        PipelineStage("report", ("inspect", "lightcurve", "spectra", "analysis")),
    )

    def __init__(self, input_data: GECAMInput, *, config: GECAMConfig):
        super().__init__(input_data, config=config)
        path = Path(input_data.event_file).expanduser()
        self.event_path = (path if path.is_absolute() else input_data.resolved_root() / path).resolve()

    def include_config_in_input_fingerprint(self):
        return False  # Quantities are normalized explicitly below, with units.

    def stage_config_dependencies(self, stage):
        values = asdict(self.config)
        for key, value in values.items():
            if isinstance(value, u.Quantity):
                unit = u.keV if key == "energy_edges" else u.s
                values[key] = {"value": value.to_value(unit).tolist(), "unit": str(unit)}
        # Include numerical dependencies as well as the upstream source hashes.
        names = ("numpy", "scipy", "astropy", "lmfit") if self.input.event_file else ("numpy", "scipy", "astropy")
        values["runtime_versions"] = {name: version(name) for name in names}
        if stage.name == "analysis":
            values["analysis_runtime_versions"] = self._analysis_runtime
        return values

    @cached_property
    def _analysis_config(self):
        edges = self.config.energy_edges.to_value(u.keV)
        return self.config.analysis or SourceAnalysisConfig(
            fit=FitConfig(model_name="cflux*powerlaw", statistic="pgstat"),
            flux_band_keV=(float(edges[0]), float(edges[-1])),
        )

    @property
    def _spectrum_detectors(self):
        if self.input.event_file:
            return set(self.config.detectors) if self.config.source_intervals is not None else set()
        return set(self.input.source_spectra)

    @cached_property
    def _analysis_runtime(self):
        detectors = self._spectrum_detectors
        spectral = bool(detectors and detectors <= self.config.responses.keys() and self._analysis_config.fit.enabled)
        try:
            return SourceAnalysisPipeline.runtime_versions(spectral)
        except ImportError as exc:
            return {"xspec_unavailable": str(exc)}

    def stage_input_dependencies(self, stage):
        paths = [self.event_path] if self.input.event_file else []
        paths.extend(self._input_path(p) for p in (*self.input.source_spectra.values(), *self.input.background_spectra.values(),
                                                   *(item.path for item in self.input.lightcurve_products)))
        paths.extend(Path(p).expanduser().resolve() for p in self.config.responses.values())
        return tuple(paths)

    def stage_code_dependencies(self, stage):
        from . import analysis
        deps = [Path(backend.__file__).resolve(), Path(analysis.__file__).resolve()]
        if stage.name == "analysis":
            deps.extend(SourceAnalysisPipeline.stage_code_dependencies(None, stage))
        if self.input.event_file:
            module = backend._backend()
            deps.extend(sorted(Path(module.__file__).parent.rglob("*.py")))
        return tuple(deps)

    def _input_path(self, path):
        path = Path(path).expanduser()
        return (path if path.is_absolute() else self.input.resolved_root()/path).resolve()

    def validate_input(self):
        c = self.config
        products = bool(self.input.source_spectra or self.input.lightcurve_products)
        if bool(self.input.event_file) == products or (self.input.event_file and self.input.background_spectra):
            raise ValueError("Supply either an event file or finished spectrum/lightcurve products")
        if self.input.event_file and self.input.met_reference_utc is not None:
            raise ValueError("The event FITS sets the native MET clock")
        if self.input.event_file and not self.event_path.is_file():
            raise FileNotFoundError(f"GECAM event file not found: {self.event_path}")
        if self.workspace == self.input.resolved_root() or any(
            self.workspace in path.parents for path in self.stage_input_dependencies(self.stages[0])
        ):
            raise ValueError("Use an output workspace separate from the input files")
        if not c.detectors or len(set(c.detectors)) != len(c.detectors) or any(
            not isinstance(d, int) or isinstance(d, bool) or not 1 <= d <= 25 for d in c.detectors
        ):
            raise ValueError("detectors must contain unique GRD integers from 1 to 25")
        if c.gain not in {"high", "low"}:
            raise ValueError("gain must be high or low; run gains separately")
        bounds = backend._values(c.time_range, u.s, "time_range")
        step = backend._values(c.time_bin, u.s, "time_bin")
        edges = backend._values(c.energy_edges, u.keV, "energy_edges")
        if bounds.shape != (2,) or bounds[1] <= bounds[0]:
            raise ValueError("time_range must be increasing [start, stop] MET")
        if step.ndim != 0 or step <= 0:
            raise ValueError("time_bin must be a positive scalar")
        if edges.ndim != 1 or len(edges) < 2 or np.any(edges <= 0) or np.any(np.diff(edges) <= 0):
            raise ValueError("energy_edges must be positive and increasing")
        if (c.source_intervals is None) != (c.background_intervals is None):
            raise ValueError("Provide both source_intervals and background_intervals")
        if set(c.responses) - set(c.detectors):
            raise ValueError("responses contains an unselected detector")
        for path in c.responses.values():
            if not Path(path).expanduser().is_file():
                raise FileNotFoundError(path)
        if products:
            if set(self.input.source_spectra) != set(self.input.background_spectra):
                raise ValueError("Finished source/background spectra must have matching detector keys")
            if set(self.input.source_spectra) - set(c.detectors):
                raise ValueError("Finished spectra contain unselected detectors")
            for path in self.stage_input_dependencies(self.stages[0]):
                if not path.is_file():
                    raise FileNotFoundError(path)
            return
        event = backend.open_event(self.event_path)
        start, stop = event.info.obs_met_range
        if bounds[0] < start or bounds[1] > stop:
            raise ValueError("time_range extends outside the event observation")
        if c.source_intervals is not None:
            src = backend._intervals(c.source_intervals, "source_intervals")
            bg = backend._intervals(c.background_intervals, "background_intervals")
            # Upstream adds one second at each end of the extraction window.
            if min(src.min(), bg.min()) - 1 < start or max(src.max(), bg.max()) + 1 > stop:
                raise ValueError("Spectrum intervals plus upstream 1 s padding exceed the observation")

    def execute_stage(self, stage, context):
        c = self.config
        if stage.name == "report":
            result = self.build_result(context)
            result["pipeline_status"] = "completed"
            path = write_json(self.workspace / "gecam_report.json", result)
            return StageResult(outputs={"report": str(path)})
        if stage.name == "analysis":
            return self._analyze(context)
        if not self.input.event_file:
            if stage.name == "inspect":
                return StageResult(data={"input_kind": "finished_products"})
            if stage.name == "lightcurve":
                return StageResult(data={"native_products": False})
            paths = {}
            for detector, path in self.input.source_spectra.items():
                paths[f"GRD{detector}_full"] = str(self._input_path(path))
                paths[f"GRD{detector}_background"] = str(self._input_path(self.input.background_spectra[detector]))
            return StageResult(outputs=paths, data={"enabled": bool(paths)})
        if stage.name == "spectra" and c.source_intervals is None:
            return StageResult(data={"enabled": False})
        event = backend.open_event(self.event_path)
        if stage.name == "inspect":
            info = event.info
            with fits.open(self.event_path) as hdus:
                header = hdus[0].header
                reference, clock_source = _met_reference(header, info.satellite_full_name, allow_mission_epoch=False)
            reference.precision = 9
            return StageResult(data={
                "event_file": str(self.event_path), "satellite": info.satellite_full_name,
                "instrument": info.instrument, "trigger_id": info.trig_id,
                "trigger_met_s": info.trig_met, "observation_met_s": info.obs_met_range,
                "met_reference_utc": reference.isot, "met_clock_source": clock_source,
                "gecamtools_version": backend._backend().get_software_version(),
            })
        output = Path(tempfile.mkdtemp(prefix=f"{stage.name}-", dir=self.workspace))
        paths, diagnostics = {}, {}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for detector in c.detectors:
                key = f"GRD{detector:02d}_{c.gain}"
                if stage.name == "lightcurve":
                    clock = context["inspect"].data
                    lc = backend.extract_lightcurve(
                        event, detector=detector, gain=c.gain, time_range=c.time_range,
                        time_bin=c.time_bin, energy_edges=c.energy_edges,
                        only_recommend=c.only_recommend, correct_dead_time=c.correct_dead_time,
                    )
                    times, counts, errors = lc.get_data(correct_by_dead_time=c.correct_dead_time)
                    _, raw, raw_errors = lc.get_data(correct_by_dead_time=False)
                    if not all(np.all(np.isfinite(a)) for a in (times, counts, errors)):
                        raise ValueError(f"Nonfinite lightcurve values for {key}; inspect dead time and selection")
                    path = output / f"{key}.npz"
                    np.savez(path, time_edges_met_s=times, counts=counts, counts_error=errors,
                             raw_counts=raw, raw_counts_error=raw_errors,
                             energy_edges_keV=lc.energy_bins, channel_edges=lc.channel_bins,
                             dead_time_s=lc.dead_time_on_bins,
                             telescope=clock["satellite"], met_reference_utc=clock["met_reference_utc"],
                             met_clock_source=clock["met_clock_source"],
                             correct_dead_time=c.correct_dead_time)
                    paths[key] = str(path)
                    diagnostics[key] = {"time_bins": len(times) - 1, "raw_counts": float(raw.sum()),
                                        "actual_energy_edges_keV": lc.energy_bins}
                elif stage.name == "spectra":
                    data = backend.extract_spectra(
                        event, detector=detector, gain=c.gain, source_intervals=c.source_intervals,
                        background_intervals=c.background_intervals, time_bin=c.spectrum_time_bin,
                        outdir=output / key, response=c.responses.get(detector),
                        channel_bin=c.channel_bin, fit_order=c.fit_order, only_recommend=c.only_recommend,
                    )
                    if not data:
                        raise ValueError(f"GECAMTools returned no spectra for {key}")
                    for native_key, products in data.items():
                        for _, *spectra in products["src_spec_list"]:
                            for spec in spectra:
                                if spec.exposure <= 0 or not np.isfinite(spec.exposure) or not all(
                                    np.all(np.isfinite(a)) for a in (spec.counts, spec.counts_err)
                                ):
                                    raise ValueError(f"Invalid spectrum values or exposure for {native_key}")
                        for kind, path in zip(("full", "background", "net"), products["spec_path_list"], strict=True):
                            paths[f"{native_key}_{kind}"] = str(Path(path).resolve())
                        diagnostics[native_key] = {"response_supplied": detector in c.responses}
        return StageResult(outputs=paths, data={"enabled": True, "detectors": diagnostics,
                                               "warnings": list(dict.fromkeys(str(w.message) for w in caught))})

    def _analyze(self, context):
        output = Path(tempfile.mkdtemp(prefix="analysis-", dir=self.workspace))
        config = self._analysis_config
        met_reference = self.input.met_reference_utc
        clock_source = "caller-supplied" if met_reference is not None else None
        if self.input.event_file:
            inspect = context["inspect"].data
            telescope = inspect["satellite"]
            met_reference = inspect["met_reference_utc"]
            clock_source = inspect["met_clock_source"]
            reference = Time(met_reference, scale="utc") + self.config.time_range[0]
            reference.precision = 9
            reference_utc = reference.isot
            curves = gecam_lightcurve_products(
                context["lightcurve"].outputs, outdir=output / "lightcurves",
                reference_utc=reference_utc, telescope=telescope, target_id=self.input.target_id,
            )
            spectra, backgrounds = {}, {}
            for key, path in context["spectra"].outputs.items():
                if key.endswith("_full"):
                    detector = int(key[2:4])  # GECAMTools keys such as bg17H_full
                    spectra[detector] = path
                    backgrounds[detector] = context["spectra"].outputs[key[:-5] + "_background"]
        else:
            curves = tuple(replace(item, path=self._input_path(item.path)) for item in self.input.lightcurve_products)
            spectra = {d: self._input_path(p) for d, p in self.input.source_spectra.items()}
            backgrounds = {d: self._input_path(p) for d, p in self.input.background_spectra.items()}
            reference_utc = curves[0].reference_utc if curves else None
        missing = set(spectra) - set(self.config.responses) if config.fit.enabled else set()
        observations = ()
        if spectra and config.fit.enabled and not missing:
            observations = prepare_gecam_spectra(
                spectra, backgrounds, {d: self.config.responses[d] for d in spectra},
                outdir=output / "prepared", energy_range=self.config.energy_edges[[0, -1]],
                target_id=self.input.target_id, gain=self.config.gain, met_reference_utc=met_reference,
            )
            if reference_utc is None:
                reference_utc = observations[0].start_utc

        paths = {f"input_lightcurve_{i}": str(item.path) for i, item in enumerate(curves)}
        for index, item in enumerate(observations):
            paths[f"prepared_{index}_source"] = str(item.spectrum.grouped_pha)
            paths[f"prepared_{index}_background"] = str(item.spectrum.background_pha)
            paths[f"prepared_{index}_response"] = str(item.spectrum.rmf)
        result = {}
        if observations or curves:
            child = SourceAnalysisPipeline(SourceAnalysisInput(
                self.input.target_id, output, output_root=output / "source", spectra=observations,
                lightcurves=curves, reference_utc=reference_utc,
            ), config=replace(config, execution=ExecutionConfig()))
            result = child.run()
            for name, stage_result in result["stages"].items():
                paths.update({f"{name}_{key}": value for key, value in stage_result["outputs"].items()})
        products_path = write_json(output / "products.json", {
            "target_id": self.input.target_id, "reference_utc": reference_utc,
            "spectra": observations, "lightcurves": curves,
            "met_reference_utc": met_reference,
            "met_clock_source": clock_source or ("native FITS clock or A/B mission epoch" if observations else None),
        })
        paths["products"] = str(products_path)
        fit_performed = result.get("stages", {}).get("fit", {}).get("data", {}).get("fit_performed", False)
        data = {"analysis": result, "fit_performed": fit_performed}
        if missing:
            message = f"Missing explicit GECAM responses for detectors {sorted(missing)}"
            return StageResult(PipelineStatus.NEEDS_REVIEW, outputs=paths, data=data, message=message)
        return StageResult(outputs=paths, data=data)

    def build_result(self, context):
        """Return available stages, including partial runs and review diagnostics."""
        return {"target_id": self.input.target_id, "workspace": str(self.workspace),
                "pipeline_status": self.status().status.value,
                "stages": {name: {"outputs": r.outputs, "data": r.data, "status": r.status.value, "message": r.message} for name, r in context.items()},
                "config": self.stage_config_dependencies(self.stages[0]),
                "science_status": "requires_review", "fit_performed": context.get("analysis", StageResult()).data.get("fit_performed", False)}
