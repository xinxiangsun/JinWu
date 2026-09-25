"""Resumable single-event GW coverage workflow."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import inspect
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any

from astropy.table import Table

from jinwu.core.config import ExecutionConfig
from jinwu.core.pipeline import InstrumentPipeline, PipelineInput, PipelineStage, StageResult, register_pipeline

from .alert import fetch_skymap_url, fetch_superevent, read_notice, skymap_from_notice
from .coverage import gbm_coverage_for_skymap
from .layers import footprint_from_circle, footprint_from_polygon, load_layers
from .models import CoverageResult, GWEvent, GWRunResult, SkyFootprint, scalar_time
from .plot import plot_allsky, plot_gbm_diagnostic
from .skymap import credible_region_stats, load_skymap, refined_probability


@dataclass(frozen=True, slots=True)
class GWPipelineInput(PipelineInput):
    """CLI-level input for one GW sky-coverage report."""

    notice: Path | str | None = None
    event: str | None = None
    notice_version: str | None = None
    skymap: Path | str | None = None
    time: str | None = None
    at: str | None = None
    layers: Path | str | None = None
    gbm_cache: Path | str | None = None
    gbm_mode: str = "auto"
    no_download: bool = False
    formats: tuple[str, ...] = ("png", "pdf")


@dataclass(frozen=True, slots=True)
class GWConfig:
    """Configuration kept intentionally independent of instrument configs."""

    name: str = "GW"
    pipeline: str = "gw"
    moc_order: int = 10
    max_moc_order: int = 13
    probability_tolerance: float = 1e-3
    execution: ExecutionConfig = field(default_factory=lambda: ExecutionConfig(resume=False))


@dataclass(slots=True)
class _GWContext:
    event: GWEvent | None = None
    skymap_path: Path | None = None
    skymap: Any = None
    layers: list[SkyFootprint] = field(default_factory=list)
    coverages: list[CoverageResult] = field(default_factory=list)
    probabilities: dict[str, float | None] = field(default_factory=dict)
    credible_regions: dict[str, Any] = field(default_factory=dict)
    credible_mocs: dict[str, Any] = field(default_factory=dict)
    moc_refinement: dict[str, Any] = field(default_factory=dict)
    output_paths: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)


@register_pipeline("gw")
class GWPipeline(InstrumentPipeline[GWPipelineInput, GWRunResult]):
    """Single-event workflow using the shared jinwu pipeline machinery."""

    stages = (
        PipelineStage("load_alert"),
        PipelineStage("load_skymap", ("load_alert",)),
        PipelineStage("gbm_coverage", ("load_skymap",)),
        PipelineStage("crossmatch", ("gbm_coverage",)),
        PipelineStage("plot", ("crossmatch",)),
        PipelineStage("report", ("plot",)),
    )

    def __init__(self, input_data: GWPipelineInput, *, config: GWConfig | None = None):
        super().__init__(input_data, config=config or GWConfig())
        self._context = _GWContext()

    def validate_input(self) -> None:
        selected = sum(value is not None for value in (self.input.notice, self.input.event, self.input.skymap))
        if selected != 1:
            raise ValueError("provide exactly one of notice, event, or skymap")
        if self.input.gbm_mode not in {"auto", "observed", "predicted", "none"}:
            raise ValueError("gbm_mode must be auto, observed, predicted, or none")

    def stage_code_dependencies(self, stage: PipelineStage) -> tuple[Path, ...]:
        """Hash the module files behind each stage so code edits invalidate caches.

        The mapping must stay a superset of each stage's real imports: the
        publish wheel-gate and ``test/test_stage_code_deps.py`` rely on these
        paths resolving to real installed files (AUD-01).
        """
        module_dir = Path(__file__).resolve().parent
        files: dict[str, tuple[str, ...]] = {
            "load_alert": ("pipeline.py", "alert.py", "models.py"),
            "load_skymap": ("skymap.py", "models.py"),
            "gbm_coverage": ("coverage.py", "skymap.py", "models.py"),
            "crossmatch": ("layers.py", "skymap.py", "models.py"),
            "plot": ("plot.py", "models.py"),
            "report": ("pipeline.py", "models.py"),
        }
        dependencies = [module_dir / name for name in files.get(stage.name, ())]
        if stage.name == "gbm_coverage":
            # GBM coverage delegates attitude selection and geometry to the
            # Fermi package; cache manifests must therefore track the actual
            # imported implementation instead of only the GW wrapper.
            from jinwu.fermi.gbm import find_gbm_poshist
            from jinwu.fermi.gbm.poshist import read_gbm_geometry

            for function in (find_gbm_poshist, read_gbm_geometry):
                source = inspect.getsourcefile(function)
                if source is not None:
                    dependencies.append(Path(source).resolve())
        return tuple(dependencies)

    def stage_input_dependencies(self, stage: PipelineStage) -> tuple[Path, ...]:
        """Fingerprint local notice/maps/layers and the GBM cache directory.

        The shared pipeline then invalidates a resumed stage when a newly
        published alert file, supplied layer, or POSHIST cache changes.  URL
        inputs remain part of the run-wide input fingerprint and are fetched
        again when the CLI is run with its default ``resume=False`` policy.
        """
        values: list[Path] = []
        if stage.name in {"load_alert", "load_skymap"}:
            for value in (self.input.notice, self.input.skymap):
                if value is None:
                    continue
                text = str(value)
                if text.lower().startswith(("http://", "https://")):
                    continue
                values.append(Path(text).expanduser().resolve())
        if stage.name in {"crossmatch", "plot", "report"} and self.input.layers is not None:
            layer_path = Path(self.input.layers).expanduser().resolve()
            values.append(layer_path)
            if layer_path.is_file():
                try:
                    payload = json.loads(layer_path.read_text(encoding="utf-8"))
                    records = payload.get("layers", payload) if isinstance(payload, dict) else payload
                    if isinstance(records, list):
                        for record in records:
                            if not isinstance(record, dict):
                                continue
                            value = (
                                record.get("moc")
                                or record.get("path")
                                or record.get("moc_path")
                                or record.get("skymap")
                                or record.get("probability_map")
                            )
                            if value:
                                source = Path(str(value))
                                values.append(source if source.is_absolute() else layer_path.parent / source)
                except (OSError, json.JSONDecodeError, TypeError):
                    pass
        if stage.name == "gbm_coverage" and self.input.gbm_cache is not None:
            cache_path = Path(self.input.gbm_cache).expanduser().resolve()
            values.append(cache_path)
            # Directory mtimes do not change when an existing POSHIST is
            # replaced in place.  Include the concrete attitude products as
            # file dependencies as well, so a newly published/reprocessed
            # file invalidates a resumed GBM stage deterministically.
            if cache_path.is_dir():
                values.extend(sorted(cache_path.rglob("glg_poshist_all_*.fit*")))
        return tuple(values)

    def execute_stage(self, stage: PipelineStage, context: dict[str, StageResult]) -> StageResult:
        self._hydrate_context(context, stage.name)
        method = getattr(self, f"_stage_{stage.name}")
        return method()

    def _hydrate_context(self, context: dict[str, StageResult], next_stage: str | None = None) -> None:
        """Rebuild lightweight in-memory objects when a stage is resumed."""
        alert_result = context.get("load_alert")
        if self._context.event is None and alert_result is not None:
            event_data = dict(alert_result.data.get("event", {}))
            event_time = event_data.get("event_time_utc")
            self._context.event = GWEvent(
                event_data.get("superevent_id", self.input.target_id),
                scalar_time(event_time) if event_time is not None else None,
                alert_type=event_data.get("alert_type", "UNKNOWN"),
                status=event_data.get("status", "active"),
                notice_version=event_data.get("notice_version"),
                skymap_source=event_data.get("skymap_source"),
                notice_source=event_data.get("notice_source"),
                metadata=event_data.get("metadata", {}),
            )
            value = alert_result.data.get("skymap_path")
            self._context.skymap_path = Path(value).expanduser().resolve() if value else None
            self._context.provenance.update(alert_result.data.get("provenance", {}))
        if self._context.skymap is None and self._context.skymap_path is not None:
            self._context.skymap = load_skymap(self._context.skymap_path)
        if self._context.skymap is not None and "skymap" not in self._context.provenance:
            self._context.provenance["skymap"] = {
                "source": str(self._context.skymap_path) if self._context.skymap_path else self._context.skymap.source,
                "ordering": self._context.skymap.ordering,
                "pixel_count": int(len(self._context.skymap.pixel_probability)),
                "max_level": self._context.skymap.max_level,
                "raw_total_probability": self._context.skymap.raw_total_probability,
                "normalized_total_probability": self._context.skymap.total_probability,
            }
        if self._context.event is not None and self._context.event.event_time is not None:
            when = scalar_time(self.input.at or self._context.event.event_time)
            if self.input.layers is not None and not self._context.layers:
                self._context.layers = [
                    layer
                    for layer in load_layers(self.input.layers, max_depth=self.config.moc_order)
                    if layer.applies_at(when)
                ]
        if (
            (self._context.skymap is not None or (self._context.event is not None and self._context.event.status == "retracted"))
            and not self._context.coverages
            and next_stage != "gbm_coverage"
            and "gbm_coverage" in context
        ):
            when = scalar_time(self.input.at or self._context.event.event_time) if self._context.event else None
            if when is not None:
                if self._context.skymap is None:
                    coverage = CoverageResult("GBM", "unknown", None, target_time=when, reasons=("skymap_unavailable",))
                elif self.input.gbm_mode == "none":
                    coverage = CoverageResult("GBM", "unknown", None, target_time=when, reasons=("disabled",))
                elif self.input.gbm_cache is None:
                    coverage = CoverageResult("GBM", "unknown", None, target_time=when, reasons=("gbm_cache_not_configured",))
                else:
                    coverage = gbm_coverage_for_skymap(
                        self._context.skymap,
                        when,
                        self.input.gbm_cache,
                        mode=self.input.gbm_mode,
                        download=not self.input.no_download,
                        order_start=self.config.moc_order,
                        order_max=self.config.max_moc_order,
                        tolerance=self.config.probability_tolerance,
                    )
                self._context.coverages = [coverage]
                self._context.probabilities["GBM"] = coverage.probability
                self._context.probabilities["GBM_geometric"] = coverage.geometric_probability
                self._context.probabilities["GBM_state"] = coverage.state_probability
        for stage_name in ("plot", "report"):
            if stage_name in context:
                self._context.output_paths.update(context[stage_name].outputs)
        skymap_result = context.get("load_skymap")
        if skymap_result is not None and not self._context.credible_regions:
            self._context.credible_regions.update(skymap_result.data.get("credible_regions", {}))
        if self._context.skymap is not None and not self._context.credible_mocs:
            self._context.credible_mocs = {
                str(level): credible_region_stats(self._context.skymap, level)["moc"]
                for level in (0.5, 0.9)
            }
        crossmatch_result = context.get("crossmatch")
        if crossmatch_result is not None:
            self._context.probabilities.update(crossmatch_result.data.get("probabilities", {}))
            self._context.moc_refinement.update(crossmatch_result.data.get("moc_refinement", {}))

    def _stage_load_alert(self) -> StageResult:
        workspace = self.workspace / "inputs"
        workspace.mkdir(parents=True, exist_ok=True)
        if self.input.notice is not None:
            event, payload = read_notice(self.input.notice)
            if event.status == "retracted":
                # A withdrawal must never turn an older GraceDB map into the
                # current result.  Keep the notice provenance and emit a
                # status-only report below.
                path, provenance = None, {"notice_only": True}
            else:
                path, provenance = skymap_from_notice(event, payload, workspace)
            if path is None and event.status != "retracted" and event.superevent_id != "local":
                event, path, provenance = fetch_superevent(event.superevent_id, workspace)
        elif self.input.event is not None:
            event, path, provenance = fetch_superevent(
                self.input.event, workspace, notice_version=self.input.notice_version,
            )
        else:
            if self.input.time is None:
                raise ValueError("--time is required with --skymap")
            value = str(self.input.skymap)
            if value.lower().startswith(("http://", "https://")):
                path, provenance = fetch_skymap_url(value, workspace, filename="local_skymap.fits")
            else:
                path = Path(value).expanduser().resolve()
                if not path.is_file():
                    raise FileNotFoundError(path)
                import hashlib

                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                provenance = {"source": str(path), "sha256": digest}
            event = GWEvent("local", scalar_time(self.input.time), alert_type="LOCAL", skymap_source=str(path))
        if event.status == "retracted":
            self._context.warnings.append("alert is retracted; the report is retained for provenance only")
        self._context.event = event
        self._context.skymap_path = Path(path).expanduser().resolve() if path is not None else None
        self._context.provenance.update(provenance)
        return StageResult(data={"event": event.to_dict(), "skymap_path": str(path) if path else None, "provenance": provenance})

    def _stage_load_skymap(self) -> StageResult:
        if self._context.skymap_path is None:
            if self._context.event is not None and self._context.event.status == "retracted":
                return StageResult(data={"source": None, "credible_regions": {}, "status": "retracted_no_skymap"})
            raise FileNotFoundError("no sky map was resolved from the alert")
        self._context.skymap = load_skymap(self._context.skymap_path)
        if self._context.event is not None:
            self._context.event = replace(self._context.event, skymap_source=str(self._context.skymap_path))
        self._context.provenance["skymap"] = {
            "source": str(self._context.skymap_path),
            "ordering": self._context.skymap.ordering,
            "pixel_count": int(len(self._context.skymap.pixel_probability)),
            "max_level": self._context.skymap.max_level,
            "raw_total_probability": self._context.skymap.raw_total_probability,
            "normalized_total_probability": self._context.skymap.total_probability,
        }
        stats_with_moc = {str(level): credible_region_stats(self._context.skymap, level) for level in (0.5, 0.9)}
        self._context.credible_mocs = {key: item["moc"] for key, item in stats_with_moc.items()}
        stats = {key: {name: value for name, value in item.items() if name != "moc"} for key, item in stats_with_moc.items()}
        self._context.credible_regions = stats
        return StageResult(data={"source": str(self._context.skymap_path), "total_probability": self._context.skymap.total_probability, "credible_regions": stats})

    def _stage_gbm_coverage(self) -> StageResult:
        assert self._context.event is not None
        if self._context.event.status == "retracted":
            coverage = CoverageResult(
                "GBM", "not_run", None, reasons=("alert_retracted",),
                metadata={"coverage_basis": "not_executed"},
            )
            self._context.coverages = [coverage]
            self._context.probabilities["GBM"] = None
            self._context.probabilities["GBM_geometric"] = None
            self._context.probabilities["GBM_state"] = None
            return StageResult(data=coverage.to_dict())
        when = scalar_time(self.input.at or self._context.event.event_time)
        if self._context.skymap is None:
            coverage = CoverageResult("GBM", "unknown", None, target_time=when, reasons=("skymap_unavailable",))
        elif self.input.gbm_mode == "none":
            coverage = CoverageResult("GBM", "unknown", None, target_time=when, reasons=("disabled",))
        elif self.input.gbm_cache is None:
            coverage = CoverageResult("GBM", "unknown", None, target_time=when, reasons=("gbm_cache_not_configured",))
        else:
            coverage = gbm_coverage_for_skymap(
                self._context.skymap, when, self.input.gbm_cache,
                mode=self.input.gbm_mode, download=not self.input.no_download,
                order_start=self.config.moc_order,
                order_max=self.config.max_moc_order,
                tolerance=self.config.probability_tolerance,
            )
        self._context.coverages = [coverage]
        self._context.probabilities["GBM"] = coverage.probability
        self._context.probabilities["GBM_geometric"] = coverage.geometric_probability
        self._context.probabilities["GBM_state"] = coverage.state_probability
        return StageResult(data=coverage.to_dict())

    def _stage_crossmatch(self) -> StageResult:
        assert self._context.event is not None
        if self._context.skymap is None:
            return StageResult(data={"probabilities": dict(self._context.probabilities), "layers": [], "moc_refinement": {}})
        when = scalar_time(self.input.at or self._context.event.event_time)
        if self.input.layers is not None:
            self._context.layers = [layer for layer in load_layers(self.input.layers, max_depth=self.config.moc_order) if layer.applies_at(when)]
        combined = None
        instrument_unions: dict[str, Any] = {}
        for layer in self._context.layers:
            is_localization = layer.kind.strip().lower() in {"localization", "localisation"}
            if layer.source is None:
                self._context.probabilities[layer.name] = None
                self._context.coverages.append(CoverageResult(
                    layer.name, "display_only", None, layer, source=None, target_time=when,
                    reasons=("missing_source",), metadata=layer.to_dict(),
                ))
                continue
            self._context.probabilities[layer.name] = self._integrate_footprint(layer)
            if is_localization:
                self._context.coverages.append(CoverageResult(
                    layer.name, "localization", None, layer, source=layer.source, target_time=when,
                    reasons=("excluded_from_instrument_coverage",), metadata=layer.to_dict(),
                ))
                continue
            combined = layer.moc if combined is None else combined.union(layer.moc)
            instrument = str(
                layer.metadata.get("instrument")
                or layer.metadata.get("mission")
                or layer.name
                or layer.kind
            ).upper()
            instrument_unions[instrument] = (
                layer.moc
                if instrument not in instrument_unions
                else instrument_unions[instrument].union(layer.moc)
            )
            self._context.coverages.append(CoverageResult(layer.name, "provided", self._context.probabilities[layer.name], layer, source=layer.source, target_time=when, metadata=layer.to_dict()))
        for instrument, moc in instrument_unions.items():
            self._context.probabilities[instrument] = self._integrate_footprint(
                SkyFootprint(f"{instrument} union", moc, kind="instrument_union")
            )
        gbm = next((item for item in self._context.coverages if item.instrument == "GBM"), None)
        if gbm is not None and gbm.footprint is not None:
            if combined is not None:
                self._context.probabilities["GBM_and_layers"] = self._integrate_footprint(
                    SkyFootprint(
                        "GBM and supplied layers",
                        gbm.footprint.moc.intersection(combined),
                        kind="intersection",
                    )
                )
            for instrument, moc in instrument_unions.items():
                self._context.probabilities[f"GBM_and_{instrument}"] = self._integrate_footprint(
                    SkyFootprint(
                        f"GBM and {instrument}",
                        gbm.footprint.moc.intersection(moc),
                        kind="intersection",
                    )
                )
        return StageResult(data={"probabilities": dict(self._context.probabilities), "layers": [layer.to_dict() for layer in self._context.layers], "moc_refinement": dict(self._context.moc_refinement)})

    def _integrate_footprint(self, footprint: SkyFootprint) -> float:
        native_order = int(getattr(footprint.moc, "max_order", self.config.max_moc_order))

        def build(order: int) -> SkyFootprint:
            record = dict(footprint.metadata)
            if record.get("shape") == "circle" or any(
                key in record for key in ("radius_deg", "error_radius_deg", "error_radius", "radius")
            ):
                ra_value = record.get("ra_deg", record.get("ra"))
                dec_value = record.get("dec_deg", record.get("dec"))
                radius_value = record.get(
                    "radius_deg",
                    record.get("error_radius_deg", record.get("error_radius", record.get("radius", 0.0))),
                )
                return footprint_from_circle(
                    footprint.name,
                    ra_deg=float(ra_value),
                    dec_deg=float(dec_value),
                    radius=float(radius_value),
                    source=footprint.source,
                    valid_at=footprint.valid_at.utc.isot if footprint.valid_at is not None else None,
                    kind=footprint.kind,
                    max_depth=order,
                    metadata=record,
                )
            ra_value = record.get("ra_deg", record.get("ra"))
            dec_value = record.get("dec_deg", record.get("dec"))
            if isinstance(ra_value, list) and isinstance(dec_value, list):
                return footprint_from_polygon(
                    footprint.name,
                    ra_deg=ra_value,
                    dec_deg=dec_value,
                    source=footprint.source,
                    valid_at=footprint.valid_at.utc.isot if footprint.valid_at is not None else None,
                    kind=footprint.kind,
                    max_depth=order,
                    metadata=record,
                )
            vertices = record.get("vertices") or record.get("polygon")
            if vertices and isinstance(vertices, list) and isinstance(vertices[0], (list, tuple)):
                return footprint_from_polygon(
                    footprint.name,
                    ra_deg=[pair[0] for pair in vertices],
                    dec_deg=[pair[1] for pair in vertices],
                    source=footprint.source,
                    valid_at=footprint.valid_at.utc.isot if footprint.valid_at is not None else None,
                    kind=footprint.kind,
                    max_depth=order,
                    metadata=record,
                )
            moc = footprint.moc
            if order < native_order:
                moc = moc.degrade_to_order(order)
            return SkyFootprint(footprint.name, moc, kind=footprint.kind)

        integral = refined_probability(
            self._context.skymap,
            build,
            order_start=self.config.moc_order,
            order_max=self.config.max_moc_order,
            tolerance=self.config.probability_tolerance,
        )
        self._context.moc_refinement[footprint.name] = integral.to_dict()
        return integral.value

    def _stage_plot(self) -> StageResult:
        assert self._context.event is not None
        if self._context.skymap is None:
            return StageResult(data={"status": "skipped_no_skymap"})
        when = scalar_time(self.input.at or self._context.event.event_time)
        credible_layers = [
            SkyFootprint(f"GW {float(level) * 100:.0f}% credible", moc, kind="credible")
            for level, moc in sorted(self._context.credible_mocs.items(), key=lambda item: float(item[0]))
        ]
        plots = plot_allsky(
            self._context.skymap,
            output=self.workspace / "sky_map",
            event_label=self._context.event.superevent_id,
            footprints=tuple(credible_layers) + tuple(self._context.layers),
            coverages=self._context.coverages,
            title_suffix=f" @ {when.utc.isot}",
        )
        gbm = next((item for item in self._context.coverages if item.instrument == "GBM"), None)
        if gbm is not None:
            plots.update(plot_gbm_diagnostic(self._context.skymap, gbm, output=self.workspace / "gbm_diagnostic", event_label=self._context.event.superevent_id))
        self._context.output_paths.update(plots)
        return StageResult(outputs=plots, data=plots)

    def _stage_report(self) -> StageResult:
        assert self._context.event is not None
        run_dir = self._run_directory(self._context.event)
        run_dir.mkdir(parents=True, exist_ok=True)
        # Plot stages share the manifest workspace.  Snapshot their immutable
        # artifacts into this source-fingerprinted run directory before the
        # report points at them, so a later alert cannot overwrite history.
        for name, value in tuple(self._context.output_paths.items()):
            source = Path(value)
            if source.is_file() and source.parent != run_dir:
                target = run_dir / source.name
                shutil.copy2(source, target)
                self._context.output_paths[name] = str(target)
        json_path = run_dir / "coverage.json"
        rows = [item.to_dict() for item in self._context.coverages]
        if rows:
            table = Table(rows=rows)
            ecsv_path = run_dir / "coverage.ecsv"
            table.write(ecsv_path, format="ascii.ecsv", overwrite=True)
            self._context.output_paths["coverage_ecsv"] = str(ecsv_path)
        self._context.output_paths["coverage_json"] = str(json_path)
        report = GWRunResult(
            self._context.event,
            str(self._context.skymap_path) if self._context.skymap_path else None,
            self._context.output_paths,
            tuple(self._context.coverages),
            self._context.probabilities,
            warnings=tuple(self._context.warnings),
            provenance=self._context.provenance,
            credible_regions=self._context.credible_regions,
            moc_refinement=self._context.moc_refinement,
        )
        json_path.write_text(json.dumps(report.to_dict(), indent=2, default=str) + "\n", encoding="utf-8")
        self._archive_run(report)
        # Include every archived artifact, not only the tabular report: when
        # the core pipeline hydrates a completed stage it must replace the
        # temporary plot-workspace paths with immutable run paths as well.
        return StageResult(outputs=dict(self._context.output_paths), data=report.to_dict())

    def _run_directory(self, event: GWEvent) -> Path:
        """Return the immutable artifact directory for this resolved input set."""
        provenance = dict(self._context.provenance)
        # Retrieval time is audit metadata, rather than a scientific input.
        # Excluding it lets an unchanged alert/map source reuse its archive.
        provenance.pop("notice_downloaded_at_utc", None)
        payload = {
            "event": event.to_dict(),
            "provenance": provenance,
            "display_time": self.input.at,
            "layers": str(self.input.layers) if self.input.layers else None,
            "gbm_mode": self.input.gbm_mode,
            "config": self.config,
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:16]
        return self.workspace / event.superevent_id / "runs" / digest

    @staticmethod
    def _atomic_json(path: Path, payload: Any) -> None:
        """Atomically replace a small JSON pointer or history file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
            json.dump(payload, handle, indent=2, default=str)
            handle.write("\n")
            temporary = Path(handle.name)
        temporary.replace(path)

    def _archive_run(self, report: GWRunResult) -> None:
        """Keep one history entry per alert version and maintain ``current.json``.

        规范：警报更新保留版本；撤回警报生成撤回状态，不继续把旧图作为
        当前结果。每次运行的完整结果本身已随 ``coverage.json`` 归档，这里
        追加一条轻量历史记录；撤回运行会移除 ``current.json`` 指针并写入
        撤回标记，而历史条目永远保留。
        """
        event = report.event
        entry = {
            "superevent_id": event.superevent_id,
            "alert_type": event.alert_type,
            "status": event.status,
            "notice_version": event.notice_version,
            "event_time_utc": event.event_time.utc.isot if event.event_time is not None else None,
            "coverage_json": report.output_paths.get("coverage_json"),
            "probabilities": dict(report.probabilities),
        }
        event_root = self.workspace / event.superevent_id
        history_path = event_root / "runs_history.json"
        try:
            history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.is_file() else []
        except (OSError, json.JSONDecodeError):
            history = []
        if not isinstance(history, list):
            history = []
        # Re-running an identical source fingerprint reuses the same archive
        # entry instead of duplicating history on resume.
        if not any(item.get("coverage_json") == entry["coverage_json"] for item in history if isinstance(item, dict)):
            history.append(entry)
        self._atomic_json(history_path, history)
        pointer = event_root / "current.json"
        if event.status == "retracted":
            # A retraction must not leave a previous alert looking current.
            pointer.unlink(missing_ok=True)
            self._atomic_json(event_root / "RETRACTED.json", entry)
        else:
            (event_root / "RETRACTED.json").unlink(missing_ok=True)
            self._atomic_json(pointer, entry)

    def build_result(self, context: dict[str, StageResult]) -> GWRunResult:
        self._hydrate_context(context)
        assert self._context.event is not None
        return GWRunResult(
            self._context.event,
            str(self._context.skymap_path) if self._context.skymap_path else None,
            self._context.output_paths,
            tuple(self._context.coverages),
            self._context.probabilities,
            warnings=tuple(self._context.warnings),
            provenance=self._context.provenance,
            credible_regions=self._context.credible_regions,
            moc_refinement=self._context.moc_refinement,
        )


def run_gw_pipeline(input_data: GWPipelineInput, *, config: GWConfig | None = None) -> GWRunResult:
    """Run a report with resume disabled by default for fresh source checks."""
    return GWPipeline(input_data, config=config).run(resume=False)
