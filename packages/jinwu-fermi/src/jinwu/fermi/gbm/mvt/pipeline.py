"""Manifest-backed standalone GBM MVT workflow, independent of spectra/XSPEC."""
from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path

import astropy.units as u
import numpy as np

from jinwu.core.pipeline import InstrumentPipeline, PipelineStage, StageResult, PipelineStatus, register_pipeline, _fingerprint
from jinwu.core.products import write_json
from jinwu.core.time import Time
from ..pipeline import _sha256, _utc_hours, fetch_gbm_products_for_interval
from ..tte import latest_products
from .models import GBMMVTConfig, GBMMVTInput, GBMMVTResult, NAI_DETECTORS
from .data import prepare_detector, bin_events, peak_snr, select_detectors
from .engine import compute_mvt, classify_mvt
from .resampling import resample_mvt


@register_pipeline("fermi.gbm.mvt")
class GBMMVTPipeline(InstrumentPipeline):
    """Prepare real counts, refine detector choice, resample, classify and plot."""
    stages = (PipelineStage("preflight"), PipelineStage("prepare", ("preflight",)),
              PipelineStage("measure", ("prepare",)), PipelineStage("report", ("prepare", "measure")))

    def __init__(self, input_data, *, config=None):
        super().__init__(input_data, config=config or GBMMVTConfig())

    def validate_input(self):
        if not isinstance(self.input, GBMMVTInput):
            raise TypeError("GBMMVTInput required")
        if self.workspace == self.input.resolved_root():
            raise ValueError("output_root must differ from the read-only data root")
        if any(self.workspace == Path(p).expanduser().resolve().parent for p in self.input.tte_paths):
            raise ValueError("output workspace cannot be a raw TTE directory")

    def stage_config_dependencies(self, stage):
        return self.config.to_dict()

    def _input_fingerprint(self):
        return _fingerprint({"input": self.input.to_dict(), "config": self.config.to_dict()})

    def stage_code_dependencies(self, stage):
        root = Path(__file__).parent
        import jinwu.core.time as time_module
        return tuple(sorted(root.rglob("*.py"))) + (root / "_vendor/mvt_snr_fit_model.npz",
                    root.parent / "tte.py", root.parent / "pipeline.py", Path(time_module.__file__))

    def _files(self):
        if self.input.tte_paths:
            return latest_products(self.input.tte_paths)
        paths = []
        hours = None
        if self.input.trigger_time is not None:
            bounds = np.vstack([self.input.source_interval.to_value(u.s), self.input.background_intervals.to_value(u.s)])
            hours = {t.utc.datetime.strftime("%y%m%d_%Hz") for t in _utc_hours(
                     self.input.trigger_time + bounds[:, 0].min() * u.s,
                     self.input.trigger_time + bounds[:, 1].max() * u.s)}
        for root in (self.input.resolved_root(), self.workspace / "cache"):
            for path in root.rglob("glg_tte_*.fit*") if root.exists() else ():
                if "_bn" in path.name:
                    if self.input.trigger_id is None or f"_{self.input.trigger_id}_" in path.name:
                        paths.append(path)
                elif hours is not None and any(f"_{h}_" in path.name for h in hours):
                    paths.append(path)
        return latest_products(paths)

    def stage_input_dependencies(self, stage):
        return self._files() if stage.name == "prepare" else ()

    def _save(self, stage, data, *, outputs=None, review=False):
        path = write_json(self.workspace / f"{stage}.json", data)
        return StageResult(PipelineStatus.NEEDS_REVIEW if review else PipelineStatus.COMPLETED,
                           {stage: str(path), **(outputs or {})}, data, data.get("reason") if review else None)

    def execute_stage(self, stage, context):
        return getattr(self, f"_stage_{stage.name}")(context)

    def _stage_preflight(self, context):
        from .engine import load_validation_curve
        load_validation_curve()
        versions = {name: importlib.metadata.version(name) for name in
                    ("numpy", "scipy", "astropy", "astro-gdt", "astro-gdt-fermi")}
        return self._save("preflight", {"settings": self.config.to_dict(), "versions": versions,
                           "license_notice": "_vendor/NOTICE; original nrbutler/mvt licensing remains unresolved before redistribution"})

    def _stage_prepare(self, context):
        from gdt.missions.fermi.gbm.tte import GbmTte
        if self.input.download:
            detectors = NAI_DETECTORS if self.config.detectors == "auto" else self.config.detectors
            if self.input.trigger_id is not None:
                from gdt.missions.fermi.gbm.finders import TriggerFinder
                finder = TriggerFinder(self.input.trigger_id.removeprefix("bn"), protocol="HTTPS")
                finder.get_tte(str(self.workspace / "cache"), dets=list(detectors))
            else:
                windows = np.vstack([self.input.source_interval.to_value(u.s), self.input.background_intervals.to_value(u.s)])
                fetch_gbm_products_for_interval(self.input.trigger_time + windows[:, 0].min() * u.s,
                    self.input.trigger_time + windows[:, 1].max() * u.s,
                    destination=self.workspace / "cache", detectors=detectors, products=("tte",))
        paths = self._files()
        if not paths:
            return self._save("prepare", {"reason": "no matching local TTE; provide files or explicitly enable download"}, review=True)
        missing_paths = [str(p) for p in paths if not p.is_file()]
        if missing_paths:
            return self._save("prepare", {"reason": f"missing explicit TTE files: {missing_paths}"}, review=True)
        trigger_time = self.input.trigger_time
        if trigger_time is None:
            times = []
            for path in paths:
                with GbmTte.open(str(path)) as tte:
                    if tte.trigtime is not None:
                        times.append(float(tte.trigtime))
            if not times or not np.allclose(times, times[0], rtol=0, atol=1e-6):
                return self._save("prepare", {"reason": "continuous or ambiguous TTE requires explicit trigger_time"}, review=True)
            trigger_time = Time(times[0], format="fermi")
        grouped = {d: [p for p in paths if f"_tte_{d}_" in p.name] for d in NAI_DETECTORS}
        if self.config.detectors != "auto":
            missing = [d for d in self.config.detectors if not grouped[d]]
            if missing:
                return self._save("prepare", {"reason": f"missing requested detectors: {missing}"}, review=True)
            grouped = {d: grouped[d] for d in self.config.detectors}
        grouped = {d: ps for d, ps in grouped.items() if ps}
        if not grouped:
            return self._save("prepare", {"reason": "no NaI TTE products in the provided files"}, review=True)
        source = self.input.source_interval.to_value(u.s)
        backgrounds = self.input.background_intervals.to_value(u.s)
        events, diagnostics = {}, {}
        try:
            for det, files in grouped.items():
                events[det], diagnostics[det] = prepare_detector(files, trigger_time, source, backgrounds, self.config)
        except (ValueError, np.linalg.LinAlgError, RuntimeError) as exc:
            return self._save("prepare", {"reason": str(exc), "background": diagnostics}, review=True)
        path = self.workspace / "prepared_events.npz"
        np.savez_compressed(path, **events)
        return self._save("prepare", {"trigger_met": float(trigger_time.to_value("fermi")),
            "trigger_utc": trigger_time.utc.isot, "source_interval_s": source.tolist(),
            "background_intervals_s": backgrounds.tolist(), "background": diagnostics,
            "available_detectors": list(events), "detector_scope": "provided NaI data",
            "data_provenance": {str(p): _sha256(p) for p in paths}}, outputs={"events": str(path)})

    def _measure_resolution(self, events, dets, source, width, iteration, resolution):
        times = np.concatenate([events[d] for d in dets])
        counts, edges = bin_events(times, source, width, max_bins=self.config.max_bins)
        base = compute_mvt(counts, bin_width=width * u.s, config=self.config)
        samples, summary = resample_mvt(counts, width * u.s, config=self.config, stream=(iteration, resolution))
        prefix = f"iteration{iteration}_bw{resolution}"
        lightcurve = self.workspace / f"{prefix}_lightcurve.npz"
        np.savez_compressed(lightcurve, counts=counts, edges_s=edges,
                            source_exposure_s=np.maximum(0., np.minimum(edges[1:], source[1]) - edges[:-1]))
        diagnostics = self.workspace / f"{prefix}_haar.npz"
        np.savez_compressed(diagnostics, **{k: v for k, v in base.diagnostics.items() if isinstance(v, np.ndarray)})
        raw = self.workspace / f"{prefix}_resamples.json"
        write_json(raw, {"random_stream": summary["random_stream"], "samples": [r.to_dict() for r in samples]})
        return {"bin_width_s": width, "detectors": list(dets), "base": base.to_dict(),
                "resampling": summary, "last_bin_source_fraction": float(min(width, source[1] - edges[-2]) / width),
                "products": {"lightcurve": str(lightcurve), "haar": str(diagnostics), "resamples": str(raw)}}

    def _stage_measure(self, context):
        data = context["prepare"].data
        with np.load(context["prepare"].outputs["events"], allow_pickle=False) as archive:
            events = {k: archive[k] for k in archive.files}
        source = data["source_interval_s"]
        backgrounds = data["background"]
        initial = .016 if self.config.t90 is not None and self.config.t90 < 1 * u.s else .064
        history = []
        if self.config.detectors == "auto":
            dets, selection = select_detectors(events, backgrounds, source, initial, max_bins=self.config.max_bins)
            history.append(selection)
        else:
            dets = self.config.detectors
        first_width = float(self.config.bin_widths[0].to_value(u.s))
        initial_record = None
        converged = self.config.detectors != "auto"
        for iteration in range(self.config.max_detector_iterations if not converged else 1):
            initial_record = self._measure_resolution(events, dets, source, first_width, iteration, 0)
            quantiles = initial_record["resampling"]["percentiles_s"]
            if self.config.detectors != "auto" or quantiles is None:
                break
            new_dets, selection = select_detectors(events, backgrounds, source, quantiles[1], max_bins=self.config.max_bins)
            times = np.concatenate([events[d] for d in dets])
            rate = sum(backgrounds[d]["source_average_rate_cps"] for d in dets)
            current_snr = peak_snr(times, source, quantiles[1], rate, max_bins=self.config.max_bins)
            gain = selection["best"]["snr"] / current_snr - 1 if current_snr > 0 else np.inf
            selection.update(previous_detectors=list(dets), fractional_snr_gain=gain)
            history.append(selection)
            if new_dets == dets or gain <= self.config.detector_snr_tolerance:
                converged = True
                break
            if iteration + 1 < self.config.max_detector_iterations:
                dets = new_dets
        rows = [initial_record]
        for j, width in enumerate(self.config.bin_widths[1:].to_value(u.s), 1):
            rows.append(self._measure_resolution(events, dets, source, float(width), iteration, j))
        times = np.concatenate([events[d] for d in dets])
        rate = sum(backgrounds[d]["source_average_rate_cps"] for d in dets)
        for row in rows:
            q = row["resampling"]["percentiles_s"]
            if q is not None:
                snr = peak_snr(times, source, q[1], rate, max_bins=self.config.max_bins)
                row["validation"] = classify_mvt(q[1] * u.s, snr)
        outputs = {f"resolution{j}_{k}": v for j, r in enumerate(rows) for k, v in r["products"].items()}
        return self._save("measure", {"resolutions": rows, "selected_detectors": list(dets),
                "detector_history": history, "detector_converged": converged}, outputs=outputs)

    def _stage_report(self, context):
        from .plots import make_mvt_plots
        prepare, measure = context["prepare"].data, context["measure"].data
        rows = measure["resolutions"]
        qs = [row["resampling"]["percentiles_s"] for row in rows]
        stable = len(qs) >= 2 and qs[-1] is not None and qs[-2] is not None and max(qs[-1][0], qs[-2][0]) <= min(qs[-1][2], qs[-2][2])
        last = rows[-1]
        q = qs[-1]
        quality = all(prepare["background"][d]["quality_passed"] for d in measure["selected_detectors"])
        label = last.get("validation", {}).get("classification", "unavailable")
        interpretation = label if stable else "unresolved_upper_limit" if q is not None else "unavailable"
        payload = {**prepare, **measure, "settings": self.config.to_dict(),
                   "versions": context["preflight"].data["versions"], "frame": "observer",
                   "energy_range_keV": self.config.energy_range.to_value(u.keV).tolist(),
                   "stability_passed": stable, "stability_rule": "overlap of conditional 16-84 percentile intervals at two finest bin widths",
                   "background_quality_passed": quality, "interpretation": interpretation,
                   "upper_bound_s": q[2] if q is not None and interpretation != "robust_measurement" else None,
                   "science_status": "completed" if quality and measure["detector_converged"] and q is not None else "needs_review",
                   "calibration_sha256": _sha256(Path(__file__).parent / "_vendor/mvt_snr_fit_model.npz"),
                   "limits": ["conditional Poisson count resampling does not propagate background-fit uncertainty",
                              "global empirical classification does not establish fastest intrinsic source variability"]}
        outputs = make_mvt_plots(payload, self.workspace)
        return self._save("report", payload, outputs=outputs)

    def build_result(self, context):
        products = {k: v for result in context.values() for k, v in result.outputs.items()}
        states = {k: r.status.value for k, r in context.items()}
        if "report" in context:
            payload = context["report"].data
        else:
            payload = {"science_status": "needs_review", "completed_stages": list(context),
                       "reason": next((r.message for r in context.values() if r.message), "partial run")}
            products["report"] = str(write_json(self.workspace / "partial_report.json", payload))
        return GBMMVTResult(payload["science_status"], payload, products, self.workspace, states)


def run_gbm_mvt(input_data, *, config=None, until=None, resume=None):
    """Run standalone MVT; return quantities in report plus all evidence paths."""
    return GBMMVTPipeline(input_data, config=config).run(until=until, resume=resume)
