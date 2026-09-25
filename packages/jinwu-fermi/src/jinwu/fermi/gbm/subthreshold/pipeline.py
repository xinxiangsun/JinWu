"""Resumable, standalone GBM subthreshold pipeline."""
from __future__ import annotations

import importlib.metadata
import json
import logging
from pathlib import Path
import numpy as np
import astropy.units as u
from astropy.table import QTable

from jinwu.core.pipeline import InstrumentPipeline, PipelineStage, StageResult, PipelineStatus, register_pipeline
from jinwu.core.products import write_json
from ..pipeline import fetch_gbm_products_for_interval, _sha256, _utc_hours, _utc_days, _merge_intervals
from .models import GBMTargetedSearchConfig, GBMTargetedSearchInput, GBMTargetedSearchResult, seconds, fingerprint
from .data import latest_products, prepare_search_data, PreparedSearchData, MeasuredHistory

logger = logging.getLogger(__name__)


def validate_search_templates(root):
    """Validate native GTS response shapes and hash every template file.

    ``root`` must contain direct/, atmo_nai/, atmo_bgo/. Returns a provenance
    mapping keyed by relative path. No resources are downloaded implicitly.
    """
    from ._vendor.utils import SkyGrid
    root = Path(root).expanduser().resolve()
    grid_size = SkyGrid(5.).size
    records = {}
    for kind, detectors in (("nai", 12), ("bgo", 2)):
        paths = [root / "direct" / f"{kind}.npy"] + [
            root / f"atmo_{kind}" / f"atmrates_az{az}_zen130.npy" for az in range(0, 360, 5)]
        if any(not path.is_file() for path in paths):
            raise ValueError(f"missing {kind} atmospheric response templates under {root}")
        for path in paths:
            value = np.load(path, mmap_mode="r", allow_pickle=False)
            if value.ndim != 4 or value.shape[0] < 3 or value.shape[1:] != (grid_size, 8, detectors):
                raise ValueError(f"unexpected template shape {value.shape}: {path}")
            tolerance = 1e-12 * max(1., float(np.max(np.abs(value))))
            if not np.isfinite(value).all() or np.any(value < -tolerance):
                raise ValueError(f"nonfinite or negative response: {path}")
            records[str(path.relative_to(root))] = {"sha256": _sha256(path), "shape": list(value.shape),
                                                  "roundoff_clipped": int(np.count_nonzero(value < 0))}
    return records


@register_pipeline("fermi.gbm.subthreshold")
class GBMTargetedSearchPipeline(InstrumentPipeline):
    """Use core manifests without importing optional search dependencies eagerly."""
    stages = (PipelineStage("preflight"), PipelineStage("data", ("preflight",)),
              PipelineStage("background", ("data",)), PipelineStage("search", ("preflight", "data", "background")),
              PipelineStage("report", ("preflight", "data", "background", "search")))

    def __init__(self, input_data, *, config=None):
        super().__init__(input_data, config=config or GBMTargetedSearchConfig())

    def validate_input(self):
        if not isinstance(self.input, GBMTargetedSearchInput):
            raise TypeError("GBMTargetedSearchInput required")
        if self.input.resolved_root() == self.workspace:
            raise ValueError("output_root must differ from the read-only data root")

    def _input_fingerprint(self):
        return fingerprint({"input": self.input.to_dict(), "config": self.config.to_dict()})

    def stage_config_dependencies(self, stage):
        return self.config.to_dict()

    def stage_code_dependencies(self, stage):
        root = Path(__file__).parent
        import jinwu.core.time as time_module
        return tuple(sorted(root.rglob("*.py"))) + (root.parent / "pipeline.py", root.parent / "poshist.py", Path(time_module.__file__))

    def _bounds(self):
        lo, hi = seconds(self.config.search_interval)
        context = max(float(seconds(self.config.background_context)), float(seconds(self.config.background_window)) + 30)
        return self.input.trigger_time + (min(-context, lo - context) * u.s), self.input.trigger_time + (max(context, hi + context) * u.s)

    def _files(self):
        start, stop = self._bounds()
        hours = {t.utc.datetime.strftime("%y%m%d_%Hz") for t in _utc_hours(start, stop)}
        days = {t.utc.datetime.strftime("%y%m%d") for t in _utc_days(start, stop)}
        ttes, poshist = list(map(Path, self.input.tte_paths)), list(map(Path, self.input.poshist_paths))
        for root in (self.input.resolved_root(), self.workspace / "cache"):
            if not root.exists():
                continue
            if not self.input.tte_paths:
                ttes.extend(p for p in root.rglob("glg_tte_*.fit*") if any(f"_{h}_" in p.name for h in hours))
            if not self.input.poshist_paths:
                poshist.extend(p for p in root.rglob("glg_poshist_all_*.fit*") if any(f"_{d}_" in p.name for d in days))
        return latest_products(ttes), latest_products(poshist)

    def stage_input_dependencies(self, stage):
        files = []
        if stage.name == "preflight":
            root = Path(self.input.template_root).expanduser().resolve()
            files.extend(sorted(root.rglob("*.npy")) if root.exists() else [root])
        if stage.name in {"data", "background", "search"}:
            ttes, poshist = self._files()
            files.extend(ttes + poshist)
        if self.input.skymap is not None:
            files.append(Path(self.input.skymap))
        if self.input.calibration is not None and stage.name == "report":
            files.append(Path(self.input.calibration))
        return tuple(files)

    def _save(self, name, payload, *, review=False, outputs=None):
        path = write_json(self.workspace / f"{name}.json", payload)
        return StageResult(PipelineStatus.NEEDS_REVIEW if review else PipelineStatus.COMPLETED,
                           {name: str(path), **(outputs or {})}, payload,
                           payload.get("reason") if review else None)

    def execute_stage(self, stage, context):
        logger.info("GBM targeted search stage: %s", stage.name)
        return getattr(self, f"_stage_{stage.name}")(context)

    def _stage_preflight(self, context):
        try:
            versions = {name: importlib.metadata.version(name) for name in
                        ("numpy", "scipy", "astro-gdt", "astro-gdt-fermi", "astropy", "healpy", "rich", "PyYAML")}
            templates = validate_search_templates(self.input.template_root)
            return self._save("preflight", {"versions": versions, "templates": templates,
                              "template_source": "https://fermi.gsfc.nasa.gov/ssc/data/analysis/gbm/templates.tar.gz"})
        except (ImportError, OSError, ValueError, importlib.metadata.PackageNotFoundError) as exc:
            return self._save("preflight", {"reason": str(exc)}, review=True)

    def _stage_data(self, context):
        if self.input.download:
            start, stop = self._bounds()
            fetch_gbm_products_for_interval(start, stop, destination=self.workspace / "cache",
                                           detectors=self.config.detectors, products=("tte", "poshist"))
        ttes, poshist = self._files()
        grouped = {det: [str(p) for p in ttes if f"_tte_{det}_" in p.name] for det in self.config.detectors}
        payload = {"tte": grouped, "poshist": list(map(str, poshist)),
                   "provenance": {str(p): _sha256(p) for p in ttes + poshist}}
        missing = [d for d, paths in grouped.items() if not paths]
        if missing or not poshist:
            payload["reason"] = f"Missing TTE detectors {missing} or measured POSHIST"
        return self._save("data", payload, review=bool(missing or not poshist))

    def _stage_background(self, context):
        data = context["data"].data
        prepared, diagnostics = prepare_search_data(data["tte"], self.input.trigger_time, self.config)
        path = self.workspace / "prepared.npz"
        prepared.save(path)
        return self._save("background", {"diagnostics": diagnostics,
                          "quality_passed": all(v["passed"] for v in diagnostics.values())}, outputs={"prepared": str(path)})

    def _stage_search(self, context):
        from .search import run_search_grid, select_search_candidates
        prepared = PreparedSearchData.open(context["background"].outputs["prepared"])
        history = MeasuredHistory(context["data"].data["poshist"])
        table, diagnostics = run_search_grid(prepared, history, self.input.template_root,
                                             self.input.trigger_time, self.config,
                                             position=self.input.position, skymap=self.input.skymap)
        outputs = {}
        candidate_rows = {}
        for score in ("loglr", "prior_loglr"):
            candidates, reasons = select_search_candidates(table, score=score, threshold=self.config.min_score,
                                                           overlap_factor=self.config.overlap_factor)
            table[f"filter_{score}"] = reasons
            path = self.workspace / f"candidates_{score}.ecsv"
            candidates.write(path, format="ascii.ecsv", overwrite=True)
            outputs[f"candidates_{score}"] = str(path)
            candidate_rows[score] = np.asarray(candidates[score], float).tolist()
        path = self.workspace / "full_search.ecsv"
        table.write(path, format="ascii.ecsv", overwrite=True)
        outputs["full_search"] = str(path)
        # Effective time is the union of actual shortest search windows,
        # excluding measured attitude or background gaps. Store absolute MET.
        shortest = table[np.isclose(table["duration"].to_value(u.s), float(seconds(self.config.min_duration)))]
        t0 = float(self.input.trigger_time.to_value("fermi"))
        starts = shortest["tstart"].to_value(u.s)
        durations = shortest["duration"].to_value(u.s)
        intervals = _merge_intervals([(t0 + start, t0 + start + duration) for start, duration, ok in zip(starts, durations, shortest["atmospheric_response"]) if ok])
        diagnostics.update(calibration_scores=candidate_rows, effective_intervals_met=intervals,
                           quality_passed=context["background"].data["quality_passed"] and
                           bool(len(table)) and bool(np.all(table["atmospheric_response"])) and not diagnostics["failed_windows"])
        return self._save("search", diagnostics, outputs=outputs)

    def _contract(self, context):
        import jinwu.core.time as time_module
        return {"settings": self.config.to_dict(), "templates": context["preflight"].data["templates"],
                "time_code_sha256": _sha256(Path(time_module.__file__)),
                "versions": context["preflight"].data["versions"],
                "shared_gbm_pipeline_sha256": _sha256(Path(__file__).parents[1] / "pipeline.py"),
                "position_deg": self.input.to_dict()["position_deg"],
                "skymap_sha256": None if self.input.skymap is None else _sha256(Path(self.input.skymap)),
                "code": {p.name if p.parent.name != "_vendor" else "vendor/" + p.name: _sha256(p)
                         for p in Path(__file__).parent.rglob("*.py")}}

    def _stage_report(self, context):
        from .calibration import estimate_candidate_far
        from .plots import make_search_plots, write_candidate_localizations
        search = context["search"]
        score = "prior_loglr" if self.input.position is not None or self.input.skymap is not None else "loglr"
        candidates = QTable.read(search.outputs[f"candidates_{score}"], format="ascii.ecsv")
        full = QTable.read(search.outputs["full_search"], format="ascii.ecsv")
        payload = {**search.data, "trigger_met": float(self.input.trigger_time.to_value("fermi")),
                   "trigger_utc": self.input.trigger_time.utc.isot,
                   "calibration_contract": self._contract(context), "search_complete": True,
                   "data_provenance": context["data"].data["provenance"], "ranking": score,
                   "science_status": "uncalibrated_candidates", "candidate_count": len(candidates),
                   "far": [None] * len(candidates), "background": context["background"].data}
        intervals = search.data["effective_intervals_met"]
        exposure = sum(b - a for a, b in intervals)
        prepared = PreparedSearchData.open(context["background"].outputs["prepared"])
        outputs = make_search_plots(full, candidates, prepared, self.workspace, ranking=score)
        if len(candidates):
            localizations, checks = write_candidate_localizations(prepared, MeasuredHistory(context["data"].data["poshist"]),
                    self.input.template_root, self.input.trigger_time, self.config, candidates, self.workspace, ranking=score)
            outputs.update(localizations)
            payload["localization_checks"] = checks
            payload["quality_passed"] = payload["quality_passed"] and all(c["response_stable"] for c in checks)
        if self.input.calibration is not None:
            calibration = json.loads(Path(self.input.calibration).read_text())
            if calibration.get("contract") != payload["calibration_contract"]:
                raise ValueError("calibration configuration does not match this search")
            if any(min(b, d) > max(a, c) for a, b in intervals for c, d in calibration["intervals_met"]):
                raise ValueError("calibration overlaps on-source search time")
            if payload["quality_passed"] and exposure > 0:
                payload["far"] = [estimate_candidate_far(float(c[score]), calibration["scores"][score],
                                  calibration["livetime_s"] * u.s, search_time=exposure * u.s) for c in candidates]
                payload["science_status"] = "empirically_calibrated_candidates"
        if not payload["quality_passed"]:
            payload["science_status"] = "needs_review"
        candidate_path = self.workspace / "candidates.ecsv"
        for name in ("far_hz", "far_upper_hz", "fap", "fap_upper"):
            candidates[name] = [np.nan if record is None or record.get(name) is None else record[name] for record in payload["far"]]
        candidates["far_hz"].unit = u.Hz
        candidates["far_upper_hz"].unit = u.Hz
        candidates.write(candidate_path, format="ascii.ecsv", overwrite=True)
        outputs["candidates"] = str(candidate_path)
        # Review is a scientific state; the report is still a completed product.
        return self._save("report", payload, outputs=outputs)

    def build_result(self, context):
        products = {key: value for stage in context.values() for key, value in stage.outputs.items()}
        diagnostics = {key: value.data for key, value in context.items()}
        if "report" in context:
            payload = context["report"].data
            return GBMTargetedSearchResult("completed", payload["science_status"],
                                          QTable.read(products["candidates"], format="ascii.ecsv"), products, diagnostics)
        payload = {"search_complete": False, "quality_passed": False, "science_status": "needs_review",
                   "completed_stages": list(context), "diagnostics": diagnostics}
        products["report"] = str(write_json(self.workspace / "partial_report.json", payload))
        return GBMTargetedSearchResult(self.status().status.value, "needs_review", QTable(), products, diagnostics)


def run_targeted_search(input_data, *, config=None, until=None, resume=None):
    """Run one external-trigger search; return candidates, diagnostics and paths."""
    return GBMTargetedSearchPipeline(input_data, config=config).run(until=until, resume=resume)
