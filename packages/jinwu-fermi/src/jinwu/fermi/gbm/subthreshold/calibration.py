"""Empirical, configuration-bound calibration of clustered search candidates."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import json
import numpy as np
import astropy.units as u
from scipy.stats import chi2

from jinwu.core.time import Time
from jinwu.core.products import write_json
from ..pipeline import _merge_intervals
from .models import seconds, fingerprint


def estimate_candidate_far(score, background_scores, livetime, *, search_time=None, confidence=0.95):
    """Estimate exceedance FAR with Poisson count uncertainty.

    Scores are dimensionless; livetime and search_time are time Quantities.
    Returns rates in Hz and probabilities, with a one-sided upper bound for
    zero exceedances. FAP = 1-exp(-FAR*T) assumes a stationary Poisson process
    of clustered events. This assumption is separate from search trials.
    """
    total = float(seconds(livetime))
    values = np.asarray(background_scores, float)
    if total <= 0 or values.ndim != 1 or not np.all(np.isfinite(values)) or not np.isfinite(score):
        raise ValueError("finite scores and positive calibration livetime required")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    count = int(np.count_nonzero(values >= score))
    alpha = 1 - confidence
    lower = 0. if count == 0 else 0.5 * chi2.ppf(alpha / 2, 2 * count) / total
    upper = (-np.log(alpha) / total if count == 0 else
             0.5 * chi2.ppf(1 - alpha / 2, 2 * (count + 1)) / total)
    far = count / total if count else None
    duration = None if search_time is None else float(seconds(search_time))
    if duration is not None and duration <= 0:
        raise ValueError("search_time must be positive")
    return {"exceedances": count, "background_livetime_s": total,
            "far_hz": far, "far_lower_hz": float(lower), "far_upper_hz": float(upper),
            "confidence": confidence, "kind": "estimate" if count else "upper_limit",
            "fap": None if far is None or duration is None else float(-np.expm1(-far * duration)),
            "fap_upper": None if duration is None else float(-np.expm1(-upper * duration)),
            "fap_assumption": "stationary Poisson process of clustered events"}


def calibration_from_searches(reports, *, output=None):
    """Build a calibration from complete, quality-qualified off-source reports.

    Duplicate reports are counted once; partially overlapping search domains
    are rejected rather than double-counting candidate events. Returns a JSON
    compatible calibration with union livetime and both score distributions.
    """
    unique, contract, intervals, scores, provenance = set(), None, [], {"loglr": [], "prior_loglr": []}, []
    for report in reports:
        payload = json.loads(Path(report).read_text()) if isinstance(report, (str, Path)) else report
        if payload.get("search_complete") is not True or payload.get("quality_passed") is not True:
            raise ValueError("calibration requires complete, quality-qualified searches")
        if contract is None:
            contract = payload["calibration_contract"]
        elif contract != payload["calibration_contract"]:
            raise ValueError("calibration search configurations do not match")
        domain = payload["effective_intervals_met"]
        identifier = fingerprint({"contract": contract, "domain": domain, "trigger_met": payload["trigger_met"]})
        if identifier in unique:
            continue
        for a, b in domain:
            if any(min(b, d) > max(a, c) for c, d in intervals):
                raise ValueError("overlapping off-source search domains must be supplied only once")
        unique.add(identifier)
        intervals.extend(domain)
        for key in scores:
            scores[key].extend(payload["calibration_scores"][key])
        provenance.append({"report": str(report) if isinstance(report, (str, Path)) else None,
                           "trigger_met": payload["trigger_met"], "data": payload.get("data_provenance")})
    intervals = _merge_intervals(intervals)
    livetime = sum(b - a for a, b in intervals)
    if contract is None or livetime <= 0:
        raise ValueError("no positive off-source search livetime")
    payload = {"schema_version": 1, "contract": contract, "livetime_s": livetime,
               "intervals_met": intervals, "scores": {k: sorted(v) for k, v in scores.items()},
               "sources": provenance, "status": "empirical_config_conditional"}
    if output is not None:
        write_json(output, payload)
    return payload


def calibrate_targeted_search(input_data, background_times, *, config=None, output=None):
    """Run the same targeted search at explicit, non-overlapping off-source times.

    Time values must be scalar Time instances or UTC strings. The original
    on-source background context is excluded. Outputs live in separate child
    workspaces; the target's calibration file is not applied to these runs.
    No automatic selection of a scientifically representative sample occurs.
    """
    from .models import GBMTargetedSearchConfig
    from .pipeline import run_targeted_search
    config = config or GBMTargetedSearchConfig()
    t0 = float(input_data.trigger_time.to_value("fermi"))
    context = max(float(seconds(config.background_context)), float(seconds(config.background_window)) + 30)
    lo, hi = seconds(config.search_interval)
    times = sorted({float((Time(t, scale="utc") if isinstance(t, str) else Time(t)).to_value("fermi"))
                    for t in background_times})
    for t in times:
        if t + hi + context > t0 + lo - context and t + lo - context < t0 + hi + context:
            raise ValueError("off-source data context overlaps the on-source data context")
    for previous, current in zip(times, times[1:]):
        if previous + hi > current + lo:
            raise ValueError("off-source search intervals overlap")
    root = input_data.resolved_output_root() / "calibration_runs"
    reports = []
    for index, met in enumerate(times):
        job = replace(input_data, target_id=f"{input_data.target_id}_off{index}",
                      trigger_time=Time(met, format="fermi"), output_root=root / f"off_{met:.6f}", calibration=None)
        execution = replace(config.execution, workspace=None)
        result = run_targeted_search(job, config=replace(config, execution=execution))
        if result.status != "completed":
            raise ValueError(f"off-source search at MET {met} requires review: {result.products.get('report')}")
        reports.append(result.products["report"])
    return calibration_from_searches(reports, output=output or root.parent / "calibration.json")
