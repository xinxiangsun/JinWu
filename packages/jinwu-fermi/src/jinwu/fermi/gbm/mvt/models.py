"""Explicit inputs, units and output states for the frozen GBM Haar method."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re

import astropy.units as u
import numpy as np

from jinwu.core.config import ExecutionConfig
from jinwu.core.pipeline import PipelineInput
from jinwu.core.products import jsonable
from jinwu.core.time import Time

UPSTREAM_COMMIT = "57e7a0fe7e5b27311acb161f622f4a68b2e0e8c7"
METHOD_VERSION = "gbm-mvt-paper-57e7a0f-v1"
NAI_DETECTORS = tuple(f"n{i:x}" for i in range(12))


def time_values(value: u.Quantity, name: str) -> np.ndarray:
    """Convert an explicit finite time Quantity to seconds."""
    if not isinstance(value, u.Quantity):
        raise TypeError(f"{name} must be a time Quantity")
    result = np.asarray(value.to_value(u.s), dtype=float)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class MVTResult:
    """One Haar estimate; an algorithmic limit is never a zero-error detection.

    ``raw_values`` retains the upstream seven-value return in seconds and
    its original flux normalization. ``diagnostics`` retains the intermediate
    scaleogram arrays. The fitted error is the upstream analytic error,
    distinct from the Monte Carlo percentile interval of a GBM run.
    """
    estimator_status: str
    mvt: u.Quantity | None
    error: u.Quantity | None
    raw_values: tuple[float, ...] | None
    diagnostics: dict = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    reason: str | None = None

    def to_dict(self, *, include_diagnostics: bool = False) -> dict:
        value = {
            "method": METHOD_VERSION, "estimator_status": self.estimator_status,
            "mvt_s": None if self.mvt is None else float(self.mvt.to_value(u.s)),
            "analytic_error_s": None if self.error is None else float(self.error.to_value(u.s)),
            "raw_values": self.raw_values, "warnings": self.warnings, "reason": self.reason,
        }
        if include_diagnostics:
            value["diagnostics"] = self.diagnostics
        return jsonable(value)


@dataclass(frozen=True, slots=True)
class GBMMVTConfig:
    """Paper-method settings, with physical quantities at the public boundary.

    The bin grid, 300 resamples and detector refinement follow Bala et al.
    (2026), arXiv:2512.16204. Haar defaults follow the frozen paper wrapper.
    ``background_order=0`` reproduces a constant background; a different
    polynomial order is explicit and recorded, rather than selected silently.
    ``t90`` only selects the paper's initial detector-ranking resolution.
    """
    name: str = "GBM Haar MVT"
    pipeline: str = "fermi.gbm.mvt"
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    energy_range: u.Quantity = field(default_factory=lambda: [8., 900.] * u.keV)
    bin_widths: u.Quantity = field(default_factory=lambda: [1., .1, .01] * u.ms)
    detectors: tuple[str, ...] | str = "auto"
    t90: u.Quantity | None = None
    n_resamples: int = 300
    seed: int = 0
    background_order: int = 0
    background_bin_width: u.Quantity = field(default_factory=lambda: .256 * u.s)
    tau_bg_max: u.Quantity = field(default_factory=lambda: .01 * u.s)
    nrepl: int = 2
    bin_fac: int = 4
    afactor: float = -1.
    snr_threshold: float = 3.
    weight: bool = True
    max_detector_iterations: int = 3
    detector_snr_tolerance: float = .01
    max_bins: int = 10_000_000
    workers: int = 1

    def __post_init__(self):
        bands = np.asarray(self.energy_range.to_value(u.keV), dtype=float)
        if bands.shape != (2,) or not np.isfinite(bands).all() or not 0 < bands[0] < bands[1]:
            raise ValueError("energy_range requires two ordered positive energies")
        widths = time_values(self.bin_widths, "bin_widths")
        if widths.ndim != 1 or not len(widths) or np.any(widths <= 0) or np.any(np.diff(widths) >= 0):
            raise ValueError("bin_widths must be positive and strictly decreasing")
        if self.detectors != "auto":
            detectors = tuple(self.detectors)
            if not detectors or len(set(detectors)) != len(detectors) or set(detectors) - set(NAI_DETECTORS):
                raise ValueError("detectors must be 'auto' or unique NaI names")
            object.__setattr__(self, "detectors", detectors)
        for name in ("background_bin_width", "tau_bg_max"):
            value = time_values(getattr(self, name), name)
            if value.shape != () or value <= 0:
                raise ValueError(f"{name} must be a positive scalar time")
        if self.t90 is not None and (time_values(self.t90, "t90").shape != () or self.t90 <= 0 * u.s):
            raise ValueError("t90 must be a positive scalar time")
        for name in ("n_resamples", "nrepl", "bin_fac", "max_detector_iterations", "max_bins", "workers"):
            if not isinstance(getattr(self, name), int) or isinstance(getattr(self, name), bool) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if self.background_order not in (0, 1, 2):
            raise ValueError("background_order must be 0, 1 or 2")
        if not np.isfinite(self.afactor) or not np.isfinite(self.snr_threshold) or self.snr_threshold <= 0:
            raise ValueError("invalid Haar afactor or snr_threshold")
        if not 0 <= self.detector_snr_tolerance < 1:
            raise ValueError("detector_snr_tolerance must be in [0, 1)")

    def haar_kwargs(self) -> dict:
        """Original keyword settings, using seconds internally."""
        return {"tau_bg_max": float(self.tau_bg_max.to_value(u.s)), "nrepl": self.nrepl,
                "bin_fac": self.bin_fac, "afactor": self.afactor,
                "snr": self.snr_threshold, "weight": self.weight}

    def to_dict(self) -> dict:
        return {"method": METHOD_VERSION, "upstream_commit": UPSTREAM_COMMIT,
                "energy_range_keV": self.energy_range.to_value(u.keV).tolist(),
                "bin_widths_s": self.bin_widths.to_value(u.s).tolist(),
                "detectors": self.detectors, "n_resamples": self.n_resamples, "seed": self.seed,
                "background_order": self.background_order,
                "background_bin_width_s": float(self.background_bin_width.to_value(u.s)),
                "t90_s": None if self.t90 is None else float(self.t90.to_value(u.s)),
                "max_detector_iterations": self.max_detector_iterations,
                "detector_snr_tolerance": self.detector_snr_tolerance, "max_bins": self.max_bins,
                "workers": self.workers,
                "haar": self.haar_kwargs()}


@dataclass(frozen=True, slots=True, kw_only=True)
class GBMMVTInput(PipelineInput):
    """Read-only local TTE or one archive trigger, with explicit time windows.

    ``source_interval`` and ``background_intervals`` are seconds relative to
    ``trigger_time``. Triggered files supply their TRIGTIME when trigger_time
    is omitted; continuous TTE requires an explicit scalar Time or UTC string.
    Downloads go only into the output workspace's cache.
    """
    source_interval: u.Quantity
    background_intervals: u.Quantity
    trigger_time: Time | str | None = None
    trigger_id: str | None = None
    tte_paths: tuple[str | Path, ...] = ()
    download: bool = False

    def to_dict(self):
        return {"target_id": self.target_id, "root": str(self.resolved_root()),
                "output_root": str(self.resolved_output_root()), "trigger_id": self.trigger_id,
                "trigger_met": None if self.trigger_time is None else float(self.trigger_time.to_value("fermi")),
                "source_interval_s": self.source_interval.to_value(u.s).tolist(),
                "background_intervals_s": self.background_intervals.to_value(u.s).tolist(),
                "tte_paths": [str(Path(p).expanduser().resolve()) for p in self.tte_paths],
                "download": self.download}

    def __post_init__(self):
        source = time_values(self.source_interval, "source_interval")
        back = time_values(self.background_intervals, "background_intervals")
        if source.shape != (2,) or source[1] <= source[0]:
            raise ValueError("source_interval requires two ordered time offsets")
        if back.ndim != 2 or back.shape[1] != 2 or not len(back) or np.any(back[:, 1] <= back[:, 0]):
            raise ValueError("background_intervals requires ordered pairs of offsets")
        if np.any(np.minimum(back[:, 1], source[1]) > np.maximum(back[:, 0], source[0])):
            raise ValueError("background intervals must not overlap the source interval")
        ordered = back[np.argsort(back[:, 0])]
        if np.any(ordered[1:, 0] < ordered[:-1, 1]):
            raise ValueError("background intervals must not overlap each other")
        if self.trigger_time is not None:
            time = Time(self.trigger_time, scale="utc") if isinstance(self.trigger_time, str) else Time(self.trigger_time)
            if not time.isscalar or not np.isfinite(time.to_value("fermi")):
                raise ValueError("trigger_time must be a finite scalar Time or UTC string")
            object.__setattr__(self, "trigger_time", time)
        if self.trigger_id is not None:
            trigger_id = str(self.trigger_id).removeprefix("bn")
            if not re.fullmatch(r"\d{9}", trigger_id):
                raise ValueError("trigger_id must be a GBM bnYYMMDDfff identifier")
            object.__setattr__(self, "trigger_id", "bn" + trigger_id)
        if self.download and self.trigger_id is None and self.trigger_time is None:
            raise ValueError("download requires trigger_id or trigger_time")


@dataclass(frozen=True, slots=True)
class GBMMVTResult:
    """A complete or partial manifest-backed GBM MVT run.

    ``summary`` separates the Haar state, the empirical paper classification,
    bin-width stability and instrument/background diagnostics.
    """
    science_status: str
    summary: dict
    products: dict[str, str]
    workspace: Path
    stage_status: dict[str, str]
