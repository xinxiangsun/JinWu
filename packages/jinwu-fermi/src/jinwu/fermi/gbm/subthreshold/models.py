"""Explicit, unit-aware contracts for the GBM targeted search."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import json
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import QTable

from jinwu.core.time import Time
from jinwu.core.config import ExecutionConfig
from jinwu.core.pipeline import PipelineInput

UPSTREAM_COMMIT = "1bc1e913f97fd7195a7e297f8d6032a5c7758894"
METHOD_VERSION = "jinwu-gts-1"
DETECTORS = tuple(f"n{i:x}" for i in range(12)) + ("b0", "b1")
TEMPLATES = ("hard", "normal", "soft")
CHANNEL_EDGES = {"nai": (0, 8, 20, 33, 51, 85, 106, 127, 128),
                 "bgo": (0, 8, 21, 40, 65, 90, 112, 124, 128)}


def seconds(value: u.Quantity) -> np.ndarray:
    """Convert an explicit time Quantity to finite seconds; reject bare values."""
    if not isinstance(value, u.Quantity):
        raise TypeError("time intervals and durations require astropy.units.Quantity")
    result = np.asarray(value.to_value(u.s), dtype=float)
    if not np.all(np.isfinite(result)):
        raise ValueError("time values must be finite")
    return result


def fingerprint(payload: dict) -> str:
    """Hash an explicitly serialized search contract, never process state."""
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class GBMTargetedSearchConfig:
    """Search settings; times are Quantities and scores are dimensionless.

    The score floor controls candidate export, not detection significance.
    References: Goldstein et al. 2019, arXiv:1903.12597 and UPSTREAM.json.
    """
    name: str = "GBM targeted search"
    pipeline: str = "fermi.gbm.subthreshold"
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    search_interval: u.Quantity = field(default_factory=lambda: [-30., 30.] * u.s)
    min_duration: u.Quantity = field(default_factory=lambda: 0.064 * u.s)
    max_duration: u.Quantity = field(default_factory=lambda: 8.192 * u.s)
    min_step: u.Quantity = field(default_factory=lambda: 0.064 * u.s)
    num_steps: int = 8
    background_context: u.Quantity = field(default_factory=lambda: 500 * u.s)
    background_window: u.Quantity = field(default_factory=lambda: 125 * u.s)
    detectors: tuple[str, ...] = DETECTORS
    min_score: float = 5.
    overlap_factor: float = 0.2
    response_tolerance: float = 0.01

    def __post_init__(self):
        interval = seconds(self.search_interval)
        if interval.shape != (2,) or interval[1] <= interval[0]:
            raise ValueError("search_interval requires two ordered time offsets")
        for name in ("min_duration", "max_duration", "min_step", "background_context", "background_window"):
            value = seconds(getattr(self, name))
            if value.shape != () or value <= 0:
                raise ValueError(f"{name} must be a positive scalar duration")
        lo, hi, step = (float(seconds(getattr(self, n))) for n in ("min_duration", "max_duration", "min_step"))
        if hi < lo or interval[1] - interval[0] < hi:
            raise ValueError("search interval must contain max_duration >= min_duration")
        for duration in (lo, hi):
            power = np.log2(duration / 0.064)
            if power < 0 or not np.isclose(power, round(power), atol=1e-10):
                raise ValueError("durations must be 0.064 s times a nonnegative power of two")
        if step > lo or not np.isclose(lo / step, round(lo / step)):
            raise ValueError("min_step must divide min_duration")
        if not isinstance(self.num_steps, int) or self.num_steps < 1:
            raise ValueError("num_steps must be a positive integer")
        if not self.detectors or len(set(self.detectors)) != len(self.detectors) or set(self.detectors) - set(DETECTORS):
            raise ValueError("detectors must be unique GBM detector names")
        if not np.isfinite(self.min_score) or self.min_score < 0:
            raise ValueError("min_score must be finite and nonnegative")
        if not 0 <= self.overlap_factor < 1 or not 0 < self.response_tolerance < 1:
            raise ValueError("invalid overlap_factor or response_tolerance")

    def to_dict(self) -> dict:
        """JSON-ready scientific settings in declared seconds, excluding execution."""
        return {"method": METHOD_VERSION, "upstream_commit": UPSTREAM_COMMIT,
                "search_interval_s": seconds(self.search_interval).tolist(),
                **{n + "_s": float(seconds(getattr(self, n))) for n in
                   ("min_duration", "max_duration", "min_step", "background_context", "background_window")},
                "num_steps": self.num_steps, "detectors": list(self.detectors),
                "min_score": self.min_score, "overlap_factor": self.overlap_factor,
                "response_tolerance": self.response_tolerance,
                "skygrid_deg": 5., "templates": list(TEMPLATES),
                "channel_edges": {key: list(value) for key, value in CHANNEL_EDGES.items()},
                "background_variance_factor": 1.}


@dataclass(frozen=True, slots=True, kw_only=True)
class GBMTargetedSearchInput(PipelineInput):
    """One external trigger. Paths are read-only except output_root/cache.

    ``trigger_time`` is a scalar Time or UTC ISO string. ``position`` and
    local HEALPix ``skymap`` are mutually exclusive spatial priors.
    """
    trigger_time: Time | str
    template_root: Path | str
    position: SkyCoord | None = None
    skymap: Path | str | None = None
    download: bool = False
    tte_paths: tuple[Path | str, ...] = ()
    poshist_paths: tuple[Path | str, ...] = ()
    calibration: Path | str | None = None

    def __post_init__(self):
        value = self.trigger_time
        time = Time(value, scale="utc") if isinstance(value, str) else Time(value)
        if not time.isscalar or not np.isfinite(time.to_value("fermi")):
            raise ValueError("trigger_time must be a finite scalar Time or UTC string")
        object.__setattr__(self, "trigger_time", time)
        if self.position is not None and self.skymap is not None:
            raise ValueError("position and skymap are mutually exclusive")
        if self.position is not None:
            if not isinstance(self.position, SkyCoord) or not self.position.isscalar:
                raise ValueError("position must be a scalar SkyCoord")
            if not np.all(np.isfinite([self.position.icrs.ra.deg, self.position.icrs.dec.deg])):
                raise ValueError("position must be finite")
        if self.skymap is not None and "://" in str(self.skymap):
            raise ValueError("skymap must be a local file")

    def to_dict(self) -> dict:
        """Serialize the input with explicit absolute time and angular units."""
        return {"target_id": self.target_id, "root": str(self.resolved_root()),
                "output_root": str(self.resolved_output_root()),
                "trigger_met": float(self.trigger_time.to_value("fermi")),
                "template_root": str(Path(self.template_root).expanduser().resolve()),
                "position_deg": None if self.position is None else
                    [float(self.position.icrs.ra.deg), float(self.position.icrs.dec.deg)],
                "skymap": None if self.skymap is None else str(Path(self.skymap).expanduser().resolve()),
                "download": self.download, "tte_paths": list(map(str, self.tte_paths)),
                "poshist_paths": list(map(str, self.poshist_paths)),
                "calibration": None if self.calibration is None else str(self.calibration)}


@dataclass(slots=True)
class GBMTargetedSearchResult:
    """Search products and separate computational/scientific status.

    Candidate columns carry seconds, degrees and photon flux units. FAR/FAP
    are absent until a matching empirical calibration is supplied.
    """
    status: str
    science_status: str
    candidates: QTable
    products: dict[str, str]
    diagnostics: dict
