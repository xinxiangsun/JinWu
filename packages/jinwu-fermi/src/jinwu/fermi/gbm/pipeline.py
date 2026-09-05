"""Reusable, headless Fermi/GBM continuous-data analysis primitives.

These functions separate coverage and data-product construction from any GUI
or source-specific workflow.  They use Fermi MET internally, while public
intervals retain :class:`astropy.time.Time` and :class:`astropy.coordinates.SkyCoord`
objects so units and time scales remain explicit.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timedelta, timezone
import csv
import functools
import gzip
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import shutil
import subprocess
from statistics import NormalDist
from typing import Any, ClassVar, Iterable, Literal, Mapping, Sequence
import warnings

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import TimeDelta
import numpy as np

from jinwu.core.config import GBM, GBMAnalysisConfig, UpperLimitConfig
from jinwu.core.pipeline import (
    InstrumentPipeline,
    PipelineInput,
    PipelineStage,
    PipelineStatus,
    StageResult,
    register_pipeline,
)
from jinwu.core.time import Time

__all__ = [
    "GBMFlareInterval",
    "GBMCoverageResult",
    "GBMDetectorSelection",
    "GBMDataManifest",
    "GBMBackgroundSummary",
    "GBMSpectralProducts",
    "check_gbm_coverage",
    "select_gbm_detectors",
    "fetch_gbm_continuous_products",
    "extract_gbm_spectral_products",
    "integrate_background_interval",
    "validate_background_residuals",
    "GBMPipelineInput",
    "GBMPipelineResult",
    "GBMPipeline",
    "background_windows",
    "tte_coverage",
    "single_response",
    "gti_intervals_for_paths",
    "read_ogip_products",
    "powerlaw_energy_flux_per_norm",
    "main",
]

_NAI_DETECTORS = tuple(f"n{index}" for index in range(10)) + ("na", "nb")
_BGO_DETECTORS = ("b0", "b1")

logger = logging.getLogger(__name__)


def _as_scalar_time(value: Time | str) -> Time:
    if isinstance(value, Time):
        time = value
    else:
        text = str(value).strip()
        try:
            time = Time(text, format="isot", scale="utc")
        except ValueError:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
            if parsed.tzinfo is not None:
                parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
            time = Time(parsed, scale="utc")
    if not time.isscalar:
        raise ValueError("GBM flare times must be scalar astropy Time values")
    return time


@dataclass(frozen=True, slots=True)
class GBMFlareInterval:
    """One sky position and UTC flare interval for a GBM coverage query."""

    source_name: str
    skycoord: SkyCoord
    start: Time
    stop: Time
    candidate_index: int | None = None

    def __post_init__(self) -> None:
        if not self.source_name.strip():
            raise ValueError("source_name must not be empty")
        if not isinstance(self.skycoord, SkyCoord) or not self.skycoord.isscalar:
            raise ValueError("skycoord must be one scalar SkyCoord")
        start = _as_scalar_time(self.start)
        stop = _as_scalar_time(self.stop)
        if not start < stop:
            raise ValueError("flare stop must be later than start")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "stop", stop)

    @classmethod
    def from_utc(
        cls,
        *,
        source_name: str,
        ra_deg: float,
        dec_deg: float,
        start_utc: str | Time,
        stop_utc: str | Time,
        candidate_index: int | None = None,
    ) -> "GBMFlareInterval":
        return cls(
            source_name=source_name,
            skycoord=SkyCoord(float(ra_deg) * u.deg, float(dec_deg) * u.deg, frame="icrs"),
            start=_as_scalar_time(start_utc),
            stop=_as_scalar_time(stop_utc),
            candidate_index=candidate_index,
        )

    @property
    def start_met(self) -> float:
        return float(self.start.to_value("fermi"))

    @property
    def stop_met(self) -> float:
        return float(self.stop.to_value("fermi"))

    @property
    def duration_s(self) -> float:
        return float((self.stop - self.start).to_value("s"))


@dataclass(frozen=True, slots=True)
class GBMCoverageResult:
    """Coverage decision for one flare after position-history/GTI intersection."""

    status: Literal["full", "partial", "none", "data_missing"]
    interval: GBMFlareInterval
    coverage_fraction: float
    covered_exposure_s: float
    segments_met: tuple[tuple[float, float], ...]
    reasons: tuple[str, ...]
    detector_angles_deg: Mapping[str, float]
    poshist_path: str | None
    tte_gti_applied: bool
    cadence_s: float | None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result.update(
            {
                "candidate_index": self.interval.candidate_index,
                "source_name": self.interval.source_name,
                "ra_deg": float(self.interval.skycoord.ra.deg),
                "dec_deg": float(self.interval.skycoord.dec.deg),
                "flare_start_utc": self.interval.start.isot,
                "flare_stop_utc": self.interval.stop.isot,
                "flare_start_met": self.interval.start_met,
                "flare_stop_met": self.interval.stop_met,
                "coverage_basis": (
                    "poshist_and_tte_gti" if self.tte_gti_applied else "poshist_geometry_only"
                ),
                "segments_met": [list(item) for item in self.segments_met],
                "detector_angles_deg": dict(self.detector_angles_deg),
            }
        )
        result.pop("interval", None)
        return result


@dataclass(frozen=True, slots=True)
class GBMDetectorSelection:
    """Geometry-selected GBM detectors for one coverage result."""

    nai: tuple[str, ...]
    bgo: tuple[str, ...]
    angles_deg: Mapping[str, float]
    max_nai_angle_deg: float
    max_bgo_angle_deg: float

    @property
    def all_detectors(self) -> tuple[str, ...]:
        return self.nai + self.bgo

    def to_dict(self) -> dict[str, Any]:
        return {
            "nai": list(self.nai),
            "bgo": list(self.bgo),
            "all_detectors": list(self.all_detectors),
            "angles_deg": dict(self.angles_deg),
            "max_nai_angle_deg": self.max_nai_angle_deg,
            "max_bgo_angle_deg": self.max_bgo_angle_deg,
        }


@dataclass(frozen=True, slots=True)
class GBMDataManifest:
    """Downloaded/reused GBM continuous products and their checksums."""

    root: Path
    poshist_paths: tuple[Path, ...] = ()
    tte_paths: tuple[Path, ...] = ()
    cspec_paths: tuple[Path, ...] = ()
    downloaded_paths: tuple[Path, ...] = ()
    reused_paths: tuple[Path, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        def describe(path: Path) -> dict[str, Any]:
            return {
                "path": str(path),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }

        return {
            "root": str(self.root),
            "poshist": [describe(path) for path in self.poshist_paths],
            "tte": [describe(path) for path in self.tte_paths],
            "cspec": [describe(path) for path in self.cspec_paths],
            "downloaded": [str(path) for path in self.downloaded_paths],
            "reused": [str(path) for path in self.reused_paths],
        }


@dataclass(frozen=True, slots=True)
class GBMBackgroundSummary:
    """Selected local polynomial-background fit metadata."""

    polynomial_order: int
    aicc: float
    statistic: float
    dof: int
    fit_ranges_met: tuple[tuple[float, float], ...]
    bin_size_s: float
    n_parameters: int | None = None
    n_observations: int | None = None
    model_selection: str = "aicc"
    residual_validation: Mapping[str, Any] | None = None
    candidate_scores: Mapping[str, float | None] = field(default_factory=dict)
    candidate_standard_errors: Mapping[str, float | None] = field(default_factory=dict)
    qualified_orders: tuple[int, ...] = ()
    # Full-range fitters are an in-process implementation detail used to
    # materialize alternate background spectra for the conservative model
    # set.  They are never serialized into JSON manifests.
    candidate_fitters: Mapping[int, Any] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )


@dataclass(frozen=True, slots=True)
class GBMSpectralProducts:
    """One detector's OGIP products generated from continuous TTE data."""

    detector: str
    source_pha: Path
    background_bak: Path
    source_ranges_met: tuple[tuple[float, float], ...]
    energy_range_keV: tuple[float, float]
    background: GBMBackgroundSummary
    background_variants: Mapping[int, Path] = field(default_factory=dict)


def check_gbm_coverage(
    interval: GBMFlareInterval,
    poshist: str | Path | Any | None,
    *,
    tte_gti: Sequence[tuple[float, float]] | None = None,
) -> GBMCoverageResult:
    """Intersect visibility, spacecraft state and optional TTE GTIs.

    The result is evaluated on the native position-history cadence.  Cells are
    represented by midpoints between samples, which makes duration accounting
    well-defined at the requested flare boundaries.
    """
    if poshist is None or (isinstance(poshist, (str, Path)) and not _is_valid_fits(Path(poshist))):
        return GBMCoverageResult(
            status="data_missing",
            interval=interval,
            coverage_fraction=0.0,
            covered_exposure_s=0.0,
            segments_met=(),
            reasons=("poshist_missing",),
            detector_angles_deg={},
            poshist_path=None if poshist is None else str(poshist),
            tte_gti_applied=tte_gti is not None,
            cadence_s=None,
        )
    if isinstance(poshist, (str, Path)):
        from gdt.missions.fermi.gbm.poshist import GbmPosHist

        try:
            history = GbmPosHist.open(poshist)
        except (OSError, TypeError, ValueError):
            return GBMCoverageResult(
                status="data_missing",
                interval=interval,
                coverage_fraction=0.0,
                covered_exposure_s=0.0,
                segments_met=(),
                reasons=("poshist_invalid",),
                detector_angles_deg={},
                poshist_path=str(poshist),
                tte_gti_applied=tte_gti is not None,
                cadence_s=None,
            )
        path_value = str(Path(poshist).resolve())
    else:
        history = poshist
        path_value = getattr(history, "filename", None)
    states = history.get_spacecraft_states()
    met = np.asarray(states.time.fermi, dtype=float)
    if met.ndim != 1 or met.size < 2:
        raise ValueError("GBM position history must contain at least two time samples")
    cadence = float(np.nanmedian(np.diff(met)))
    # GDT coordinate transformations are vectorized over the supplied frame.
    # Restrict them to the flare-neighbouring state cells instead of transforming
    # an entire day (normally ~86k samples) for a ten-second interval.
    state_window = (met >= interval.start_met - cadence) & (met <= interval.stop_met + cadence)
    if np.count_nonzero(state_window) < 2:
        return GBMCoverageResult(
            status="data_missing",
            interval=interval,
            coverage_fraction=0.0,
            covered_exposure_s=0.0,
            segments_met=(),
            reasons=("poshist_time_range_missing",),
            detector_angles_deg={},
            poshist_path=path_value,
            tte_gti_applied=tte_gti is not None,
            cadence_s=cadence,
        )
    met = met[state_window]
    frame = history.get_spacecraft_frame()[state_window]
    with warnings.catch_warnings():
        # The calculation is deliberately performed in the spacecraft frame;
        # Astropy's non-rotation warning is expected here and not a data-QC
        # failure.  Persisted coverage still comes from GDT's exact geometry.
        warnings.filterwarnings("ignore", message="transforming other coordinates")
        visible = np.asarray(frame.location_visible(interval.skycoord), dtype=bool)
    good = np.asarray(states["good"], dtype=bool)[state_window]
    saa = np.asarray(states["saa"], dtype=bool)[state_window]
    if not (visible.shape == good.shape == saa.shape == met.shape):
        raise ValueError("position-history state arrays do not share one time axis")
    in_interval = (met >= interval.start_met) & (met <= interval.stop_met)
    valid = in_interval & visible & good & ~saa
    edges = _sample_edges(met)
    segments = _mask_to_intervals(edges, valid, interval.start_met, interval.stop_met)
    if tte_gti is not None:
        segments = _intersect_intervals(segments, _validate_intervals(tte_gti))
    exposure = float(sum(stop - start for start, stop in segments))
    fraction = min(1.0, max(0.0, exposure / interval.duration_s))
    tolerance = max(1e-6, 0.51 * cadence)
    if exposure <= 0.0:
        status: Literal["full", "partial", "none", "data_missing"] = "none"
    elif interval.duration_s - exposure <= tolerance:
        status = "full"
    else:
        status = "partial"
    reasons: list[str] = []
    if np.any(in_interval & ~visible):
        reasons.append("earth_occulted")
    if np.any(in_interval & saa):
        reasons.append("saa")
    if np.any(in_interval & ~good & ~saa):
        reasons.append("spacecraft_not_good")
    if tte_gti is not None and exposure < _interval_duration(_mask_to_intervals(edges, valid, interval.start_met, interval.stop_met)):
        reasons.append("tte_gti_gap")
    angles = _detector_angles(frame, interval.skycoord, valid)
    return GBMCoverageResult(
        status=status,
        interval=interval,
        coverage_fraction=fraction,
        covered_exposure_s=exposure,
        segments_met=tuple(segments),
        reasons=tuple(reasons),
        detector_angles_deg=angles,
        poshist_path=path_value,
        tte_gti_applied=tte_gti is not None,
        cadence_s=cadence,
    )


def select_gbm_detectors(
    coverage: GBMCoverageResult,
    *,
    max_nai_angle: u.Quantity | float = 60.0 * u.deg,
    max_nai: int = 3,
    max_bgo_angle: u.Quantity | float = 90.0 * u.deg,
    max_bgo: int = 2,
) -> GBMDetectorSelection:
    """Select the best-facing NaI and BGO detectors from coverage geometry."""
    nai_limit = _angle_degrees(max_nai_angle, "max_nai_angle")
    bgo_limit = _angle_degrees(max_bgo_angle, "max_bgo_angle")
    if max_nai < 1 or max_bgo < 1:
        raise ValueError("max_nai and max_bgo must be positive")
    angles = {str(name): float(value) for name, value in coverage.detector_angles_deg.items()}
    nai = tuple(
        name
        for name, angle in sorted(angles.items(), key=lambda item: item[1])
        if name in _NAI_DETECTORS and angle <= nai_limit
    )[:max_nai]
    bgo = tuple(
        name
        for name, angle in sorted(angles.items(), key=lambda item: item[1])
        if name in _BGO_DETECTORS and angle <= bgo_limit
    )[:max_bgo]
    return GBMDetectorSelection(
        nai=nai,
        bgo=bgo,
        angles_deg={name: angles[name] for name in nai + bgo},
        max_nai_angle_deg=nai_limit,
        max_bgo_angle_deg=bgo_limit,
    )


def fetch_gbm_continuous_products(
    interval: GBMFlareInterval,
    *,
    destination: str | Path,
    detectors: Iterable[str] = (),
    products: Sequence[Literal["poshist", "tte", "cspec"]] = ("poshist",),
    context_s: float = 0.0,
    verbose: bool = False,
) -> GBMDataManifest:
    """Fetch only continuous GBM products covering an interval.

    Existing files are reused by filename and never overwritten.  TTE files
    are queried hour-by-hour because the Fermi archive serves continuous TTE in
    hourly chunks, while CSPEC and position history are daily products.
    """
    requested = set(products)
    invalid = requested.difference({"poshist", "tte", "cspec"})
    if invalid:
        raise ValueError(f"Unknown GBM products: {sorted(invalid)}")
    context = float(context_s)
    if not math.isfinite(context) or context < 0:
        raise ValueError("context_s must be finite and non-negative")
    root = Path(destination).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    detector_names = tuple(str(item).lower() for item in detectors)
    start = interval.start - TimeDelta(context, format="sec")
    stop = interval.stop + TimeDelta(context, format="sec")
    from gdt.missions.fermi.gbm.finders import ContinuousFinder

    downloaded: list[Path] = []
    reused: list[Path] = []
    poshist_paths: list[Path] = []
    tte_paths: list[Path] = []
    cspec_paths: list[Path] = []
    for day in _utc_days(start, stop):
        finder = ContinuousFinder(day)
        day_dir = root / "daily" / day.datetime.strftime("%Y/%m/%d") / "current"
        day_dir.mkdir(parents=True, exist_ok=True)
        if "poshist" in requested:
            paths, was_downloaded = _fetch_named_files(
                finder, finder.ls_poshist(), day_dir, verbose=verbose
            )
            poshist_paths.extend(paths)
            (downloaded if was_downloaded else reused).extend(paths)
        if "cspec" in requested:
            names = _filter_detector_filenames(finder.ls_cspec(), detector_names)
            paths, was_downloaded = _fetch_named_files(finder, names, day_dir, verbose=verbose)
            cspec_paths.extend(paths)
            (downloaded if was_downloaded else reused).extend(paths)
    if "tte" in requested:
        for hour in _utc_hours(start, stop):
            finder = ContinuousFinder(hour)
            hour_dir = root / "daily" / hour.datetime.strftime("%Y/%m/%d") / "current"
            hour_dir.mkdir(parents=True, exist_ok=True)
            names = _filter_detector_filenames(finder.ls_tte(), detector_names)
            paths, was_downloaded = _fetch_named_files(finder, names, hour_dir, verbose=verbose)
            tte_paths.extend(paths)
            (downloaded if was_downloaded else reused).extend(paths)
    return GBMDataManifest(
        root=root,
        poshist_paths=tuple(_unique_paths(poshist_paths)),
        tte_paths=tuple(_unique_paths(tte_paths)),
        cspec_paths=tuple(_unique_paths(cspec_paths)),
        downloaded_paths=tuple(_unique_paths(downloaded)),
        reused_paths=tuple(_unique_paths(reused)),
    )


def extract_gbm_spectral_products(
    tte_paths: Sequence[str | Path],
    *,
    detector: str,
    source_ranges_met: Sequence[tuple[float, float]],
    background_ranges_met: Sequence[tuple[float, float]],
    energy_range_keV: tuple[float, float],
    output_dir: str | Path,
    background_bin_s: float = 4.0,
) -> GBMSpectralProducts:
    """Create one OGIP PHA/BAK pair from TTE with local polynomial background.

    This function intentionally handles one detector and one continuous source
    segment.  Callers that have disjoint GBM coverage must call it per segment
    and retain separate responses, rather than silently bridging a gap.
    """
    ranges = tuple(_validate_intervals(source_ranges_met))
    background_ranges = tuple(_validate_intervals(background_ranges_met))
    if not ranges:
        raise ValueError("source_ranges_met must contain at least one interval")
    if not background_ranges:
        raise ValueError("background_ranges_met must contain at least one interval")
    e_min, e_max = (float(value) for value in energy_range_keV)
    if not 0.0 < e_min < e_max:
        raise ValueError("energy_range_keV must be a positive increasing pair")
    if float(background_bin_s) <= 0:
        raise ValueError("background_bin_s must be positive")
    # Continuous GBM MET is of order 1e8 s.  Normalize the polynomial basis
    # before fitting so order-1/2 normal matrices are well conditioned; this
    # is an in-process compatibility patch and never edits the installed GDT
    # package.
    _ensure_gdt_polynomial_basis()
    paths = [Path(path).expanduser().resolve() for path in tte_paths]
    if not paths or any(not path.is_file() for path in paths):
        raise FileNotFoundError("all TTE paths must exist before spectrum extraction")
    from gdt.core.background.binned import Polynomial
    from gdt.core.background.fitter import BackgroundFitter
    from gdt.core.binning.unbinned import bin_by_time
    from gdt.missions.fermi.gbm.tte import GbmTte

    ttes = [GbmTte.open(path) for path in paths]
    tte = ttes[0] if len(ttes) == 1 else GbmTte.merge(ttes, force_unique=False)
    outdir = Path(output_dir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    source_pha = tte.to_pha(time_ranges=list(ranges), energy_range=(e_min, e_max))
    source_name = f"gbm_{detector}_source.pha"
    source_pha.write(outdir, filename=source_name, poisson_errs=True, overwrite=True)
    fit_start = min(start for start, _ in background_ranges)
    fit_stop = max(stop for _, stop in background_ranges)
    phaii = tte.to_phaii(
        bin_by_time,
        float(background_bin_s),
        time_range=(fit_start, fit_stop),
        energy_range=(e_min, e_max),
    )
    fitter, summary = _select_polynomial_background(
        phaii,
        background_ranges,
        background_bin_s=float(background_bin_s),
    )
    if len(ranges) != 1:
        raise ValueError("extract one continuous GBM coverage segment at a time")
    bak = integrate_background_interval(fitter, ranges[0])
    background_name = f"gbm_{detector}_background.bak"
    bak.write(outdir, filename=background_name, poisson_errs=False, overwrite=True)
    background_variants: dict[int, Path] = {}
    # Materialize only models that passed the residual gate and were within
    # one standard error of the best predictive score.  The fit stage will
    # run the same response-aware profile for each available variant and
    # retain the most conservative upper bound.
    for order, candidate_fitter in summary.candidate_fitters.items():
        if int(order) == int(summary.polynomial_order):
            continue
        variant = integrate_background_interval(candidate_fitter, ranges[0])
        variant_name = f"gbm_{detector}_background_order{int(order)}.bak"
        variant.write(
            outdir,
            filename=variant_name,
            poisson_errs=False,
            overwrite=True,
        )
        background_variants[int(order)] = outdir / variant_name
    return GBMSpectralProducts(
        detector=str(detector).lower(),
        source_pha=outdir / source_name,
        background_bak=outdir / background_name,
        source_ranges_met=ranges,
        energy_range_keV=(e_min, e_max),
        background=summary,
        background_variants=background_variants,
    )


def integrate_background_interval(fitter: Any, interval: tuple[float, float]) -> Any:
    """Integrate a fitted GDT background exactly over one source interval.

    ``BackgroundFitter.to_bak`` first evaluates the model on every original
    PHAII bin and can therefore include a whole boundary bin.  Calling
    ``interpolate_bins([start], [stop])`` evaluates the fitted model on the
    requested interval itself and preserves its true exposure before creating
    the OGIP BAK object.  A missing interpolation API is a hard compatibility
    error rather than a silent exposure rescale.
    """
    start, stop = _validate_intervals((interval,))[0]
    interpolate = getattr(fitter, "interpolate_bins", None)
    if not callable(interpolate):
        raise RuntimeError("GDT BackgroundFitter lacks interpolate_bins; exact BAK integration unavailable")
    tstart = np.asarray([start], dtype=float)
    tstop = np.asarray([stop], dtype=float)
    rates = interpolate(tstart, tstop)
    # Build the integrated spectrum from the interpolated *rate* directly.
    # ``BackgroundRates.integrate_time`` uses an edge-snapped time mask and
    # therefore includes whole 4-s bins for a short source interval.  The
    # fitter's ``get_exposure(..., scale=True)`` supplies the dead-time-aware
    # exposure of the requested interval without changing the model rate.
    from gdt.core.background.primitives import BackgroundSpectrum

    rate_values = np.asarray(getattr(rates, "rates", ()), dtype=float)
    uncertainty_values = np.asarray(getattr(rates, "rate_uncertainty", ()), dtype=float)
    if rate_values.ndim == 2:
        bin_exposure = np.asarray(getattr(rates, "exposure", ()), dtype=float)
        if bin_exposure.shape != (rate_values.shape[0],) or np.any(bin_exposure <= 0):
            raise RuntimeError("interpolated background has invalid per-bin exposure")
        total_bin_exposure = float(np.sum(bin_exposure))
        rate_values = np.sum(rate_values * bin_exposure[:, None], axis=0) / total_bin_exposure
        uncertainty_values = np.sqrt(
            np.sum((uncertainty_values * bin_exposure[:, None]) ** 2, axis=0)
        ) / total_bin_exposure
    elif rate_values.ndim == 1:
        rate_values = rate_values.reshape(-1)
        uncertainty_values = uncertainty_values.reshape(-1)
    else:
        raise RuntimeError("interpolated background rates have an invalid shape")
    if rate_values.size == 0 or uncertainty_values.shape != rate_values.shape:
        raise RuntimeError("interpolated background rates have inconsistent shapes")
    data_obj = getattr(fitter, "_data_obj", None)
    exposure_value = None
    get_exposure = getattr(data_obj, "get_exposure", None)
    if callable(get_exposure):
        try:
            exposure_value = float(get_exposure((start, stop), scale=True))
        except TypeError:
            # ``Phaii.get_exposure`` forwards to its underlying
            # ``TimeEnergyBins`` object but does not expose the ``scale``
            # keyword.  Use that lower-level implementation when available;
            # its scaled result is the only one that respects partial edge
            # bins for a short source interval.
            data_bins = getattr(data_obj, "data", None)
            scaled_get_exposure = getattr(data_bins, "get_exposure", None)
            if callable(scaled_get_exposure):
                exposure_value = float(scaled_get_exposure((start, stop), scale=True))
            else:
                exposure_value = float(get_exposure((start, stop)))
    # Lightweight GDT-compatible objects may expose the exact requested
    # exposure on the interpolated result rather than on _data_obj. It
    # is acceptable to use that value; fabricating stop-start would treat
    # dead time and GTI gaps as live time and silently bias the spectrum.
    if exposure_value is None:
        interpolated_exposure = np.asarray(getattr(rates, "exposure", ()), dtype=float)
        if interpolated_exposure.size == 1:
            exposure_value = float(interpolated_exposure.reshape(-1)[0])
    if exposure_value is None or not math.isfinite(exposure_value) or exposure_value <= 0:
        raise RuntimeError(
            "fitted background does not expose the exact source-interval exposure; "
            "refusing to substitute geometric duration"
        )
    emin = np.asarray(getattr(rates, "emin", getattr(rates, "lo_edges", ())), dtype=float)
    emax = np.asarray(getattr(rates, "emax", getattr(rates, "hi_edges", ())), dtype=float)
    if emin.shape != rate_values.shape or emax.shape != rate_values.shape:
        raise RuntimeError("interpolated background energy bounds do not match channels")
    spectrum = BackgroundSpectrum(
        rate_values,
        uncertainty_values,
        emin,
        emax,
        np.full(rate_values.size, exposure_value, dtype=float),
    )
    from gdt.core.pha import Bak
    from gdt.core.data_primitives import Gti

    headers = getattr(data_obj, "headers", None)
    kwargs = {}
    if headers is not None:
        # GDT ``FileHeaders`` intentionally implements ``keys``/``__getitem__``
        # rather than ``Mapping.get``.  Preserve only the SPECTRUM header
        # cards accepted by ``Bak.from_data`` and fall back to PRIMARY for
        # lightweight test doubles.
        spectrum_headers = None
        for key in ("SPECTRUM", "PRIMARY"):
            try:
                spectrum_headers = headers[key]
            except (KeyError, TypeError, AttributeError):
                continue
            if spectrum_headers is not None:
                break
        if spectrum_headers is not None:
            kwargs.update(dict(spectrum_headers))
    return Bak.from_data(spectrum, gti=Gti.from_list([(start, stop)]), **kwargs)


def validate_background_residuals(
    observed_rates: np.ndarray,
    predicted_rates: np.ndarray,
    rate_errors: np.ndarray,
    *,
    times: np.ndarray | None = None,
) -> dict[str, Any]:
    """Check mean, trend and nominal Gaussian residual coverage.

    Arrays may be ``(time, channel)`` or one-dimensional.  This is a compact
    diagnostic gate for polynomial-background extrapolation; it does not turn
    a failed check into an extra systematic error.  Callers should mark the
    resulting upper limit ``needs_review`` when ``passed`` is false.
    """
    observed = np.asarray(observed_rates, dtype=float)
    predicted = np.asarray(predicted_rates, dtype=float)
    errors = np.asarray(rate_errors, dtype=float)
    if observed.shape != predicted.shape or observed.shape != errors.shape:
        raise ValueError("observed, predicted and rate_errors must have matching shapes")
    if observed.size == 0 or np.any(~np.isfinite(observed)) or np.any(~np.isfinite(predicted)):
        raise ValueError("background residual arrays must be finite and non-empty")
    if np.any(~np.isfinite(errors)) or np.any(errors <= 0):
        raise ValueError("background rate_errors must be finite and positive")
    standardized = (observed - predicted) / errors
    flat = standardized.reshape(-1)
    mean = float(np.mean(flat))
    std = float(np.std(flat, ddof=1)) if flat.size > 1 else 0.0
    mean_half_width = 1.96 * (std / math.sqrt(flat.size)) if flat.size > 1 else math.inf
    mean_ok = abs(mean) <= mean_half_width
    trend_slope = 0.0
    trend_ok = True
    if times is not None:
        time_values = np.asarray(times, dtype=float)
        if observed.ndim == 1:
            if time_values.shape != observed.shape:
                raise ValueError("times must match the time axis")
            y = flat
            x = time_values
        else:
            if time_values.shape != (observed.shape[0],):
                raise ValueError("times must have one value per time bin")
            x = np.repeat(time_values, observed.shape[1])
            y = flat
        centered = x - np.mean(x)
        denominator = float(np.sum(centered**2))
        if denominator > 0:
            trend_slope = float(np.sum(centered * y) / denominator)
            trend_scale = float(np.std(y, ddof=1) / math.sqrt(max(1, y.size)))
            trend_ok = abs(trend_slope) <= 3.0 * trend_scale / max(float(np.std(x, ddof=1)), 1e-30)
    within_68_count = int(np.sum(np.abs(flat) <= 1.0))
    within_95_count = int(np.sum(np.abs(flat) <= 1.96))
    sample_count = int(flat.size)
    within_68 = within_68_count / sample_count
    within_95 = within_95_count / sample_count
    # Treat the nominal Gaussian coverages as binomial probabilities.  A
    # fixed fraction cut (for example 0.50--0.86) is too permissive for large
    # control samples and too strict for small ones; the exact interval makes
    # the finite-sample uncertainty explicit in the diagnostic payload.
    coverage_68_interval = _binomial_interval(
        within_68_count, sample_count, confidence=0.95
    )
    coverage_95_interval = _binomial_interval(
        within_95_count, sample_count, confidence=0.95
    )
    nominal_68 = float(NormalDist().cdf(1.0) - NormalDist().cdf(-1.0))
    nominal_95 = float(NormalDist().cdf(1.96) - NormalDist().cdf(-1.96))
    coverage_68_ok = coverage_68_interval[0] <= nominal_68 <= coverage_68_interval[1]
    coverage_95_ok = coverage_95_interval[0] <= nominal_95 <= coverage_95_interval[1]
    coverage_ok = bool(coverage_68_ok and coverage_95_ok)
    return {
        "passed": bool(mean_ok and trend_ok and coverage_ok),
        "mean": mean,
        "mean_95_half_width": float(mean_half_width),
        "trend_slope": trend_slope,
        "coverage_68": within_68,
        "coverage_95": within_95,
        "coverage_68_nominal": nominal_68,
        "coverage_95_nominal": nominal_95,
        "coverage_68_interval": list(coverage_68_interval),
        "coverage_95_interval": list(coverage_95_interval),
        "coverage_68_ok": bool(coverage_68_ok),
        "coverage_95_ok": bool(coverage_95_ok),
        "sample_count": sample_count,
    }


def _binomial_interval(
    successes: int,
    trials: int,
    *,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Return a finite-sample two-sided binomial interval.

    SciPy is already an optional dependency of the GBM fitting stack, but the
    Wilson calculation keeps diagnostics usable with lightweight GDT test
    doubles when ``scipy.stats`` is unavailable.  This interval is a check on
    predictive coverage only; it is never folded into the source likelihood.
    """
    n = int(trials)
    k = int(successes)
    level = float(confidence)
    if n <= 0 or k < 0 or k > n or not 0.0 < level < 1.0:
        raise ValueError("invalid binomial interval inputs")
    tail = (1.0 - level) / 2.0
    try:
        from scipy.stats import beta

        lower = 0.0 if k == 0 else float(beta.ppf(tail, k, n - k + 1))
        upper = 1.0 if k == n else float(beta.ppf(1.0 - tail, k + 1, n - k))
        if math.isfinite(lower) and math.isfinite(upper):
            return max(0.0, lower), min(1.0, upper)
    except Exception:  # pragma: no cover - lightweight installations only
        pass
    z = NormalDist().inv_cdf(1.0 - tail)
    proportion = k / n
    denominator = 1.0 + z * z / n
    center = (proportion + z * z / (2.0 * n)) / denominator
    half = z * math.sqrt(
        proportion * (1.0 - proportion) / n + z * z / (4.0 * n * n)
    ) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def _background_channel_count(phaii: Any, fitter: Any) -> int:
    """Return the number of valid energy channels used by GDT background fits."""
    for owner in (phaii, getattr(phaii, "data", None), getattr(fitter, "_data_obj", None)):
        if owner is None:
            continue
        for name in ("num_chans", "num_channels", "nchan"):
            value = getattr(owner, name, None)
            if value is not None:
                try:
                    integer = int(value)
                except (TypeError, ValueError):
                    continue
                if integer > 0:
                    return integer
        counts = getattr(owner, "counts", None)
        if counts is not None:
            array = np.asarray(counts)
            if array.ndim >= 2 and array.shape[-1] > 0:
                return int(array.shape[-1])
    statistic = np.asarray(getattr(fitter, "statistic", []), dtype=float)
    if statistic.ndim > 0 and statistic.size > 1:
        return int(statistic.size)
    return 1


def _background_holdout_score(
    phaii: Any,
    ranges: tuple[tuple[float, float], ...],
    *,
    order: int,
) -> float | None:
    """Return the mean held-out Poisson deviance for one order.

    The public helper keeps its historical scalar return value.  The
    selection stage uses the detailed helper as well so model-set
    qualification can account for score uncertainty.
    """
    statistics = _background_holdout_statistics(phaii, ranges, order=order)
    return None if statistics is None else statistics[0]


def _background_holdout_statistics(
    phaii: Any,
    ranges: tuple[tuple[float, float], ...],
    *,
    order: int,
) -> tuple[float, float] | None:
    """Score a polynomial order on contiguous held-out background blocks.

    The score is the Poisson deviance of predicted counts, evaluated on the
    last fifth of each requested background window after fitting the earlier
    four fifths.  It is deliberately a predictive check rather than a second
    likelihood term.  ``None`` means that the GDT object does not expose a
    usable time/count grid, in which case the caller records an AICc fallback.
    """
    from gdt.core.background.binned import Polynomial
    from gdt.core.background.fitter import BackgroundFitter

    data = getattr(phaii, "data", phaii)
    tstart = np.asarray(getattr(data, "tstart", ()), dtype=float)
    tstop = np.asarray(getattr(data, "tstop", ()), dtype=float)
    counts = np.asarray(getattr(data, "counts", ()), dtype=float)
    exposure = np.asarray(getattr(data, "exposure", ()), dtype=float)
    if (
        tstart.ndim != 1
        or tstop.shape != tstart.shape
        or counts.ndim != 2
        or exposure.shape != tstart.shape
        or counts.shape[0] != tstart.size
        or np.any(~np.isfinite(tstart))
        or np.any(~np.isfinite(tstop))
        or np.any(~np.isfinite(counts))
        or np.any(~np.isfinite(exposure))
        or np.any(exposure <= 0)
    ):
        return None
    train_ranges: list[tuple[float, float]] = []
    holdout_ranges: list[tuple[float, float]] = []
    for start, stop in ranges:
        width = float(stop - start)
        if width <= 0:
            continue
        split = float(start + 0.8 * width)
        # A polynomial of order k needs at least k+2 independent time bins;
        # short windows are left to the AICc fallback rather than extrapolated.
        train_mask = (tstart >= start) & (tstop <= split) & (tstop > tstart)
        valid_mask = (tstart >= split) & (tstop <= stop) & (tstop > tstart)
        if int(np.sum(train_mask)) < int(order + 2) or int(np.sum(valid_mask)) < 2:
            continue
        train_ranges.append((float(start), split))
        holdout_ranges.append((split, float(stop)))
    if not train_ranges or not holdout_ranges:
        return None
    try:
        fitter = BackgroundFitter.from_phaii(
            phaii,
            Polynomial,
            time_ranges=train_ranges,
        )
        fitter.fit(order=order)
        deviance_terms: list[float] = []
        for (hold_start, hold_stop) in holdout_ranges:
            mask = (tstart >= hold_start) & (tstop <= hold_stop) & (tstop > tstart)
            if not np.any(mask):
                continue
            predicted_obj = fitter.interpolate_bins(tstart[mask], tstop[mask])
            predicted_rates = np.asarray(predicted_obj.rates, dtype=float)
            if predicted_rates.ndim == 1:
                predicted_rates = predicted_rates.reshape(1, -1)
            observed_counts = counts[mask]
            exposure_values = exposure[mask]
            if predicted_rates.shape != observed_counts.shape:
                return None
            expected_counts = np.maximum(predicted_rates * exposure_values[:, None], 1e-12)
            if np.any(~np.isfinite(expected_counts)) or np.any(expected_counts <= 0):
                return None
            # Poisson deviance remains well-defined at zero observed counts.
            log_term = np.zeros_like(observed_counts, dtype=float)
            positive = observed_counts > 0
            log_term[positive] = observed_counts[positive] * np.log(
                observed_counts[positive] / expected_counts[positive]
            )
            terms = 2.0 * (expected_counts - observed_counts + log_term)
            deviance_terms.extend(float(value) for value in terms.ravel())
        if not deviance_terms:
            return None
        values = np.asarray(deviance_terms, dtype=float)
        if np.any(~np.isfinite(values)):
            return None
        mean = float(np.mean(values))
        standard_error = (
            float(np.std(values, ddof=1) / math.sqrt(values.size))
            if values.size > 1
            else math.inf
        )
        return mean, standard_error
    except Exception as exc:
        logger.debug("background holdout order %d unavailable: %s", order, exc)
        return None


def _select_polynomial_background(phaii: Any, ranges: tuple[tuple[float, float], ...], *, background_bin_s: float) -> tuple[Any, GBMBackgroundSummary]:
    from gdt.core.background.binned import Polynomial
    from gdt.core.background.fitter import BackgroundFitter

    candidates: list[
        tuple[float, Any, GBMBackgroundSummary, float | None, float | None]
    ] = []
    for order in (0, 1, 2):
        # FIX(C8): protect against singular matrix in polynomial background fitting
        try:
            fitter = BackgroundFitter.from_phaii(phaii, Polynomial, time_ranges=list(ranges))
            fitter.fit(order=order)
        except np.linalg.LinAlgError:
            logger.warning("polynomial order %d fit singular; skipping", order)
            continue
        except Exception as exc:  # noqa: BLE001 - fitting may fail for various numerical reasons
            logger.warning("polynomial order %d fit failed: %s", order, exc)
            continue
        statistic = float(np.nansum(np.asarray(fitter.statistic, dtype=float)))
        dof = int(np.nansum(np.asarray(fitter.dof, dtype=float)))
        # The polynomial is fitted independently for every valid energy
        # channel.  Counting only ``order + 1`` under-penalizes multi-channel
        # models and yields an invalid AICc.  Derive the number of channels
        # from the PHAII data (falling back to the statistic vector for small
        # test doubles) and count one coefficient set per channel.
        n_channels = _background_channel_count(phaii, fitter)
        n_parameters = (order + 1) * n_channels
        data_counts = np.asarray(getattr(getattr(phaii, "data", None), "counts", ()))
        n_data = int(data_counts.size) if data_counts.ndim == 2 else 0
        n_observations = max(n_parameters + 2, dof + n_parameters, n_data)
        correction = 2.0 * n_parameters * (n_parameters + 1) / (n_observations - n_parameters - 1)
        aicc = statistic + 2.0 * n_parameters + correction
        residual_validation = _validate_fitted_background(phaii, fitter, ranges=ranges)
        summary = GBMBackgroundSummary(
            polynomial_order=order,
            aicc=float(aicc),
            statistic=statistic,
            dof=dof,
            fit_ranges_met=ranges,
            bin_size_s=float(background_bin_s),
            n_parameters=int(n_parameters),
            n_observations=int(n_observations),
            model_selection="aicc_fallback",
            residual_validation=residual_validation,
        )
        holdout_statistics = _background_holdout_statistics(
            phaii,
            ranges,
            order=order,
        )
        holdout_score = (
            None if holdout_statistics is None else float(holdout_statistics[0])
        )
        holdout_standard_error = (
            None if holdout_statistics is None else float(holdout_statistics[1])
        )
        candidates.append(
            (aicc, fitter, summary, holdout_score, holdout_standard_error)
        )
    # FIX(C8): if all polynomial orders failed, raise an explicit error
    if not candidates:
        raise RuntimeError("all polynomial background orders failed (singular matrix or fitting error)")
    holdout_scores = {
        str(item[2].polynomial_order): (None if item[3] is None else float(item[3]))
        for item in candidates
    }
    holdout_standard_errors = {
        str(item[2].polynomial_order): (
            None if item[4] is None else float(item[4])
        )
        for item in candidates
    }
    holdout_candidates = [item for item in candidates if item[3] is not None]
    if len(holdout_candidates) >= 2:
        # ``BackgroundFitter.parameters`` is not stable across GDT releases;
        # use the selected tuple's identity instead of relying on it.
        selected_item = min(holdout_candidates, key=lambda item: float(item[3]))
        selected_summary = selected_item[2]
        selected_order = int(selected_summary.polynomial_order)
        best_score = float(selected_item[3])
        best_error = float(selected_item[4]) if selected_item[4] is not None else 0.0
        if not math.isfinite(best_error):
            best_error = 0.0
        qualified_orders = tuple(
            int(item[2].polynomial_order)
            for item in holdout_candidates
            if float(item[3]) <= best_score + best_error
            and isinstance(item[2].residual_validation, Mapping)
            and bool(item[2].residual_validation.get("passed", False))
        )
        # The hold-out fitters were trained on only the first 80% of each
        # control block.  Refit every qualified order on the complete control
        # windows before materialising a source-interval BAK; otherwise an
        # alternate model would be evaluated with a different exposure/model
        # contract from the selected one.
        final_fitters: dict[int, Any] = {}
        for order in qualified_orders:
            try:
                final_fitter = BackgroundFitter.from_phaii(
                    phaii,
                    Polynomial,
                    time_ranges=list(ranges),
                )
                final_fitter.fit(order=int(order))
                final_fitters[int(order)] = final_fitter
            except Exception as exc:
                logger.warning(
                    "qualified background order %d refit failed: %s", order, exc
                )
        if selected_order in final_fitters:
            selected_summary = replace(
                selected_summary,
                model_selection="poisson_deviance_holdout_qualified",
                candidate_scores=holdout_scores,
                candidate_standard_errors=holdout_standard_errors,
                qualified_orders=tuple(sorted(final_fitters)),
                candidate_fitters=final_fitters,
            )
            return final_fitters[selected_order], selected_summary
        logger.warning(
            "selected holdout background order %d could not be refit on all windows",
            selected_order,
        )
    _, fitter, summary, _, _ = min(candidates, key=lambda item: item[0])
    selected_order = int(summary.polynomial_order)
    qualified_orders = (
        (selected_order,)
        if isinstance(summary.residual_validation, Mapping)
        and bool(summary.residual_validation.get("passed", False))
        else ()
    )
    summary = replace(
        summary,
        candidate_scores=holdout_scores,
        candidate_standard_errors=holdout_standard_errors,
        qualified_orders=qualified_orders,
        candidate_fitters={
            int(item[2].polynomial_order): item[1]
            for item in candidates
            if int(item[2].polynomial_order) in qualified_orders
        },
    )
    return fitter, summary


def _validate_fitted_background(
    phaii: Any,
    fitter: Any,
    *,
    ranges: Sequence[tuple[float, float]] | None = None,
) -> dict[str, Any] | None:
    """Run residual diagnostics on background windows only.

    ``phaii`` normally spans the source interval as well as both off-source
    windows.  Including the source bins in a background residual check would
    turn a real flare into an apparent background-model failure, so the
    optional ``ranges`` argument explicitly restricts the diagnostic to the
    continuous off-source blocks used by the fitter.
    """
    data = getattr(phaii, "data", phaii)
    tstart = getattr(data, "tstart", None)
    tstop = getattr(data, "tstop", None)
    counts = getattr(data, "counts", None)
    exposure = getattr(data, "exposure", None)
    if any(value is None for value in (tstart, tstop, counts, exposure)):
        return None
    try:
        tstart_array = np.asarray(tstart, dtype=float)
        tstop_array = np.asarray(tstop, dtype=float)
        count_array = np.asarray(counts, dtype=float)
        exposure_array = np.asarray(exposure, dtype=float)
        if count_array.ndim != 2 or exposure_array.shape != (count_array.shape[0],):
            return None
        if np.any(tstop_array <= tstart_array) or np.any(exposure_array <= 0):
            return {"passed": False, "error": "invalid_time_or_exposure_grid"}
        interpolated = fitter.interpolate_bins(tstart_array, tstop_array)
        predicted = np.asarray(interpolated.rates, dtype=float)
        if predicted.shape != count_array.shape:
            return None
        if ranges is not None:
            selected = np.zeros(tstart_array.shape, dtype=bool)
            for start, stop in ranges:
                # Only complete bins inside an off-source block are valid
                # residual controls.  A bin crossing a source/background
                # boundary can contain source photons (or a guard interval)
                # and must not be allowed into the background diagnostic.
                selected |= (tstart_array >= float(start)) & (
                    tstop_array <= float(stop)
                )
            if not np.any(selected):
                return {
                    "passed": False,
                    "error": "no_background_bins_for_residual_validation",
                    "ranges_met": [list(item) for item in ranges],
                }
            tstart_array = tstart_array[selected]
            tstop_array = tstop_array[selected]
            count_array = count_array[selected]
            exposure_array = exposure_array[selected]
            predicted = predicted[selected]
        observed = count_array / exposure_array[:, None]
        errors = np.sqrt(np.maximum(count_array, 1.0)) / exposure_array[:, None]
        times = 0.5 * (tstart_array + tstop_array)
        result = validate_background_residuals(observed, predicted, errors, times=times)
        result["ranges_met"] = (
            None if ranges is None else [list(item) for item in ranges]
        )
        result["background_bin_count"] = int(tstart_array.size)
        return result
    except Exception as exc:
        return {"passed": False, "error": f"{type(exc).__name__}:{exc}"}


def _sample_edges(times: np.ndarray) -> np.ndarray:
    if np.any(np.diff(times) <= 0):
        raise ValueError("position-history times must be strictly increasing")
    centers = 0.5 * (times[1:] + times[:-1])
    first = times[0] - (centers[0] - times[0])
    last = times[-1] + (times[-1] - centers[-1])
    return np.concatenate(([first], centers, [last]))


def _angle_degrees(value: u.Quantity | float, name: str) -> float:
    degrees = float(value.to_value(u.deg)) if isinstance(value, u.Quantity) else float(value)
    if not math.isfinite(degrees) or not 0.0 < degrees <= 180.0:
        raise ValueError(f"{name} must be a finite angle in (0, 180] degrees")
    return degrees


def _mask_to_intervals(edges: np.ndarray, mask: np.ndarray, start: float, stop: float) -> list[tuple[float, float]]:
    intervals = []
    for index, value in enumerate(np.asarray(mask, dtype=bool)):
        if not value or edges[index + 1] <= start or edges[index] >= stop:
            continue
        left = max(float(edges[index]), start)
        right = min(float(edges[index + 1]), stop)
        # Keep requested flare boundaries exact in persisted provenance rather
        # than exposing sub-nanosecond floating-point conversion noise.
        if abs(left - start) < 1e-6:
            left = start
        if abs(right - stop) < 1e-6:
            right = stop
        intervals.append((left, right))
    return _merge_intervals(intervals)


def _validate_intervals(intervals: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    checked = []
    for start, stop in intervals:
        left = float(start)
        right = float(stop)
        if not math.isfinite(left) or not math.isfinite(right) or not left < right:
            raise ValueError("all GTI intervals must be finite increasing pairs")
        checked.append((left, right))
    return _merge_intervals(checked)


def _merge_intervals(intervals: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    if not intervals:
        return []
    merged: list[list[float]] = []
    for start, stop in sorted(intervals):
        if not merged or start > merged[-1][1] + 1e-9:
            merged.append([start, stop])
        else:
            merged[-1][1] = max(merged[-1][1], stop)
    return [(float(start), float(stop)) for start, stop in merged]


def _intersect_intervals(left: Sequence[tuple[float, float]], right: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = []
    for left_start, left_stop in left:
        for right_start, right_stop in right:
            start = max(left_start, right_start)
            stop = min(left_stop, right_stop)
            if start < stop:
                result.append((start, stop))
    return _merge_intervals(result)


def _interval_duration(intervals: Sequence[tuple[float, float]]) -> float:
    return float(sum(stop - start for start, stop in intervals))


def _detector_angles(frame: Any, coordinate: SkyCoord, valid_mask: np.ndarray) -> dict[str, float]:
    if not np.any(valid_mask):
        return {}
    from gdt.missions.fermi.gbm.detectors import GbmDetectors

    selected_frame = frame[valid_mask]
    angles: dict[str, float] = {}
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="transforming other coordinates")
        for detector in GbmDetectors:
            direction = detector.skycoord(selected_frame)
            # FIX(M9): use median instead of mean for detector angles, robust against attitude jumps
            angles[detector.name] = float(np.nanmedian(direction.separation(coordinate).to_value(u.deg)))
    return angles


def _utc_days(start: Time, stop: Time) -> list[Time]:
    current = Time(start.datetime.date().isoformat(), format="iso", scale="utc")
    end = Time(stop.datetime.date().isoformat(), format="iso", scale="utc")
    result = []
    while current <= end:
        result.append(current)
        current += TimeDelta(86400.0, format="sec")
    return result


def _utc_hours(start: Time, stop: Time) -> list[Time]:
    start_dt = start.datetime.replace(minute=0, second=0, microsecond=0)
    stop_dt = stop.datetime.replace(minute=0, second=0, microsecond=0)
    result = []
    current = start_dt
    while current <= stop_dt:
        result.append(Time(current, scale="utc"))
        current += timedelta(hours=1)
    return result


def _filter_detector_filenames(names: Sequence[str], detectors: Sequence[str]) -> list[str]:
    if not detectors:
        return list(names)
    requested = {str(detector).lower() for detector in detectors}
    return [name for name in names if any(f"_{detector}_" in name.lower() for detector in requested)]


def _fetch_named_files(finder: Any, names: Sequence[str], destination: Path, *, verbose: bool) -> tuple[list[Path], bool]:
    if not names:
        return [], False
    existing = []
    fetched = []
    for name in names:
        path = destination / Path(name).name
        if path.is_file() and _is_valid_fits(path):
            existing.append(path)
        else:
            was_downloaded = _resume_archive_file(finder, name, path, verbose=verbose)
            if was_downloaded and _is_valid_fits(path):
                fetched.append(path)
            # FIX(H5): log and remove corrupt downloads so next run retries
            elif was_downloaded:
                logger.warning("downloaded file failed FITS validation: %s", path)
                path.unlink(missing_ok=True)
    valid_fetched = [path for path in fetched if _is_valid_fits(path)]
    return [*existing, *valid_fetched], bool(valid_fetched)


def _unique_paths(paths: Iterable[Path]) -> list[Path]:
    result: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            result.append(resolved)
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_valid_fits(path: Path) -> bool:
    """Return whether a cached Fermi product is a non-empty readable FITS file."""
    if not path.is_file() or path.stat().st_size < 2880:
        return False
    try:
        from astropy.io import fits

        # FIX(H3): use memmap + lazy_load to avoid loading entire file into memory
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with fits.open(path, memmap=True, lazy_load_hdus=True) as hdul:
                _ = hdul[0].header
                return len(hdul) > 0
    except (OSError, TypeError, ValueError, EOFError):
        return False


def _resume_archive_file(finder: Any, name: str, destination: Path, *, verbose: bool) -> bool:
    """Download one archive file with HTTP Range resume support.

    GDT's finder is still the authoritative source for the date directory and
    filenames.  This small transport wrapper preserves partial bytes across
    interrupted interactive runs, rather than treating them as a finished file
    or repeatedly starting a multi-megabyte transfer from zero.
    """
    import requests

    base_url = getattr(getattr(finder, "_protocol", None), "_url", None)
    cwd = getattr(finder, "_cwd", None)
    if not base_url or not cwd:
        raise RuntimeError("GDT finder does not expose an HTTP archive location")
    url = f"{str(base_url).rstrip('/')}/{str(cwd).strip('/')}/{name}"
    offset = destination.stat().st_size if destination.is_file() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    try:
        with requests.get(url, headers=headers, stream=True, timeout=(20.0, 180.0)) as response:
            # FIX(H4): handle HTTP 416 by removing stale partial download and retrying
            if response.status_code == 416:
                response.close()
                destination.unlink(missing_ok=True)
                with requests.get(url, stream=True, timeout=(20.0, 180.0)) as retry_resp:
                    retry_resp.raise_for_status()
                    with destination.open("wb") as fh:
                        for chunk in retry_resp.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                fh.write(chunk)
                return True
            if offset and response.status_code == 200:
                offset = 0
            response.raise_for_status()
            mode = "ab" if offset and response.status_code == 206 else "wb"
            with destination.open(mode) as handle:
                for block in response.iter_content(chunk_size=1024 * 1024):
                    if block:
                        handle.write(block)
        if verbose:
            print(f"GBM archive {name}: {destination.stat().st_size} bytes")
        logger.info("GBM archive %s: %d bytes", name, destination.stat().st_size)
        return True
    # FIX(H13): catch OSError from disk write failures (e.g. disk full)
    except (requests.RequestException, OSError) as exc:
        if verbose:
            print(f"GBM archive transfer interrupted for {name}: {exc}")
        logger.warning("GBM archive transfer interrupted for %s: %s", name, exc)
        return False


# ---------------------------------------------------------------------------
# Resumable single-target GBM pipeline ("fermi.gbm")
# ---------------------------------------------------------------------------

KEV_TO_ERG = 1.602176634e-9


def _ensure_gdt_polynomial_basis() -> None:
    """Stabilize GDT ``Polynomial`` background fits for continuous data.

    GDT evaluates the time basis as raw ``MET**k`` integrals.  For trigger
    data (times referenced to the trigger) that is well conditioned, but the
    continuous daily TTE used here carries absolute MET ~8e8 s, so the order-1
    and order-2 normal matrices reach condition numbers ~1e19/1e28 and
    ``np.linalg.inv`` fails with "Singular matrix".  The patch normalizes the
    basis to ``[-1, 1]`` over the span cached at fit time; every later
    evaluation (model, uncertainty, interpolate/to_bak extrapolation under the
    source window) reuses the same cached span, so the fitted coefficients
    keep exactly their original meaning.
    """
    from gdt.core.background.binned import Polynomial

    if getattr(Polynomial, "_jinwu_normalized_basis", False):
        return

    # FIX(H2): verify GDT Polynomial API compatibility before patching
    import inspect
    for attr in ("_eval_basis", "_eval_model"):
        if not hasattr(Polynomial, attr):
            raise RuntimeError(
                f"GDT Polynomial class missing {attr!r}; jinwu patch incompatible"
            )
    sig = inspect.signature(Polynomial._eval_basis)
    if len(sig.parameters) != 3:
        raise RuntimeError(
            f"GDT Polynomial._eval_basis signature changed: {sig}; jinwu patch may be incompatible"
        )
    logger.info("GDT Polynomial._eval_basis signature: %s", sig)

    def _edges(self, tstart, tstop):
        t0, t1 = self._norm_span
        mid = 0.5 * (t0 + t1)
        half = 0.5 * (t1 - t0)
        s_lo = (np.asarray(tstart, dtype=float) - mid) / half
        s_hi = (np.asarray(tstop, dtype=float) - mid) / half
        ds = s_hi - s_lo
        zerowidth = ds <= 0.0
        ds[zerowidth] = 2e-6 / half
        s_hi[zerowidth] = s_lo[zerowidth] + ds[zerowidth]
        return s_lo, s_hi, ds

    def _eval_basis(self, tstart, tstop):
        if getattr(self, "_norm_span", None) is None:
            # FIX(C6): guard against degenerate time span (t0 == t1) causing NaN propagation
            t0, t1 = float(np.min(tstart)), float(np.max(tstop))
            if t0 >= t1:
                t0, t1 = t0 - 1.0, t1 + 1.0
            object.__setattr__(self, "_norm_span", (t0, t1))
        s_lo, s_hi, ds = _edges(self, tstart, tstop)
        basis_func = np.array(
            [
                (s_hi ** (i + 1.0) - s_lo ** (i + 1.0)) / ((i + 1.0) * ds)
                for i in range(self._order + 1)
            ]
        )
        return np.tile(basis_func[:, :, np.newaxis], self._numchans)

    def _eval_model(self, tstart, tstop):
        s_lo, s_hi, ds = _edges(self, tstart, tstop)
        model = np.zeros((tstart.size, self._numchans))
        for i in range(self._order + 1):
            model += (
                self._coeff[i, :]
                * (
                    (s_hi[:, np.newaxis] ** (i + 1.0) - s_lo[:, np.newaxis] ** (i + 1.0))
                    / ((i + 1.0) * ds[:, np.newaxis])
                )
            ).astype(float)
        if model.ndim == 1:
            model = model.reshape(-1, 1)
        return model

    _eval_basis.__name__ = "_eval_basis"
    _eval_model.__name__ = "_eval_model"
    Polynomial._eval_basis = _eval_basis
    Polynomial._eval_model = _eval_model
    Polynomial._jinwu_normalized_basis = True


def powerlaw_energy_flux_per_norm(gamma: float, elo: float, ehi: float) -> float:
    """Energy flux (erg cm^-2 s^-1) of XSPEC *powerlaw* with norm=1 in a band.

    XSPEC powerlaw photon density is norm * (E/1keV)^-gamma in ph/cm^2/s/keV.
    """
    exponent = 2.0 - float(gamma)
    if abs(exponent) < 1e-9:
        integral = math.log(ehi / elo)
    else:
        integral = (ehi**exponent - elo**exponent) / exponent
    return KEV_TO_ERG * integral


def tte_coverage(paths: Sequence[str | Path]) -> tuple[float, float]:
    """Union time coverage of several GBM TTE files in Fermi MET."""
    from gdt.missions.fermi.gbm.tte import GbmTte

    lo, hi = math.inf, -math.inf
    for path in paths:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tte = GbmTte.open(str(path))
        lo = min(lo, float(tte.time_range[0]))
        hi = max(hi, float(tte.time_range[1]))
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        raise ValueError(f"invalid TTE coverage from {paths}")
    return lo, hi


def gti_intervals_for_paths(
    paths: Sequence[str | Path],
    interval: tuple[float, float],
) -> tuple[tuple[float, float], ...]:
    """Return merged TTE GTI overlap intervals in absolute Fermi MET.

    The response and spectrum stages must use the exposure that was actually
    available in the requested source interval.  This helper reads the GTI
    extensions from each TTE file, intersects them with ``interval`` and
    merges touching intervals.  It never turns a wall-clock interval into a
    scaled exposure estimate.
    """
    start, stop = _validate_intervals((interval,))[0]
    from gdt.missions.fermi.gbm.tte import GbmTte

    selected: list[tuple[float, float]] = []
    for path in paths:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tte = GbmTte.open(str(path))
        gti = getattr(tte, "gti", None)
        if gti is None or not hasattr(gti, "as_list"):
            raise ValueError(f"TTE {path} has no readable GTI")
        for left, right in gti.as_list():
            overlap_left = max(start, float(left))
            overlap_right = min(stop, float(right))
            if overlap_right > overlap_left:
                selected.append((overlap_left, overlap_right))
    if not selected:
        return ()
    selected.sort()
    merged: list[list[float]] = []
    for left, right in selected:
        if merged and left <= merged[-1][1] + 1e-9:
            merged[-1][1] = max(merged[-1][1], right)
        else:
            merged.append([left, right])
    return tuple((float(left), float(right)) for left, right in merged)


def background_windows(
    seg: tuple[float, float],
    coverage: tuple[float, float],
    *,
    guard_s: float,
    context_min_s: float,
    context_max_s: float,
) -> tuple[list[tuple[float, float]], list[str]]:
    """Background windows flanking the source segment, adjacent to the event.

    Windows are anchored at the source side with the far side truncated to
    ``context_max_s``; anything shorter than ``context_min_s`` is flagged but
    still used down to a quarter of ``context_min_s``.
    """
    flags: list[str] = []
    windows: list[tuple[float, float]] = []
    sides = (
        ("pre", coverage[0], seg[0] - guard_s),
        ("post", seg[1] + guard_s, coverage[1]),
    )
    for name, start, stop in sides:
        length = stop - start
        if length <= 0:
            continue
        used = min(context_max_s, length)
        if used < context_min_s:
            flags.append(f"short_background_{name}:{used:.0f}s<{context_min_s:.0f}s")
        # FIX(M15): document the 1/4 minimum background fraction threshold
        # Polynomial background fitting requires at least ~75s (order=0) of data;
        # context_min_s/4 is the empirical lower bound below which results are unreliable.
        _MIN_BACKGROUND_FRACTION = 0.25
        if used >= context_min_s * _MIN_BACKGROUND_FRACTION:
            # Anchor the window at the source side; the far side is truncated.
            if name == "pre":
                windows.append((stop - used, stop))
            else:
                windows.append((start, start + used))
    if not windows:
        flags.append("no_usable_background_window")
    return windows, flags


def _detector_of(filename: str) -> str | None:
    import re

    match = re.search(r"glg_(?:tte|cspec)_([a-z0-9]{1,2})_", filename)
    return match.group(1) if match else None


def single_response(
    rsp_path: str | Path,
    mid_met: float,
    out_dir: str | Path,
    detector: str,
    *,
    time_intervals: Sequence[tuple[float, float]] | None = None,
    energy_band_keV: tuple[float, float] | None = None,
    photon_index: float = 2.0,
    midpoint_tolerance: float = 0.01,
    return_metadata: bool = False,
) -> Path | tuple[Path, dict[str, Any]]:
    """Return an OGIP single-matrix RSP covering one source segment.

    For RSP2 inputs the default is a midpoint interpolation for backwards
    compatibility.  When ``time_intervals`` are supplied (normally the TTE
    GTI overlap), the matrices are exposure weighted across those intervals.
    A midpoint approximation is retained only when its folded power-law rate
    differs from the exposure-weighted rate by at most ``midpoint_tolerance``;
    otherwise the weighted response is written.  The choice and comparison
    are returned when ``return_metadata=True`` and are also persisted by the
    pipeline report.

    The output is post-processed for the two consumers of this file:

    - PyXspec ``fakeit`` requires the OGIP "SPECRESP MATRIX" name, while
      ``jinwu.core.io.read_rmf`` requires the legacy "MATRIX" name: keep the
      latter, ``fakeit`` accepts both.
    - jinwu maps ``fakeit`` output channels (1-based) through the response
      EBOUNDS CHANNEL column, so stored 0-based channels are renumbered.
    """
    from astropy.io import fits

    rsp_path = Path(rsp_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, Any] = {
        "input": str(rsp_path.resolve()),
        "detector": str(detector).lower(),
        "time_weighting": "not_applicable",
        "midpoint_relative_difference": None,
        "midpoint_tolerance": float(midpoint_tolerance),
        "gti_intervals_met": None,
    }
    if not math.isfinite(float(midpoint_tolerance)) or not 0.0 <= float(midpoint_tolerance) < 1.0:
        raise ValueError("midpoint_tolerance must be finite and in [0, 1)")
    is_rsp2 = rsp_path.name.lower().endswith((".rsp2", ".rsp2.gz"))
    rsp2_input = rsp_path
    temporary_rsp2: Path | None = None
    if is_rsp2 and rsp_path.name.lower().endswith(".gz"):
        # GDT accepts an uncompressed RSP2 path more reliably across releases;
        # keep the archive untouched and stage a private copy for reading.
        token = "".join(
            character if character.isalnum() or character in "._-" else "_"
            for character in rsp_path.stem
        )
        temporary_rsp2 = out_dir / f".{token or 'input'}.rsp2"
        with gzip.open(rsp_path, "rb") as source, temporary_rsp2.open("wb") as target:
            shutil.copyfileobj(source, target)
        rsp2_input = temporary_rsp2
    try:
        if is_rsp2:
            from gdt.missions.fermi.gbm.response import GbmRsp2

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rsp2 = GbmRsp2.open(str(rsp2_input))
            rsp, response_meta = _select_rsp2_response(
                rsp2,
                mid_met=float(mid_met),
                time_intervals=time_intervals,
                energy_band_keV=energy_band_keV,
                photon_index=float(photon_index),
                midpoint_tolerance=float(midpoint_tolerance),
            )
            metadata.update(response_meta)
            out = out_dir / f"gbm_{detector}_response.rsp"
            if out.exists():
                out.unlink()
            rsp.write(str(out_dir), filename=out.name)
        else:
            # Never mutate a caller's downloaded/reference response while
            # normalizing channels or extension names.  The pipeline writes all
            # post-processing below its isolated workspace.
            out = out_dir / f"gbm_{detector}_response.rsp"
            if rsp_path.resolve() == out.resolve():
                raise ValueError(
                    "single_response output must be separate from the input response; "
                    "reference RSP files are read-only"
                )
            if rsp_path.resolve() != out.resolve():
                shutil.copy2(rsp_path, out)
            metadata["time_weighting"] = "single_matrix_input"
        if not out.exists():
            raise RuntimeError(f"single-matrix response was not written: {out}")
        with fits.open(out, mode="update") as hdul:
            for hdu in hdul:
                if hdu.name.upper() == "SPECRESP MATRIX":
                    hdu.name = "MATRIX"
                    break
            ebounds = hdul["EBOUNDS"]
            col_idx = ebounds.columns.names.index("CHANNEL") + 1
            channels = np.asarray(ebounds.data["CHANNEL"], dtype=int)
            tlmin = int(ebounds.header.get(f"TLMIN{col_idx}", int(channels.min())))
            shift = 1 - tlmin
            if shift:
                ebounds.data["CHANNEL"] = channels + shift
                ebounds.header[f"TLMIN{col_idx}"] = 1
                tlmax = ebounds.header.get(f"TLMAX{col_idx}")
                if tlmax is not None:
                    ebounds.header[f"TLMAX{col_idx}"] = int(tlmax) + shift
            hdul.flush()
        with fits.open(out, memmap=False) as hdul:
            names = {hdu.name.upper() for hdu in hdul}
            if "EBOUNDS" not in names or "MATRIX" not in names:
                raise ValueError(f"{out} lacks EBOUNDS/MATRIX extensions: {[h.name for h in hdul]}")
        if return_metadata:
            return out, metadata
        return out
    finally:
        if temporary_rsp2 is not None:
            try:
                temporary_rsp2.unlink()
            except OSError:
                pass


def _rsp2_time_coordinate(rsp2: Any, value: float) -> float:
    """Map absolute or trigger-relative MET to the RSP2 time coordinate."""
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError("response time must be finite")
    low = float(np.min(np.asarray(rsp2.tstart, dtype=float)))
    high = float(np.max(np.asarray(rsp2.tstop, dtype=float)))
    candidates = [numeric]
    trigger = getattr(rsp2, "trigtime", None)
    if trigger is not None and math.isfinite(float(trigger)):
        relative = numeric - float(trigger)
        # Continuous GBM TTE/GTI times are absolute MET (~1e8--1e9 s),
        # whereas GbmRsp2 stores its DRM grid relative to TRIGTIME.  FITS
        # round-off at a response edge can be a few milliseconds, so retain
        # the relative value and let the interval caller clip it safely.
        if abs(numeric) > 1.0e6:
            return float(relative)
        candidates.append(relative)
    inside = [candidate for candidate in candidates if low <= candidate <= high]
    if inside:
        # Prefer the value already expressed in the compact RSP2 coordinate
        # when both candidates happen to lie in range.
        return float(inside[0])
    # A boundary can fall just outside a response due to FITS round-off.  Do
    # not silently select the first/last DRM for a wholly disjoint interval.
    nearest = min(candidates, key=lambda candidate: min(abs(candidate - low), abs(candidate - high)))
    if nearest < low - 1e-6 or nearest > high + 1e-6:
        raise ValueError(
            f"time {numeric} does not overlap RSP2 range [{low}, {high}]"
        )
    return float(np.clip(nearest, low, high))


def _rsp2_exposure_intervals(
    rsp2: Any,
    intervals: Sequence[tuple[float, float]],
) -> tuple[tuple[float, float], ...]:
    """Convert and clip absolute/relative intervals to the RSP2 coordinate."""
    if not intervals:
        return ()
    low = float(np.min(np.asarray(rsp2.tstart, dtype=float)))
    high = float(np.max(np.asarray(rsp2.tstop, dtype=float)))
    result: list[tuple[float, float]] = []
    for left, right in intervals:
        if not math.isfinite(float(left)) or not math.isfinite(float(right)) or float(right) <= float(left):
            raise ValueError("response time_intervals must contain increasing finite pairs")
        start = _rsp2_time_coordinate(rsp2, float(left))
        stop = _rsp2_time_coordinate(rsp2, float(right))
        if stop < start:
            raise ValueError("response time interval changes ordering after time conversion")
        start = max(low, start)
        stop = min(high, stop)
        if stop > start:
            result.append((start, stop))
    result.sort()
    merged: list[list[float]] = []
    for left, right in result:
        if merged and left <= merged[-1][1] + 1e-9:
            merged[-1][1] = max(merged[-1][1], right)
        else:
            merged.append([left, right])
    return tuple((float(left), float(right)) for left, right in merged)


def _rsp2_folded_rate(rsp: Any, *, photon_index: float, energy_band_keV: tuple[float, float] | None) -> float:
    """Fold a unit-normalized power law through a GDT response for comparison."""
    emin, emax = (0.0, math.inf) if energy_band_keV is None else tuple(float(v) for v in energy_band_keV)
    if not 0.0 <= emin < emax:
        raise ValueError("energy_band_keV must be increasing and non-negative")
    channel_low = np.asarray(rsp.ebounds.low_edges(), dtype=float)
    channel_high = np.asarray(rsp.ebounds.high_edges(), dtype=float)
    mask = (channel_low < emax) & (channel_high > emin)
    if not np.any(mask):
        raise ValueError("energy_band_keV does not overlap response channels")

    def powerlaw(params: Sequence[float], energy: np.ndarray) -> np.ndarray:
        return float(params[0]) * np.power(np.asarray(energy, dtype=float), -float(params[1]))

    values = np.asarray(rsp.drm.fold_spectrum(powerlaw, [1.0, photon_index], channel_mask=mask), dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("response folded rate is non-finite")
    return float(np.sum(finite))


def _exposure_weighted_rsp2(rsp2: Any, intervals: Sequence[tuple[float, float]]) -> Any:
    """Construct an exposure-weighted single DRM from an RSP2 object."""
    from gdt.core.data_primitives import ResponseMatrix

    if not intervals:
        raise ValueError("at least one interval is required for a weighted RSP2")
    reference = rsp2[0]
    boundaries = {float(value) for interval in intervals for value in interval}
    boundaries.update(float(value) for value in np.asarray(rsp2.tstart, dtype=float))
    boundaries.update(float(value) for value in np.asarray(rsp2.tstop, dtype=float))
    ordered = sorted(boundaries)
    weighted = np.zeros_like(np.asarray(reference.drm.matrix, dtype=float))
    total = 0.0
    for left, right in zip(ordered[:-1], ordered[1:], strict=True):
        overlap = sum(
            max(0.0, min(right, stop) - max(left, start))
            for start, stop in intervals
        )
        if overlap <= 0:
            continue
        center = 0.5 * (left + right)
        if getattr(rsp2, "num_drms", 1) == 1:
            matrix = np.asarray(reference.drm.matrix, dtype=float)
        else:
            matrix = np.asarray(rsp2.interpolate(center).drm.matrix, dtype=float)
        weighted += matrix * overlap
        total += overlap
    if total <= 0:
        raise ValueError("RSP2 intervals have no overlap with response DRMs")
    weighted /= total
    drm = ResponseMatrix(
        weighted,
        reference.drm.photon_bins.low_edges(),
        reference.drm.photon_bins.high_edges(),
        rsp2.ebounds.low_edges(),
        rsp2.ebounds.high_edges(),
    )
    cls = type(reference)
    trigger = getattr(reference, "trigtime", None)
    start = intervals[0][0] + trigger if trigger is not None else intervals[0][0]
    stop = intervals[-1][1] + trigger if trigger is not None else intervals[-1][1]
    headers = reference.headers.copy() if getattr(reference, "headers", None) is not None else None
    result = cls.from_data(
        drm,
        start_time=float(start),
        stop_time=float(stop),
        trigger_time=trigger,
        headers=headers,
        detector=getattr(reference, "detector", None),
    )
    for attribute in ("_fchan", "_nchan", "_ngrp"):
        if hasattr(reference, attribute):
            setattr(result, attribute, getattr(reference, attribute))
    return result


def _select_rsp2_response(
    rsp2: Any,
    *,
    mid_met: float,
    time_intervals: Sequence[tuple[float, float]] | None,
    energy_band_keV: tuple[float, float] | None,
    photon_index: float,
    midpoint_tolerance: float,
) -> tuple[Any, dict[str, Any]]:
    """Select midpoint or exposure-weighted response and report the gate."""
    midpoint_time = _rsp2_time_coordinate(rsp2, mid_met)
    rsp_low = float(np.min(np.asarray(rsp2.tstart, dtype=float)))
    rsp_high = float(np.max(np.asarray(rsp2.tstop, dtype=float)))
    if midpoint_time < rsp_low - 1e-6 or midpoint_time > rsp_high + 1e-6:
        raise ValueError(
            f"midpoint {mid_met} does not overlap RSP2 range "
            f"[{rsp_low}, {rsp_high}]"
        )
    midpoint_time = float(np.clip(midpoint_time, rsp_low, rsp_high))
    midpoint = rsp2[0] if getattr(rsp2, "num_drms", 1) == 1 else rsp2.interpolate(midpoint_time)
    metadata: dict[str, Any] = {
        "rsp2_time_coordinate": midpoint_time,
        "rsp2_trigger_time": getattr(rsp2, "trigtime", None),
        "rsp2_num_drms": int(getattr(rsp2, "num_drms", 1)),
        "gti_intervals_relative": None,
        "time_weighting": "midpoint_no_gti",
        "midpoint_relative_difference": None,
        "midpoint_approximation_allowed": False,
    }
    if not time_intervals:
        return midpoint, metadata
    intervals = _rsp2_exposure_intervals(rsp2, time_intervals)
    if not intervals:
        raise ValueError("time_intervals do not overlap the RSP2 response")
    weighted = _exposure_weighted_rsp2(rsp2, intervals)
    midpoint_rate = _rsp2_folded_rate(midpoint, photon_index=photon_index, energy_band_keV=energy_band_keV)
    weighted_rate = _rsp2_folded_rate(weighted, photon_index=photon_index, energy_band_keV=energy_band_keV)
    relative_difference = abs(midpoint_rate - weighted_rate) / max(abs(weighted_rate), 1e-30)
    use_midpoint = relative_difference <= midpoint_tolerance
    metadata.update(
        {
            "gti_intervals_relative": [list(item) for item in intervals],
            "gti_exposure_s": float(sum(stop - start for start, stop in intervals)),
            "midpoint_folded_rate": midpoint_rate,
            "weighted_folded_rate": weighted_rate,
            "midpoint_relative_difference": float(relative_difference),
            "midpoint_approximation_allowed": bool(use_midpoint),
            "time_weighting": "midpoint_within_tolerance" if use_midpoint else "exposure_weighted",
        }
    )
    return (midpoint if use_midpoint else weighted), metadata


def _find_ogip_table(
    hdul: Any,
    *,
    required_columns: set[str],
    preferred_names: Sequence[str] = (),
) -> Any:
    """Find an OGIP table by columns instead of relying on extension order."""
    preferred = {str(name).upper() for name in preferred_names}
    fallback = None
    for hdu in hdul:
        data = getattr(hdu, "data", None)
        names = {
            str(name).upper(): name
            for name in (getattr(data, "names", None) or ())
        }
        if not required_columns.issubset(names):
            continue
        if str(getattr(hdu, "name", "")).upper() in preferred:
            return hdu
        if fallback is None:
            fallback = hdu
    if fallback is None:
        joined = ", ".join(sorted(required_columns))
        raise ValueError(f"OGIP product lacks a table with columns: {joined}")
    return fallback


def _find_ogip_ebounds(hdul: Any) -> Any:
    """Find an EBOUNDS table by name, with a column-based compatibility fallback."""
    fallback = None
    for hdu in hdul:
        data = getattr(hdu, "data", None)
        names = {
            str(name).upper(): name
            for name in (getattr(data, "names", None) or ())
        }
        if {"E_MIN", "E_MAX"}.issubset(names):
            if str(getattr(hdu, "name", "")).upper() == "EBOUNDS":
                return hdu
            if fallback is None:
                fallback = hdu
    if fallback is None:
        raise ValueError("OGIP product lacks an EBOUNDS table")
    return fallback


def _ogip_exposure(hdu: Any, hdul: Any) -> float:
    """Read and validate an OGIP exposure, falling back to PRIMARY cards."""
    value = hdu.header.get("EXPOSURE")
    if value is None and len(hdul):
        value = hdul[0].header.get("EXPOSURE")
    try:
        exposure = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("OGIP product lacks a finite positive EXPOSURE") from exc
    if not math.isfinite(exposure) or exposure <= 0:
        raise ValueError("OGIP product EXPOSURE must be finite and positive")
    return exposure


def read_ogip_products(pha_path: str | Path, bak_path: str | Path, rsp_path: str | Path) -> dict[str, Any]:
    """Read one PHA/BAK pair aligned to the response channel grid.

    GDT writes the PHA with every detector channel plus ``QUALITY`` flags for
    the out-of-band channels, while the BAK carries only the in-band channels;
    the official RSP keeps the full detector grid.  The returned arrays are
    restricted to the in-band channels and ``channel_mask`` marks those same
    channels on the response grid so folded templates align.
    """
    from astropy.io import fits

    pha_path = Path(pha_path)
    bak_path = Path(bak_path)
    rsp_path = Path(rsp_path)
    with fits.open(pha_path, memmap=False) as hdul:
        pha = _find_ogip_table(
            hdul,
            required_columns={"COUNTS", "QUALITY"},
            preferred_names=("SPECTRUM", "PHA"),
        )
        exposure = _ogip_exposure(pha, hdul)
        names = {str(name).upper(): name for name in (pha.data.names or ())}
        quality = np.asarray(pha.data[names["QUALITY"]], dtype=int)
        counts = np.asarray(pha.data[names["COUNTS"]], dtype=float)
        if quality.shape != counts.shape:
            raise ValueError("PHA QUALITY and COUNTS must have the same shape")
        if np.any(~np.isfinite(counts)) or np.any(counts < 0):
            raise ValueError("PHA COUNTS must be finite and non-negative")
        ebounds = _find_ogip_ebounds(hdul)
        e_names = {str(name).upper(): name for name in (ebounds.data.names or ())}
        p_lo = np.asarray(ebounds.data[e_names["E_MIN"]], dtype=float)
        p_hi = np.asarray(ebounds.data[e_names["E_MAX"]], dtype=float)
        if p_lo.shape != counts.shape or p_hi.shape != counts.shape:
            raise ValueError("PHA EBOUNDS and COUNTS must have the same shape")
    in_band = quality == 0
    if not np.any(in_band):
        raise ValueError("PHA has no quality==0 in-band channels")
    with fits.open(bak_path, memmap=False) as hdul:
        bak = _find_ogip_table(
            hdul,
            required_columns={"COUNTS", "STAT_ERR"},
            preferred_names=("SPECTRUM", "BACKGROUND", "BAK"),
        )
        bak_exposure = _ogip_exposure(bak, hdul)
        bak_names = {str(name).upper(): name for name in (bak.data.names or ())}
        bak_counts = np.asarray(bak.data[bak_names["COUNTS"]], dtype=float)
        bak_stat = np.asarray(bak.data[bak_names["STAT_ERR"]], dtype=float)
        if bak_counts.shape != bak_stat.shape:
            raise ValueError("BAK COUNTS and STAT_ERR must have the same shape")
        if (
            np.any(~np.isfinite(bak_counts))
            or np.any(bak_counts < 0)
            or np.any(~np.isfinite(bak_stat))
            or np.any(bak_stat < 0)
        ):
            raise ValueError("BAK COUNTS/STAT_ERR must be finite and non-negative")
        bak_sys = (
            np.asarray(bak.data[bak_names["SYS_ERR"]], dtype=float)
            if "SYS_ERR" in bak_names
            else None
        )
        ebounds = _find_ogip_ebounds(hdul)
        e_names = {str(name).upper(): name for name in (ebounds.data.names or ())}
        b_lo = np.asarray(ebounds.data[e_names["E_MIN"]], dtype=float)
        b_hi = np.asarray(ebounds.data[e_names["E_MAX"]], dtype=float)
        if b_lo.shape != bak_counts.shape or b_hi.shape != bak_counts.shape:
            raise ValueError("BAK EBOUNDS and COUNTS must have the same shape")
        background_covariance = _background_covariance_from_hdul(hdul, bak_counts.size)
    # The BAK must represent the source interval itself.  Do not rescale its
    # integrated counts to the PHA exposure: doing so hides boundary/GTI
    # mistakes and changes the fitted background normalization.  The exact
    # interval helper above writes the actual integrated exposure; grossly
    # inconsistent products are rejected below.
    exposure_ratio = bak_exposure / exposure
    if bak_exposure <= 0 or not 0.95 <= exposure_ratio <= 1.05:
        raise ValueError(f"BAK exposure {bak_exposure} grossly inconsistent with PHA {exposure}")
    # FIX(C4): relax tolerance to match float32 precision of FITS EBOUNDS
    # (consistent with PHA-RSP alignment which uses atol=0.01)
    if bak_counts.size != int(in_band.sum()) or not (
        np.allclose(b_lo, p_lo[in_band], rtol=1e-4, atol=0.01)
        and np.allclose(b_hi, p_hi[in_band], rtol=1e-4, atol=0.01)
    ):
        raise ValueError("BAK energy grid does not match the PHA in-band channels")
    # FIX(M5): validate response matrix before returning
    with fits.open(rsp_path, memmap=False) as hdul:
        rsp_ebounds = _find_ogip_ebounds(hdul)
        e_names = {str(name).upper(): name for name in (rsp_ebounds.data.names or ())}
        r_lo = np.asarray(rsp_ebounds.data[e_names["E_MIN"]], dtype=float)
        r_hi = np.asarray(rsp_ebounds.data[e_names["E_MAX"]], dtype=float)
        if r_lo.shape != r_hi.shape or r_lo.ndim != 1 or r_lo.size == 0:
            raise ValueError("RSP EBOUNDS have invalid shape")
        if np.any(~np.isfinite(r_lo)) or np.any(~np.isfinite(r_hi)) or np.any(r_hi <= r_lo):
            raise ValueError("RSP EBOUNDS must be finite increasing intervals")
        detchans = int(rsp_ebounds.header.get("DETCHANS", r_lo.size))
        if detchans != r_lo.size:
            raise ValueError(
                f"RSP DETCHANS={detchans} does not match EBOUNDS channels={r_lo.size}"
            )
        matrix_hdu = _find_ogip_table(
            hdul,
            required_columns={"MATRIX"},
            preferred_names=("MATRIX", "SPECRESP MATRIX", "SPECRESP"),
        )
        matrix_names = {
            str(name).upper(): name for name in (matrix_hdu.data.names or ())
        }
        raw_matrix = matrix_hdu.data[matrix_names["MATRIX"]]
        # Variable-length FITS arrays come back as object arrays of rows.
        try:
            rsp_matrix = np.asarray(raw_matrix, dtype=float)
        except (TypeError, ValueError):
            rsp_matrix = np.asarray(
                [np.asarray(row, dtype=float) for row in raw_matrix],
                dtype=float,
            )
        if rsp_matrix.dtype == object:
            rsp_matrix = np.asarray(
                [np.asarray(row, dtype=float) for row in raw_matrix],
                dtype=float,
            )
        if (
            rsp_matrix.ndim != 2
            or not np.all(np.isfinite(rsp_matrix))
            or np.any(rsp_matrix < 0)
        ):
            raise ValueError(
                "RSP MATRIX must be a finite non-negative two-dimensional array"
            )
        # RMF MATRIX rows are incident-photon energy bins (ENERG_LO/HI),
        # whereas EBOUNDS rows are detector channels.  They are normally
        # different lengths (for example 140 photon bins and 128 GBM
        # channels), so compare the matrix rows with its own energy grid.
        matrix_energy_rows = None
        if {"ENERG_LO", "ENERG_HI"}.issubset(matrix_names):
            matrix_energy_rows = np.asarray(
                matrix_hdu.data[matrix_names["ENERG_LO"]], dtype=float
            )
            matrix_energy_hi = np.asarray(
                matrix_hdu.data[matrix_names["ENERG_HI"]], dtype=float
            )
            if (
                matrix_energy_rows.ndim != 1
                or matrix_energy_hi.shape != matrix_energy_rows.shape
                or np.any(~np.isfinite(matrix_energy_rows))
                or np.any(~np.isfinite(matrix_energy_hi))
                or np.any(matrix_energy_hi <= matrix_energy_rows)
            ):
                raise ValueError("RSP MATRIX ENERG_LO/ENERG_HI grid is invalid")
        if matrix_energy_rows is not None and rsp_matrix.shape[0] != matrix_energy_rows.size:
            raise ValueError(
                "RSP MATRIX rows do not match its photon-energy grid: "
                f"{rsp_matrix.shape[0]} != {matrix_energy_rows.size}"
            )
        if rsp_matrix.shape[1] != detchans:
            raise ValueError(
                "RSP MATRIX detector-channel columns do not match DETCHANS: "
                f"{rsp_matrix.shape[1]} != {detchans}"
            )
        row_sums = rsp_matrix.sum(axis=1)
        if np.any(row_sums == 0):
            n_zero = int(np.sum(row_sums == 0))
            logger.warning("RSP has %d zero-response channels", n_zero)
    mask = np.zeros(detchans, dtype=bool)
    if p_lo.size == r_lo.size and np.allclose(p_lo, r_lo, atol=0.01) and np.allclose(p_hi, r_hi, atol=0.01):
        # Same channel grid up to CSPEC float32 rounding: map by index.
        mask[in_band] = True
    else:
        # FIX(M1): relax per-channel matching — if approximate grid match succeeds,
        # log warning instead of error for ambiguous (multi-hit) channels
        for lo, hi in zip(p_lo[in_band], p_hi[in_band]):
            hits = np.where((r_lo < hi) & (r_hi > lo))[0]
            if hits.size == 1:
                mask[hits[0]] = True
            elif hits.size > 1:
                # Multiple overlapping channels: pick the one with best energy overlap
                overlaps = np.minimum(hi, r_hi[hits]) - np.maximum(lo, r_lo[hits])
                best = hits[int(np.argmax(overlaps))]
                logger.warning(
                    "PHA channel %.1f-%.1f keV matches %d RSP channels; picking best overlap (channel %d)",
                    lo, hi, hits.size, best,
                )
                mask[best] = True
            else:
                raise ValueError(f"response channel for PHA bounds {lo}-{hi} keV not found")
    background_model = bak_counts
    background_sigma = bak_stat
    if background_covariance is None:
        if np.any(background_sigma <= 0):
            background_sigma = np.sqrt(np.clip(background_model, 0.0, None))
            # A zero-count background channel has no Gaussian variance.  Keep
            # a finite floor for the profile engine while recording that this
            # is a fallback, not a measured covariance.
            background_sigma = np.maximum(background_sigma, 1e-12)
            sigma_note = "background_sigma=sqrt(model_counts)"
        else:
            sigma_note = "background_sigma=bak_stat_err_propagated"
        # Keep the ordinary OGIP STAT_ERR path diagonal/analytic.  Building a
        # dense diagonal matrix here needlessly sends every real GBM spectrum
        # through the much slower correlated-background optimizer and can hit
        # its function-evaluation limit for 128 channels.  A full covariance
        # is reserved for a genuinely supplied COVARIANCE extension below.
        covariance = None
    else:
        covariance = background_covariance
        background_sigma = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        background_sigma = np.maximum(background_sigma, 1e-12)
        sigma_note = "background_covariance=BAK covariance extension"
    if bak_sys is not None and background_covariance is None:
        if bak_sys.shape != background_model.shape or np.any(~np.isfinite(bak_sys)) or np.any(bak_sys < 0):
            raise ValueError("BAK SYS_ERR is invalid")
        # Keep the reported per-channel scale synchronized with the final
        # diagonal variance without materializing a dense matrix or passing
        # SYS_ERR a second time downstream.
        background_sigma = np.sqrt(
            np.maximum(background_sigma**2 + (np.abs(background_model) * bak_sys) ** 2, 0.0)
        )
    elif bak_sys is not None:
        # A full COVARIANCE extension is already the supplied Gaussian
        # nuisance covariance.  Adding BAK.SYS_ERR on top would double count
        # a systematics term when the writer has folded it into that matrix.
        if bak_sys.shape != background_model.shape or np.any(~np.isfinite(bak_sys)) or np.any(bak_sys < 0):
            raise ValueError("BAK SYS_ERR is invalid")
    return {
        "counts": counts[in_band],
        "background_model": background_model,
        "background_sigma": background_sigma,
        "background_covariance": covariance,
        "exposure_s": exposure,
        "background_exposure_s": bak_exposure,
        "channel_mask": mask,
        "sigma_note": sigma_note,
        "background_error_source": (
            "BAK.STAT_ERR"
            if bak_sys is None
            else "BAK covariance extension (BAK.SYS_ERR not added)"
            if background_covariance is not None
            else "BAK.STAT_ERR+BAK.SYS_ERR"
        ),
        "background_covariance_source": (
            "BAK covariance extension"
            if background_covariance is not None
            else "BAK.STAT_ERR diagonal"
        ),
        "systematic_error_applied": bool(
            bak_sys is not None and background_covariance is None
        ),
        "systematic_error_ignored_with_covariance": bool(
            bak_sys is not None and background_covariance is not None
        ),
    }


def _background_covariance_from_hdul(hdul: Any, channels: int) -> np.ndarray | None:
    """Read an optional channel covariance matrix from a BAK FITS file.

    OGIP background files usually carry only ``STAT_ERR``.  Some calibrated
    products additionally provide a ``COVARIANCE``/``COVAR`` extension or a
    matrix-valued table column.  When present it replaces the diagonal
    approximation; callers must not add ``STAT_ERR`` again.
    """
    candidates: list[np.ndarray] = []
    extension_names = {"COVARIANCE", "COVAR", "COV_MATRIX", "BACKGROUND_COVARIANCE"}
    for hdu in hdul:
        data = getattr(hdu, "data", None)
        if data is None:
            continue
        extname = str(getattr(hdu, "name", "")).upper()
        names = {str(name).upper(): name for name in (getattr(data, "names", None) or ())}
        for key in ("COVARIANCE", "COVAR", "COV_MATRIX", "MATRIX"):
            actual = names.get(key)
            if actual is None:
                continue
            try:
                values = np.asarray(data[actual], dtype=float)
            except (TypeError, ValueError):
                continue
            if values.ndim == 2 and values.shape == (channels, channels):
                candidates.append(values)
            elif values.ndim == 1 and values.size == channels * channels:
                candidates.append(values.reshape(channels, channels))
        if extname in extension_names:
            try:
                values = np.asarray(data, dtype=float)
            except (TypeError, ValueError):
                values = np.asarray([], dtype=float)
            if values.ndim == 2 and values.shape == (channels, channels):
                candidates.append(values)
    if not candidates:
        return None
    covariance = np.asarray(candidates[0], dtype=float)
    if np.any(~np.isfinite(covariance)) or not np.allclose(
        covariance, covariance.T, rtol=1e-8, atol=1e-12
    ):
        raise ValueError("BAK covariance must be finite and symmetric")
    try:
        np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as exc:
        raise ValueError("BAK covariance must be positive definite") from exc
    return covariance


# FIX(M4): stable CSV column order with fixed fields first
_FIXED_CSV_FIELDS = (
    "target_id", "source_name", "geometry_status", "coverage_fraction",
    "start_utc", "stop_utc", "segment_met", "nai_detectors", "bgo_detectors",
    "model", "gamma_primary", "response_backend", "analysis_status",
    "science_result", "flux_erg_cm2_s", "upper_limit_erg_cm2_s",
    "quality_flags",
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(_FIXED_CSV_FIELDS)
    extra = sorted({key for row in rows for key in row} - set(fields))
    fields.extend(extra)
    # Only include fields that actually exist in the data
    fields = [f for f in fields if any(f in row for row in rows)]
    if not fields:
        raise ValueError(f"Refusing to write {path}: no rows, so no header columns can be derived")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _official_response_available(executable: str) -> bool:
    return shutil.which(executable) is not None


def _drm_gen_available() -> tuple[bool, str | None]:
    """Probe the pure-Python response stack without requiring it."""
    try:
        import gbm_drm_gen  # noqa: F401
        import gbmgeometry  # noqa: F401
        import responsum  # noqa: F401
    except ImportError as exc:
        return False, f"gbm_drm_gen stack not importable: {exc}"
    if not os.environ.get("BALROG_DB"):
        return False, "BALROG_DB environment variable is not set"
    return True, None


def _resolve_response_backend(analysis: GBMAnalysisConfig) -> tuple[str | None, str]:
    """Pick an available response backend under the configured policy."""
    official_ok = _official_response_available(analysis.response_generator_executable)
    drm_ok, drm_note = _drm_gen_available()
    backend = analysis.response_backend
    if backend == "official":
        if official_ok:
            return "official", "official SA_GBM_RSP_Gen.pl selected"
        return None, f"official response generator unavailable: {analysis.response_generator_executable!r} not on PATH"
    if backend == "gbm_drm_gen":
        if drm_ok:
            return "gbm_drm_gen", "pure-Python gbm_drm_gen selected"
        return None, f"gbm_drm_gen unavailable: {drm_note}"
    # auto: prefer the official generator, degrade to the pure-Python stack.
    if official_ok:
        return "official", "official SA_GBM_RSP_Gen.pl selected"
    if drm_ok:
        return "gbm_drm_gen", "auto-degraded to pure-Python gbm_drm_gen"
    return None, f"no response generator available (official missing; gbm_drm_gen: {drm_note})"


@dataclass(frozen=True)
class GBMPipelineInput(PipelineInput):
    """One sky position and UTC interval for the GBM continuous pipeline.

    ``root`` is the read-only search root for already-downloaded GBM products
    (for example a shared Fermi archive cache); new downloads always go to
    ``<output>/cache``.  ``download=False`` keeps the pipeline strictly local.
    """

    source_name: str = ""
    ra_deg: float = float("nan")
    dec_deg: float = float("nan")
    start_utc: str = ""
    stop_utc: str = ""
    download: bool = False
    source_windows_met: tuple[tuple[float, float], ...] | None = None
    background_windows_met: tuple[tuple[float, float], ...] | None = None
    poshist_paths: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        source = str(self.source_name).strip()
        if not source:
            raise ValueError("GBMPipelineInput source_name must not be empty")
        ra = float(self.ra_deg)
        dec = float(self.dec_deg)
        if not math.isfinite(ra) or not 0.0 <= ra < 360.0:
            raise ValueError("GBMPipelineInput ra_deg must be in [0, 360)")
        if not math.isfinite(dec) or not -90.0 <= dec <= 90.0:
            raise ValueError("GBMPipelineInput dec_deg must be in [-90, 90]")
        # Normalize through GBMFlareInterval so the manifest fingerprint stores
        # canonical UTC strings instead of opaque astropy objects.
        interval = GBMFlareInterval.from_utc(
            source_name=source,
            ra_deg=ra,
            dec_deg=dec,
            start_utc=str(self.start_utc),
            stop_utc=str(self.stop_utc),
        )
        object.__setattr__(self, "source_name", source)
        object.__setattr__(self, "ra_deg", ra)
        object.__setattr__(self, "dec_deg", dec)
        object.__setattr__(self, "start_utc", interval.start.isot)
        object.__setattr__(self, "stop_utc", interval.stop.isot)
        for name in ("source_windows_met", "background_windows_met"):
            value = getattr(self, name)
            if value is not None:
                checked = tuple((float(left), float(right)) for left, right in value)
                if not checked:
                    raise ValueError(f"GBMPipelineInput {name} must not be empty when given")
                for left, right in checked:
                    if not math.isfinite(left) or not math.isfinite(right) or not left < right:
                        raise ValueError(f"GBMPipelineInput {name} must contain finite increasing MET pairs")
                object.__setattr__(self, name, checked)
        object.__setattr__(self, "poshist_paths", tuple(str(path) for path in self.poshist_paths))

    # FIX(M13): cache interval object to avoid repeated astropy Time/SkyCoord construction
    @functools.cached_property
    def interval(self) -> GBMFlareInterval:
        return GBMFlareInterval.from_utc(
            source_name=self.source_name,
            ra_deg=self.ra_deg,
            dec_deg=self.dec_deg,
            start_utc=self.start_utc,
            stop_utc=self.stop_utc,
        )


@dataclass(frozen=True, slots=True)
class GBMPipelineResult:
    """Public outcome of one GBM pipeline run, including partial reviews."""

    target_id: str
    source_name: str
    science_status: str
    science_result: str
    message: str | None = None
    stages: Mapping[str, str] = field(default_factory=dict)
    products: Mapping[str, str] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()


@register_pipeline("fermi.gbm")
class GBMPipeline(InstrumentPipeline["GBMPipelineInput", GBMPipelineResult]):
    """Resumable GBM continuous-data pipeline for one sky target.

    Stages run local-first: ``download=False`` never touches the network and
    every gate that cannot be passed scientifically stops the run with
    ``needs_review`` instead of fabricating a flux or upper limit.

    Stage data contracts (keys produced in ``StageResult.data``):
      coverage:  {status, segments_met, coverage_fraction, covered_exposure_s,
                  detector_angles_deg, poshist_path, tte_gti_applied, cadence_s, reasons}
      detectors: {all_detectors, nai, bgo}  (GBMDetectorSelection dict)
      download:  {tte: {det: [path]}, cspec: {det: [path]}, poshist: path}
      windows:   {segment_met, source_explicit, per_detector: {det: {background_windows_met, flags}}, warnings}
      spectra:   {per_detector: {det: {pha, bak, kind, band_keV}}}
      response:  {rsp: {det: path}, response_backend}
      fit:       {science_result, fits: {nai|bgo: [{gamma, ...}]}, fractional_systematics,
                  significance: {nai, bgo, primary, threshold_sigma}, warnings}
    """

    stages: ClassVar[tuple[PipelineStage, ...]] = (
        PipelineStage("preflight"),
        PipelineStage("coverage", ("preflight",)),
        PipelineStage("detectors", ("coverage",)),
        PipelineStage("download", ("coverage", "detectors")),
        PipelineStage("windows", ("coverage", "download")),
        PipelineStage("lightcurve", ("download", "windows")),
        PipelineStage("spectra", ("download", "windows")),
        PipelineStage("response", ("detectors", "download", "spectra")),
        PipelineStage("fit", ("spectra", "response")),
        PipelineStage("report", ("coverage", "windows", "fit")),
    )

    def __init__(self, input_data: GBMPipelineInput, *, config: Any):
        super().__init__(input_data, config=config)
        self._stage_handlers = {
            "preflight": self._stage_preflight,
            "coverage": self._stage_coverage,
            "detectors": self._stage_detectors,
            "download": self._stage_download,
            "windows": self._stage_windows,
            "lightcurve": self._stage_lightcurve,
            "spectra": self._stage_spectra,
            "response": self._stage_response,
            "fit": self._stage_fit,
            "report": self._stage_report,
        }

    # -- framework hooks ----------------------------------------------------

    def validate_input(self) -> None:
        # GBMPipelineInput.__post_init__ already validates coordinates, times
        # and windows; rebuild the interval to fail fast on any drift.
        _ = self.input.interval

    def include_config_in_input_fingerprint(self) -> bool:
        """Keep GBM preparation caches independent from fit-only options."""
        return False

    def stage_config_dependencies(self, stage: PipelineStage) -> Any:
        config = self.config
        common = {
            "instrument": (config.name, config.pipeline),
            "execution": config.execution,
        }
        analysis = getattr(config, "gbm_analysis", None)
        if stage.name in {"preflight", "coverage", "detectors", "download"}:
            return {**common, "gbm_analysis": analysis}
        if stage.name in {"windows", "lightcurve", "spectra"}:
            return {**common, "gbm_analysis": analysis}
        if stage.name == "response":
            return {**common, "gbm_analysis": analysis, "spectrum": config.spectrum}
        if stage.name == "fit":
            return {
                **common,
                "gbm_analysis": analysis,
                "spectrum": config.spectrum,
                "fitting": config.fitting,
                "upper_limit": config.upper_limit,
            }
        if stage.name == "report":
            return {
                **common,
                "gbm_analysis": analysis,
                "spectrum": config.spectrum,
                "fitting": config.fitting,
                "upper_limit": config.upper_limit,
                "reporting": config.reporting,
            }
        return config

    def execute_stage(
        self,
        stage: PipelineStage,
        context: Mapping[str, StageResult],
    ) -> StageResult:
        handler = self._stage_handlers.get(stage.name)
        if handler is None:
            raise RuntimeError(f"Unknown GBM pipeline stage {stage.name!r}")
        logger.info("GBM pipeline stage %s: starting", stage.name)
        result = handler(context)
        logger.info("GBM pipeline stage %s: %s", stage.name, result.status.value)
        return result

    def build_result(self, context: Mapping[str, StageResult]) -> GBMPipelineResult:
        stage_status = {stage.name: "pending" for stage in self.stages}
        products: dict[str, str] = {}
        warnings_list: list[str] = []
        review_messages: list[str] = []
        for name, result in context.items():
            stage_status[name] = result.status.value
            products.update(result.outputs)
            for flag in result.data.get("warnings", ()) if isinstance(result.data, dict) else ():
                warnings_list.append(str(flag))
            if result.status == PipelineStatus.NEEDS_REVIEW and result.message:
                review_messages.append(result.message)
        fit_result = context.get("fit")
        science_result = "not_computed"
        if fit_result is not None and fit_result.status == PipelineStatus.COMPLETED:
            science_result = str(fit_result.data.get("science_result", "not_computed"))
        executed = {name for name, status in stage_status.items() if status != "pending"}
        all_names = {stage.name for stage in self.stages}
        fit_needs_review = bool(
            fit_result is not None
            and isinstance(fit_result.data, Mapping)
            and fit_result.data.get("analysis_status") == "needs_review"
        )
        report_result = context.get("report")
        report_needs_review = bool(
            report_result is not None
            and isinstance(report_result.data, Mapping)
            and report_result.data.get("analysis_status") == "needs_review"
        )
        if review_messages:
            science_status = "needs_review"
            message = "; ".join(review_messages)
        elif fit_needs_review or report_needs_review:
            science_status = "needs_review"
            message = "background residual validation requires review"
        elif executed == all_names and all(
            context[name].status == PipelineStatus.COMPLETED for name in all_names
        ):
            science_status = "complete"
            message = None
        elif any(context[name].status == PipelineStatus.FAILED for name in executed):
            science_status = "failed"
            message = "one or more stages failed"
        else:
            science_status = "partial"
            message = "pipeline stopped before the final stage"
        return GBMPipelineResult(
            target_id=self.input.target_id,
            source_name=self.input.source_name,
            science_status=science_status,
            science_result=science_result,
            message=message,
            stages=stage_status,
            products=products,
            warnings=tuple(sorted(set(warnings_list))),
        )

    # -- shared helpers ------------------------------------------------------

    @property
    def _analysis(self) -> GBMAnalysisConfig:
        analysis = getattr(self.config, "gbm_analysis", None)
        return analysis if isinstance(analysis, GBMAnalysisConfig) else GBMAnalysisConfig()

    @property
    def _interval(self) -> GBMFlareInterval:
        return self.input.interval

    @property
    def _cache_dir(self) -> Path:
        return self.workspace / "cache"

    def _search_roots(self) -> list[Path]:
        roots = [self.input.resolved_root(), self._cache_dir]
        return [root for root in roots if root.is_dir()]

    def _coverage_from_payload(self, payload: Mapping[str, Any]) -> GBMCoverageResult:
        return GBMCoverageResult(
            status=payload["status"],
            interval=self._interval,
            coverage_fraction=float(payload["coverage_fraction"]),
            covered_exposure_s=float(payload["covered_exposure_s"]),
            segments_met=tuple(tuple(segment) for segment in payload["segments_met"]),
            reasons=tuple(payload["reasons"]),
            detector_angles_deg=dict(payload["detector_angles_deg"]),
            poshist_path=payload.get("poshist_path"),
            tte_gti_applied=bool(payload["tte_gti_applied"]),
            cadence_s=payload.get("cadence_s"),
        )

    def _write_json(self, path: Path, payload: Mapping[str, Any]) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        # FIX(M10): atomic write to prevent corrupt JSON on interruption
        import tempfile as _tempfile
        fd, tmp = _tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2)
            Path(tmp).replace(path)
        except BaseException:
            os.unlink(tmp)
            raise
        return path

    # -- stages --------------------------------------------------------------

    def _stage_preflight(self, context: Mapping[str, StageResult]) -> StageResult:
        analysis = self._analysis
        perl = shutil.which("perl")
        perl_module: bool | None = None
        if perl is not None:
            # FIX(M11): add timeout to prevent infinite hang if Perl/loader is broken
            process = subprocess.run(
                [perl, "-MAstro::FITS::CFITSIO", "-e", "1"],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
            perl_module = process.returncode == 0
        drm_ok, drm_note = _drm_gen_available()
        try:
            import xspec  # noqa: F401

            xspec_available = True
        except ImportError:
            xspec_available = False
        backend, backend_note = _resolve_response_backend(analysis)
        report = {
            "checks": {
                "gdt_data": shutil.which("gdt-data"),
                "response_generator": shutil.which(analysis.response_generator_executable),
                "perl": perl,
                "perl_astro_fits_cfitsio": perl_module,
                "gbm_drm_gen_available": drm_ok,
                "gbm_drm_gen_note": drm_note,
                "balrog_db": os.environ.get("BALROG_DB"),
                "xspec_available": xspec_available,
            },
            "response_backend_policy": analysis.response_backend,
            "resolved_backend": backend,
            "backend_note": backend_note,
        }
        path = self._write_json(self.workspace / "preflight.json", report)
        return StageResult(
            outputs={"preflight": str(path)},
            data={"resolved_backend": backend, "backend_note": backend_note},
        )

    def _poshist_day_stamps(self, interval: GBMFlareInterval) -> list[str]:
        """UTC day stamps for every day the interval spans (YYMMDD)."""
        day = interval.start.datetime.replace(hour=0, minute=0, second=0, microsecond=0)
        last_day = interval.stop.datetime.date()
        stamps: list[str] = []
        while day.date() <= last_day:
            stamps.append(day.strftime("%y%m%d"))
            day += timedelta(days=1)
        return stamps

    def _find_poshist_files(self, interval: GBMFlareInterval) -> list[Path]:
        """Readable poshist candidates for every UTC day the interval spans."""
        candidates: dict[Path, None] = {}
        for stamp in self._poshist_day_stamps(interval):
            patterns = (
                f"glg_poshist_all_{stamp}_v*.fit",
                f"glg_poshist_all_{stamp}_v*.fit.gz",
            )
            for root in self._search_roots():
                for pattern in patterns:
                    for path in sorted(root.rglob(pattern)):
                        if _is_valid_fits(path):
                            candidates.setdefault(path.resolve(), None)
        return list(candidates)

    def _select_covering_poshist(
        self, interval: GBMFlareInterval, paths: Iterable[Path]
    ) -> tuple[Path | None, GBMCoverageResult | None]:
        """First poshist whose time range actually covers the interval.

        A file merely named for the start day may not cover the interval
        (intervals crossing UTC midnight need the following day's file), so
        each candidate is verified with the real coverage check.
        """
        best_path: Path | None = None
        best_coverage: GBMCoverageResult | None = None
        for path in paths:
            coverage = check_gbm_coverage(interval, path)
            if coverage.status != "data_missing":
                return path, coverage
            if best_coverage is None:
                best_path, best_coverage = path, coverage
        return best_path, best_coverage

    def _stage_coverage(self, context: Mapping[str, StageResult]) -> StageResult:
        interval = self._interval
        candidates = [Path(path).expanduser() for path in self.input.poshist_paths]
        if not candidates:
            candidates = self._find_poshist_files(interval)
        poshist_path, coverage = self._select_covering_poshist(interval, candidates)
        if (coverage is None or coverage.status == "data_missing") and self.input.download:
            # No usable local file, or the local file does not cover the
            # interval: fetch the position history for the interval's days.
            manifest = fetch_gbm_continuous_products(
                interval,
                destination=self._cache_dir,
                products=("poshist",),
            )
            fetched_path, fetched_coverage = self._select_covering_poshist(
                interval, manifest.poshist_paths
            )
            if fetched_coverage is not None and fetched_coverage.status != "data_missing":
                poshist_path, coverage = fetched_path, fetched_coverage
        if coverage is None:
            coverage = check_gbm_coverage(interval, None)
        payload = coverage.to_dict()
        out = self._write_json(self.workspace / "coverage.json", payload)
        if coverage.status in {"none", "data_missing"}:
            reasons = ", ".join(coverage.reasons) or "no visibility"
            return StageResult(
                PipelineStatus.NEEDS_REVIEW,
                outputs={"coverage": str(out)},
                data=payload,
                message=f"GBM coverage {coverage.status}: {reasons}",
            )
        return StageResult(outputs={"coverage": str(out)}, data=payload)

    def _stage_detectors(self, context: Mapping[str, StageResult]) -> StageResult:
        coverage = self._coverage_from_payload(context["coverage"].data)
        analysis = self._analysis
        selection = select_gbm_detectors(
            coverage,
            max_nai_angle=analysis.max_nai_angle_deg,
            max_nai=analysis.max_nai_detectors,
            max_bgo_angle=analysis.max_bgo_angle_deg,
            max_bgo=analysis.max_bgo_detectors,
        )
        payload = selection.to_dict()
        out = self._write_json(self.workspace / "detector_selection.json", payload)
        # FIX(C3): empty detector list would crash downstream stages
        if not selection.all_detectors:
            return StageResult(
                PipelineStatus.NEEDS_REVIEW,
                outputs={"detector_selection": str(out)},
                data=payload,
                message="E014: no detector passes angle thresholds",
            )
        return StageResult(outputs={"detector_selection": str(out)}, data=payload)

    def _find_product_files(
        self, kind: str, detector: str, roots: Sequence[Path]
    ) -> list[Path]:
        found: list[Path] = []
        seen: set[Path] = set()
        for root in roots:
            for pattern in (
                f"glg_{kind}_{detector}_*.fit",
                f"glg_{kind}_{detector}_*.fit.gz",
            ):
                for path in sorted(root.rglob(pattern)):
                    resolved = path.resolve()
                    if resolved in seen or not path.is_file() or path.stat().st_size < 2880:
                        continue
                    seen.add(resolved)
                    found.append(resolved)
        return found

    def _stage_download(self, context: Mapping[str, StageResult]) -> StageResult:
        analysis = self._analysis
        detectors = tuple(context["detectors"].data["all_detectors"])
        interval = self._interval
        # Fetch far enough to span the largest allowed background windows.
        context_s = analysis.background_context_max_s + analysis.background_guard_min_s

        def scan() -> tuple[dict[str, list[Path]], dict[str, list[Path]]]:
            roots = self._search_roots()
            tte = {det: self._find_product_files("tte", det, roots) for det in detectors}
            cspec = {det: self._find_product_files("cspec", det, roots) for det in detectors}
            return tte, cspec

        tte, cspec = scan()
        missing = [det for det in detectors if not tte[det] or not cspec[det]]
        if missing and self.input.download:
            fetch_gbm_continuous_products(
                interval,
                destination=self._cache_dir,
                detectors=detectors,
                products=("tte", "cspec", "poshist"),
                context_s=context_s,
            )
            tte, cspec = scan()
            missing = [det for det in detectors if not tte[det] or not cspec[det]]
        payload = {
            "tte": {det: [str(path) for path in tte[det]] for det in detectors},
            "cspec": {det: [str(path) for path in cspec[det]] for det in detectors},
            "poshist_path": context["coverage"].data.get("poshist_path"),
        }
        manifest_path = self._write_json(self.workspace / "data_manifest.json", payload)
        if missing:
            return StageResult(
                PipelineStatus.NEEDS_REVIEW,
                outputs={"data_manifest": str(manifest_path)},
                data=payload,
                message=(
                    f"GBM products incomplete for detectors {missing}; provide data "
                    f"under {self.input.resolved_root()} or rerun with --download"
                ),
            )
        # TODO(H10): TTE files are loaded independently in _stage_lightcurve, _stage_spectra,
        # and _stage_windows. Consider pre-loading and caching TTE objects in StageResult.data
        # during _stage_download to reduce peak memory usage for multi-detector long-duration data.
        return StageResult(outputs={"data_manifest": str(manifest_path)}, data=payload)

    def _stage_windows(self, context: Mapping[str, StageResult]) -> StageResult:
        analysis = self._analysis
        segments = [tuple(segment) for segment in context["coverage"].data["segments_met"]]
        if self.input.source_windows_met is not None:
            if len(self.input.source_windows_met) != 1:
                return StageResult(
                    PipelineStatus.NEEDS_REVIEW,
                    message="exactly one source window is supported per run",
                )
            seg = self.input.source_windows_met[0]
            source_explicit = True
        else:
            if len(segments) != 1:
                return StageResult(
                    PipelineStatus.NEEDS_REVIEW,
                    message=(
                        f"expected one continuous coverage segment, found {len(segments)}; "
                        "pass an explicit --window to select one"
                    ),
                )
            seg = segments[0]
            source_explicit = False
        detectors = tuple(context["detectors"].data["all_detectors"])
        tte_by_det = context["download"].data["tte"]
        per_detector: dict[str, dict[str, Any]] = {}
        blocked: list[str] = []
        all_flags: list[str] = []
        for det in detectors:
            if self.input.background_windows_met is not None:
                windows: list[tuple[float, float]] = list(self.input.background_windows_met)
                flags: list[str] = []
            else:
                coverage_range = tte_coverage(tte_by_det[det])
                windows, flags = background_windows(
                    seg,
                    coverage_range,
                    guard_s=analysis.background_guard_min_s,
                    context_min_s=analysis.background_context_min_s,
                    context_max_s=analysis.background_context_max_s,
                )
            all_flags.extend(f"det_{det}:{flag}" for flag in flags)
            if not windows:
                blocked.append(f"det_{det}:no_usable_background_window")
                continue
            # The fitted background is integrated over the source segment from
            # the PHAII envelope spanned by the windows, so the windows must
            # bracket the source.  One-sided backgrounds across a data gap
            # (e.g. SAA) are never extrapolated silently.
            # FIX(C1): strict inequality avoids false E013 when source window is adjacent to context window
            bracketed = any(w[1] < seg[0] for w in windows) and any(w[0] > seg[1] for w in windows)
            if not bracketed:
                # FIX(M12): distinguish auto-selected vs user-provided background windows
                origin = "user-provided" if self.input.background_windows_met is not None else "auto-selected"
                blocked.append(f"det_{det}:no_bracketing_background({origin})")
            per_detector[det] = {
                "background_windows_met": [list(window) for window in windows],
                "flags": flags,
            }
        payload = {
            "segment_met": list(seg),
            "source_explicit": source_explicit,
            "per_detector": per_detector,
            "warnings": tuple(all_flags),
        }
        out = self._write_json(self.workspace / "windows.json", payload)
        if blocked:
            return StageResult(
                PipelineStatus.NEEDS_REVIEW,
                outputs={"windows": str(out)},
                data=payload,
                message="background window gate failed: " + "; ".join(blocked),
            )
        return StageResult(outputs={"windows": str(out)}, data=payload)

    def _stage_lightcurve(self, context: Mapping[str, StageResult]) -> StageResult:
        analysis = self._analysis
        seg = tuple(context["windows"].data["segment_met"])
        selection = context["detectors"].data
        tte_by_det = context["download"].data["tte"]
        out_dir = self.workspace / "lightcurve"
        out_dir.mkdir(parents=True, exist_ok=True)
        warnings_list: list[str] = []
        try:
            from gdt.core.binning.unbinned import bin_by_time
            from gdt.missions.fermi.gbm.tte import GbmTte
        except ImportError as exc:
            warnings_list.append(f"gdt_unavailable:{exc}")
            return StageResult(
                outputs={"lightcurve_dir": str(out_dir)},
                data={"warnings": tuple(warnings_list)},
            )
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        for det in tuple(selection["all_detectors"]):
            band = analysis.nai_band_keV if det in selection["nai"] else analysis.bgo_band_keV
            npz_path = out_dir / f"gbm_{det}_lightcurve.npz"
            png_path = out_dir / f"gbm_{det}_lightcurve.png"
            try:
                ttes = [GbmTte.open(path) for path in tte_by_det[det]]
                tte = ttes[0] if len(ttes) == 1 else GbmTte.merge(ttes, force_unique=False)
                lightcurve = tte.to_lightcurve(
                    bin_by_time,
                    analysis.lc_bin_s,
                    time_range=seg,
                    energy_range=band,
                )
                times = np.asarray(lightcurve.times, dtype=float)
                rates = np.asarray(lightcurve.rates, dtype=float)
                rate_errors = np.asarray(lightcurve.rate_errors, dtype=float)
                np.savez(
                    npz_path,
                    time=times,
                    rate=rates,
                    rate_err=rate_errors,
                    tstart=float(seg[0]),
                    tstop=float(seg[1]),
                    energy_range_keV=np.asarray(band, dtype=float),
                    bin_s=float(analysis.lc_bin_s),
                    detector=det,
                )
                figure, axis = plt.subplots(figsize=(8.0, 4.0), dpi=150)
                try:
                    offset = times - seg[0]
                    axis.errorbar(offset, rates, yerr=rate_errors, fmt=".", markersize=3, lw=0.8)
                    axis.axvspan(0.0, seg[1] - seg[0], color="tab:red", alpha=0.15, label="source window")
                    axis.set_xlabel(f"time since segment start (s; MET {seg[0]:.1f})")
                    axis.set_ylabel(f"count rate (cts/s; {band[0]:g}-{band[1]:g} keV)")
                    axis.set_title(f"GBM {det} lightcurve — {self.input.source_name}")
                    axis.legend(loc="best", fontsize=8)
                    figure.tight_layout()
                    figure.savefig(png_path)
                finally:
                    plt.close(figure)
            # FIX(H1): propagate system-level errors; only swallow data/analysis errors
            except (MemoryError, OSError) as exc:
                raise
            except Exception as exc:
                import traceback
                logger.warning("GBM lightcurve failed for %s:\n%s", det, traceback.format_exc())
                warnings_list.append(f"det_{det}:lightcurve_failed:{exc}")
        return StageResult(
            outputs={"lightcurve_dir": str(out_dir)},
            data={"warnings": tuple(warnings_list)},
        )

    def _stage_spectra(self, context: Mapping[str, StageResult]) -> StageResult:
        analysis = self._analysis
        _ensure_gdt_polynomial_basis()
        windows_data = context["windows"].data
        seg = tuple(windows_data["segment_met"])
        selection = context["detectors"].data
        tte_by_det = context["download"].data["tte"]
        spectra_dir = self.workspace / "spectra"
        spectra_dir.mkdir(parents=True, exist_ok=True)
        from astropy.io import fits

        per_detector: dict[str, dict[str, Any]] = {}
        warnings_list: list[str] = []
        for det in tuple(selection["all_detectors"]):
            kind = "nai" if det in selection["nai"] else "bgo"
            band = analysis.nai_band_keV if kind == "nai" else analysis.bgo_band_keV
            windows = [
                tuple(window)
                for window in windows_data["per_detector"][det]["background_windows_met"]
            ]
            try:
                products = extract_gbm_spectral_products(
                    tte_by_det[det],
                    detector=det,
                    source_ranges_met=[seg],
                    background_ranges_met=windows,
                    energy_range_keV=band,
                    output_dir=spectra_dir,
                )
                with fits.open(products.source_pha, memmap=False) as hdul:
                    pha_hdu = _find_ogip_table(
                        hdul,
                        required_columns={"COUNTS"},
                        preferred_names=("SPECTRUM", "PHA"),
                    )
                    pha_names = {
                        str(name).upper(): name
                        for name in (pha_hdu.data.names or ())
                    }
                    counts_total = float(
                        np.sum(np.asarray(pha_hdu.data[pha_names["COUNTS"]], dtype=float))
                    )
            except Exception as exc:
                warnings_list.append(
                    f"det_{det}:spectrum_failed:{type(exc).__name__}:{exc}"
                )
                logger.warning("detector %s spectrum extraction failed: %s", det, exc)
                continue
            # Zero source counts are a valid Poisson non-detection. Preserve the
            # PHA/BAK pair so the response-aware upper-limit stage can use it;
            # attach a review flag instead of discarding the detector.
            if counts_total <= 0.0:
                warnings_list.append(f"det_{det}:zero_source_counts")
                logger.warning("detector %s: zero source counts; retaining for upper limit", det)
            per_detector[det] = {
                "kind": kind,
                "band_keV": list(band),
                "pha": str(products.source_pha),
                "bak": str(products.background_bak),
                "background": {
                    "polynomial_order": products.background.polynomial_order,
                    "aicc": products.background.aicc,
                    "bin_size_s": products.background.bin_size_s,
                    "n_parameters": products.background.n_parameters,
                    "n_observations": products.background.n_observations,
                    "model_selection": products.background.model_selection,
                    "residual_validation": products.background.residual_validation,
                    "candidate_scores": dict(products.background.candidate_scores),
                    "candidate_standard_errors": dict(
                        products.background.candidate_standard_errors
                    ),
                    "qualified_orders": list(products.background.qualified_orders),
                },
                "background_variants": {
                    str(order): str(path)
                    for order, path in products.background_variants.items()
                },
                "source_counts_total": counts_total,
                "zero_source_counts": bool(counts_total <= 0.0),
            }
        # Only a complete extraction failure prevents the response/fitting
        # stages. An all-zero but structurally valid PHA remains useful for a
        # non-detection upper bound and carries its warning in the manifest.
        if not per_detector:
            return StageResult(
                PipelineStatus.NEEDS_REVIEW,
                data={"warnings": tuple(warnings_list)},
                message="no detector produced a valid source/background spectrum",
            )
        payload = {"segment_met": list(seg), "per_detector": per_detector, "warnings": tuple(warnings_list)}
        out = self._write_json(self.workspace / "spectra.json", payload)
        return StageResult(outputs={"spectra": str(out), "spectra_dir": str(spectra_dir)}, data=payload)

    def _generate_responses_official(
        self,
        context: Mapping[str, StageResult],
        seg: tuple[float, float],
        detectors: tuple[str, ...],
        workdir: Path,
        analysis: GBMAnalysisConfig,
    ) -> dict[str, Path]:
        """Run the official generator once and map detectors to RSP files."""
        from .response import generate_gbm_response

        download_data = context["download"].data
        cspec_paths: list[Path] = []
        for det in detectors:
            paths = [Path(path) for path in download_data["cspec"][det]]
            if not paths:
                raise FileNotFoundError(f"no CSPEC files for detector {det}")
            cspec_paths.extend(paths)
        poshist_path = download_data.get("poshist_path")
        if not poshist_path:
            raise FileNotFoundError("no poshist recorded for response generation")
        # FIX(H16): deduplicate CSPEC paths to avoid redundant symlinks
        for path in set(cspec_paths):
            link = workdir / Path(path).name
            if not link.exists():
                link.symlink_to(Path(path).resolve())
        link = workdir / Path(poshist_path).name
        if not link.exists():
            link.symlink_to(Path(poshist_path).resolve())
        # A previously failed attempt may have left responses behind; the
        # wrapper tracks new outputs by directory diff, so start clean.
        for stale in workdir.glob("*.rsp*"):
            stale.unlink()
        generate_gbm_response(
            ra_deg=self.input.ra_deg,
            dec_deg=self.input.dec_deg,
            start_met=seg[0],
            stop_met=seg[1],
            detectors=list(detectors),
            workdir=workdir,
            executable=analysis.response_generator_executable,
            timeout_s=analysis.response_timeout_s,
        )
        mapping: dict[str, Path] = {}
        for path in sorted(workdir.glob("*.rsp*")):
            detector = _detector_of(path.name.replace(".gz", ""))
            if detector:
                mapping.setdefault(detector, path)
        missing = [det for det in detectors if det not in mapping]
        if missing:
            raise RuntimeError(f"response generator produced no RSP for {missing}")
        return mapping

    def _tte_covering(self, paths: Sequence[str], met: float) -> Path:
        """The TTE file whose time range contains ``met``."""
        from gdt.missions.fermi.gbm.tte import GbmTte

        for path in paths:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tte = GbmTte.open(str(path))
            lo, hi = float(tte.time_range[0]), float(tte.time_range[1])
            if lo <= met <= hi:
                return Path(path)
        raise RuntimeError(f"no TTE file covers MET {met:.1f}: {[str(path) for path in paths]}")

    def _generate_responses_drm_gen(
        self,
        context: Mapping[str, StageResult],
        seg: tuple[float, float],
        detectors: tuple[str, ...],
        workdir: Path,
    ) -> dict[str, Path]:
        """Pure-Python responses from ``gbm_drm_gen`` at the segment midpoint."""
        from gbm_drm_gen import DRMGenTTE

        download_data = context["download"].data
        poshist_path = download_data.get("poshist_path")
        if not poshist_path:
            raise FileNotFoundError("no poshist recorded for response generation")
        mid = 0.5 * (seg[0] + seg[1])
        mapping: dict[str, Path] = {}
        for det in detectors:
            try:
                tte_file = self._tte_covering(download_data["tte"][det], mid)
                cspec_paths = download_data["cspec"][det]
                if not cspec_paths:
                    raise FileNotFoundError(f"no CSPEC files for detector {det}")
                generator = DRMGenTTE(
                    tte_file=str(tte_file),
                    poshist=str(poshist_path),
                    cspecfile=str(cspec_paths[0]),
                    time=mid,
                    mat_type=0,
                )
                out = workdir / f"glg_rsp_{det}_jinwu.rsp"
                generator.to_fits(
                    self.input.ra_deg,
                    self.input.dec_deg,
                    str(out),
                    overwrite=True,
                )
                if not out.is_file():
                    raise RuntimeError(f"gbm_drm_gen did not write {out}")
                mapping[det] = out
            except Exception as exc:
                # One detector's response failure should remain recoverable;
                # the response stage will retain other detectors and report
                # this detector in ``failed_detectors``.
                logger.warning(
                    "GBM response generation failed for %s: %s: %s",
                    det,
                    type(exc).__name__,
                    exc,
                )
        return mapping

    def _stage_response(self, context: Mapping[str, StageResult]) -> StageResult:
        analysis = self._analysis
        backend, backend_note = _resolve_response_backend(analysis)
        if backend is None:
            return StageResult(
                PipelineStatus.NEEDS_REVIEW,
                data={"response_backend": None, "backend_note": backend_note},
                message=f"response_generator_missing: {backend_note}",
            )
        seg = tuple(context["windows"].data["segment_met"])
        detectors = tuple(context["spectra"].data["per_detector"].keys())
        out_dir = self.workspace / "response"
        rsp_work = out_dir / "rsp_work"
        rsp_work.mkdir(parents=True, exist_ok=True)
        try:
            if backend == "official":
                raw_map = self._generate_responses_official(
                    context,
                    seg,
                    detectors,
                    rsp_work,
                    analysis,
                )
            else:
                raw_map = self._generate_responses_drm_gen(
                    context,
                    seg,
                    detectors,
                    rsp_work,
                )
        except Exception as exc:
            reason = (
                f"response_generation_failed:{type(exc).__name__}:{exc}"
            )
            payload = {
                "response_backend": backend,
                "backend_note": backend_note,
                "rsp": {},
                "response_selection": {},
                "warnings": (reason,),
                "failed_detectors": tuple(detectors),
            }
            out = self._write_json(out_dir / "response.json", payload)
            return StageResult(
                PipelineStatus.NEEDS_REVIEW,
                outputs={"response": str(out), "response_dir": str(out_dir)},
                data=payload,
                message=reason,
            )
        mid = 0.5 * (seg[0] + seg[1])
        tte_by_det = context["download"].data.get("tte", {})
        rsp_paths: dict[str, str] = {}
        response_selection: dict[str, dict[str, Any]] = {}
        response_warnings: list[str] = []
        for det in detectors:
            # A response may cover a source interval containing GTI gaps.  The
            # response weighting uses only the actual TTE GTI overlap, while
            # the PHA exposure remains the complete source exposure written by
            # the extractor.
            try:
                gti_intervals = gti_intervals_for_paths(tte_by_det.get(det, ()), seg)
            except Exception as exc:
                response_warnings.append(
                    f"det_{det}:response_gti_unavailable:{type(exc).__name__}:{exc}"
                )
                continue
            if not gti_intervals:
                response_warnings.append(f"det_{det}:response_gti_no_overlap")
                continue
            try:
                response_value = single_response(
                    raw_map[det],
                    mid,
                    out_dir,
                    det,
                    time_intervals=gti_intervals,
                    energy_band_keV=(
                        analysis.nai_band_keV
                        if det in context["detectors"].data["nai"]
                        else analysis.bgo_band_keV
                    ),
                    photon_index=analysis.photon_index_primary,
                    return_metadata=True,
                )
            except Exception as exc:
                response_warnings.append(
                    f"det_{det}:response_processing_failed:{type(exc).__name__}:{exc}"
                )
                continue
            rsp_path, metadata = response_value
            rsp_paths[det] = str(rsp_path)
            metadata["gti_intervals_met"] = [list(item) for item in gti_intervals]
            response_selection[det] = metadata
        payload = {
            "response_backend": backend,
            "backend_note": backend_note,
            "rsp": rsp_paths,
            "response_selection": response_selection,
            "warnings": tuple(response_warnings),
            "failed_detectors": tuple(
                det for det in detectors if det not in rsp_paths
            ),
        }
        if not rsp_paths:
            return StageResult(
                PipelineStatus.NEEDS_REVIEW,
                data=payload,
                message="no detector response was generated; " + "; ".join(response_warnings),
            )
        out = self._write_json(out_dir / "response.json", payload)
        return StageResult(outputs={"response": str(out), "response_dir": str(out_dir)}, data=payload)

    def _stage_fit(self, context: Mapping[str, StageResult]) -> StageResult:
        analysis = self._analysis
        try:
            import xspec

            xspec.Xset.chatter = 0
            xspec.Xset.logChatter = 0
        except Exception as exc:
            return StageResult(
                PipelineStatus.NEEDS_REVIEW,
                message=(
                    "xspec_unavailable: PyXspec could not initialize for the "
                    f"response-aware profile likelihood ({type(exc).__name__}: {exc})"
                ),
                data={
                    "science_result": "not_computed",
                    "warnings": (
                        f"xspec_unavailable:{type(exc).__name__}:{exc}",
                    ),
                },
            )
        seg = tuple(context["windows"].data["segment_met"])
        spectra_data = context["spectra"].data
        rsp_paths = context["response"].data["rsp"]
        windows_per_det = context["windows"].data["per_detector"]
        fit_dir = self.workspace / "fit"
        fit_dir.mkdir(parents=True, exist_ok=True)
        groups: dict[str, list[dict[str, Any]]] = {"nai": [], "bgo": []}
        warnings_list: list[str] = []
        for det, spec in spectra_data["per_detector"].items():
            try:
                ogip = read_ogip_products(spec["pha"], spec["bak"], rsp_paths[det])
            except Exception as exc:
                # A corrupt or channel-mismatched detector product must not
                # prevent an independent detector group from being analysed.
                warnings_list.append(
                    f"det_{det}:ogip_validation_failed:{type(exc).__name__}:{exc}"
                )
                continue
            if ogip["sigma_note"].startswith("background_sigma=sqrt"):
                warnings_list.append(f"det_{det}:bak_stat_err_zero")
            groups[spec["kind"]].append(
                {
                    "detector": det,
                    "kind": spec["kind"],
                    "band": tuple(spec["band_keV"]),
                    "rsp": Path(rsp_paths[det]),
                    "pha": Path(spec["pha"]),
                    "bak": Path(spec["bak"]),
                    "background_variants": {
                        int(order): Path(path)
                        for order, path in spec.get("background_variants", {}).items()
                    },
                    "background_order": int(
                        spec.get("background", {}).get("polynomial_order", -1)
                    ),
                    "background_windows_met": windows_per_det[det]["background_windows_met"],
                    "ogip": ogip,
                    "background_validation": spec.get("background", {}).get("residual_validation"),
                }
            )
            validation = spec.get("background", {}).get("residual_validation")
            if validation is None:
                # A profile based on a background model without an off-source
                # residual check is conditional only.  Keep the detector in
                # the joint fit for diagnostics, but prevent the report from
                # presenting its result as a validated science measurement.
                warnings_list.append(f"det_{det}:background_residuals_unavailable")
            elif isinstance(validation, Mapping) and not validation.get("passed", False):
                warnings_list.append(f"det_{det}:background_residuals_need_review")
        fits_by_group: dict[str, list[dict[str, Any]]] = {"nai": [], "bgo": []}
        systematics: dict[str, float] = {}
        for kind, group in groups.items():
            if not group:
                continue
            band = group[0]["band"]
            # Background covariance is already carried by the Gaussian
            # nuisance in ``read_ogip_products``.  Converting its uncertainty
            # into an additional fractional systematic double-counts the same
            # information.  A non-zero floor remains an explicit opt-in for a
            # separately calibrated analysis, never a 5% default.
            fractional = float(analysis.fractional_systematic_floor or 0.0)
            systematics[kind] = fractional
            if fractional > 0:
                warnings_list.append(
                    f"{kind}_explicit_fractional_systematic={fractional:.3f}"
                )
            for gamma in analysis.photon_indices:
                logger.info("GBM fit: %s group gamma=%g", kind, gamma)
                orders = sorted(
                    {
                        int(item["background_order"])
                        for item in group
                        if int(item.get("background_order", -1)) >= 0
                    }
                    | {
                        int(order)
                        for item in group
                        for order in item.get("background_variants", {})
                    }
                )
                variants: list[dict[str, Any]] = []
                for order in orders or [-1]:
                    variant_group = []
                    for item in group:
                        variant = item["background_variants"].get(order)
                        variant_group.append(
                            item
                            if variant is None
                            else {**item, "bak": variant}
                        )
                    variant_dir = fit_dir / kind
                    if order >= 0:
                        variant_dir = variant_dir / f"background_order{order}"
                    try:
                        variant_result = _run_gamma_fit(
                            variant_group,
                            band,
                            gamma,
                            seg,
                            variant_dir,
                            fractional,
                            analysis,
                            background_order=(None if order < 0 else order),
                            upper_limit_policy=self.config.upper_limit,
                        )
                    except Exception as exc:
                        reason = (
                            f"detector_group_{kind}_gamma_{gamma:g}_fit_failed:"
                            f"{type(exc).__name__}:{exc}"
                        )
                        warnings_list.append(reason)
                        logger.warning("%s", reason)
                        variant_result = _unavailable_gbm_fit(
                            band=band,
                            gamma=gamma,
                            background_order=(None if order < 0 else order),
                            reason=reason,
                            sigma=analysis.significance_threshold_sigma,
                            false_alarm_probability=self.config.upper_limit.detection_false_alarm_probability,
                            target_power=self.config.upper_limit.detection_power,
                        )
                    variants.append(variant_result)
                # Upper limits are the quantity that must remain conservative
                # across acceptable background models.  For a detected fit,
                # retaining the largest bound is still the safe reported
                # profile while the individual variant outputs stay auditable.
                selected = max(
                    variants,
                    key=lambda item: (
                        -math.inf
                        if item.get("amplitude_upper") is None
                        else float(item["amplitude_upper"])
                    ),
                )
                if len(variants) > 1:
                    selected = {
                        **selected,
                        "background_model_variants": variants,
                        "conservative_background_model_set": [
                            item.get("background_order") for item in variants
                        ],
                    }
                fits_by_group[kind].append(selected)
        primary_nai = next(
            (
                fit
                for fit in fits_by_group["nai"]
                if fit["gamma"] == analysis.photon_index_primary
                and _gbm_fit_is_ready(fit)
            ),
            None,
        )
        primary_bgo = next(
            (
                fit
                for fit in fits_by_group["bgo"]
                if fit["gamma"] == analysis.photon_index_primary
                and _gbm_fit_is_ready(fit)
            ),
            None,
        )
        primary_fit = primary_nai or primary_bgo
        if primary_fit is None:
            payload = {
                "analysis_status": "needs_review",
                "science_result": "not_computed",
                "fits": fits_by_group,
                "observed_upper_bound": _unavailable_gbm_observed(
                    band=tuple(analysis.bat_comparable_band_keV),
                    gamma=analysis.photon_index_primary,
                    reason=(
                        "No detector group produced a finite validated "
                        "PHA/BAK/RSP profile."
                    ),
                    sigma=analysis.significance_threshold_sigma,
                ),
                "detection_sensitivity": _unavailable_gbm_sensitivity(
                    band=tuple(analysis.bat_comparable_band_keV),
                    reason=(
                        "No detector group produced a finite validated "
                        "PHA/BAK/RSP profile."
                    ),
                    sigma=analysis.significance_threshold_sigma,
                    false_alarm_probability=self.config.upper_limit.detection_false_alarm_probability,
                    target_power=self.config.upper_limit.detection_power,
                ),
                "background": [
                    item.get("background")
                    for group_items in fits_by_group.values()
                    for item in group_items
                    if isinstance(item, Mapping) and item.get("background") is not None
                ],
                "fractional_systematics": systematics,
                "warnings": tuple(warnings_list),
            }
            out = self._write_json(fit_dir / "fit_summary.json", payload)
            return StageResult(
                outputs={"fit": str(out), "fit_dir": str(fit_dir)},
                data=payload,
            )
        threshold = analysis.significance_threshold_sigma
        significant = primary_fit["significance_sigma"] >= threshold
        science_result = (
            f"detection_{threshold:g}sigma" if significant else f"upper_limit_{threshold:g}sigma"
        )
        background_review = any(
            str(flag).endswith(
                ("background_residuals_need_review", "background_residuals_unavailable")
            )
            for flag in warnings_list
        )
        fit_review = any(
            not _gbm_fit_is_ready(item)
            for group_items in fits_by_group.values()
            for item in group_items
        )
        payload = {
            "analysis_status": "needs_review" if background_review or fit_review else "ready",
            "science_result": science_result,
            "fits": fits_by_group,
            # Expose the primary pair at the file root while retaining every
            # detector/group/gamma variant under ``fits`` for auditability.
            "observed_upper_bound": primary_fit.get("observed_upper_bound"),
            "detection_sensitivity": primary_fit.get("detection_sensitivity"),
            "background": [
                item.get("background")
                for group_items in fits_by_group.values()
                for item in group_items
                if isinstance(item, Mapping) and item.get("background") is not None
            ],
            "fractional_systematics": systematics,
            "significance": {
                "nai": primary_nai["significance_sigma"] if primary_nai else None,
                "bgo": primary_bgo["significance_sigma"] if primary_bgo else None,
                "primary": primary_fit["significance_sigma"],
                "threshold_sigma": threshold,
            },
            "warnings": tuple(warnings_list),
        }
        out = self._write_json(fit_dir / "fit_summary.json", payload)
        return StageResult(outputs={"fit": str(out), "fit_dir": str(fit_dir)}, data=payload)

    def _stage_report(self, context: Mapping[str, StageResult]) -> StageResult:
        analysis = self._analysis
        coverage = context["coverage"].data
        windows = context["windows"].data
        spectra = context.get("spectra")
        response = context.get("response")
        fit = context["fit"]
        fit_data = fit.data if fit.status == PipelineStatus.COMPLETED else {}
        seg = tuple(windows["segment_met"])
        fits_by_group = fit_data.get("fits", {})
        primary_nai = next(
            (
                item
                for item in fits_by_group.get("nai", [])
                if item["gamma"] == analysis.photon_index_primary
                and _gbm_fit_is_ready(item)
            ),
            None,
        )
        primary_bgo = next(
            (
                item
                for item in fits_by_group.get("bgo", [])
                if item["gamma"] == analysis.photon_index_primary
                and _gbm_fit_is_ready(item)
            ),
            None,
        )
        primary_fit = primary_nai or primary_bgo
        threshold = analysis.significance_threshold_sigma
        science_result = str(fit_data.get("science_result", "not_computed"))
        # FIX(H9): safe access in case spectra stage returned NEEDS_REVIEW without per_detector
        detectors = spectra.data.get("per_detector", {}) if spectra is not None else {}
        row: dict[str, Any] = {
            "target_id": self.input.target_id,
            "source_name": self.input.source_name,
            "geometry_status": coverage["status"],
            "coverage_fraction": coverage["coverage_fraction"],
            "start_utc": self.input.start_utc,
            "stop_utc": self.input.stop_utc,
            "segment_met": json.dumps(list(seg)),
            "nai_detectors": ",".join(det for det, spec in detectors.items() if spec["kind"] == "nai"),
            "bgo_detectors": ",".join(det for det, spec in detectors.items() if spec["kind"] == "bgo"),
            "model": "powerlaw_fixed_index_profile",
            "gamma_primary": analysis.photon_index_primary,
            "response_backend": response.data.get("response_backend", "") if response is not None else "",
            "analysis_status": (
                "needs_review"
                if str(fit_data.get("analysis_status", "")) == "needs_review"
                else "pgstat_profile_complete"
                if fit.status == PipelineStatus.COMPLETED
                else "not_run"
            ),
            "science_result": science_result,
        }
        if primary_nai is not None:
            row["nai_significance_sigma"] = f"{primary_nai['significance_sigma']:.6e}"
        if primary_bgo is not None:
            row["bgo_significance_sigma"] = f"{primary_bgo['significance_sigma']:.6e}"
        if primary_fit is not None:
            significant = primary_fit["significance_sigma"] >= threshold
            amplitude = primary_fit["amplitude_mle"] if significant else primary_fit["amplitude_upper"]
            comparable = amplitude * powerlaw_energy_flux_per_norm(
                primary_fit["gamma"], *analysis.bat_comparable_band_keV
            )
            if significant:
                row["flux_erg_cm2_s"] = f"{comparable:.6e}"
            else:
                row["upper_limit_erg_cm2_s"] = f"{comparable:.6e}"
            for key, entry, band in (
                ("nai_8_900_erg_cm2_s", primary_nai, analysis.nai_band_keV),
                ("bgo_200_40000_erg_cm2_s", primary_bgo, analysis.bgo_band_keV),
                ("bat_14_195_erg_cm2_s", primary_fit, analysis.bat_comparable_band_keV),
            ):
                if entry is None:
                    row[key] = ""
                    continue
                group_significant = entry["significance_sigma"] >= threshold
                group_amplitude = entry["amplitude_mle"] if group_significant else entry["amplitude_upper"]
                row[key] = f"{group_amplitude * powerlaw_energy_flux_per_norm(entry['gamma'], *band):.6e}"
            for label, native_band in (("nai", analysis.nai_band_keV), ("bgo", analysis.bgo_band_keV)):
                for gamma in analysis.photon_index_sensitivity:
                    if gamma == analysis.photon_index_primary:
                        continue
                    match = next(
                        (
                            item
                            for item in fits_by_group.get(label, [])
                            if item["gamma"] == gamma and _gbm_fit_is_ready(item)
                        ),
                        None,
                    )
                    if match is None:
                        continue
                    key = f"gamma_{str(gamma).replace('.', 'p')}_{label}_flux_upper_erg_cm2_s"
                    row[key] = f"{match['amplitude_upper'] * powerlaw_energy_flux_per_norm(gamma, *native_band):.6e}"
        collected_warnings: list[str] = []
        for result in context.values():
            if isinstance(result.data, dict):
                collected_warnings.extend(str(flag) for flag in result.data.get("warnings", ()))
        row["quality_flags"] = ";".join(sorted(set(collected_warnings))) or "clean"
        report_dir = self.workspace / "report"
        summary = {
            "target_id": self.input.target_id,
            "source_name": self.input.source_name,
            "ra_deg": self.input.ra_deg,
            "dec_deg": self.input.dec_deg,
            "segment_met": list(seg),
            "coverage": coverage,
            "windows": windows,
            "summary_row": dict(row),
            "fits": fits_by_group,
            "observed_upper_bound": (
                primary_fit.get("observed_upper_bound")
                if primary_fit is not None
                else fit_data.get(
                    "observed_upper_bound",
                    _unavailable_gbm_observed(
                        band=tuple(analysis.bat_comparable_band_keV),
                        gamma=analysis.photon_index_primary,
                        reason="GBM fit did not produce a primary detector result.",
                        sigma=analysis.significance_threshold_sigma,
                    ),
                )
            ),
            "detection_sensitivity": (
                primary_fit.get("detection_sensitivity")
                if primary_fit is not None
                else fit_data.get(
                    "detection_sensitivity",
                    _unavailable_gbm_sensitivity(
                        band=tuple(analysis.bat_comparable_band_keV),
                        reason="GBM fit did not produce a primary detector result.",
                        sigma=analysis.significance_threshold_sigma,
                        false_alarm_probability=self.config.upper_limit.detection_false_alarm_probability,
                        target_power=self.config.upper_limit.detection_power,
                    ),
                )
            ),
            "background": [
                item.get("background")
                for group_items in fits_by_group.values()
                for item in group_items
                if isinstance(item, Mapping) and item.get("background") is not None
            ],
            "science_result": science_result,
            "analysis_status": row["analysis_status"],
            "warnings": sorted(set(collected_warnings)),
        }
        json_path = self._write_json(report_dir / "gbm_summary.json", summary)
        csv_path = report_dir / "summary_row.csv"
        _write_csv(csv_path, [dict(row)])
        return StageResult(
            outputs={"report": str(json_path), "report_csv": str(csv_path)},
            data={
                "analysis_status": row["analysis_status"],
                "science_result": science_result,
                "observed_upper_bound": summary["observed_upper_bound"],
                "detection_sensitivity": summary["detection_sensitivity"],
                "background": summary["background"],
                "warnings": tuple(sorted(set(collected_warnings))),
            },
        )


def _unavailable_gbm_observed(
    *,
    band: tuple[float, float],
    gamma: float,
    reason: str,
    sigma: float = 3.0,
) -> dict[str, Any]:
    """Return the stable observed-bound schema for an unavailable GBM fit."""
    sigma_value = float(sigma)
    if not math.isfinite(sigma_value) or sigma_value <= 0:
        raise ValueError("sigma must be finite and positive")
    return {
        "value": None,
        "unit": "xspec_powerlaw_norm_ph_cm2_s_keV_at_1keV",
        "energy_band": list(band),
        "confidence_level": float(NormalDist().cdf(sigma_value)),
        "confidence_convention": "one_sided_gaussian_equivalent",
        "spectral_model": f"powerlaw_gamma{float(gamma):g}",
        "signed_mle": None,
        "constrained_mle": None,
        "construction": "unavailable",
        "calibration_status": "unavailable",
        "profile_status": "unavailable",
        "reason": str(reason),
    }


def _unavailable_gbm_sensitivity(
    *,
    band: tuple[float, float],
    reason: str,
    sigma: float = 3.0,
    target_power: float = 0.90,
    false_alarm_probability: float | None = None,
) -> dict[str, Any]:
    """Return the stable detection-sensitivity schema for an unavailable fit."""
    sigma_value = float(sigma)
    power_value = float(target_power)
    if not math.isfinite(sigma_value) or sigma_value <= 0:
        raise ValueError("sigma must be finite and positive")
    if not math.isfinite(power_value) or not 0 < power_value < 1:
        raise ValueError("target_power must be between 0 and 1")
    alpha = (
        float(1.0 - NormalDist().cdf(sigma_value))
        if false_alarm_probability is None
        else float(false_alarm_probability)
    )
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("false_alarm_probability must be between 0 and 1")
    return {
        "status": "unavailable",
        "calibration_status": "unavailable",
        "value": None,
        "unit": "xspec_powerlaw_norm_ph_cm2_s_keV_at_1keV",
        "energy_band": list(band),
        "false_alarm_probability": alpha,
        "target_power": power_value,
        "achieved_power": None,
        "search_scope": "fixed_position",
        "trial_count": 0,
        "reason": str(reason),
    }


def _gbm_fit_is_ready(value: Mapping[str, Any] | None) -> bool:
    """Return whether one detector/group profile has a finite upper bound."""
    if not isinstance(value, Mapping):
        return False
    if str(value.get("fit_status", "ready")) != "ready":
        return False
    upper = value.get("amplitude_upper")
    try:
        return upper is not None and math.isfinite(float(upper)) and float(upper) > 0.0
    except (TypeError, ValueError):
        return False


def _unavailable_gbm_fit(
    *,
    band: tuple[float, float],
    gamma: float,
    background_order: int | None,
    reason: str,
    sigma: float,
    false_alarm_probability: float | None = None,
    target_power: float = 0.90,
) -> dict[str, Any]:
    """Keep one detector failure auditable without aborting other detectors."""
    return {
        "gamma": float(gamma),
        "band": list(band),
        "background_order": background_order,
        "fit_status": "failed",
        "reason": str(reason),
        "amplitude_mle": None,
        "amplitude_upper": None,
        "significance_sigma": None,
        "null_statistic": None,
        "fit_statistic": None,
        "unit_flux_native": float(powerlaw_energy_flux_per_norm(gamma, *band)),
        "flux_upper_native": None,
        "observed_upper_bound": _unavailable_gbm_observed(
            band=band,
            gamma=gamma,
            reason=reason,
            sigma=sigma,
        ),
        "detection_sensitivity": _unavailable_gbm_sensitivity(
            band=band,
            reason=reason,
            sigma=sigma,
            false_alarm_probability=false_alarm_probability,
            target_power=target_power,
        ),
        "background": {
            "likelihood": "poisson_gaussian_profile",
            "provenance": [],
            "covariance_source": [],
            "residual_validation": [],
        },
        "warnings": [str(reason)],
    }


def _run_gamma_fit(
    group: Sequence[Mapping[str, Any]],
    band: tuple[float, float],
    gamma: float,
    seg: tuple[float, float],
    out_dir: Path,
    fractional_systematic: float,
    analysis: GBMAnalysisConfig,
    *,
    background_order: int | None = None,
    upper_limit_policy: UpperLimitConfig | None = None,
) -> dict[str, Any]:
    """Profile one fixed-shape power law over a joint detector group."""
    from jinwu.core.upperlimit import (
        CountTemplateModel,
        UpperLimitObservation,
        XspecCountPredictor,
        estimate_upper_limit,
    )

    unit_flux = powerlaw_energy_flux_per_norm(gamma, *band)
    model = CountTemplateModel(
        name=f"powerlaw_gamma{gamma:g}",
        predictor=XspecCountPredictor(model_expression="powerlaw", parameters=(gamma, 1.0)),
        amplitude_unit="xspec_powerlaw_norm_ph_cm2_s_keV_at_1keV",
        flux_per_amplitude=unit_flux,
    )
    observations = [
        UpperLimitObservation(
            name=item["detector"],
            source_counts=item["ogip"]["counts"],
            background_model=item["ogip"]["background_model"],
            background_sigma=(
                None
                if item["ogip"].get("background_covariance") is not None
                else item["ogip"]["background_sigma"]
            ),
            background_covariance=item["ogip"].get("background_covariance"),
            exposure_s=item["ogip"]["exposure_s"],
            response_path=str(item["rsp"]),
            metadata={
                "channel_mask": item["ogip"]["channel_mask"],
                "background_likelihood": "poisson_gaussian_profile",
                "covariance_source": item["ogip"].get(
                    "background_covariance_source", "BAK.STAT_ERR diagonal"
                ),
                "residual_validation": item.get("background_validation"),
            },
        )
        for item in group
    ]
    kind = group[0]["kind"]
    base_policy = upper_limit_policy or UpperLimitConfig(
        strategy="modeled_count_spectrum",
        background_likelihood="poisson_gaussian_profile",
        response_folding="rsp",
        combine="joint_detectors",
        result_modes=("observed_upper_bound", "detection_sensitivity"),
        default_sigma=analysis.significance_threshold_sigma,
        spectral_index=float(gamma),
        calibration="empirical_if_available",
        calibration_mode="empirical_if_available",
        fractional_background_systematic=None,
        enabled=True,
        unavailable_reason=None,
    )
    # A GBM gamma-grid member changes only the fixed photon index.  Preserve
    # the caller's result-mode, confidence, false-alarm, power and calibration
    # policy so disabling sensitivity or changing the confidence level actually
    # reaches the shared execution engine.  The likelihood/response/combine
    # fields are kept explicit because GBM's source-Poisson plus Gaussian-
    # background contract is not interchangeable with BAT's signed-rate path.
    gbm_policy = replace(
        base_policy,
        strategy="modeled_count_spectrum",
        background_likelihood="poisson_gaussian_profile",
        response_folding="rsp",
        combine="joint_detectors",
        spectral_index=float(gamma),
        enabled=True,
        unavailable_reason=None,
    )
    configured_fractional = float(fractional_systematic)
    if (
        upper_limit_policy is not None
        and base_policy.fractional_background_systematic is not None
    ):
        configured_fractional = float(base_policy.fractional_background_systematic)
    instrument = GBM(
        detector=group[0]["detector"].upper(),
        name=f"GBM_{kind.upper()}_joint",
        energy_range_keV=band,
        upper_limit=replace(
            gbm_policy,
            fractional_background_systematic=configured_fractional,
        ),
    )
    # TODO(M6): add timeout protection for estimate_upper_limit — requires thread-level
    # timeout since PyXspec has no native support; pathological data could cause hang
    result = estimate_upper_limit(
        observations,
        model=model,
        instrument_config=instrument,
        interval=seg,
        energy_band=band,
        output_dir=out_dir / f"gamma_{gamma:g}",
        plots=False,
    )
    bound = result.observed_upper_bound
    # FIX(M2): log warning if numerical noise produces negative TS value
    ts_raw = bound.null_statistic
    if ts_raw < 0:
        logger.warning("negative TS=%.2f, clamping to 0", ts_raw)
    significance = math.sqrt(max(0.0, ts_raw))
    background_provenance = [
        {
            "detector": str(item["detector"]),
            "pha": str(item["pha"]),
            "bak": str(item["bak"]),
            "response": str(item["rsp"]),
            "polynomial_order": background_order,
            "residual_validation": item.get("background_validation"),
        }
        for item in group
    ]
    covariance_sources = [
        str(item["ogip"].get("background_covariance_source", "BAK.STAT_ERR diagonal"))
        for item in group
    ]
    residual_validation = [
        item.get("background_validation")
        for item in group
    ]
    return {
        "gamma": float(gamma),
        "band": list(band),
        "background_order": background_order,
        "amplitude_mle": float(bound.amplitude_mle),
        "amplitude_upper": float(bound.amplitude_upper),
        "significance_sigma": float(significance),
        "null_statistic": float(bound.null_statistic),
        "fit_statistic": float(bound.fit_statistic),
        "unit_flux_native": float(unit_flux),
        "flux_upper_native": float(bound.flux_upper) if bound.flux_upper is not None else None,
        "observed_upper_bound": {
            "value": float(bound.amplitude_upper),
            "unit": bound.amplitude_unit,
            "energy_band": list(band),
            "confidence_level": bound.level.confidence,
            "signed_mle": bound.signed_mle,
            "constrained_mle": bound.constrained_mle,
            "construction": bound.construction,
            "calibration_status": bound.calibration_status,
            "confidence_convention": bound.confidence_convention,
            "profile_status": bound.profile_status,
        },
        "background": {
            "likelihood": "poisson_gaussian_profile",
            "provenance": background_provenance,
            "covariance_source": covariance_sources,
            "residual_validation": residual_validation,
        },
        "detection_sensitivity": (
            None
            if result.detection_sensitivity is None
            else {
                "status": result.detection_sensitivity.status,
                "calibration_status": result.detection_sensitivity.calibration_status,
                "value": result.detection_sensitivity.amplitude,
                "unit": result.detection_sensitivity.amplitude_unit,
                "energy_band": list(band),
                "false_alarm_probability": result.detection_sensitivity.false_alarm_probability,
                "target_power": result.detection_sensitivity.target_power,
                "achieved_power": result.detection_sensitivity.achieved_power,
                "search_scope": result.detection_sensitivity.search_scope,
                "trial_count": result.detection_sensitivity.trial_count,
                "reason": result.detection_sensitivity.reason,
            }
        ),
        "warnings": [str(warning) for warning in result.warnings],
    }


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: ``python -m jinwu.fermi.gbm.pipeline``."""
    from jinwu.core.config import GBMContinuous
    from jinwu.core.pipeline import pipeline

    parser = argparse.ArgumentParser(
        description="Resumable Fermi/GBM continuous-data pipeline for one target."
    )
    parser.add_argument("target_id")
    parser.add_argument("--ra", type=float, required=True, help="ICRS right ascension (deg)")
    parser.add_argument("--dec", type=float, required=True, help="ICRS declination (deg)")
    parser.add_argument("--start", required=True, help="interval start (UTC ISO 8601)")
    parser.add_argument("--stop", required=True, help="interval stop (UTC ISO 8601)")
    parser.add_argument("--root", type=Path, required=True, help="GBM data search root (local-first)")
    parser.add_argument("--output", "--output-dir", dest="output", type=Path, default=None)
    parser.add_argument("--source-name", default=None)
    parser.add_argument("--download", action="store_true", help="fetch missing archive products")
    parser.add_argument(
        "--window",
        nargs=2,
        type=float,
        action="append",
        metavar=("START_MET", "STOP_MET"),
        default=[],
        help="explicit source window in Fermi MET (default: coverage segment)",
    )
    parser.add_argument(
        "--background-window",
        nargs=2,
        type=float,
        action="append",
        metavar=("START_MET", "STOP_MET"),
        default=[],
        help="explicit background window in Fermi MET (default: flanking windows)",
    )
    parser.add_argument("--poshist", action="append", default=[], help="explicit position-history file")
    parser.add_argument(
        "--response-backend",
        choices=("auto", "official", "gbm_drm_gen"),
        default=None,
        help="override the configured response backend",
    )
    parser.add_argument("--until", choices=tuple(stage.name for stage in GBMPipeline.stages), default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    config = (
        GBMContinuous(response_backend=args.response_backend)
        if args.response_backend is not None
        else GBMContinuous()
    )
    job = GBMPipelineInput(
        target_id=args.target_id,
        root=args.root,
        output_root=args.output,
        source_name=args.source_name or args.target_id,
        ra_deg=args.ra,
        dec_deg=args.dec,
        start_utc=args.start,
        stop_utc=args.stop,
        download=args.download,
        source_windows_met=tuple(tuple(pair) for pair in args.window) or None,
        background_windows_met=tuple(tuple(pair) for pair in args.background_window) or None,
        poshist_paths=tuple(args.poshist),
    )
    result = pipeline(config, job).run(until=args.until, resume=not args.no_resume)
    print(
        json.dumps(
            {
                "science_status": result.science_status,
                "science_result": result.science_result,
                "message": result.message,
                "stages": dict(result.stages),
                "products": dict(result.products),
                "warnings": list(result.warnings),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if result.science_status == "needs_review":
        return 2
    if result.science_status == "failed":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
