"""Reusable, headless Fermi/GBM continuous-data analysis primitives.

These functions separate coverage and data-product construction from any GUI
or source-specific workflow.  They use Fermi MET internally, while public
intervals retain :class:`astropy.time.Time` and :class:`astropy.coordinates.SkyCoord`
objects so units and time scales remain explicit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
import hashlib
import math
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence
import warnings

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import TimeDelta
import numpy as np

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
]

_NAI_DETECTORS = tuple(f"n{index}" for index in range(10)) + ("na", "nb")
_BGO_DETECTORS = ("b0", "b1")


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


@dataclass(frozen=True, slots=True)
class GBMSpectralProducts:
    """One detector's OGIP products generated from continuous TTE data."""

    detector: str
    source_pha: Path
    background_bak: Path
    source_ranges_met: tuple[tuple[float, float], ...]
    energy_range_keV: tuple[float, float]
    background: GBMBackgroundSummary


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
    bak = fitter.to_bak(time_range=ranges[0])
    background_name = f"gbm_{detector}_background.bak"
    bak.write(outdir, filename=background_name, poisson_errs=False, overwrite=True)
    return GBMSpectralProducts(
        detector=str(detector).lower(),
        source_pha=outdir / source_name,
        background_bak=outdir / background_name,
        source_ranges_met=ranges,
        energy_range_keV=(e_min, e_max),
        background=summary,
    )


def _select_polynomial_background(phaii: Any, ranges: tuple[tuple[float, float], ...], *, background_bin_s: float) -> tuple[Any, GBMBackgroundSummary]:
    from gdt.core.background.binned import Polynomial
    from gdt.core.background.fitter import BackgroundFitter

    candidates: list[tuple[float, Any, GBMBackgroundSummary]] = []
    for order in (0, 1, 2):
        fitter = BackgroundFitter.from_phaii(phaii, Polynomial, time_ranges=list(ranges))
        fitter.fit(order=order)
        statistic = float(np.nansum(np.asarray(fitter.statistic, dtype=float)))
        dof = int(np.nansum(np.asarray(fitter.dof, dtype=float)))
        parameters = np.asarray(fitter.parameters, dtype=float)
        n_parameters = max(1, int(parameters.size))
        n_observations = max(n_parameters + 2, dof + n_parameters)
        correction = 2.0 * n_parameters * (n_parameters + 1) / (n_observations - n_parameters - 1)
        aicc = statistic + 2.0 * n_parameters + correction
        summary = GBMBackgroundSummary(
            polynomial_order=order,
            aicc=float(aicc),
            statistic=statistic,
            dof=dof,
            fit_ranges_met=ranges,
            bin_size_s=float(background_bin_s),
        )
        candidates.append((aicc, fitter, summary))
    _, fitter, summary = min(candidates, key=lambda item: item[0])
    return fitter, summary


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
            angles[detector.name] = float(np.nanmean(direction.separation(coordinate).to_value(u.deg)))
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
            if _resume_archive_file(finder, name, path, verbose=verbose) and _is_valid_fits(path):
                fetched.append(path)
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

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with fits.open(path, memmap=False, lazy_load_hdus=False) as hdul:
                for hdu in hdul:
                    if hdu.data is not None:
                        _ = hdu.data.shape
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
        return True
    except requests.RequestException as exc:
        if verbose:
            print(f"GBM archive transfer interrupted for {name}: {exc}")
        return False


def _quarantine_invalid(path: Path) -> Path:
    """Move an invalid cache entry aside so an archive client can resume safely."""
    candidate = path.with_name(f"{path.name}.invalid")
    suffix = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.name}.invalid.{suffix}")
        suffix += 1
    path.rename(candidate)
    return candidate
