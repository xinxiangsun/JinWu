"""Resumable normal-pointing pipeline for Einstein Probe WXT products."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
from typing import Any, Callable, Mapping, Sequence, cast

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time, TimeDelta
from astropy.wcs import WCS
import astropy.units as u
from ...core.config import InstrumentConfig
from ...core.galactic import resolve_galactic_absorption
from ...core.instruments import Catalog, DataFile, Manifest, SpectrumBundle, scan
from ...core.time import mission_time_format, time_from_mission_seconds
from ...core.pipeline import (
    InstrumentPipeline,
    PipelineInput,
    PipelineStage,
    PipelineStatus,
    StageResult,
    register_pipeline,
)
from ...core.utils import li_ma_snr
from ...core.products import (
    FluxCurveResult,
    NetLightcurve,
    build_flux_curve,
    build_artifact_index,
    build_net_lightcurve,
    build_quicklook_flux_curve,
    collect_xselect_artifacts,
    collect_runtime_environment,
    load_net_lightcurve,
    plot_net_lightcurve,
    render_observation_summary,
    render_wechat_fit_messages,
    save_flux_curve,
    save_net_lightcurve,
)
from ...core.xselect import (
    XSelectRunResult,
    build_effective_ds9_region,
    extract_products_with_xselect,
)

__all__ = [
    "BackgroundScalingResult",
    "ExposureMeasure",
    "TimeResolvedSegment",
    "WXTObservationFiles",
    "WXTPointingInput",
    "WXTPointingPipeline",
    "WXTPointingResult",
    "measure_region_exposure",
    "merge_bayesian_blocks_for_spectra",
]


Extractor = Callable[..., XSelectRunResult]


@dataclass(frozen=True, slots=True, kw_only=True)
class WXTPointingInput(PipelineInput):
    """Inputs that identify one normal-pointing WXT source analysis."""

    ra_deg: float
    dec_deg: float
    source_id: str | None = None
    obsid: str | None = None
    detector: str | None = None
    source_region: Path | str | None = None
    background_region: Path | str | None = None
    auto_approve_regions: bool = False
    trigger_time_utc: str | None = None
    redshift: float = 0.0


@dataclass(frozen=True, slots=True)
class WXTObservationFiles:
    cleaned_event: Path
    exposure_map: Path
    rmf: Path
    arf: Path
    obsid: str
    detector: str
    source_id: str | None
    exposure_correction: Path | None = None
    arm_region: Path | None = None
    gti: Path | None = None
    source_region: Path | None = None
    background_region: Path | None = None
    source_catalog: Path | None = None
    pipeline_source_pha: Path | None = None
    pipeline_background_pha: Path | None = None


@dataclass(frozen=True, slots=True)
class ExposureMeasure:
    exposure_sum: float
    geometric_area_pixels: float
    valid_area_pixels: float
    coverage_fraction: float
    zero_exposure_fraction: float
    nan_fraction: float


@dataclass(frozen=True, slots=True)
class BackgroundScalingResult:
    alpha: float
    source: ExposureMeasure
    background: ExposureMeasure
    background_before_arm: ExposureMeasure
    arm_excluded_exposure: float
    method: str = "exposure_map_ratio"


@dataclass(frozen=True, slots=True)
class TimeResolvedSegment:
    index: int
    start: float
    stop: float
    n_on: float
    n_off: float
    alpha: float
    net_counts: float
    significance: float


@dataclass(slots=True)
class WXTPointingResult:
    workspace: Path
    status: str
    files: WXTObservationFiles | None = None
    alpha: float | None = None
    duration: dict[str, Any] | None = None
    t90_source_pha: Path | None = None
    t90_background_pha: Path | None = None
    segments: list[TimeResolvedSegment] = field(default_factory=list)
    report: Path | None = None
    summary: Path | None = None
    plot_groups: dict[str, tuple[Path, ...]] = field(default_factory=dict)

    def summary_text(self) -> str:
        """Return the copy-ready Chinese announcement, when available."""

        if self.summary is None or not self.summary.is_file():
            return f"WXT pipeline status={self.status}; workspace={self.workspace}"
        return self.summary.read_text(encoding="utf-8").strip()

    def display(self, *, print_summary: bool = True) -> None:
        """Display diagnostics, then print the copy-ready announcement last."""

        try:
            from IPython.display import Image, SVG, display
        except ImportError:
            for title, paths in self.plot_groups.items():
                print(f"{title}: {', '.join(map(str, paths))}")
            if print_summary:
                print(self.summary_text())
            return
        for title, paths in self.plot_groups.items():
            existing = [path for path in paths if path.is_file()]
            if not existing:
                continue
            print(f"\n【{title}】")
            for path in existing:
                if path.suffix.lower() == ".svg":
                    display(SVG(filename=str(path)))
                else:
                    display(Image(filename=str(path)))
        if print_summary:
            print(f"\n{self.summary_text()}")


def _safe_token(value: str) -> str:
    token = "".join(char if char.isalnum() or char in "._-" else "_" for char in value)
    token = "_".join(part for part in token.split("_") if part)
    if not token:
        raise ValueError("target_id must contain a filename-safe character")
    return token


def _candidate_id(obsid: str, detector: str, source_id: str | None) -> str:
    detector_numbers = "".join(re.findall(r"\d+", detector))
    detector_token = detector_numbers or _safe_token(detector).lower()
    return f"ep{obsid}wxt{detector_token}{source_id or ''}".lower()


def _single_file(
    manifest: Manifest,
    role: str,
    *,
    required: bool,
    source_id: str | None = None,
) -> Path | None:
    candidates = manifest.by_role(role)
    if role == "exposure":
        processed = [item for item in candidates if "rawinstr" not in item.path.name.lower()]
        if processed:
            candidates = processed
    if role == "gti":
        pointed = [item for item in candidates if "po_clgti" in item.path.name.lower()]
        if pointed:
            candidates = pointed
    if role == "source_region" and source_id is not None:
        canonical_suffix = f"{source_id.lower()}.reg"
        canonical = [item for item in candidates if item.path.name.lower().endswith(canonical_suffix)]
        if canonical:
            candidates = canonical
    if source_id is not None:
        exact = [item for item in candidates if item.source_id == source_id]
        if exact:
            candidates = exact
        else:
            candidates = [item for item in candidates if item.source_id is None]
    if len(candidates) > 1:
        # WXT work directories often contain copied fit products below the
        # official L2/L3 root. Prefer the shallowest canonical product, while
        # preserving an ambiguity error for multiple peers at the same depth.
        depths = {
            item.path: len(item.path.relative_to(manifest.root).parts)
            for item in candidates
        }
        minimum_depth = min(depths.values())
        shallow = [item for item in candidates if depths[item.path] == minimum_depth]
        if len(shallow) == 1:
            candidates = shallow
        else:
            hashes = {_file_hash(item.path) for item in shallow}
            if len(hashes) == 1:
                candidates = [sorted(shallow, key=lambda item: item.path.name)[0]]
            else:
                names = ", ".join(str(item.path.relative_to(manifest.root)) for item in shallow)
                raise ValueError(f"Ambiguous WXT {role}: {names}")
    if not candidates:
        if required:
            raise ValueError(f"WXT directory is missing required role: {role}")
        return None
    return candidates[0].path.resolve()


def _choose_manifest(catalog: Catalog, input_data: WXTPointingInput) -> Manifest:
    candidates = [manifest for manifest in catalog.manifests if manifest.instrument.upper() == "WXT"]
    if input_data.obsid is not None:
        candidates = [manifest for manifest in candidates if manifest.obsid == input_data.obsid]
    if input_data.detector is not None:
        detector = input_data.detector.upper()
        candidates = [
            manifest for manifest in candidates if (manifest.detector or "").upper() == detector
        ]
    if len(candidates) != 1:
        labels = ", ".join(
            f"{item.obsid}/{item.detector}" for item in candidates
        ) or "none"
        raise ValueError(
            "WXT input must resolve to exactly one observation/detector manifest; "
            f"candidates: {labels}"
        )
    return candidates[0]


def _resolve_source_id(manifest: Manifest, requested: str | None) -> str | None:
    source_ids = sorted(
        {
            item.source_id
            for item in manifest.files
            if item.source_id is not None
            and item.role in {"arf", "source_region", "background_region", "source_pha"}
        }
    )
    if requested is not None:
        normalized = requested.lower()
        if source_ids and normalized not in {item.lower() for item in source_ids}:
            raise ValueError(
                f"WXT source_id {requested!r} not found; available: {', '.join(source_ids)}"
            )
        return next((item for item in source_ids if item.lower() == normalized), requested)
    if len(source_ids) > 1:
        raise ValueError(f"WXT directory has multiple sources; choose one of: {', '.join(source_ids)}")
    return source_ids[0] if source_ids else None


def discover_wxt_files(input_data: WXTPointingInput) -> WXTObservationFiles:
    catalog = scan(input_data.resolved_root(), instrument="wxt")
    manifest = _choose_manifest(catalog, input_data)
    source_id = _resolve_source_id(manifest, input_data.source_id)

    bundle = next(
        (item for item in manifest.bundles if item.source_id == source_id and item.ready),
        None,
    )
    arf = bundle.arf.path.resolve() if bundle and bundle.arf else _single_file(
        manifest, "arf", required=True, source_id=source_id
    )
    rmf = bundle.rmf.path.resolve() if bundle and bundle.rmf else _single_file(
        manifest, "rmf", required=True, source_id=source_id
    )
    if arf is None or rmf is None:  # protected by required=True
        raise RuntimeError("WXT response discovery failed")

    cleaned_event = _single_file(manifest, "cleaned_event", required=True)
    exposure = _single_file(manifest, "exposure", required=True)
    if cleaned_event is None or exposure is None:
        raise RuntimeError("WXT event/exposure discovery failed")
    if not manifest.obsid or not manifest.detector:
        raise ValueError("WXT manifest lacks obsid or detector identity")

    return WXTObservationFiles(
        cleaned_event=cleaned_event,
        exposure_map=exposure,
        exposure_correction=_single_file(manifest, "exposure_correction", required=False),
        arm_region=_single_file(manifest, "arm_region", required=False),
        rmf=rmf,
        arf=arf,
        gti=_single_file(manifest, "gti", required=False),
        source_region=_single_file(
            manifest, "source_region", required=False, source_id=source_id
        ),
        background_region=_single_file(
            manifest, "background_region", required=False, source_id=source_id
        ),
        source_catalog=_single_file(manifest, "source_catalog", required=False),
        pipeline_source_pha=(
            bundle.source_pha.path.resolve() if bundle and bundle.source_pha else None
        ),
        pipeline_background_pha=(
            bundle.background_pha.path.resolve() if bundle and bundle.background_pha else None
        ),
        obsid=manifest.obsid,
        detector=manifest.detector,
        source_id=source_id,
    )


def _region_to_mask(region, wcs: WCS, shape: tuple[int, int], mode: str) -> np.ndarray:
    if hasattr(region, "to_pixel"):
        pixel_region = region.to_pixel(wcs)
    elif hasattr(region, "to_mask"):
        pixel_region = region
    else:  # pragma: no cover - guarded by the regions parser
        raise TypeError(f"Unsupported region object: {type(region).__name__}")
    if hasattr(pixel_region, "inner_radius") and hasattr(pixel_region, "outer_radius"):
        from regions import CirclePixelRegion

        outer = CirclePixelRegion(pixel_region.center, pixel_region.outer_radius)
        inner = CirclePixelRegion(pixel_region.center, pixel_region.inner_radius)
        outer_image = outer.to_mask(mode=mode).to_image(shape)
        inner_image = inner.to_mask(mode=mode).to_image(shape)
        outer_array = np.zeros(shape) if outer_image is None else np.asarray(outer_image, dtype=float)
        inner_array = np.zeros(shape) if inner_image is None else np.asarray(inner_image, dtype=float)
        return np.clip(outer_array - inner_array, 0.0, 1.0)
    try:
        region_mask = pixel_region.to_mask(mode=mode)
    except NotImplementedError:
        region_mask = pixel_region.to_mask(mode="subpixels", subpixels=8)
    image = region_mask.to_image(shape)
    if image is None:
        return np.zeros(shape, dtype=float)
    return np.clip(np.asarray(image, dtype=float), 0.0, 1.0)


def _combined_region_mask(
    region_paths: Sequence[Path],
    wcs: WCS,
    header: fits.Header,
    shape: tuple[int, int],
    *,
    mode: str,
) -> np.ndarray:
    try:
        from regions import Regions
    except ImportError as exc:  # pragma: no cover - depends on optional runtime env
        raise ImportError(
            "WXT region/exposure processing requires the 'regions' package"
        ) from exc
    include = np.zeros(shape, dtype=float)
    exclude = np.zeros(shape, dtype=float)
    n_include = 0
    for path_index, path in enumerate(region_paths):
        text = path.read_text(encoding="utf-8", errors="replace")
        significant = [
            line.strip().lower()
            for line in text.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        frames = {"physical", "image", "fk5", "icrs", "galactic", "ecliptic", "fk4"}
        frame_values = {line.split(";", 1)[0].strip() for line in significant} & frames
        has_frame = bool(frame_values)
        physical = (not has_frame) or ("physical" in frame_values)
        if physical:
            text = _physical_region_to_image(text, header)
            if not has_frame:
                text = "image\n" + text
            else:
                text = re.sub(r"(?im)^\s*physical\s*$", "image", text)
        for region in Regions.parse(text, format="ds9"):
            mask = _region_to_mask(region, wcs, shape, mode)
            is_include = bool(region.meta.get("include", True))
            if path_index > 0:
                # ``exclusion_regions`` are semantic masks: a positive
                # polygon in an ARM file is still an area to remove.
                exclude = np.maximum(exclude, mask)
            elif is_include:
                include = np.maximum(include, mask)
                n_include += 1
            else:
                exclude = np.maximum(exclude, mask)
    if n_include == 0:
        raise ValueError("Effective extraction region contains no inclusion shape")
    return include * (1.0 - exclude)


def _physical_region_to_image(text: str, header: fits.Header) -> str:
    """Apply a FITS physical-to-image LTM/LTV transform to DS9 shapes."""
    sx = float(header.get("LTM1_1", 1.0))
    sy = float(header.get("LTM2_2", 1.0))
    ox = float(header.get("LTV1", 0.0))
    oy = float(header.get("LTV2", 0.0))
    radial_scale = 0.5 * (abs(sx) + abs(sy))
    if not all(np.isfinite(value) for value in (sx, sy, ox, oy)) or sx == 0 or sy == 0:
        raise ValueError("Exposure map has invalid LTM/LTV physical-coordinate transform")

    shape_pattern = re.compile(
        r"^(?P<indent>\s*)(?P<exclude>-?)(?P<shape>circle|annulus|polygon|ellipse|box)"
        r"\s*\((?P<args>[^)]*)\)(?P<tail>.*)$",
        re.IGNORECASE,
    )

    def transform_line(line: str) -> str:
        match = shape_pattern.match(line)
        if not match:
            return line
        try:
            values = [float(value.strip()) for value in match.group("args").split(",")]
        except ValueError as exc:
            raise ValueError(f"Unsupported physical DS9 region arguments: {line}") from exc
        shape = match.group("shape").lower()
        if len(values) < 2:
            raise ValueError(f"Physical DS9 region lacks center coordinates: {line}")
        values[0] = sx * values[0] + ox
        values[1] = sy * values[1] + oy
        if shape == "polygon":
            if len(values) % 2:
                raise ValueError(f"Physical polygon has an odd coordinate count: {line}")
            for index in range(2, len(values), 2):
                values[index] = sx * values[index] + ox
                values[index + 1] = sy * values[index + 1] + oy
        elif shape in {"circle", "annulus"}:
            for index in range(2, len(values)):
                values[index] *= radial_scale
        elif shape in {"ellipse", "box"}:
            if len(values) >= 4:
                values[2] *= abs(sx)
                values[3] *= abs(sy)
        args = ",".join(f"{value:.10g}" for value in values)
        return (
            f"{match.group('indent')}{match.group('exclude')}{match.group('shape')}"
            f"({args}){match.group('tail')}"
        )

    return "\n".join(transform_line(line) for line in text.splitlines()) + "\n"


def _measure_mask(exposure: np.ndarray, mask: np.ndarray) -> ExposureMeasure:
    area = float(mask.sum())
    if not np.isfinite(area) or area <= 0:
        raise ValueError("Region has zero geometric area on the exposure map")
    finite = np.isfinite(exposure)
    positive = finite & (exposure > 0)
    valid_area = float((mask * positive).sum())
    nan_area = float((mask * (~finite)).sum())
    zero_area = float((mask * (finite & (exposure <= 0))).sum())
    exposure_sum = float(np.sum(mask * np.where(positive, exposure, 0.0)))
    if exposure_sum <= 0:
        raise ValueError("Region has no positive exposure")
    return ExposureMeasure(
        exposure_sum=exposure_sum,
        geometric_area_pixels=area,
        valid_area_pixels=valid_area,
        coverage_fraction=valid_area / area,
        zero_exposure_fraction=zero_area / area,
        nan_fraction=nan_area / area,
    )


def measure_region_exposure(
    exposure_map: str | Path,
    include_region: str | Path,
    *,
    exclusion_regions: Sequence[str | Path] = (),
    mask_mode: str = "exact",
) -> ExposureMeasure:
    """Integrate one exposure image over the exact effective extraction region."""
    path = Path(exposure_map)
    with fits.open(path, memmap=False) as hdul:
        image_hdu = next((hdu for hdu in hdul if hdu.data is not None and hdu.data.ndim == 2), None)
        if image_hdu is None:
            raise ValueError(f"Exposure map contains no 2-D image: {path}")
        exposure = np.asarray(image_hdu.data, dtype=float)
        header = image_hdu.header.copy()
        wcs = WCS(header)
    paths = [Path(include_region), *(Path(item) for item in exclusion_regions)]
    mask = _combined_region_mask(paths, wcs, header, exposure.shape, mode=mask_mode)
    return _measure_mask(exposure, mask)


def _write_source_region(path: Path, ra: float, dec: float, radius_arcsec: float) -> None:
    path.write_text(
        f"# Region file format: DS9 version 4.1\nfk5\n"
        f"circle({ra:.7f},{dec:.7f},{radius_arcsec:.3f}\")\n",
        encoding="ascii",
    )


def _sector_polygon(
    center: SkyCoord,
    start_deg: float,
    stop_deg: float,
    inner_arcsec: float,
    outer_arcsec: float,
    *,
    samples: int = 80,
) -> str:
    angles = np.linspace(start_deg, stop_deg, samples) * u.deg
    outer = center.directional_offset_by(angles, outer_arcsec * u.arcsec)
    inner = center.directional_offset_by(angles[::-1], inner_arcsec * u.arcsec)
    coords = zip(
        np.concatenate((outer.ra.deg, inner.ra.deg)),
        np.concatenate((outer.dec.deg, inner.dec.deg)),
    )
    return "polygon(" + ",".join(f"{ra:.7f},{dec:.7f}" for ra, dec in coords) + ")"


def _write_background_region(
    path: Path,
    ra: float,
    dec: float,
    roll_deg: float,
    sectors: Sequence[tuple[float, float, float, float]],
) -> None:
    center = SkyCoord(ra, dec, unit="deg", frame="fk5")
    roll = float(roll_deg) % 90.0
    lines = ["# Region file format: DS9 version 4.1", "fk5"]
    lines.extend(
        _sector_polygon(center, roll + start, roll + stop, inner, outer)
        for start, stop, inner, outer in sectors
    )
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


_DS9_REGION_SHAPE = re.compile(
    r"^\s*(?:[a-z][a-z0-9_]*\s*;\s*)?(?P<sign>[+-]?)\s*"
    r"(?P<shape>circle|annulus|polygon|ellipse|box|sector)\s*\(",
    re.IGNORECASE,
)


def _validate_arm_region(path: str | Path) -> tuple[str, ...]:
    """Require a WXT ARM mask to contain at least one DS9 shape."""

    region_path = Path(path)
    shape_lines: list[str] = []
    for raw_line in region_path.read_text(
        encoding="utf-8", errors="replace"
    ).splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.lower().startswith("global"):
            continue
        match = _DS9_REGION_SHAPE.match(line)
        if match is None:
            continue
        shape_lines.append(line)
    if not shape_lines:
        raise ValueError(f"WXT ARM region contains no DS9 shapes: {region_path}")
    return tuple(shape_lines)


def _event_roll(path: Path) -> float | None:
    with fits.open(path, memmap=False) as hdul:
        for hdu in hdul:
            for key in ("PA_PNT", "ROLL_PNT", "ROLL"):
                value = hdu.header.get(key)
                if value is not None:
                    try:
                        return float(value)
                    except (TypeError, ValueError):
                        continue
    return None


def _fits_extension(hdul: fits.HDUList, preferred: str):
    try:
        return hdul[preferred]
    except KeyError:
        return hdul[1]


def _write_event_scaling(path: Path, backscal: float, exposure_sum: float) -> None:
    with fits.open(path, mode="update", memmap=False) as hdul:
        event_hdu = _fits_extension(hdul, "EVENTS")
        for header in (hdul[0].header, event_hdu.header):
            header["BACKSCAL"] = (float(backscal), "Jinwu source/background scale proxy")
            header["JWEXPSUM"] = (float(exposure_sum), "Exposure-map sum in extraction region")
            header["JWBKSMTH"] = ("EXPMAP_RATIO", "Jinwu BACKSCAL metadata method")
            header.add_history("Jinwu event BACKSCAL is ratio metadata, not an OGIP PHA area.")
        hdul.flush()


def _pha_values(path: Path) -> tuple[float, float]:
    with fits.open(path, memmap=False) as hdul:
        header = _fits_extension(hdul, "SPECTRUM").header
        exposure = float(header.get("EXPOSURE", 0.0))
        backscal = float(header.get("BACKSCAL", 0.0))
    if exposure <= 0 or backscal <= 0:
        raise ValueError(f"PHA has invalid EXPOSURE/BACKSCAL: {path}")
    return exposure, backscal


def _set_pha_backscal(path: Path, value: float) -> None:
    with fits.open(path, mode="update", memmap=False) as hdul:
        header = _fits_extension(hdul, "SPECTRUM").header
        header["BACKSCAL"] = (float(value), "Exposure-map matched background scaling")
        header.add_history("Jinwu adjusted BACKSCAL from source/background exposure-map ratio.")
        hdul.flush()


def _finalize_pha_pair(
    source: Path,
    background: Path,
    *,
    alpha: float,
    rmf: Path,
    arf: Path,
    alpha_rtol: float,
    exposure_rtol: float,
) -> tuple[Path, Path, Path, Path]:
    src_exp, src_backscal = _pha_values(source)
    bkg_exp, _ = _pha_values(background)
    if not np.isclose(src_exp, bkg_exp, rtol=exposure_rtol, atol=1e-6):
        raise ValueError(
            f"Source/background PHA exposures differ: {src_exp} vs {bkg_exp}"
        )
    _set_pha_backscal(background, src_backscal / alpha)

    staged_rmf = source.parent / rmf.name
    staged_arf = source.parent / arf.name
    for original, staged in ((rmf, staged_rmf), (arf, staged_arf)):
        if original.resolve() != staged.resolve():
            shutil.copy2(original, staged)
    with fits.open(source, mode="update", memmap=False) as hdul:
        header = _fits_extension(hdul, "SPECTRUM").header
        header["BACKFILE"] = background.name
        header["RESPFILE"] = staged_rmf.name
        header["ANCRFILE"] = staged_arf.name
        hdul.flush()

    src_exp, src_backscal = _pha_values(source)
    bkg_exp, bkg_backscal = _pha_values(background)
    alpha_counts = (src_exp / bkg_exp) * (src_backscal / bkg_backscal)
    if not np.isclose(alpha_counts, alpha, rtol=alpha_rtol, atol=0.0):
        raise ValueError(f"PHA alpha mismatch: expected {alpha}, reconstructed {alpha_counts}")
    for path in (source, background, staged_rmf, staged_arf):
        with fits.open(path, memmap=False):
            pass
    return source, background, staged_rmf, staged_arf


def _pha_total_counts(path: Path, pi_range: tuple[int, int] | None = None) -> int:
    with fits.open(path, memmap=False) as hdul:
        data = _fits_extension(hdul, "SPECTRUM").data
        if data is None or "COUNTS" not in data.names:
            raise ValueError(f"PHA lacks COUNTS: {path}")
        counts = np.asarray(data["COUNTS"], dtype=np.int64)
        if pi_range is not None:
            if "CHANNEL" not in data.names:
                raise ValueError(f"PHA lacks CHANNEL for band count check: {path}")
            channels = np.asarray(data["CHANNEL"], dtype=int)
            lo, hi = pi_range
            counts = counts[(channels >= lo) & (channels <= hi)]
        return int(counts.sum())


def _event_interval_count(path: Path, start: float, stop: float) -> int:
    times = _event_times(path)
    return int(np.count_nonzero((times >= start) & (times <= stop)))


def _validate_ogip_bundle(
    source: Path,
    background: Path,
    rmf: Path,
    arf: Path,
    *,
    fit_energy_range_keV: tuple[float, float],
) -> dict[str, Any]:
    with fits.open(source, memmap=False) as src_hdul:
        src_hdu = _fits_extension(src_hdul, "SPECTRUM")
        src_data = src_hdu.data
        if src_data is None or "CHANNEL" not in src_data.names:
            raise ValueError(f"Source PHA lacks CHANNEL: {source}")
        channels = np.asarray(src_data["CHANNEL"], dtype=int)
        header = src_hdu.header
        for keyword, expected in (
            ("BACKFILE", background.name),
            ("RESPFILE", rmf.name),
            ("ANCRFILE", arf.name),
        ):
            if str(header.get(keyword, "")).strip() != expected:
                raise ValueError(f"{source.name} has invalid {keyword}: {header.get(keyword)!r}")

    rmf_channels = None
    with fits.open(rmf, memmap=False) as rmf_hdul:
        for hdu in rmf_hdul:
            data = hdu.data
            if data is not None and getattr(data, "names", None) and "CHANNEL" in data.names:
                rmf_channels = np.asarray(data["CHANNEL"], dtype=int)
                break
    if rmf_channels is not None and channels.size:
        if not np.all(np.isin(channels, rmf_channels)):
            raise ValueError("PHA CHANNEL values are not covered by RMF EBOUNDS")

    arf_bounds = None
    with fits.open(arf, memmap=False) as arf_hdul:
        for hdu in arf_hdul:
            data = hdu.data
            names = getattr(data, "names", None)
            if data is not None and names and {"ENERG_LO", "ENERG_HI"} <= set(names):
                lo = np.asarray(data["ENERG_LO"], dtype=float)
                hi = np.asarray(data["ENERG_HI"], dtype=float)
                if lo.size and hi.size:
                    arf_bounds = (float(np.nanmin(lo)), float(np.nanmax(hi)))
                break
    if arf_bounds is not None:
        fit_lo, fit_hi = fit_energy_range_keV
        if arf_bounds[0] > fit_lo or arf_bounds[1] < fit_hi:
            raise ValueError(
                f"ARF range {arf_bounds} does not cover fit range {fit_energy_range_keV}"
            )
    return {
        "pha_channels": int(channels.size),
        "rmf_channel_check": rmf_channels is not None,
        "arf_energy_range_keV": arf_bounds,
    }


def _event_times(path: Path) -> np.ndarray:
    with fits.open(path, memmap=False) as hdul:
        data = _fits_extension(hdul, "EVENTS").data
        if data is None or "TIME" not in data.names:
            raise ValueError(f"Event file lacks TIME: {path}")
        return np.asarray(data["TIME"], dtype=float)


def _count_interval(times: np.ndarray, start: float, stop: float, *, final: bool = False) -> int:
    side = "right" if final else "left"
    left = int(np.searchsorted(times, start, side="left"))
    right = int(np.searchsorted(times, stop, side=side))
    return max(right - left, 0)


def merge_bayesian_blocks_for_spectra(
    edges: Sequence[float],
    source_times: Sequence[float],
    background_times: Sequence[float],
    *,
    alpha: float,
    t_start: float,
    t_stop: float,
    minimum_net_counts: float,
    minimum_significance: float,
) -> list[TimeResolvedSegment]:
    """Merge adjacent duration blocks until each segment is fit-worthy."""
    clipped = np.asarray(edges, dtype=float)
    clipped = clipped[np.isfinite(clipped)]
    clipped = np.clip(clipped, t_start, t_stop)
    clipped = np.unique(np.concatenate(([t_start], clipped, [t_stop])))
    if clipped.size < 2:
        return []
    src = np.sort(np.asarray(source_times, dtype=float))
    bkg = np.sort(np.asarray(background_times, dtype=float))
    segments: list[TimeResolvedSegment] = []
    pending_start = float(clipped[0])
    pending_on = 0.0
    pending_off = 0.0

    for index, (left, right) in enumerate(zip(clipped[:-1], clipped[1:])):
        final = index == clipped.size - 2
        pending_on += _count_interval(src, float(left), float(right), final=final)
        pending_off += _count_interval(bkg, float(left), float(right), final=final)
        net = pending_on - alpha * pending_off
        significance = li_ma_snr(pending_on, pending_off, alpha, signed=True)
        if net >= minimum_net_counts and significance >= minimum_significance:
            segments.append(
                TimeResolvedSegment(
                    index=len(segments),
                    start=pending_start,
                    stop=float(right),
                    n_on=pending_on,
                    n_off=pending_off,
                    alpha=alpha,
                    net_counts=net,
                    significance=significance,
                )
            )
            pending_start = float(right)
            pending_on = 0.0
            pending_off = 0.0

    if pending_on > 0 or pending_off > 0:
        if not segments:
            return []
        previous = segments.pop()
        n_on = previous.n_on + pending_on
        n_off = previous.n_off + pending_off
        segments.append(
            TimeResolvedSegment(
                index=previous.index,
                start=previous.start,
                stop=t_stop,
                n_on=n_on,
                n_off=n_off,
                alpha=alpha,
                net_counts=n_on - alpha * n_off,
                significance=li_ma_snr(n_on, n_off, alpha, signed=True),
            )
        )
    return segments


def _json_dump(path: Path, payload: Any) -> Path:
    def clean(value):
        if isinstance(value, Mapping):
            return {str(key): clean(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(item) for item in value]
        if hasattr(value, "tolist"):
            return clean(value.tolist())
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(clean(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _event_mjdref(path: Path) -> tuple[float, str] | None:
    with fits.open(path, memmap=False) as hdul:
        headers = [hdu.header for hdu in hdul]
        for header in headers:
            if "MJDREF" in header:
                mjdref = float(header["MJDREF"])
                return mjdref, str(header.get("TIMESYS", "TT")).lower()
            if "MJDREFI" in header or "MJDREFF" in header:
                mjdref = float(header.get("MJDREFI", 0.0)) + float(header.get("MJDREFF", 0.0))
                return mjdref, str(header.get("TIMESYS", "TT")).lower()
    return None


def _pipeline_t0(
    event: Path,
    trigger_time_utc: str | None,
    t100_start: float,
    mission: str | None,
) -> dict[str, Any]:
    """Return the single T0 definition used by duration outputs and plots."""
    if trigger_time_utc is not None:
        reference = _event_mjdref(event)
        trigger = Time(trigger_time_utc, scale="utc")
        mission_time = None
        if reference is not None:
            mjdref, scale = reference
            mission_time = float((trigger - Time(mjdref, format="mjd", scale=scale)).to_value("s"))
        return {
            "source": "external_trigger",
            "mission_time_s": mission_time,
            "utc": trigger.utc.isot,
        }
    anchor = time_from_mission_seconds(mission, t100_start)
    return {
        "source": "t100_start",
        "mission_time_s": float(t100_start),
        "utc": anchor.utc.isot if anchor is not None else None,
    }


def _time_reference_provenance(
    event: Path, anchor: Time | None, mission_time_s: float | None
) -> dict[str, Any]:
    """Record, but do not replace, the FITS clock consistency check."""
    reference = _event_mjdref(event)
    if reference is None:
        return {"available": False, "matches_mission_clock": None}
    mjdref, scale = reference
    result: dict[str, Any] = {
        "available": True,
        "mjdref": mjdref,
        "timesys": scale.upper(),
    }
    if anchor is None or mission_time_s is None or not np.isfinite(mission_time_s):
        result["matches_mission_clock"] = None
        return result
    from_fits = Time(mjdref, format="mjd", scale=scale) + TimeDelta(
        float(mission_time_s), format="sec"
    )
    delta_s = float((anchor - from_fits).to_value("sec"))
    result.update(
        {
            "utc_from_fits": from_fits.utc.isot,
            "mission_clock_delta_s": delta_s,
            "matches_mission_clock": abs(delta_s) <= 1e-3,
        }
    )
    return result


@register_pipeline("ep.wxt.pointing")
class WXTPointingPipeline(InstrumentPipeline[WXTPointingInput, WXTPointingResult]):
    """Normal-pointing WXT workflow with explicit region review."""

    stages = (
        PipelineStage("discover"),
        PipelineStage("galactic_absorption", ("discover",)),
        PipelineStage("pipeline_spectrum", ("discover",)),
        PipelineStage("regions", ("discover", "galactic_absorption")),
        PipelineStage("provisional_events", ("discover", "regions")),
        PipelineStage("exposure_arm_qc", ("discover", "regions", "provisional_events")),
        PipelineStage("final_events", ("discover", "regions", "exposure_arm_qc")),
        PipelineStage("duration", ("final_events", "exposure_arm_qc")),
        PipelineStage("lightcurves", ("discover", "regions", "final_events", "exposure_arm_qc", "duration")),
        PipelineStage("t100_spectra", ("discover", "regions", "duration", "exposure_arm_qc")),
        PipelineStage("t90_spectra", ("discover", "regions", "duration", "exposure_arm_qc")),
        PipelineStage("ogip_finalize", ("discover", "final_events", "duration", "t100_spectra", "t90_spectra", "exposure_arm_qc")),
        PipelineStage("bayesian_block_spectra", ("discover", "regions", "duration", "final_events", "ogip_finalize", "exposure_arm_qc")),
        PipelineStage("fit", ("discover", "galactic_absorption", "pipeline_spectrum", "ogip_finalize", "bayesian_block_spectra")),
        PipelineStage("fluxcurve", ("lightcurves", "duration", "fit", "bayesian_block_spectra")),
        PipelineStage("report", tuple(stage for stage in (
            "discover", "galactic_absorption", "regions", "provisional_events",
            "exposure_arm_qc", "final_events", "lightcurves", "duration",
            "pipeline_spectrum", "t100_spectra", "t90_spectra", "ogip_finalize",
            "bayesian_block_spectra", "fit",
            "fluxcurve",
        ))),
    )

    def __init__(
        self,
        input_data: WXTPointingInput,
        *,
        config: InstrumentConfig,
        extractor: Extractor = extract_products_with_xselect,
        nhtot_query: Callable[..., dict[str, Any]] | None = None,
    ):
        super().__init__(input_data, config=config)
        self.extractor = extractor
        self.nhtot_query = nhtot_query

    def run(
        self,
        *,
        until: str | None = None,
        resume: bool | None = None,
    ) -> WXTPointingResult:
        """Run the WXT pipeline with a concrete result type for callers and IDEs."""

        return cast(WXTPointingResult, super().run(until=until, resume=resume))

    def stage_code_dependencies(self, stage: PipelineStage) -> tuple[Path, ...]:
        core = Path(__file__).resolve().parents[2] / "core"
        dependencies = {
            "galactic_absorption": (core / "galactic.py", core / "utils.py"),
            "pipeline_spectrum": (),
            "provisional_events": (core / "xselect.py",),
            "final_events": (core / "xselect.py",),
            "lightcurves": (core / "products.py", core / "time.py", core / "xselect.py"),
            "duration": (
                core / "products.py",
                core / "plot.py",
                core / "ops.py",
                core / "time.py",
                core / "io.py",
                core / "data.py",
            ),
            "t100_spectra": (core / "xselect.py",),
            "t90_spectra": (core / "xselect.py",),
            "bayesian_block_spectra": (core / "xselect.py",),
            "fit": (core / "fit.py", core / "products.py", core / "plot.py"),
            "fluxcurve": (core / "products.py",),
            "report": (core / "products.py",),
        }
        return dependencies.get(stage.name, ())

    @property
    def approval_path(self) -> Path:
        return self.workspace / "regions" / "approved.json"

    def approve_regions(self, *, note: str = "approved") -> Path:
        """Approve the current region proposal and allow downstream stages."""
        region_manifest = self.workspace / "regions" / "regions.json"
        if not region_manifest.is_file():
            raise RuntimeError("Run the pipeline through exposure_arm_qc before approval")
        payload = json.loads(region_manifest.read_text(encoding="utf-8"))
        approval = {
            "note": note,
            "region_manifest_sha256": _file_hash(region_manifest),
        }
        return _json_dump(self.approval_path, approval)

    def validate_input(self) -> None:
        if self.config.name.upper() != "WXT":
            raise ValueError(f"WXTPointingPipeline requires WXT config, got {self.config.name}")
        root = self.input.resolved_root()
        if not root.is_dir():
            raise FileNotFoundError(f"WXT product directory does not exist: {root}")
        if not np.isfinite(float(self.input.ra_deg)) or not 0.0 <= float(self.input.ra_deg) < 360.0:
            raise ValueError("ra_deg must be finite and in [0, 360)")
        if not np.isfinite(float(self.input.dec_deg)) or not -90.0 <= float(self.input.dec_deg) <= 90.0:
            raise ValueError("dec_deg must be finite and in [-90, 90]")
        if self.input.trigger_time_utc is not None:
            try:
                Time(self.input.trigger_time_utc, scale="utc")
            except Exception as exc:
                raise ValueError("trigger_time_utc must be a valid UTC time") from exc

    def _files(self, context: Mapping[str, StageResult]) -> WXTObservationFiles:
        data = context["discover"].data
        path_fields = {
            "cleaned_event", "exposure_map", "exposure_correction", "arm_region",
            "rmf", "arf", "gti", "source_region", "background_region", "source_catalog",
            "pipeline_source_pha", "pipeline_background_pha",
        }
        payload = {
            key: (Path(value) if key in path_fields and value is not None else value)
            for key, value in data.items()
        }
        return WXTObservationFiles(**payload)

    def _prefix(self, files: WXTObservationFiles) -> str:
        detector = files.detector.replace("CMOS", "CMOS")
        return _safe_token(f"ep{files.obsid}_wxt{detector}_{self.input.target_id}")

    def _extract(
        self,
        files: WXTObservationFiles,
        *,
        outdir: Path,
        label: str,
        role: str,
        products: Sequence[str],
        regions: Sequence[Path],
        time_range: tuple[float, float] | None = None,
        stage: str,
        filter_energy_band: bool = True,
        band_name: str = "full",
    ) -> XSelectRunResult:
        band = self.config.extraction.bands.get(band_name)
        if band is None:
            raise ValueError(f"Unknown extraction band: {band_name}")
        pha_range = band.pi_range if band is not None and filter_energy_band else None
        with self.stage_environment(stage) as env:
            return self.extractor(
                files.cleaned_event,
                outdir,
                products=products,
                prefix=self._prefix(files),
                label=label,
                role=role,
                time_range=time_range,
                time_format=self.config.extraction.time_format,
                pha_range=pha_range,
                region=regions,
                lc_binsize=(
                    self.config.extraction.lightcurve_binsize_s
                    if "lightcurve" in products else None
                ),
                image_binsize=(
                    self.config.extraction.image_binsize if "image" in products else None
                ),
                overwrite=True,
                env=env,
                timeout=(
                    self.config.extraction.timeout_s
                    or self.config.execution.command_timeout_s
                ),
            )

    def execute_stage(self, stage: PipelineStage, context: Mapping[str, StageResult]) -> StageResult:
        handler = getattr(self, f"_stage_{stage.name}")
        return handler(context)

    def _stage_discover(self, context) -> StageResult:
        files = discover_wxt_files(self.input)
        outputs = {
            key: str(value)
            for key, value in asdict(files).items()
            if isinstance(value, Path)
        }
        data = {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in asdict(files).items()
        }
        return StageResult(outputs=outputs, data=data)

    def _stage_galactic_absorption(self, context) -> StageResult:
        config = self.config.galactic_absorption
        cache = self.workspace / "galactic_absorption" / "nhtot.json"
        result = resolve_galactic_absorption(
            self.input.ra_deg,
            self.input.dec_deg,
            cache_path=cache,
            equinox=config.equinox,
            service=config.service,
            timeout_s=config.timeout_s,
            use_cache=config.use_cache,
            query=self.nhtot_query,
        )
        return StageResult(outputs={"nhtot": str(cache)}, data=result.to_dict())

    def _stage_pipeline_spectrum(self, context) -> StageResult:
        """Register and validate the source spectrum delivered in the input directory.

        This product is intentionally not rewritten: its background scaling and
        response links belong to the upstream WXT pipeline and must remain
        distinguishable from Jinwu's T100/T90 re-extractions.
        """
        files = self._files(context)
        source = files.pipeline_source_pha
        background = files.pipeline_background_pha
        if source is None or background is None:
            raise ValueError(
                "Input WXT directory has no complete pipeline spectrum bundle "
                "(source PHA, background PHA, RMF, ARF) for the selected source"
            )
        source_exp, source_backscal = _pha_values(source)
        background_exp, background_backscal = _pha_values(background)
        alpha_counts = (source_exp / background_exp) * (source_backscal / background_backscal)
        if not np.isfinite(alpha_counts) or alpha_counts <= 0.0:
            raise ValueError(f"Input pipeline PHA has invalid reconstructed alpha: {alpha_counts}")
        response_checks = _validate_ogip_bundle(
            source,
            background,
            files.rmf,
            files.arf,
            fit_energy_range_keV=(
                self.config.spectrum.fit_energy_range_keV or self.config.energy_range_keV
            ),
        )
        return StageResult(
            outputs={
                "source_pha": str(source),
                "background_pha": str(background),
                "rmf": str(files.rmf),
                "arf": str(files.arf),
            },
            data={
                "time_basis": "input_pipeline_native_gti",
                "source_exposure": source_exp,
                "background_exposure": background_exp,
                "alpha_counts": alpha_counts,
                "response_checks": response_checks,
            },
        )

    def _copy_or_generate_regions(self, files: WXTObservationFiles) -> tuple[Path, Path, str, str]:
        region_dir = self.workspace / "regions"
        region_dir.mkdir(parents=True, exist_ok=True)
        source_out = region_dir / "source.reg"
        background_out = region_dir / "background.reg"

        source_input = Path(self.input.source_region).expanduser().resolve() if self.input.source_region else files.source_region
        background_input = Path(self.input.background_region).expanduser().resolve() if self.input.background_region else files.background_region
        if source_input is not None:
            shutil.copy2(source_input, source_out)
            source_origin = "user" if self.input.source_region else "official"
        else:
            if self.input.ra_deg is None or self.input.dec_deg is None:
                raise ValueError("Generating a WXT source region requires ra_deg and dec_deg")
            radius = self.config.regions.source_radius_arcsec
            if radius is None:
                raise ValueError("WXT source_radius_arcsec is not configured")
            _write_source_region(source_out, self.input.ra_deg, self.input.dec_deg, radius)
            source_origin = "generated"

        if background_input is not None:
            shutil.copy2(background_input, background_out)
            background_origin = "user" if self.input.background_region else "official"
        else:
            if self.input.ra_deg is None or self.input.dec_deg is None:
                raise ValueError("Generating a WXT background region requires ra_deg and dec_deg")
            roll = _event_roll(files.cleaned_event)
            if roll is None:
                raise ValueError("Generating a WXT background region requires PA_PNT/ROLL")
            _write_background_region(
                background_out,
                self.input.ra_deg,
                self.input.dec_deg,
                roll,
                self.config.regions.background_sectors,
            )
            background_origin = "generated"
        return source_out, background_out, source_origin, background_origin

    def _stage_regions(self, context) -> StageResult:
        files = self._files(context)
        source, background, source_origin, background_origin = self._copy_or_generate_regions(files)
        arm_shapes: tuple[str, ...] = ()
        effective_background = background
        if files.arm_region is not None:
            arm_shapes = _validate_arm_region(files.arm_region)
            effective_background = build_effective_ds9_region(
                background,
                (files.arm_region,),
                self.workspace / "regions" / "background_effective.reg",
                default_frame="physical",
            )
        region_manifest = _json_dump(
            self.workspace / "regions" / "regions.json",
            {
                "source": str(source),
                "background": str(background),
                "background_effective": str(effective_background),
                "arm": str(files.arm_region) if files.arm_region else None,
                "source_origin": source_origin,
                "background_origin": background_origin,
                "source_sha256": _file_hash(source),
                "background_sha256": _file_hash(background),
                "arm_sha256": _file_hash(files.arm_region) if files.arm_region else None,
                "background_effective_semantics": (
                    "background_minus_arm" if files.arm_region else "background"
                ),
                "background_extraction_regions": [str(effective_background)],
                "arm_exclusion_shape_count": len(arm_shapes),
            },
        )
        return StageResult(
            outputs={
                "source_region": str(source),
                "background_region": str(background),
                "background_effective_region": str(effective_background),
                "manifest": str(region_manifest),
            },
            data={"source_origin": source_origin, "background_origin": background_origin},
        )

    def _region_paths(self, context) -> tuple[Path, Path, list[Path]]:
        outputs = context["regions"].outputs
        source = Path(outputs["source_region"])
        background = Path(outputs["background_region"])
        effective_background = Path(outputs["background_effective_region"])
        return source, background, [effective_background]

    def _stage_provisional_events(self, context) -> StageResult:
        files = self._files(context)
        source_region, _, background_regions = self._region_paths(context)
        outdir = self.workspace / "provisional"
        src = self._extract(
            files, outdir=outdir, label="provisional_full", role="src",
            products=("events", "image"), regions=[source_region], stage="provisional_events_src",
        )
        bkg = self._extract(
            files, outdir=outdir, label="provisional_full", role="bkg",
            products=("events", "image"), regions=background_regions, stage="provisional_events_bkg",
        )
        required = {"source_event": src.events, "source_image": src.image, "background_event": bkg.events, "background_image": bkg.image}
        if any(value is None for value in required.values()):
            raise RuntimeError("XSELECT did not return all provisional WXT products")
        outputs = {key: str(value) for key, value in required.items() if value is not None}
        outputs.update(collect_xselect_artifacts(src).outputs("source"))
        outputs.update(collect_xselect_artifacts(bkg).outputs("background"))
        return StageResult(outputs=outputs)

    def _stage_exposure_arm_qc(self, context) -> StageResult:
        files = self._files(context)
        source_region, background_region, background_regions = self._region_paths(context)
        mode = self.config.background_scaling.mask_mode
        source = measure_region_exposure(files.exposure_map, source_region, mask_mode=mode)
        before = measure_region_exposure(files.exposure_map, background_region, mask_mode=mode)
        background = measure_region_exposure(
            files.exposure_map,
            background_region,
            exclusion_regions=(files.arm_region,) if files.arm_region else (),
            mask_mode=mode,
        )
        alpha = source.exposure_sum / background.exposure_sum
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError(f"Invalid WXT exposure-map alpha: {alpha}")
        scaling = BackgroundScalingResult(
            alpha=alpha,
            source=source,
            background=background,
            background_before_arm=before,
            arm_excluded_exposure=max(before.exposure_sum - background.exposure_sum, 0.0),
        )
        qc = _json_dump(self.workspace / "regions" / "exposure_qc.json", asdict(scaling))
        needs_review = (
            self.config.regions.require_review
            and not self.input.auto_approve_regions
            and not self.approval_path.is_file()
        )
        low_coverage = min(source.coverage_fraction, background.coverage_fraction) < self.config.regions.minimum_coverage_fraction
        if self.input.auto_approve_regions and not self.approval_path.exists():
            self.approve_regions(note="auto-approved by WXTPointingInput")
        data = asdict(scaling)
        data["low_coverage_warning"] = low_coverage
        if needs_review:
            return StageResult(
                status=PipelineStatus.NEEDS_REVIEW,
                outputs={"qc": str(qc), "region_manifest": context["regions"].outputs["manifest"]},
                data=data,
                message="Review source/background/ARM exposure diagnostics and call approve_regions()",
            )
        outputs = {"qc": str(qc)}
        if self.approval_path.is_file():
            outputs["approval"] = str(self.approval_path)
        return StageResult(outputs=outputs, data=data)

    def _stage_final_events(self, context) -> StageResult:
        files = self._files(context)
        source_region, _, background_regions = self._region_paths(context)
        alpha = float(context["exposure_arm_qc"].data["alpha"])
        source_sum = float(context["exposure_arm_qc"].data["source"]["exposure_sum"])
        background_sum = float(context["exposure_arm_qc"].data["background"]["exposure_sum"])
        outdir = self.workspace / "events"
        src = self._extract(
            files, outdir=outdir, label="search_full", role="src",
            products=("events",), regions=[source_region], stage="final_events_src",
        )
        bkg = self._extract(
            files, outdir=outdir, label="search_full", role="bkg",
            products=("events",), regions=background_regions, stage="final_events_bkg",
        )
        if src.events is None or bkg.events is None:
            raise RuntimeError("XSELECT did not return final source/background events")
        _write_event_scaling(src.events, alpha, source_sum)
        _write_event_scaling(bkg.events, 1.0, background_sum)
        outputs = {"source_event": str(src.events), "background_event": str(bkg.events)}
        outputs.update(collect_xselect_artifacts(src).outputs("source"))
        outputs.update(collect_xselect_artifacts(bkg).outputs("background"))
        return StageResult(outputs=outputs, data={"alpha": alpha})

    def _stage_lightcurves(self, context) -> StageResult:
        files = self._files(context)
        source_region, _, background_regions = self._region_paths(context)
        alpha = float(context["exposure_arm_qc"].data["alpha"])
        duration = context["duration"].data
        t0 = duration.get("t0") or {}
        t0_met = t0.get("mission_time_s")
        try:
            timezero = float(t0_met)
        except (TypeError, ValueError):
            timezero = math.nan
        if not np.isfinite(timezero):
            raise RuntimeError("Duration stage did not provide a finite T0 mission time for light curves")
        time_axis = dict(duration.get("time_axis") or {})
        time_anchor = None
        utc = t0.get("utc")
        if utc is not None:
            try:
                time_anchor = Time(str(utc), scale="utc")
            except Exception:
                time_axis["relative_seconds_only"] = True
                time_axis["warning"] = (
                    "Duration UTC anchor could not be reconstructed; "
                    "light curves use relative T0 seconds without a UTC label."
                )
        outputs: dict[str, str] = {}
        bands_data: dict[str, Any] = {}
        for band_name, band in self.config.extraction.bands.items():
            outdir = self.workspace / "lightcurves" / band_name
            src = self._extract(
                files, outdir=outdir, label=f"{band_name}_fixed", role="src",
                products=("lightcurve",), regions=[source_region],
                stage=f"lightcurve_{band_name}_src", band_name=band_name,
            )
            bkg = self._extract(
                files, outdir=outdir, label=f"{band_name}_fixed", role="bkg",
                products=("lightcurve",), regions=background_regions,
                stage=f"lightcurve_{band_name}_bkg", band_name=band_name,
            )
            if src.lightcurve is None or bkg.lightcurve is None:
                raise RuntimeError(f"XSELECT did not return {band_name} light curves")
            net = build_net_lightcurve(src.lightcurve, bkg.lightcurve, alpha=alpha)
            base = outdir / f"{self._prefix(files)}_{band_name}_net_lc"
            saved = save_net_lightcurve(net, base)
            plots = ()
            if self.config.plotting.enabled:
                plots = plot_net_lightcurve(
                    net, base.with_name(base.name + "_plot"),
                    title=f"{self.input.target_id} WXT {band_name} count light curve",
                    timezero=timezero,
                    timezero_obj=time_anchor,
                    formats=self.config.plotting.formats,
                    dpi=self.config.plotting.dpi,
                )
                if self.config.plotting.required and len(plots) != len(self.config.plotting.formats):
                    raise RuntimeError(f"Required {band_name} light-curve plots were not produced")
            prefix = f"{band_name}_source"
            outputs.update(collect_xselect_artifacts(src).outputs(prefix))
            outputs.update(collect_xselect_artifacts(bkg).outputs(f"{band_name}_background"))
            outputs.update({f"{band_name}_{key}": str(path) for key, path in saved.items()})
            outputs.update({f"{band_name}_plot_{path.suffix.lstrip('.')}": str(path) for path in plots})
            bands_data[band_name] = {
                "energy_range_keV": list(band.energy_range_keV),
                "pi_range": list(band.pi_range) if band.pi_range else None,
                "alpha": alpha,
                "net_ecsv": str(saved["net_ecsv"]),
                "time_axis": {
                    **time_axis,
                    "t0_met": timezero,
                    "t0_utc": utc,
                    "anchor_source": t0.get("source"),
                },
            }
        manifest = _json_dump(self.workspace / "lightcurves" / "lightcurves.json", bands_data)
        outputs["manifest"] = str(manifest)
        return StageResult(outputs=outputs, data={"bands": bands_data})

    def _stage_duration(self, context) -> StageResult:
        from ...core.ops import txx

        source = Path(context["final_events"].outputs["source_event"])
        background = Path(context["final_events"].outputs["background_event"])
        alpha = float(context["exposure_arm_qc"].data["alpha"])
        cfg = self.config.duration
        duration = txx(
            source,
            background=background,
            alpha=alpha,
            p0=cfg.p0,
            block_snr_threshold=cfg.block_snr_threshold,
            cumulative_mode=cfg.cumulative_mode,
            evt_binsize=cfg.event_binsize_s,
            nmc=cfg.nmc,
            seed=cfg.seed,
        )
        start = float(duration.get("t90_tstart", math.nan))
        stop = float(duration.get("t90_tstop", math.nan))
        if not np.isfinite(start) or not np.isfinite(stop) or stop <= start:
            raise RuntimeError("Txx did not produce a finite positive T90 interval")
        duration["negative_net_policy"] = "clip_to_zero"
        duration["evt_binsize"] = cfg.event_binsize_s
        duration["diagnostic_min_t100_bins"] = cfg.diagnostic_min_t100_bins
        duration["focus_t100"] = cfg.focus_t100
        duration["diagnostic_context_fraction"] = cfg.diagnostic_context_fraction
        duration["t0"] = _pipeline_t0(
            source,
            self.input.trigger_time_utc,
            float(duration["t100_tstart"]),
            self.config.mission,
        )
        time_anchor = None
        time_warning = None
        if self.input.trigger_time_utc is not None:
            time_anchor = Time(self.input.trigger_time_utc, scale="utc")
        else:
            time_anchor = time_from_mission_seconds(
                self.config.mission, duration["t0"]["mission_time_s"]
            )
            if time_anchor is None:
                time_warning = (
                    f"No registered MET-to-UTC conversion for mission "
                    f"{self.config.mission!r}; duration plots use relative seconds."
                )
        duration["time_axis"] = {
            "anchor_source": duration["t0"]["source"],
            "mission": self.config.mission,
            "time_format": mission_time_format(self.config.mission),
            "utc": duration["t0"]["utc"],
            "relative_seconds_only": time_anchor is None,
            "warning": time_warning,
            "fits_time_reference": _time_reference_provenance(
                source, time_anchor, duration["t0"]["mission_time_s"]
            ),
        }
        path = _json_dump(self.workspace / "duration" / "duration.json", duration)
        outputs = {"duration": str(path)}
        if self.config.plotting.enabled:
            from ...core.plot import plot_event_txx

            plot_base = self.workspace / "duration" / f"{self.input.target_id}_wxt_txx"
            t0_seconds = duration["t0"].get("mission_time_s")
            plot_timezero = time_anchor if time_anchor is not None else t0_seconds
            for extension in self.config.plotting.formats:
                plot_path = plot_base.with_suffix(f".{extension}")
                figure, _ = plot_event_txx(
                    source,
                    duration,
                    background=background,
                    alpha=alpha,
                    srcname=self.input.target_id,
                    title=f"{self.input.target_id} WXT duration",
                    out=plot_path,
                    timezero=plot_timezero,
                    min_t100_bins=cfg.diagnostic_min_t100_bins,
                    focus_t100=cfg.focus_t100,
                    t100_context_fraction=cfg.diagnostic_context_fraction,
                )
                try:
                    import matplotlib.pyplot as plt

                    plt.close(figure)
                except Exception:
                    pass
                outputs[f"plot_{extension}"] = str(plot_path)
            if self.config.plotting.required:
                missing = [value for key, value in outputs.items() if key.startswith("plot_") and not Path(value).is_file()]
                if missing:
                    raise RuntimeError(f"Required duration plots were not produced: {missing}")
        return StageResult(outputs=outputs, data=json.loads(path.read_text()))

    def _extract_duration_spectrum(
        self,
        context: Mapping[str, StageResult],
        *,
        label: str,
        start: float,
        stop: float,
    ) -> StageResult:
        files = self._files(context)
        source_region, _, background_regions = self._region_paths(context)
        outdir = self.workspace / "spectra" / label
        src = self._extract(
            files, outdir=outdir, label=f"{label}_full", role="src",
            products=("spectrum",), regions=[source_region], time_range=(start, stop),
            stage=f"{label}_spectrum_src", filter_energy_band=False,
        )
        bkg = self._extract(
            files, outdir=outdir, label=f"{label}_full", role="bkg",
            products=("spectrum",), regions=background_regions, time_range=(start, stop),
            stage=f"{label}_spectrum_bkg", filter_energy_band=False,
        )
        if src.spectrum is None or bkg.spectrum is None:
            raise RuntimeError(f"XSELECT did not return {label} source/background PHA")
        alpha = float(context["exposure_arm_qc"].data["alpha"])
        _, source_backscal = _pha_values(src.spectrum)
        _set_pha_backscal(bkg.spectrum, source_backscal / alpha)
        outputs = {"source_pha": str(src.spectrum), "background_pha": str(bkg.spectrum)}
        outputs.update(collect_xselect_artifacts(src).outputs("source"))
        outputs.update(collect_xselect_artifacts(bkg).outputs("background"))
        return StageResult(outputs=outputs, data={"start": start, "stop": stop, "time_basis": label})

    def _stage_t100_spectra(self, context) -> StageResult:
        duration = context["duration"].data
        return self._extract_duration_spectrum(
            context,
            label="t100",
            start=float(duration["t100_tstart"]),
            stop=float(duration["t100_tstop"]),
        )

    def _stage_t90_spectra(self, context) -> StageResult:
        duration = context["duration"].data
        return self._extract_duration_spectrum(
            context,
            label="t90",
            start=float(duration["t90_tstart"]),
            stop=float(duration["t90_tstop"]),
        )

    def _stage_ogip_finalize(self, context) -> StageResult:
        files = self._files(context)
        alpha = float(context["exposure_arm_qc"].data["alpha"])
        source_events = Path(context["final_events"].outputs["source_event"])
        background_events = Path(context["final_events"].outputs["background_event"])
        output: dict[str, str] = {}
        products: dict[str, Any] = {}
        for label in ("t100", "t90"):
            stage = context[f"{label}_spectra"]
            source, background, rmf, arf = _finalize_pha_pair(
                Path(stage.outputs["source_pha"]),
                Path(stage.outputs["background_pha"]),
                alpha=alpha,
                rmf=files.rmf,
                arf=files.arf,
                alpha_rtol=self.config.background_scaling.alpha_rtol,
                exposure_rtol=self.config.background_scaling.exposure_rtol,
            )
            start = float(stage.data["start"])
            stop = float(stage.data["stop"])
            count_checks = {
                "source_event_counts": _event_interval_count(source_events, start, stop),
                "source_pha_counts": _pha_total_counts(
                    source, self.config.extraction.bands["full"].pi_range
                ),
                "background_event_counts": _event_interval_count(background_events, start, stop),
                "background_pha_counts": _pha_total_counts(
                    background, self.config.extraction.bands["full"].pi_range
                ),
            }
            if count_checks["source_event_counts"] != count_checks["source_pha_counts"]:
                raise ValueError(f"{label} source event/PHA count mismatch: {count_checks}")
            if count_checks["background_event_counts"] != count_checks["background_pha_counts"]:
                raise ValueError(f"{label} background event/PHA count mismatch: {count_checks}")
            n_on = float(count_checks["source_event_counts"])
            n_off = float(count_checks["background_event_counts"])
            count_checks.update(
                {
                    "n_on": n_on,
                    "n_off": n_off,
                    "alpha": alpha,
                    "net_counts": n_on - alpha * n_off,
                    "significance": li_ma_snr(n_on, n_off, alpha, signed=True),
                }
            )
            response_checks = _validate_ogip_bundle(
                source,
                background,
                rmf,
                arf,
                fit_energy_range_keV=(
                    self.config.spectrum.fit_energy_range_keV or self.config.energy_range_keV
                ),
            )
            output.update(
                {
                    f"{label}_source_pha": str(source),
                    f"{label}_background_pha": str(background),
                    f"{label}_rmf": str(rmf),
                    f"{label}_arf": str(arf),
                }
            )
            products[label] = {
                "start": start,
                "stop": stop,
                "count_checks": count_checks,
                "response_checks": response_checks,
            }
        output.update(
            {
                "source_pha": output["t90_source_pha"],
                "background_pha": output["t90_background_pha"],
                "rmf": output["t90_rmf"],
                "arf": output["t90_arf"],
            }
        )
        return StageResult(
            outputs=output,
            data={
                "alpha": alpha,
                "products": products,
                "count_checks": products["t90"]["count_checks"],
                "response_checks": products["t90"]["response_checks"],
            },
        )

    def _stage_bayesian_block_spectra(self, context) -> StageResult:
        cfg = self.config.bayesian_block_spectra
        segments_path = self.workspace / "spectra" / "bayesian_blocks" / "segments.json"
        if not cfg.enabled:
            _json_dump(segments_path, [])
            return StageResult(outputs={"segments": str(segments_path)}, data={"segments": []})
        duration = context["duration"].data
        source_event = Path(context["final_events"].outputs["source_event"])
        background_event = Path(context["final_events"].outputs["background_event"])
        alpha = float(context["exposure_arm_qc"].data["alpha"])
        segments = merge_bayesian_blocks_for_spectra(
            duration["bb_edges_time"],
            _event_times(source_event),
            _event_times(background_event),
            alpha=alpha,
            t_start=float(duration["t90_tstart"]),
            t_stop=float(duration["t90_tstop"]),
            minimum_net_counts=cfg.minimum_net_counts,
            minimum_significance=cfg.minimum_significance,
        )
        files = self._files(context)
        source_region, _, background_regions = self._region_paths(context)
        outputs: dict[str, str] = {"segments": str(segments_path)}
        segment_payload = []
        for segment in segments:
            label = f"bb{segment.index:03d}_full"
            outdir = self.workspace / "spectra" / "bayesian_blocks" / f"bb{segment.index:03d}"
            src = self._extract(
                files, outdir=outdir, label=label, role="src", products=("spectrum",),
                regions=[source_region], time_range=(segment.start, segment.stop),
                stage=f"bb{segment.index:03d}_src", filter_energy_band=False,
            )
            bkg = self._extract(
                files, outdir=outdir, label=label, role="bkg", products=("spectrum",),
                regions=background_regions, time_range=(segment.start, segment.stop),
                stage=f"bb{segment.index:03d}_bkg", filter_energy_band=False,
            )
            if src.spectrum is None or bkg.spectrum is None:
                raise RuntimeError(f"XSELECT did not return PHA for {label}")
            source, background, rmf, arf = _finalize_pha_pair(
                src.spectrum, bkg.spectrum, alpha=alpha, rmf=files.rmf, arf=files.arf,
                alpha_rtol=self.config.background_scaling.alpha_rtol,
                exposure_rtol=self.config.background_scaling.exposure_rtol,
            )
            key = f"bb{segment.index:03d}"
            outputs[f"{key}_source_pha"] = str(source)
            outputs[f"{key}_background_pha"] = str(background)
            outputs[f"{key}_rmf"] = str(rmf)
            outputs[f"{key}_arf"] = str(arf)
            outputs.update(collect_xselect_artifacts(src).outputs(f"{key}_source"))
            outputs.update(collect_xselect_artifacts(bkg).outputs(f"{key}_background"))
            item = asdict(segment)
            item.update({"source_pha": str(source), "background_pha": str(background), "rmf": str(rmf), "arf": str(arf)})
            segment_payload.append(item)
        _json_dump(segments_path, segment_payload)
        return StageResult(outputs=outputs, data={"segments": segment_payload})

    def _fit_one(
        self,
        label: str,
        source: Path,
        background: Path,
        rmf: Path,
        arf: Path,
        *,
        galactic_nh_1e22: float,
        candidate_keys: Sequence[str] | None = None,
    ):
        from ...core.fit import fit_xray_models
        from ...core.spectrum_prep import prepare_spectra

        source_file = DataFile(source, "source_pha", "jinwu_pipeline", "EP", "WXT", source_id=label)
        background_file = DataFile(background, "background_pha", "jinwu_pipeline", "EP", "WXT", source_id=label)
        arf_file = DataFile(arf, "arf", "jinwu_pipeline", "EP", "WXT", source_id=label)
        rmf_file = DataFile(rmf, "rmf", "jinwu_pipeline", "EP", "WXT", source_id=label)
        bundle = SpectrumBundle(
            source_file, background_file, arf_file, rmf_file,
            None, None, detector="WXT", source_id=label,
        )
        fit_dir = self.workspace / "fit" / label
        catalog = Catalog(
            root=source.parent,
            manifests=[
                Manifest(
                    root=source.parent,
                    mission="EP",
                    instrument="WXT",
                    layout="jinwu_wxt_pipeline",
                    detector="WXT",
                    files=[source_file, background_file, arf_file, rmf_file],
                    bundles=[bundle],
                )
            ],
        )
        prepared_catalog = prepare_spectra(
            catalog,
            outdir=fit_dir / "prepared",
            group_min=self.config.spectrum.group_min_counts,
            overwrite=True,
        )
        if not prepared_catalog.spectra or not prepared_catalog.spectra[0].ready:
            detail = "; ".join(prepared_catalog.diagnostics) or "unknown preparation failure"
            raise RuntimeError(f"Unable to prepare {label} spectrum: {detail}")
        prepared = prepared_catalog.spectra[0]
        return fit_xray_models(
            prepared,
            outdir=fit_dir,
            model_class=self.config.fitting.model_class,
            absorption_mode=self.config.fitting.absorption_mode,
            candidate_keys=(
                candidate_keys
                if candidate_keys is not None
                else self.config.fitting.candidate_keys
            ),
            selection_metric=self.config.fitting.selection_metric,
            stat_method=self.config.fitting.statistic,
            abundance=self.config.fitting.abundance,
            cross_section=self.config.fitting.cross_section,
            galactic_nh_1e22=galactic_nh_1e22,
            redshift=self.input.redshift,
            calculate_errors=self.config.fitting.calculate_errors,
            error_delta_stat=self.config.fitting.error_delta_stat,
            srcname=self.input.target_id,
            instname=f"WXT_{label}",
            plot_formats=(
                self.config.plotting.formats if self.config.plotting.enabled else ()
            ),
            plot_density=self.config.plotting.dpi,
            plot_required=self.config.plotting.enabled and self.config.plotting.required,
        )

    def _stage_fit(self, context) -> StageResult:
        summary_path = self.workspace / "fit" / "fit_summary.json"
        if not self.config.fitting.enabled:
            _json_dump(summary_path, {"enabled": False, "fits": {}})
            return StageResult(outputs={"summary": str(summary_path)}, data={"fits": {}})
        galactic_nh = float(context["galactic_absorption"].data["tbabs_nh_1e22"])
        ogip = context["ogip_finalize"].outputs
        comparisons: dict[str, Any] = {}

        def run_comparison(label: str, source: Path, background: Path, rmf: Path, arf: Path):
            result = self._fit_one(
                label, source, background, rmf, arf,
                galactic_nh_1e22=galactic_nh,
            )
            comparisons[label] = result
            return result

        pipeline_spectrum = context["pipeline_spectrum"].outputs
        run_comparison(
            "pipeline",
            Path(pipeline_spectrum["source_pha"]),
            Path(pipeline_spectrum["background_pha"]),
            Path(pipeline_spectrum["rmf"]),
            Path(pipeline_spectrum["arf"]),
        )
        run_comparison(
            "t100",
            Path(ogip["t100_source_pha"]),
            Path(ogip["t100_background_pha"]),
            Path(ogip["t100_rmf"]),
            Path(ogip["t100_arf"]),
        )
        run_comparison(
            "t90", Path(ogip["source_pha"]), Path(ogip["background_pha"]),
            Path(ogip["rmf"]), Path(ogip["arf"]),
        )

        def adopted_fit(value: Any) -> dict[str, Any]:
            return value.adopted_fit if hasattr(value, "adopted_fit") else value

        t90_adopted = adopted_fit(comparisons["t90"])
        adopted_key = t90_adopted.get("model_key")
        for item in context["bayesian_block_spectra"].data.get("segments", []):
            key = f"bb{int(item['index']):03d}"
            comparisons[key] = self._fit_one(
                key, Path(item["source_pha"]), Path(item["background_pha"]),
                Path(item["rmf"]), Path(item["arf"]),
                galactic_nh_1e22=galactic_nh,
                candidate_keys=(adopted_key,) if adopted_key else None,
            )

        fit_results = {key: adopted_fit(value) for key, value in comparisons.items()}
        compact = {
            key: {
                "statistics": value.get("statistics"),
                "model": value.get("model"),
                "parameters": value.get("parameters"),
                "flux_abs": value.get("flux_abs"),
                "rate": value.get("rate"),
                "fit_products": value.get("fit_products"),
                "report_txt": value.get("report_txt"),
                "plot_fit": value.get("plot_fit"),
                "plot_fits": value.get("plot_fits"),
                "warnings": value.get("warnings"),
                "xspec_settings": value.get("xspec_settings"),
                "model_key": value.get("model_key"),
                "model_family": value.get("model_family"),
                "absorption_mode": value.get("absorption_mode"),
                "metrics": value.get("metrics"),
                "derived_parameters": value.get("derived_parameters"),
                "requested_statistic": self.config.fitting.statistic,
                "effective_statistic": value.get("effective_statistic", "wstat"),
            }
            for key, value in fit_results.items()
        }
        comparison_payload = {
            key: (
                value.to_dict()
                if hasattr(value, "to_dict")
                else {
                    "adopted_key": compact[key].get("model_key"),
                    "adopted_reason": "legacy single-model fit",
                    "candidates": {compact[key].get("model_key") or "legacy": compact[key]},
                }
            )
            for key, value in comparisons.items()
        }
        failure_logs: dict[str, dict[str, str]] = {}
        for label, value in comparisons.items():
            for candidate_key in getattr(value, "failures", {}):
                path = self.workspace / "fit" / label / "models" / candidate_key / "fit_failure.log"
                if path.is_file():
                    failure_logs.setdefault(label, {})[candidate_key] = str(path)
        _json_dump(
            summary_path,
            {
                "enabled": True,
                "fits": compact,
                "model_comparisons": comparison_payload,
                "failure_logs": failure_logs,
            },
        )
        outputs = {"summary": str(summary_path)}
        for key, value in compact.items():
            if value.get("report_txt"):
                outputs[f"{key}_report"] = str(value["report_txt"])
            products = value.get("fit_products") or {}
            for product_key, product_value in products.items():
                if product_key == "plots":
                    for index, path in enumerate(product_value):
                        outputs[f"{key}_plot_{index}"] = str(path)
                elif product_value:
                    outputs[f"{key}_{product_key}"] = str(product_value)
        for label, value in comparisons.items():
            if hasattr(value, "comparison_json") and value.comparison_json:
                outputs[f"{label}_model_comparison_json"] = str(value.comparison_json)
            if hasattr(value, "comparison_txt") and value.comparison_txt:
                outputs[f"{label}_model_comparison_txt"] = str(value.comparison_txt)
            for candidate_key, candidate in getattr(value, "candidates", {}).items():
                products = candidate.get("fit_products") or {}
                for product_key, product_value in products.items():
                    if product_key == "plots":
                        for index, path in enumerate(product_value):
                            outputs[f"{label}_{candidate_key}_plot_{index}"] = str(path)
                    elif product_value:
                        outputs[f"{label}_{candidate_key}_{product_key}"] = str(product_value)
                summary = self.workspace / "fit" / label / "models" / candidate_key / "summary_zh.txt"
                if summary.is_file():
                    outputs[f"{label}_{candidate_key}_summary_zh"] = str(summary)
            for candidate_key in getattr(value, "failures", {}):
                candidate_dir = self.workspace / "fit" / label / "models" / candidate_key
                failure_json = candidate_dir / "fit_failure.json"
                failure_summary = candidate_dir / "summary_zh.txt"
                failure_log = candidate_dir / "fit_failure.log"
                if failure_json.is_file():
                    outputs[f"{label}_{candidate_key}_failure_json"] = str(failure_json)
                if failure_summary.is_file():
                    outputs[f"{label}_{candidate_key}_summary_zh"] = str(failure_summary)
                if failure_log.is_file():
                    outputs[f"{label}_{candidate_key}_xspec_log"] = str(failure_log)
        return StageResult(
            outputs=outputs,
            data={
                "fits": compact,
                "model_comparisons": comparison_payload,
                "failure_logs": failure_logs,
            },
        )

    def _stage_fluxcurve(self, context) -> StageResult:
        output_base = self.workspace / "fluxcurve" / f"{self.input.target_id}_wxt_fluxcurve"
        fits_by_label = context["fit"].data.get("fits", {})
        if not self.config.flux_curve.enabled or "t90" not in fits_by_label:
            status = "disabled" if not self.config.flux_curve.enabled else "fit_unavailable"
            manifest = _json_dump(output_base.with_suffix(".json"), {"status": status})
            return StageResult(outputs={"manifest": str(manifest)}, data={"status": status})

        energy_range = (
            self.config.flux_curve.energy_range_keV
            or self.config.spectrum.fit_energy_range_keV
            or self.config.energy_range_keV
        )
        curve = build_flux_curve(
            fits_by_label,
            context["bayesian_block_spectra"].data.get("segments", []),
            context["duration"].data,
            energy_range_keV=energy_range,
            include_t90_aggregate=self.config.flux_curve.include_t90_aggregate,
        )
        quicklook = ()
        if self.config.flux_curve.build_quicklook:
            full_path = context["lightcurves"].data["bands"]["full"]["net_ecsv"]
            net_curve = load_net_lightcurve(full_path)
            aggregate = curve.aggregate_points[0] if curve.aggregate_points else None
            if aggregate is not None:
                quicklook = build_quicklook_flux_curve(
                    net_curve,
                    t90_flux=aggregate.flux,
                    t90_start=float(context["duration"].data["t90_tstart"]),
                    t90_stop=float(context["duration"].data["t90_tstop"]),
                    energy_range_keV=energy_range,
                    model_key=fits_by_label["t90"].get("model_key"),
                    model_expression=fits_by_label["t90"].get("model"),
                )
        curve = FluxCurveResult(
            science_points=curve.science_points,
            aggregate_points=curve.aggregate_points,
            quicklook_points=quicklook,
            status=curve.status,
        )
        outputs_saved = save_flux_curve(
            curve,
            output_base,
            formats=self.config.plotting.formats if self.config.plotting.enabled else (),
            dpi=self.config.plotting.dpi,
            title=f"{self.input.target_id} WXT unabsorbed 0.5-4 keV flux curve",
        )
        if self.config.plotting.enabled and self.config.plotting.required:
            missing_formats = [
                extension for extension in self.config.plotting.formats
                if f"plot_{extension}" not in outputs_saved
            ]
            if missing_formats:
                raise RuntimeError(f"Required flux-curve plots were not produced: {missing_formats}")
        positive_quicklook = [
            point
            for point in curve.quicklook_points
            if np.isfinite(point.flux) and point.flux > 0.0
        ]
        quicklook_peak = max(positive_quicklook, key=lambda point: point.flux, default=None)
        return StageResult(
            outputs={key: str(path) for key, path in outputs_saved.items()},
            data={
                "status": curve.status,
                "n_science_points": len(curve.science_points),
                "n_aggregate_points": len(curve.aggregate_points),
                "n_quicklook_points": len(curve.quicklook_points),
                "science_points": [asdict(point) for point in curve.science_points],
                "aggregate_points": [asdict(point) for point in curve.aggregate_points],
                "quicklook_peak_flux": quicklook_peak.flux if quicklook_peak else None,
                "quicklook_peak_time": quicklook_peak.time if quicklook_peak else None,
                "quicklook_peak_significance": (
                    quicklook_peak.significance if quicklook_peak else None
                ),
            },
        )

    def _stage_report(self, context) -> StageResult:
        report = self.workspace / "report" / "wxt_pipeline_report.json"
        payload = {
            "target_id": self.input.target_id,
            "config": asdict(self.config),
            "stages": {
                name: {"outputs": result.outputs, "data": result.data}
                for name, result in context.items()
            },
            "environment": collect_runtime_environment(),
        }
        fits = context["fit"].data.get("fits", {})
        t90_fit = fits.get("t90", {})
        products = context["ogip_finalize"].data.get("products", {})
        fit_intervals: dict[str, dict[str, Any]] = {
            "pipeline": {
                "display_name": "输入目录原始 pipeline 整段观测谱",
                "description": (
                    "原始 WXT pipeline GTI；"
                    f"源谱曝光 {context['pipeline_spectrum'].data.get('source_exposure', 'N/A')} s"
                ),
            },
            "t100": {
                "display_name": "T100 时段谱",
                "description": "按自动 T100 起止时间抽取",
                "start_met": products.get("t100", {}).get("start"),
                "stop_met": products.get("t100", {}).get("stop"),
            },
            "t90": {
                "display_name": "T90 时段谱",
                "description": "按自动 T90 起止时间抽取",
                "start_met": products.get("t90", {}).get("start"),
                "stop_met": products.get("t90", {}).get("stop"),
            },
        }
        for segment in context["bayesian_block_spectra"].data.get("segments", []):
            label = f"bb{int(segment['index']):03d}"
            fit_intervals[label] = {
                "display_name": f"时间分辨谱 {label}",
                "description": (
                    f"Bayesian-block 合并段；净计数 {float(segment['net_counts']):.4g}；"
                    f"signed Li & Ma {float(segment['significance']):.4g} sigma"
                ),
                "start_met": segment.get("start"),
                "stop_met": segment.get("stop"),
            }
        discover = context["discover"].data
        summary_payload = {
            "target_id": self.input.target_id,
            "candidate_id": _candidate_id(
                str(discover.get("obsid")),
                str(discover.get("detector")),
                discover.get("source_id"),
            ),
            "obsid": discover.get("obsid"),
            "detector": discover.get("detector"),
            "source_id": discover.get("source_id"),
            "ra_deg": self.input.ra_deg,
            "dec_deg": self.input.dec_deg,
            "t0": context["duration"].data.get("t0", {}),
            "duration": context["duration"].data,
            "t90_spectrum": {
                "start_met": context["duration"].data.get("t90_tstart"),
                "stop_met": context["duration"].data.get("t90_tstop"),
                "t0_met": context["duration"].data.get("t0", {}).get("mission_time_s"),
            },
            "t90_counts": context["ogip_finalize"].data.get("count_checks", {}),
            "galactic_absorption": context["galactic_absorption"].data,
            "fit": t90_fit,
            "fits": fits,
            "model_comparisons": context["fit"].data.get("model_comparisons", {}),
            "fit_intervals": fit_intervals,
            "integrated_fits": {
                label: fits.get(label, {})
                for label in ("pipeline", "t100")
            },
            "flux_curve": context["fluxcurve"].data,
            "energy_band": list(
                self.config.spectrum.fit_energy_range_keV or self.config.energy_range_keV
            ),
            "redshift": self.input.redshift,
            "outputs": {
                "report": str(report),
                "workspace": str(self.workspace),
                "spectrum_plot": next(
                    iter(t90_fit.get("plot_fits") or []), None
                ),
                "flux_curve": context["fluxcurve"].outputs.get("plot_png")
                or context["fluxcurve"].outputs.get("ecsv"),
            },
        }
        report_dir = self.workspace / "report"
        report_dir.mkdir(parents=True, exist_ok=True)
        detail_text = render_observation_summary(summary_payload)
        detail_txt = report_dir / "summary_detail_zh.txt"
        detail_txt.write_text(detail_text + "\n", encoding="utf-8")

        wechat_messages = render_wechat_fit_messages(summary_payload)
        wechat_dir = report_dir / "wechat"
        wechat_dir.mkdir(parents=True, exist_ok=True)
        wechat_paths: dict[str, str] = {}
        candidate_blocks: list[str] = []
        adopted_blocks: list[str] = []
        comparisons = summary_payload.get("model_comparisons") or {}
        for message_key, message in wechat_messages.items():
            label, candidate_key = message_key.split("__", 1)
            selected = str((comparisons.get(label) or {}).get("adopted_key") or "")
            if not selected:
                selected = str((fits.get(label) or {}).get("model_key") or "")
            if label == "event":
                status = "无拟合"
            else:
                status = "自动采用" if candidate_key == selected else "候选对比"
            interval_name = str(
                (fit_intervals.get(label) or {}).get("display_name")
                or ("未生成能谱拟合" if label == "event" else label)
            )
            safe_key = _safe_token(message_key)
            message_path = wechat_dir / f"{safe_key}_zh.txt"
            message_path.write_text(message + "\n", encoding="utf-8")
            wechat_paths[message_key] = str(message_path)
            block = f"【{interval_name} | {candidate_key} | {status}】\n{message}"
            if label != "event" and candidate_key == selected:
                adopted_blocks.append(block)
            else:
                candidate_blocks.append(block)

        summary_sections: list[str] = []
        if candidate_blocks:
            summary_sections.append("【候选模型对比】\n\n" + "\n\n".join(candidate_blocks))
        if adopted_blocks:
            summary_sections.append("【自动采用模型】\n\n" + "\n\n".join(adopted_blocks))
        summary_text = "\n\n".join(summary_sections)
        summary_txt = self.workspace / "report" / "summary_zh.txt"
        summary_txt.write_text(summary_text + "\n", encoding="utf-8")
        summary_payload["outputs"]["summary_detail_zh"] = str(detail_txt)
        summary_payload["outputs"]["summary_zh"] = str(summary_txt)
        summary_payload["outputs"]["wechat_messages"] = wechat_paths
        summary_json = _json_dump(report_dir / "summary.json", summary_payload)

        report_products = {
            "summary_json": str(summary_json),
            "summary_txt": str(summary_txt),
            "summary_detail_txt": str(detail_txt),
            **{f"wechat_{_safe_token(key)}": value for key, value in wechat_paths.items()},
        }
        payload["report_products"] = report_products
        payload["artifacts"] = build_artifact_index(
            {
                **{name: result.outputs for name, result in context.items()},
                "report_products": report_products,
            }
        )
        _json_dump(report, payload)
        if self.config.reporting.print_summary:
            print(summary_text)
        return StageResult(
            outputs={
                "report": str(report),
                "summary_json": str(summary_json),
                "summary_txt": str(summary_txt),
                "summary_detail_txt": str(detail_txt),
                **{
                    f"wechat_{_safe_token(key)}": value
                    for key, value in wechat_paths.items()
                },
            },
            data={
                "summary": summary_text,
                "summary_detail": detail_text,
                "wechat_messages": wechat_paths,
            },
        )

    def build_result(self, context: Mapping[str, StageResult]) -> WXTPointingResult:
        files = self._files(context) if "discover" in context else None
        alpha = None
        if "exposure_arm_qc" in context:
            alpha = float(context["exposure_arm_qc"].data["alpha"])
        duration = context.get("duration").data if "duration" in context else None
        ogip = context.get("ogip_finalize")
        segments = []
        if "bayesian_block_spectra" in context:
            segments = [
                TimeResolvedSegment(**{key: item[key] for key in (
                    "index", "start", "stop", "n_on", "n_off", "alpha", "net_counts", "significance"
                )})
                for item in context["bayesian_block_spectra"].data.get("segments", [])
            ]
        plot_groups: dict[str, tuple[Path, ...]] = {}

        def add_plot(title: str, candidates: Sequence[str | Path | None]) -> None:
            paths = [Path(path) for path in candidates if path]
            png = next((path for path in paths if path.suffix.lower() == ".png"), None)
            selected = png or next(
                (path for path in paths if path.suffix.lower() == ".svg"),
                None,
            )
            if selected is not None:
                plot_groups[title] = (selected,)

        if "duration" in context:
            add_plot(
                "T100/T90/T50 时标诊断",
                (context["duration"].outputs.get("plot_png"), context["duration"].outputs.get("plot_svg")),
            )
        if "lightcurves" in context:
            for band in ("full", "soft", "hard"):
                add_plot(
                    f"{band} 能段净计数光变",
                    (
                        context["lightcurves"].outputs.get(f"{band}_plot_png"),
                        context["lightcurves"].outputs.get(f"{band}_plot_svg"),
                    ),
                )
        if "fit" in context:
            fit_names = {
                "pipeline": "输入目录原始 pipeline 整段能谱拟合",
                "t100": "T100 时段能谱拟合",
                "t90": "T90 时段能谱拟合",
            }
            fit_data = context["fit"].data.get("fits", {})
            labels = [label for label in ("pipeline", "t100", "t90") if label in fit_data]
            labels.extend(sorted(label for label in fit_data if label.startswith("bb")))
            labels.extend(label for label in fit_data if label not in labels)
            for label in labels:
                item = fit_data[label]
                add_plot(
                    fit_names.get(label, f"时间分辨能谱拟合 {label}"),
                    item.get("plot_fits") or (item.get("plot_fit"),),
                )
        if "fluxcurve" in context:
            add_plot(
                "0.5-4 keV flux curve",
                (context["fluxcurve"].outputs.get("plot_png"), context["fluxcurve"].outputs.get("plot_svg")),
            )
        return WXTPointingResult(
            workspace=self.workspace,
            status=self.status().status.value,
            files=files,
            alpha=alpha,
            duration=duration,
            t90_source_pha=Path(ogip.outputs["source_pha"]) if ogip else None,
            t90_background_pha=Path(ogip.outputs["background_pha"]) if ogip else None,
            segments=segments,
            report=Path(context["report"].outputs["report"]) if "report" in context else None,
            summary=(
                Path(context["report"].outputs["summary_txt"])
                if "report" in context else None
            ),
            plot_groups=plot_groups,
        )
