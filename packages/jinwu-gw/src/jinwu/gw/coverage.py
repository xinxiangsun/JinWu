"""Coverage calculations that bridge a GW sky map and existing GBM geometry."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal
import hashlib
import warnings

import astropy.units as u
import numpy as np

from .models import CoverageResult, SkyFootprint, SkyMap, scalar_time
from .skymap import ProbabilityIntegral, probability_in_footprint, refined_probability
from .layers import spherical_cap_moc


def _cone_builder(lon_deg: float, lat_deg: float, radius_deg: float):
    """Return a callable rebuilding one spherical cap at a given MOC order."""
    from astropy.coordinates import SkyCoord

    center = SkyCoord(float(lon_deg) * u.deg, float(lat_deg) * u.deg, frame="icrs")
    radius = u.Quantity(float(radius_deg), u.deg)

    def build(order: int):
        return spherical_cap_moc(center, radius, max_depth=int(order))[0]

    return build


def _union_of(builders, order: int):
    union = builders[0](order)
    for builder in builders[1:]:
        union = union.union(builder(order))
    return union


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _moc_from_mask(skymap: SkyMap, mask: np.ndarray):
    from mocpy import MOC

    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return MOC.new_empty(max_depth=skymap.max_level)
    return MOC.from_healpix_cells(
        ipix=np.asarray(skymap.ipix)[mask],
        depth=np.asarray(skymap.levels, dtype=np.uint8)[mask],
        max_depth=skymap.max_level,
    )


def _nearest_state(history: Any, when: Any) -> tuple[Any, Any, bool, bool, int]:
    states = history.get_spacecraft_states()
    met = np.asarray(states.time.fermi, dtype=float)
    target_met = float(when.to_value("fermi"))
    index = int(np.argmin(np.abs(met - target_met)))
    frames = history.get_spacecraft_frame()
    # GDT exposes ``SpacecraftFrame.at`` for interpolation.  Use it for the
    # sky-mask calculation so Earth blocking and detector angles correspond to
    # the same instant as ``read_gbm_geometry``; retain the nearest state only
    # for the discrete SAA/good flags.  Lightweight test doubles may expose
    # only indexed frames, in which case the nearest sample is the documented
    # fallback.
    try:
        frame = frames.at(when)
    except (AttributeError, TypeError, ValueError):
        frame = frames[index]
    return frame, states, bool(np.asarray(states["good"])[index]), bool(np.asarray(states["saa"])[index]), index


def gbm_coverage_for_skymap(
    skymap: SkyMap,
    target_time: Any,
    cache: str | Path,
    *,
    mode: Literal["auto", "observed", "predicted", "none"] = "auto",
    download: bool = True,
    max_nai_angle: float = 60.0,
    max_bgo_angle: float = 90.0,
    order_start: int = 10,
    order_max: int = 13,
    tolerance: float = 1e-3,
) -> CoverageResult:
    """Compute GBM sky probability visible at one target time.

    The reported probabilities integrate the sky map over analytically
    rebuilt regions -- the Earth-blocking cap complement and the detector
    response cones -- rasterized at increasing MOC orders until adjacent
    estimates agree within ``tolerance`` (spec: order 10 -> 13, |dP| < 1e-3;
    an unconverged boundary is recorded, never silently trusted).  The
    pixel-center masks based on GDT's ``location_visible``/``detector_angle``
    are retained as footprints for plotting and cross-validation, and every
    result stays a geometry statement: a POSHIST-state pass is not a TTE
    exposure.
    """
    target = scalar_time(target_time)
    try:
        from jinwu.fermi.gbm import find_gbm_poshist
        from jinwu.fermi.gbm.poshist import read_gbm_geometry
        from gdt.missions.fermi.gbm.poshist import GbmPosHist
    except ImportError as exc:
        return CoverageResult("GBM", "unknown", None, source=None, target_time=target, reasons=(f"dependency_missing:{exc}",))
    selection = find_gbm_poshist(target, cache, mode=mode, download=download)
    if selection.path is None or selection.reference_time is None:
        return CoverageResult(
            "GBM", "unknown", None, source=str(selection.path) if selection.path else None,
            reference_time=selection.reference_time, target_time=target,
            reasons=(selection.reason or "poshist_missing",),
            metadata={"poshist": selection.to_dict(), "coverage_basis": "unknown"},
        )
    try:
        poshist_provenance = selection.to_dict()
        poshist_provenance["sha256"] = _sha256(selection.path)
        history = GbmPosHist.open(selection.path)
        frame, _, _, _, index = _nearest_state(history, selection.reference_time)
        # Reuse the shared one-instant POSHIST geometry primitive.  In addition
        # to the state flags it supplies scalar ICRS detector pointings for the
        # diagnostic plot; the pixel masks below still use GDT's vectorized
        # visibility/angle methods on the exact same reference frame.
        geometry = read_gbm_geometry(
            history,
            selection.reference_time,
            source="predicted" if selection.status == "predicted_30_orbit" else "real",
            reference_time=selection.reference_time,
        )
        good, saa = geometry.good, geometry.saa
        coords = skymap.skycoord
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="transforming other coordinates")
            earth_visible = np.asarray(frame.location_visible(coords), dtype=bool)
        if not good or saa:
            state_mask = np.zeros(earth_visible.shape, dtype=bool)
        else:
            state_mask = earth_visible
        detector_mask = np.zeros(earth_visible.shape, dtype=bool)
        angles: dict[str, float] = {}
        detector_centers = {
            name: [float(ra_deg), float(dec_deg)]
            for name, ra_deg, dec_deg in geometry.detector_pointings
        }
        if getattr(frame, "detectors", None) is not None:
            for name in tuple(f"n{i}" for i in range(10)) + ("na", "nb"):
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", message="transforming other coordinates")
                        values = np.asarray(frame.detector_angle(name, coords).to_value(u.deg), dtype=float)
                except (AttributeError, KeyError, ValueError):
                    continue
                angles[name] = float(np.nanmin(values)) if values.size else float("nan")
                detector_mask |= values <= float(max_nai_angle)
            for name in ("b0", "b1"):
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", message="transforming other coordinates")
                        values = np.asarray(frame.detector_angle(name, coords).to_value(u.deg), dtype=float)
                except (AttributeError, KeyError, ValueError):
                    continue
                angles[name] = float(np.nanmin(values)) if values.size else float("nan")
                detector_mask |= values <= float(max_bgo_angle)
        if not np.any(detector_mask):
            detector_mask = state_mask.copy()
        geometric_moc = _moc_from_mask(skymap, earth_visible)
        geom_moc = _moc_from_mask(skymap, state_mask)
        detector_moc = _moc_from_mask(skymap, state_mask & detector_mask)
        geometric = SkyFootprint(
            "GBM geometric visibility",
            geometric_moc,
            kind="gbm_geometry",
            source=str(selection.path),
            valid_at=selection.reference_time,
        )
        geom = SkyFootprint(
            "GBM state-filtered visibility",
            geom_moc,
            kind="gbm_state",
            source=str(selection.path),
            valid_at=selection.reference_time,
        )
        det = SkyFootprint(
            "GBM detector visibility",
            detector_moc,
            kind="gbm_detectors",
            source=str(selection.path),
            valid_at=selection.reference_time,
            metadata={"angles_deg": angles, "detector_centers": detector_centers},
        )
        # Pixel-center integrals (GDT mask granularity = sky-map resolution),
        # retained as the cross-validation baseline for the refined boundary.
        detector_mask_probability = probability_in_footprint(skymap, det)
        geometric_mask_probability = probability_in_footprint(skymap, geometric)
        state_mask_probability = probability_in_footprint(skymap, geom)

        # Refined boundary integrals: rebuild the Earth-blocking cap and the
        # detector response cones at increasing MOC orders.  The Earth radius
        # formula matches GDT ``SpacecraftFrame.earth_angular_radius``, so both
        # approaches describe the same geometry at the same instant.
        refinement_notes: list[str] = []
        geometry_integral: ProbabilityIntegral | None = None
        detector_integral: ProbabilityIntegral | None = None
        if np.isfinite(geometry.earth_angular_radius_deg) and geometry.earth_angular_radius_deg > 0:
            try:
                occulted_builder = _cone_builder(
                    float(geometry.nadir.icrs.ra.deg),
                    float(geometry.nadir.icrs.dec.deg),
                    geometry.earth_angular_radius_deg,
                )

                def visible_builder(order: int):
                    return occulted_builder(order).complement()

                geometry_integral = refined_probability(
                    skymap, visible_builder,
                    order_start=order_start, order_max=order_max, tolerance=tolerance,
                )
                cone_builders = [
                    _cone_builder(ra_deg, dec_deg, max_nai_angle if name.startswith("n") else max_bgo_angle)
                    for name, ra_deg, dec_deg in geometry.detector_pointings
                ]
                if cone_builders:

                    def detector_builder(order: int):
                        return _union_of(cone_builders, order).intersection(
                            occulted_builder(order).complement()
                        )

                    detector_integral = refined_probability(
                        skymap, detector_builder,
                        order_start=order_start, order_max=order_max, tolerance=tolerance,
                    )
            except (ValueError, TypeError, MemoryError) as exc:
                refinement_notes.append(f"refinement_fallback:{exc}")
        else:
            refinement_notes.append("refinement_unavailable:earth_angular_radius")

        geometric_probability = (
            geometry_integral.value if geometry_integral is not None else geometric_mask_probability
        )
        detector_probability = (
            detector_integral.value if detector_integral is not None else detector_mask_probability
        )
        # POSHIST state is a gate on the coverage result.  Keep geometric
        # visibility for diagnostics, but never report a detector probability
        # while GBM is in SAA or marks the state unusable.
        if not good or saa:
            detector_probability = 0.0
        state_probability = geometric_probability if (good and not saa) else 0.0
        reasons = tuple(
            reason
            for reason in (
                "earth_occulted" if np.any(~earth_visible) else None,
                "saa" if saa else None,
                "spacecraft_not_good" if not good else None,
            )
            if reason is not None
        )
        # Promote the converged analytic MOCs into the public footprints.
        # Plotting and later GBM/EP intersections must use the same regions as
        # the primary reported integrals, rather than the coarser pixel-center
        # masks retained above only as a GDT cross-check.
        if geometry_integral is not None:
            refined_geometric_moc = visible_builder(geometry_integral.order)
            geometric = SkyFootprint(
                "GBM geometric visibility", refined_geometric_moc,
                kind="gbm_geometry", source=str(selection.path), valid_at=selection.reference_time,
                metadata={"moc_order": geometry_integral.order, "integral": geometry_integral.to_dict()},
            )
            geom = SkyFootprint(
                "GBM state-filtered visibility",
                refined_geometric_moc if good and not saa else _moc_from_mask(skymap, np.zeros_like(earth_visible)),
                kind="gbm_state", source=str(selection.path), valid_at=selection.reference_time,
                metadata={"moc_order": geometry_integral.order, "state_gate": bool(good and not saa)},
            )
        if detector_integral is not None:
            refined_detector_moc = detector_builder(detector_integral.order)
            det = SkyFootprint(
                "GBM detector visibility",
                refined_detector_moc if good and not saa else _moc_from_mask(skymap, np.zeros_like(earth_visible)),
                kind="gbm_detectors", source=str(selection.path), valid_at=selection.reference_time,
                metadata={
                    "angles_deg": angles, "detector_centers": detector_centers,
                    "moc_order": detector_integral.order,
                    "integral": detector_integral.to_dict(),
                    "state_gate": bool(good and not saa),
                },
            )
        return CoverageResult(
            "GBM", selection.status, detector_probability, det,
            geometry_footprint=geometric,
            state_footprint=geom,
            geometry_probability=geometric_probability,
            geometric_probability=geometric_probability,
            state_probability=state_probability,
            source=str(selection.path), reference_time=selection.reference_time, target_time=target,
            reasons=reasons,
            metadata={
                "poshist": poshist_provenance,
                "state_index": index,
                "coverage_basis": "poshist_geometry_only",
                "earth_visible_pixels": int(np.count_nonzero(earth_visible)),
                "detector_centers": detector_centers,
                "detector_angles_deg": angles,
                "geometry_state": geometry.to_dict(),
                "geometry_integral": geometry_integral.to_dict() if geometry_integral else None,
                "detector_integral": detector_integral.to_dict() if detector_integral else None,
                "mask_probabilities": {
                    "geometry": geometric_mask_probability,
                    "state": state_mask_probability,
                    "detector": detector_mask_probability,
                },
                "refinement_notes": tuple(refinement_notes),
            },
        )
    except (OSError, ValueError, TypeError, IndexError, AttributeError, RuntimeError) as exc:
        return CoverageResult("GBM", "unknown", None, source=str(selection.path), reference_time=selection.reference_time, target_time=target, reasons=(f"geometry_failed:{exc}",), metadata={"poshist": selection.to_dict()})
