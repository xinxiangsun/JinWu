"""EP/BAT and other time-tagged sky footprint readers."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import astropy.units as u
from astropy.coordinates import SkyCoord
import numpy as np

from .models import SkyFootprint, scalar_time


def _adaptive_cap_moc(center: SkyCoord, radius: u.Quantity, max_depth: int):
    """Rasterize a spherical cap with a conservative nested-HEALPix boundary.

    This is used only when MOCPy's cone wrapper returns an area incompatible
    with its own pixel-resolution bound (observed for symmetric coordinates
    with some mocpy/Astropy combinations).  Interior cells are retained at
    their first wholly-contained level and boundary cells are retained at the
    requested maximum level.
    """
    import healpy as hp
    from mocpy import MOC

    depth = int(max_depth)
    cap_radius = float(radius.to_value(u.rad))
    vector = np.asarray(center.icrs.cartesian.xyz.value, dtype=float).reshape(3)
    vector /= np.linalg.norm(vector)
    candidates = np.arange(12, dtype=np.int64)
    selected_ipix: list[np.ndarray] = []
    selected_depth: list[np.ndarray] = []
    for order in range(depth + 1):
        nside = 2**order
        x, y, z = hp.pix2vec(nside, candidates, nest=True)
        dot = np.clip(vector[0] * x + vector[1] * y + vector[2] * z, -1.0, 1.0)
        distance = np.arccos(dot)
        pixel_radius = float(hp.max_pixrad(nside))
        inside = distance + pixel_radius <= cap_radius
        outside = distance - pixel_radius > cap_radius
        take = inside if order < depth else ~outside
        if np.any(take):
            selected_ipix.append(candidates[take])
            selected_depth.append(np.full(np.count_nonzero(take), order, dtype=np.uint8))
        boundary = ~(inside | outside)
        if order == depth or not np.any(boundary):
            break
        candidates = (candidates[boundary, None] * 4 + np.arange(4, dtype=np.int64)).reshape(-1)
    if not selected_ipix:
        return MOC.new_empty(max_depth=depth)
    return MOC.from_healpix_cells(
        ipix=np.concatenate(selected_ipix),
        depth=np.concatenate(selected_depth),
        max_depth=depth,
    )


def spherical_cap_moc(
    center: SkyCoord,
    radius: u.Quantity | float,
    *,
    max_depth: int,
) -> tuple[Any, str]:
    """Return a cap MOC and the construction method used.

    ``MOC.from_cone`` is preferred.  Its sky fraction is checked against the
    analytic cap area; an adaptive HEALPix construction replaces only a
    demonstrably bad cone result, preserving a finite-resolution MOC rather
    than approximating the cap with a fixed polygon.
    """
    from mocpy import MOC

    radius_q = u.Quantity(radius, u.deg).to(u.deg)
    expected_fraction = (1.0 - math.cos(float(radius_q.to_value(u.rad)))) / 2.0
    # At order 10, a 1e-4 all-sky fraction is already looser than the cap
    # boundary discretization in normal operation yet rejects the known
    # symmetric-coordinate cone failure by orders of magnitude.
    tolerance = max(1e-6, 1e-4)
    try:
        moc = MOC.from_cone(
            lon=center.icrs.ra,
            lat=center.icrs.dec,
            radius=radius_q,
            max_depth=int(max_depth),
        )
        if abs(float(moc.sky_fraction) - expected_fraction) <= tolerance:
            return moc, "mocpy_from_cone"
    except (TypeError, ValueError, RuntimeError):
        pass
    return _adaptive_cap_moc(center.icrs, radius_q, int(max_depth)), "adaptive_healpix"


def footprint_from_circle(
    name: str,
    *,
    ra_deg: float,
    dec_deg: float,
    radius: u.Quantity | float,
    source: str | None = None,
    valid_at: str | None = None,
    valid_from: str | None = None,
    valid_to: str | None = None,
    kind: str = "localization",
    max_depth: int = 10,
    metadata: dict[str, Any] | None = None,
) -> SkyFootprint:
    """Create an ICRS circular footprint or localization."""
    center = SkyCoord(float(ra_deg) * u.deg, float(dec_deg) * u.deg, frame="icrs")
    radius_q = u.Quantity(radius, u.deg).to(u.deg)
    if radius_q.ndim or not math.isfinite(float(radius_q.value)) or float(radius_q.value) <= 0:
        raise ValueError("footprint radius must be a finite positive angle")
    moc, method = spherical_cap_moc(center, radius_q, max_depth=int(max_depth))
    return SkyFootprint(
        name=name,
        moc=moc,
        kind=kind,
        source=source,
        valid_at=scalar_time(valid_at) if valid_at else None,
        center=center,
        metadata={**(metadata or {}), "geometry_method": method, "native_moc_order": int(max_depth)},
        valid_from=scalar_time(valid_from) if valid_from else None,
        valid_to=scalar_time(valid_to) if valid_to else None,
    )


def footprint_from_polygon(
    name: str,
    *,
    ra_deg: Iterable[float],
    dec_deg: Iterable[float],
    source: str | None = None,
    valid_at: str | None = None,
    valid_from: str | None = None,
    valid_to: str | None = None,
    kind: str = "footprint",
    max_depth: int = 10,
    metadata: dict[str, Any] | None = None,
) -> SkyFootprint:
    """Create a footprint from an ICRS polygon."""
    from mocpy import MOC

    center = SkyCoord(list(ra_deg), list(dec_deg), unit="deg", frame="icrs")
    if center.size < 3:
        raise ValueError("a polygon needs at least three vertices")
    moc = MOC.from_polygon_skycoord(center, max_depth=int(max_depth))
    # Average unit vectors rather than RA/DEC scalars so a polygon crossing
    # 0/360 degrees receives a sensible representative center for plots.
    vector = center.cartesian.xyz.mean(axis=1)
    representative_cartesian = SkyCoord(
        x=vector[0], y=vector[1], z=vector[2], unit="", representation_type="cartesian", frame="icrs"
    ).icrs
    representative = SkyCoord(
        representative_cartesian.spherical.lon,
        representative_cartesian.spherical.lat,
        frame="icrs",
    )
    return SkyFootprint(
        name=name,
        moc=moc,
        kind=kind,
        source=source,
        valid_at=scalar_time(valid_at) if valid_at else None,
        center=representative,
        metadata=metadata or {},
        valid_from=scalar_time(valid_from) if valid_from else None,
        valid_to=scalar_time(valid_to) if valid_to else None,
    )


def _footprint_from_record(record: dict[str, Any], *, base: Path, max_depth: int) -> SkyFootprint:
    name = str(record.get("name") or record.get("instrument") or "layer")
    source = record.get("source")
    valid_at = record.get("time") or record.get("valid_at") or record.get("observation_time")
    valid_from = record.get("valid_from") or record.get("start_time") or record.get("valid_start")
    valid_to = record.get("valid_to") or record.get("stop_time") or record.get("valid_end")
    ra_value = record.get("ra_deg", record.get("ra"))
    dec_value = record.get("dec_deg", record.get("dec"))
    radius_value = record.get(
        "radius_deg",
        record.get("error_radius_deg", record.get("error_radius" , record.get("radius"))),
    )
    default_kind = "localization" if any(
        key in record for key in ("error_radius", "error_radius_deg", "confidence", "confidence_level")
    ) else "footprint"
    kind = str(record.get("kind", default_kind))
    if "skymap" in record or "probability_map" in record:
        map_path = Path(record.get("skymap") or record.get("probability_map"))
        if not map_path.is_absolute():
            map_path = base / map_path
        from .skymap import credible_region, load_skymap, sky_map_footprint

        localization_map = load_skymap(map_path)
        confidence = float(record.get("credible_level", record.get("confidence", 0.9)))
        moc = credible_region(localization_map, confidence, max_depth=max_depth) if confidence < 1.0 else sky_map_footprint(localization_map)
        return SkyFootprint(
            name=name,
            moc=moc,
            kind=kind if kind != "footprint" else "localization",
            source=source or str(map_path),
            valid_at=scalar_time(valid_at) if valid_at else None,
            valid_from=scalar_time(valid_from) if valid_from else None,
            valid_to=scalar_time(valid_to) if valid_to else None,
            metadata={**record, "skymap_source": str(map_path), "credible_level": confidence},
        )
    if "moc" in record or "path" in record or "moc_path" in record:
        path = Path(record.get("moc") or record.get("path") or record.get("moc_path"))
        if not path.is_absolute():
            path = base / path
        from mocpy import MOC

        moc = MOC.from_fits(path)
        return SkyFootprint(
            name=name, moc=moc, kind=kind, source=source or str(path),
            valid_at=scalar_time(valid_at) if valid_at else None,
            valid_from=scalar_time(valid_from) if valid_from else None,
            valid_to=scalar_time(valid_to) if valid_to else None,
            metadata={key: value for key, value in record.items() if key not in {"name", "kind", "source", "time", "valid_at", "valid_from", "valid_to", "start_time", "stop_time", "moc", "path"}},
        )
    if (
        not (record.get("vertices") or record.get("polygon"))
        and (
            any(key in record for key in ("radius_deg", "error_radius_deg", "error_radius", "radius"))
            or record.get("shape") == "circle"
        )
    ):
        return footprint_from_circle(
            name,
            ra_deg=ra_value,
            dec_deg=dec_value,
            radius=float(radius_value or 0.0),
            source=source,
            valid_at=valid_at,
            valid_from=valid_from,
            valid_to=valid_to,
            kind=kind,
            max_depth=max_depth,
            metadata=record,
        )
    vertices = record.get("vertices") or record.get("polygon")
    if vertices and isinstance(vertices, list) and isinstance(vertices[0], (list, tuple)):
        ra_values = [pair[0] for pair in vertices]
        dec_values = [pair[1] for pair in vertices]
        return footprint_from_polygon(
            name,
            ra_deg=ra_values,
            dec_deg=dec_values,
            source=source,
            valid_at=valid_at,
            valid_from=valid_from,
            valid_to=valid_to,
            kind=kind,
            max_depth=max_depth,
            metadata=record,
        )
    if (
        ra_value is not None
        and dec_value is not None
        and isinstance(ra_value, list)
        and isinstance(dec_value, list)
    ):
        return footprint_from_polygon(
            name,
            ra_deg=ra_value,
            dec_deg=dec_value,
            source=source,
            valid_at=valid_at,
            valid_from=valid_from,
            valid_to=valid_to,
            kind=kind,
            max_depth=max_depth,
            metadata=record,
        )
    raise ValueError(f"unsupported footprint record for {name!r}")


def load_layers(path: str | Path, *, max_depth: int = 10) -> list[SkyFootprint]:
    """Load a JSON layer list.

    Records may point to a MOC FITS file, describe a circle using
    ``ra_deg/dec_deg/radius_deg``, or describe a polygon using coordinate
    lists.  A top-level ``{"layers": [...]}`` wrapper is also accepted.
    """
    path = Path(path).expanduser().resolve()
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    records = payload.get("layers", payload) if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError("layers JSON must be a list or contain a 'layers' list")
    return [_footprint_from_record(dict(record), base=path.parent, max_depth=max_depth) for record in records]
