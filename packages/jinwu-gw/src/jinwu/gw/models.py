"""Data contracts used by :mod:`jinwu.gw`.

The objects in this module intentionally keep provenance alongside numerical
values.  A coverage probability without its sky-map and attitude source is
not a reproducible science product.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import astropy.units as u
from astropy.coordinates import SkyCoord

from jinwu.core.time import Time


def scalar_time(value: Time | str) -> Time:
    """Parse one scalar UTC time without silently accepting an array.

    GWpy's optional ``LIGOTimeGPS`` scalar exposes ``gpsSeconds`` and a
    numeric conversion; accepting that protocol keeps public GW times in the
    jinwu core time type without making GWpy a runtime dependency.
    """
    if isinstance(value, Time):
        result = value.utc
    elif isinstance(value, (int, float)) or hasattr(value, "gpsSeconds"):
        # GraceDB's compact superevent representation uses GPS seconds in
        # ``t_0``; JSON notices normally carry an ISO-8601 string.
        result = Time(float(value), format="ligo", scale="utc")
    else:
        text = str(value).strip().replace("Z", "+00:00")
        try:
            result = Time(text, format="isot", scale="utc")
        except ValueError:
            parsed = datetime.fromisoformat(text)
            if parsed.tzinfo is not None:
                parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
            result = Time(parsed, scale="utc")
    if not result.isscalar:
        raise ValueError("GW times must be scalar")
    return result


@dataclass(frozen=True, slots=True)
class GWEvent:
    """One LVK alert identity and its event time."""

    superevent_id: str
    event_time: Time | None
    alert_type: str = "UNKNOWN"
    status: str = "active"
    notice_version: str | None = None
    skymap_source: str | None = None
    notice_source: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.superevent_id.strip():
            raise ValueError("superevent_id must not be empty")
        object.__setattr__(self, "alert_type", self.alert_type.upper())
        status = self.status.lower()
        if status in {"withdrawn", "withdrawal", "retraction"}:
            status = "retracted"
        object.__setattr__(self, "status", status)
        if self.event_time is None:
            if status != "retracted":
                raise ValueError("only a retracted GW event may omit event_time")
        else:
            object.__setattr__(self, "event_time", scalar_time(self.event_time))

    def to_dict(self) -> dict[str, Any]:
        return {
            "superevent_id": self.superevent_id,
            "event_time_utc": self.event_time.utc.isot if self.event_time is not None else None,
            "alert_type": self.alert_type,
            "status": self.status,
            "notice_version": self.notice_version,
            "skymap_source": self.skymap_source,
            "notice_source": self.notice_source,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class SkyMap:
    """Normalized HEALPix localization table.

    ``uniq`` is always nested UNIQ indexing, ``prob_density_sr`` is in sr^-1,
    and ``pixel_probability`` is density times pixel area.  Mixed-resolution
    LVK maps and flat HEALPix maps are represented by the same contract.
    ``raw_table``/``raw_pixel_probability`` retain the values before optional
    unit normalization for provenance and later re-analysis.
    """

    uniq: Any
    levels: Any
    ipix: Any
    prob_density_sr: Any
    pixel_area_sr: Any
    pixel_probability: Any
    table: Any
    source: str | None = None
    ordering: str = "NUNIQ"
    total_probability: float = 0.0
    raw_total_probability: float | None = None
    raw_table: Any | None = None
    raw_pixel_probability: Any | None = None

    def __post_init__(self) -> None:
        import numpy as np

        arrays = [
            np.asarray(self.uniq),
            np.asarray(self.levels),
            np.asarray(self.ipix),
            np.asarray(self.prob_density_sr),
            np.asarray(self.pixel_area_sr),
            np.asarray(self.pixel_probability),
        ]
        if len({array.shape for array in arrays}) != 1:
            raise ValueError("sky-map columns must have identical shapes")
        if arrays[0].ndim != 1 or arrays[0].size == 0:
            raise ValueError("sky map must contain at least one pixel")
        if np.any(~np.isfinite(arrays[3])) or np.any(arrays[3] < 0):
            raise ValueError("sky-map probability density must be finite and non-negative")
        if np.any(~np.isfinite(arrays[4])) or np.any(arrays[4] <= 0):
            raise ValueError("sky-map pixel areas must be finite and positive")
        if np.any(~np.isfinite(arrays[5])) or np.any(arrays[5] < 0):
            raise ValueError("sky-map pixel probabilities must be finite and non-negative")
        total = float(np.sum(arrays[5]))
        if not np.isfinite(total) or total <= 0:
            raise ValueError("sky map has no positive probability")
        object.__setattr__(self, "total_probability", total)
        if self.raw_total_probability is None:
            object.__setattr__(self, "raw_total_probability", total)
        if self.raw_table is None:
            object.__setattr__(self, "raw_table", self.table)
        if self.raw_pixel_probability is None:
            object.__setattr__(self, "raw_pixel_probability", arrays[5].copy())

    @property
    def max_level(self) -> int:
        import numpy as np

        return int(np.max(np.asarray(self.levels)))

    @property
    def skycoord(self) -> SkyCoord:
        """Return the center of every pixel as an ICRS coordinate."""
        import astropy_healpix as ah
        import numpy as np

        levels = np.asarray(self.levels, dtype=int)
        ipix = np.asarray(self.ipix, dtype=np.int64)
        ra = np.empty(levels.size, dtype=float)
        dec = np.empty(levels.size, dtype=float)
        for level in np.unique(levels):
            indices = np.flatnonzero(levels == level)
            hp = ah.HEALPix(nside=ah.level_to_nside(int(level)), order="nested", frame="icrs")
            coordinates = hp.healpix_to_skycoord(ipix[indices])
            ra[indices] = np.asarray(coordinates.ra.deg, dtype=float)
            dec[indices] = np.asarray(coordinates.dec.deg, dtype=float)
        return SkyCoord(ra, dec, unit="deg", frame="icrs")

    def normalized(self) -> "SkyMap":
        """Return a copy normalized to unit total probability."""
        import numpy as np

        total = float(np.sum(self.pixel_probability))
        density = np.asarray(self.prob_density_sr) / total
        probability = np.asarray(self.pixel_probability) / total
        table = self.table.copy(copy_data=True)
        table["PROBDENSITY"] = density
        return SkyMap(
            self.uniq, self.levels, self.ipix, density, self.pixel_area_sr,
            probability, table, self.source, self.ordering, 1.0,
            self.raw_total_probability, self.raw_table, self.raw_pixel_probability,
        )


@dataclass(frozen=True, slots=True)
class SkyFootprint:
    """A time-tagged MOC footprint from EP, BAT, GBM, or another instrument."""

    name: str
    moc: Any
    kind: str = "footprint"
    source: str | None = None
    valid_at: Time | None = None
    center: SkyCoord | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    valid_from: Time | None = None
    valid_to: Time | None = None

    def __post_init__(self) -> None:
        for field_name in ("valid_at", "valid_from", "valid_to"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, scalar_time(value))
        if self.valid_from is not None and self.valid_to is not None and self.valid_from > self.valid_to:
            raise ValueError("footprint valid_from must not be later than valid_to")

    def applies_at(self, when: Time | str | None) -> bool:
        if when is None:
            return True
        target = scalar_time(when)
        if self.valid_from is not None and target < scalar_time(self.valid_from):
            return False
        if self.valid_to is not None and target > scalar_time(self.valid_to):
            return False
        if self.valid_from is not None or self.valid_to is not None:
            return True
        if self.valid_at is None:
            return True
        return abs((target - scalar_time(self.valid_at)).to_value("s")) < 1.0

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "name": self.name,
            "kind": self.kind,
            "source": self.source,
            "valid_at_utc": self.valid_at.utc.isot if self.valid_at is not None else None,
            "valid_from_utc": self.valid_from.utc.isot if self.valid_from is not None else None,
            "valid_to_utc": self.valid_to.utc.isot if self.valid_to is not None else None,
            "metadata": dict(self.metadata),
        }
        if self.center is not None:
            result.update({"ra_deg": float(self.center.ra.deg), "dec_deg": float(self.center.dec.deg)})
        return result


@dataclass(frozen=True, slots=True)
class Localization:
    """A candidate position or error region displayed on a sky map."""

    name: str
    center: SkyCoord
    radius: u.Quantity | None = None
    source: str | None = None
    confidence: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.center, SkyCoord) or not self.center.isscalar:
            raise ValueError("localization center must be a scalar SkyCoord")
        if self.radius is not None:
            radius = u.Quantity(self.radius).to(u.deg)
            if radius.ndim or radius.value <= 0:
                raise ValueError("localization radius must be positive")
            object.__setattr__(self, "radius", radius)
        if self.confidence is not None:
            confidence = float(self.confidence)
            if not 0.0 < confidence <= 1.0:
                raise ValueError("localization confidence must lie in (0, 1]")
            object.__setattr__(self, "confidence", confidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ra_deg": float(self.center.ra.deg),
            "dec_deg": float(self.center.dec.deg),
            "radius_deg": float(self.radius.to_value(u.deg)) if self.radius is not None else None,
            "source": self.source,
            "confidence": self.confidence,
        }


@dataclass(frozen=True, slots=True)
class CoverageResult:
    """Coverage probabilities and attitude provenance for one instrument."""

    instrument: str
    status: str
    probability: float | None
    footprint: SkyFootprint | None = None
    geometry_footprint: SkyFootprint | None = None
    state_footprint: SkyFootprint | None = None
    geometry_probability: float | None = None
    geometric_probability: float | None = None
    state_probability: float | None = None
    source: str | None = None
    reference_time: Time | None = None
    target_time: Time | None = None
    reasons: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        import math

        for field_name in ("probability", "geometry_probability", "geometric_probability", "state_probability"):
            value = getattr(self, field_name)
            if value is None:
                continue
            numeric = float(value)
            if not math.isfinite(numeric) or numeric < -1e-12 or numeric > 1.0 + 1e-12:
                raise ValueError(f"{field_name} must lie in [0, 1] or be None")
            object.__setattr__(self, field_name, min(1.0, max(0.0, numeric)))
        if self.geometry_probability is None and self.geometric_probability is not None:
            object.__setattr__(self, "geometry_probability", self.geometric_probability)
        elif self.geometric_probability is None and self.geometry_probability is not None:
            object.__setattr__(self, "geometric_probability", self.geometry_probability)

    def to_dict(self) -> dict[str, Any]:
        return {
            "instrument": self.instrument,
            "status": self.status,
            "probability": self.probability,
            "geometry_probability": self.geometry_probability,
            "geometric_probability": self.geometric_probability,
            "state_probability": self.state_probability,
            "footprint": self.footprint.to_dict() if self.footprint else None,
            "geometry_footprint": self.geometry_footprint.to_dict() if self.geometry_footprint else None,
            "state_footprint": self.state_footprint.to_dict() if self.state_footprint else None,
            "source": self.source,
            "reference_time_utc": self.reference_time.utc.isot if self.reference_time else None,
            "target_time_utc": self.target_time.utc.isot if self.target_time else None,
            "reasons": list(self.reasons),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class GWRunResult:
    """Serializable result returned by :class:`GWPipeline`."""

    event: GWEvent
    skymap_source: str | None
    output_paths: Mapping[str, str]
    coverages: tuple[CoverageResult, ...]
    probabilities: Mapping[str, float | None]
    spectral_analysis: str = "not_run"
    warnings: tuple[str, ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)
    credible_regions: Mapping[str, Any] = field(default_factory=dict)
    moc_refinement: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event": self.event.to_dict(),
            "skymap_source": self.skymap_source,
            "output_paths": dict(self.output_paths),
            "coverages": [item.to_dict() for item in self.coverages],
            "probabilities": dict(self.probabilities),
            "spectral_analysis": self.spectral_analysis,
            "warnings": list(self.warnings),
            "provenance": dict(self.provenance),
            "credible_regions": dict(self.credible_regions),
            "moc_refinement": dict(self.moc_refinement),
        }
