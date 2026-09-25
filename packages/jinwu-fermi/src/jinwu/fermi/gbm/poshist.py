"""Time-indexed POSHIST access and single-time spacecraft geometry.

The download interface deliberately takes only a time and a cache directory:
attitude files are mission products, and inventing source coordinates just to
reuse an interval-based downloader would couple geometry to unrelated science
inputs.  The 30-orbit reference selection itself lives in
:func:`jinwu.fermi.gbm.pipeline.find_gbm_poshist` (RapidGBM, Wang et al. 2025);
this module exposes the shared time-keyed primitives and the one-instant
geometry snapshot used by both the legacy GBM entry points and ``jinwu.gw``.

Earth-blocking geometry mirrors GDT's ``SpacecraftFrame.location_visible``
(``gdt/core/coords/spacecraft/frame.py``): a direction is visible when its
separation from the geocenter exceeds the Earth angular radius
``arcsin(R_earth / (R_earth + altitude))``.  Keeping the same formula here
guarantees that MOC-based region integrals and pixel-wise ``location_visible``
queries agree at the same instant.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import warnings

import astropy.units as u
import numpy as np
from astropy.coordinates import CartesianRepresentation, SkyCoord

from jinwu.core.time import Time

from .pipeline import (
    _as_scalar_time,
    _download_poshist_for_day,
    _poshist_paths_for_day,
)

__all__ = [
    "GBMGeometryState",
    "fetch_poshist_for_time",
    "read_gbm_geometry",
]

# WGS84 semi-major axis, used only for the reported geocentric altitude.
_EARTH_RADIUS_M = 6378137.0


def fetch_poshist_for_time(
    utc: Time | str,
    cache: str | Path,
    *,
    download: bool = True,
    verbose: bool = False,
) -> Path:
    """Return a POSHIST file covering the UTC day that contains ``utc``.

    Existing valid files in the cache are reused and never overwritten; the
    cache layout mirrors HEASARC (``<cache>/daily/YYYY/MM/DD/current``) so the
    result is interchangeable with :func:`fetch_gbm_continuous_products`.
    """
    when = _as_scalar_time(utc)
    cache_path = Path(cache).expanduser().resolve()
    if not cache_path.is_dir():
        if not download:
            raise FileNotFoundError(
                f"no POSHIST cached for {when.utc.isot} under {cache_path}; "
                "pass download=True to fetch it"
            )
        cache_path.mkdir(parents=True, exist_ok=True)
    else:
        cache_path.mkdir(parents=True, exist_ok=True)
    paths = _poshist_paths_for_day(cache_path, when)
    if not paths:
        if not download:
            raise FileNotFoundError(
                f"no POSHIST cached for {when.utc.isot} under {cache_path}; "
                "pass download=True to fetch it"
            )
        paths = _download_poshist_for_day(cache_path, when, verbose=verbose)
    if not paths:
        raise FileNotFoundError(f"no POSHIST product available for {when.utc.isot}")
    # Keep the newest reprocessed product when several versions are cached for
    # one UTC day; the finder returns names in ascending vNN order.
    return paths[-1]


@dataclass(frozen=True, slots=True)
class GBMGeometryState:
    """One-instant Fermi geometry snapshot from POSHIST.

    Positions are geocentric inertial (POSHIST ``POS_*``, metres); the
    Earth angular radius follows GDT's ``SpacecraftFrame.earth_angular_radius``
    so region integrals built from this state reproduce
    ``SpacecraftFrame.location_visible`` exactly at this instant.
    """

    time: Time
    position_m: tuple[float, float, float]
    radius_m: float
    altitude_m: float
    subpoint_lon_deg: float
    subpoint_lat_deg: float
    earth_angular_radius_deg: float
    nadir: SkyCoord
    zenith: SkyCoord
    detector_pointings: tuple[tuple[str, float, float], ...]
    saa: bool
    good: bool
    source: str = "real"
    poshist_path: Path | None = None
    reference_time: Time | None = None

    def to_dict(self) -> dict[str, Any]:
        def scalar_degrees(value: Any) -> float:
            return float(np.asarray(value).reshape(-1)[0])

        return {
            "time_utc": self.time.utc.isot,
            "position_m": list(self.position_m),
            "radius_m": self.radius_m,
            "altitude_m": self.altitude_m,
            "subpoint_lon_deg": self.subpoint_lon_deg,
            "subpoint_lat_deg": self.subpoint_lat_deg,
            "earth_angular_radius_deg": self.earth_angular_radius_deg,
            "nadir_ra_deg": scalar_degrees(self.nadir.icrs.ra.deg),
            "nadir_dec_deg": scalar_degrees(self.nadir.icrs.dec.deg),
            "zenith_ra_deg": scalar_degrees(self.zenith.icrs.ra.deg),
            "zenith_dec_deg": scalar_degrees(self.zenith.icrs.dec.deg),
            "detector_pointings": {
                name: [ra_deg, dec_deg] for name, ra_deg, dec_deg in self.detector_pointings
            },
            "saa": self.saa,
            "good": self.good,
            "source": self.source,
            "poshist_path": str(self.poshist_path) if self.poshist_path else None,
            "reference_time_utc": self.reference_time.utc.isot if self.reference_time else None,
        }


def _detector_pointings(state_frame: Any) -> tuple[tuple[str, float, float], ...]:
    try:
        from gdt.missions.fermi.gbm.detectors import GbmDetectors
    except ImportError:
        return ()
    pointings: list[tuple[str, float, float]] = []
    with warnings.catch_warnings():
        # GDT mixes coordinate frames (spacecraft -> ICRS) and astropy warns on
        # each implicit transform; the transform itself is the intended path.
        warnings.filterwarnings("ignore", message="transforming other coordinates")
        for detector in GbmDetectors:
            direction = detector.skycoord(state_frame)
            if not direction.isscalar:
                direction = direction[0]
            icrs = direction.transform_to("icrs")
            # ``SkyCoord.transform_to`` returns a length-one coordinate for a
            # scalar ``SpacecraftFrame`` on some GDT/Astropy combinations.
            # Flattening here keeps the public snapshot scalar and avoids
            # leaking an implementation-dependent array shape.
            ra_deg = float(np.asarray(icrs.ra.deg).reshape(-1)[0])
            dec_deg = float(np.asarray(icrs.dec.deg).reshape(-1)[0])
            pointings.append((detector.name, ra_deg, dec_deg))
    return tuple(pointings)


def read_gbm_geometry(
    poshist: Any,
    t: Time | str,
    *,
    source: str = "real",
    reference_time: Time | str | None = None,
    max_interpolation_gap: u.Quantity = 5.0 * u.s,
) -> GBMGeometryState:
    """Read one interpolated geometry state at time ``t`` from POSHIST.

    ``poshist`` may be a file path or any object exposing GDT's
    ``get_spacecraft_frame``/``get_spacecraft_states`` surface.  Geometry uses
    GDT's Slerp/linear interpolation (``SpacecraftFrame.at``); the SAA/good
    flags come from the nearest state sample.  ``t`` must lie inside the
    file's validity range: attitude extrapolation is forbidden.
    """
    when = _as_scalar_time(t)
    path: Path | None = None
    history = poshist
    if not hasattr(poshist, "get_spacecraft_frame"):
        from gdt.missions.fermi.gbm.poshist import GbmPosHist

        path = Path(poshist).expanduser().resolve()
        history = GbmPosHist.open(path)
    frame = history.get_spacecraft_frame()
    samples = frame.obstime
    met = float(when.to_value("fermi"))
    sample_met = np.asarray(samples.to_value("fermi"), dtype=float)
    if sample_met.size == 0:
        raise ValueError("POSHIST contains no time samples")
    sample_min = float(sample_met.min())
    sample_max = float(sample_met.max())
    if met < sample_min or met > sample_max:
        raise ValueError(
            f"requested time {when.utc.isot} lies outside POSHIST validity "
            f"({samples[np.argmin(sample_met)].utc.isot} .. "
            f"{samples[np.argmax(sample_met)].utc.isot}); extrapolation is not allowed"
        )
    gap = u.Quantity(max_interpolation_gap, u.s).to(u.s)
    if not np.isfinite(gap.value) or gap <= 0 * u.s:
        raise ValueError("max_interpolation_gap must be a finite positive duration")
    nearest_gap = float(np.min(np.abs(sample_met - met))) * u.s
    if nearest_gap > gap:
        raise ValueError(
            f"requested time {when.utc.isot} is {nearest_gap.to_value(u.s):.3f} s from the "
            f"nearest POSHIST sample, exceeding max_interpolation_gap={gap.to_value(u.s):.3f} s"
        )
    # ``SpacecraftFrame.at`` interpolates inside the recorded interval.  Do
    # not clamp an out-of-range request to an endpoint: that would turn a
    # missing attitude sample into an unannounced extrapolation.
    attitude_time = Time(met, format="fermi")
    state_frame = frame.at(attitude_time) if sample_met.size > 1 else frame

    position = state_frame.obsgeoloc
    if not isinstance(position, CartesianRepresentation):
        position = CartesianRepresentation(position)
    radius = float(position.norm().to_value(u.m))
    xyz = tuple(float(value) for value in position.xyz.to_value(u.m))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="transforming other coordinates")
        # Earth occultation is centred on the spacecraft-to-geocentre vector,
        # not on a body-fixed spacecraft axis.  Attitude quaternions still
        # determine detector pointings through ``_detector_pointings``.
        geocenter = SkyCoord(state_frame.geocenter).transform_to("icrs")
        if not geocenter.isscalar:
            geocenter = geocenter.reshape(-1)[0]
        nadir = SkyCoord(geocenter.ra, geocenter.dec, frame="icrs")
        zenith = SkyCoord((nadir.ra + 180 * u.deg).wrap_at(360 * u.deg), -nadir.dec, frame="icrs")
    if not nadir.isscalar:
        nadir = nadir.reshape(-1)[0]
    if not zenith.isscalar:
        zenith = zenith.reshape(-1)[0]
    earth_radius_deg = float(state_frame.earth_angular_radius.to_value(u.deg))
    try:
        geodetic = state_frame.earth_location.to_geodetic()
        subpoint_lon = float(geodetic.lon.to_value(u.deg))
        subpoint_lat = float(geodetic.lat.to_value(u.deg))
    except (AttributeError, TypeError, ValueError):
        subpoint_lon = float("nan")
        subpoint_lat = float("nan")

    saa = False
    good = True
    if hasattr(history, "get_spacecraft_states"):
        states = history.get_spacecraft_states()
        state_met = np.asarray(states.time.to_value("fermi"), dtype=float)
        index = int(np.argmin(np.abs(state_met - met)))
        saa = bool(np.asarray(states["saa"])[index])
        good = bool(np.asarray(states["good"])[index])

    reference = _as_scalar_time(reference_time) if reference_time is not None else None
    return GBMGeometryState(
        time=when,
        position_m=xyz,
        radius_m=radius,
        altitude_m=radius - _EARTH_RADIUS_M,
        subpoint_lon_deg=subpoint_lon,
        subpoint_lat_deg=subpoint_lat,
        earth_angular_radius_deg=earth_radius_deg,
        nadir=nadir,
        zenith=zenith,
        detector_pointings=_detector_pointings(state_frame),
        saa=saa,
        good=good,
        source=source,
        poshist_path=path,
        reference_time=reference,
    )
