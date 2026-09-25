"""HEALPix and MOC operations for gravitational-wave localizations.

Layering: local-file parsing of the pixel contract (nested UNIQ, density in
sr^-1) lives in :mod:`jinwu.core.skymap` and is shared with instruments; this
module adds the GW-specific capabilities on top — trusted-source URL fetching
with per-redirect validation, alert/notice provenance, credible regions and
MOC integration.  ``jinwu.gw.skymap.load_skymap`` remains the richer public
entry point for GW workflows.
"""

from __future__ import annotations

from dataclasses import dataclass
import io
from pathlib import Path
from typing import Any, Callable

import astropy.units as u
from astropy.io import fits
from astropy.table import Table
import numpy as np

from .models import SkyMap


def _unit_density(values: Any) -> np.ndarray:
    unit = getattr(values, "unit", None)
    if unit is None:
        return np.asarray(values, dtype=float)
    quantity = u.Quantity(values)
    if quantity.unit.is_equivalent(u.sr**-1):
        return quantity.to_value(u.sr**-1)
    if quantity.unit.is_equivalent(u.deg**-2):
        return quantity.to_value(u.sr**-1)
    raise ValueError(f"unsupported sky-map density unit: {quantity.unit}")


def _density_column(table_hdu: Any, name: Any) -> np.ndarray:
    """Read a FITS density column while honoring its optional TUNIT card."""
    values = np.asarray(table_hdu.data[name], dtype=float)
    unit = getattr(table_hdu.columns[name], "unit", None)
    if unit:
        values = _unit_density(values * u.Unit(str(unit)))
    return values


def _uniq_columns(uniq: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import astropy_healpix as ah

    levels, ipix = ah.uniq_to_level_ipix(np.asarray(uniq, dtype=np.uint64))
    return np.asarray(uniq, dtype=np.uint64), np.asarray(levels, dtype=np.int64), np.asarray(ipix, dtype=np.int64)


def _make_map(
    uniq: np.ndarray,
    levels: np.ndarray,
    ipix: np.ndarray,
    density: np.ndarray,
    area: np.ndarray,
    *,
    source: str | None,
    normalize: bool,
    ordering: str = "NUNIQ",
    raw_table: Table | None = None,
) -> SkyMap:
    probability = np.asarray(density, dtype=float) * np.asarray(area, dtype=float)
    table = Table({"UNIQ": np.asarray(uniq, dtype=np.uint64), "PROBDENSITY": np.asarray(density, dtype=float)})
    result = SkyMap(
        uniq, levels, ipix, density, area, probability, table,
        source=source, ordering=ordering, raw_table=raw_table,
    )
    if normalize:
        result = result.normalized()
    return result


def load_skymap(path: str | Path, *, normalize: bool = True) -> SkyMap:
    """Read an LVK multi-order or ordinary HEALPix FITS sky map.

    The returned density is always in sr^-1 and the HEALPix indices are
    nested UNIQ values.  A map with negative, non-finite, or zero probability
    is rejected before any plotting is attempted.
    """
    raw_path = str(path)
    if raw_path.lower().startswith(("http://", "https://")):
        # Sky-map URLs occur in untrusted notices, so use the alert HTTP
        # helper which validates every redirect target as well.
        from .alert import fetch_http_bytes

        content, final_url = fetch_http_bytes(raw_path)
        fileobj: Any = io.BytesIO(content)
        source = final_url
    else:
        local_path = Path(path).expanduser().resolve()
        if not local_path.is_file():
            raise FileNotFoundError(local_path)
        fileobj = local_path
        source = str(local_path)
    with fits.open(fileobj, memmap=False) as hdul:
        table_hdu = next((hdu for hdu in hdul[1:] if getattr(hdu, "data", None) is not None), None)
        if table_hdu is None:
            raise ValueError(f"no HEALPix table found in {path}")
        names = {str(name).upper(): name for name in table_hdu.columns.names}
        header = table_hdu.header
        primary_header = hdul[0].header
        data = table_hdu.data
        raw_table = Table(data, copy=True)
        if "UNIQ" in names and "PROBDENSITY" in names:
            uniq = np.asarray(data[names["UNIQ"]], dtype=np.uint64)
            uniq, levels, ipix = _uniq_columns(uniq)
            density = _density_column(table_hdu, names["PROBDENSITY"])
            import astropy_healpix as ah

            area = ah.nside_to_pixel_area(ah.level_to_nside(levels)).to_value(u.sr)
            return _make_map(
                uniq, levels, ipix, density, area,
                source=source, normalize=normalize, ordering="NUNIQ",
                raw_table=raw_table,
            )

        prob_name = names.get("PROB") or names.get("PROBABILITY")
        density_name = names.get("PROBDENSITY")
        if prob_name is None and density_name is None:
            raise ValueError("HEALPix table must contain PROB or PROBDENSITY")
        values = np.asarray(data[density_name or prob_name], dtype=float)
        nside = int(header.get("NSIDE", primary_header.get("NSIDE", 0)))
        if nside <= 0:
            nside_float = np.sqrt(values.size / 12.0)
            nside = int(round(nside_float))
            if 12 * nside * nside != values.size:
                raise ValueError("cannot infer NSIDE from flat HEALPix map")
        if 12 * nside * nside != values.size:
            raise ValueError("NSIDE does not match flat HEALPix map length")
        if nside & (nside - 1):
            raise ValueError("HEALPix NSIDE must be a power of two")
        ordering = str(header.get("ORDERING", primary_header.get("ORDERING", "RING"))).upper()
        if not (ordering.startswith("RING") or ordering.startswith("NEST")):
            raise ValueError(f"unsupported HEALPix ordering: {ordering}")
        ipix = np.arange(values.size, dtype=np.int64)
        import astropy_healpix as ah

        hp = ah.HEALPix(nside=nside, order="nested", frame="icrs")
        if ordering.startswith("RING"):
            ipix = np.asarray(hp.ring_to_nested(ipix), dtype=np.int64)
        level = int(round(np.log2(nside)))
        levels = np.full(values.size, level, dtype=np.int64)
        uniq = (4 * nside * nside + ipix).astype(np.uint64)
        area_value = 4.0 * np.pi / values.size
        area = np.full(values.size, area_value, dtype=float)
        if prob_name is not None:
            density = values / area_value
        else:
            density = _density_column(table_hdu, density_name)
        return _make_map(
            uniq, levels, ipix, density, area,
            source=source, normalize=normalize, ordering=ordering,
            raw_table=raw_table,
        )


def sky_map_footprint(skymap: SkyMap, *, max_depth: int | None = None):
    """Return an MOC covering all cells represented by ``skymap``."""
    from mocpy import MOC

    depth = skymap.max_level if max_depth is None else int(max_depth)
    return MOC.from_healpix_cells(
        ipix=np.asarray(skymap.ipix, dtype=np.int64),
        depth=np.asarray(skymap.levels, dtype=np.uint8),
        max_depth=depth,
    )


def _credible_indices(skymap: SkyMap, level: float) -> np.ndarray:
    """Select whole native pixels in a highest-density credible region.

    Multi-order maps must be ranked by probability density, never by each
    cell's integrated probability.  Equal-density cells are stably ordered by
    UNIQ, and the first cell reaching the requested cumulative probability is
    included in full.
    """
    if not 0.0 < float(level) <= 1.0:
        raise ValueError("credible level must lie in (0, 1]")
    density = np.asarray(skymap.prob_density_sr, dtype=float)
    by_uniq = np.argsort(np.asarray(skymap.uniq, dtype=np.uint64), kind="stable")
    order = by_uniq[np.argsort(-density[by_uniq], kind="stable")]
    cumulative = np.cumsum(np.asarray(skymap.pixel_probability, dtype=float)[order])
    count = int(np.searchsorted(cumulative, float(level), side="left")) + 1
    return order[:count]


def credible_region(skymap: SkyMap, level: float = 0.9, *, max_depth: int | None = None):
    """Build the greedy highest-density credible region as a native-cell MOC."""
    from mocpy import MOC

    selected = _credible_indices(skymap, level)
    # Coarsening selected cells would leak or remove probability at a region
    # boundary, making the MOC inconsistent with the reported statistics.
    depth = max(skymap.max_level, int(max_depth)) if max_depth is not None else skymap.max_level
    return MOC.from_healpix_cells(
        ipix=np.asarray(skymap.ipix, dtype=np.int64)[selected],
        depth=np.asarray(skymap.levels, dtype=np.uint8)[selected],
        max_depth=depth,
    )


def credible_region_stats(skymap: SkyMap, level: float = 0.9) -> dict[str, Any]:
    """Return MOC, area and actual cumulative probability for one level."""
    selected = _credible_indices(skymap, level)
    return {
        "moc": credible_region(skymap, level),
        "area_deg2": float(np.sum(np.asarray(skymap.pixel_area_sr)[selected]) / (np.pi / 180.0) ** 2),
        "probability": float(np.sum(np.asarray(skymap.pixel_probability)[selected])),
        "level": float(level),
    }


def probability_in_footprint(skymap: SkyMap, footprint: Any) -> float:
    """Integrate the normalized sky-map probability inside an MOC footprint.

    MOCPy intersects the MOC with the multi-order map through UNIQ tree
    semantics, so cells of one side fully contained in a cell of the other are
    counted exactly; only the boundary granularity of the MOC itself limits
    the result.  :func:`refined_probability` drives that granularity to
    convergence for analytically defined regions.
    """
    value = float(footprint.moc.probability_in_multiordermap(skymap.table))
    return float(np.clip(value, 0.0, 1.0))


@dataclass(frozen=True, slots=True)
class ProbabilityIntegral:
    """One region-integral estimate plus its MOC-refinement provenance."""

    value: float
    order: int
    converged: bool
    history: tuple[tuple[int, float], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "order": self.order,
            "converged": self.converged,
            "history": [[order, value] for order, value in self.history],
        }


def refined_probability(
    skymap: SkyMap,
    region_builder: Callable[[int], Any],
    *,
    order_start: int = 10,
    order_max: int = 13,
    tolerance: float = 1e-3,
) -> ProbabilityIntegral:
    """Integrate the map over a region rebuilt at increasing MOC orders.

    ``region_builder(order)`` must return the region MOC rasterized at that
    HEALPix order.  The estimate stops when the absolute probability change
    between adjacent orders drops below ``tolerance``; reaching ``order_max``
    first leaves ``converged=False`` so callers can record the unconverged
    status instead of silently trusting the last number.
    """
    order_start = int(order_start)
    order_max = int(order_max)
    if not 0 <= order_start <= order_max:
        raise ValueError("order_start must satisfy 0 <= order_start <= order_max")
    tolerance = float(tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    history: list[tuple[int, float]] = []
    previous: float | None = None
    for order in range(order_start, order_max + 1):
        region = region_builder(order)
        # ``region_builder`` may return a bare MOC or a footprint wrapper.
        moc = getattr(region, "moc", region)
        value = float(np.clip(float(moc.probability_in_multiordermap(skymap.table)), 0.0, 1.0))
        history.append((order, value))
        if previous is not None and abs(value - previous) < tolerance:
            return ProbabilityIntegral(value, order, True, tuple(history))
        previous = value
    return ProbabilityIntegral(previous if previous is not None else 0.0, order_max, False, tuple(history))
