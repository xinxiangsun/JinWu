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

from astropy.table import Table
import numpy as np

from .models import SkyMap


def load_skymap(path: str | Path, *, normalize: bool = True) -> SkyMap:
    """Read an LVK multi-order or ordinary HEALPix FITS sky map.

    File parsing is delegated to :func:`jinwu.core.skymap.load_skymap_with_table`,
    which also returns the raw FITS table for provenance.  The returned density
    is always in sr^-1 and the HEALPix indices are nested UNIQ values.  A map
    with negative, non-finite, or zero probability is rejected before plotting.
    """
    from jinwu.core.skymap import load_skymap_with_table

    raw_path = str(path)
    source_name = None
    if raw_path.lower().startswith(("http://", "https://")):
        # Sky-map URLs occur in untrusted notices, so use the alert HTTP
        # helper which validates every redirect target as well.
        from .alert import fetch_http_bytes

        content, final_url = fetch_http_bytes(raw_path)
        fileobj: Any = io.BytesIO(content)
        source_name = final_url
    else:
        fileobj = Path(path).expanduser().resolve()
    data, raw_table = load_skymap_with_table(
        fileobj, normalize=False, source_name=source_name,
    )
    table = Table({
        "UNIQ": np.asarray(data.uniq, dtype=np.uint64),
        "PROBDENSITY": np.asarray(data.prob_density_sr, dtype=float),
    })
    result = SkyMap(
        data.uniq, data.levels, data.ipix, data.prob_density_sr,
        data.pixel_area_sr, data.pixel_probability, table,
        source=data.source, ordering=data.ordering, raw_table=raw_table,
    )
    return result.normalized() if normalize else result


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
