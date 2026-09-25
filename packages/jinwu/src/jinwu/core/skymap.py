"""Pure-data HEALPix probability-map reader shared by instruments and GW tools.

This module is deliberately dependency-light: it only needs ``numpy``,
``astropy`` and ``astropy-healpix`` — *not* the heavier ``mocpy`` / ``healpy``
stack used by :mod:`jinwu.gw` for MOC integration and plotting.  Instruments
(e.g. the Fermi/GBM subthreshold search) can therefore weight a source
probability sky map without pulling in the full ``jinwu-gw`` distribution.

Layering contract (breaks the former ``jinwu-fermi -> jinwu-gw`` dependency):

* ``jinwu.core.skymap`` — parse a local FITS file into :class:`SkyMapData`.
* ``jinwu.gw.skymap`` — builds on the same pixel contract and adds
  alert-provenance fetching, credible regions and MOC integration.  Its public
  ``load_skymap`` / ``SkyMap`` names remain valid for backward compatibility.

The pixel contract is identical in both layers: nested HEALPix UNIQ indexing,
density in sr^-1, per-pixel probability = density * area.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import astropy.units as u
import numpy as np
from astropy.io import fits

__all__ = ["SkyMapData", "load_skymap", "sky_map_pixel_vectors"]

#: HEALPix order up to which a coarse native cell is subdivided with constant
#: density when a caller needs unit vectors per pixel (matches the GBM
#: targeted-search default; finer cells keep their original mass).
DEFAULT_VECTOR_ORDER = 6


@dataclass(frozen=True, slots=True)
class SkyMapData:
    """Normalized HEALPix probability map (nested UNIQ, density in sr^-1).

    ``uniq`` is the nested HEALPix UNIQ index (``4*4**level + ipix``);
    ``prob_density_sr`` is in sr^-1; ``pixel_probability`` is density times
    pixel area and sums to ``total_probability`` (1.0 when normalized).
    ``ordering`` records the original FITS ORDERING card.
    """

    uniq: Any
    levels: Any
    ipix: Any
    prob_density_sr: Any
    pixel_area_sr: Any
    pixel_probability: Any
    ordering: str = "NUNIQ"
    source: str | None = None
    total_probability: float = field(init=False)

    def __post_init__(self) -> None:
        uniq = np.asarray(self.uniq, dtype=np.uint64)
        levels = np.asarray(self.levels, dtype=np.int64)
        ipix = np.asarray(self.ipix, dtype=np.int64)
        density = np.asarray(self.prob_density_sr, dtype=float)
        area = np.asarray(self.pixel_area_sr, dtype=float)
        prob = np.asarray(self.pixel_probability, dtype=float)
        shapes = {a.shape for a in (uniq, levels, ipix, density, area, prob)}
        if len(shapes) != 1:
            raise ValueError("sky-map columns must have identical shapes")
        if uniq.ndim != 1 or uniq.size == 0:
            raise ValueError("sky map must contain at least one pixel")
        if np.any(~np.isfinite(density)) or np.any(density < 0):
            raise ValueError("probability density must be finite and non-negative")
        if np.any(~np.isfinite(area)) or np.any(area <= 0):
            raise ValueError("pixel areas must be finite and positive")
        if np.any(~np.isfinite(prob)) or np.any(prob < 0):
            raise ValueError("pixel probabilities must be finite and non-negative")
        total = float(np.sum(prob))
        if not np.isfinite(total) or total <= 0:
            raise ValueError("sky map has no positive probability")
        object.__setattr__(self, "uniq", uniq)
        object.__setattr__(self, "levels", levels)
        object.__setattr__(self, "ipix", ipix)
        object.__setattr__(self, "prob_density_sr", density)
        object.__setattr__(self, "pixel_area_sr", area)
        object.__setattr__(self, "pixel_probability", prob)
        object.__setattr__(self, "total_probability", total)


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


def _finish(uniq, levels, ipix, density, area, *, ordering, source, normalize) -> SkyMapData:
    density = np.asarray(density, dtype=float)
    area = np.asarray(area, dtype=float)
    probability = density * area
    if normalize:
        total = float(probability.sum())
        if total > 0:
            probability = probability / total
            density = probability / area
    return SkyMapData(
        uniq=uniq, levels=levels, ipix=ipix,
        prob_density_sr=density, pixel_area_sr=area, pixel_probability=probability,
        ordering=ordering, source=source,
    )


def load_skymap(path: str | Path, *, normalize: bool = True) -> SkyMapData:
    """Read a local LVK multi-order (NUNIQ) or flat HEALPix FITS probability map.

    Remote URLs are intentionally **not** supported here; fetching untrusted
    alert URLs with per-redirect validation lives in :mod:`jinwu.gw.alert` and
    :mod:`jinwu.gw.skymap`.  Density is always returned in sr^-1.
    """
    import astropy_healpix as ah

    local_path = Path(path).expanduser().resolve()
    if not local_path.is_file():
        raise FileNotFoundError(local_path)
    with fits.open(local_path, memmap=False) as hdul:
        table_hdu = next(
            (hdu for hdu in hdul[1:] if getattr(hdu, "data", None) is not None), None
        )
        if table_hdu is None:
            raise ValueError(f"no HEALPix table found in {path}")
        names = {str(n).upper(): n for n in table_hdu.columns.names}
        header = table_hdu.header
        primary_header = hdul[0].header

        # ---- Multi-order map (NUNIQ / PROBDENSITY) ----
        if "UNIQ" in names and "PROBDENSITY" in names:
            uniq_raw = np.asarray(table_hdu.data[names["UNIQ"]], dtype=np.uint64)
            levels, ipix = ah.uniq_to_level_ipix(uniq_raw)
            density = _density_column(table_hdu, names["PROBDENSITY"])
            area = ah.nside_to_pixel_area(ah.level_to_nside(levels)).to_value(u.sr)
            return _finish(
                uniq_raw, levels, ipix, density, area,
                ordering="NUNIQ", source=str(local_path), normalize=normalize,
            )

        # ---- Flat map (PROB or PROBDENSITY) ----
        prob_name = names.get("PROB") or names.get("PROBABILITY")
        density_name = names.get("PROBDENSITY")
        if prob_name is None and density_name is None:
            raise ValueError("HEALPix table must contain PROB or PROBDENSITY")
        values = np.asarray(table_hdu.data[density_name or prob_name], dtype=float)
        nside = int(header.get("NSIDE", primary_header.get("NSIDE", 0)))
        if nside <= 0:
            nside = int(round(float(np.sqrt(values.size / 12.0))))
        if 12 * nside * nside != values.size:
            raise ValueError("NSIDE does not match flat HEALPix map length")
        if nside & (nside - 1):
            raise ValueError("HEALPix NSIDE must be a power of two")
        ordering = str(header.get("ORDERING", primary_header.get("ORDERING", "RING"))).upper()
        if not (ordering.startswith("RING") or ordering.startswith("NEST")):
            raise ValueError(f"unsupported HEALPix ordering: {ordering}")
        ipix = np.arange(values.size, dtype=np.int64)
        if ordering.startswith("RING"):
            hp = ah.HEALPix(nside=nside, order="nested", frame="icrs")
            ipix = np.asarray(hp.ring_to_nested(ipix), dtype=np.int64)
        level = int(round(float(np.log2(nside))))
        levels = np.full(values.size, level, dtype=np.int64)
        uniq = (4 * nside * nside + ipix).astype(np.uint64)
        area_value = 4.0 * np.pi / values.size
        area = np.full(values.size, area_value, dtype=float)
        if prob_name is not None:
            density = values / area_value
        else:
            density = _density_column(table_hdu, density_name)
        return _finish(
            uniq, levels, ipix, density, area,
            ordering=ordering, source=str(local_path), normalize=normalize,
        )


def sky_map_pixel_vectors(
    skymap: SkyMapData, *, max_order: int = DEFAULT_VECTOR_ORDER
) -> tuple[np.ndarray, np.ndarray]:
    """Expand a map into per-(sub)pixel ICRS unit vectors and probability mass.

    Cells coarser than ``max_order`` are subdivided with constant density; finer
    cells keep their original mass.  This is the representation consumed by the
    GBM targeted search's spatial prior.  Returns ``(unit_vectors, masses)``
    with ``unit_vectors`` of shape ``(N, 3)`` and ``masses`` summing to
    ``skymap.total_probability``.
    """
    import astropy_healpix as ah

    vectors, probabilities = [], []
    for level in np.unique(skymap.levels):
        select = skymap.levels == level
        cells = skymap.ipix[select]
        prob = skymap.pixel_probability[select]
        factor = 4 ** max(0, max_order - int(level))
        cells = (cells[:, None] * factor + np.arange(factor)).ravel()
        prob = np.repeat(prob / factor, factor)
        hp = ah.HEALPix(nside=2 ** max(max_order, int(level)), order="nested", frame="icrs")
        vectors.append(hp.healpix_to_skycoord(cells).cartesian.xyz.value.T)
        probabilities.append(prob)
    return np.concatenate(vectors), np.concatenate(probabilities)
