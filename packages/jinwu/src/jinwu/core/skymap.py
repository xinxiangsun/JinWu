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
        """规范化数组并检查概率字段 / Coerce and validate map columns.

        各字段须为同形状非空一维数组；密度、面积和概率按各自规则校验。
        记录概率总和，但不在此自动归一化或核验 UNIQ 与像素层级的对应。
        Require equally shaped, nonempty 1-D columns and valid density, area,
        and mass. Store the total; do not normalize or cross-check UNIQ encoding.
        """
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
    """将角概率密度换算为 sr^-1 / Convert probability density to sr^-1.

    无单位数组按 sr^-1 解释；带单位值必须与逆立体角等价，否则报错。
    Bare arrays are interpreted as sr^-1. Unit-bearing input must convert to
    inverse solid angle; unsupported units raise ValueError.
    """
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
    """读取并换算 FITS 密度列 / Read a FITS density column in sr^-1.

    有 TUNIT 时按列单位换算，无 TUNIT 时按 sr^-1；返回浮点数组。
    Honor a present TUNIT card; otherwise assume sr^-1. Return a float array.
    """
    values = np.asarray(table_hdu.data[name], dtype=float)
    unit = getattr(table_hdu.columns[name], "unit", None)
    if unit:
        values = _unit_density(values * u.Unit(str(unit)))
    return values


def _finish(uniq, levels, ipix, density, area, *, ordering, source, normalize) -> SkyMapData:
    """从像素密度构造概率图 / Build a probability map from density and area.

    density 单位 sr^-1，area 单位 sr；概率质量为两者乘积。normalize=True
    且总概率为正时，同时缩放质量与密度；最终交给 SkyMapData 校验。
    Density is sr^-1 and area is sr; mass is their product. Normalize both
    mass and density when requested and the total is positive, then validate
    through SkyMapData. Input/source/order metadata are preserved.
    """
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
    """读取本地 HEALPix 概率 FITS / Read a local HEALPix probability FITS.

    Parameters
    ----------
    path : str or Path
        本地文件，支持 UNIQ/PROBDENSITY 多阶图，或含 PROB、PROBABILITY、
        PROBDENSITY 的平面图；URL 获取由 jinwu.gw 的告警层负责。
        Local multi-order UNIQ/PROBDENSITY or flat PROB/PROBABILITY/PROBDENSITY
        file. Remote fetching belongs to the jinwu.gw alert layer.
    normalize : bool
        默认 True，将正的总概率缩放到 1，并同步密度。
        Default True: scale a positive total mass to 1 and update density.

    Returns
    -------
    SkyMapData
        统一为 nested UNIQ，密度单位 sr^-1、面积单位 sr，像素概率无量纲。
        ordering 保留原平面图排序信息；source 为绝对路径。
        Nested UNIQ representation with density in sr^-1, area in sr and
        dimensionless pixel mass. Retain original flat ordering and source path.

    Raises
    ------
    FileNotFoundError, ValueError
        文件缺失、列/NSIDE/排序不兼容或概率字段无效；FITS 读取异常也上抛。
        Missing file or invalid columns, NSIDE, ordering or probabilities.
        FITS read errors propagate. Requires optional astropy-healpix at call time.
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
            # FITS K stores signed int64 and astropy-healpix's bit-scan ufunc
            # accepts that dtype.  Casting to uint64 first fails on clean CI
            # installs even for valid positive UNIQ indices.
            uniq_signed = np.asarray(table_hdu.data[names["UNIQ"]], dtype=np.int64)
            if np.any(uniq_signed < 4):
                raise ValueError("NUNIQ indices must be at least 4")
            levels, ipix = ah.uniq_to_level_ipix(uniq_signed)
            uniq_raw = uniq_signed.astype(np.uint64)
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
    """生成 ICRS 像素向量与概率质量 / Expand a map into vectors and masses.

    skymap 为 SkyMapData；max_order 为细分目标阶数，默认 6。粗于该阶的
    像素按均匀密度细分，较细像素保留原质量。返回 ``(unit_vectors, masses)``，
    形状为 (N, 3) 与 (N,)，均无量纲，质量总和保留原图总概率。
    ``skymap`` is SkyMapData; ``max_order`` defaults to 6. Coarser cells split
    at constant density; finer cells retain their mass. Return dimensionless
    ICRS Cartesian unit vectors (N, 3) and masses (N,), preserving total mass.

    输出按原像素层级分组，不能假定与输入数组一一对应；提高阶数会按
    4 的幂增加细分像素数量和内存需求。用于 GBM 搜索的空间先验。
    Output is grouped by original level, not necessarily input order. Higher
    orders increase subdivisions and memory by powers of 4. Used by GBM spatial
    priors; requires optional astropy-healpix at call time.
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
