"""High-level dataset containers.

These classes wrap lower-level OGIP data structures (e.g. LightcurveData,
PhaData, EventData) and provide a uniform, higher-level interface for
selection, slicing, merging, and background handling.

The API is intentionally minimal at this stage and can be extended as
more concrete analysis needs arise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from jinwu.core.file import LightcurveData, PhaData

__all__ = [
    "LightcurveDataset",
    "SpectrumDataset",
    "JointDataset",
    "netdata",
]

@dataclass(slots=True)
class LightcurveDataset:
    """High-level light-curve container.

    Parameters
    ----------
    data : LightcurveData
        The underlying OGIP-like light-curve data object.
    label : str, optional
        A human-readable label (e.g. instrument/band name).
    background : LightcurveDataset | None, optional
        Optional background light curve associated with this source.

    Notes
    -----
    This class is meant to be a thin convenience wrapper around
    :class:`jinwu.core.file.LightcurveData`, adding methods for selection
    and simple arithmetic. It does **not** try to hide the underlying
    data object; you can always access ``.data`` directly.
    """

    data: LightcurveData
    label: Optional[str] = None
    background: Optional["LightcurveDataset"] = None
    # area_ratio: how many "source areas" correspond to one background area.
    # Typically area_ratio = A_src / A_bkg. It is *not* inferred automatically
    # here; callers should compute it from RegionArea/region keywords or
    # BACKSCAL and set it explicitly when creating the dataset.
    area_ratio: Optional[float] = None

    # ---- basic views ----

    @property
    def time(self) -> np.ndarray:
        return self.data.time

    @property
    def value(self) -> np.ndarray:
        return self.data.value

    @property
    def error(self) -> Optional[np.ndarray]:
        return self.data.error

    # ---- simple selection/slicing ----

    def select_time(self, tmin: float | None = None, tmax: float | None = None) -> "LightcurveDataset":
        """Return a new dataset restricted to a time interval.

        The selection is half-open in spirit: [tmin, tmax]. If any is
        ``None``, it is left unbounded on that side.
        """

        t = self.data.time
        mask = np.ones_like(t, dtype=bool)
        if tmin is not None:
            mask &= t >= tmin
        if tmax is not None:
            mask &= t <= tmax
        return self._apply_mask(mask)

    def _apply_mask(self, mask: np.ndarray) -> "LightcurveDataset":
        d = self.data
        new = LightcurveData(
            kind=d.kind,
            path=d.path,
            time=d.time[mask],
            value=d.value[mask],
            error=d.error[mask] if d.error is not None else None,
            dt=d.dt,
            exposure=d.exposure,
            is_rate=d.is_rate,
            header=d.header,
            meta=d.meta,
            headers_dump=d.headers_dump,
            region=d.region,
        )
        bg = self.background._apply_mask(mask) if self.background is not None else None
        return LightcurveDataset(
            data=new,
            label=self.label,
            background=bg,
            area_ratio=self.area_ratio,
        )

    # ---- simple background subtraction hook ----

    def subtract_background(self) -> "LightcurveDataset":
        """Return a background-subtracted light curve.

        公式
        ------
        使用

        ``value_net = value_src - area_ratio * value_bkg``

        其中 ``area_ratio`` 一般由调用方根据源/背景区域面积
        （例如 :class:`jinwu.core.file.RegionArea` 或 BACKSCAL）
        预先计算并填写到 ``LightcurveDataset.area_ratio`` 中。

        若 ``area_ratio`` 为 ``None`` 或背景缺失，则直接返回自身
        （不做减背景）。

        误差按独立误差传播：

        ``err_net = sqrt(err_src^2 + (area_ratio * err_bkg)^2)``。

        当前实现仍然假定源与背景 time 轴完全对齐；更通用的
        重采样/对齐可在上层工具中实现。
        """

        if self.background is None or self.area_ratio is None:
            return self

        src = self.data
        bkg = self.background.data
        if src.time.shape != bkg.time.shape or not np.allclose(src.time, bkg.time):
            raise ValueError("Source and background light curves must share the same time bins for now.")

        area_ratio = float(self.area_ratio)
        val = src.value - area_ratio * bkg.value
        if src.error is not None or bkg.error is not None:
            src_err = src.error if src.error is not None else np.zeros_like(src.value)
            bkg_err = bkg.error if bkg.error is not None else np.zeros_like(bkg.value)
            err = np.sqrt(src_err**2 + (area_ratio * bkg_err) ** 2)
        else:
            err = None

        new = LightcurveData(
            kind=src.kind,
            path=src.path,
            time=src.time,
            value=val,
            error=err,
            dt=src.dt,
            exposure=src.exposure,
            is_rate=src.is_rate,
            header=src.header,
            meta=src.meta,
            headers_dump=src.headers_dump,
            region=src.region,
        )
        return LightcurveDataset(data=new, label=self.label, background=None)


@dataclass(slots=True)
class SpectrumDataset:
    """High-level spectral dataset container.

    Parameters
    ----------
    data : PhaData
        The underlying OGIP spectral data (SPECTRUM + optional EBOUNDS).
    label : str, optional
        A human-readable label for this spectrum (e.g. instrument, epoch).
    background : SpectrumDataset | None, optional
        Optional associated background spectrum.
    """

    data: PhaData
    label: Optional[str] = None
    background: Optional["SpectrumDataset"] = None


@dataclass(slots=True)
class JointDataset:
    """A simple container for multiple datasets.

    This can hold any combination of light curves and spectra that
    conceptually belong to the same astrophysical source/event.
    """

    lightcurves: List[LightcurveDataset]
    spectra: List[SpectrumDataset]

    def add_lightcurve(self, lc: LightcurveDataset) -> None:
        self.lightcurves.append(lc)

    def add_spectrum(self, spec: SpectrumDataset) -> None:
        self.spectra.append(spec)


def _region_area_value(lc: LightcurveData) -> Optional[float]:
    reg = lc.region
    if reg is None or reg.area is None:
        return None
    if reg.area <= 0:
        return None
    try:
        return float(reg.area)
    except Exception:
        return None


def _infer_area_ratio(source: LightcurveData, background: LightcurveData) -> Optional[float]:
    src_area = _region_area_value(source)
    bkg_area = _region_area_value(background)
    if src_area is None or bkg_area is None or bkg_area == 0:
        return None
    return src_area / bkg_area


def netdata(
    source: LightcurveData,
    background: Optional[LightcurveData] = None,
    *,
    label: Optional[str] = None,
    background_label: Optional[str] = None,
    area_ratio: Optional[float] = None,
) -> LightcurveDataset:
    """Return a light-curve dataset (net if possible).

    行为概述
    --------
    1. 先把源/背景光变包装成 :class:`LightcurveDataset`；
    2. 若未提供 ``area_ratio`` 且源/背景区域面积可得，则自动计算；
    3. 若背景和 ``area_ratio`` 均可用，则立即执行减背景并返回净光变；
       否则返回带背景引用的原始数据集，方便上层后续手动处理。
    """

    background_ds = None
    if background is not None:
        background_ds = LightcurveDataset(data=background, label=background_label)

    ratio = area_ratio
    if ratio is None and background is not None:
        ratio = _infer_area_ratio(source, background)

    dataset = LightcurveDataset(
        data=source,
        label=label,
        background=background_ds,
        area_ratio=ratio,
    )
    if background_ds is None or dataset.area_ratio is None:
        return dataset
    return dataset.subtract_background()
