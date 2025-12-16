"""High-level dataset containers.

These classes wrap lower-level OGIP data structures (e.g. LightcurveData,
PhaData, EventData) and provide a uniform, higher-level interface for
selection, slicing, merging, and background handling.

The API is intentionally minimal at this stage and can be extended as
more concrete analysis needs arise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Literal, Union

import numpy as np

from jinwu.core.file import LightcurveData, PhaData
from jinwu.core.file import RegionArea
from jinwu.core.ops import rebin_lightcurve

__all__ = [
    "LightcurveDataset",
    "SpectrumDataset",
    "JointDataset",
    "netdata",
]


# ---- typing helpers -------------------------------------------------------

LightcurveInput = LightcurveData | PhaData


def _coerce_lightcurve(obj: LightcurveInput, *, arg_name: str) -> LightcurveData:
    """Ensure the input is a LightcurveData instance.

    Accepts any OgipData but raises a clear error if the provided object
    is not actually a light curve (kind != 'lc'). This allows callers who
    use ``readfits`` without specifying ``kind='lc'`` to pass the result
    directly without extra casting while still keeping runtime safety.
    """

    if isinstance(obj, LightcurveData):
        return obj
    kind = getattr(obj, "kind", None)
    raise TypeError(
        f"{arg_name} must be a LightcurveData (kind='lc'), got {type(obj).__name__} with kind={kind!r}."
    )

@dataclass(slots=True)
class LightcurveDataset:
    """光变曲线容器类（支持多条曲线统一绘图）
    
    参数
    ----
    data : List[LightcurveData] | LightcurveData
        单个或多个光变曲线
    labels : List[str] | str | None, optional
        每条曲线的标签
    
    示例
    ----
    >>> ds = LightcurveDataset(data=[lc1, lc2, lc3], labels=["Src", "Bkg", "Net"])
    >>> ds.plot(ykind='rate', multiband=True)
    >>> ds = lc1 + lc2 + lc3  # 链式创建
    >>> ds = ds + lc4  # 添加新曲线
    """
    
    data: List[LightcurveData]
    labels: Optional[List[str]] = None
    
    def __post_init__(self):
        """确保 data 是列表"""
        if not isinstance(self.data, list):
            self.data = [self.data]
        if self.labels is not None and not isinstance(self.labels, list):
            self.labels = [self.labels]
        if self.labels is not None and len(self.labels) != len(self.data):
            raise ValueError(f"labels length ({len(self.labels)}) != data length ({len(self.data)})")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index: int) -> LightcurveData:
        return self.data[index]
    
    def __add__(self, other: Union[LightcurveData, 'LightcurveDataset']) -> 'LightcurveDataset':
        """添加新的光变曲线到容器
        
        示例
        ----
        >>> ds = ds + new_lc
        >>> ds = ds1 + ds2  # 合并两个 dataset
        """
        if isinstance(other, LightcurveData):
            return LightcurveDataset(
                data=self.data + [other],
                labels=self.labels
            )
        elif isinstance(other, LightcurveDataset):
            new_labels = None
            if self.labels is not None or other.labels is not None:
                new_labels = (self.labels or [None]*len(self.data)) + (other.labels or [None]*len(other.data))
            return LightcurveDataset(
                data=self.data + other.data,
                labels=new_labels
            )
        else:
            return NotImplemented
    
    def plot(self, *, ax=None, ykind: Literal['auto', 'rate', 'counts', 'flux'] = 'auto', 
              multiband: Union[bool, str] = "auto", colors=None, title=None, 
              grid: bool = True, **kwargs):
        """绘制光变曲线
        
        参数
        ----
        multiband : bool | 'auto', default='auto'
            True: 多子图模式；False: 叠加模式；'auto': 自动选择
        colors : list[str], optional
            每条曲线的颜色
        其他参数传递给 plot_lightcurve
        """
        from jinwu.core.plot import plot_lightcurve
        import matplotlib.pyplot as plt
        
        # 单条曲线：直接绘制
        if len(self.data) == 1:
            label = self.labels[0] if self.labels else None
            color = colors[0] if colors else None
            return plot_lightcurve(
                self.data[0], ax=ax, ykind=ykind, multiband=multiband,
                color=color, label=label, title=title, grid=grid, **kwargs
            )
        
        # 多条曲线
        if multiband == True or (multiband == "auto" and len(self.data) > 3):
            # 多子图模式
            fig, axes = plt.subplots(len(self.data), 1, figsize=(10, 3*len(self.data)), sharex=True)
            if not isinstance(axes, np.ndarray):
                axes = [axes]
            for i, lc in enumerate(self.data):
                label = self.labels[i] if self.labels else None
                color = colors[i] if colors else None
                plot_lightcurve(lc, ax=axes[i], ykind=ykind, color=color, label=label, grid=grid, **kwargs)
            if title:
                fig.suptitle(title)
            return axes
        else:
            # 叠加模式
            if ax is None:
                fig, ax = plt.subplots(figsize=(10, 6))
            for i, lc in enumerate(self.data):
                label = self.labels[i] if self.labels else None
                color = colors[i] if colors else None
                plot_lightcurve(lc, ax=ax, ykind=ykind, color=color, label=label, grid=grid, **kwargs)
            if title:
                ax.set_title(title)
            if self.labels:
                ax.legend()
            return ax


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


def netdata(
    source: LightcurveInput,
    background: Optional[LightcurveInput] = None,
    *,
    ratio: Optional[float] = None,
    use_exposure_weighted_ratio: bool = True,
    offset: float = 0.0,
) -> LightcurveData:
    """计算净光变曲线（源 - 背景）
    
    这是核心的背景减除函数，支持 LightcurveData 和未来的 PhaData。
    所有减法操作（包括 `src - bkg`）最终都调用此函数。
    
    参数
    ----
    source : LightcurveData | OgipData
        源光变曲线
    background : LightcurveData | OgipData | None
        背景光变曲线
    ratio : float, optional
        源背景缩放比例（None 则自动计算）
    use_exposure_weighted_ratio : bool, default=True
        自动计算时是否使用 (区域面积×曝光时间) 比值
    offset : float, default=0.0
        额外计数偏移（在计数空间减去）
    
    返回
    ----
    LightcurveData
        净光变曲线（若无背景则返回源本身）
    
    示例
    ----
    >>> net = netdata(src, bkg)  # 自动计算 ratio
    >>> net = netdata(src, bkg, ratio=1.5)  # 手动 ratio
    >>> net = src - bkg  # 等效于 netdata(src, bkg)
    
    说明
    ----
    算法流程：
    1. 确定 ratio（自动或手动）
    2. 对齐时间轴（若需要 rebin 背景）
    3. 转换到计数空间（使用 bin_exposure）
    4. 执行减法：net = src - ratio * bkg - offset
    5. 误差传播：err² = src_err² + (ratio * bkg_err)²
    6. 转回原始单位（rate/counts）
    7. 零曝光 bin 标记为 NaN
    """
    import numpy as np
    
    src_lc = _coerce_lightcurve(source, arg_name="source")
    
    if background is None:
        return src_lc
    
    bkg_lc = _coerce_lightcurve(background, arg_name="background")
    
    # ========== 1. 确定 ratio ==========
    if ratio is None:
        if use_exposure_weighted_ratio:
            src_area = src_lc.region.area if (src_lc.region and src_lc.region.area) else None
            bkg_area = bkg_lc.region.area if (bkg_lc.region and bkg_lc.region.area) else None
            src_exp = src_lc.exposure if src_lc.exposure else ValueError("Source exposure is None")
            bkg_exp = bkg_lc.exposure if bkg_lc.exposure else ValueError("Background exposure is None")
            
            if src_area is None or bkg_area is None or bkg_area == 0:
                raise ValueError(
                    "Cannot infer ratio: region area missing. "
                    "Provide ratio explicitly or ensure both have valid region.area"
                )
            ratio = (src_area * src_exp) / (bkg_area * bkg_exp)
        else:
            src_area = src_lc.region.area if (src_lc.region and src_lc.region.area) else None
            bkg_area = bkg_lc.region.area if (bkg_lc.region and bkg_lc.region.area) else None
            if src_area is None or bkg_area is None or bkg_area == 0:
                raise ValueError("Cannot infer ratio from areas. Provide explicitly.")
            ratio = src_area / bkg_area
    
    # ========== 2. 对齐时间轴 ==========
    bkg_aligned = bkg_lc
    if not (src_lc.time.shape == bkg_lc.time.shape and np.allclose(src_lc.time, bkg_lc.time)):
        from jinwu.core.ops import rebin_lightcurve
        if src_lc.dt is None:
            raise ValueError("Cannot align: source.dt is None")
        bkg_aligned = rebin_lightcurve(
            bkg_lc, binsize=src_lc.dt, method='sum',
            align_ref=src_lc.timezero if src_lc.timezero else None
        )
    
    # ========== 3. 转换到计数空间 ==========
    def _to_counts(lc):
        """将 LightcurveData 转为计数空间，返回 (counts, err_counts, exposure_array)"""
        exp = lc.bin_exposure if lc.bin_exposure is not None else (lc.dt if lc.dt is not None else 1.0)
        exp_arr = np.asarray(exp, dtype=float)
        
        if lc.is_rate:
            counts = lc.value * exp_arr
            err_counts = (lc.error * exp_arr) if lc.error is not None else np.sqrt(np.maximum(counts, 0.0))
        else:
            counts = lc.value
            err_counts = lc.error if lc.error is not None else np.sqrt(np.maximum(counts, 0.0))
        
        return counts, err_counts, exp_arr
    
    src_counts, src_err, src_exp_arr = _to_counts(src_lc)
    bkg_counts, bkg_err, _ = _to_counts(bkg_aligned)
    
    # ========== 4. 执行减法（计数空间） ==========
    net_counts = src_counts - float(ratio) * bkg_counts - float(offset)
    net_var = (src_err ** 2) + (float(ratio) ** 2) * (bkg_err ** 2)
    
    # ========== 5. 转换回原始单位 ==========
    if src_lc.is_rate:
        out_value = net_counts / src_exp_arr
        out_err = np.sqrt(net_var) / src_exp_arr
    else:
        out_value = net_counts
        out_err = np.sqrt(net_var)
    
    # ========== 6. 处理零曝光 bin ==========
    if src_lc.bin_exposure is not None:
        zero_mask = (np.asarray(src_lc.bin_exposure, dtype=float) == 0.0)
        if np.any(zero_mask):
            out_value = out_value.astype(float).copy()
            out_err = out_err.astype(float).copy()
            out_value[zero_mask] = np.nan
            out_err[zero_mask] = np.nan
    
    # ========== 7. 构造结果 ==========
    net = LightcurveData(
        time=src_lc.time, value=out_value, error=out_err,
        dt=src_lc.dt, exposure=src_lc.exposure, is_rate=src_lc.is_rate,
        bin_exposure=src_lc.bin_exposure,
        timezero=src_lc.timezero, timezero_obj=src_lc.timezero_obj,
        bin_lo=src_lc.bin_lo, bin_hi=src_lc.bin_hi,
        tstart=src_lc.tstart, tseg=src_lc.tseg,
        gti_start=src_lc.gti_start, gti_stop=src_lc.gti_stop,
        quality=src_lc.quality, fracexp=src_lc.fracexp,
        backscal=src_lc.backscal, areascal=src_lc.areascal,
        telescop=src_lc.telescop, timesys=src_lc.timesys, mjdref=src_lc.mjdref,
        path=src_lc.path, header=src_lc.header, meta=src_lc.meta,
        headers_dump=src_lc.headers_dump, region=src_lc.region,
        columns=src_lc.columns, ratio=ratio,
    )
    
    return net
