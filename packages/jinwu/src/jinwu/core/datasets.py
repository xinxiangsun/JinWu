"""High-level dataset containers.

These classes wrap lower-level OGIP data structures (e.g. LightcurveData,
PhaData, EventData) and provide a uniform, higher-level interface for
selection, slicing, merging, and background handling.

The API is intentionally minimal at this stage and can be extended as
more concrete analysis needs arise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Literal, Union, cast

import numpy as np

from jinwu.core.base import LightcurveDataBase, PhaBase, RegionArea
from jinwu.core.ops import rebin_lightcurve

if TYPE_CHECKING:
    from jinwu.core.data import LightcurveData, PhaData

__all__ = [
    "LightcurveDataset",
    "SpectrumDataset",
    "JointDataset",
    "netdata",
]


# ---- typing helpers -------------------------------------------------------

LightcurveInput = LightcurveDataBase | PhaBase


def _coerce_lightcurve(obj: LightcurveInput, *, arg_name: str) -> LightcurveDataBase:
    """Ensure the input is a LightcurveData instance.

    Accepts any OgipData but raises a clear error if the provided object
    is not actually a light curve (kind != 'lc'). This allows callers who
    use ``readfits`` without specifying ``kind='lc'`` to pass the result
    directly without extra casting while still keeping runtime safety.

    校验并返回原光变对象，不创建副本。接受 LightcurveDataBase，或在
    notebook 重载后具有 kind='lc' 与必要属性的对象；其他类型抛 TypeError。
    当前不接受 PHA 能谱，也不检查时间/数值数组的科学有效性。
    Return the original lightcurve without copying. Accept LightcurveDataBase
    or a reloaded kind='lc' object with required attributes. Reject other types,
    including PHA spectra, with TypeError; do not validate array contents here.
    """

    # Fast-path: normal case (no module reload / single class identity)
    if isinstance(obj, LightcurveDataBase):
        return obj

    kind = getattr(obj, "kind", None)

    # Duck-typing fallback: accept any object that looks like a lightcurve.
    # This keeps runtime behavior stable across reloads and avoids surprising
    # failures in interactive notebooks.
    if kind == "lc":
        required_attrs = (
            "time",
            "value",
            "error",
            "dt",
            "timezero",
            "bin_exposure",
            "exposure",
            "region",
            "is_rate",
        )
        if all(hasattr(obj, attr) for attr in required_attrs):
            return cast(LightcurveDataBase, obj)

    raise TypeError(
        f"{arg_name} must be a LightcurveData (kind='lc'), got {type(obj).__name__} from {type(obj).__module__} with kind={kind!r}."
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
    
    data: List[LightcurveDataBase]
    labels: Optional[List[str]] = None
    
    def __post_init__(self):
        """统一曲线与标签列表 / Normalize lightcurve and label lists.

        非 list 输入包装为单项列表；标签数量不匹配抛 ValueError。
        不复制成员光变或校验其科学内容。
        Wrap non-list data/labels as one-item lists. Unequal label/data counts
        raise ValueError; member lightcurves are neither copied nor validated.
        """
        if not isinstance(self.data, list):
            self.data = [self.data]
        if self.labels is not None and not isinstance(self.labels, list):
            self.labels = [self.labels]
        if self.labels is not None and len(self.labels) != len(self.data):
            raise ValueError(f"labels length ({len(self.labels)}) != data length ({len(self.data)})")
    
    def __len__(self) -> int:
        """返回曲线条数 / Return the number of lightcurves, not time bins."""
        return len(self.data)
    
    def __getitem__(self, index: int) -> LightcurveDataBase:
        """按索引访问原曲线 / Return the referenced lightcurve by list index.

        支持底层列表的负索引和切片；切片实际返回列表，不复制数据。
        Follow list indexing, including negative indices and slices; slices
        return a list despite the scalar type hint. No data copy is made.
        """
        return self.data[index]
    
    def __add__(self, other: Union[LightcurveDataBase, 'LightcurveDataset']) -> 'LightcurveDataset':
        """添加新的光变曲线到容器
        
        示例
        ----
        >>> ds = ds + new_lc
        >>> ds = ds1 + ds2  # 合并两个 dataset

        Return a new dataset containing references to the concatenated curves;
        this operation does not merge time bins. Two datasets retain labels
        only if both provide them. Adding a bare curve reuses current labels,
        so a labeled dataset can raise ValueError for the resulting mismatch.
        Unsupported operands return NotImplemented.
        返回新容器但不合并时间分箱、不复制曲线；已有标签的容器直接加
        单条曲线时，当前标签未扩展，可能因数量不匹配抛出 ValueError。
        """
        if isinstance(other, LightcurveDataBase):
            return LightcurveDataset(
                data=self.data + [other],
                labels=self.labels
            )
        elif isinstance(other, LightcurveDataset):
            new_labels = None
            if self.labels is not None and other.labels is not None:
                new_labels = self.labels + other.labels
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

        Delegate values/units and kwargs to core.plot.plot_lightcurve. A single
        curve returns that plotting result; several curves return one overlay
        Axes or an array of shared-x Axes. Auto uses separate panels above three
        curves. Multi-panel mode creates a new figure even if ax was supplied.
        This creates artists and does not save output files.
        数值与单位解释交由 plot_lightcurve；多曲线返回叠加 Axes 或子图数组。
        auto 在超过三条曲线时创建多面板；多面板不复用传入的 ax，不自动保存。
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
        """追加光变容器引用 / Append a lightcurve dataset by reference.

        原位修改列表，返回 None；不做时间对齐或校验。
        Mutate the list, return None; no time alignment or validation.
        """
        self.lightcurves.append(lc)

    def add_spectrum(self, spec: SpectrumDataset) -> None:
        """追加能谱容器引用 / Append a spectral dataset by reference.

        原位修改列表，返回 None；不执行联合拟合或响应校验。
        Mutate the list, return None; no joint fit or response validation.
        """
        self.spectra.append(spec)


def netdata(
    source: LightcurveInput,
    background: Optional[LightcurveInput] = None,
    *,
    ratio: Optional[float] = None,
    use_exposure_weighted_ratio: bool = True,
    offset: float = 0.0,
) -> LightcurveDataBase:
    """计算净光变曲线（源 - 背景）
    
    这是核心的光变背景减除函数；当前只接受光变对象，不支持 PHA 能谱。
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

    English contract
    ----------------
    source/background must be lightcurves, with value in counts or counts/s
    according to is_rate and time/exposure in consistent seconds. With no
    background, return source itself and ignore ratio/offset; no copy is made.
    Otherwise create a new instance of the source class and preserve its metadata.

    ratio is a fixed, dimensionless multiplier in count space. If omitted,
    infer (source area * total exposure)/(background area * total exposure),
    or only the area ratio when use_exposure_weighted_ratio=False. Required
    missing area/exposure metadata raises ValueError. Ratio uncertainty is
    not propagated. offset is a deterministic count offset subtracted per bin.

    Different grids trigger background rebinning anchored to the source's
    first bin edge, followed by a strict grid check. TIMEZERO must agree;
    callers must convert time references before subtraction. Misaligned grids
    are rejected rather than subtracted by array position.
    Rates are converted using bin_exposure, falling back to dt, then 1.0.
    Missing errors use sqrt(max(counts, 0)). Independent variances add as
    source_variance + ratio**2 * background_variance; covariance is not modeled.

    Negative net counts are retained. The primary value/error fields of explicit
    zero-exposure source bins become NaN; nonpositive/nonfinite exposure also
    yields NaN when converting to rate. Auxiliary counts fields can still hold
    count-space values. Preserved err_dist metadata does not establish that the
    background-subtracted data follow a Poisson distribution.

    注意：ratio 为计数空间的固定无量纲系数，未传播其不确定度；offset
    为逐 bin 减去的确定性计数偏移。无背景时返回源原对象并忽略 offset。
    输入时间系统须已一致。净计数允许为负；来源 err_dist 元数据不能
    证明扣背景后的净数据仍服从 Poisson 分布，误差传播未考虑协方差。
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
            src_exp = src_lc.exposure
            bkg_exp = bkg_lc.exposure
            
            if src_area is None or bkg_area is None or bkg_area == 0:
                raise ValueError(
                    "Cannot infer ratio: region area missing. "
                    "Provide ratio explicitly or ensure both have valid region.area"
                )
            if src_exp is None or bkg_exp is None or bkg_exp == 0:
                raise ValueError(
                    "Cannot infer exposure-weighted ratio: exposure missing or zero. "
                    "Provide ratio explicitly or ensure both have valid exposure"
                )
            ratio = (src_area * src_exp) / (bkg_area * bkg_exp)
        else:
            src_area = src_lc.region.area if (src_lc.region and src_lc.region.area) else None
            bkg_area = bkg_lc.region.area if (bkg_lc.region and bkg_lc.region.area) else None
            if src_area is None or bkg_area is None or bkg_area == 0:
                raise ValueError("Cannot infer ratio from areas. Provide explicitly.")
            ratio = src_area / bkg_area
    
    # ========== 2. 对齐时间轴 ==========
    if src_lc.time is None or bkg_lc.time is None:
        raise ValueError("source/background time array is None")

    src_time = np.asarray(src_lc.time, dtype=float)

    from jinwu.core.ops import _infer_bin_geometry
    src_origin = float(src_lc.timezero or 0.0)
    bkg_origin = float(bkg_lc.timezero or 0.0)
    if not np.isfinite(src_origin + bkg_origin) or not np.isclose(src_origin, bkg_origin, rtol=0, atol=1e-7):
        raise ValueError("source/background TIMEZERO differs; convert to a common time reference first")
    src_lo, src_hi, _ = _infer_bin_geometry(src_lc)

    def grid_matches(lc):
        lo, hi, _ = _infer_bin_geometry(lc)
        arrays = ((src_time, np.asarray(lc.time, dtype=float)), (src_lo, lo), (src_hi, hi))
        return all(a.shape == b.shape and np.all(np.isfinite(a))
                   and np.all(np.isfinite(b)) and np.allclose(a, b, rtol=0, atol=1e-7)
                   for a, b in arrays)

    bkg_aligned = bkg_lc
    if not grid_matches(bkg_lc):
        if src_lc.dt is None:
            raise ValueError("Cannot align: source.dt is None")

        binsize_arr = np.asarray(src_lc.dt, dtype=float)
        binsize = float(np.median(binsize_arr)) if binsize_arr.ndim > 0 else float(binsize_arr)
        if not np.isfinite(binsize) or binsize <= 0:
            raise ValueError(f"Cannot align: invalid source binsize={binsize}")

        bkg_lo, bkg_hi, _ = _infer_bin_geometry(bkg_lc)
        # Aggregation is valid only for complete background bins contained in
        # the source bins. Rebinning cannot recover counts from shifted edges.
        if (not src_lo.size or not bkg_lo.size
                or not np.allclose(src_hi - src_lo, binsize, rtol=0, atol=1e-7)
                or not np.isclose(bkg_lo[0], src_lo[0], rtol=0, atol=1e-7)
                or not np.isclose(bkg_hi[-1], src_hi[-1], rtol=0, atol=1e-7)
                or not np.allclose(bkg_lo[1:], bkg_hi[:-1], rtol=0, atol=1e-7)):
            raise ValueError("Cannot align source/background: incomplete or shifted bin coverage")
        membership = np.searchsorted(src_hi, bkg_lo + 1e-7, side='right')
        if np.any(membership >= src_hi.size) or np.any(bkg_hi > src_hi[membership] + 1e-7):
            raise ValueError("Cannot align source/background: background bins cross source edges")

        bkg_aligned = rebin_lightcurve(
            bkg_lc, binsize=binsize, method='auto',
            align_ref=float(src_lo[0]) if src_lo.size else None
        )
        if not grid_matches(bkg_aligned):
            raise ValueError("source/background bin centers and edges do not align after rebinning")
    
    # ========== 3. 转换到计数空间 ==========
    def _to_counts(lc):
        """转到计数空间 / Return counts, count error and exposure array.

        曝光优先 bin_exposure，再 dt，最后 1.0；rate 乘曝光还原计数。
        无误差时取 sqrt(max(counts, 0))；value 缺失抛 ValueError。
        Prefer bin_exposure, then dt, then 1.0. Multiply rates/errors by exposure;
        absent errors use sqrt(max(counts, 0)). Missing value raises ValueError.
        """
        if lc.value is None:
            raise ValueError("LightcurveData.value is None")
        exp = lc.bin_exposure if lc.bin_exposure is not None else (lc.dt if lc.dt is not None else 1.0)
        exp_arr = np.asarray(exp, dtype=float)
        val = np.asarray(lc.value, dtype=float)
        
        if lc.is_rate:
            counts = val * exp_arr
            err_counts = (np.asarray(lc.error, dtype=float) * exp_arr) if lc.error is not None else np.sqrt(np.maximum(counts, 0.0))
        else:
            counts = val
            err_counts = np.asarray(lc.error, dtype=float) if lc.error is not None else np.sqrt(np.maximum(counts, 0.0))
        
        return counts, err_counts, exp_arr
    
    src_counts, src_err, src_exp_arr = _to_counts(src_lc)
    bkg_counts, bkg_err, _ = _to_counts(bkg_aligned)
    
    # ========== 4. 执行减法（计数空间） ==========
    net_counts = src_counts - float(ratio) * bkg_counts - float(offset)
    net_var = (src_err ** 2) + (float(ratio) ** 2) * (bkg_err ** 2)
    
    # ========== 5. 转换回原始单位 ==========
    if src_lc.is_rate:
        valid_exp = np.isfinite(src_exp_arr) & (src_exp_arr > 0)
        out_value = np.full_like(net_counts, np.nan, dtype=float)
        out_err = np.full_like(net_counts, np.nan, dtype=float)
        out_value[valid_exp] = net_counts[valid_exp] / src_exp_arr[valid_exp]
        out_err[valid_exp] = np.sqrt(net_var[valid_exp]) / src_exp_arr[valid_exp]
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
    if src_lc.is_rate:
        out_rate = out_value
        out_rate_err = out_err
        out_counts = net_counts
        out_counts_err = np.sqrt(net_var)
    else:
        out_counts = out_value
        out_counts_err = out_err
        out_rate = np.full_like(net_counts, np.nan, dtype=float)
        out_rate_err = np.full_like(net_counts, np.nan, dtype=float)
        valid_exp = np.isfinite(src_exp_arr) & (src_exp_arr > 0)
        out_rate[valid_exp] = out_counts[valid_exp] / src_exp_arr[valid_exp]
        out_rate_err[valid_exp] = out_counts_err[valid_exp] / src_exp_arr[valid_exp]

    bin_lo = src_lc.bin_lo
    bin_hi = src_lc.bin_hi
    bin_width = src_lc.bin_width
    binning = src_lc.binning
    if (bin_lo is None or bin_hi is None or bin_width is None) and src_lc.time is not None:
        try:
            lo, hi, bw = src_lc._resolve_bin_geometry()
            bin_lo = lo
            bin_hi = hi
            bin_width = bw
            if bw.size > 0 and np.allclose(bw, float(np.median(bw)), rtol=1e-8, atol=1e-12):
                binning = 'uniform'
            else:
                binning = 'variable'
        except Exception:
            pass

    net_cls = type(src_lc)
    net = net_cls(
        time=src_lc.time, value=out_value, error=out_err,
        dt=src_lc.dt, exposure=src_lc.exposure, is_rate=src_lc.is_rate,
        bin_exposure=src_lc.bin_exposure,
        timezero=src_lc.timezero, timezero_obj=src_lc.timezero_obj,
        bin_lo=bin_lo, bin_hi=bin_hi, bin_width=bin_width, binning=binning,
        counts=out_counts, rate=out_rate,
        counts_err=out_counts_err, rate_err=out_rate_err,
        err_dist=src_lc.err_dist,
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
