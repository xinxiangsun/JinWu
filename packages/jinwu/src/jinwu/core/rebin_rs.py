"""Rust-accelerated rebinning for LightcurveData.

This module provides a fast Rust implementation of the core rebinning
algorithm used by `ops.rebin_lightcurve()`.  The Python implementation
remains the default; use `rebin_lightcurve_rs()` as an accelerated
alternative with identical numerical results.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional

import numpy as np

try:
    from jinwurs import rebin_counts_core, rebin_finalize
    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False

if TYPE_CHECKING:
    from jinwu.core.data import LightcurveData

__all__ = ["rebin_lightcurve_rs", "_HAS_RUST"]


def rebin_lightcurve_rs(
    lc: "LightcurveData",
    binsize: float,
    method: Literal["auto", "sum", "mean"] = "auto",
    *,
    align_ref: Optional[float] = None,
    empty_bin: Literal["zero", "nan"] = "zero",
) -> "LightcurveData":
    """用 Rust 重分箱一维光变 / Rebin a one-dimensional lightcurve with Rust.

    复用 Python 层的 bin 几何与曝光解析，把重叠累加交给 jinwurs。
    数值等价性取决于两种后端的具体版本与输入，应以对应回归为准。
    Reuse Python geometry/exposure helpers and delegate overlap accumulation
    to jinwurs. Numerical equivalence depends on backend versions and inputs;
    establish it with the corresponding regression comparisons.

    Parameters
    ----------
    lc : LightcurveData
        一维光变，counts 或 counts/s 由 is_rate 决定；多能段须先切片。
        One-dimensional counts or counts/s according to is_rate; slice bands
        first. Input time reference is preserved, not transformed.
    binsize : float
        目标正 bin 宽，单位与时间相同，通常秒；小于最大原 bin 宽时
        自动提升到该宽度，本入口不会细分到更短的目标宽度。
        Desired positive width in the time unit, usually seconds; raised to
        the maximum original width if smaller. This entry does not refine bins.
    method : {'auto', 'sum', 'mean'}
        auto 对 rate 用 mean，对 counts 用 sum；sum 输出计数，mean
        输出计数除以曝光，非直接对输入数值求算术平均。
        Auto uses mean for rates and sum for counts. Sum returns counts;
        mean divides accumulated counts by exposure, not an arithmetic mean.
    align_ref : float or None
        新网格左边缘参考，时间单位/零点同 lc；None 使用最早原左边缘。
        Left-edge reference in the same unit/zero point; None uses the first
        original left edge. A supplied reference controls the covered grid.
    empty_bin : {'zero', 'nan'}
        零曝光分箱的值/误差填充策略，传递给 Rust finalize。
        Value/error filling for zero-exposure bins, passed to Rust finalize.

    Returns
    -------
    LightcurveData
        新的均匀网格对象，保留来源元数据，设置 bin 边界及计数或率字段。
        显式 bin_exposure 存在时按重叠传播，否则 mean 用目标 bin 宽归一化。
        New uniform-grid object with source metadata, bin edges and counts/rate
        fields. Explicit bin exposure is propagated; otherwise mean uses target
        bin width as its denominator. Source arrays are not modified.

    Raises
    ------
    ImportError, NotImplementedError
        缺少 jinwurs，或输入为多能段数组；不会自动回退到 Python 后端。
        Missing jinwurs or multi-band input; no automatic Python fallback.

    Notes
    -----
    旧计数按 overlap/old_width 分配，方差按独立输入的线性传播累加。
    缺失误差时使用 sqrt(max(counts, 0))。同一旧 bin 的分数拆分可使
    输出 bin 相关；本结果未返回协方差矩阵，也未重新生成 Poisson 事件。
    Allocate old counts by overlap/old_width and add linearly propagated
    independent-input variances. Missing errors use sqrt(max(counts, 0)).
    Fractional splitting can correlate outputs; no covariance matrix or newly
    sampled Poisson events is produced.
    """
    if not _HAS_RUST:
        raise ImportError(
            "jinwurs Rust extension not installed. "
            "Install with: pip install jinwu-core  or  maturin develop"
        )

    from jinwu.core.data import LightcurveData
    from jinwu.core.ops import _infer_bin_geometry, _effective_exposure_from_lc

    if method == "auto":
        method = "mean" if lc.is_rate else "sum"

    if lc.value.ndim > 1:
        raise NotImplementedError(
            "Rebin for multi-band LC not yet supported; slice bands first."
        )

    t = np.asarray(lc.time, dtype=float)
    if t.size == 0:
        return LightcurveData(
            path=lc.path, time=np.array([], dtype=float),
            value=np.array([], dtype=float), error=None,
            dt=binsize, exposure=lc.exposure, bin_exposure=None,
            is_rate=lc.is_rate, header=lc.header, meta=lc.meta,
            headers_dump=lc.headers_dump, region=lc.region,
            bin_width=np.array([], dtype=float), binning="unknown",
        )

    # --- identical geometry logic as Python version ---
    orig_left, orig_right, orig_width = _infer_bin_geometry(lc)

    max_bin = float(np.max(orig_width)) if orig_width.size else float(binsize)
    if binsize < max_bin:
        binsize = max_bin

    if align_ref is not None:
        ref = float(align_ref)
    else:
        ref = float(orig_left.min())

    tmax = float(orig_right.max())
    nbins = max(1, int(np.ceil((tmax - ref) / binsize)))
    edges = ref + np.arange(nbins + 1, dtype=float) * binsize
    centers = 0.5 * (edges[:-1] + edges[1:])

    vals = np.asarray(lc.value, dtype=float)
    errs = np.asarray(lc.error, dtype=float) if lc.error is not None else None
    orig_eff_expo = _effective_exposure_from_lc(lc, orig_width)

    if lc.is_rate:
        orig_counts = vals * orig_eff_expo
        orig_err_counts = errs * orig_eff_expo if errs is not None else None
    else:
        orig_counts = vals.copy()
        orig_err_counts = errs.copy() if errs is not None else None

    if orig_err_counts is None:
        orig_err_counts = np.sqrt(np.maximum(orig_counts, 0.0))
    # ensure contiguous C-order arrays for Rust FFI
    orig_counts = np.ascontiguousarray(orig_counts, dtype=np.float64)
    orig_err_counts = np.ascontiguousarray(orig_err_counts, dtype=np.float64)
    orig_left = np.ascontiguousarray(orig_left, dtype=np.float64)
    orig_right = np.ascontiguousarray(orig_right, dtype=np.float64)
    orig_width = np.ascontiguousarray(orig_width, dtype=np.float64)
    edges = np.ascontiguousarray(edges, dtype=np.float64)

    # --- Rust core ---
    orig_bin_expos = getattr(lc, "bin_exposure", None)
    if orig_bin_expos is not None:
        orig_bin_expos = np.ascontiguousarray(
            np.asarray(orig_bin_expos, dtype=np.float64)
        )
    orig_eff_expo_contig = np.ascontiguousarray(orig_eff_expo, dtype=np.float64)

    # 方法：counts 守恒式重分组——旧 bin 按时间重叠比例 frac=overlap/width 分配到新 bin（counts_j=Σ c_i·frac_ij），误差按独立项线性传播 var_j=Σ(ε_i·frac_ij)²（无误差数组时取 Poisson ε=√c；rate 输入先乘有效曝光还原 counts）；sum 法直接输出计数、mean/rate 法除以新 bin 有效曝光。注意同一旧 bin 跨越多个新 bin 时输出 bin 误差相关，属分数拆分的固有近似
    # 参考：Gaussian 线性误差传播标准式（Bevington & Robinson 2003, Data Reduction and Error Analysis for the Physical Sciences, 3rd ed., McGraw-Hill）；与 ops.rebin_lightcurve 数值逐点一致（模块 docstring 声明）
    new_counts, new_var, new_exposure = rebin_counts_core(
        orig_counts,
        orig_err_counts,
        orig_left,
        orig_right,
        orig_width,
        orig_eff_expo_contig if orig_bin_expos is not None else None,
        edges,
    )
    # Rust returns PyArray1 — get as numpy
    new_counts = np.asarray(new_counts)
    new_var = np.asarray(new_var)
    new_exposure = np.asarray(new_exposure)

    # --- finalize ---
    out_is_rate = method != "sum"
    empty_nan = empty_bin == "nan"

    if orig_bin_expos is not None:
        denom = new_exposure
    else:
        denom = np.full_like(new_counts, binsize, dtype=np.float64)

    out_value, out_err = rebin_finalize(
        new_counts, new_var, denom, method, empty_nan
    )
    out_value = np.asarray(out_value)
    out_err = np.asarray(out_err)

    ret_bin_exposure = new_exposure if orig_bin_expos is not None else None

    return LightcurveData(
        path=lc.path,
        time=centers,
        value=out_value,
        error=out_err,
        dt=binsize,
        timezero=getattr(lc, "timezero", -1),
        timezero_obj=getattr(lc, "timezero_obj", None),
        bin_lo=edges[:-1],
        bin_hi=edges[1:],
        tstart=getattr(lc, "tstart", None),
        tseg=getattr(lc, "tseg", None),
        bin_width=np.diff(edges),
        binning="uniform",
        exposure=float(np.sum(ret_bin_exposure)) if ret_bin_exposure is not None else lc.exposure,
        bin_exposure=ret_bin_exposure,
        is_rate=out_is_rate,
        err_dist=getattr(lc, "err_dist", None),
        counts=None if out_is_rate else out_value,
        rate=out_value if out_is_rate else None,
        counts_err=None if out_is_rate else out_err,
        rate_err=out_err if out_is_rate else None,
        gti_start=getattr(lc, "gti_start", None),
        gti_stop=getattr(lc, "gti_stop", None),
        quality=getattr(lc, "quality", None),
        fracexp=getattr(lc, "fracexp", None),
        backscal=getattr(lc, "backscal", None),
        areascal=getattr(lc, "areascal", None),
        telescop=getattr(lc, "telescop", None),
        timesys=getattr(lc, "timesys", None),
        mjdref=getattr(lc, "mjdref", None),
        header=lc.header,
        meta=lc.meta,
        headers_dump=lc.headers_dump,
        region=lc.region,
        columns=getattr(lc, "columns", ()),
        ratio=getattr(lc, "ratio", None),
    )
