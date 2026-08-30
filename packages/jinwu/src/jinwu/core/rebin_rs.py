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
    """Rebin a LightcurveData using the Rust-accelerated engine.

    Numerically identical to `ops.rebin_lightcurve()`, but the core
    double-loop aggregation runs in compiled Rust.

    Parameters
    ----------
    lc : LightcurveData
        Input lightcurve.
    binsize : float
        Desired new-bin width in seconds.
    method : {'auto', 'sum', 'mean'}
        Aggregation method.
    align_ref : float or None
        Left-edge alignment reference time.
    empty_bin : {'zero', 'nan'}
        How to fill bins with zero exposure.

    Returns
    -------
    LightcurveData
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
