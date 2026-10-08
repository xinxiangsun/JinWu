"""Unit boundary around the frozen Haar computation; no new MVT estimator."""
from __future__ import annotations

from importlib.resources import files
import warnings
import astropy.units as u
import numpy as np
from scipy.interpolate import interp1d

from .models import GBMMVTConfig, METHOD_VERSION, MVTResult, time_values


def compute_mvt(counts, count_errors=None, bin_width=None, *, config=None,
                diagnostics=True) -> MVTResult:
    """Run the frozen GBM_MVT_paper Haar algorithm on a uniform light curve.

    Parameters
    ----------
    counts, count_errors : array-like or Quantity
        Per-bin counts and their one-sigma errors, in counts. Explicit errors
        are required for signed or non-Poisson light curves. With no errors,
        use exactly the paper wrapper's sqrt(counts) for nonnegative counts.
    bin_width : Quantity
        Positive uniform time bin width. Gaps must be handled before this call.
    config : GBMMVTConfig, optional
        Frozen algorithm settings. No environment or session state is read.
    diagnostics : bool
        Retain scaleogram and denoising arrays for inspection and plotting.

    Returns
    -------
    MVTResult
        MVT and original analytic error in seconds, the original seven-value
        return, explicit measurement/limit/failure state, and optional arrays.
        A zero-error upstream limit remains a limit, never a detection.

    References
    ----------
    Golkhou & Butler (2014), https://arxiv.org/abs/1403.4254;
    Bala et al. (2026), https://arxiv.org/abs/2512.16204.
    """
    from ._vendor.haar_power_mod import haar_power_mod
    settings = config or GBMMVTConfig()
    dt = time_values(bin_width, "bin_width")
    if dt.shape != () or dt <= 0:
        raise ValueError("bin_width must be a positive scalar time")
    values = np.array(counts.to_value(u.ct) if isinstance(counts, u.Quantity) else counts,
                      dtype=float, copy=True)
    if values.ndim != 1 or values.size < 4 or not np.isfinite(values).all():
        raise ValueError("counts must be a finite one-dimensional array with at least four bins")
    if count_errors is None:
        if np.any(values < 0):
            raise ValueError("signed counts require explicit count_errors")
        errors = np.sqrt(values)
    else:
        errors = np.array(count_errors.to_value(u.ct) if isinstance(count_errors, u.Quantity)
                          else count_errors, dtype=float, copy=True)
        if errors.shape != values.shape or not np.isfinite(errors).all() or np.any(errors < 0):
            raise ValueError("count_errors must be finite, nonnegative and match counts")
    detail = {}
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", RuntimeWarning)
        try:
            raw = tuple(float(x) for x in haar_power_mod(
                values, errors, min_dt=float(dt), doplot=False, zerocheck=False,
                verbose=False, _diagnostics=detail, _capture_arrays=diagnostics, **settings.haar_kwargs()))
        except (ValueError, IndexError, FloatingPointError, ZeroDivisionError,
                np.linalg.LinAlgError) as exc:
            return MVTResult("failed", None, None, None, {},
                             tuple(dict.fromkeys(str(w.message) for w in captured)),
                             f"{type(exc).__name__}: {exc}")
    messages = tuple(dict.fromkeys(str(w.message) for w in captured))
    status = "measurement" if detail.get("otype") == "measurement" else "upper_limit"
    if not np.isfinite(raw[2]) or raw[2] <= 0:
        status = "unavailable"
    if status == "measurement" and (not np.isfinite(raw[3]) or raw[3] <= 0):
        status = "unavailable"
    if detail.get("optimizer_success") is False:
        messages += ("upstream optimizer reported failure; original return retained",)
    valid = status in {"measurement", "upper_limit"}
    return MVTResult(status, raw[2] * u.s if valid else None,
                     raw[3] * u.s if status == "measurement" else None, raw,
                     detail if diagnostics else {}, messages,
                     None if valid else "upstream produced no usable positive estimate")


def load_validation_curve(model_path=None) -> dict:
    """Load the pinned paper curve, without pickle or extrapolation policy.

    The four log10 arrays retain the upstream normalization (MVT in ms,
    dimensionless SNR). This helper performs format validation only.
    """
    path = model_path if model_path is not None else files(__package__ + "._vendor").joinpath("mvt_snr_fit_model.npz")
    with path.open("rb") if hasattr(path, "open") else open(path, "rb") as handle:
        with np.load(handle, allow_pickle=False) as source:
            model = {k: source[k].copy() for k in
                     ("mvt_grid_log", "snr_median_log", "snr_lower_log", "snr_upper_log")}
    grid = model["mvt_grid_log"]
    if grid.ndim != 1 or grid.size < 2 or np.any(np.diff(grid) <= 0):
        raise ValueError("invalid validation-curve grid")
    if any(a.shape != grid.shape or not np.isfinite(a).all() for a in model.values()):
        raise ValueError("invalid validation-curve arrays")
    if np.any(model["snr_lower_log"] > model["snr_median_log"]) or np.any(model["snr_median_log"] > model["snr_upper_log"]):
        raise ValueError("validation confidence band is not ordered")
    return model


def classify_mvt(mvt: u.Quantity, snr_mvt: float, *, model_path=None) -> dict:
    """Apply the paper's unchanged in-domain log-space classification.

    ``snr_mvt`` must use the paper's peak-total-counts/sqrt(background-counts)
    definition at the measured MVT bin width. It is not a detection sigma.
    Out-of-domain inputs are explicitly uncalibrated, rather than silently
    extrapolated as in the upstream UI helper.
    """
    value = np.asarray(mvt.to_value(u.ms), dtype=float)
    snr = np.asarray(snr_mvt, dtype=float)
    if value.shape != () or snr.shape != () or not np.isfinite(value) or value <= 0 or not np.isfinite(snr) or snr <= 0:
        raise ValueError("mvt and snr_mvt must be positive finite scalars")
    model = load_validation_curve(model_path)
    grid = model["mvt_grid_log"]
    x = np.log10(value)
    result = {"method": METHOD_VERSION, "mvt_ms": float(value), "snr_mvt": float(snr),
              "calibration_range_ms": (10 ** grid[[0, -1]]).tolist(),
              "calibration_scope": "empirical paper curve; not instrument-wide significance"}
    if x < grid[0] or x > grid[-1]:
        return {**result, "classification": "uncalibrated", "reason": "outside model domain"}
    bounds = [float(interp1d(grid, model[k], kind="linear", bounds_error=True)(x))
              for k in ("snr_lower_log", "snr_median_log", "snr_upper_log")]
    label = "upper_limit" if np.log10(snr) < bounds[0] else "robust_measurement" if np.log10(snr) > bounds[2] else "likely_upper_limit"
    return {**result, "classification": label, "snr_bounds": (10 ** np.array(bounds)).tolist()}


def summarize_resamples(results: list[MVTResult]) -> dict:
    """Preserve paper's measured-sample percentiles and report every other state.

    Upstream selected mvt_err_ms > 0 and required at least two measurements.
    These conditional percentiles retain that procedure; censored bounds are
    kept separately and never mixed into the measurement distribution.
    """
    counts = {s: sum(r.estimator_status == s for r in results) for s in
              ("measurement", "upper_limit", "unavailable", "failed")}
    samples = np.array([round(float(r.mvt.to_value(u.ms)), 3) / 1000 for r in results
                        if r.estimator_status == "measurement" and round(float(r.error.to_value(u.ms)), 3) > 0])
    limits = [float(r.mvt.to_value(u.s)) for r in results if r.estimator_status == "upper_limit"]
    quantiles = np.percentile(samples, [16, 50, 84]).tolist() if len(samples) >= 2 else None
    return {"n_resamples": len(results), "state_counts": counts,
            "measurement_fraction": counts["measurement"] / len(results) if results else 0.,
            "n_summary_measurements": len(samples),
            "upstream_rounding": "MVT and analytic error rounded to 0.001 ms before error>0 selection and percentiles",
            "percentiles_s": quantiles, "percentile_levels": [16, 50, 84],
            "percentile_condition": "conditional on valid upstream measurements (not censored limits)",
            "algorithm_upper_limits_s": limits,
            "uncertainty_method": "Poisson resampling of observed per-bin counts; fixed settings"}
