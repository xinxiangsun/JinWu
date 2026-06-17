"""Test Rust rebinning vs Python reference implementation.

Verifies numerical identity between `ops.rebin_lightcurve()` and
`rebin_rs.rebin_lightcurve_rs()` across a range of inputs.
"""

import numpy as np
import pytest

from jinwu.core.data import LightcurveData
from jinwu.core.ops import rebin_lightcurve
from jinwu.core.rebin_rs import rebin_lightcurve_rs, _HAS_RUST


def make_lc(
    n: int = 100,
    dt: float = 10.0,
    is_rate: bool = False,
    seed: int = 42,
) -> LightcurveData:
    """Build a synthetic LightcurveData with Poisson-ish counts."""
    rng = np.random.default_rng(seed)
    t_start = 0.0
    times = t_start + dt * (np.arange(n, dtype=float) + 0.5)
    true_rate = 5.0 + 2.0 * np.sin(2 * np.pi * times / 500.0)
    counts = rng.poisson(true_rate * dt).astype(float)

    kwargs = dict(
        path=None,
        header={},
        meta=None,
        headers_dump=None,
        time=times,
        dt=dt,
        exposure=float(n * dt),
        is_rate=is_rate,
    )
    if is_rate:
        kwargs["value"] = counts / dt
        kwargs["error"] = np.sqrt(np.maximum(counts, 0)) / dt
        kwargs["rate"] = kwargs["value"]
        kwargs["rate_err"] = kwargs["error"]
    else:
        kwargs["value"] = counts
        kwargs["error"] = np.sqrt(np.maximum(counts, 0))
        kwargs["counts"] = counts
        kwargs["counts_err"] = kwargs["error"]
    return LightcurveData(**kwargs)


@pytest.mark.skipif(not _HAS_RUST, reason="jinwurs Rust extension not installed")
class TestRebinIdentity:
    """Numerical identity between Python and Rust rebinning."""

    @pytest.mark.parametrize("binsize", [5.0, 20.0, 50.0, 100.0])
    @pytest.mark.parametrize("method", ["sum", "mean"])
    @pytest.mark.parametrize("is_rate", [False, True])
    def test_identity(self, binsize, method, is_rate):
        lc = make_lc(n=200, dt=3.0, is_rate=is_rate)
        py_result = rebin_lightcurve(lc, binsize=binsize, method=method)
        rs_result = rebin_lightcurve_rs(lc, binsize=binsize, method=method)

        np.testing.assert_allclose(
            py_result.time, rs_result.time, rtol=0, atol=1e-12,
            err_msg="time mismatch",
        )
        np.testing.assert_allclose(
            py_result.value, rs_result.value, rtol=0, atol=1e-12,
            err_msg="value mismatch",
        )
        np.testing.assert_allclose(
            py_result.error, rs_result.error, rtol=0, atol=1e-12,
            err_msg="error mismatch",
        )

    def test_identity_with_align_ref(self):
        lc = make_lc(n=100, dt=7.0)
        py_result = rebin_lightcurve(lc, binsize=30.0, align_ref=0.0)
        rs_result = rebin_lightcurve_rs(lc, binsize=30.0, align_ref=0.0)
        np.testing.assert_allclose(py_result.value, rs_result.value, atol=1e-12)

    def test_identity_small_binsize_enforced(self):
        """When binsize < max orig bin, it's silently raised."""
        lc = make_lc(n=50, dt=10.0)
        py_result = rebin_lightcurve(lc, binsize=3.0)
        rs_result = rebin_lightcurve_rs(lc, binsize=3.0)
        np.testing.assert_allclose(py_result.value, rs_result.value, atol=1e-12)

    def test_identity_empty_bin_nan(self):
        lc = make_lc(n=10, dt=1.0)
        # Use huge binsize to get zero-exposure bins at edge
        py_result = rebin_lightcurve(lc, binsize=100.0, empty_bin="nan")
        rs_result = rebin_lightcurve_rs(lc, binsize=100.0, empty_bin="nan")
        np.testing.assert_allclose(
            py_result.value, rs_result.value, atol=1e-12, equal_nan=True
        )

    def test_identity_empty_input(self):
        lc = LightcurveData(
            path=None, header={}, meta=None, headers_dump=None,
            time=np.array([], dtype=float), value=np.array([], float),
            error=None, dt=1.0, exposure=0.0, is_rate=False,
        )
        py_result = rebin_lightcurve(lc, binsize=10.0)
        rs_result = rebin_lightcurve_rs(lc, binsize=10.0)
        assert len(py_result.time) == len(rs_result.time) == 0


@pytest.mark.skipif(not _HAS_RUST, reason="jinwurs Rust extension not installed")
def test_rebin_performance():
    """Benchmark: Rust should be faster than Python for large lightcurves."""
    import time

    lc = make_lc(n=2000, dt=1.0)

    # Python
    t0 = time.perf_counter()
    for _ in range(50):
        _ = rebin_lightcurve(lc, binsize=5.0)
    t_py = (time.perf_counter() - t0) * 1000

    # Rust
    t0 = time.perf_counter()
    for _ in range(50):
        _ = rebin_lightcurve_rs(lc, binsize=5.0)
    t_rs = (time.perf_counter() - t0) * 1000

    print(f"\n  Python: {t_py:.1f}ms  |  Rust: {t_rs:.1f}ms  |  {t_py / t_rs:.1f}x")
    # not a hard assertion — just informational
