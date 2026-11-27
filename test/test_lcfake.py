"""Regression tests for shared NPZ light-curve simulation preparation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jinwu.lf import lcfake


ROOT = Path(__file__).resolve().parents[1]
REAL_NPZ_PATHS = (
    "test/EP260809a_wqypipe/lc_data_new/EP260809a_lc_auto_obs.npz",
    "reviews/evidence/release-0.2.2-20261008/wxt_template.npz",
)


def _config(model: str, exposure: float = 100.0) -> lcfake.XspecConfig:
    return lcfake.XspecConfig(
        arf=f"{model}.arf",
        rmf=f"{model}.rmf",
        background=None if model == "ref" else "target.bkg",
        model=model,
        params=(2.0, 1e-4),
        band=(0.5, 4.0),
        exposure=exposure,
        back_exposure=2.0 * exposure,
        xspec_abund="wilm",
        xspec_xsect="vern",
        xspec_cosmo="cosmo",
        allow_prompting=False,
    )


def test_k_config_kwargs_preserve_every_xspec_field():
    cfg = _config("ref")

    kwargs = lcfake._k_config_kwargs(cfg)

    assert kwargs == {
        "arf": "ref.arf",
        "rmf": "ref.rmf",
        "background": None,
        "model": "ref",
        "params": (2.0, 1e-4),
        "band": (0.5, 4.0),
        "exposure": 100.0,
        "back_exposure": 200.0,
        "xspec_abund": "wilm",
        "xspec_xsect": "vern",
        "xspec_cosmo": "cosmo",
        "allow_prompting": False,
    }


def test_flux_on_target_grid_matches_original_redshift_and_rebin_math(tmp_path, monkeypatch):
    npz = tmp_path / "template.npz"
    np.savez(
        npz,
        time=np.arange(4.0),
        corrected_counts_src=np.array([2.0, 4.0, 8.0, 16.0]),
    )
    calls = []

    def fake_get_K(**kwargs):
        calls.append(kwargs)
        return 2.0 if kwargs["model"] == "ref" else 4.0

    monkeypatch.setattr(lcfake._K_FACTORY, "get_K", fake_get_K)
    cfg_ref = _config("ref")
    cfg_tgt = _config("tgt")

    time, dt, flux, K_ref, K_tgt, S_t, dt_in = lcfake._flux_on_target_grid(
        str(npz),
        cfg_ref,
        cfg_tgt,
        area_ratio=None,
        input_background_rate=None,
        z0=0.0,
        z=1.0,
        T0=None,
        target_dt=0.75,
    )

    np.testing.assert_allclose(time, np.arange(8) * 0.75, rtol=0, atol=0)
    np.testing.assert_allclose(flux, [1.0, 1.375, 1.75, 2.25, 3.0, 3.75, 5.0, 6.5], rtol=0, atol=0)
    assert (dt, K_ref, K_tgt, S_t, dt_in) == (0.75, 2.0, 4.0, 2.0, 1.0)
    assert [call["model"] for call in calls] == ["ref", "tgt"]
    assert calls[0] == lcfake._k_config_kwargs(cfg_ref)
    assert calls[1] == lcfake._k_config_kwargs(cfg_tgt)


def test_flux_on_target_grid_keeps_no_redshift_and_default_target_config(tmp_path, monkeypatch):
    npz = tmp_path / "template.npz"
    np.savez(npz, time=np.arange(4.0), corrected_counts_src=np.array([2.0, 4.0, 8.0, 16.0]))
    calls = []

    def fake_get_K(**kwargs):
        calls.append(kwargs)
        return 2.0

    monkeypatch.setattr(lcfake._K_FACTORY, "get_K", fake_get_K)
    cfg = _config("ref")

    time, dt, flux, K_ref, K_tgt, S_t, dt_in = lcfake._flux_on_target_grid(
        str(npz),
        cfg,
        None,
        area_ratio=None,
        input_background_rate=None,
        z0=None,
        z=None,
        T0=123.0,
        target_dt=0.25,
    )

    np.testing.assert_array_equal(time, np.arange(4.0))
    np.testing.assert_allclose(flux, [1.0, 2.0, 4.0, 8.0], rtol=0, atol=0)
    assert (dt, K_ref, K_tgt, S_t, dt_in) == (1.0, 2.0, 2.0, 1.0, 1.0)
    assert len(calls) == 2
    assert calls[0] == calls[1] == lcfake._k_config_kwargs(cfg)


def test_generate_redshift_lightcurves_reuses_k_config_kwargs(monkeypatch):
    calls = []

    def fake_get_K_with_values(**kwargs):
        calls.append(kwargs)
        return 2.0, 4.0, 2.0

    class Posterior:
        area_ratio = 0.5

        def sample_off(self, *, t, size, rng):
            return np.zeros(size)

    monkeypatch.setattr(lcfake._K_FACTORY, "get_K_with_values", fake_get_K_with_values)
    cfg = _config("tgt")

    curves = lcfake.generate_redshift_lightcurves(
        time_norm=np.array([0.0, 1.0]),
        counts_norm=np.array([1.0, 2.0]),
        z0=0.0,
        t100=2.0,
        dt=1.0,
        mean_rate_z0=0.0,
        K_z0=0.0,
        z=1.0,
        background_posterior=Posterior(),
        cfg=cfg,
        n_curves=2,
        rng=np.random.default_rng(7),
    )

    assert calls == [lcfake._k_config_kwargs(cfg)]
    assert len(curves) == 2
    for time, counts_on, counts_off in curves:
        np.testing.assert_array_equal(time, [0.0, 4.0])
        np.testing.assert_array_equal(counts_off, [0, 0])
        assert counts_on.shape == (2,)


@pytest.mark.real_data
@pytest.mark.parametrize("relative_path", REAL_NPZ_PATHS)
def test_fake_builders_share_target_grid_on_real_npz(relative_path, monkeypatch):
    path = ROOT / relative_path
    if not path.is_file():
        pytest.skip(f"real NPZ fixture unavailable: {relative_path}")
    monkeypatch.setattr(lcfake._K_FACTORY, "get_K", lambda **kwargs: 2.0)
    cfg = _config("ref")

    single = lcfake.build_fake_from_npz(
        str(path), cfg, area_ratio=0.5, add_poisson=False
    )
    pair = lcfake.build_fake_on_off_from_npz(
        str(path), cfg, alpha=0.5, add_poisson=False
    )

    np.testing.assert_array_equal(pair.time, single.time)
    np.testing.assert_array_equal(pair.counts_on, single.counts)
    np.testing.assert_array_equal(pair.counts_off, np.zeros_like(pair.counts_on))
    assert pair.dt == single.dt
    for key in ("K_ref", "K_tgt", "S_t", "dt_in", "dt_out"):
        assert pair.meta[key] == single.meta[key]
