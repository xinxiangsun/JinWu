"""Offline regressions for the time-keyed POSHIST primitives and 30-orbit selection.

Covers :func:`jinwu.fermi.gbm.estimate_gbm_orbit_period` (RapidGBM circular-orbit
estimator), :func:`jinwu.fermi.gbm.fetch_poshist_for_time`,
:func:`jinwu.fermi.gbm.find_gbm_poshist` (observed / predicted_30_orbit / unknown,
single-day degradation) and :func:`jinwu.fermi.gbm.read_gbm_geometry` (one-instant
geometry snapshot with GDT interpolation semantics).

Run in the ``hea`` environment::

    conda run -n hea python -m pytest test/test_gbm_poshist_predict.py -q -p no:cacheprovider
"""

from __future__ import annotations

from pathlib import Path

import astropy.units as u
import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time

from jinwu.core.time import Time as JinwuTime
from jinwu.fermi.gbm import (
    estimate_gbm_orbit_period,
    fetch_poshist_for_time,
    find_gbm_poshist,
    read_gbm_geometry,
)


ORBIT_RADIUS_M = 6.9e6


def _write_position_history(path: Path, *, radius_m: float = ORBIT_RADIUS_M, n: int = 360) -> Path:
    """Write a minimal circular-orbit POSHIST with only the POS_* columns filled."""
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    columns = fits.ColDefs([
        fits.Column(name="POS_X", format="D", array=radius_m * np.cos(angles)),
        fits.Column(name="POS_Y", format="D", array=radius_m * np.sin(angles)),
        fits.Column(name="POS_Z", format="D", array=np.zeros(n)),
    ])
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(columns)]).writeto(path, overwrite=True)
    return path


def test_estimate_gbm_orbit_period_matches_analytic_circle(tmp_path: Path):
    path = _write_position_history(tmp_path / "poshist.fit")
    estimated = estimate_gbm_orbit_period([path]).to_value(u.s)
    from astropy.constants import GM_earth

    analytic = float(2.0 * np.pi * np.sqrt((ORBIT_RADIUS_M * u.m) ** 3 / GM_earth).to_value(u.s))
    assert estimated == pytest.approx(analytic, rel=1e-6)


def test_estimate_gbm_orbit_period_rejects_missing_positions(tmp_path: Path):
    columns = fits.ColDefs([fits.Column(name="SCLK_UTC", format="D", array=np.arange(4.0))])
    path = tmp_path / "empty.fit"
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(columns)]).writeto(path)
    with pytest.raises(ValueError, match="no usable spacecraft positions"):
        estimate_gbm_orbit_period([path])


def test_fetch_poshist_for_time_reuses_cached_day(tmp_path: Path):
    day = JinwuTime("2024-01-01T12:00:00", scale="utc")
    directory = tmp_path / "daily" / "2024" / "01" / "01" / "current"
    directory.mkdir(parents=True)
    cached = _write_position_history(directory / "glg_poshist_all_240101_v00.fit")
    result = fetch_poshist_for_time(day, tmp_path, download=False)
    assert result == cached


def test_fetch_poshist_for_time_without_cache_and_download(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="download=True"):
        fetch_poshist_for_time("2024-01-01T12:00:00", tmp_path, download=False)


@pytest.fixture()
def offline_finder(monkeypatch):
    """Force :func:`find_gbm_poshist` fully offline with controllable day maps."""

    def install(day_files: dict[int, list[Path]], time_ranges: dict[Path, tuple[Time, Time]]):
        from datetime import date

        def fake_paths_for_day(cache, when):
            target_date = date(2024, 1, 1)
            when_date = when.datetime.date()
            offset = (when_date - target_date).days
            return list(day_files.get(offset, []))

        monkeypatch.setattr(
            "jinwu.fermi.gbm.pipeline._poshist_paths_for_day", fake_paths_for_day
        )
        monkeypatch.setattr(
            "jinwu.fermi.gbm.pipeline._download_poshist_for_day",
            lambda *args, **kwargs: [],
        )

        def fake_time_range(path):
            return time_ranges[path]

        monkeypatch.setattr(
            "jinwu.fermi.gbm.pipeline._poshist_time_range", fake_time_range
        )

    return install


def test_find_gbm_poshist_predicted_single_day_is_degraded(tmp_path: Path, offline_finder, monkeypatch):
    from astropy.constants import GM_earth

    period = float(2.0 * np.pi * np.sqrt((ORBIT_RADIUS_M * u.m) ** 3 / GM_earth).to_value(u.s))
    target = JinwuTime("2024-01-01T12:00:00", scale="utc")
    reference = target - 30 * period * u.s
    file_day2 = _write_position_history(tmp_path / "day2.fit")
    # The reference time lands ~2 UTC days before the target for a ~95 min orbit.
    offline_finder(
        {-2: [file_day2]},
        {file_day2: (reference - 600 * u.s, reference + 600 * u.s)},
    )
    selection = find_gbm_poshist(target, tmp_path, mode="auto", download=False)
    assert selection.status == "predicted_30_orbit"
    assert selection.path == file_day2
    assert selection.period_s == pytest.approx(period, rel=1e-3)
    assert selection.reference_time is not None
    assert abs((selection.reference_time - reference).to_value(u.s)) < 1.0
    assert selection.degraded is True


def test_find_gbm_poshist_two_days_period_not_degraded(tmp_path: Path, offline_finder):
    from astropy.constants import GM_earth

    period = float(2.0 * np.pi * np.sqrt((ORBIT_RADIUS_M * u.m) ** 3 / GM_earth).to_value(u.s))
    target = JinwuTime("2024-01-01T12:00:00", scale="utc")
    reference = target - 30 * period * u.s
    file_day1 = _write_position_history(tmp_path / "day1.fit")
    file_day2 = _write_position_history(tmp_path / "day2.fit", radius_m=ORBIT_RADIUS_M * 1.0005)
    offline_finder(
        {-1: [file_day1], -2: [file_day2]},
        {file_day1: (reference - 600 * u.s, reference + 600 * u.s),
         file_day2: (reference - 600 * u.s, reference + 600 * u.s)},
    )
    selection = find_gbm_poshist(target, tmp_path, mode="auto", download=False)
    assert selection.status == "predicted_30_orbit"
    assert selection.degraded is False


def test_find_gbm_poshist_reference_outside_validity_is_unknown(tmp_path: Path, offline_finder):
    target = JinwuTime("2024-01-01T12:00:00", scale="utc")
    file_day1 = _write_position_history(tmp_path / "day1.fit")
    offline_finder(
        {-1: [file_day1]},
        {file_day1: (target - 86400 * u.s, target - 80000 * u.s)},
    )
    selection = find_gbm_poshist(target, tmp_path, mode="auto", download=False)
    assert selection.status == "unknown"
    assert selection.reason == "reference_time_outside_available_poshist"


def test_find_gbm_poshist_observed_target_day(tmp_path: Path, offline_finder):
    target = JinwuTime("2024-01-01T12:00:00", scale="utc")
    file_day0 = _write_position_history(tmp_path / "day0.fit")
    offline_finder({0: [file_day0]}, {})
    selection = find_gbm_poshist(target, tmp_path, mode="auto", download=False)
    assert selection.status == "observed"
    assert selection.path == file_day0
    assert selection.reference_time == target
    assert selection.degraded is False


class _FakeStates:
    def __init__(self, met: np.ndarray, saa: np.ndarray) -> None:
        self.time = JinwuTime(met, format="fermi")
        self._saa = saa

    def __getitem__(self, key: str):
        if key == "saa":
            return self._saa
        if key == "good":
            return ~self._saa
        raise KeyError(key)


class _FakeFrame:
    """Duck-typed GDT SpacecraftFrame with circular motion and identity attitude."""

    def __init__(self, met: np.ndarray, radius_m: float) -> None:
        from astropy.coordinates import CartesianRepresentation
        from gdt.core.coords.quaternion import Quaternion
        from gdt.core.coords.spacecraft.frame import SpacecraftFrame

        self.obstime = JinwuTime(met, format="fermi")
        angles = np.linspace(0.0, np.pi, met.size)
        position = CartesianRepresentation(
            x=radius_m * np.cos(angles) * u.m,
            y=radius_m * np.sin(angles) * u.m,
            z=np.zeros(met.size) * u.m,
        )
        # Identity rotation keeps the spacecraft axes aligned with ICRS so the
        # zenith direction transforms to the ICRS pole without extra assumptions.
        quaternion = Quaternion(np.tile([0.0, 0.0, 0.0, 1.0], (met.size, 1)))
        self._frame = SpacecraftFrame(
            obstime=self.obstime,
            obsgeoloc=position,
            quaternion=quaternion,
        )

    def get_spacecraft_frame(self):
        return self._frame

    def get_spacecraft_states(self):
        return self._states

    def set_states(self, states) -> None:
        self._states = states


def test_read_gbm_geometry_interpolates_and_flags(tmp_path: Path):
    from astropy.constants import R_earth

    met = np.arange(0.0, 600.0, 30.0)
    poshist = _FakeFrame(met, ORBIT_RADIUS_M)
    poshist.set_states(_FakeStates(met, saa=np.array([False] * 19 + [True])))

    inside = JinwuTime(305.0, format="fermi")
    state = read_gbm_geometry(poshist, inside)
    # Linear interpolation of the position vector cuts inside the circle
    # (chord effect), so the interpolated radius sits slightly below R.
    assert ORBIT_RADIUS_M * 0.99 < state.radius_m < ORBIT_RADIUS_M
    at_sample = read_gbm_geometry(poshist, JinwuTime(300.0, format="fermi"))
    assert at_sample.radius_m == pytest.approx(ORBIT_RADIUS_M, rel=1e-9)
    assert at_sample.altitude_m == pytest.approx(ORBIT_RADIUS_M - 6378137.0, rel=1e-3)
    from astropy.constants import R_earth

    expected_radius_deg = float(
        np.degrees(np.arcsin(R_earth.to_value(u.m) / ORBIT_RADIUS_M))
    )
    assert at_sample.earth_angular_radius_deg == pytest.approx(expected_radius_deg, rel=1e-2)
    assert state.good is True and state.saa is False
    # Earth blocking follows the geocentre direction, independently of the
    # attitude quaternion (which only controls detector pointings).
    expected_geocenter = poshist.get_spacecraft_frame().at(inside).geocenter
    assert state.nadir.separation(expected_geocenter).to_value(u.arcsec) < 1e-5
    assert state.nadir.separation(state.zenith).to_value(u.arcsec) == pytest.approx(180.0 * 3600.0, abs=1e-4)

    near_saa = JinwuTime(565.0, format="fermi")
    state_saa = read_gbm_geometry(poshist, near_saa)
    assert state_saa.saa is True and state_saa.good is False


def test_read_gbm_geometry_rejects_extrapolation(tmp_path: Path):
    met = np.arange(0.0, 600.0, 30.0)
    poshist = _FakeFrame(met, ORBIT_RADIUS_M)
    poshist.set_states(_FakeStates(met, saa=np.zeros(met.size, dtype=bool)))
    with pytest.raises(ValueError, match="outside POSHIST validity"):
        read_gbm_geometry(poshist, JinwuTime(6000.0, format="fermi"))


def test_read_gbm_geometry_rejects_a_large_interpolation_gap():
    met = np.arange(0.0, 600.0, 30.0)
    poshist = _FakeFrame(met, ORBIT_RADIUS_M)
    poshist.set_states(_FakeStates(met, saa=np.zeros(met.size, dtype=bool)))
    with pytest.raises(ValueError, match="nearest POSHIST sample"):
        read_gbm_geometry(poshist, JinwuTime(315.0, format="fermi"), max_interpolation_gap=5 * u.s)
