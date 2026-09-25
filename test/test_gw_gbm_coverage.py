"""Offline GBM coverage regressions with analytic region probabilities.

The geometry used by :func:`jinwu.gw.coverage.gbm_coverage_for_skymap` is
replaced by fakes whose Earth-blocking and detector-angle masks reproduce the
GDT formulas (``separation(pos, geocenter) > earth_angular_radius`` and
``detector_angle <= limit``), so the MOC-refined integrals can be compared
against closed-form spherical caps on a uniform all-sky map:

    P(cap with angular radius rho) = (1 - cos rho) / 2.

The pixel-center GDT-mask numbers embedded in the same result provide the
cross-validation baseline at the sky-map resolution.

Run in the ``hea`` environment::

    conda run -n hea python -m pytest test/test_gw_gbm_coverage.py -q -p no:cacheprovider
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import warnings

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time

from jinwu.core.time import Time as JinwuTime
from jinwu.gw.coverage import gbm_coverage_for_skymap
from jinwu.gw.skymap import load_skymap


NADIR_RA_DEG = 0.0
NADIR_DEC_DEG = 0.0
EARTH_RADIUS_DEG = 60.0
DETECTOR_RA_DEG = 180.0
DETECTOR_DEC_DEG = 0.0


def _uniform_flat_map(path: Path, *, nside: int = 2) -> Path:
    """One uniform all-sky map: every pixel carries the same probability."""
    npix = 12 * nside * nside
    probability = np.full(npix, 1.0 / npix)
    columns = fits.ColDefs([fits.Column(name="PROB", format="D", array=probability)])
    table = fits.BinTableHDU.from_columns(columns)
    table.header["NSIDE"] = nside
    table.header["ORDERING"] = "RING"
    fits.HDUList([fits.PrimaryHDU(), table]).writeto(path, overwrite=True)
    return path


def _cap_fraction(radius_deg: float) -> float:
    return (1.0 - np.cos(np.radians(radius_deg))) / 2.0


@dataclass
class _FakeSelection:
    status: str = "observed"
    target_time: Any = None
    reference_time: Any = None
    path: Path | None = None
    period_s: float | None = None
    orbit_count: int = 30
    reason: str | None = None
    downloaded: tuple[Path, ...] = ()
    degraded: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "degraded": self.degraded}


@dataclass
class _FakeGeometry:
    good: bool = True
    saa: bool = False
    earth_angular_radius_deg: float = EARTH_RADIUS_DEG
    detector_pointings: tuple = (
        ("n0", DETECTOR_RA_DEG, DETECTOR_DEC_DEG),
    )
    provenance_calls: list = field(default_factory=list)

    @property
    def nadir(self) -> SkyCoord:
        return SkyCoord(NADIR_RA_DEG * u.deg, NADIR_DEC_DEG * u.deg, frame="icrs")

    def to_dict(self) -> dict[str, Any]:
        self.provenance_calls.append(1)
        return {"fake": True}


class _FakeMaskFrame:
    """Mimics GDT ``SpacecraftFrame`` visibility/angle methods analytically."""

    def at(self, when: Time) -> "_FakeMaskFrame":
        return self

    def location_visible(self, coords: SkyCoord) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            separation = coords.separation(self._nadir)
        return np.asarray(separation.deg > EARTH_RADIUS_DEG, dtype=bool)

    @property
    def _nadir(self) -> SkyCoord:
        return SkyCoord(NADIR_RA_DEG * u.deg, NADIR_DEC_DEG * u.deg, frame="icrs")

    def detector_angle(self, name: str, coords: SkyCoord) -> u.Quantity:
        centers = {"n0": (DETECTOR_RA_DEG, DETECTOR_DEC_DEG)}
        if name not in centers:
            # Detectors without a fake pointing can never pass an angle cut.
            return np.full(np.shape(coords.ra), 180.0) * u.deg
        ra_deg, dec_deg = centers[name]
        center = SkyCoord(ra_deg * u.deg, dec_deg * u.deg, frame="icrs")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return coords.separation(center).to_value(u.deg) * u.deg


class _StateTable:
    """Dict-like access to boolean state columns, mirroring GDT TimeSeries."""

    def __init__(self, time, saa: np.ndarray, good: np.ndarray) -> None:
        self.time = time
        self._columns = {"saa": saa, "good": good}

    def __getitem__(self, key: str) -> np.ndarray:
        return self._columns[key]


class _FakePosHist:
    def __init__(self) -> None:
        met = np.array([0.0, 100.0])
        self._states = _StateTable(
            JinwuTime(met, format="fermi"),
            saa=np.array([False, False]),
            good=np.array([True, True]),
        )

    def get_spacecraft_frame(self) -> _FakeMaskFrame:
        return _FakeMaskFrame()

    def get_spacecraft_states(self) -> SimpleNamespace:
        return self._states


@pytest.fixture()
def offline_gbm(monkeypatch, tmp_path: Path):
    """Patch finder/geometry/poshist so the coverage math runs fully offline."""
    import gdt.missions.fermi.gbm.poshist as gdt_poshist
    import jinwu.fermi.gbm as fermi_gbm

    poshist_file = tmp_path / "glg_poshist_all_240101_v00.fit"
    columns = fits.ColDefs([fits.Column(name="DUMMY", format="D", array=np.zeros(2))])
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(columns)]).writeto(poshist_file)

    reference = JinwuTime("2024-01-01T12:00:00", scale="utc")
    selection = _FakeSelection(reference_time=reference, target_time=reference, path=poshist_file)
    geometry = _FakeGeometry()
    history = _FakePosHist()

    # ``coverage`` imports the finder from the package and the geometry reader
    # from the ``poshist`` submodule, so both patch points are required.
    monkeypatch.setattr("jinwu.fermi.gbm.find_gbm_poshist", lambda *args, **kwargs: selection)
    monkeypatch.setattr("jinwu.fermi.gbm.read_gbm_geometry", lambda *args, **kwargs: geometry)
    monkeypatch.setattr(
        "jinwu.fermi.gbm.poshist.read_gbm_geometry", lambda *args, **kwargs: geometry
    )
    monkeypatch.setattr(gdt_poshist.GbmPosHist, "open", classmethod(lambda cls, path: history))
    return geometry


def test_gbm_refined_probabilities_match_analytic_caps(tmp_path: Path, offline_gbm):
    skymap = load_skymap(_uniform_flat_map(tmp_path / "map.fits"))
    result = gbm_coverage_for_skymap(
        skymap, "2024-01-01T12:00:00", tmp_path / "cache", mode="observed", download=False,
    )
    assert result.status == "observed", result.reasons
    # Visible region = full sky minus the 60-degree Earth cap centered on the
    # nadir -> analytic visible probability is 1 - (1 - cos 60)/2 = 0.75.
    assert result.geometry_probability == pytest.approx(1.0 - _cap_fraction(EARTH_RADIUS_DEG), abs=2e-3)
    # One NaI cone with a 60-degree radius around the anti-nadir direction is
    # fully visible -> its cap fraction is the detector-limited probability.
    assert result.probability == pytest.approx(_cap_fraction(EARTH_RADIUS_DEG), abs=2e-3)
    assert result.state_probability == pytest.approx(result.geometry_probability, abs=1e-9)
    integral = result.metadata["geometry_integral"]
    assert integral is not None and integral["converged"] is True
    assert len(integral["history"]) >= 2
    # Pixel-center mask baseline (GDT formulas at sky-map resolution) agrees
    # with the refined boundary within the pixel granularity.
    masks = result.metadata["mask_probabilities"]
    assert masks["geometry"] == pytest.approx(result.geometry_probability, abs=0.15)


def test_gbm_state_filter_zeroes_probability_on_saa(tmp_path: Path, offline_gbm):
    offline_gbm.saa = True
    skymap = load_skymap(_uniform_flat_map(tmp_path / "map.fits"))
    result = gbm_coverage_for_skymap(
        skymap, "2024-01-01T12:00:00", tmp_path / "cache", mode="observed", download=False,
    )
    assert result.probability == 0.0
    assert result.state_probability == 0.0
    assert "saa" in result.reasons
    # The geometric visibility itself is unaffected by the SAA flag.
    assert result.geometry_probability > 0.5


def test_gbm_unknown_selection_reports_reason(tmp_path: Path, monkeypatch):
    import jinwu.fermi.gbm as fermi_gbm

    selection = _FakeSelection(reference_time=None, path=None, reason="target_poshist_missing")
    monkeypatch.setattr(fermi_gbm, "find_gbm_poshist", lambda *args, **kwargs: selection)
    skymap = load_skymap(_uniform_flat_map(tmp_path / "map.fits"))
    result = gbm_coverage_for_skymap(
        skymap, "2024-01-01T12:00:00", tmp_path / "cache", mode="auto", download=False,
    )
    assert result.status == "unknown"
    assert result.probability is None
    assert result.reasons == ("target_poshist_missing",)
