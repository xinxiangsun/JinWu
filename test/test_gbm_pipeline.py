from __future__ import annotations

from dataclasses import replace

import astropy.units as u
from astropy.coordinates import SkyCoord
import numpy as np
import pytest

from jinwu.core.config import instrument
from jinwu.core.time import Time
from jinwu.core.upperlimit import (
    CountTemplateModel,
    UpperLimitObservation,
    profile_source_amplitude,
)
from jinwu.fermi.gbm import (
    GBMCoverageResult,
    GBMFlareInterval,
    check_gbm_coverage,
    select_gbm_detectors,
)
from jinwu.fermi.gbm.response import build_gbm_response_command
import jinwu.fermi.gbm.pipeline as gbm_pipeline


class _FakeStates:
    def __init__(self, times, good, saa):
        self.time = Time(times, format="fermi")
        self._columns = {"good": np.asarray(good), "saa": np.asarray(saa)}

    def __getitem__(self, name):
        return self._columns[name]


class _FakeFrame:
    def __init__(self, visible):
        self.visible = np.asarray(visible)

    def location_visible(self, coordinate):
        del coordinate
        return self.visible

    def __getitem__(self, index):
        return _FakeFrame(self.visible[index])


class _FakePosHist:
    filename = "fake_poshist.fit"

    def __init__(self, times, good, saa, visible):
        self.states = _FakeStates(times, good, saa)
        self.frame = _FakeFrame(visible)

    def get_spacecraft_states(self):
        return self.states

    def get_spacecraft_frame(self):
        return self.frame


def _interval():
    return GBMFlareInterval(
        source_name="unit-test",
        skycoord=SkyCoord(10.0 * u.deg, 20.0 * u.deg),
        start=Time(100.0, format="fermi"),
        stop=Time(106.0, format="fermi"),
        candidate_index=1,
    )


def test_coverage_full_and_tte_intersection(monkeypatch):
    monkeypatch.setattr(
        gbm_pipeline,
        "_detector_angles",
        lambda frame, coordinate, mask: {"n0": 20.0, "n1": 50.0, "b0": 70.0},
    )
    poshist = _FakePosHist(
        np.arange(100.0, 107.0),
        [True] * 7,
        [False] * 7,
        [True] * 7,
    )
    full = check_gbm_coverage(_interval(), poshist)
    assert full.status == "full"
    assert full.covered_exposure_s == pytest.approx(6.0)
    assert full.segments_met[0][0] == pytest.approx(100.0)
    assert full.segments_met[0][1] == pytest.approx(106.0)

    partial = check_gbm_coverage(_interval(), poshist, tte_gti=((101.0, 105.0),))
    assert partial.status == "partial"
    assert partial.covered_exposure_s == pytest.approx(4.0)
    assert partial.reasons == ("tte_gti_gap",)


def test_coverage_reason_and_detector_selection(monkeypatch):
    monkeypatch.setattr(
        gbm_pipeline,
        "_detector_angles",
        lambda frame, coordinate, mask: {"n0": 40.0, "n1": 61.0, "n2": 10.0, "b0": 89.0, "b1": 91.0},
    )
    poshist = _FakePosHist(
        np.arange(100.0, 107.0),
        [True, True, False, True, True, True, True],
        [False, False, False, False, False, False, False],
        [True, True, True, False, True, True, True],
    )
    coverage = check_gbm_coverage(_interval(), poshist)
    assert coverage.status == "partial"
    assert set(coverage.reasons) == {"earth_occulted", "spacecraft_not_good"}
    selection = select_gbm_detectors(coverage)
    assert selection.nai == ("n2", "n0")
    assert selection.bgo == ("b0",)


def test_missing_poshist_is_not_no_coverage():
    result = check_gbm_coverage(_interval(), None)
    assert result.status == "data_missing"
    assert result.reasons == ("poshist_missing",)


def test_interval_accepts_iso8601_utc_offset_strings():
    interval = GBMFlareInterval.from_utc(
        source_name="offset-time",
        ra_deg=10.0,
        dec_deg=20.0,
        start_utc="2025-03-21T18:40:38.275050+00:00",
        stop_utc="2025-03-21T18:40:48.775050+00:00",
    )
    assert interval.duration_s == pytest.approx(10.5)


def test_safe_response_command_uses_native_detector_names(tmp_path):
    command = build_gbm_response_command(
        ra_deg=12.5,
        dec_deg=-30.0,
        start_met=100.0,
        stop_met=101.0,
        detectors=("n0", "BGO_2"),
        workdir=tmp_path,
    )
    assert command.detector_names == ("n0", "b1")
    assert "-d0" in command.arguments
    assert "-d13" in command.arguments
    assert command.arguments[-1] == str(tmp_path.resolve())


def test_profile_amplitude_reuses_gbm_gaussian_background_contract():
    config = instrument("GBM", detector="N0")
    config.upper_limit = replace(config.upper_limit, result_modes=("observed_upper_bound",))
    observation = UpperLimitObservation(
        name="n0",
        source_counts=np.array([10.0, 12.0]),
        unit_source_counts=np.array([1.0, 1.0]),
        background_model=np.array([4.0, 4.0]),
        background_sigma=np.array([1.0, 1.0]),
    )
    model = CountTemplateModel(name="fixed-powerlaw", flux_per_amplitude=1.0)
    result = profile_source_amplitude(
        observation,
        model=model,
        instrument_config=config,
        interval=(0.0, 1.0),
        energy_band=(8.0, 900.0),
    )
    assert result.amplitude_mle > 0.0
    assert result.significance_sigma > 0.0
    assert result.instrument == "GBM_N0"
