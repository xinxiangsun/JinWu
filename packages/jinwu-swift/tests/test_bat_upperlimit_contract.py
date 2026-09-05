from __future__ import annotations

from dataclasses import replace

import astropy.units as u
import numpy as np
import pytest

from jinwu.core.config import BATSurvey
from jinwu.core.upperlimit import (
    CountTemplateModel,
    OneSidedLevel,
    UpperLimitObservation,
    estimate_upper_limit,
)
from jinwu.swift.bat.survey import (
    BATSurveySensitivityAdapter,
    SurveyRatePoint,
    estimate_bat_survey_sensitivity,
)


def _control(index: int, snr: float) -> SurveyRatePoint:
    noise = np.full(8, 1.0)
    rates = np.full(8, snr / np.sqrt(8.0))
    return SurveyRatePoint(
        obsid="control",
        pointing_id=f"p{index}",
        source=None,
        time_start=float(index),
        time_stop=float(index + 1),
        exposure_s=1.0,
        rate=float(np.sum(rates)),
        rate_error=float(np.sqrt(8.0)),
        snr=float(snr),
        pcode=0.5,
        bad_bin=False,
        quality="SUCCESS",
        rate_unit="count/s",
        error_source="RATE_ERR",
        source_file="controls.csv",
        channel_rate=tuple(rates),
        channel_error=tuple(noise),
        channel_snr=tuple(rates),
        background_variance=tuple(noise),
        native_channel_count=8,
        total_band_is_last=False,
    )


def _well_sampled_controls() -> tuple[SurveyRatePoint, ...]:
    """Create a null sample large enough to resolve a 3-sigma tail."""
    rng = np.random.default_rng(12345)
    statistics = rng.normal(size=20_000)
    return tuple(_control(index, float(value)) for index, value in enumerate(statistics))


def test_bat_defaults_request_separate_observed_and_sensitivity_products():
    config = BATSurvey()
    assert config.upper_limit.background_likelihood == "gaussian_net_rate"
    assert config.upper_limit.spectral_index == pytest.approx(2.0)
    assert config.upper_limit.result_modes == (
        "observed_upper_bound",
        "detection_sensitivity",
    )
    assert config.upper_limit.detection_false_alarm_probability == pytest.approx(
        0.0013498980316301
    )


def test_bat_survey_config_is_compatible_with_shared_profile_engine():
    result = estimate_upper_limit(
        UpperLimitObservation(
            name="bat-survey",
            source_counts=np.asarray([8.0, 9.0]),
            background_model=np.asarray([7.0, 7.0]),
            background_sigma=np.asarray([1.0, 1.0]),
            unit_source_counts=np.ones(2),
        ),
        model=CountTemplateModel(name="unit", flux_per_amplitude=1.0),
        instrument_config=BATSurvey(),
        interval=(0.0, 1.0),
        energy_band=(14.0, 195.0),
        plots=False,
    )
    assert result.observed_upper_bound is not None
    assert result.detection_sensitivity is not None
    assert result.detection_sensitivity.status == "unavailable"


def test_bat_empirical_sensitivity_requires_a_populated_null_tail():
    controls = _well_sampled_controls()
    result = estimate_bat_survey_sensitivity(
        controls,
        unit_source_rate=np.ones(8) * u.ct / u.s,
        target_power=0.90,
    )
    assert result.status == "ready"
    assert result.calibration_status == "empirical_fixed_position"
    assert result.null_trials == 20_000
    assert result.false_alarm_probability == pytest.approx(0.0013498980316301)
    assert result.amplitude is not None and result.amplitude > 0


def test_bat_sensitivity_does_not_accept_invalid_or_short_controls():
    result = estimate_bat_survey_sensitivity(
        (_control(0, 0.0),),
        unit_source_rate=np.ones(8) * u.ct / u.s,
    )
    assert result.status == "unavailable"
    assert result.calibration_status == "unavailable"
    assert "exceedances" in (result.reason or "")


def test_bat_sensitivity_uses_explicit_all_band_total_once():
    """The native TOTSNR numerator is the product's total-band value."""
    rng = np.random.default_rng(2468)
    statistics = rng.normal(size=5_000)
    controls = []
    for index, value in enumerate(statistics):
        point = _control(index, float(value))
        explicit_total = float(point.rate) + 2.0
        controls.append(
            replace(
                point,
                rate=explicit_total,
                channel_rate=tuple(point.channel_rate or ()) + (explicit_total,),
                total_band_is_last=True,
            )
        )
    result = estimate_bat_survey_sensitivity(
        tuple(controls),
        unit_source_rate=np.ones(8) * u.ct / u.s,
        false_alarm_probability=0.01,
        min_null_exceedances=20,
    )
    expected = float(
        np.quantile(
            statistics + 2.0 / np.sqrt(8.0),
            0.99,
            method="higher",
        )
    )
    assert result.status == "ready"
    assert result.threshold == pytest.approx(expected)


def test_bat_sensitivity_rejects_a_contaminated_empirical_tail():
    controls = tuple(_control(i, 5.0 if i < 20 else 0.0) for i in range(100))
    result = estimate_bat_survey_sensitivity(
        controls,
        unit_source_rate=np.ones(8) * u.ct / u.s,
    )
    assert result.status == "unavailable"
    assert result.calibration_status == "needs_review"
    assert "outside the 95% binomial interval" in (result.reason or "")


def test_bat_adapter_exposes_auditable_calibration_protocol():
    controls = _well_sampled_controls()
    adapter = BATSurveySensitivityAdapter(
        controls=controls,
        unit_source_rate=np.ones(8) * u.ct / u.s,
    )
    result = adapter.estimate_sensitivity(
        (),
        level=OneSidedLevel.from_sigma(3.0),
        policy=BATSurvey().upper_limit,
    )
    assert result.status == "ready"
    metadata = adapter.calibrate(
        (),
        level=OneSidedLevel.from_sigma(3.0),
        target_power=0.90,
        seed=7,
    )
    assert metadata["construction"] == "bat_native_totsnr_null_and_injection"
    assert metadata["seed"] == 7
