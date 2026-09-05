from __future__ import annotations

from dataclasses import asdict, replace
from statistics import NormalDist
from types import SimpleNamespace

from astropy.io import fits
import numpy as np
import pytest

from jinwu.core.config import instrument
from jinwu.core.upperlimit import (
    DEFAULT_CHAIN_LEVELS,
    DEFAULT_ONE_SIDED_CHAIN_LEVELS,
    CountTemplateModel,
    DetectionSensitivity,
    UpperLimit,
    UpperLimitObservation,
    estimate_upper_limit,
)
import jinwu.core.upperlimit as upperlimit


def _write_chain(path):
    columns = [
        fits.Column(name="PhoIndex__1", format="D", array=np.arange(1.0, 101.0)),
        fits.Column(name="FIT_STATISTIC", format="D", array=np.arange(100.0)),
    ]
    fits.HDUList(
        [fits.PrimaryHDU(), fits.BinTableHDU.from_columns(columns, name="CHAIN")]
    ).writeto(path)
    return path


def _observed_only_config(name: str):
    cfg = instrument(name)
    cfg.upper_limit = replace(
        cfg.upper_limit,
        result_modes=("observed_upper_bound",),
    )
    return cfg


def test_legacy_and_explicit_one_sided_chain_levels_are_distinct(tmp_path):
    assert DEFAULT_CHAIN_LEVELS == {
        "1sigma": pytest.approx(0.6826894921370859),
        "90%": pytest.approx(0.9),
        "2sigma": pytest.approx(0.9544997361036416),
        "3sigma": pytest.approx(0.9973002039367398),
    }
    assert DEFAULT_ONE_SIDED_CHAIN_LEVELS["1sigma"] == pytest.approx(
        NormalDist().cdf(1.0)
    )
    assert DEFAULT_ONE_SIDED_CHAIN_LEVELS["2sigma"] == pytest.approx(
        NormalDist().cdf(2.0)
    )
    assert DEFAULT_ONE_SIDED_CHAIN_LEVELS["3sigma"] == pytest.approx(
        NormalDist().cdf(3.0)
    )

    path = _write_chain(tmp_path / "chain.fits")
    legacy = UpperLimit("PhoIndex").from_chain(path)
    explicit = UpperLimit("PhoIndex").from_chain(
        path,
        levels=DEFAULT_ONE_SIDED_CHAIN_LEVELS,
    )

    assert any("0.997300" in warning for warning in legacy.warnings)
    assert explicit.warnings == []
    assert legacy.limits["3sigma"].upper != explicit.limits["3sigma"].upper
    point = asdict(explicit.limits["3sigma"])
    assert point["confidence_convention"] == "posterior_quantile"
    assert point["central_coverage"] is None
    assert point["one_sided_probability"] == pytest.approx(NormalDist().cdf(3.0))


def test_profile_error_probability_metadata():
    assert upperlimit._profile_central_coverage(2.706) == pytest.approx(0.9, abs=5e-5)
    assert upperlimit._profile_one_sided_probability(2.706) == pytest.approx(
        0.95, abs=2e-5
    )
    assert upperlimit._profile_central_coverage(9.0) == pytest.approx(
        0.9973002039367398
    )
    assert upperlimit._profile_one_sided_probability(9.0) == pytest.approx(
        0.9986501019683699
    )


def test_observed_bound_reuses_one_profile_fit(monkeypatch):
    calls = 0
    original = upperlimit._ProfileLikelihood.fit

    def counted(self):
        nonlocal calls
        calls += 1
        return original(self)

    monkeypatch.setattr(upperlimit._ProfileLikelihood, "fit", counted)
    observation = UpperLimitObservation(
        name="CMOS1",
        source_counts=np.array([3.0, 2.0]),
        background_counts=np.array([5.0, 4.0]),
        alpha=0.2,
        unit_source_counts=np.array([1.0, 0.5]),
    )
    result = estimate_upper_limit(
        observation,
        model=CountTemplateModel(name="unit"),
        instrument_config=_observed_only_config("WXT"),
        interval=(0.0, 10.0),
        energy_band=(0.5, 4.0),
        plots=False,
    )

    assert calls == 1
    assert result.observed_upper_bound is not None
    assert np.all(result.likelihood_scan["delta_stat"] >= 0.0)


def test_covariance_conditioning_uses_final_covariance():
    cfg = _observed_only_config("GBM")
    observation = UpperLimitObservation(
        name="NAI1",
        source_counts=np.array([10.0, 12.0]),
        background_model=np.array([8.0, 10.0]),
        background_covariance=np.diag([1.0, 1e9]),
        unit_source_counts=np.ones(2),
    )
    warnings_list: list[str] = []
    prepared = upperlimit._prepare_observation(
        observation,
        model=CountTemplateModel(name="unit"),
        interval=(0.0, 1.0),
        energy_band=(8.0, 1000.0),
        policy=cfg.upper_limit,
        warnings_list=warnings_list,
    )
    assert prepared.background_cholesky is not None
    assert prepared.background_covariance_condition == pytest.approx(1e9)
    assert any("ill-conditioned" in warning for warning in warnings_list)

    singular = replace(
        observation,
        background_covariance=np.diag([1.0, 1.1e12]),
    )
    with pytest.raises(ValueError, match="numerically singular"):
        upperlimit._prepare_observation(
            singular,
            model=CountTemplateModel(name="unit"),
            interval=(0.0, 1.0),
            energy_band=(8.0, 1000.0),
            policy=cfg.upper_limit,
            warnings_list=[],
        )


def test_sensitivity_adapter_receives_raw_covariance_once():
    captured = {}

    class Adapter:
        def estimate_sensitivity(self, observations, **kwargs):
            captured["observation"] = observations[0]
            captured["policy"] = kwargs["policy"]
            return DetectionSensitivity(
                amplitude=1.0,
                amplitude_unit="model normalization",
                flux=None,
                flux_unit=None,
                fluence=None,
                fluence_unit=None,
                threshold=9.0,
                false_alarm_confidence=NormalDist().cdf(3.0),
                target_power=0.9,
                achieved_power=0.9,
                null_trials=10,
                signal_trials=10,
                status="ready",
            )

    cfg = instrument("GBM")
    cfg.upper_limit = replace(
        cfg.upper_limit,
        fractional_background_systematic=0.2,
    )
    raw_covariance = np.diag([4.0, 9.0])
    observation = UpperLimitObservation(
        name="NAI1",
        source_counts=np.array([10.0, 12.0]),
        background_model=np.array([8.0, 10.0]),
        background_covariance=raw_covariance,
        unit_source_counts=np.ones(2),
    )
    estimate_upper_limit(
        observation,
        model=CountTemplateModel(name="unit"),
        instrument_config=cfg,
        interval=(0.0, 1.0),
        energy_band=(8.0, 1000.0),
        sensitivity_adapter=Adapter(),
        plots=False,
    )

    adapter_observation = captured["observation"]
    assert np.array_equal(adapter_observation.background_covariance, raw_covariance)
    assert adapter_observation.metadata["fractional_background_systematic_applied"] is False
    assert captured["policy"].fractional_background_systematic == pytest.approx(0.2)


def test_sensitivity_uses_explicit_false_alarm_probability():
    cfg = instrument("GBM")
    cfg.upper_limit = replace(
        cfg.upper_limit,
        result_modes=("observed_upper_bound", "detection_sensitivity"),
        calibration="asymptotic",
        calibration_mode="conditional_model",
        detection_false_alarm_probability=0.01,
        detection_power=0.5,
        signal_trials=8,
    )
    result = estimate_upper_limit(
        UpperLimitObservation(
            name="NAI1",
            source_counts=np.array([6.0, 5.0]),
            background_model=np.array([4.0, 4.0]),
            background_sigma=np.array([1.0, 1.0]),
            unit_source_counts=np.ones(2),
        ),
        model=CountTemplateModel(name="unit", flux_per_amplitude=1.0),
        instrument_config=cfg,
        interval=(0.0, 1.0),
        energy_band=(8.0, 1000.0),
        plots=False,
    )
    sensitivity = result.detection_sensitivity
    assert sensitivity is not None
    assert sensitivity.false_alarm_probability == pytest.approx(0.01)
    assert sensitivity.false_alarm_confidence == pytest.approx(0.99)
    expected_threshold = NormalDist().inv_cdf(0.99) ** 2
    assert sensitivity.threshold == pytest.approx(expected_threshold)


def test_energy_selection_uses_channel_overlap():
    observation = UpperLimitObservation(name="det", source_counts=np.ones(3))
    selected = upperlimit._select_template_channels(
        np.array([1.0, 2.0, 3.0]),
        e_min=np.array([0.0, 1.0, 2.0]),
        e_max=np.array([1.0, 2.0, 3.0]),
        energy_band=(0.9, 1.1),
        observation=observation,
    )
    assert np.array_equal(selected, np.array([1.0, 2.0]))


def test_fakeit_pha_reader_aligns_ebounds_by_channel(tmp_path):
    spectrum = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="CHANNEL", format="I", array=np.array([2, 1])),
            fits.Column(name="COUNTS", format="D", array=np.array([20.0, 10.0])),
        ],
        name="SPECTRUM",
    )
    spectrum.header["EXPOSURE"] = 1.0
    ebounds = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="CHANNEL", format="I", array=np.array([1, 2])),
            fits.Column(name="E_MIN", format="D", array=np.array([0.0, 1.0])),
            fits.Column(name="E_MAX", format="D", array=np.array([1.0, 2.0])),
        ],
        name="EBOUNDS",
    )
    path = tmp_path / "fake.pha"
    fits.HDUList([fits.PrimaryHDU(), spectrum, ebounds]).writeto(path)

    counts, e_min, e_max = upperlimit._read_fakeit_expected_counts(
        path,
        response_path=tmp_path / "unused.rsp",
    )
    assert np.array_equal(counts, np.array([20.0, 10.0]))
    assert np.array_equal(e_min, np.array([1.0, 0.0]))
    assert np.array_equal(e_max, np.array([2.0, 1.0]))


def test_fakeit_pha_reader_falls_back_to_response_ebounds(monkeypatch, tmp_path):
    import jinwu.core.io as io

    monkeypatch.setattr(
        io,
        "read_pha",
        lambda path: SimpleNamespace(
            counts=np.array([20.0, 10.0]),
            channels=np.array([2, 1]),
            ebounds=None,
        ),
    )
    monkeypatch.setattr(
        io,
        "read_rmf",
        lambda path: SimpleNamespace(
            channel=np.array([1, 2]),
            e_min=np.array([0.0, 1.0]),
            e_max=np.array([1.0, 2.0]),
        ),
    )

    counts, e_min, e_max = upperlimit._read_fakeit_expected_counts(
        tmp_path / "fake.pha",
        response_path=tmp_path / "response.rsp",
    )
    assert np.array_equal(counts, np.array([20.0, 10.0]))
    assert np.array_equal(e_min, np.array([1.0, 0.0]))
    assert np.array_equal(e_max, np.array([2.0, 1.0]))


def test_gecam_requires_explicit_calibration_identity():
    with pytest.raises(ValueError, match="explicit detector"):
        instrument("GECAM")
    cfg = instrument(
        "GECAM",
        detector="GRD01",
        energy_range_keV=(20.0, 1000.0),
    )
    assert cfg.name == "GECAM_GRD01"
    assert cfg.energy_range_keV == (20.0, 1000.0)


def test_count_predictor_is_public():
    assert "CountPredictor" in upperlimit.__all__
