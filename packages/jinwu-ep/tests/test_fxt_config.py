from dataclasses import replace

import pytest

from jinwu.core.config import FXT
from jinwu.ep.fxt import FXTInput, FXTPipeline


@pytest.mark.parametrize(
    ("configured_band", "explicit_band", "expected_band"),
    [
        ((0.3, 10.0), None, (0.3, 10.0)),
        ((0.3, 10.0), (0.7, 8.0), (0.7, 8.0)),
        (None, None, (0.4, 9.0)),
    ],
)
def test_fxt_fit_band_precedence(tmp_path, configured_band, explicit_band, expected_band):
    config = FXT(energy_range_keV=(0.4, 9.0))
    config.spectrum = replace(config.spectrum, fit_energy_range_keV=configured_band)
    data = FXTInput(target_id="test", root=tmp_path, ra_deg=10, dec_deg=20)

    pipeline = FXTPipeline(data, config=config, fit_energy_range=explicit_band)

    assert pipeline.fit_band == expected_band
    assert config.spectrum.fit_energy_range_keV == configured_band


def test_fxt_default_fit_band_is_owned_by_the_preset(tmp_path):
    data = FXTInput(target_id="test", root=tmp_path, ra_deg=10, dec_deg=20)
    pipeline = FXTPipeline(data)

    assert pipeline.config.energy_range_keV == (0.3, 10.0)
    assert pipeline.fit_band == pipeline.config.spectrum.fit_energy_range_keV == (0.5, 10.0)
