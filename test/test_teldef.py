"""Numerical contracts for Teldef inverse-gnomonic projection."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from jinwu.ftools import teldef_helpers
from jinwu.ftools.teldef import Teldef


ROOT = Path(__file__).resolve().parents[1]
REAL_TELDEF = ROOT / "test/data/swx20230701v001.teldef"


def test_inverse_gnomonic_returns_radians_and_rejects_singular_plane():
    ra, dec = teldef_helpers.inverse_gnomonic(
        0.1, -0.2, math.radians(50.0), math.radians(30.0)
    )

    expected_ra = 50.0 + math.degrees(math.atan2(
        0.1,
        math.cos(math.radians(30.0)) + 0.2 * math.sin(math.radians(30.0)),
    ))
    assert math.degrees(ra) == pytest.approx(expected_ra)
    assert math.degrees(dec) == pytest.approx(18.597574400823223, abs=1e-12)
    with pytest.raises(ValueError, match="Inverse gnomonic singular"):
        teldef_helpers.inverse_gnomonic(
            0.0, math.cos(math.pi / 2), 0.0, math.pi / 2
        )


def test_det2sky_cached_path_does_not_recompute_pointing_trigonometry(monkeypatch):
    mapper = teldef_helpers.Det2Sky(
        np.eye(3), 3500.0, 0.04, 0.04, (300.0, 300.0), 50.0, 30.0
    )

    def fail(*args, **kwargs):
        raise AssertionError("cached pointing trigonometry was recomputed")

    monkeypatch.setattr(teldef_helpers.math, "sin", fail)
    monkeypatch.setattr(teldef_helpers.math, "cos", fail)

    ra, dec = mapper.apply(300.0, 300.0)
    assert ra == pytest.approx(50.0, abs=1e-12)
    assert dec == pytest.approx(30.0, abs=1e-12)


@pytest.mark.real_data
@pytest.mark.skipif(not REAL_TELDEF.is_file(), reason="Swift XRT TELDEF fixture unavailable")
def test_real_teldef_inverse_projection_paths_remain_finite():
    teldef = Teldef.from_file(REAL_TELDEF)
    assert teldef.align is not None
    assert teldef.focal_length is not None
    teldef._build_and_cache_det2sky(50.0, 30.0)

    for x, y in ((280.0, 300.0), (250.0, 300.0), (200.0, 300.0), (100.0, 300.0)):
        direct = teldef.det_to_sky_with_pointing(x, y, 50.0, 30.0)
        cached = teldef.det2sky.apply(x, y)
        assert np.all(np.isfinite(direct))
        assert np.all(np.isfinite(cached))
