from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
from astropy.io import fits
import pytest

from jinwu.gw.alert import read_notice, skymap_from_notice
from jinwu.fermi.gbm import estimate_gbm_orbit_period, find_gbm_poshist
from jinwu.gw.layers import footprint_from_circle
from jinwu.gw.pipeline import GWPipelineInput, run_gw_pipeline
from jinwu.gw.skymap import credible_region_stats, load_skymap, probability_in_footprint


def _flat_map(path: Path, *, ordering: str = "RING") -> bytes:
    probability = np.arange(1, 13, dtype=float)
    probability /= probability.sum()
    columns = fits.ColDefs([fits.Column(name="PROB", format="D", array=probability)])
    table = fits.BinTableHDU.from_columns(columns)
    table.header["NSIDE"] = 1
    table.header["ORDERING"] = ordering
    fits.HDUList([fits.PrimaryHDU(), table]).writeto(path)
    return path.read_bytes()


def test_flat_healpix_probability_and_credible_region(tmp_path: Path):
    path = tmp_path / "map.fits"
    _flat_map(path)
    skymap = load_skymap(path)
    assert skymap.total_probability == pytest.approx(1.0)
    stats = credible_region_stats(skymap, 0.5)
    assert stats["probability"] >= 0.5
    assert stats["area_deg2"] > 0
    circle = footprint_from_circle("EP", ra_deg=0, dec_deg=0, radius=90, max_depth=3)
    probability = probability_in_footprint(skymap, circle)
    assert 0.0 < probability < 1.0


def test_credible_region_stats_describes_the_same_region_as_its_moc(tmp_path: Path):
    path = tmp_path / "map.fits"
    _flat_map(path)
    skymap = load_skymap(path)
    stats = credible_region_stats(skymap, 0.5)
    moc_probability = float(stats["moc"].probability_in_multiordermap(skymap.table))
    assert stats["probability"] == pytest.approx(moc_probability, abs=1e-12)


@pytest.mark.parametrize("ra_deg,dec_deg", [(0.0, 0.0), (1.0, 0.0), (0.0, 75.0)])
def test_circle_footprint_uses_a_spherical_cap_not_an_inscribed_polygon(ra_deg: float, dec_deg: float):
    radius_deg = 30.0
    footprint = footprint_from_circle(
        "WXT", ra_deg=ra_deg, dec_deg=dec_deg, radius=radius_deg, kind="footprint", max_depth=13,
    )
    expected = (1.0 - np.cos(np.deg2rad(radius_deg))) / 2.0
    assert footprint.moc.sky_fraction == pytest.approx(expected, abs=1e-4)


def test_notice_embedded_base64_map(tmp_path: Path):
    map_path = tmp_path / "embedded.fits"
    raw = _flat_map(map_path)
    notice_path = tmp_path / "notice.json"
    notice_path.write_text(
        json.dumps({
            "superevent_id": "SLOCAL",
            "alert_type": "INITIAL",
            "time_created": "2024-01-01T00:00:01Z",
            "event": {"time": "2024-01-01T00:00:00Z", "skymap": base64.b64encode(raw).decode()},
        }),
        encoding="utf-8",
    )
    event, payload = read_notice(notice_path)
    resolved, provenance = skymap_from_notice(event, payload, tmp_path / "resolved")
    assert resolved is not None and resolved.is_file()
    assert provenance["embedded"] is True


def test_pipeline_writes_static_report_without_gbm(tmp_path: Path):
    map_path = tmp_path / "map.fits"
    _flat_map(map_path)
    output = tmp_path / "result"
    result = run_gw_pipeline(
        GWPipelineInput(
            target_id="local",
            root=output,
            output_root=output,
            skymap=map_path,
            time="2024-01-01T00:00:00Z",
            gbm_mode="none",
        )
    )
    assert result.event.superevent_id == "local"
    assert result.spectral_analysis == "not_run"
    assert Path(result.output_paths["coverage_json"]).is_file()
    assert Path(result.output_paths["allsky_png"]).is_file()
    assert Path(result.output_paths["allsky_pdf"]).is_file()
    assert result.provenance["sha256"]
    assert "credible_regions" in result.to_dict()
    assert Path(result.output_paths["gbm_png"]).is_file()


def test_gbm_poshist_policy_and_period(tmp_path: Path):
    positions = np.column_stack([
        np.full(4, 6.9e6), np.zeros(4), np.zeros(4)
    ])
    columns = fits.ColDefs([
        fits.Column(name="POS_X", format="D", array=positions[:, 0]),
        fits.Column(name="POS_Y", format="D", array=positions[:, 1]),
        fits.Column(name="POS_Z", format="D", array=positions[:, 2]),
    ])
    path = tmp_path / "poshist.fit"
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(columns)]).writeto(path)
    period = estimate_gbm_orbit_period([path])
    assert period.to_value("s") > 5000
    selection = find_gbm_poshist("2024-01-01T00:00:00Z", tmp_path, mode="none", download=False)
    assert selection.status == "unknown"
    assert selection.reason == "disabled"
