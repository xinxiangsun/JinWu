"""Opt-in HEASoft and local-data acceptance tests for BAT survey products."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from jinwu.swift.bat.survey import BATSurveyInput, BATSurveyPipeline


_AT20G_ROOT = Path("/home/xinxiang/research/batsurvey_lmjagn/batdata")
_AT20G_PRODUCTS = _AT20G_ROOT / "00097302084_surveyresult"
_AT20G_PHA = (
    _AT20G_PRODUCTS
    / "PHA_files"
    / "AT20G_J182338-345412_survey_point_20241981206.pha"
)
_RUN_REAL = os.environ.get("JINWU_RUN_REAL_DATA", "").strip().lower() in {
    "1",
    "true",
    "yes",
}


@pytest.mark.heasoft
@pytest.mark.real_data
@pytest.mark.skipif(
    not _RUN_REAL,
    reason="set JINWU_RUN_REAL_DATA=1 to run local HEASoft acceptance data",
)
@pytest.mark.skipif(
    not _AT20G_PHA.is_file(),
    reason="reference AT20G 00097302084 PHA is not available",
)
def test_at20g_existing_survey_pha_runs_real_upper_limit(tmp_path):
    """Run the actual local PHA/RSP through validation, XSPEC and reporting."""
    result = BATSurveyPipeline(
        BATSurveyInput(
            target_id="AT20G_J182338-345412",
            root=_AT20G_ROOT,
            output_root=tmp_path / "at20g",
            source_name="AT20G J182338-345412",
            obsids=("00097302084",),
            survey_products_dir=_AT20G_PRODUCTS,
            time_windows=(("2024-07-16T12:06:10", "2024-07-16T12:19:58"),),
        )
    ).run(resume=False)

    assert result.full_pipeline
    assert result.science_status == "complete"
    base = tmp_path / "at20g" / "products" / "AT20G_J182338-345412"
    preflight = json.loads((base / "preflight.json").read_text(encoding="utf-8"))
    assert all(preflight["dependency_checks"].values())
    assert preflight["caldb_mode"] in {"local", "remote"}
    assert preflight["runtime_versions"]["xspec"].endswith("/12.15.1")
    assert preflight["runtime_sources"]["heasoftpy"].endswith(
        "/heasoftpy/__init__.py"
    )
    assert {item["name"] for item in preflight["caldb_provenance"]} == {
        "CALDBCONFIG",
        "CALDBALIAS",
    }
    assert all(len(item["sha256"]) == 64 for item in preflight["caldb_provenance"])

    spectra = json.loads((base / "spectra.json").read_text(encoding="utf-8"))
    assert len(spectra["spectra"]) == 1
    assert spectra["spectra"][0]["valid"]
    assert spectra["spectra"][0]["channels"] == 8

    fits = json.loads((base / "fits.json").read_text(encoding="utf-8"))
    assert len(fits["results"]) == 1
    item = fits["results"][0]
    assert "upperlim" not in Path(item["pha"]).name.lower()
    assert item["status"] == "upper_limit_ready"
    profile = item["result"]
    assert profile["normalization_at_lower_bound"]
    assert profile["profile_delta_stat"] == pytest.approx(9.0, abs=0.02)
    assert profile["profile_delta_tolerance"] == pytest.approx(0.02)
    assert profile["flux_erg_cm2_s"] > 0
