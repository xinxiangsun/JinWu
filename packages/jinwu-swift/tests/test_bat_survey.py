from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
import sys
import tarfile
import time
from types import SimpleNamespace

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
import numpy as np
import pytest

from jinwu.core.config import BATSurvey, FitConfig, SwiftBATSurveyConfig, UpperLimitConfig, instrument
from jinwu.core.fit import _freeze_prepared_parameters
from jinwu.core.pipeline import PipelineStage, StageResult
from jinwu.core.time import Time
import jinwu.swift.bat.survey as survey_module
from jinwu.swift.bat.survey import (
    BATSurveyInput,
    BATSurveyPipeline,
    calculate_background_scale,
    gti_overlap_duration,
    parse_area_table,
    read_gti_intervals,
    read_bat_survey_rates,
    safe_extract_archive,
    select_overlapping_pointings,
    signed_snr,
    validate_survey_pha,
)


def test_response_kind_strips_transport_compression():
    assert survey_module._response_is_arf(Path("source.arf"))
    assert survey_module._response_is_arf(Path("source.arf.gz"))
    assert survey_module._response_is_arf(Path("source.arf.bz2"))
    assert not survey_module._response_is_arf(Path("source.rsp.gz"))


def test_bat_survey_profiles_and_input_contract(tmp_path):
    default = instrument("BATSurvey")
    lmjagn = BATSurvey(profile="lmjagn")
    assert default.pipeline == "swift.bat.survey"
    assert default.survey.detthresh == 10000
    assert default.survey.min_pcode == 0.05
    assert default.selection.detthresh == 10000
    assert lmjagn.survey.detthresh == 8000
    assert lmjagn.survey.min_pcode == 0.01
    assert lmjagn.mosaic.min_pcode == 0.01
    assert BATSurvey(processes=2).survey.processes == 2

    data = BATSurveyInput(
        target_id="target",
        root=tmp_path,
        coord=(10.0, -20.0),
        time_windows=(("2024-01-01", "2024-01-02"),),
    )
    assert data.time_windows[0][0].startswith("2024-01-01")
    assert data.skycoord.ra.deg == pytest.approx(10.0)
    skycoord_input = BATSurveyInput(
        target_id="skycoord-target",
        root=tmp_path,
        skycoord=SkyCoord(11.0, -21.0, unit="deg"),
    )
    assert skycoord_input.skycoord.dec.deg == pytest.approx(-21.0)
    with pytest.raises(ValueError, match="either coord or skycoord"):
        BATSurveyInput(
            target_id="ambiguous-coordinate",
            root=tmp_path,
            coord=(10.0, -20.0),
            skycoord=SkyCoord(11.0, -21.0, unit="deg"),
        )
    assert BATSurveyInput(target_id="obsid", root=tmp_path, obsids=(97302084,)).obsids == (
        "00097302084",
    )
    met_pair = BATSurveyInput(
        target_id="met-window",
        root=tmp_path,
        time_windows=(
            Time("2024-01-01", scale="utc"),
            Time("2024-01-02", scale="utc"),
        ),
    )
    assert len(met_pair.time_windows) == 1


def test_bat_upper_limit_policy_is_canonical_with_legacy_mapping():
    legacy = SwiftBATSurveyConfig(
        upper_limit_photon_index=2.7,
        upper_limit_delta_stat=16.0,
    )
    mapped = BATSurvey(survey=legacy)
    assert mapped.upper_limit.spectral_index == pytest.approx(2.7)
    assert mapped.upper_limit.default_sigma == pytest.approx(4.0)

    explicit = UpperLimitConfig(
        strategy="coded_mask_spectrum",
        background_likelihood="gaussian_net_rate",
        response_folding="rsp",
        result_modes=("observed_upper_bound", "detection_sensitivity"),
        default_sigma=5.0,
        spectral_index=1.8,
        calibration="empirical_if_available",
        calibration_mode="empirical_if_available",
        enabled=True,
        unavailable_reason=None,
    )
    chosen = BATSurvey(survey=legacy, upper_limit=explicit)
    assert chosen.upper_limit.spectral_index == pytest.approx(1.8)
    assert chosen.upper_limit.default_sigma == pytest.approx(5.0)


def test_signed_rate_and_area_scale(tmp_path):
    assert signed_snr(-4e-4, 7e-4) < 0
    area = tmp_path / "source.area.gz"
    with gzip.open(area, "wt", encoding="utf-8") as handle:
        handle.write("# start stop background_area\n")
        handle.write("0 10 200\n")
        handle.write("10 20 100\n")
    rows = parse_area_table(area)
    assert len(rows) == 2
    assert calculate_background_scale(area, source_area=10 * u.pixel**2, time_s=5) == pytest.approx(0.05)
    with pytest.raises(TypeError):
        calculate_background_scale(area, source_area=10)  # type: ignore[arg-type]


def test_rate_column_contract_and_overlap(tmp_path):
    table = tmp_path / "rates.csv"
    table.write_text(
        "OBSID,IMAGE_ID,NAME,TSTART,TSTOP,EXPOSURE,CENT_RATE,BKG_VAR,PCODEFR,BADBIN,STATUS\n"
        "00000000001,p1,T,100,200,100,-0.0004,0.0007,0.2,False,SUCCESS\n"
        "00000000001,p2,T,200,300,100,0.01,0.001,0.2,False,SUCCESS\n",
        encoding="utf-8",
    )
    points = read_bat_survey_rates(table, source_name="T")
    assert points[0].rate == pytest.approx(-0.0004)
    assert points[0].rate_error == pytest.approx(0.0007)
    assert points[0].error_source == "BKG_VAR"
    selected = select_overlapping_pointings(points, 150, 250, min_pcode=0.05)
    assert [item["point"]["pointing_id"] for item in selected] == ["p1", "p2"]
    assert selected[0]["overlap_s"] == pytest.approx(50)
    assert selected[0]["effective_exposure_s"] == pytest.approx(100)
    with pytest.raises(ValueError):
        select_overlapping_pointings(points, 200, 200.0)


def test_all_zero_rates_are_preserved_and_marked_for_review(tmp_path):
    table = tmp_path / "zero.csv"
    table.write_text(
        "OBSID,IMAGE_ID,NAME,TSTART,TSTOP,EXPOSURE,CENT_RATE,BKG_VAR,PCODEFR,STATUS\n"
        "1,p1,T,100,200,100,0,0.001,0.2,SUCCESS\n",
        encoding="utf-8",
    )
    points = read_bat_survey_rates(table, source_name="T")
    assert points[0].rate == 0.0
    pipeline = BATSurveyPipeline(
        BATSurveyInput(
            target_id="zero-target",
            root=tmp_path,
            survey_products_dir=table,
            source_name="T",
        )
    )
    result = pipeline.run(resume=False)
    assert result.quality_status == "needs_review"
    lightcurve = Path(result.products["lightcurve"]["lightcurve"]).read_text(encoding="utf-8")
    assert "all_zero_survey_rates" in lightcurve


def test_source_catalog_aliases_are_matched_when_reading_rates(tmp_path):
    table = tmp_path / "alias.csv"
    table.write_text(
        "OBSID,IMAGE_ID,NAME,TSTART,TSTOP,EXPOSURE,CENT_RATE,BKG_VAR,PCODEFR,STATUS\n"
        "1,p1,Mrk_79,100,200,100,0.01,0.001,0.2,SUCCESS\n",
        encoding="utf-8",
    )
    points = read_bat_survey_rates(table, source_name="Mrk 79")
    assert len(points) == 1


def test_compressed_raw_observation_files_are_validated(tmp_path):
    for relative in ("bat/survey", "bat/hk", "auxil"):
        directory = tmp_path / relative
        directory.mkdir(parents=True)
        file_name = {
            "bat/survey": "sw00000000001bsvpbo4b69g0fb4.dph.gz",
            "bat/hk": "sw00000000001bdecb.hk.gz",
            "auxil": "sw00000000001uat.fits.gz",
        }[relative]
        target = directory / file_name
        with gzip.open(target, "wb") as handle:
            fits.PrimaryHDU().writeto(handle)
    from jinwu.swift.bat.survey import validate_observation_directory

    valid, diagnostics = validate_observation_directory(tmp_path)
    assert valid, diagnostics


def test_fits_vector_rate_preserves_eight_bands_and_total(tmp_path):
    path = tmp_path / "target.cat"
    columns = [
        fits.Column(name="NAME", format="12A", array=["T"]),
        fits.Column(name="TIME", format="D", unit="s", array=[100.0]),
        fits.Column(name="TIME_STOP", format="D", unit="s", array=[200.0]),
        fits.Column(name="EXPOSURE", format="D", unit="s", array=[80.0]),
        fits.Column(name="IMAGE_ID", format="12A", array=["p1"]),
        fits.Column(name="PCODEFR", format="E", array=[0.2]),
        fits.Column(name="CENT_RATE", format="9D", unit="count/s", array=[np.arange(9)]),
        fits.Column(name="BKG_VAR", format="9D", unit="count/s", array=[np.ones(9)]),
        fits.Column(name="VECTSNR", format="9D", array=[np.arange(9)]),
    ]
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(columns)]).writeto(path)
    point = read_bat_survey_rates(path, source_name="T")[0]
    assert point.rate == pytest.approx(8.0)
    assert point.rate_source == "native_central_rate"
    assert point.native_channel_count == 8
    assert point.total_band_is_last
    assert len(point.channel_rate or ()) == 9
    assert point.rate_unit == "count/s"


def test_eight_band_cat_derives_total_without_reusing_last_band(tmp_path):
    path = tmp_path / "eight_band.cat"
    columns = [
        fits.Column(name="NAME", format="12A", array=["T"]),
        fits.Column(name="TIME", format="D", array=[100.0]),
        fits.Column(name="TIME_STOP", format="D", array=[200.0]),
        fits.Column(name="EXPOSURE", format="D", array=[80.0]),
        fits.Column(name="IMAGE_ID", format="12A", array=["p1"]),
        fits.Column(name="PCODEFR", format="E", array=[0.2]),
        fits.Column(name="CENT_RATE", format="8D", unit="count/s", array=[np.ones(8)]),
        fits.Column(name="RATE_ERR", format="8D", unit="count/s", array=[np.full(8, 2.0)]),
        fits.Column(name="BKG_VAR", format="8D", unit="count/s", array=[np.full(8, 3.0)]),
        fits.Column(name="VECTSNR", format="8D", array=[np.full(8, 0.5)]),
    ]
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(columns)]).writeto(path)
    point = read_bat_survey_rates(path, source_name="T")[0]
    assert point.rate == pytest.approx(8.0)
    assert point.rate_error == pytest.approx(2.0 * np.sqrt(8))
    assert point.snr == pytest.approx(8.0 / (3.0 * np.sqrt(8)))
    assert point.native_channel_count == 8
    assert point.total_band_is_last
    assert len(point.channel_rate or ()) == 9


def test_invalid_rate_error_uses_explicit_bkg_var_fallback(tmp_path):
    path = tmp_path / "fallback.cat"
    columns = [
        fits.Column(name="NAME", format="12A", array=["T"]),
        fits.Column(name="TIME", format="D", array=[100.0]),
        fits.Column(name="TIME_STOP", format="D", array=[200.0]),
        fits.Column(name="CENT_RATE", format="D", array=[-1.0]),
        fits.Column(name="RATE_ERR", format="D", array=[np.nan]),
        fits.Column(name="BKG_VAR", format="D", array=[0.25]),
    ]
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(columns)]).writeto(path)
    point = read_bat_survey_rates(path, source_name="T")[0]
    assert point.rate_error == pytest.approx(0.25)
    assert point.error_source == "BKG_VAR"


def test_gti_union_overlap_does_not_double_count_or_scale_exposure(tmp_path):
    path = tmp_path / "point.gti"
    hdu = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="START", format="D", array=[0.0, 5.0]),
            fits.Column(name="STOP", format="D", array=[10.0, 15.0]),
        ],
        name="STDGTI",
    )
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(path)
    intervals = read_gti_intervals(path)
    assert gti_overlap_duration(intervals, 4, 12) == pytest.approx(8)


def test_gti_timezero_is_applied_in_seconds(tmp_path):
    path = tmp_path / "timezero.gti"
    hdu = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="START", format="D", array=[0.0]),
            fits.Column(name="STOP", format="D", array=[10.0]),
        ],
        name="STDGTI",
    )
    hdu.header["TIMEZERO"] = 100.0
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(path)
    assert read_gti_intervals(path) == ((100.0, 110.0),)


def test_pha_time_interval_applies_timeunit_and_timezero(tmp_path):
    path = tmp_path / "timed.pha"
    hdu = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="CHANNEL", format="I", array=[0]),
            fits.Column(name="RATE", format="D", array=[0.0]),
            fits.Column(name="STAT_ERR", format="D", array=[1.0]),
        ],
        name="SPECTRUM",
    )
    hdu.header.update(
        {"TSTART": 1000.0, "TSTOP": 2000.0, "TIMEUNIT": "ms", "TIMEZERO": 1000.0}
    )
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(path)
    assert survey_module._pha_time_interval(path) == (2.0, 3.0)


def test_numeric_dat_distinguishes_flux_ecf_and_native_rate(tmp_path):
    derived = tmp_path / "derived.dat"
    # Real Burst Analyser fourteen-column layout: ECF is column 12;
    # columns 7/8 include ECF uncertainty and are not statistical rate errors.
    derived.write_text(",".join(str(value) for value in [1, 0.1, 0.1, 4, 0.4, -0.3, 2, -2.5, 1.9, 0.1, -0.1, 2, 0.1, -0.1]) + "\n")
    point = read_bat_survey_rates(derived)[0]
    assert point.rate_source == "flux_over_ecf"
    assert point.rate == pytest.approx(2)
    assert point.rate_error == pytest.approx(0.2)
    native = tmp_path / "native.dat"
    native.write_text(",".join(str(value) for value in [1, 0.1, 0.1, 4, 0.4, 0.3, -1, 0.2, 0.3] + [0] * 6) + "\n")
    point = read_bat_survey_rates(native)[0]
    assert point.rate_source == "native_rate"
    assert point.rate == pytest.approx(-1)
    assert point.rate_error == pytest.approx(0.3)


def test_invalid_fourteen_column_ecf_is_retained_as_unavailable_rate(tmp_path):
    path = tmp_path / "invalid_ecf.dat"
    path.write_text(",".join(str(value) for value in [
        10, 2, -2, 1.5e-10, 3e-11, -3e-11, 4e-11, -5e-11,
        1.9, 0.1, -0.1, 0.0, 1e-13, -1e-13,
    ]) + "\n", encoding="utf-8")
    points = read_bat_survey_rates(path)
    assert len(points) == 1
    assert points[0].rate is None
    assert points[0].rate_source == "flux_over_ecf"
    assert not select_overlapping_pointings(points, 0, 20)


def test_qdp_rate_is_explicitly_native_and_not_an_exposure_interval(tmp_path):
    path = tmp_path / "WTCURVE.qdp"
    path.write_text(
        "READ SERR 1\n"
        "! Time TimePos TimeNeg Rate RatePos RateNeg\n"
        "10 2 2 -0.25 0.05 0.04\n",
        encoding="utf-8",
    )
    point = read_bat_survey_rates(path)[0]
    assert point.rate_source == "native_qdp_rate"
    assert point.rate == pytest.approx(-0.25)
    assert point.rate_error == pytest.approx(0.05)
    assert point.time_stop is None
    symmetric = tmp_path / "symmetric.qdp"
    symmetric.write_text("READ SERR 1\n10 -0.3 0.06\n", encoding="utf-8")
    symmetric_point = read_bat_survey_rates(symmetric)[0]
    assert symmetric_point.rate == pytest.approx(-0.3)
    assert symmetric_point.rate_error == pytest.approx(0.06)


def test_pipeline_existing_file_and_cache_invalidation(tmp_path):
    survey = tmp_path / "survey.csv"
    survey.write_text(
        "OBSID,IMAGE_ID,NAME,TSTART,TSTOP,EXPOSURE,CENT_RATE,BKG_VAR,PCODEFR,STATUS\n"
        "1,p1,T,100,200,100,0.01,0.001,0.2,SUCCESS\n",
        encoding="utf-8",
    )
    input_data = BATSurveyInput(
        target_id="cache-target",
        root=tmp_path,
        survey_products_dir=survey,
        source_name="T",
    )
    first = BATSurveyPipeline(input_data).run(resume=False)
    assert first.full_pipeline
    lightcurve = Path(first.products["lightcurve"]["lightcurve"])
    assert '"rate": 0.01' in lightcurve.read_text(encoding="utf-8")
    time.sleep(0.02)
    survey.write_text(
        survey.read_text(encoding="utf-8").replace("0.01,0.001", "0.02,0.001"),
        encoding="utf-8",
    )
    second = BATSurveyPipeline(input_data).run(resume=True)
    assert '"rate": 0.02' in Path(second.products["lightcurve"]["lightcurve"]).read_text(encoding="utf-8")


def test_bat_fit_config_fingerprint_is_scoped_to_downstream_stages(tmp_path):
    survey = tmp_path / "survey.csv"
    survey.write_text(
        "OBSID,IMAGE_ID,NAME,TSTART,TSTOP,EXPOSURE,CENT_RATE,BKG_VAR,PCODEFR,STATUS\n"
        "1,p1,T,100,200,100,0.01,0.001,0.2,SUCCESS\n",
        encoding="utf-8",
    )
    data = BATSurveyInput(
        target_id="scoped-cache",
        root=tmp_path,
        survey_products_dir=survey,
        source_name="T",
    )
    default = BATSurveyPipeline(data, config=BATSurvey())
    changed = BATSurveyPipeline(
        data,
        config=BATSurvey(
            fitting=FitConfig(model_name="cflux*powerlaw", statistic="chi", error_delta_stat=2.0)
        ),
    )
    assert default._input_fingerprint() == changed._input_fingerprint()
    assert default._stage_config_fingerprint(PipelineStage("lightcurve")) == changed._stage_config_fingerprint(
        PipelineStage("lightcurve")
    )
    assert default._stage_config_fingerprint(PipelineStage("fit")) != changed._stage_config_fingerprint(
        PipelineStage("fit")
    )


def test_until_stage_is_distinct_from_full_pipeline(tmp_path):
    survey = tmp_path / "survey.csv"
    survey.write_text(
        "OBSID,IMAGE_ID,NAME,TSTART,TSTOP,EXPOSURE,CENT_RATE,BKG_VAR,PCODEFR,STATUS\n"
        "1,p1,T,100,200,100,0.01,0.001,0.2,SUCCESS\n",
        encoding="utf-8",
    )
    input_data = BATSurveyInput(
        target_id="until-target",
        root=tmp_path,
        output_root=tmp_path / "output",
        survey_products_dir=survey,
        source_name="T",
        time_windows=(("1970-01-01T00:01:40", "1970-01-01T00:03:20"),),
    )

    stopped = BATSurveyPipeline(input_data).run(until="lightcurve", resume=False)
    assert not stopped.full_pipeline
    assert stopped.status.value == "pending"
    assert "report" not in stopped.products

    completed = BATSurveyPipeline(input_data).run(resume=True)
    assert completed.full_pipeline
    assert "report" in completed.products


def test_download_uses_bounded_retries_and_timeout(tmp_path):
    calls = []

    def download_swiftdata(table, reload=False, jobs=10, **kwargs):
        calls.append((list(table), bool(reload), jobs, kwargs))
        ok = len(calls) >= 2
        return {str(item): {"success": ok} for item in table}

    from jinwu.swift.bat.survey import BatAnalysisSurveyBackend

    backend = BatAnalysisSurveyBackend(module=SimpleNamespace(download_swiftdata=download_swiftdata))
    result = backend.download(
        obsids=("00000001",),
        destination=tmp_path / "raw",
        retries=3,
        retry_wait_s=0,
        timeout_s=17,
    )
    assert len(calls) == 2
    assert calls[0][1] is False and calls[1][1] is True
    assert calls[0][3]["timeout"] == 17
    assert result["00000001"]["success"]


def test_download_retries_transient_exceptions_without_resubmitting_forever(tmp_path):
    calls = []

    def download_swiftdata(table, **kwargs):
        calls.append(tuple(table))
        if len(calls) == 1:
            raise TimeoutError("temporary")
        return {str(item): {"success": True} for item in table}

    from jinwu.swift.bat.survey import BatAnalysisSurveyBackend

    backend = BatAnalysisSurveyBackend(module=SimpleNamespace(download_swiftdata=download_swiftdata))
    result = backend.download(
        obsids=("00000001",),
        destination=tmp_path / "raw",
        retries=3,
        retry_wait_s=0,
        timeout_s=17,
    )
    assert calls == [("00000001",), ("00000001",)]
    assert result["00000001"]["success"]


def test_download_does_not_repeat_an_ambiguous_remote_result(tmp_path):
    calls = []

    def download_swiftdata(observations, **kwargs):
        calls.append(tuple(observations))
        return {"task_id": "submitted-without-per-observation-status"}

    from jinwu.swift.bat.survey import BatAnalysisSurveyBackend

    backend = BatAnalysisSurveyBackend(
        module=SimpleNamespace(download_swiftdata=download_swiftdata)
    )
    result = backend.download(
        obsids=("00000000001",),
        destination=tmp_path / "raw",
        retries=3,
        retry_wait_s=0,
    )
    assert calls == [("00000000001",)]
    assert result["00000000001"]["error"] == "ambiguous_download_result"


def test_mosaic_window_members_keep_full_exposure_and_mark_shared_rows(tmp_path):
    from jinwu.swift.bat.survey import BatAnalysisSurveyBackend

    base = 900.0
    inventory = tmp_path / "outventory_all.fits"
    table = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="TSTART", format="D", array=[base, base + 100]),
            fits.Column(name="TSTOP", format="D", array=[base + 200, base + 300]),
            fits.Column(name="EXPOSURE", format="D", array=[100, 100]),
            fits.Column(name="IMAGE_STATUS", format="L", array=[True, True]),
            fits.Column(name="PCODEFR", format="E", array=[0.2, 0.2]),
            fits.Column(name="POINTING", format="12A", array=["p1", "p2"]),
        ]
    )
    fits.HDUList([fits.PrimaryHDU(), table]).writeto(inventory)
    start1 = Time(base + 50, format="swiftmet").utc.isot
    stop1 = Time(base + 150, format="swiftmet").utc.isot
    start2 = Time(base + 150, format="swiftmet").utc.isot
    stop2 = Time(base + 250, format="swiftmet").utc.isot
    members = BatAnalysisSurveyBackend._inventory_window_members(
        inventory,
        ((start1, stop1), (start2, stop2)),
        min_pcode=0.15,
    )
    assert [item["id"] for item in members[0]["members"]] == ["p1", "p2"]
    assert members[0]["members"][0]["exposure_s"] == pytest.approx(100)
    assert members[0]["members"][0]["overlap_s"] == pytest.approx(100)
    assert [item["id"] for item in members[1]["members"]] == ["p1", "p2"]
    assert set(BatAnalysisSurveyBackend._shared_members(members)) == {"p1", "p2"}


def test_safe_archive_and_pha_validation(tmp_path):
    archive = tmp_path / "safe.tar"
    source = tmp_path / "payload.txt"
    source.write_text("ok", encoding="utf-8")
    with tarfile.open(archive, "w") as handle:
        handle.add(source, arcname="nested/payload.txt")
    extracted = safe_extract_archive(archive, tmp_path / "extract")
    assert extracted[0].exists()

    bad_archive = tmp_path / "bad.tar"
    with tarfile.open(bad_archive, "w") as handle:
        info = tarfile.TarInfo("../escape.txt")
        info.size = 0
        handle.addfile(info)
    with pytest.raises(ValueError):
        safe_extract_archive(bad_archive, tmp_path / "bad-extract")

    response = tmp_path / "source.rsp"
    response_ebounds = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="CHANNEL", format="I", array=np.arange(8, dtype=np.int16)),
            fits.Column(name="E_MIN", format="E", array=np.arange(8, dtype=np.float32)),
            fits.Column(name="E_MAX", format="E", array=np.arange(8, dtype=np.float32) + 1),
        ],
        name="EBOUNDS",
    )
    fits.HDUList([fits.PrimaryHDU(), response_ebounds]).writeto(response)
    column = fits.Column(name="CHANNEL", format="I", array=np.arange(8, dtype=np.int16))
    rate = fits.Column(name="RATE", format="E", array=np.ones(8, dtype=np.float32))
    stat_err = fits.Column(name="STAT_ERR", format="E", array=np.ones(8, dtype=np.float32))
    hdu = fits.BinTableHDU.from_columns([column, rate, stat_err])
    hdu.header["EXPOSURE"] = 100.0
    hdu.header["BACKSCAL"] = 1.0
    hdu.header["RESPFILE"] = response.name
    pha = tmp_path / "source.pha"
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(pha)
    result = validate_survey_pha(pha)
    assert result.valid
    assert result.channels == 8
    assert result.response == response


def test_pha_response_channel_mismatch_is_rejected(tmp_path):
    response = tmp_path / "mismatch.rsp"
    ebounds = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="CHANNEL", format="I", array=np.arange(7, dtype=np.int16)),
            fits.Column(name="E_MIN", format="E", array=np.arange(7, dtype=np.float32)),
            fits.Column(name="E_MAX", format="E", array=np.arange(7, dtype=np.float32) + 1),
        ],
        name="EBOUNDS",
    )
    fits.HDUList([fits.PrimaryHDU(), ebounds]).writeto(response)
    hdu = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="CHANNEL", format="I", array=np.arange(8, dtype=np.int16)),
            fits.Column(name="RATE", format="E", array=np.ones(8, dtype=np.float32)),
            fits.Column(name="STAT_ERR", format="E", array=np.ones(8, dtype=np.float32)),
        ]
    )
    hdu.header["EXPOSURE"] = 10.0
    hdu.header["BACKSCAL"] = 1.0
    hdu.header["RESPFILE"] = response.name
    pha = tmp_path / "mismatch.pha"
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(pha)
    result = validate_survey_pha(pha)
    assert not result.valid
    assert any(item.startswith("response_channel_mismatch") for item in result.diagnostics)


def test_pha_response_matrix_without_ebounds_is_rejected(tmp_path):
    response = tmp_path / "matrix_only.rsp"
    matrix = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="CHANNEL", format="I", array=np.arange(8, dtype=np.int16)),
            fits.Column(name="MATRIX", format="8E", array=np.ones((8, 8), dtype=np.float32)),
        ],
        name="MATRIX",
    )
    fits.HDUList([fits.PrimaryHDU(), matrix]).writeto(response)
    hdu = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="CHANNEL", format="I", array=np.arange(8, dtype=np.int16)),
            fits.Column(name="RATE", format="E", array=np.ones(8, dtype=np.float32)),
            fits.Column(name="STAT_ERR", format="E", array=np.ones(8, dtype=np.float32)),
        ]
    )
    hdu.header["EXPOSURE"] = 10.0
    hdu.header["BACKSCAL"] = 1.0
    hdu.header["RESPFILE"] = response.name
    pha = tmp_path / "matrix_only.pha"
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(pha)
    result = validate_survey_pha(pha)
    assert not result.valid
    assert "response_missing_EBOUNDS" in result.diagnostics


def test_existing_cache_is_copied_before_backend_load(tmp_path):
    from jinwu.swift.bat.survey import BatAnalysisSurveyBackend

    source = tmp_path / "raw-cache"
    source.mkdir()
    (source / "batsurvey.pickle").write_bytes(b"cache")
    (source / ".batsurvey_complete").write_text("", encoding="utf-8")
    output = tmp_path / "run" / "survey" / "00000000001"
    staged = BatAnalysisSurveyBackend._staged_result_cache(
        "00000000001", source, output
    )
    assert staged != source
    assert (staged / "batsurvey.pickle").read_bytes() == b"cache"
    assert source.joinpath(".local_pfile").exists() is False


def test_diagnostic_fits_do_not_count_as_survey_products(tmp_path):
    """Image/statistics FITS alone cannot satisfy the survey product gate."""
    diagnostic = tmp_path / "stats_obs.fits"
    table = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="OBS_ID", format="16A", array=["00000000001"]),
            fits.Column(name="EXPOSURE", format="D", array=[100.0]),
        ],
        name="STATS_OBS",
    )
    fits.HDUList([fits.PrimaryHDU(), table]).writeto(diagnostic)
    assert not survey_module._has_product(diagnostic)

    rate_product = tmp_path / "sources_tot.fits"
    rate_table = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="RATE", format="D", array=[0.1]),
            fits.Column(name="RATE_ERR", format="D", array=[0.01]),
        ],
        name="SOURCES",
    )
    fits.HDUList([fits.PrimaryHDU(), rate_table]).writeto(rate_product)
    assert survey_module._has_product(rate_product)


def test_pipeline_writes_partial_report_without_products(tmp_path):
    result = BATSurveyPipeline(
        BATSurveyInput(
            target_id="empty",
            root=tmp_path,
            time_windows=(("2024-01-01", "2024-01-02"),),
        )
    ).run()
    assert result.status.value == "completed"
    assert result.science_status == "partial"
    report = Path(result.products["report"]["report"])
    assert report.exists()


def test_explicit_fetcher_allows_query_without_existing_local_root(tmp_path):
    calls = []

    def fetcher(input_data):
        calls.append(input_data.target_id)
        return [{"obsid": "00000000001", "source": "tap"}]

    missing_root = tmp_path / "not_created_yet"
    result = BATSurveyPipeline(
        BATSurveyInput(
            target_id="online-target",
            root=missing_root,
            coord=(1.0, 2.0),
            time_windows=(("2024-01-01", "2024-01-02"),),
            query=True,
            fetcher=fetcher,
        )
    ).run(resume=False)
    assert calls == ["online-target"]
    assert result.full_pipeline
    discovery = Path(result.products["discover"]["discovery"]).read_text(encoding="utf-8")
    assert "00000000001" in discovery


class _FakeComponent:
    parameterNames = ("PhoIndex",)

    def __init__(self):
        self.PhoIndex = type("Parameter", (), {"values": [2.0], "frozen": False})()


class _FakeModel:
    componentNames = ("powerlaw",)

    def __init__(self):
        self.powerlaw = _FakeComponent()


def test_fit_prepared_freezes_named_parameter():
    model = _FakeModel()
    _freeze_prepared_parameters(model, {"powerlaw.PhoIndex": 2.0})
    assert model.powerlaw.PhoIndex.values == 2.0
    assert model.powerlaw.PhoIndex.frozen


def _write_synthetic_survey_pha(tmp_path: Path, *, matrix: bool = True) -> tuple[Path, Path]:
    """Write an OGIP-like eight-channel survey PHA and sibling response."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    response = tmp_path / "T_point_p1.rsp"
    response_hdus = [fits.PrimaryHDU()]
    if matrix:
        response_hdus.append(
            fits.BinTableHDU.from_columns(
                [
                    fits.Column(name="ENERG_LO", format="E", array=np.linspace(14, 180, 8)),
                    fits.Column(name="ENERG_HI", format="E", array=np.linspace(20, 195, 8)),
                    fits.Column(name="N_GRP", format="I", array=np.ones(8, dtype=np.int16)),
                    fits.Column(name="F_CHAN", format="1I", array=np.zeros((8, 1), dtype=np.int16)),
                    fits.Column(name="N_CHAN", format="1I", array=np.full((8, 1), 8, dtype=np.int16)),
                    fits.Column(name="MATRIX", format="8E", array=np.ones((8, 8), dtype=np.float32)),
                ],
                name="SPECRESP MATRIX",
            )
        )
        response_hdus[-1].header["HDUCLAS2"] = "RSP_MATRIX"
        response_hdus[-1].header["DETCHANS"] = 8
    response_hdus.append(
        fits.BinTableHDU.from_columns(
            [
                fits.Column(name="CHANNEL", format="I", array=np.arange(8, dtype=np.int16)),
                fits.Column(name="E_MIN", format="E", array=np.array([14, 20, 24, 35, 50, 75, 100, 150], dtype=np.float32)),
                fits.Column(name="E_MAX", format="E", array=np.array([20, 24, 35, 50, 75, 100, 150, 195], dtype=np.float32)),
            ],
            name="EBOUNDS",
        )
    )
    response_hdus[-1].header["HDUCLAS2"] = "EBOUNDS"
    fits.HDUList(response_hdus).writeto(response)

    pha = tmp_path / "T_survey_point_p1.pha"
    spectrum = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="CHANNEL", format="I", array=np.arange(8, dtype=np.int16)),
            fits.Column(name="RATE", format="E", unit="count/s", array=np.full(8, 1.0, dtype=np.float32)),
            fits.Column(name="STAT_ERR", format="E", unit="count/s", array=np.full(8, 0.1, dtype=np.float32)),
            fits.Column(name="SYS_ERR", format="E", unit="count/s", array=np.full(8, 0.02, dtype=np.float32)),
        ],
        name="SPECTRUM",
    )
    spectrum.header.update(
        {
            "EXPOSURE": 100.0,
            "BACKSCAL": 1.0,
            "RESPFILE": response.name,
            "TSTART": 100.0,
            "TSTOP": 200.0,
            "TIMEUNIT": "s",
            "POISSERR": False,
        }
    )
    fits.HDUList([fits.PrimaryHDU(), spectrum]).writeto(pha)
    return pha, response


def test_science_ready_pha_requires_matrix_and_preserves_systematic_error(tmp_path):
    pha, response = _write_synthetic_survey_pha(tmp_path)
    valid = validate_survey_pha(pha, require_matrix=True)
    assert valid.valid
    assert valid.channels == 8
    assert valid.response == response

    with fits.open(response, mode="update", memmap=False) as hdus:
        hdus[1].data["MATRIX"][0][0] = -1.0
        hdus.flush()
    invalid_matrix = validate_survey_pha(pha, require_matrix=True)
    assert not invalid_matrix.valid
    assert "response_invalid_MATRIX" in invalid_matrix.diagnostics

    no_matrix_pha, no_matrix_response = _write_synthetic_survey_pha(
        tmp_path / "without_matrix", matrix=False
    )
    no_matrix = validate_survey_pha(no_matrix_pha, require_matrix=True)
    assert not no_matrix.valid
    assert "response_missing_MATRIX" in no_matrix.diagnostics

    table = tmp_path / "rates_with_systematics.csv"
    table.write_text(
        "OBSID,IMAGE_ID,NAME,TSTART,TSTOP,EXPOSURE,CENT_RATE,RATE_ERR,BKG_VAR,SYS_ERR,PCODEFR,STATUS\n"
        "1,p1,T,100,200,100,1.0,0.1,0.2,0.03,0.2,SUCCESS\n",
        encoding="utf-8",
    )
    point = read_bat_survey_rates(table, source_name="T")[0]
    assert point.rate == pytest.approx(1.0)
    assert point.rate_error == pytest.approx(0.1)
    assert point.error_source == "RATE_ERR"
    assert point.systematic_error == pytest.approx((0.03,))
    assert point.rate_unit == "count/s"


def test_fits_rate_times_apply_timeunit_and_timezero(tmp_path):
    """Normalize numeric OGIP timing cards to the canonical Swift-MET seconds."""
    path = tmp_path / "rates_ms.fits"
    table = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="TSTART", format="D", array=[1000.0]),
            fits.Column(name="TSTOP", format="D", array=[2000.0]),
            fits.Column(name="EXPOSURE", format="D", array=[1.0]),
            fits.Column(name="RATE", format="D", unit="count/s", array=[-0.2]),
            fits.Column(name="STAT_ERR", format="D", unit="count/s", array=[0.1]),
        ],
        name="SOURCES",
    )
    table.header["TIMEUNIT"] = "ms"
    table.header["TIMEZERO"] = 1000.0
    fits.HDUList([fits.PrimaryHDU(), table]).writeto(path)

    point = read_bat_survey_rates(path)[0]
    assert point.time_start == pytest.approx(2.0)
    assert point.time_stop == pytest.approx(3.0)
    assert point.rate == pytest.approx(-0.2)


def test_response_extensions_are_order_independent(tmp_path):
    pha, response = _write_synthetic_survey_pha(tmp_path)
    with fits.open(response, memmap=False) as hdus:
        reordered = fits.HDUList(
            [hdus[0].copy(), hdus[2].copy(), hdus[1].copy()]
        )
    reordered.writeto(response, overwrite=True)
    validation = validate_survey_pha(pha, require_matrix=True)
    assert validation.valid


def test_pha_inventory_filters_source_window_and_legacy_upper_limit_products(tmp_path):
    survey_dir = tmp_path / "products"
    good, _ = _write_synthetic_survey_pha(survey_dir / "T_point_p1")
    with fits.open(good, mode="update", memmap=False) as hdus:
        hdus[1].header["SOURCE"] = "T"
        hdus[1].header["OBSID"] = "00000000001"
        hdus[1].header["POINTING"] = "p1"
        hdus.flush()

    other, _ = _write_synthetic_survey_pha(survey_dir / "other_point_p2")
    with fits.open(other, mode="update", memmap=False) as hdus:
        hdus[1].header["SOURCE"] = "Other source"
        hdus[1].header["OBSID"] = "00000000001"
        hdus[1].header["POINTING"] = "p2"
        hdus.flush()

    outside, _ = _write_synthetic_survey_pha(survey_dir / "T_point_p3")
    with fits.open(outside, mode="update", memmap=False) as hdus:
        hdus[1].header["SOURCE"] = "T"
        hdus[1].header["OBSID"] = "00000000001"
        hdus[1].header["POINTING"] = "p3"
        hdus[1].header["TSTART"] = 500.0
        hdus[1].header["TSTOP"] = 600.0
        hdus.flush()

    legacy_dir = survey_dir / "legacy"
    legacy, legacy_response = _write_synthetic_survey_pha(legacy_dir)
    legacy_named = legacy_dir / "T_bkgnsigma_3_upperlim.pha"
    legacy.rename(legacy_named)
    legacy_response.unlink()
    pipeline = BATSurveyPipeline(
        BATSurveyInput(
            target_id="T",
            root=tmp_path,
            source_name="T",
            obsids=(1,),
            survey_products_dir=survey_dir,
            time_windows=(("2001-01-01T00:01:40", "2001-01-01T00:03:20"),),
        )
    )
    records = [{"obsid": "00000000001", "result_dir": str(survey_dir)}]
    # The light-curve catalogue commonly prefixes image identifiers with
    # ``point_`` while the PHA header/filename stores the bare identifier.
    selected = [{"point": {"obsid": "00000000001", "pointing_id": "point_p1"}}]
    files = pipeline._pha_files(records, selected=selected)
    assert files == [good.resolve()]
    assert {item["reason"] for item in pipeline._pha_exclusions} >= {
        "source_mismatch:Other source",
        "pointing_outside_selection:p3",
        "manual_or_legacy_upper_limit_product",
    }
    assert all(path not in files for path in (other, outside, legacy))


def test_offline_pipeline_runs_survey_spectra_fit_and_report(tmp_path, monkeypatch):
    survey_dir = tmp_path / "00000000001_surveyresult"
    survey_dir.mkdir()
    pha, response = _write_synthetic_survey_pha(survey_dir)
    rates = survey_dir / "T_merged_pointings_lc.csv"
    rates.write_text(
        "OBSID,IMAGE_ID,NAME,TSTART,TSTOP,EXPOSURE,CENT_RATE,RATE_ERR,BKG_VAR,PCODEFR,STATUS\n"
        "1,p1,T,100,200,100,1.0,0.1,0.1,0.2,SUCCESS\n",
        encoding="utf-8",
    )

    # Keep this contract test independent of optional HEASoft while still
    # exercising the real stage transitions and science-ready PHA gate.
    def fake_preflight(self, _context):
        payload = {
            "dependency_checks": {"batanalysis": True, "heasoftpy": True, "pyxspec": True, "xspec_executable": True, "caldb": True},
            "warnings": [],
            "cache_reusable": True,
        }
        path = self._output("preflight.json")
        survey_module.write_json(path, payload)
        return StageResult(outputs={"preflight": str(path)}, data=payload)

    fit_calls = []

    def fake_fit(prepared, **kwargs):
        fit_calls.append((prepared, kwargs))
        return {"statistic": 2.0, "params": {"PhoIndex": 1.8, "norm": 0.5}}

    monkeypatch.setattr(BATSurveyPipeline, "_stage_preflight", fake_preflight)
    monkeypatch.setattr(survey_module, "fit_prepared", fake_fit)
    result = BATSurveyPipeline(
        BATSurveyInput(
            target_id="T",
            root=tmp_path,
            output_root=tmp_path / "output",
            source_name="T",
            obsids=(1,),
            survey_products_dir=survey_dir,
        ),
        config=BATSurvey(processes=2, internal_threads=1, task_timeout_s=17),
    ).run(resume=False)

    assert result.full_pipeline
    assert result.science_status == "complete"
    assert fit_calls and fit_calls[0][1]["stat_method"] == "chi"
    assert fit_calls[0][1]["model_name"] == "cflux*powerlaw"
    assert fit_calls[0][1]["error_delta_stat"] == pytest.approx(1.0)
    spectra = json.loads(Path(result.products["spectra"]["spectra"]).read_text())
    assert spectra["spectra"][0]["valid"]
    fits = json.loads(Path(result.products["fit"]["fit"]).read_text())
    assert fits["results"][0]["status"] == "fit_complete"
    report = json.loads(Path(result.products["report"]["report"]).read_text())
    assert report["energy_band_keV"] == [14.0, 195.0]
    assert report["rate_unit"] == "count/s/fully_illuminated_detector"
    assert report["fit_results"][0]["status"] == "fit_complete"
    assert report["mosaic"]["detection_basis"] == "mosaic_source_catalog_snr"
    assert Path(pha).is_file() and Path(response).is_file()


def test_survey_upper_limit_rechecks_profile_and_rejects_bad_status(tmp_path, monkeypatch):
    """Exercise the Gaussian survey upper-limit gates without real XSPEC."""
    pha = tmp_path / "source.pha"
    response = tmp_path / "source.rsp"
    pha.write_bytes(b"placeholder")
    response.write_bytes(b"placeholder")
    state = {"status": "FFFFFFFFF", "upper": 0.5}

    class _Parameter:
        def __init__(self):
            self._norm = 0.0
            self.frozen = False

        @property
        def values(self):
            return [self._norm, 2.0, 0.0, 0.0, 0.0, 10.0]

        @values.setter
        def values(self, value):
            if isinstance(value, (list, tuple, np.ndarray)):
                self._norm = float(value[0])
            else:
                self._norm = float(value)

    parameter = _Parameter()

    class _Model:
        def __call__(self, index):
            assert index == 2
            return parameter

    class _AllModels:
        def __init__(self):
            self.model = _Model()

        def __call__(self, index):
            assert index == 1
            return self.model

        def calcFlux(self, _band):
            return None

        def clear(self):
            return None

    class _Fit:
        @property
        def statistic(self):
            # A quadratic profile with a minimum at norm=0: norm=0.5 gives
            # the requested delta-statistic of 9.
            return 1.0 + 36.0 * parameter.values[0] ** 2

    class _AllData:
        def __call__(self, index):
            assert index == 1
            return SimpleNamespace(flux=(1.0e-10,))

        def clear(self):
            return None

    chain_state = {"cleared": False}

    class _AllChains:
        def clear(self):
            chain_state["cleared"] = True

    fake_xspec = SimpleNamespace(
        AllModels=_AllModels(),
        AllData=_AllData(),
        AllChains=_AllChains(),
        Fit=_Fit(),
    )

    class _UpperLimit:
        def __init__(self, _index):
            pass

        def error(self, **_kwargs):
            point = SimpleNamespace(
                upper=state["upper"],
                status=state["status"],
            )
            return SimpleNamespace(limits={"norm": point})

    import jinwu.core.upperlimit as upperlimit_module

    monkeypatch.setitem(sys.modules, "xspec", fake_xspec)
    monkeypatch.setattr(survey_module, "PreparedSpectrum", lambda **kwargs: kwargs)
    monkeypatch.setattr(survey_module, "fit_prepared", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(upperlimit_module, "UpperLimit", _UpperLimit)

    ready = survey_module.profile_survey_upper_limit(
        pha,
        response,
        output_dir=tmp_path / "ready",
    )
    assert ready.status == "upper_limit_ready"
    assert ready.normalization_at_lower_bound
    assert ready.parameter_status == "FFFFFFFFF"
    assert ready.profile_delta_stat == pytest.approx(9.0, abs=0.02)
    assert chain_state["cleared"]

    state.update(status="TTTTTTTTT", upper=0.5)
    failed_status = survey_module.profile_survey_upper_limit(
        pha,
        response,
        output_dir=tmp_path / "bad-status",
    )
    assert failed_status.status == "failed"
    assert any("upper_profile_xspec_status" in item for item in failed_status.diagnostics)

    state.update(status="FFFFFFFFF", upper=10.0)
    failed_bound = survey_module.profile_survey_upper_limit(
        pha,
        response,
        output_dir=tmp_path / "hard-bound",
    )
    assert failed_bound.status == "failed"
    assert "upper_profile_reached_hard_bound" in failed_bound.diagnostics


def test_headas_environment_is_stage_local_and_uses_conda_heasoft(tmp_path, monkeypatch):
    prefix = tmp_path / "conda"
    headas = prefix / "heasoft"
    (headas / "bin").mkdir(parents=True)
    (headas / "lib").mkdir()
    (headas / "syspfiles").mkdir()
    monkeypatch.delenv("HEADAS", raising=False)
    monkeypatch.setenv("CONDA_PREFIX", str(prefix))
    pipeline = BATSurveyPipeline(
        BATSurveyInput(target_id="env", root=tmp_path, output_root=tmp_path / "output")
    )
    environment = pipeline._headas_environment("fit")
    assert environment["HEADAS"] == str(headas.resolve())
    assert environment["PFILES"].startswith(str((tmp_path / "output" / ".pfiles" / "fit").resolve()))
    assert environment["PFILES"].endswith(";" + str((headas / "syspfiles").resolve()))
    assert environment["HOME"].startswith(str((tmp_path / "output" / ".heasoft_home" / "fit").resolve()))
    assert str(headas / "bin") == environment["PATH"].split(os.pathsep)[0]
    assert environment["LHEASOFT"] == str(headas.resolve())
    assert environment["PERL5LIB"] == str((headas / "lib" / "perl").resolve())


def test_calibration_config_change_invalidates_spectrum_stage(tmp_path, monkeypatch):
    config_file = tmp_path / "caldb.config"
    config_file.write_text("version-a\n", encoding="utf-8")
    monkeypatch.setenv("CALDBCONFIG", str(config_file))
    pipeline = BATSurveyPipeline(
        BATSurveyInput(target_id="fingerprint", root=tmp_path, output_root=tmp_path / "output")
    )
    stage = PipelineStage("spectra")
    first = pipeline._stage_input_fingerprint(stage)
    config_file.write_text("version-b\n", encoding="utf-8")
    second = pipeline._stage_input_fingerprint(stage)
    assert first != second


def test_mosaic_detection_uses_its_own_source_catalog(tmp_path):
    catalog = tmp_path / "sources_tot.cat"
    columns = [
        fits.Column(name="NAME", format="24A", array=["T"]),
        fits.Column(name="TIME", format="D", array=[100.0]),
        fits.Column(name="TIME_STOP", format="D", array=[200.0]),
        fits.Column(name="EXPOSURE", format="D", array=[100.0]),
        fits.Column(name="RATE", format="D", array=[0.1]),
        fits.Column(name="RATE_ERR", format="D", array=[0.1]),
        fits.Column(name="VECTSNR", format="D", array=[2.0]),
    ]
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(columns)]).writeto(catalog)
    measurements = survey_module._mosaic_source_measurements(
        catalog, source_name="T", detection_threshold=3.0
    )
    assert measurements[0]["status"] == "not_detected"
    assert measurements[0]["detection_basis"] == "mosaic_source_catalog_snr"
