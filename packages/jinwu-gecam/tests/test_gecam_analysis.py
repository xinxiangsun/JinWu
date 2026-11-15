"""Protect GECAM timing, detector associations and product-only analysis."""
from pathlib import Path
from types import SimpleNamespace

import astropy.units as u
from astropy.io import fits
from astropy.table import Table
import numpy as np
import pytest

from jinwu.core.time import Time
from jinwu.gecam import GECAMConfig, GECAMInput, GECAMPipeline, backend, gecam_lightcurve_products
from jinwu.gecam.analysis import _met_reference, prepare_gecam_spectra


def test_finished_curves_with_missing_response(tmp_path, monkeypatch):
    def unexpected_backend():
        pytest.fail("Finished products must not load GECAMTools")
    monkeypatch.setattr(backend, "_backend", unexpected_backend)
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    counts = inputs / "counts.npz"
    np.savez(counts, time_edges_met_s=[10, 11, 13], energy_edges_keV=[39, 102],
             counts=[[2., 6.]], counts_error=[[1., 2.]], correct_dead_time=True)
    reference = "2019-01-01T00:00:10"
    curves = gecam_lightcurve_products({"GRD17_high": counts}, outdir=inputs, reference_utc=reference,
                                      target_id="test")
    table = Table.read(curves[0].path)
    np.testing.assert_allclose(table["time_stop"], [1, 3], atol=1e-8)
    np.testing.assert_array_equal(table["value"].quantity.to_value(u.ct / u.s), [2., 3.])
    np.testing.assert_array_equal(table["error"].quantity.to_value(u.ct / u.s), [1., 1.])
    assert table.meta["actual_energy_range_keV"] == [39., 102.]
    assert table.meta["met_clock_source"] == "assumed GECAM-A/B mission epoch"
    inp = GECAMInput("test", inputs, output_root=tmp_path / "run", lightcurve_products=curves,
                     source_spectra={17: counts}, background_spectra={17: counts})
    config = GECAMConfig(detectors=(17,), gain="high", time_range=[10, 13] * u.s,
                         time_bin=1 * u.s, energy_edges=[40, 100] * u.keV)
    result = GECAMPipeline(inp, config=config).run()
    stage = result["stages"]["analysis"]
    assert result["pipeline_status"] == "needs_review"
    assert not result["fit_performed"]
    assert "Missing explicit GECAM responses for detectors [17]" in stage["message"]
    assert Path(stage["outputs"]["lightcurves_lightcurve_0"]).is_file()


def test_native_met_reference():
    header = fits.Header({"MJDREFI": 58484, "MJDREFF": .00080074074,
                          "TIMESYS": "TT", "TIMEUNIT": "s", "TIMEZERO": 2.0})
    epoch, source = _met_reference(header, "GECAM-B")
    assert abs((epoch - Time("2019-01-01T00:00:02", scale="utc")).to_value(u.s)) < 1e-6
    assert source.startswith("FITS")
    with pytest.raises(ValueError, match="GECAM-C/HEBS MET reference"):
        _met_reference({}, "GECAM-C")
    with pytest.raises(ValueError, match="Incomplete GECAM FITS time reference"):
        _met_reference({}, "GECAM-B", allow_mission_epoch=False)


def test_native_lightcurve_clock(tmp_path):
    counts = tmp_path / "counts.npz"
    np.savez(counts, time_edges_met_s=[10, 13], energy_edges_keV=[39, 102], counts=[[6.]],
             counts_error=[[2.]], correct_dead_time=False, telescope="GECAM-C",
             met_reference_utc="2022-07-01T00:00:00.123456789", met_clock_source="FITS MJDREF/TIMESYS")
    curve, = gecam_lightcurve_products({"GRD17_high": counts}, outdir=tmp_path,
                                       reference_utc="2022-07-01T00:00:10")
    table = Table.read(curve.path)
    assert curve.instrument == "GECAM-C GRD17_high"
    assert table.meta["met_clock_source"] == "FITS MJDREF/TIMESYS"
    assert table.meta["met_reference_utc"] == "2022-07-01T00:00:00.123456789"
    np.testing.assert_allclose(table["time_start"], [.123456789], rtol=0, atol=1e-9)


def test_matching_type_ii_rows_and_response(tmp_path, monkeypatch):
    import jinwu.gecam.analysis as analysis

    grouping = [[1, -1, 1]]
    ebounds = fits.BinTableHDU.from_columns([
        fits.Column(name="CHANNEL", format="I", array=[0, 1, 2]),
        fits.Column(name="E_MIN", format="E", array=[40, 50, 60]),
        fits.Column(name="E_MAX", format="E", array=[50, 60, 70]),
    ], name="EBOUNDS")
    source, background, rsp = (tmp_path / f"{stem}.fits" for stem in ("source", "background", "rsp"))
    for path, counts, err in ((source, [4., 0., 9.], [2., 0., 3.]), (background, [2.1, 0., 6.3], [1., 0., .5])):
        spec = fits.BinTableHDU.from_columns([
            fits.Column(name="SPEC_NUM", format="I", array=[1]),
            fits.Column(name="TIME", format="D", array=[74431599.6]),
            fits.Column(name="TIMEDEL", format="D", array=[10.]),
            fits.Column(name="EXPOSURE", format="D", array=[9.5]),
            fits.Column(name="CHANNEL", format="3I", array=[[0, 1, 2]]),
            fits.Column(name="COUNTS", format="3D", array=[counts]),
            fits.Column(name="STAT_ERR", format="3D", array=[err]),
            fits.Column(name="GROUPING", format="3I", array=grouping),
            fits.Column(name="QUALITY", format="3I", array=[[0, 0, 0]]),
        ], name="SPECTRUM")
        spec.header.update({"TELESCOP": "GECAM-B", "DETNUMS": "17"})
        fits.HDUList([fits.PrimaryHDU(), spec, ebounds]).writeto(path)
    fits.HDUList([fits.PrimaryHDU(header=fits.Header({"DETNAM": "bg17H"})), ebounds]).writeto(rsp)
    monkeypatch.setattr(analysis, "prepare_ogip_spectrum", lambda source, **kwargs: SimpleNamespace(diagnostics=[], **kwargs))
    args = {17: source}, {17: background}, {17: rsp}
    options = dict(outdir=tmp_path / "prepared", energy_range=[40, 70] * u.keV, target_id="test", gain="high")
    observation, = prepare_gecam_spectra(*args, **options)
    assert observation.statistic == "pgstat"
    assert observation.spectrum.source_poisson is True and observation.spectrum.background_poisson is False
    assert observation.spectrum.source_row == observation.spectrum.background_row == 1
    assert abs((Time(observation.start_utc) - Time("2021-05-11T11:26:39.6")).to_value(u.s)) < 1e-6
    assert abs((Time(observation.stop_utc) - Time("2021-05-11T11:26:49.6")).to_value(u.s)) < 1e-6
    assert any("Background FITS clock absent" in item for item in observation.spectrum.diagnostics)
    with pytest.raises(ValueError, match="Response detector/gain"):
        prepare_gecam_spectra(*args, **(options | {"gain": "low"}))
    with fits.open(background, mode="update") as hdus:
        hdus["SPECTRUM"].data["EXPOSURE"][:] = 9.0
    with pytest.raises(ValueError, match="Full and background EXPOSURE"):
        prepare_gecam_spectra(*args, **options)
    for path in (source, background):
        with fits.open(path, mode="update") as hdus:
            hdus["SPECTRUM"].header["TELESCOP"] = "GECAM-C"
            hdus["SPECTRUM"].data["EXPOSURE"][:] = 9.5
    with pytest.raises(ValueError, match="GECAM-C/HEBS MET reference"):
        prepare_gecam_spectra(*args, **options)
    observation, = prepare_gecam_spectra(*args, **options, met_reference_utc="2022-07-01T00:00:00")
    assert abs((Time(observation.start_utc) - Time("2024-11-08T11:26:39.6")).to_value(u.s)) < 1e-6
    for path, offset in ((source, 0.), (background, 2.)):
        with fits.open(path, mode="update") as hdus:
            hdus[0].header.update({"MJDREFI": 59761, "MJDREFF": .00080074074,
                                   "TIMESYS": "TT", "TIMEUNIT": "s", "TIMEZERO": offset})
    with pytest.raises(ValueError, match="Full and background MET references"):
        prepare_gecam_spectra(*args, **options)
    with pytest.raises(ValueError, match="Full and background MET references"):
        prepare_gecam_spectra(*args, **options, met_reference_utc="2022-07-01T00:00:00")
    with fits.open(background, mode="update") as hdus:
        # Missing TIMEZERO is equivalent to an explicit zero, even though the
        # headers are not identical. Row TIME values alone cannot prove this.
        del hdus[0].header["TIMEZERO"]
    observation, = prepare_gecam_spectra(*args, **options)
    assert not any("Background FITS clock absent" in item for item in observation.spectrum.diagnostics)
    assert abs((Time(observation.start_utc) - Time("2024-11-08T11:26:39.6")).to_value(u.s)) < 1e-6
