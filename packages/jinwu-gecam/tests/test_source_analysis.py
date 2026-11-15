"""Boundary regressions for shared analysis contracts, without a PyXspec runtime."""

from dataclasses import replace

import astropy.units as u
from astropy.table import Table
import pytest

from jinwu.core import fit as fit_module
from jinwu.core.source_analysis import (
    LightcurveProduct,
    SourceAnalysisConfig,
    SourceAnalysisInput,
    SourceAnalysisPipeline,
    SpectralObservation,
)
from jinwu.core.spectrum_prep import PreparedSpectrum


T0 = "2026-01-19T00:13:08.8249"


def test_lightcurve_rebasing_resume_and_zero_error(tmp_path):
    path = tmp_path / "lc.ecsv"
    ref = "2026-01-19T00:14:08.8249"
    table = Table({
        "time_start": [0, 1] * u.min, "time_stop": [1, 2] * u.min,
        "value": [6, 0] * u.ct / u.min, "error": [1, 0] * u.ct / u.s,
    })
    table.meta.update({"target_id": "target", "reference_utc": ref, "kind": "total_rate"})
    table.write(path, format="ascii.ecsv")
    product = LightcurveProduct("lc.ecsv", "GECAM GRD17", "40–100 keV", ref)
    config = SourceAnalysisConfig(lightcurve_model="powerlaw")
    data = SourceAnalysisInput("target", tmp_path, tmp_path / "out", lightcurves=(product,), reference_utc=T0)
    pipeline = SourceAnalysisPipeline(data, config=config)

    result = pipeline.run()
    lc = result["stages"]["lightcurves"]
    summary = lc["data"]["lightcurves"][0]
    assert result["pipeline_status"] == "completed" and not result["fit_performed"]
    assert summary["time_range_s"] == pytest.approx([60, 180])
    assert summary["unit"] == "ct / min"
    assert summary["metadata"]["kind"] == "total_rate"
    assert summary["temporal_fit_status"] == "unavailable"
    assert pipeline.run(resume=True) == result

    table["value"][0] = 9
    table.write(path, format="ascii.ecsv", overwrite=True)
    updated = pipeline.run(resume=True)["stages"]["lightcurves"]
    assert updated["data"]["lightcurves"][0]["max"] == 9
    assert updated["outputs"]["lightcurve_0"] != lc["outputs"]["lightcurve_0"]

    wrong = replace(data, lightcurves=(replace(product, target_id="other"),))
    with pytest.raises(ValueError, match="another target"):
        SourceAnalysisPipeline(wrong, config=config).validate_input()


@pytest.mark.parametrize(
    "first_stop, second_stop, tolerance",
    [("18.8249", "18.8249", 0.001), ("09.8249", "10.8249", 2.0)],
    ids=["mismatched-bounds", "no-shared-time"],
)
def test_joint_epoch_rejects_mismatched_bounds_before_loading_xspec(tmp_path, first_stop, second_stop, tolerance):
    spec = PreparedSpectrum("FXT", "1", "FXTA", None, "extraction", None, "a.pha", None, None,
                            "a.rmf", 1, (0.3, 10.0))
    first = SpectralObservation(spec, T0, f"2026-01-19T00:13:{first_stop}", "cstat", "prompt")
    second = replace(first, start_utc="2026-01-19T00:13:09.8249",
                     stop_utc=f"2026-01-19T00:13:{second_stop}",
                     spectrum=replace(spec, module="FXTB", grouped_pha="b.pha"))
    data = SourceAnalysisInput("target", tmp_path, spectra=(first, second), reference_utc=T0)
    with pytest.raises(ValueError, match="different time intervals"):
        SourceAnalysisPipeline(data, config=SourceAnalysisConfig(simultaneity_tolerance_s=tolerance)).validate_input()


def test_joint_and_noncontemporaneous_routing(tmp_path, monkeypatch):
    inputs = tmp_path / "input"
    inputs.mkdir()
    for name in ("a", "b", "c"):
        for ext in ("pha", "rmf"):
            (inputs / f"{name}.{ext}").touch()

    spec = PreparedSpectrum("FXT", "1", "FXTA", None, "extraction", None, "a.pha", None, None,
                            "a.rmf", 1, (0.3, 10.0))
    stop = "2026-01-19T00:13:18.8249"
    first = SpectralObservation(spec, T0, stop, "cstat", "prompt")
    second = replace(first, statistic="pgstat", spectrum=replace(
        spec, instrument="GECAM", obsid="other", module="GRD17", grouped_pha="b.pha", rmf="b.rmf",
        energy_range_keV=(40.0, 100.0),
    ))
    third = replace(first, start_utc=stop, stop_utc="2026-01-19T00:13:28.8249", epoch="afterglow",
                    spectrum=replace(spec, obsid="2", grouped_pha="c.pha", rmf="c.rmf"))

    data = SourceAnalysisInput("target", inputs, tmp_path / "out", (first, second, third), (), T0)
    monkeypatch.setattr(SourceAnalysisPipeline, "runtime_versions", staticmethod(lambda spectral: {"xspec": "mock"}))
    calls = []

    def mock_fit(spectra, *, outdir, **kwargs):
        calls.append((spectra, kwargs))
        outdir.mkdir()
        path = outdir / "fit.mock"
        path.touch()
        keys = list(kwargs["statistics"])
        bands = {key: {"emin": s.energy_range_keV[0], "emax": s.energy_range_keV[1]}
                 for key, s in zip(keys, spectra)}
        flux = {"energy_range_keV": [0.5, 4.0], "frame": "observer", "erg_cm2_s": 1e-9,
                "error_lo": 1e-10, "error_hi": None, "error_status": "upper_open", "definition": "mock cflux"}
        return {
            "per_group": [{"member_spectra": [key], "flux_unabs": flux} for key in keys],
            "flux_unabs": flux, "energy_ranges": bands, "noticed_energy_ranges": bands,
            "cflux_extrapolated_spectra": [key for key in keys if key.startswith("other:")],
            "xspec_settings": {"error_delta_stat": 1.0},
            "fit_products": {name: str(path) for name in ("result_json", "report_txt", "xspec_log", "xcm")}
                            | {"plots": [str(path)]},
        }

    monkeypatch.setattr(fit_module, "fit_spectral", mock_fit)
    result = SourceAnalysisPipeline(data).run()
    assert [len(spectra) for spectra, _ in calls] == [2, 1]
    assert calls[0][1]["statistics"] == {"1:FXTA": "cstat", "other:GRD17": "pgstat"}
    assert all(kwargs["explicit_bindings"] for _, kwargs in calls)
    rows = result["stages"]["lightcurves"]["data"]["flux_rows"]
    assert [row["epoch"] for row in rows] == ["prompt", "afterglow"]
    assert rows[0]["error_status"] == "upper_open"
    assert rows[0]["time_s"] == pytest.approx(5)
    assert rows[0]["cflux_extrapolated_spectra"] == ["other:GRD17"]
    assert rows[1]["member_spectra"] == ["2:FXTA"]
